# pipeline.py  —  simulates live pipeline using a file
#
# Supports two VAD backends selectable at runtime:
#   yamnet   — raw-waveform input, 521-class AudioSet model (Google)
#   tinyvad  — log-mel input, custom int8 2-class model trained in Stage 2
#
# Usage:
#   python speech_simulator.py <wav> [yamnet|tinyvad] [model_path] [whisper_model]
#
# Stage context: this is the software-only reference pipeline (Stage 1/2).
# Latency numbers here become the baseline that PicoRV32+accelerator must beat.

import numpy as np
import soundfile as sf
import tflite_runtime.interpreter as tflite
import subprocess, time, tempfile, os, csv, datetime

SAMPLE_RATE = 16000
CHUNK_SEC   = 1

# TinyVAD feature constants — must match train_tiny_vad.py exactly.
N_MEL    = 40
N_FRAMES = 49
_N_FFT   = 512
_WIN_LEN = 400
_HOP_LEN = 160

# ── mel filterbank (built once, reused every chunk) ───────────────────────────

def _build_mel_fb():
    """HTK mel filterbank matrix [N_MEL, n_fft//2+1], norm=None.

    Matches torchaudio.transforms.MelSpectrogram defaults (mel_scale='htk',
    norm=None).  We build this once at startup and cache it — it's constant
    for the lifetime of the process and will be reused for every audio chunk.

    This is also the function we'll port to C when writing the PicoRV32
    firmware feature extractor.  The filterbank is just a fixed float matrix
    multiply, which maps neatly to the int8 MAC accelerator in Stage 3.
    """
    def hz_to_mel(f): return 2595.0 * np.log10(1.0 + f / 700.0)
    def mel_to_hz(m): return 700.0 * (10.0 ** (m / 2595.0) - 1.0)

    n_freqs   = _N_FFT // 2 + 1
    fft_freqs = np.linspace(0.0, SAMPLE_RATE / 2.0, n_freqs)
    mel_pts   = np.linspace(hz_to_mel(80.0), hz_to_mel(7600.0), N_MEL + 2)
    hz_pts    = mel_to_hz(mel_pts)

    fb = np.zeros((N_MEL, n_freqs), dtype=np.float32)
    for m in range(N_MEL):
        lo, ctr, hi = hz_pts[m], hz_pts[m + 1], hz_pts[m + 2]
        up   = (fft_freqs >= lo)  & (fft_freqs <= ctr)
        down = (fft_freqs >  ctr) & (fft_freqs <= hi)
        if ctr > lo:
            fb[m, up]   = (fft_freqs[up]   - lo)  / (ctr - lo)
        if hi > ctr:
            fb[m, down] = (hi - fft_freqs[down]) / (hi  - ctr)
    return fb

_MEL_FB = _build_mel_fb()   # computed once at import time

# ── feature extraction for TinyVAD ───────────────────────────────────────────

def extract_logmel(chunk):
    """Convert a 1-second float32 PCM chunk → [N_FRAMES, N_MEL] log-mel.

    Pure numpy — no librosa/torch dependency.  Parameters match
    train_tiny_vad.py exactly (n_fft=512, win=400, hop=160, n_mels=40,
    fmin=80, fmax=7600, HTK mel scale, no filterbank normalization).

    Steps:
      1. Reflect-pad by n_fft//2 on each side  (matches torchaudio center=True)
      2. Frame with periodic Hann window        (matches torch.hann_window)
      3. rfft → power spectrum
      4. Mel filterbank matrix multiply
      5. log(mel + 1e-6)

    This function doubles as the golden reference for the C feature extractor
    we'll write in Stage 2.  Every step here has a direct C equivalent.
    """
    # 1. center-pad (reflect) so frame 0 is centered on sample 0
    chunk = np.pad(chunk, _N_FFT // 2, mode='reflect')

    # 2. periodic Hann window of size win_length, zero-padded to n_fft
    #    np.hanning gives a symmetric window; [:-1] makes it periodic
    win = np.hanning(_WIN_LEN + 1)[:-1].astype(np.float32)
    pad_w = _N_FFT - _WIN_LEN
    win   = np.pad(win, (pad_w // 2, pad_w - pad_w // 2))

    # 3. frame + rfft using stride tricks (no copy of the signal)
    n_frames = 1 + (len(chunk) - _N_FFT) // _HOP_LEN
    frames   = np.lib.stride_tricks.as_strided(
        chunk,
        shape=(n_frames, _N_FFT),
        strides=(chunk.strides[0] * _HOP_LEN, chunk.strides[0]),
    )
    power = (np.abs(np.fft.rfft(frames * win, n=_N_FFT)) ** 2).T  # [n_freqs, n_frames]

    # 4 & 5. mel filterbank then log
    log_mel = np.log(_MEL_FB @ power.astype(np.float32) + 1e-6).T  # [n_frames, N_MEL]

    log_mel = log_mel[:N_FRAMES]
    pad = N_FRAMES - log_mel.shape[0]
    if pad > 0:
        log_mel = np.pad(log_mel, ((0, pad), (0, 0)))
    return log_mel.astype(np.float32)   # [49, 40]

# ── VAD inference — TinyVAD (int8) ───────────────────────────────────────────

def run_vad_tinyvad(interpreter, chunk):
    """Run the custom int8 TinyVAD model on one audio chunk.

    Because the TFLite model was exported with full integer quantization
    (inference_input_type=INT8, inference_output_type=INT8), we must:
      1. quantize float32 log-mel → int8 using the stored scale/zero_point
      2. invoke the interpreter
      3. dequantize int8 logits → float32, then softmax

    Returns (probs, latency_s) where probs[0]=silence, probs[1]=speech.
    """
    inp_detail = interpreter.get_input_details()[0]
    out_detail = interpreter.get_output_details()[0]

    log_mel = extract_logmel(chunk)                     # [49, 40] float32

    i_scale, i_zero = inp_detail['quantization']
    q_input = np.clip(
        np.round(log_mel / i_scale + i_zero), -128, 127
    ).astype(np.int8)
    interpreter.set_tensor(inp_detail['index'], q_input[np.newaxis])  # [1,49,40]

    t0 = time.perf_counter()
    interpreter.invoke()
    lat = time.perf_counter() - t0

    logits_int8 = interpreter.get_tensor(out_detail['index'])[0]  # [2]
    o_scale, o_zero = out_detail['quantization']
    logits_f = (logits_int8.astype(np.float32) - o_zero) * o_scale

    # numerically stable softmax
    lf = logits_f - logits_f.max()
    probs = np.exp(lf) / np.exp(lf).sum()
    return probs, lat

# ── VAD inference — YAMNet (float32) ─────────────────────────────────────────

def run_vad_yamnet(interpreter, chunk):
    """Run YAMNet on one audio chunk.

    YAMNet takes a 1D float32 waveform (no batch dim) and outputs 521 AudioSet
    class scores.  Index 0 = 'Speech'.  This was the original placeholder
    before TinyVAD was trained — kept here for A/B comparison during Stage 1/2
    so we can verify TinyVAD matches or beats YAMNet accuracy on our data.
    """
    inp = interpreter.get_input_details()[0]
    out = interpreter.get_output_details()[0]

    target_len = inp['shape'][0]   # YAMNet expects exactly 15360 samples
    if len(chunk) >= target_len:
        chunk = chunk[:target_len]
    else:
        chunk = np.pad(chunk, (0, target_len - len(chunk)))

    interpreter.set_tensor(inp['index'], chunk.astype(np.float32))
    t0 = time.perf_counter()
    interpreter.invoke()
    lat = time.perf_counter() - t0

    scores = interpreter.get_tensor(out['index'])[0]   # [521]
    return scores, lat

# ── logging ───────────────────────────────────────────────────────────────────

def log_result(wav_path, vad_type, whisper_model, vad_lats, whisper_lat, transcript):
    row = {
        "timestamp":       datetime.datetime.now().isoformat(),
        "file":            wav_path,
        "vad_type":        vad_type,
        "model":           whisper_model,
        "chunks":          len(vad_lats),
        "vad_lat_avg_ms":  round(sum(vad_lats) / len(vad_lats), 1),
        "vad_lat_max_ms":  round(max(vad_lats), 1),
        "whisper_lat_s":   round(whisper_lat, 2),
        "transcript_words": len(transcript.split()),
    }
    write_header = not os.path.exists("results.csv")
    with open("results.csv", "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=row.keys())
        if write_header:
            w.writeheader()
        w.writerow(row)
    print(row)

# ── audio loading ─────────────────────────────────────────────────────────────

def load_audio(path):
    audio, sr = sf.read(path, dtype='float32')
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != SAMPLE_RATE:
        raise ValueError(f"Expected 16kHz, got {sr}Hz. Resample first.")
    return audio

# ── main pipeline ─────────────────────────────────────────────────────────────

def main(wav_path, vad_type="tinyvad", model_path=None, whisper_model="tiny.en"):
    if model_path is None:
        model_path = "tiny_vad_int8.tflite" if vad_type == "tinyvad" else "yamnet.tflite"

    print(f"VAD backend : {vad_type}  ({model_path})")
    print(f"Whisper     : {whisper_model}")
    print(f"Audio file  : {wav_path}\n")

    interpreter = tflite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    audio  = load_audio(wav_path)
    chunks = [audio[i:i+SAMPLE_RATE] for i in range(0, len(audio) - SAMPLE_RATE, SAMPLE_RATE)]

    transcript    = ""
    speech_chunks = []
    vad_lats      = []

    for i, chunk in enumerate(chunks):
        if vad_type == "tinyvad":
            probs, lat = run_vad_tinyvad(interpreter, chunk)
            is_speech  = probs[1] > 0.5
            score      = probs[1]
        else:
            scores, lat = run_vad_yamnet(interpreter, chunk)
            is_speech   = scores[0] > 0.3   # YAMNet index 0 = Speech
            score       = scores[0]

        print(f"  chunk {i:02d}: speech={is_speech}  vad_lat={lat*1000:.1f}ms  score={score:.2f}")
        vad_lats.append(lat * 1000)
        if is_speech:
            speech_chunks.append(chunk)

    whisper_lat = 0
    if speech_chunks:
        print(f"\nSpeech detected in {len(speech_chunks)}/{len(chunks)} chunks — running whisper.cpp...")
        combined = np.concatenate(speech_chunks)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            sf.write(f.name, combined, SAMPLE_RATE)
            t0 = time.perf_counter()
            result = subprocess.run(
                ["./whisper.cpp/build/bin/whisper-cli",
                 "-m", f"whisper.cpp/models/ggml-{whisper_model}.bin",
                 "-f", f.name, "--no-timestamps"],
                capture_output=True, text=True
            )
            whisper_lat = time.perf_counter() - t0
            print(f"Whisper latency: {whisper_lat:.2f}s")
            if result.returncode != 0:
                print("STDERR:", result.stderr[:500])
                print("STDOUT:", result.stdout[:500])
            else:
                transcript = result.stdout.strip()
                print("Transcript:", transcript)
                os.unlink(f.name)
    else:
        print("No speech detected — whisper skipped.")

    log_result(wav_path, vad_type, whisper_model, vad_lats, whisper_lat, transcript)


if __name__ == "__main__":
    import sys
    # speech_simulator.py <wav> [yamnet|tinyvad] [model_path] [whisper_model]
    wav    = sys.argv[1]
    vad    = sys.argv[2] if len(sys.argv) > 2 else "tinyvad"
    mpath  = sys.argv[3] if len(sys.argv) > 3 else None
    wmodel = sys.argv[4] if len(sys.argv) > 4 else "tiny.en"
    main(wav, vad, mpath, wmodel)
