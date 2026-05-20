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

# librosa is used only for TinyVAD log-mel extraction.
# Install with: pip install librosa
try:
    import librosa
    _LIBROSA_OK = True
except ImportError:
    _LIBROSA_OK = False

SAMPLE_RATE = 16000
CHUNK_SEC   = 1

# TinyVAD feature constants — must match train_tiny_vad.py exactly so that
# the feature vectors fed to the TFLite model are identical to training.
N_MEL    = 40
N_FRAMES = 49

# ── feature extraction for TinyVAD ───────────────────────────────────────────

def extract_logmel(chunk):
    """Convert a 1-second float32 PCM chunk → [N_FRAMES, N_MEL] log-mel array.

    Parameters match train_tiny_vad.py: n_fft=512, win=400, hop=160,
    n_mels=40, fmin=80, fmax=7600.  This is critical — a mismatch here would
    silently produce wrong activations even with correct weights.

    This function will also serve as the golden reference when we later write
    the C feature-extraction code for PicoRV32 firmware (Stage 2 milestone).
    """
    if not _LIBROSA_OK:
        raise ImportError("librosa is required for TinyVAD. Run: pip install librosa")
    mel = librosa.feature.melspectrogram(
        y=chunk, sr=SAMPLE_RATE,
        n_fft=512, win_length=400, hop_length=160,
        n_mels=N_MEL, fmin=80.0, fmax=7600.0,
        htk=True, norm=None,   # match torchaudio defaults used during training
    )
    log_mel = np.log(mel + 1e-6).T          # [frames, N_MEL], matches training
    log_mel = log_mel[:N_FRAMES]
    pad = N_FRAMES - log_mel.shape[0]
    if pad > 0:
        log_mel = np.pad(log_mel, ((0, pad), (0, 0)))
    return log_mel.astype(np.float32)       # [49, 40]

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
