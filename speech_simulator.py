# pipeline.py  —  simulates live pipeline using a file
import numpy as np
import soundfile as sf
import tflite_runtime.interpreter as tflite
import subprocess, time, tempfile, os

SAMPLE_RATE = 16000
CHUNK_SEC   = 1

import csv, datetime

def log_result(wav_path, whisper_model, vad_lats, whisper_lat, transcript):
    row = {
        "timestamp": datetime.datetime.now().isoformat(),
        "file": wav_path,
        "model": whisper_model,
        "chunks": len(vad_lats),
        "vad_lat_avg_ms": round(sum(vad_lats) / len(vad_lats), 1),
        "vad_lat_max_ms": round(max(vad_lats), 1),
        "whisper_lat_s": round(whisper_lat, 2),
        "transcript_words": len(transcript.split()),
    }
    write_header = not os.path.exists("results.csv")
    with open("results.csv", "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=row.keys())
        if write_header:
            w.writeheader()
        w.writerow(row)
    print(row)


def load_audio(path):
    audio, sr = sf.read(path, dtype='float32')
    if audio.ndim > 1:
        audio = audio.mean(axis=1)          # stereo → mono
    if sr != SAMPLE_RATE:
        # simple resample — or use librosa/soxr if available
        raise ValueError(f"Expected 16kHz, got {sr}Hz. Resample first.")
    return audio

def run_vad(interpreter, chunk):
    inp = interpreter.get_input_details()[0]
    out = interpreter.get_output_details()[0]

    # YAMNet expects a 1D float32 waveform — no batch dimension
    target_len = inp['shape'][0]   # typically 15360 for YAMNet
    if len(chunk) >= target_len:
        chunk = chunk[:target_len]
    else:
        chunk = np.pad(chunk, (0, target_len - len(chunk)))

    interpreter.set_tensor(inp['index'], chunk.astype(np.float32))
    t0 = time.perf_counter()
    interpreter.invoke()
    lat = time.perf_counter() - t0
    scores = interpreter.get_tensor(out['index'])[0]
    return scores, lat

def main(wav_path, model_path="yamnet.tflite", whisper_model="tiny.en"):
    interpreter = tflite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    audio = load_audio(wav_path)
    chunks = [audio[i:i+SAMPLE_RATE] for i in range(0, len(audio)-SAMPLE_RATE, SAMPLE_RATE)]
    transcript = ""
    speech_chunks = []
    vad_lats = []
    for i, chunk in enumerate(chunks):
        scores, lat = run_vad(interpreter, chunk)
        is_speech = scores[0] > 0.3      # YAMNet index 0 = speech
        print(f"  chunk {i:02d}: speech={is_speech}  vad_lat={lat*1000:.1f}ms  score={scores[0]:.2f}")
        vad_lats.append(lat*1000)
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
                ["./whisper.cpp/build/bin/whisper-cli", "-m", f"whisper.cpp/models/ggml-{whisper_model}.bin",
                 "-f", f.name, "--no-timestamps"],
                capture_output=True, text=True
            )
            whisper_lat = time.perf_counter()-t0
            print(f"Whisper latency: {whisper_lat:.2f}s")
            print(f"Return code: {result.returncode}")
            if result.returncode != 0:
                print("STDERR:", result.stderr[:500])
                print("STDOUT:", result.stdout[:500])
            else:
                transcript = result.stdout.strip()
                print("Transcript:", transcript)
                os.unlink(f.name)
    else:
        print("No speech detected — whisper skipped.")
    log_result(wav_path, whisper_model, vad_lats, whisper_lat, transcript)   

if __name__ == "__main__":
    import sys
    main(sys.argv[1])

