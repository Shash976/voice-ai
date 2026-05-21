# gen_test_vectors.py
#
# Generates golden test vectors from the TFLite model and real audio.
# Outputs:
#   firmware/tinyengine_port/tiny_vad_test_vectors.h
#
# Each vector is:
#   - input:    int8[49*40] log-mel feature (quantized to model's input scale/zp)
#   - expected: int8[2]     output logits
#   - label:    0=silence 1=speech
#
# Run:  python sw/tinyml_reference/gen_test_vectors.py
#
# Requires: tflite_runtime, soundfile, numpy

import sys, pathlib, random
import numpy as np
import soundfile as sf

ROOT = pathlib.Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from speech_simulator import extract_logmel, SAMPLE_RATE

try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    import tensorflow.lite as tflite

TFLITE_PATH = ROOT / "tiny_vad_int8.tflite"
OUT_H       = ROOT / "firmware" / "tinyengine_port" / "tiny_vad_test_vectors.h"
DATA_DIR    = ROOT / "speech_commands"
N_VECTORS   = 64   # 32 speech + 32 silence
N_MEL       = 40
N_FRAMES    = 49

# ── load model ────────────────────────────────────────────────────────────────

interp = tflite.Interpreter(model_path=str(TFLITE_PATH))
interp.allocate_tensors()
inp_detail = interp.get_input_details()[0]
out_detail = interp.get_output_details()[0]

sc_in  = inp_detail["quantization"][0]
zp_in  = int(inp_detail["quantization"][1])
sc_out = out_detail["quantization"][0]
zp_out = int(out_detail["quantization"][1])

def run_tflite(log_mel_f32):
    """log_mel_f32: [49, 40] float32 → int8[2] output logits"""
    # quantize input
    q = np.clip(np.round(log_mel_f32 / sc_in + zp_in), -128, 127).astype(np.int8)
    interp.set_tensor(inp_detail["index"], q[np.newaxis])
    interp.invoke()
    return interp.get_tensor(out_detail["index"])[0].copy()  # int8[2]

def load_wav(path):
    wav, _ = sf.read(str(path), dtype="float32")
    if wav.ndim > 1: wav = wav.mean(1)
    if len(wav) < SAMPLE_RATE:
        wav = np.pad(wav, (0, SAMPLE_RATE - len(wav)))
    return wav[:SAMPLE_RATE]

# ── collect samples ───────────────────────────────────────────────────────────

speech_wavs  = []
silence_wavs = []

if DATA_DIR.exists():
    speech_words = ["yes", "no", "up", "down", "stop", "go", "on", "off"]
    for word in speech_words:
        files = list((DATA_DIR / word).glob("*.wav"))
        random.shuffle(files)
        for f in files[:4]:
            try:
                speech_wavs.append(load_wav(f))
            except Exception:
                pass
    bg_dir = DATA_DIR / "_background_noise_"
    if bg_dir.exists():
        for bf in bg_dir.glob("*.wav"):
            wav, _ = sf.read(str(bf), dtype="float32")
            if wav.ndim > 1: wav = wav.mean(1)
            for s in range(0, len(wav) - SAMPLE_RATE, SAMPLE_RATE):
                if len(silence_wavs) >= N_VECTORS // 2:
                    break
                silence_wavs.append(wav[s:s + SAMPLE_RATE])

# Fallback: use whisper.cpp sample audio for speech vectors
if len(speech_wavs) < N_VECTORS // 2:
    samples_dir = ROOT / "whisper.cpp" / "samples"
    for wav_path in sorted(samples_dir.glob("*.wav")):
        try:
            wav, sr = sf.read(str(wav_path), dtype="float32")
            if wav.ndim > 1: wav = wav.mean(1)
            # resample if needed (jfk.wav is 16kHz so should be fine)
            if sr != SAMPLE_RATE:
                print(f"  WARNING: {wav_path.name} is {sr}Hz, expected {SAMPLE_RATE}Hz — skipping")
                continue
            # chunk into 1-second pieces
            for s in range(0, len(wav) - SAMPLE_RATE, SAMPLE_RATE):
                if len(speech_wavs) >= N_VECTORS // 2:
                    break
                speech_wavs.append(wav[s:s + SAMPLE_RATE].copy())
            if len(speech_wavs) >= N_VECTORS // 2:
                break
        except Exception as e:
            print(f"  WARNING: could not load {wav_path}: {e}")
    if speech_wavs:
        print(f"Loaded {len(speech_wavs)} speech chunks from whisper.cpp/samples/")

# Silence fallback: near-zero noise floor (not random — matches real mic noise)
if len(silence_wavs) < N_VECTORS // 2:
    rng = np.random.default_rng(42)
    while len(silence_wavs) < N_VECTORS // 2:
        silence_wavs.append((rng.standard_normal(SAMPLE_RATE) * 0.002).astype(np.float32))
    print(f"Generated {N_VECTORS // 2} synthetic silence vectors (near-zero noise floor)")

if len(speech_wavs) == 0:
    print("WARNING: no speech audio found anywhere — test vectors will be meaningless")
    rng = np.random.default_rng(0)
    for _ in range(N_VECTORS // 2):
        speech_wavs.append((rng.standard_normal(SAMPLE_RATE) * 0.05).astype(np.float32))

speech_wavs  = speech_wavs[:N_VECTORS // 2]
silence_wavs = silence_wavs[:N_VECTORS // 2]
print(f"Speech vectors:  {len(speech_wavs)}")
print(f"Silence vectors: {len(silence_wavs)}")

# ── run TFLite on each sample ─────────────────────────────────────────────────

vectors = []
for wav, label in [(w, 1) for w in speech_wavs] + [(w, 0) for w in silence_wavs]:
    feat_f32 = extract_logmel(wav)                    # [49, 40] float32
    inp_q    = np.clip(np.round(feat_f32 / sc_in + zp_in), -128, 127).astype(np.int8)
    out_q    = run_tflite(feat_f32)                   # int8[2]
    pred     = 0 if out_q[0] > out_q[1] else 1
    vectors.append((inp_q, out_q, label))

# ── accuracy check ────────────────────────────────────────────────────────────

correct = sum(
    (1 if v[1][1] > v[1][0] else 0) == v[2]
    for v in vectors
)
print(f"TFLite accuracy on test vectors: {correct}/{len(vectors)}")

# ── write C header ────────────────────────────────────────────────────────────

def fmt_i8_array(arr):
    return ", ".join(str(int(v)) for v in arr.flatten())

with open(OUT_H, "w") as f:
    f.write("/* tiny_vad_test_vectors.h — auto-generated by gen_test_vectors.py */\n")
    f.write("#ifndef TINY_VAD_TEST_VECTORS_H\n")
    f.write("#define TINY_VAD_TEST_VECTORS_H\n")
    f.write("#include <stdint.h>\n\n")
    f.write(f"#define N_TEST_VECTORS {len(vectors)}\n")
    f.write(f"#define VEC_INPUT_LEN  {N_FRAMES * N_MEL}\n")
    f.write(f"#define VEC_OUTPUT_LEN 2\n\n")

    f.write(f"static const int8_t test_inputs[{len(vectors)}][{N_FRAMES * N_MEL}] = {{\n")
    for inp_q, _, _ in vectors:
        f.write(f"    {{ {fmt_i8_array(inp_q)} }},\n")
    f.write("};\n\n")

    f.write(f"static const int8_t test_expected_outputs[{len(vectors)}][2] = {{\n")
    for _, out_q, _ in vectors:
        f.write(f"    {{ {fmt_i8_array(out_q)} }},\n")
    f.write("};\n\n")

    f.write(f"static const int8_t test_labels[{len(vectors)}] = {{\n    ")
    f.write(", ".join(str(v[2]) for v in vectors))
    f.write("\n};\n\n")

    f.write("#endif /* TINY_VAD_TEST_VECTORS_H */\n")

print(f"Wrote {OUT_H}")
