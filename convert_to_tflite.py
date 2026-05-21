# convert_to_tflite.py
# Transfers weights from trained PyTorch TinyVAD → Keras → int8 TFLite.
# Architecture must match train_tiny_vad.py exactly.
#
# Run after training:  python convert_to_tflite.py
# Output:              tiny_vad_int8.tflite

import numpy as np
import torch
import tensorflow as tf
from train_tiny_vad import TinyVAD, N_MEL, N_FRAMES

# ── load PyTorch weights ──────────────────────────────────────────────────────

pt = TinyVAD()
pt.load_state_dict(torch.load("tiny_vad_best.pt", map_location="cpu"))
pt.eval()
s = pt.state_dict()

# ── build equivalent Keras model ──────────────────────────────────────────────
# Architecture: Conv1d(40→32,k=5,s=2) BN ReLU
#               Conv1d(32→64,k=3,s=2) BN ReLU
#               GlobalAvgPool
#               Dense(64→32) BN ReLU
#               Dense(32→2)
#
# NOTE: BN is fused into the preceding conv/dense during quantization.
# We fold BN parameters manually before building Keras so the Keras model
# is already BN-free — this produces cleaner int8 export.

def fold_bn(w_pt, b_pt, gamma, beta, mean, var, eps=1e-5):
    """Fold BN into preceding layer weights/bias."""
    std = np.sqrt(var + eps)
    scale = gamma / std
    w_folded = w_pt * scale[:, None] if w_pt.ndim == 2 else w_pt * scale[:, None, None]
    b_folded = beta + (b_pt - mean) * scale
    return w_folded, b_folded

# -- Conv0 + BN0 --
w0 = s["net.0.weight"].numpy()  # [32, 40, 5]
b0 = s["net.0.bias"].numpy()    # [32]  (PyTorch Conv1d has bias=True by default when no BN;
                                 #        but our model has Conv1d→BN so bias is absorbed by BN)
g0  = s["net.1.weight"].numpy()
be0 = s["net.1.bias"].numpy()
m0  = s["net.1.running_mean"].numpy()
v0  = s["net.1.running_var"].numpy()
w0f, b0f = fold_bn(w0, b0, g0, be0, m0, v0)

# -- Conv1 + BN1 --
w1 = s["net.3.weight"].numpy()  # [64, 32, 3]
b1 = s["net.3.bias"].numpy()
g1  = s["net.4.weight"].numpy()
be1 = s["net.4.bias"].numpy()
m1  = s["net.4.running_mean"].numpy()
v1  = s["net.4.running_var"].numpy()
w1f, b1f = fold_bn(w1, b1, g1, be1, m1, v1)

# -- Dense0 + BN2 --
wd0 = s["head.1.weight"].numpy()  # [32, 64]
bd0 = s["head.1.bias"].numpy()
g2  = s["head.2.weight"].numpy()
be2 = s["head.2.bias"].numpy()
m2  = s["head.2.running_mean"].numpy()
v2  = s["head.2.running_var"].numpy()
wd0f, bd0f = fold_bn(wd0, bd0, g2, be2, m2, v2)

# -- Dense1 (no BN) --
wd1 = s["head.5.weight"].numpy()  # [2, 32]
bd1 = s["head.5.bias"].numpy()

# ── build BN-free Keras model ─────────────────────────────────────────────────

inp = tf.keras.Input(shape=(N_FRAMES, N_MEL), name="log_mel")
x   = tf.keras.layers.Conv1D(32, 5, strides=2, padding="same",
                              activation="relu", use_bias=True)(inp)
x   = tf.keras.layers.Conv1D(64, 3, strides=2, padding="same",
                              activation="relu", use_bias=True)(x)
x   = tf.keras.layers.GlobalAveragePooling1D()(x)
x   = tf.keras.layers.Dense(32, activation="relu", use_bias=True)(x)
out = tf.keras.layers.Dense(2, use_bias=True)(x)
km  = tf.keras.Model(inp, out)

# PyTorch Conv1d weight: [out, in, k] → Keras Conv1D weight: [k, in, out]
def set_conv(layer, w, b):
    layer.set_weights([np.transpose(w, (2, 1, 0)), b])

# PyTorch Linear weight: [out, in] → Keras Dense weight: [in, out]
def set_dense(layer, w, b):
    layer.set_weights([w.T, b])

set_conv( km.layers[1], w0f, b0f)
set_conv( km.layers[2], w1f, b1f)
set_dense(km.layers[4], wd0f, bd0f)
set_dense(km.layers[5], wd1, bd1)

# ── sanity check ──────────────────────────────────────────────────────────────

dummy_np = np.random.randn(4, N_FRAMES, N_MEL).astype(np.float32)
dummy_pt = torch.tensor(dummy_np)
with torch.no_grad():
    out_pt = torch.softmax(pt(dummy_pt), dim=1).numpy()
out_keras = tf.nn.softmax(km(dummy_np)).numpy()
diff = np.abs(out_pt - out_keras).max()
print(f"Max output diff PyTorch vs Keras: {diff:.6f}  (target <0.01 after BN fold)")
if diff > 0.01:
    print("WARNING: large diff — check BN parameter key names match model.")

# ── representative dataset for calibration ────────────────────────────────────

import pathlib, soundfile as sf, torchaudio, sys
sys.path.insert(0, ".")
from speech_simulator import extract_logmel

def rep_data():
    """Use real speech samples if available, else random noise."""
    data_dir = pathlib.Path("speech_commands")
    samples = []
    if data_dir.exists():
        for wav_path in list(data_dir.glob("*/*.wav"))[:200]:
            try:
                wav, sr = sf.read(str(wav_path), dtype="float32")
                if wav.ndim > 1: wav = wav.mean(1)
                if len(wav) < 16000:
                    wav = np.pad(wav, (0, 16000 - len(wav)))
                feat = extract_logmel(wav[:16000])
                samples.append(feat[np.newaxis])
            except Exception:
                continue
    if len(samples) < 50:
        samples += [np.random.randn(1, N_FRAMES, N_MEL).astype(np.float32)
                    for _ in range(200 - len(samples))]
    for s in samples:
        yield [s.astype(np.float32)]

# ── int8 TFLite export ────────────────────────────────────────────────────────

conv = tf.lite.TFLiteConverter.from_keras_model(km)
conv.optimizations = [tf.lite.Optimize.DEFAULT]
conv.representative_dataset = rep_data
conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
conv.inference_input_type  = tf.int8
conv.inference_output_type = tf.int8

tflite_bytes = conv.convert()
open("tiny_vad_int8.tflite", "wb").write(tflite_bytes)
print(f"Saved tiny_vad_int8.tflite — {len(tflite_bytes)/1024:.1f} KB")
