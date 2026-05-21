# export_weights.py
#
# Reads tiny_vad_int8.tflite, extracts all layer weights / biases /
# quantization parameters, and generates:
#
#   firmware/tinyengine_port/tiny_vad_weights.h
#
# Run:  python sw/tinyml_reference/export_weights.py
#
# TFLite uses per-channel weight quantization by default.
# d["quantization"] returns (0.0, 0) for per-channel tensors;
# actual per-channel scales live in d["quantization_parameters"]["scales"].

import sys, pathlib
import numpy as np

try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    import tensorflow.lite as tflite

ROOT        = pathlib.Path(__file__).parent.parent.parent
TFLITE_PATH = ROOT / "tiny_vad_int8.tflite"
OUT_H       = ROOT / "firmware" / "tinyengine_port" / "tiny_vad_weights.h"

# ── load model ────────────────────────────────────────────────────────────────

interp = tflite.Interpreter(model_path=str(TFLITE_PATH))
interp.allocate_tensors()
tensor_details = {d["index"]: d for d in interp.get_tensor_details()}

def get_tensor(idx):
    return interp.get_tensor(idx).copy()

def per_tensor_quant(idx):
    """Returns (scale: float, zp: int) for activation tensors (always per-tensor)."""
    d = tensor_details[idx]
    scale, zp = d["quantization"]
    if scale == 0.0:
        # fall back to quantization_parameters
        qp = d.get("quantization_parameters", {})
        scales = qp.get("scales", [0.0])
        zps    = qp.get("zero_points", [0])
        scale, zp = float(scales[0]), int(zps[0])
    return float(scale), int(zp)

def per_channel_weight_quant(idx):
    """Returns (scales: float[], zps: int[]) for weight tensors.
    TFLite per-channel weights have scales[out_ch] and zp=0 for all."""
    d = tensor_details[idx]
    qp = d.get("quantization_parameters", {})
    scales = qp.get("scales", None)
    zps    = qp.get("zero_points", None)
    if scales is None or len(scales) == 0:
        # per-tensor fallback
        sc, zp = d["quantization"]
        n = d["shape"][0]
        return np.full(n, sc, dtype=np.float64), np.zeros(n, dtype=np.int32)
    return np.array(scales, dtype=np.float64), np.array(zps, dtype=np.int32)

# ── quantized multiplier decomposition ───────────────────────────────────────

def quantize_multiplier(real_multiplier):
    """Decompose real_multiplier → (int32 Q31 fixed-point, int shift).

    real_multiplier ≈ q * 2^{-31} * 2^{-shift}

    shift > 0 = right-shift (typical conv/dense)
    shift < 0 = left-shift  (can occur for pool when sc_in > sc_out)
    """
    if real_multiplier == 0.0:
        return 0, 0
    s = 0
    while real_multiplier < 0.5:
        real_multiplier *= 2.0
        s += 1
    while real_multiplier >= 1.0:
        real_multiplier /= 2.0
        s -= 1
    q = int(round(real_multiplier * (1 << 31)))
    if q == (1 << 31):
        q //= 2
        s -= 1
    assert 0 <= q <= (1 << 31), f"bad q={q}"
    return q, s

def per_channel_multipliers(scale_in, weight_scales, scale_out):
    """Compute per-channel (mult[], rshift[]) arrays."""
    mults  = []
    rshifts = []
    for sc_w in weight_scales:
        m, s = quantize_multiplier(float(scale_in) * float(sc_w) / float(scale_out))
        mults.append(m)
        rshifts.append(s)
    return np.array(mults, dtype=np.int32), np.array(rshifts, dtype=np.int32)

# ── parse ops ─────────────────────────────────────────────────────────────────

ops = interp._get_ops_details()

conv_ops  = []
dense_ops = []
pool_ops  = []

for op in ops:
    name = op.get("op_name", "")
    ins, outs = op["inputs"], op["outputs"]
    if name in ("CONV_2D", "DEPTHWISE_CONV_2D"):
        conv_ops.append((ins[0], ins[1], ins[2], outs[0]))
    elif name == "FULLY_CONNECTED":
        dense_ops.append((ins[0], ins[1], ins[2], outs[0]))
    elif name in ("AVERAGE_POOL_2D", "MEAN"):
        pool_ops.append((ins[0], outs[0]))

print(f"Found {len(conv_ops)} conv ops, {len(dense_ops)} dense ops, {len(pool_ops)} pool ops")

# ── collect layer info ────────────────────────────────────────────────────────

layers = []

for i, (in_idx, w_idx, b_idx, out_idx) in enumerate(conv_ops):
    sc_in,  zp_in  = per_tensor_quant(in_idx)
    sc_out, zp_out = per_tensor_quant(out_idx)
    w_scales, w_zps = per_channel_weight_quant(w_idx)

    W = get_tensor(w_idx)   # [out_ch, 1, k, in_ch]  (TFLite Conv2D layout)
    B = get_tensor(b_idx)   # int32 [out_ch]

    # Transpose to [out_ch, in_ch, k] for the C conv1d layout
    W = W.squeeze(1)                # [out_ch, k, in_ch]
    W = W.transpose(0, 2, 1)        # [out_ch, in_ch, k]

    mults, rshifts = per_channel_multipliers(sc_in, w_scales, sc_out)
    layers.append({"type": "conv", "idx": i,
                   "W": W, "B": B,
                   "sc_in": sc_in, "zp_in": zp_in,
                   "sc_out": sc_out, "zp_out": zp_out,
                   "mults": mults, "rshifts": rshifts,
                   "w_scales": w_scales})

for i, (in_idx, w_idx, b_idx, out_idx) in enumerate(dense_ops):
    sc_in,  zp_in  = per_tensor_quant(in_idx)
    sc_out, zp_out = per_tensor_quant(out_idx)
    w_scales, w_zps = per_channel_weight_quant(w_idx)

    W = get_tensor(w_idx)   # [out, in]  (TFLite FC layout)
    B = get_tensor(b_idx)

    mults, rshifts = per_channel_multipliers(sc_in, w_scales, sc_out)
    layers.append({"type": "dense", "idx": i,
                   "W": W, "B": B,
                   "sc_in": sc_in, "zp_in": zp_in,
                   "sc_out": sc_out, "zp_out": zp_out,
                   "mults": mults, "rshifts": rshifts,
                   "w_scales": w_scales})

pool_info = None
if pool_ops:
    in_idx, out_idx = pool_ops[0]
    sc_in,  zp_in  = per_tensor_quant(in_idx)
    sc_out, zp_out = per_tensor_quant(out_idx)
    pool_mult, pool_rshift = quantize_multiplier(sc_in / sc_out)
    pool_info = {"sc_in": sc_in, "zp_in": zp_in,
                 "sc_out": sc_out, "zp_out": zp_out,
                 "mult": pool_mult, "rshift": pool_rshift}

input_idx  = interp.get_input_details()[0]["index"]
output_idx = interp.get_output_details()[0]["index"]
sc_input,  zp_input  = per_tensor_quant(input_idx)
sc_output, zp_output = per_tensor_quant(output_idx)

# ── print summary ─────────────────────────────────────────────────────────────

print(f"\nInput  quantization: scale={sc_input:.6f} zp={zp_input}")
for l in layers:
    nz = np.count_nonzero(l["mults"])
    print(f"  [{l['type']} {l['idx']}] W{l['W'].shape}  "
          f"zp_in={l['zp_in']}  zp_out={l['zp_out']}  "
          f"non-zero mults: {nz}/{len(l['mults'])}  "
          f"rshift range: [{l['rshifts'].min()},{l['rshifts'].max()}]")
if pool_info:
    print(f"  [pool]  sc_in={pool_info['sc_in']:.5f} zp_in={pool_info['zp_in']}  "
          f"sc_out={pool_info['sc_out']:.5f} zp_out={pool_info['zp_out']}  "
          f"mult={pool_info['mult']} rshift={pool_info['rshift']}")
print(f"Output quantization: scale={sc_output:.6f} zp={zp_output}")

# ── C array helpers ───────────────────────────────────────────────────────────

def c_i8_array(name, arr):
    flat = arr.flatten().astype(np.int8)
    vals = ", ".join(str(int(v)) for v in flat)
    return f"static const int8_t {name}[{len(flat)}] = {{\n    {vals}\n}};\n"

def c_i32_array(name, arr):
    flat = arr.flatten().astype(np.int32)
    vals = ", ".join(str(int(v)) for v in flat)
    return f"static const int32_t {name}[{len(flat)}] = {{\n    {vals}\n}};\n"

# ── generate header ───────────────────────────────────────────────────────────

lines = []
lines.append("/* tiny_vad_weights.h — auto-generated by export_weights.py. DO NOT EDIT. */")
lines.append("#ifndef TINY_VAD_WEIGHTS_H")
lines.append("#define TINY_VAD_WEIGHTS_H")
lines.append("#include <stdint.h>")
lines.append("")
lines.append("/* Model-level input/output quantization */")
lines.append(f"#define INPUT_SCALE   {sc_input}f")
lines.append(f"#define INPUT_ZP      {zp_input}")
lines.append(f"#define OUTPUT_SCALE  {sc_output}f")
lines.append(f"#define OUTPUT_ZP     {zp_output}")
lines.append("")

conv_layers  = [l for l in layers if l["type"] == "conv"]
dense_layers = [l for l in layers if l["type"] == "dense"]

for i, l in enumerate(conv_layers):
    tag = f"CONV{i}"
    out_ch = l["W"].shape[0]
    lines.append(f"/* {tag}: W{l['W'].shape} */")
    lines.append(f"#define {tag}_ZP_IN   {l['zp_in']}")
    lines.append(f"#define {tag}_ZP_OUT  {l['zp_out']}")
    lines.append(c_i8_array( f"{tag.lower()}_w",      l["W"]))
    lines.append(c_i32_array(f"{tag.lower()}_b",      l["B"]))
    lines.append(c_i32_array(f"{tag.lower()}_mult",   l["mults"]))
    lines.append(c_i32_array(f"{tag.lower()}_rshift", l["rshifts"]))

if pool_info:
    lines.append("/* POOL */")
    lines.append(f"#define POOL_ZP_IN   {pool_info['zp_in']}")
    lines.append(f"#define POOL_ZP_OUT  {pool_info['zp_out']}")
    lines.append(f"#define POOL_MULT    {pool_info['mult']}")
    lines.append(f"#define POOL_RSHIFT  {pool_info['rshift']}")
    lines.append("")

for i, l in enumerate(dense_layers):
    tag = f"FC{i}"
    lines.append(f"/* {tag}: W{l['W'].shape} */")
    lines.append(f"#define {tag}_ZP_IN   {l['zp_in']}")
    lines.append(f"#define {tag}_ZP_OUT  {l['zp_out']}")
    lines.append(c_i8_array( f"{tag.lower()}_w",      l["W"]))
    lines.append(c_i32_array(f"{tag.lower()}_b",      l["B"]))
    lines.append(c_i32_array(f"{tag.lower()}_mult",   l["mults"]))
    lines.append(c_i32_array(f"{tag.lower()}_rshift", l["rshifts"]))

lines.append("#endif /* TINY_VAD_WEIGHTS_H */")

OUT_H.parent.mkdir(parents=True, exist_ok=True)
OUT_H.write_text("\n".join(lines))
print(f"\nWrote {OUT_H}")
