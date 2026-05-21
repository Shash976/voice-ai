# export_weights.py
#
# Reads tiny_vad_int8.tflite, extracts all layer weights / biases /
# quantization parameters, and generates:
#
#   firmware/tinyengine_port/tiny_vad_weights.h   — C arrays and #defines
#
# Run:  python sw/tinyml_reference/export_weights.py
#
# Requires: tflite_runtime  (or tensorflow)

import sys, pathlib
import numpy as np

try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    import tensorflow.lite as tflite

ROOT = pathlib.Path(__file__).parent.parent.parent
TFLITE_PATH = ROOT / "tiny_vad_int8.tflite"
OUT_H       = ROOT / "firmware" / "tinyengine_port" / "tiny_vad_weights.h"

# ── load model ────────────────────────────────────────────────────────────────

interp = tflite.Interpreter(model_path=str(TFLITE_PATH))
interp.allocate_tensors()

tensor_details = {d["index"]: d for d in interp.get_tensor_details()}

def get_tensor(idx):
    return interp.get_tensor(idx).copy()

def quant(idx):
    d = tensor_details[idx]
    scale, zp = d["quantization"]
    return float(scale), int(zp)

# ── parse ops ─────────────────────────────────────────────────────────────────
# We iterate all ops, collect conv2d/fully_connected ops in order, and extract
# their input/weight/bias/output tensors.

ops = interp._get_ops_details()

conv_ops   = []  # (input_idx, weight_idx, bias_idx, output_idx)
dense_ops  = []
pool_ops   = []

OP_CONV2D          = "CONV_2D"
OP_DEPTHWISE       = "DEPTHWISE_CONV_2D"
OP_FC              = "FULLY_CONNECTED"
OP_AVG_POOL        = "AVERAGE_POOL_2D"
OP_MEAN            = "MEAN"

for op in ops:
    name = op.get("op_name", "")
    ins  = op["inputs"]
    outs = op["outputs"]
    if name in (OP_CONV2D, OP_DEPTHWISE):
        conv_ops.append((ins[0], ins[1], ins[2], outs[0]))
    elif name == OP_FC:
        dense_ops.append((ins[0], ins[1], ins[2], outs[0]))
    elif name in (OP_AVG_POOL, OP_MEAN):
        pool_ops.append((ins[0], outs[0]))

print(f"Found {len(conv_ops)} conv ops, {len(dense_ops)} dense ops, {len(pool_ops)} pool ops")

# ── quantized multiplier decomposition ───────────────────────────────────────

def quantize_multiplier(real_multiplier):
    """Decompose real_multiplier into (int32 Q31 fixed-point, right_shift >= 0).

    real_multiplier ≈ quantized_multiplier * 2^{-31} * 2^{-right_shift}

    This matches TFLite's QuantizeMultiplier() in tensorflow/lite/kernels/internal/quantization_utils.cc
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
    # s < 0 means left-shift (valid for pool layers where sc_in > sc_out)
    return q, s

def layer_multiplier(scale_in, scale_w, scale_out):
    return quantize_multiplier(float(scale_in) * float(scale_w) / float(scale_out))

# ── collect layer info ────────────────────────────────────────────────────────

layers = []

for i, (in_idx, w_idx, b_idx, out_idx) in enumerate(conv_ops):
    sc_in, zp_in   = quant(in_idx)
    sc_w,  zp_w    = quant(w_idx)
    sc_out, zp_out = quant(out_idx)
    W = get_tensor(w_idx)   # TFLite Conv2D stores as [out_ch, 1, 1, in_ch*k] for 1D via reshape
    B = get_tensor(b_idx)   # int32
    mult, rshift = layer_multiplier(sc_in, sc_w, sc_out)
    layers.append({
        "type": "conv", "idx": i,
        "W": W, "B": B,
        "sc_in": sc_in, "zp_in": zp_in,
        "sc_w":  sc_w,  "zp_w":  zp_w,
        "sc_out": sc_out, "zp_out": zp_out,
        "mult": mult, "rshift": rshift,
        "in_idx": in_idx, "out_idx": out_idx,
    })

for i, (in_idx, w_idx, b_idx, out_idx) in enumerate(dense_ops):
    sc_in, zp_in   = quant(in_idx)
    sc_w,  zp_w    = quant(w_idx)
    sc_out, zp_out = quant(out_idx)
    W = get_tensor(w_idx)   # [out, in]
    B = get_tensor(b_idx)   # int32
    mult, rshift = layer_multiplier(sc_in, sc_w, sc_out)
    layers.append({
        "type": "dense", "idx": i,
        "W": W, "B": B,
        "sc_in": sc_in, "zp_in": zp_in,
        "sc_w":  sc_w,  "zp_w":  zp_w,
        "sc_out": sc_out, "zp_out": zp_out,
        "mult": mult, "rshift": rshift,
        "in_idx": in_idx, "out_idx": out_idx,
    })

# pool layer effective scale
pool_info = None
if pool_ops:
    in_idx, out_idx = pool_ops[0]
    sc_in, zp_in   = quant(in_idx)
    sc_out, zp_out = quant(out_idx)
    pool_info = {"sc_in": sc_in, "zp_in": zp_in, "sc_out": sc_out, "zp_out": zp_out,
                 "in_idx": in_idx, "out_idx": out_idx}

# input tensor of the full model
input_idx  = interp.get_input_details()[0]["index"]
output_idx = interp.get_output_details()[0]["index"]
sc_input, zp_input   = quant(input_idx)
sc_output, zp_output = quant(output_idx)

# ── print summary ─────────────────────────────────────────────────────────────

print(f"\nInput  quantization: scale={sc_input:.6f} zp={zp_input}")
for l in layers:
    print(f"  [{l['type']} {l['idx']}] W{l['W'].shape}  "
          f"scale_in={l['sc_in']:.5f} zp_in={l['zp_in']}  "
          f"scale_out={l['sc_out']:.5f} zp_out={l['zp_out']}  "
          f"mult={l['mult']} rshift={l['rshift']}")
if pool_info:
    print(f"  [pool]  scale_in={pool_info['sc_in']:.5f} zp_in={pool_info['zp_in']}  "
          f"scale_out={pool_info['sc_out']:.5f} zp_out={pool_info['zp_out']}")
print(f"Output quantization: scale={sc_output:.6f} zp={zp_output}")

# ── C array helpers ───────────────────────────────────────────────────────────

def c_array_i8(name, arr):
    flat = arr.flatten().astype(np.int8)
    vals = ", ".join(str(int(v)) for v in flat)
    return f"static const int8_t {name}[{len(flat)}] = {{\n    {vals}\n}};\n"

def c_array_i32(name, arr):
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

lines.append(f"/* Model-level input/output quantization */")
lines.append(f"#define INPUT_SCALE   {sc_input}f")
lines.append(f"#define INPUT_ZP      {zp_input}")
lines.append(f"#define OUTPUT_SCALE  {sc_output}f")
lines.append(f"#define OUTPUT_ZP     {zp_output}")
lines.append("")

# conv layers
conv_layer_list = [l for l in layers if l["type"] == "conv"]
for i, l in enumerate(conv_layer_list):
    tag = f"CONV{i}"
    lines.append(f"/* {tag}: weight shape {l['W'].shape} */")
    lines.append(f"#define {tag}_SCALE_IN   {l['sc_in']}f")
    lines.append(f"#define {tag}_ZP_IN      {l['zp_in']}")
    lines.append(f"#define {tag}_SCALE_OUT  {l['sc_out']}f")
    lines.append(f"#define {tag}_ZP_OUT     {l['zp_out']}")
    lines.append(f"#define {tag}_MULT       {l['mult']}")
    lines.append(f"#define {tag}_RSHIFT     {l['rshift']}")
    lines.append(c_array_i8( f"{tag.lower()}_w", l["W"]))
    lines.append(c_array_i32(f"{tag.lower()}_b", l["B"]))

# pool
if pool_info:
    lines.append(f"/* POOL */")
    lines.append(f"#define POOL_SCALE_IN   {pool_info['sc_in']}f")
    lines.append(f"#define POOL_ZP_IN      {pool_info['zp_in']}")
    lines.append(f"#define POOL_SCALE_OUT  {pool_info['sc_out']}f")
    lines.append(f"#define POOL_ZP_OUT     {pool_info['zp_out']}")
    pool_mult, pool_rshift = quantize_multiplier(pool_info["sc_in"] / pool_info["sc_out"])
    lines.append(f"#define POOL_MULT       {pool_mult}")
    lines.append(f"#define POOL_RSHIFT     {pool_rshift}")
    lines.append("")

# dense layers
dense_layer_list = [l for l in layers if l["type"] == "dense"]
for i, l in enumerate(dense_layer_list):
    tag = f"FC{i}"
    lines.append(f"/* {tag}: weight shape {l['W'].shape} */")
    lines.append(f"#define {tag}_SCALE_IN   {l['sc_in']}f")
    lines.append(f"#define {tag}_ZP_IN      {l['zp_in']}")
    lines.append(f"#define {tag}_SCALE_OUT  {l['sc_out']}f")
    lines.append(f"#define {tag}_ZP_OUT     {l['zp_out']}")
    lines.append(f"#define {tag}_MULT       {l['mult']}")
    lines.append(f"#define {tag}_RSHIFT     {l['rshift']}")
    lines.append(c_array_i8( f"{tag.lower()}_w", l["W"]))
    lines.append(c_array_i32(f"{tag.lower()}_b", l["B"]))

lines.append("#endif /* TINY_VAD_WEIGHTS_H */")

OUT_H.parent.mkdir(parents=True, exist_ok=True)
OUT_H.write_text("\n".join(lines))
print(f"\nWrote {OUT_H}")
