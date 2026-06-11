# 01 — The Model & int8 Quantization (Stages 1–2)

This covers how TinyVAD is trained, how it becomes 8-bit integers, and how it turns
into C that runs everywhere. **Runs on Windows** (PyTorch/TensorFlow live there).

---

## What TinyVAD is

A tiny 1D convolutional neural network. Its only job is binary classification:
**speech vs. silence**.

- **Input:** 1 second of audio, turned into a *log-mel spectrogram* — a
  `[49 time frames × 40 mel-frequency bins]` int8 tensor.
- **Output:** 2 logits `[silence_score, speech_score]`. Bigger one wins.

### Architecture (5 layers)

| Layer | Operation | Output shape |
|-------|-----------|--------------|
| Conv0 (kernel=5, stride=2, pad=2) | Slide a 5-frame filter over time → 32 features | `[25 × 32]` |
| Conv1 (kernel=3, stride=2, pad=1) | Slide a 3-frame filter → 64 features | `[13 × 64]` |
| GlobalAvgPool | Average over the time axis → one vector | `[64]` |
| FC0 (dense) | Matrix-vector multiply 64→32 + ReLU | `[32]` |
| FC1 (dense) | Matrix-vector multiply 32→2 (the logits) | `[2]` |

Total work ≈ **242K multiply-accumulates** per inference. That number is why the
pure-software version is slow (~11.2M CPU cycles) and why the accelerator exists.

### The layout rule you must never break

**All tensors are stored in `[time, channel]` order** (time outer, channel inner) —
this matches TFLite's NHWC convention, *not* PyTorch's `[channel, time]`.

```
inp[t * n_channels + c]      // NOT inp[c * n_time + t]
```

> 🐛 Historical bug: swapping this layout produced completely wrong outputs. The C
> engine, the weight export, and the accelerator all assume `[time, channel]`. If
> you ever see garbage logits, suspect a layout mismatch first.

---

## Why int8?

The custom chip has **no floating-point unit**. Everything must be integer math.
Quantization also makes weights 4× smaller (8-bit vs 32-bit float), so the whole
model fits in a couple kilobytes of RAM.

### The quantization scheme (you'll see this everywhere)

```
real_value = scale * (int8_value − zero_point)
```

- **Inputs/activations:** one `scale` + one `zero_point` per tensor.
- **Weights:** *per-channel* int8 — each output channel has its own scale.
- **Requantization** (after a layer's int32 accumulation, to get back to int8):

  ```
  real_mult = (input_scale × weight_scale) / output_scale
  ```

  This float multiplier is decomposed into a fixed-point pair `(q_mult, rshift)`:

  ```
  result ≈ (accumulator × q_mult) >> 31 >> rshift      // shift can be negative = left shift
  ```

  See `requantize()` in
  [`../firmware/tinyengine_port/tiny_vad_infer.c:44`](../firmware/tinyengine_port/tiny_vad_infer.c)
  — int64 intermediate, rounds before the Q31 shift, handles negative shift (the
  pool layer needs a left shift). **This exact function is duplicated in three
  places** and they must stay identical:
  - the C engine (`tiny_vad_infer.c`)
  - the accelerator model (`accel_requantize()` in `sim/verilator/sim_main.cpp:81`)
  - (implicitly) the TFLite reference

---

## The files

| File | Role | Machine |
|------|------|---------|
| [`../train_tiny_vad.py`](../train_tiny_vad.py) | Train the CNN on Google Speech Commands v2 | Windows |
| [`../convert_to_tflite.py`](../convert_to_tflite.py) | `.pt` → ONNX → int8 `.tflite` | Windows |
| [`../sw/tinyml_reference/export_weights.py`](../sw/tinyml_reference/export_weights.py) | Read TFLite → emit `tiny_vad_weights.h` | Windows |
| [`../sw/tinyml_reference/gen_test_vectors.py`](../sw/tinyml_reference/gen_test_vectors.py) | Run TFLite on samples → emit `tiny_vad_test_vectors.h` | Windows |
| [`../firmware/tinyengine_port/tiny_vad_infer.c`](../firmware/tinyengine_port/tiny_vad_infer.c) | The hand-written int8 inference engine | both |
| `../firmware/tinyengine_port/tiny_vad_weights.h` | **Auto-generated** weight tables — do not edit | — |
| `../firmware/tinyengine_port/tiny_vad_test_vectors.h` | **Auto-generated** 64 test inputs + labels — do not edit | — |
| [`../firmware/tinyengine_port/test_infer_host.c`](../firmware/tinyengine_port/test_infer_host.c) | x86 harness: C engine vs TFLite | WSL/x86 |

### The C inference engine in one breath

[`tiny_vad_infer.c`](../firmware/tinyengine_port/tiny_vad_infer.c) is plain,
portable C — **no stdlib, no malloc**. It uses four static scratch buffers
(`buf0`..`buf3`, ~2 KB total) so it can run bare-metal on the RISC-V chip. The public
entry point is:

```c
int tiny_vad_infer(const int8_t *input, int8_t *logits);
// returns 1 if speech (logits[1] > logits[0]), else 0
```

Internally it just calls `conv1d → conv1d → global_avg_pool → dense → dense`.

#### The hook trick (important — this is how Stage 4 plugs in)

`conv1d()` and `dense()` each start with:

```c
if (tinyvad_conv1d_hook) { tinyvad_conv1d_hook(...); return; }
```

Two global function pointers (`tinyvad_conv1d_hook`, `tinyvad_dense_hook`) default to
`NULL` (= run the software path). When the firmware sets them to the accelerator
driver functions, *every* conv/dense call transparently goes to hardware instead —
the inference code itself is unchanged. That's the entire Stage-3 → Stage-4 switch.

---

## How to run it (Windows, in the Python venv)

```powershell
# Step 1 — train (downloads ~2.3 GB Speech Commands dataset on first run)
python train_tiny_vad.py
#   → tiny_vad_best.pt, tiny_vad.onnx

# Step 2 — quantize to int8 TFLite
python convert_to_tflite.py
#   → tiny_vad_int8.tflite       (the source of truth for weights + vectors)

# Step 3 — regenerate the C headers (RE-RUN THESE WHENEVER THE MODEL CHANGES)
python sw/tinyml_reference/export_weights.py     # → tiny_vad_weights.h
python sw/tinyml_reference/gen_test_vectors.py   # → tiny_vad_test_vectors.h
```

> You usually do **not** need to retrain. The trained model + headers are already
> checked in. You only re-run steps 1–3 if you deliberately change the architecture.

---

## Sanity check the C engine (WSL or any x86)

```bash
cd firmware/tinyengine_port
make host          # gcc → x86 binary test_infer_host
./test_infer_host  # runs the C engine against the baked-in vectors
```

Expected: 64/64 vectors pass, **max logit error vs TFLite ≤ 3 LSB**.

"3 LSB" = the int8 C engine differs from the TFLite float reference by at most 3
integer counts — quantization plus accumulation-order rounding (the C engine
iterates layers in a different loop order than TFLite's kernels; worst case is
one vector at 3 LSB, predicted labels unaffected). This proves the C engine is correct
*before* you cross-compile it for RISC-V. If this fails, fix it here first; do not
proceed to the simulator.

---

## Mental model / what to remember

- The `.tflite` file is the **single source of truth**. Everything downstream
  (weights header, test vectors, C math) is derived from it and validated against it.
- Three implementations of the same int8 math must agree bit-for-bit. Correctness =
  "64/64 vectors pass" at every stage.
- `[time, channel]` layout, always.
- Two headers are generated — regenerate, don't edit.

Next: [02_firmware_and_simulation.md](02_firmware_and_simulation.md) — getting this
engine running on a simulated RISC-V CPU.
