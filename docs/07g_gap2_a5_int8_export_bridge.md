# 07g — Gap 2 A5: the int8 export format bridge

Stage 7 Gap 2's A5: turn A4's ORT calibration output into the actual
deployable blobs `firmware/quartznet/quartznet_infer.c` consumes — real
calibrated int8 weights, not `quartznet_ref.make_blobs()`'s seeded-random
placeholder.

## Result: G2.5 PASSES

```
$ python3 quartznet_export_int8.py --audio <real LibriSpeech clip> --t-out 70
...
  zero-point propagation cross-checked against real qparams: 187/187 descriptors consistent
  quartznet_weights.bin  18,847,040 B  (== EXPECTED_PARAMS: True)

$ python3 quartznet_ref.py --weights-from build/quartznet_real
...
transcript  : "as for etching"  (14 symbols)

$ ./test_quartznet_host build/quartznet_real
=== build/quartznet_real ===
  descriptors 187   T_out 70   T_in 140   T_TILE 32   DW_CH_TILE 64
  per-op   :  dw 77/77  pw 95/95  add 15/15
  activations: 187/187 descriptors bit-exact
  transcript : MATCH  (14 symbols)
PASS — 1/1 configurations bit-exact
```

Real English out of the real int8 descriptor blobs, matched byte-for-byte
by the C interpreter — `"as for etching"` on the first 1.4s of
`1272-128104-0008.flac` (ground truth: *"AS FOR ETCHINGS THEY ARE OF TWO
KINDS..."*). `make -C firmware/quartznet host-real` reproduces this from
scratch.

## Two real blockers found and fixed along the way

**`quartznet_descriptors.DescriptorTable` hardcoded fake zero points.**
`zp_out = -128 if relu else 0` is the *seeded-random* placeholder — random
weights have no real skew to calibrate a non-zero zero point against. Real
calibrated `out_zp` on non-ReLU layers ranges −78 to +93. Using the
placeholder on real weights would silently corrupt every downstream
descriptor's `in_zp`/`res_zp`/`out_zp` (they propagate via
`_resolve_zps()`). Fixed with an optional `zp_out=` override (backward
compatible — `None` keeps the placeholder for `make_blobs()`'s existing
seeded-random path). `quartznet_export_int8.py` passes the real per-descriptor
values; a cross-check inside the exporter (`tbl.zp_in[i]`/`tbl.zp_res[i]`
against the same values independently pulled from the ONNX graph) confirmed
0 mismatches across all 187 descriptors — a wrong ADD-node-to-descriptor
pairing or a bad zero point would very likely have broken this self-check.

**`export_weights.py` (needed for `quantize_multiplier`) couldn't be
imported.** It's a 250-line top-to-bottom script, not a module — importing
it unconditionally runs a `tflite_runtime`/`tensorflow` import at module
scope, and neither is installed on this machine (that TinyVAD export already
ran once, its output is committed). Refactored: `quantize_multiplier()` and
`per_channel_multipliers()` (the two functions Stage 7 needs) stay at module
level, dependency-free; everything TFLite-specific moved into a new `main()`.
No behavior change to the TinyVAD path — verified `quantize_multiplier`'s
output is unchanged by running it against known values, though the full
`tiny_vad_weights.h` regeneration itself couldn't be re-run here (no
tflite/tensorflow installed) to prove byte-identity end-to-end.

## The ADD requantization math — the plan's actual unfinished item

`quartznet_ref._acc_add`/`quartznet_infer.c`'s `tile_add` already implement
the real *structure* (pre-scale both operands into a shared domain via
`qmult[0]`/`qmult[1]`, combine, then `qmult[2]` into the output domain) —
only `make_blobs()`'s *values* were a placeholder
(`quantize_multiplier(0.5)` for both operands, correct only when both
scales are equal, never true on the real model: measured `s_main/s_res`
ranges 0.75–3.82 across the 15 real Adds). The real formula (TFLite
`reference_ops::Add`'s `twice_max` normalization):

```
twice_max = 2 * max(s_main, s_res)
qmult[0], rshift[0] = quantize_multiplier(s_main / twice_max)
qmult[1], rshift[1] = quantize_multiplier(s_res  / twice_max)
qmult[2], rshift[2] = quantize_multiplier(twice_max / (2**20 * s_out))
```

Verified numerically against 200,000 random int8 operand pairs per Add: 14
of 15 exact to 0 counts, 1 of 15 (the largest scale ratio, 3.82×) exact to
within 1 count — a `round()`-tie, not a formula error. Provably cannot
overflow int32: each operand multiplier is ≤0.5 by construction, so the
worst-case combined accumulator is 12.45% of int32 range for any input.

## Bias domain: A4's claim re-verified, not re-trusted

A4's docstring claims ORT's `bias_int32` is drop-in usable with zero
conversion (`bias_scale == in_scale * w_scale`). Re-checked independently
here (this session has found real bugs hiding behind confident-sounding
docstrings before — `BN_EPS`, `ctc_greedy`'s cast): `quartznet_ref.replay()`
computes `acc = Σ(in − in_zp)·w + bias`, with no `−in_zp·w` cross-term
folded into `bias` anywhere, matching exactly what a QDQ-format ONNX graph's
`Conv` node computes (`DequantizeLinear` recovers `Σ(q_x − zp_x)·q_w·s_x·s_w
+ b_i32·s_b`, which only equals the same thing when `s_b == s_x·s_w` —
already verified bit-exact for all 171 convs in A4). Confirmed correct;
`bias_int32_{lid}` really is drop-in.

## New / changed files

- **`sw/tinyml_reference/quartznet_export_int8.py`** (new) — the format
  bridge: loads A4's `qparams_ort.npz` + walks `quartznet_int8.onnx` for the
  15 Add nodes' real scales, builds the descriptor table with real zero
  points, fills the weight/qparam blobs (per-channel `(q_mult, rshift)` via
  `export_weights.quantize_multiplier`, imported not reimplemented), and
  quantizes a real mel input with the network's *actual* calibrated input
  scale/zero-point (`in_scale=0.038837`, `in_zp=-44` — not
  `quartznet_audio.IN_ZP_DEFAULT=-20`, which is the seeded-random
  placeholder's own convention). Also emits `quartznet_meta.json` (scales
  the descriptor table itself doesn't carry — needed to dequantize logits or
  re-quantize a new clip against this specific calibrated model).
- **`sw/tinyml_reference/quartznet_descriptors.py`** — `DescriptorTable`
  gains the `zp_out=` override described above.
- **`sw/tinyml_reference/export_weights.py`** — refactored into an
  importable module (see above); no behavior change.
- **`sw/tinyml_reference/quartznet_ref.py`** — new `--weights-from <dir>`
  flag: skips `make_blobs()`'s random-weight calibration entirely and
  re-executes real blobs via `replay()` (already "the pure re-execution path
  from emitted blobs alone" — exactly what this needs). `t_out` is *derived*
  from `quartznet_input.bin`'s own length, never taken from `--t-out`,
  removing the sharpest footgun in this design (an exporter/replay `t_out`
  mismatch would otherwise silently produce a golden the C interpreter can
  never match).
- **`firmware/quartznet/Makefile`** — new `goldens-real`/`host-real`
  targets, kept separate from `goldens`/`host` (which must keep working on a
  bare gcc+numpy machine with no venv, no LibriSpeech, no `onnx`).

## Verification

```
$ make -C firmware/quartznet host-real
...
PASS — 1/1 configurations bit-exact

$ make -C firmware/quartznet host
...
PASS — 2/2 configurations bit-exact   # unaffected regression
```
