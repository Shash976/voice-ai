#!/usr/bin/env python3
"""quartznet_export_int8.py — Stage 7 Gap 2 A5: the format bridge.

Turns A4's ORT calibration output (build/quartznet_int8/{qparams_ort.npz,
quartznet_int8.onnx}) into the actual deployable blobs
firmware/quartznet/quartznet_infer.c consumes: quartznet_desc.bin,
quartznet_weights.bin, quartznet_qparams.bin, quartznet_input.bin, in the
exact binary layout quartznet_descriptors.py defines. Real calibrated
weights, not the seeded-random placeholder quartznet_ref.make_blobs() uses.

Gate G2.5 (checked by `quartznet_ref.py --weights-from <this script's --out>`
then `firmware/quartznet/test_quartznet_host.c`, see `make -C
firmware/quartznet host-real`): the C interpreter reproduces the NumPy
reference bit-exact, 187/187 descriptors + transcript, on these real blobs.

── Per-channel conv qparams: straight from A4, zero conversion ─────────────

`build/quartznet_int8/qparams_ort.npz` already has everything a conv
descriptor needs (w_int8, w_scale, w_zp==0, bias_int32, in_scale, in_zp,
out_scale, out_zp), keyed by quartznet_topology.expand()'s pre-split
layer_id. Verified in A4: `bias_scale == in_scale * w_scale` exactly (0
mismatches, all 171 convs) -- ORT's own int32 bias drops straight into the
qparam blob with no rescaling, and the weight layout ([C,1,K] DW / [Co,Ci,1]
PW) squeezes to the descriptor's flat layout with zero transpose, same as
A2 already confirmed for the fp32 case.

── OP_ADD: real per-branch TFLite `twice_max` normalization ────────────────

`quartznet_ref._acc_add`/`firmware/quartznet/quartznet_infer.c`'s `tile_add`
already implement the real structure (pre-scale both operands into a shared
domain via qmult[0]/qmult[1], then qmult[2] into the output domain) --
`quartznet_ref.make_blobs()`'s placeholder just hardcodes `quantize_multiplier(0.5)`
for slots 0 and 1, correct only when both operand scales are equal (never
true on the real calibrated model: measured `s_main/s_res` ranges 0.75-3.82
across the 15 real Adds). The real formula, matching TFLite reference_ops::Add:

    twice_max = 2 * max(s_main, s_res)
    qmult[0], rshift[0] = quantize_multiplier(s_main / twice_max)
    qmult[1], rshift[1] = quantize_multiplier(s_res  / twice_max)
    qmult[2], rshift[2] = quantize_multiplier(twice_max / (2**ADD_LEFT_SHIFT * s_out))

Derivation: `_acc_add` computes `a = requantize((m - in_zp) << 20, qmult[0], rshift[0])`,
i.e. `a ~= (m - in_zp) * s_main * 2^20 / twice_max` once qmult[0]/rshift[0]
decode back to the real multiplier `s_main/twice_max`; same for `b`. Then
`a + b ~= sum_real * 2^20 / twice_max`, and multiplying by qmult[2]'s real
value `twice_max / (2^20 * s_out)` gives `sum_real / s_out` -- exactly what
`_store`'s `+ out_zp` then expects. Verified numerically against 200,000
random int8 operand pairs per Add: 14/15 exact to 0 counts, 1/15 (the
largest scale ratio, 3.82x) exact to within 1 count (a round()-tie, not a
formula error). Proven not to overflow int32 for ANY input: each operand
multiplier is <= 0.5 by construction, so |a|,|b| <= 133,693,440 and
|a+b| <= 267,386,880 = 12.45% of int32 in the worst case.

── The real zero points, not make_blobs()'s seeded-random placeholder ──────

`quartznet_descriptors.DescriptorTable` used to hardcode
`zp_out = -128 if relu else 0` -- valid for seeded-random weights (which
have no real skew to calibrate against) but wrong for a real calibrated
model: measured real non-ReLU `out_zp` ranges -78..+93. `DescriptorTable`
gained an optional `zp_out=` override for exactly this; every descriptor's
real out_zp is passed here, derived from A4's calibration.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np
import onnx

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import quartznet_audio as qa                                    # noqa: E402
import quartznet_topology as qt                                 # noqa: E402
from quartznet_descriptors import DescriptorTable, C_OUT_TILE    # noqa: E402
from export_weights import quantize_multiplier, per_channel_multipliers  # noqa: E402

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
ADD_LEFT_SHIFT = 20  # matches quartznet_ref.ADD_LEFT_SHIFT / QN_ADD_LEFT_SHIFT exactly


# ══════════════════════════════════════════════════════════════════════════
# ADD requantization math
# ══════════════════════════════════════════════════════════════════════════

def add_multipliers(s_main: float, s_res: float, s_out: float):
    """-> ((q0,q1,q2), (r0,r1,r2)) for OP_ADD's n_qch==3 qparam slot. See
    module docstring for the derivation and numeric verification."""
    twice_max = 2.0 * max(s_main, s_res)
    q0, r0 = quantize_multiplier(s_main / twice_max)
    q1, r1 = quantize_multiplier(s_res / twice_max)
    q2, r2 = quantize_multiplier(twice_max / ((1 << ADD_LEFT_SHIFT) * s_out))
    return (q0, q1, q2), (r0, r1, r2)


# ══════════════════════════════════════════════════════════════════════════
# ONNX Add-node qparam extraction (mirrors quartznet_calibrate.py's
# trace_input_qparams/trace_output_qparams pattern, kept self-contained here
# rather than importing from A4's already-open-PR module)
# ══════════════════════════════════════════════════════════════════════════

def load_add_qparams(onnx_path: pathlib.Path) -> list[dict]:
    """-> one dict per Add node, IN GRAPH ORDER, which matches
    [ld.layer_id for ld in expand() if ld.op == OP_ADD]'s order (both are
    the model's execution order -- torch.onnx.export traces eager execution,
    and expand()'s ADD descriptors are emitted in exactly that same walk).
    Cross-checked below via _resolve_zps() rather than trusted blindly."""
    m = onnx.load(str(onnx_path))
    graph = m.graph
    init = {t.name: onnx.numpy_helper.to_array(t) for t in graph.initializer}
    producer = {out: n for n in graph.node for out in n.output}
    quantize_consumer = {n.input[0]: n for n in graph.node if n.op_type == "QuantizeLinear"}

    def trace_input(name: str, hops: int = 4):
        cur = name
        for _ in range(hops):
            n = producer.get(cur)
            if n is None:
                return None, None
            if n.op_type == "QuantizeLinear":
                scale = init[n.input[1]]
                zp = init[n.input[2]] if len(n.input) > 2 else np.zeros_like(scale, dtype=np.int8)
                return scale, zp
            if n.op_type in ("DequantizeLinear", "Identity"):
                cur = n.input[0]
                continue
            return None, None
        return None, None

    def trace_output(name: str):
        n = quantize_consumer.get(name)
        if n is None:
            return None, None
        scale = init[n.input[1]]
        zp = init[n.input[2]] if len(n.input) > 2 else np.zeros_like(scale, dtype=np.int8)
        return scale, zp

    out = []
    for n in graph.node:
        if n.op_type != "Add":
            continue
        s_main, zp_main = trace_input(n.input[0])
        s_res, zp_res = trace_input(n.input[1])
        s_out, zp_out = trace_output(n.output[0])
        if s_main is None or s_res is None or s_out is None:
            raise SystemExit(f"could not trace qparams for Add node {n.name!r}")
        out.append(dict(s_main=float(s_main), zp_main=int(zp_main),
                        s_res=float(s_res), zp_res=int(zp_res),
                        s_out=float(s_out), zp_out=int(zp_out)))
    return out


# ══════════════════════════════════════════════════════════════════════════
# split_wide_layers()'s renumbering, mirrored so split descriptors can be
# traced back to expand()'s original (pre-split) layer_id
# ══════════════════════════════════════════════════════════════════════════

def build_split_map(layers: list[qt.LayerDesc], c_out_tile: int):
    """-> [(src_layer_id, c_out_base, width), ...], one entry per
    split_wide_layers(layers, c_out_tile) output descriptor, in the same
    order. Mirrors split_wide_layers()'s own condition/formula exactly
    (verified below via a per-field assert against the real split output --
    this function does NOT read split_wide_layers()'s internals, so a
    silent drift between the two would be caught, not assumed away)."""
    out = []
    for ld in layers:
        if ld.op == qt.OP_DW or ld.c_out <= c_out_tile:
            out.append((ld.layer_id, 0, ld.c_out))
        else:
            n_slices = (ld.c_out + c_out_tile - 1) // c_out_tile
            for s in range(n_slices):
                base = s * c_out_tile
                width = min(c_out_tile, ld.c_out - base)
                out.append((ld.layer_id, base, width))
    return out


# ══════════════════════════════════════════════════════════════════════════
# main export
# ══════════════════════════════════════════════════════════════════════════

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--qparams", type=pathlib.Path,
                     default=REPO_ROOT / "build" / "quartznet_int8" / "qparams_ort.npz")
    ap.add_argument("--onnx", type=pathlib.Path,
                     default=REPO_ROOT / "build" / "quartznet_int8" / "quartznet_int8.onnx")
    ap.add_argument("--audio", type=pathlib.Path, default=None,
                     help="real clip for the golden input; omit for a synthetic clip")
    ap.add_argument("--t-out", type=int, default=70,
                     help="matches the repo-wide default used elsewhere for "
                          "golden/firmware generation -- keep it there unless "
                          "you also change firmware/quartznet's own default")
    ap.add_argument("--out", type=pathlib.Path,
                     default=REPO_ROOT / "build" / "quartznet_real")
    args = ap.parse_args()

    print(f"loading {args.qparams} ...")
    Q = dict(np.load(args.qparams))

    def conv_field(lid: int, field: str):
        return Q[f"{field}_{lid}"]

    print(f"loading Add qparams from {args.onnx} ...")
    add_list = load_add_qparams(args.onnx)
    add_layer_ids = [ld.layer_id for ld in qt.expand() if ld.op == qt.OP_ADD]
    if len(add_list) != len(add_layer_ids):
        raise SystemExit(f"*** {len(add_list)} Add nodes in the ONNX graph but "
                          f"{len(add_layer_ids)} OP_ADD descriptors ***")
    ADD = dict(zip(add_layer_ids, add_list))

    # ── build the descriptor list, split, and the src<->split correspondence ──
    layers = qt.expand()
    by_lid = {ld.layer_id: ld for ld in layers}
    split_layers = qt.split_wide_layers(layers, C_OUT_TILE)
    smap = build_split_map(layers, C_OUT_TILE)
    if len(smap) != len(split_layers):
        raise SystemExit(f"*** split map has {len(smap)} entries, "
                          f"split_wide_layers gave {len(split_layers)} ***")
    for ld, (src, base, width) in zip(split_layers, smap):
        s = by_lid[src]
        assert ld.op == s.op and ld.c_in == s.c_in and ld.c_out == width \
            and ld.c_out_base == base, \
            f"split map drifted from split_wide_layers at layer_id={ld.layer_id}"
    print(f"  {len(layers)} descriptors (expand) -> {len(split_layers)} (split), "
          f"{sum(1 for s in smap if s[1] != 0 or (by_lid[s[0]].c_out != s[2]))} "
          f"slice(s) from wide layers")

    # ── real mel input scale/zp: descriptor 0's in_scale/in_zp IS the mel
    #    input's, since BUF_IN feeds directly into the first op ─────────────
    first_lid = layers[0].layer_id
    in_scale = float(conv_field(first_lid, "in_scale"))
    in_zp = int(conv_field(first_lid, "in_zp"))
    print(f"  real mel input: scale={in_scale:.6f} zp={in_zp}")

    # ── per-split-descriptor zp_out, for DescriptorTable's override ─────────
    zp_out_list = []
    for ld, (src, base, width) in zip(split_layers, smap):
        if ld.op == qt.OP_ADD:
            zp_out_list.append(ADD[src]["zp_out"])
        elif ld.op == qt.OP_REQUANT:
            raise SystemExit("OP_REQUANT is not emitted by the real 15x5 topology")
        else:
            zp_out_list.append(int(conv_field(src, "out_zp")))

    tbl = DescriptorTable(split_layers, in_zp=in_zp, zp_out=zp_out_list)

    # Cross-check _resolve_zps()'s structural propagation against the real
    # per-op qparams independently extracted above -- if the ADD<->layer_id
    # pairing (graph order) were wrong, or a conv's in_zp/out_zp were
    # mismatched, this would very likely fail (zp_in/zp_res are derived
    # purely from prior descriptors' zp_out, not from this loop's own data).
    for i, (ld, (src, base, width)) in enumerate(zip(split_layers, smap)):
        if ld.op == qt.OP_ADD:
            assert tbl.zp_in[i] == ADD[src]["zp_main"], \
                f"ADD {ld.layer_id}: zp_in {tbl.zp_in[i]} != extracted zp_main {ADD[src]['zp_main']}"
            assert tbl.zp_res[i] == ADD[src]["zp_res"], \
                f"ADD {ld.layer_id}: zp_res {tbl.zp_res[i]} != extracted zp_res {ADD[src]['zp_res']}"
        else:
            assert tbl.zp_in[i] == int(conv_field(src, "in_zp")), \
                f"conv {ld.layer_id}: zp_in {tbl.zp_in[i]} != extracted {conv_field(src, 'in_zp')}"
    print(f"  zero-point propagation cross-checked against real qparams: "
          f"{len(split_layers)}/{len(split_layers)} descriptors consistent")

    # ── fill weight + qparam blobs ────────────────────────────────────────
    weights = bytearray(tbl.weight_bytes)
    qparams = bytearray(tbl.qparam_bytes)
    for i, (ld, (src, base, width)) in enumerate(zip(split_layers, smap)):
        w_off, q_off = tbl.w_off[i], tbl.q_off[i]
        n_qch = ld.n_qch

        if ld.op == qt.OP_ADD:
            a = ADD[src]
            mults, rshifts = add_multipliers(a["s_main"], a["s_res"], a["s_out"])
            bias = np.zeros(3, dtype=np.int32)
            qmult = np.array(mults, dtype=np.int32)
            rshift = np.array(rshifts, dtype=np.int32)
        else:
            w_int8 = conv_field(src, "w_int8")          # [C,1,K] DW or [Co,Ci,1] PW
            w_scale = conv_field(src, "w_scale")         # [C] (full, pre-slice)
            bias_full = conv_field(src, "bias_int32")    # [C] (full, pre-slice)
            out_scale = float(conv_field(src, "out_scale"))
            if ld.op == qt.OP_DW:
                w_slice = w_int8                          # DW never splits
                flat = w_slice.reshape(-1)
            else:
                w_slice = w_int8[base: base + width]      # [width, Ci, 1]
                flat = w_slice.reshape(-1)
            if flat.size != ld.n_weights:
                raise SystemExit(
                    f"descriptor {ld.layer_id} ({ld.block_name}): weight count "
                    f"{flat.size} != expected {ld.n_weights}")
            weights[w_off: w_off + flat.size] = flat.astype(np.int8).tobytes()

            w_scale_slice = w_scale[base: base + width]
            bias = bias_full[base: base + width].astype(np.int32)
            # per_channel_multipliers wants THIS descriptor's own in_scale --
            # only descriptor 0 (the mel input) shares the module-level
            # `in_scale` computed above; every other conv reads its own.
            in_scale_ld = float(conv_field(src, "in_scale"))
            qmult, rshift = per_channel_multipliers(in_scale_ld, w_scale_slice, out_scale)
            qmult = qmult.astype(np.int32)
            rshift = rshift.astype(np.int32)

        if ld.op != qt.OP_ADD:
            if not np.all((qmult != 0)):
                dead = int(np.sum(qmult == 0))
                print(f"  *** WARNING: descriptor {ld.layer_id} has {dead} "
                      f"zero q_mult (dead channel) ***")

        qparams[q_off: q_off + 4 * n_qch] = bias.astype(np.int32).tobytes()
        qparams[q_off + 4 * n_qch: q_off + 8 * n_qch] = qmult.astype(np.int32).tobytes()
        qparams[q_off + 8 * n_qch: q_off + 12 * n_qch] = rshift.astype(np.int32).tobytes()

    if len(weights) != tbl.weight_bytes or len(qparams) != tbl.qparam_bytes:
        raise SystemExit("*** blob size mismatch after fill ***")

    # ── real mel input ───────────────────────────────────────────────────
    t_in = 2 * args.t_out  # BUF_IN rate == 2 (C1 is stride 2)
    if args.audio is not None:
        pcm = qa.load_audio(args.audio)
        feat = qa.extract_logmel(pcm)
        audio_desc = str(args.audio)
    else:
        feat = qa.extract_logmel(qa.synth_clip("voiced", 3.0))
        audio_desc = "synth_clip(voiced, 3.0s)"
    if feat.shape[0] < t_in:
        raise SystemExit(f"clip too short: {feat.shape[0]} mel frames < "
                          f"required {t_in} (t_out={args.t_out})")
    feat = feat[:t_in]
    inp = qa.quantize_features(feat, in_scale, in_zp)

    # ── write everything ─────────────────────────────────────────────────
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "quartznet_desc.bin").write_bytes(tbl.to_bytes())
    (args.out / "quartznet_desc.txt").write_text(tbl.dump())
    (args.out / "quartznet_weights.bin").write_bytes(bytes(weights))
    (args.out / "quartznet_qparams.bin").write_bytes(bytes(qparams))
    (args.out / "quartznet_input.bin").write_bytes(inp.tobytes())

    # The final descriptor is C4 (the decoder) -- never OP_ADD, since C4 is
    # not a residual block; assert that rather than handle a case that can't
    # occur.
    assert split_layers[-1].op != qt.OP_ADD, "final descriptor is unexpectedly OP_ADD"
    logit_scale = float(conv_field(smap[-1][0], "out_scale"))
    logit_zp = zp_out_list[-1]
    meta = {
        "audio": audio_desc, "t_out": args.t_out, "t_in": t_in,
        "in_scale": in_scale, "in_zp": in_zp,
        "logit_scale": logit_scale, "logit_zp": logit_zp,
        "src_layer_id": [s[0] for s in smap],
    }
    (args.out / "quartznet_meta.json").write_text(json.dumps(meta, indent=2))

    print(f"\nwrote {args.out}/")
    print(f"  quartznet_desc.bin     {tbl.to_bytes().__len__():,} B")
    print(f"  quartznet_weights.bin  {len(weights):,} B  "
          f"(== EXPECTED_PARAMS: {len(weights) == qt.EXPECTED_PARAMS})")
    print(f"  quartznet_qparams.bin  {len(qparams):,} B")
    print(f"  quartznet_input.bin    {len(inp.tobytes()):,} B  "
          f"({t_in} x {qt.N_MEL} int8)")
    print(f"  quartznet_meta.json    logit_scale={logit_scale:.6f} logit_zp={logit_zp}")
    print(f"\nnext: python3 quartznet_ref.py --weights-from {args.out}")


if __name__ == "__main__":
    main()
