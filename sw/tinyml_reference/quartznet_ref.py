# quartznet_ref.py
#
# NumPy int8 golden model for QuartzNet 15x5.  Walks the binary descriptor table
# emitted by quartznet_descriptors.py and executes it with whole-tensor time axes —
# deliberately NOT tiled, so that comparing it against the tile-based C interpreter
# in firmware/quartznet/quartznet_infer.c proves the tiling exact as a side effect.
#
# Run:
#   python sw/tinyml_reference/quartznet_ref.py --out DIR              # full 15x5
#   python sw/tinyml_reference/quartznet_ref.py --out DIR --reduced    # small config
#
# Emits, into DIR:
#   quartznet_desc.bin / .txt   descriptor table + human-readable dump
#   quartznet_weights.bin       int8 weight blob
#   quartznet_qparams.bin       int32 bias / q_mult / rshift blob
#   quartznet_input.bin         int8 mel input, [time, mel]
#   quartznet_golden.bin        per-descriptor output activations + transcript
#   quartznet_golden.txt        summary, per-descriptor stats, transcript
#
# ── Real weights vs. these ────────────────────────────────────────────────────
#
# torch / onnx / NeMo are not installed and NeMo is a multi-GB dependency, so the
# blobs here are REPRODUCIBLY SEEDED RANDOM int8 at the real topology's exact
# shapes.  That is enough to validate the descriptor format, the tiling, and every
# arithmetic path — which is the point.  Real weights slot into the same blobs
# unchanged; see QUARTZNET_MODEL_ACQUISITION.md.
#
# The requantize multipliers are not random: they are CALIBRATED in a forward pass
# (the same thing a real post-training-quantization pass does) so that each layer's
# accumulator distribution actually lands in int8 range.  Without this the test is
# worthless — everything saturates or collapses to the zero point and bit-exactness
# becomes trivial to satisfy.
#
# ── Numerics ──────────────────────────────────────────────────────────────────
#
# requantize() below is bit-identical to firmware/tinyengine_port/tiny_vad_infer.c:44-56,
# including NEGATIVE shift meaning a LEFT shift.  Accumulation is int32 (ACC_W=32);
# the worst case here is 1024 * 255 * 127 = 33,162,240, six bits inside int32.
# Pointwise accumulation goes through float64 matmul, which is exact for these
# magnitudes (every partial sum is well under 2^53) and hits BLAS instead of NumPy's
# very slow integer matmul path.

from __future__ import annotations

import argparse
import pathlib
import struct
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from quartznet_descriptors import (              # noqa: E402
    BUFREC_BYTES, DESC_BYTES, F_FUSE_NEXT, F_RELU, HEADER_BYTES, HEADER_FIELDS,
    QN_DESC_MAGIC, build_table,
)
from quartznet_topology import (                 # noqa: E402
    BLANK_IDX, BUF_LOGITS, BUF_NAMES, BlockSpec, LABELS, N_CLASSES,
    OP_ADD, OP_DW, OP_NAMES, OP_PW, OP_REQUANT,
)

GOLDEN_MAGIC = 0x4F474E51        # 'QNGO'
ADD_LEFT_SHIFT = 20              # TFLite ADD's fixed pre-scale headroom


# ── Q31 multiplier decomposition — same math as export_weights.py:63-85 ───────

def quantize_multiplier(real_multiplier: float) -> tuple[int, int]:
    """Decompose real_multiplier -> (int32 Q31 mantissa, shift).

    real_multiplier ~= q * 2^-31 * 2^-shift
    shift > 0 = right shift, shift < 0 = left shift.  Identical to
    sw/tinyml_reference/export_weights.py:quantize_multiplier().
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


def requantize(x, q_mult, shift):
    """Vectorized twin of tiny_vad_infer.c:44-56. All arithmetic in int64.

    val = (q_mult * x + 2^30) >> 31        (Q31, round-half-up)
    shift > 0 : val = (val + 2^(shift-1)) >> shift
    shift < 0 : val = val << -shift
    """
    x = np.asarray(x, dtype=np.int64)
    q = np.asarray(q_mult, dtype=np.int64)
    s = np.asarray(shift, dtype=np.int64)
    val = q * x + np.int64(1 << 30)
    val >>= np.int64(31)
    sp = np.maximum(s, 0)
    rnd = np.where(sp > 0, np.left_shift(np.int64(1), np.maximum(sp - 1, 0)),
                   np.int64(0))
    right = (val + rnd) >> sp
    left = val << np.maximum(-s, 0)
    return np.where(s > 0, right, np.where(s < 0, left, val))


def clamp_i8(x):
    return np.clip(x, -128, 127).astype(np.int8)


# ── descriptor table parsing (reads the .bin, so the packing itself is tested) ─

def parse_table(blob: bytes) -> tuple[dict, list[dict], list[dict]]:
    hdr_vals = struct.unpack_from("<16I", blob, 0)
    hdr = dict(zip(HEADER_FIELDS, hdr_vals))
    hdr["in_zp"] = struct.unpack_from("<i", blob, 9 * 4)[0]
    assert hdr["magic"] == QN_DESC_MAGIC, "bad descriptor-table magic"
    assert hdr["desc_stride"] == DESC_BYTES

    bufs = []
    for b in range(hdr["n_buffers"]):
        ch, rate = struct.unpack_from("<2I", blob, HEADER_BYTES + b * BUFREC_BYTES)
        bufs.append({"channels": ch, "rate": rate})

    descs = []
    for i in range(hdr["n_desc"]):
        w = struct.unpack_from("<16I", blob, hdr["desc_off"] + i * DESC_BYTES)
        descs.append(unpack_record(w))
    return hdr, bufs, descs


def _s8(v: int) -> int:
    return v - 256 if v >= 128 else v


def unpack_record(w: tuple[int, ...]) -> dict:
    return {
        "op":         w[0] & 0xFF,
        "flags":     (w[0] >> 8) & 0xFF,
        "in_buf":    (w[0] >> 16) & 0xFF,
        "out_buf":   (w[0] >> 24) & 0xFF,
        "c_in":       w[1] & 0xFFFF,
        "c_out":     (w[1] >> 16) & 0xFFFF,
        "k":          w[2] & 0xFFFF,
        "stride":    (w[2] >> 16) & 0xFF,
        "dilation":  (w[2] >> 24) & 0xFF,
        "pad":        w[3] & 0xFFFF,
        "res_buf":   (w[3] >> 16) & 0xFF,
        "in_stride":  w[4] & 0xFFFF,
        "out_stride": (w[4] >> 16) & 0xFFFF,
        "in_zp":     _s8(w[5] & 0xFF),
        "out_zp":    _s8((w[5] >> 8) & 0xFF),
        "res_zp":    _s8((w[5] >> 16) & 0xFF),
        "in_off":     w[6],
        "out_off":    w[7],
        "res_off":    w[8],
        "w_off":      w[9],
        "bias_off":   w[10],
        "qmult_off":  w[11],
        "rshift_off": w[12],
        "acc_off":    w[13],
        "layer_id":   w[14] & 0xFFFF,
        "block_id":  (w[14] >> 16) & 0xFFFF,
    }


def n_qch(d: dict) -> int:
    return 3 if d["op"] == OP_ADD else d["c_out"]


def n_weights(d: dict) -> int:
    if d["op"] == OP_DW:
        return d["c_out"] * d["k"]
    if d["op"] == OP_PW:
        return d["c_out"] * d["c_in"] * d["k"]
    return 0


# ── the interpreter ───────────────────────────────────────────────────────────

class RefRunner:
    """Executes a parsed descriptor table over whole-tensor time axes."""

    def __init__(self, hdr, bufs, descs, weights: np.ndarray, t_out: int):
        self.hdr, self.bufs, self.descs = hdr, bufs, descs
        self.weights = weights
        self.t_out = t_out
        self.bias: list[np.ndarray] = []
        self.qmult: list[np.ndarray] = []
        self.rshift: list[np.ndarray] = []
        self.acc_max = 0
        self.buf = [np.zeros((b["rate"] * t_out, b["channels"]), dtype=np.int8)
                    for b in bufs]

    # -- helpers ----------------------------------------------------------------

    def _w(self, d: dict) -> np.ndarray:
        n = n_weights(d)
        return self.weights[d["w_off"]: d["w_off"] + n].astype(np.int64)

    def _src(self, d: dict, buf_id: int, off: int, stride: int, t: int) -> np.ndarray:
        """Read a [t, stride]-shaped view starting at byte offset `off`."""
        b = self.buf[buf_id]
        assert b.shape[0] >= t, f"buffer {BUF_NAMES[buf_id]} too short: {b.shape[0]} < {t}"
        base_c = off % b.shape[1]
        return b[:t, base_c: base_c + stride]

    def _store(self, d: dict, acc: np.ndarray, i: int) -> np.ndarray:
        """requantize -> +zp -> ReLU -> clamp -> write the output slice."""
        self.acc_max = max(self.acc_max, int(np.abs(acc).max(initial=0)))
        assert np.abs(acc).max(initial=0) < 2**31, "int32 accumulator overflow"
        # ADD packs [q_a, q_b, q_out] (n_qch == 3): the first two pre-scale the
        # operands in _acc_add, so only the third applies here. Every other op
        # carries one multiplier per output channel and broadcasts over time.
        if d["op"] == OP_ADD:
            qm, rs = self.qmult[i][2], self.rshift[i][2]
        else:
            qm, rs = self.qmult[i][None, :], self.rshift[i][None, :]
        r = requantize(acc, qm, rs) + d["out_zp"]
        if d["flags"] & F_RELU:
            r = np.maximum(r, d["out_zp"])
        out = clamp_i8(r)
        dst = self.buf[d["out_buf"]]
        c0 = d["out_off"]
        dst[: self.t_out, c0: c0 + d["c_out"]] = out
        return out

    # -- per-op accumulators ----------------------------------------------------

    def _acc_dw(self, d: dict) -> np.ndarray:
        c, K = d["c_out"], d["k"]
        w = self._w(d).reshape(c, K)
        t_in = self.t_out * d["stride"]
        src = self._src(d, d["in_buf"], d["in_off"], d["in_stride"], t_in)
        x = src[:, :c].astype(np.int64) - d["in_zp"]
        t_idx = np.arange(self.t_out)
        acc = np.zeros((self.t_out, c), dtype=np.int64)
        for k in range(K):
            pos = t_idx * d["stride"] + k * d["dilation"] - d["pad"]
            ok = (pos >= 0) & (pos < t_in)
            xv = np.where(ok[:, None], x[np.clip(pos, 0, t_in - 1)], 0)
            acc += xv * w[:, k][None, :]
        return acc

    def _acc_pw(self, d: dict) -> np.ndarray:
        c_in, c_out = d["c_in"], d["c_out"]
        assert d["k"] == 1, "only K=1 pointwise is emitted by this topology"
        w = self._w(d).reshape(c_out, c_in)
        src = self._src(d, d["in_buf"], d["in_off"], d["in_stride"], self.t_out)
        x = src[:, :c_in].astype(np.float64) - d["in_zp"]
        # float64 matmul is exact here: |partial| <= 1024*255*127 << 2^53.
        return (x @ w.astype(np.float64).T).astype(np.int64)

    def _acc_add(self, d: dict, i: int) -> np.ndarray:
        """TFLite ADD: pre-scale both operands into a shared domain, then sum."""
        c = d["c_out"]
        m = self.buf[d["in_buf"]][: self.t_out, :c].astype(np.int64) - d["in_zp"]
        r = self.buf[d["res_buf"]][: self.t_out, :c].astype(np.int64) - d["res_zp"]
        qm, rs = self.qmult[i], self.rshift[i]
        a = requantize(m << ADD_LEFT_SHIFT, qm[0], rs[0])
        b = requantize(r << ADD_LEFT_SHIFT, qm[1], rs[1])
        return a + b

    def _acc_requant(self, d: dict) -> np.ndarray:
        c = d["c_out"]
        src = self._src(d, d["in_buf"], d["in_off"], d["in_stride"], self.t_out)
        return src[:, :c].astype(np.int64) - d["in_zp"]

    def accumulate(self, i: int) -> np.ndarray:
        d = self.descs[i]
        if d["op"] == OP_DW:
            return self._acc_dw(d)
        if d["op"] == OP_PW:
            return self._acc_pw(d)
        if d["op"] == OP_ADD:
            return self._acc_add(d, i)
        if d["op"] == OP_REQUANT:
            return self._acc_requant(d)
        raise ValueError(f"unknown op {d['op']}")


# ── calibrated blob generation ────────────────────────────────────────────────

def make_blobs(tbl, t_out: int, seed: int = 20260727, target: float = 48.0):
    """Generate seeded random weights, calibrated qparams, and a random mel input.

    One forward pass does both jobs: for each descriptor the accumulator is computed
    first, its 99.5th percentile is measured, and the multiplier is chosen so that
    percentile maps to `target` counts.  Downstream layers then see realistically
    distributed int8, exactly as they would with real weights.
    """
    rng = np.random.default_rng(seed)
    blob = tbl.to_bytes()
    hdr, bufs, descs = parse_table(blob)

    weights = rng.integers(-127, 128, size=tbl.weight_bytes, dtype=np.int64).astype(np.int8)
    run = RefRunner(hdr, bufs, descs, weights, t_out)

    t_in = bufs[0]["rate"] * t_out
    inp = rng.integers(-128, 128, size=(t_in, hdr["in_ch"]), dtype=np.int64).astype(np.int8)
    run.buf[0][:, :] = inp

    qblob = bytearray(tbl.qparam_bytes)
    goldens: list[np.ndarray] = []
    stats: list[dict] = []

    for i, d in enumerate(descs):
        nq = n_qch(d)

        if d["op"] == OP_ADD:
            # Both operands are pre-scaled by 0.5 into a shared domain (TFLite's
            # twice_max normalization with sc_main == sc_res), then the sum is
            # calibrated down to int8.
            q0, s0 = quantize_multiplier(0.5)
            run.qmult.append(np.array([q0, q0, 0], dtype=np.int64))
            run.rshift.append(np.array([s0, s0, 0], dtype=np.int64))
            acc = run.accumulate(i)
            scale = _calib_scale(acc, target)
            q2, s2 = quantize_multiplier(scale)
            run.qmult[i] = np.array([q0, q0, q2], dtype=np.int64)
            run.rshift[i] = np.array([s0, s0, s2], dtype=np.int64)
            bias = np.zeros(nq, dtype=np.int64)
        else:
            run.qmult.append(np.zeros(nq, dtype=np.int64))
            run.rshift.append(np.zeros(nq, dtype=np.int64))
            acc = run.accumulate(i)
            # bias at ~10% of the accumulator spread, then folded in
            spread = max(1.0, float(np.percentile(np.abs(acc), 99.5)))
            bias = np.round(rng.normal(0.0, 0.1 * spread, nq)).astype(np.int64)
            acc = acc + bias[None, :]
            # per-CHANNEL multipliers, like TFLite's per-channel weight quantization
            per_ch = np.percentile(np.abs(acc), 99.5, axis=0)
            qm = np.zeros(nq, dtype=np.int64)
            rs = np.zeros(nq, dtype=np.int64)
            for c in range(nq):
                q, s = quantize_multiplier(target / max(per_ch[c], 1.0))
                qm[c], rs[c] = q, s
            run.qmult[i], run.rshift[i] = qm, rs

        out = run._store(d, acc, i)
        goldens.append(np.ascontiguousarray(out))

        struct.pack_into(f"<{nq}i", qblob, d["bias_off"], *bias.tolist())
        struct.pack_into(f"<{nq}i", qblob, d["qmult_off"], *run.qmult[i].tolist())
        struct.pack_into(f"<{nq}i", qblob, d["rshift_off"], *run.rshift[i].tolist())

        stats.append({
            "acc_absmax": int(np.abs(acc).max(initial=0)),
            "out_min": int(out.min(initial=0)),
            "out_max": int(out.max(initial=0)),
            "out_uniq": int(np.unique(out).size),
        })

    return {
        "table": blob, "hdr": hdr, "bufs": bufs, "descs": descs,
        "weights": weights, "qparams": bytes(qblob), "input": inp,
        "goldens": goldens, "stats": stats, "runner": run,
    }


def _calib_scale(acc: np.ndarray, target: float) -> float:
    return target / max(float(np.percentile(np.abs(acc), 99.5)), 1.0)


def replay(blob: bytes, weights: np.ndarray, qparams: bytes,
           inp: np.ndarray, t_out: int):
    """Re-run from the emitted blobs alone. Must reproduce make_blobs() exactly."""
    hdr, bufs, descs = parse_table(blob)
    run = RefRunner(hdr, bufs, descs, weights, t_out)
    run.buf[0][:, :] = inp
    outs = []
    for i, d in enumerate(descs):
        nq = n_qch(d)
        run.bias.append(np.frombuffer(qparams, np.int32, nq, d["bias_off"]).astype(np.int64))
        run.qmult.append(np.frombuffer(qparams, np.int32, nq, d["qmult_off"]).astype(np.int64))
        run.rshift.append(np.frombuffer(qparams, np.int32, nq, d["rshift_off"]).astype(np.int64))
        acc = run.accumulate(i)
        if d["op"] != OP_ADD:
            acc = acc + run.bias[i][None, :]
        outs.append(np.ascontiguousarray(run._store(d, acc, i)))
    return run, outs


# ── CTC greedy decode ─────────────────────────────────────────────────────────

def ctc_greedy(logits: np.ndarray, blank: int = BLANK_IDX) -> tuple[list[int], str]:
    """argmax per frame, collapse repeats, drop blanks.

    Operating on the int8 logits directly is valid: the decoder output is
    per-tensor quantized, so one positive scale and one zero point are shared by
    all classes and argmax is order-preserving.

    Also used on plain fp32 logits (quartznet_run_fp32.py) -- argmax is
    dtype-agnostic, so no cast is needed for either caller. A `.astype(np.int32)`
    cast used to sit here; it was inert for int8 inputs (already representable)
    but silently truncated fp32 logits to integers before comparison, corrupting
    the argmax (caught via all-empty greedy transcripts on the fp32 path).
    """
    ids = np.argmax(np.asarray(logits), axis=1)
    out: list[int] = []
    prev = -1
    for v in ids:
        v = int(v)
        if v != prev and v != blank:
            out.append(v)
        prev = v
    text = "".join(LABELS[i] if i < len(LABELS) else "?" for i in out)
    return out, text


# ── golden-file emission ──────────────────────────────────────────────────────

def write_golden(path: pathlib.Path, goldens: list[np.ndarray], t_out: int,
                 t_in: int, transcript: str) -> int:
    index = []
    off = 0
    for g in goldens:
        index.append((off, g.size))
        off += g.size
    tr = transcript.encode()
    hdr = struct.pack("<8I", GOLDEN_MAGIC, len(goldens), t_out, t_in,
                      N_CLASSES, len(tr), off, 0)
    body = b"".join(struct.pack("<2I", o, n) for o, n in index)
    data = b"".join(g.tobytes() for g in goldens)
    blob = hdr + body + data + tr
    path.write_bytes(blob)
    return len(blob)


# ── reduced config (fast iteration + exercises OP_REQUANT) ────────────────────

REDUCED_BLOCKS = [
    BlockSpec("C1", 48, 11, 1, False, True, stride=2),
    BlockSpec("B1.0", 48, 13, 3, True, True),
    BlockSpec("B2.0", 96, 15, 2, True, True),
    BlockSpec("C2", 96, 17, 1, False, True, dilation=2),
    BlockSpec("C3", 128, 1, 1, False, False),
    BlockSpec("C4", N_CLASSES, 1, 1, False, False),
]


def reduced_table():
    """Small model with the same op mix, plus a trailing OP_REQUANT.

    OP_REQUANT never appears in the 15x5 table, so without this it would be dead,
    untested code in both the reference and the firmware.  Here it rescales the
    logits in place — a pure per-element op, so in-place is well defined.
    """
    from quartznet_topology import LayerDesc

    tbl = build_table(c_out_tile=64, blocks=REDUCED_BLOCKS)
    last = tbl.layers[-1]
    tbl.layers.append(LayerDesc(
        layer_id=len(tbl.layers), block_id=last.block_id + 1,
        block_name="C4/rescale", op=OP_REQUANT,
        c_in=N_CLASSES, c_out=N_CLASSES, k=1, relu=False,
        in_buf=BUF_LOGITS, out_buf=BUF_LOGITS,
        in_stride=N_CLASSES, out_stride=N_CLASSES))
    # rebuild the derived tables now that a descriptor was appended
    return type(tbl)(tbl.layers, in_zp=tbl.in_zp, t_tile=tbl.t_tile,
                     dw_ch_tile=tbl.dw_ch_tile, c_out_tile=tbl.c_out_tile)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description="QuartzNet int8 NumPy golden model")
    ap.add_argument("--out", default=None, help="output directory")
    ap.add_argument("--reduced", action="store_true",
                    help="small config (fast; also exercises OP_REQUANT)")
    ap.add_argument("--t-out", type=int, default=70,
                    help="output frames (default 70 = 2 full T_TILE=32 tiles + a "
                         "6-frame ragged tail)")
    ap.add_argument("--seed", type=int, default=20260727)
    args = ap.parse_args()

    root = pathlib.Path(__file__).resolve().parent.parent.parent
    sub = "quartznet_reduced" if args.reduced else "quartznet"
    out = pathlib.Path(args.out) if args.out else root / "build" / sub
    out.mkdir(parents=True, exist_ok=True)

    tbl = reduced_table() if args.reduced else build_table()
    print(f"config      : {'REDUCED' if args.reduced else 'QuartzNet 15x5'}")
    print(f"descriptors : {len(tbl.layers)}")
    print(f"T_out       : {args.t_out}  (T_TILE={tbl.t_tile})")

    b = make_blobs(tbl, args.t_out, seed=args.seed)
    hdr, descs = b["hdr"], b["descs"]

    # Self-check: replaying from the emitted blobs must reproduce the calibration
    # pass byte-for-byte, otherwise the qparam blob and the goldens disagree.
    run2, outs2 = replay(b["table"], b["weights"], b["qparams"], b["input"], args.t_out)
    bad = [i for i, (a, c) in enumerate(zip(b["goldens"], outs2)) if not np.array_equal(a, c)]
    if bad:
        raise SystemExit(f"*** replay diverged from calibration on descriptors {bad[:8]} ***")
    print(f"replay      : {len(outs2)}/{len(outs2)} descriptors reproduce from blobs")

    ids, text = ctc_greedy(run2.buf[BUF_LOGITS][: args.t_out, :hdr["n_classes"]],
                           hdr["blank_idx"])

    (out / "quartznet_desc.bin").write_bytes(b["table"])
    (out / "quartznet_desc.txt").write_text(tbl.dump())
    (out / "quartznet_weights.bin").write_bytes(b["weights"].tobytes())
    (out / "quartznet_qparams.bin").write_bytes(b["qparams"])
    (out / "quartznet_input.bin").write_bytes(b["input"].tobytes())
    gsz = write_golden(out / "quartznet_golden.bin", b["goldens"],
                       args.t_out, b["bufs"][0]["rate"] * args.t_out, text)

    # summary
    L = [f"QuartzNet reference run  ({'reduced' if args.reduced else '15x5'})",
         f"  seed        {args.seed}",
         f"  descriptors {len(descs)}",
         f"  T_out       {args.t_out}   T_in {b['bufs'][0]['rate'] * args.t_out}",
         f"  weights     {len(b['weights']):,} B",
         f"  qparams     {len(b['qparams']):,} B",
         f"  golden      {gsz:,} B  ({sum(g.size for g in b['goldens']):,} activation bytes)",
         f"  max |acc|   {max(s['acc_absmax'] for s in b['stats']):,}  "
         f"(int32 limit {2**31 - 1:,})",
         "",
         f"{'idx':>4} {'name':<26} {'op':>7} {'shape':>12} {'|acc|max':>12} "
         f"{'out[min,max]':>16} {'uniq':>5}"]
    for i, (d, s) in enumerate(zip(descs, b["stats"])):
        L.append(f"{i:>4} {tbl.layers[i].block_name:<26} {OP_NAMES[d['op']]:>7} "
                 f"{args.t_out:>5}x{d['c_out']:<6} {s['acc_absmax']:>12,} "
                 f"{'[' + str(s['out_min']) + ',' + str(s['out_max']) + ']':>16} "
                 f"{s['out_uniq']:>5}")
    L += ["", f"CTC greedy: {len(ids)} symbols", f'transcript: "{text}"']
    (out / "quartznet_golden.txt").write_text("\n".join(L) + "\n")

    dead = [i for i, s in enumerate(b["stats"]) if s["out_uniq"] < 4]
    print(f"max |acc|   : {max(s['acc_absmax'] for s in b['stats']):,} "
          f"({100.0 * max(s['acc_absmax'] for s in b['stats']) / (2**31 - 1):.2f}% of int32)")
    print(f"output range: median {int(np.median([s['out_uniq'] for s in b['stats']]))} "
          f"distinct int8 values per layer; {len(dead)} near-degenerate layers")
    print(f'transcript  : "{text}"  ({len(ids)} symbols)')
    print(f"wrote       : {out}")


if __name__ == "__main__":
    main()
