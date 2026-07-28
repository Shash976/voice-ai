# quartznet_topology.py
#
# The QuartzNet 15x5 topology as data, plus the expansion from "15 blocks of R=5
# separable convolutions" down to the flat list of primitive layer descriptors the
# firmware interpreter and the accelerator address generator actually execute.
#
# Run:  python sw/tinyml_reference/quartznet_topology.py
#
# Source of truth: NeMo v1.23 `quartznet_15x5.yaml` (verified against the paper's
# stated 18.9 M parameters).  Do not re-derive the block table from the paper text —
# the yaml is what the released `stt_en_quartznet15x5` checkpoint was built from.
#
# ── What a NeMo "separable" sub-block actually is ─────────────────────────────
#
#   JasperBlock(repeat=R, residual=True, separable=True) unrolls to
#
#       for r in 0..R-1:
#           depthwise  Conv1d(c, c, K, stride, dilation, groups=c)   <- carries stride
#           pointwise  Conv1d(c, c_out, 1)
#           BatchNorm1d(c_out)
#           ReLU                         (SKIPPED on the last repeat)
#       residual path:  Conv1d(c_in, c_out, 1) + BatchNorm1d(c_out)
#       out = ReLU(main + residual)
#
#   There is NO BatchNorm and NO activation between the depthwise and the pointwise —
#   they are two `MaskedConv1d`s back to back inside one `_get_conv_bn_layer` call.
#   In an int8 pipeline the depthwise output still has to be requantized to int8 to
#   feed the pointwise, so it gets its own (q_mult, rshift) pair with relu=0.
#
#   BatchNorm folds into the pointwise weights and the requantize scales, exactly the
#   way train_tiny_vad.py already folds BN for TinyVAD.  It never appears as an op.
#
# ── Tensor layout ─────────────────────────────────────────────────────────────
#
#   [time, channel] everywhere — TFLite NHWC convention, matching tiny_vad_infer.c.
#   NOT PyTorch's [channel, time].  CLAUDE.md flags this as the source of past
#   total-garbage bugs; every offset computation below assumes
#
#       element (t, c)  is at   buf[base + t * stride_channels + c]
#
#   where `stride_channels` is the FULL channel count of the tensor (the row pitch),
#   which is not necessarily the number of channels this descriptor computes — see
#   the c_out / out_stride split in quartznet_descriptors.py.

from __future__ import annotations

import math
from dataclasses import dataclass, field

# ── Op codes (must match QN_OP_* in firmware/quartznet/quartznet_infer.h) ──────

OP_DW      = 0   # depthwise conv, groups == c_in == c_out, K taps, carries stride
OP_PW      = 1   # pointwise conv == matvec per frame, K == 1
OP_ADD     = 2   # residual add of two int8 tensors (TFLite ADD semantics)
OP_REQUANT = 3   # rescale/copy a tensor between quantization domains

OP_NAMES = {OP_DW: "dw", OP_PW: "pw", OP_ADD: "add", OP_REQUANT: "requant"}

# ── Buffer IDs (must match QN_BUF_* in quartznet_infer.h) ─────────────────────
#
# Five logical activation tensors.  A and B ping-pong the main path; R holds the
# residual projection; IN is the mel input (at the pre-C1 frame rate); LOGITS is
# the decoder output.

BUF_IN     = 0
BUF_A      = 1
BUF_B      = 2
BUF_R      = 3
BUF_LOGITS = 4
N_BUFFERS  = 5

BUF_NAMES = {BUF_IN: "IN", BUF_A: "A", BUF_B: "B", BUF_R: "R", BUF_LOGITS: "LOG"}

# Buffers at the *input* frame rate (before C1's stride-2 depthwise).  Everything
# else lives at the output frame rate.
BUF_RATE = {BUF_IN: 2, BUF_A: 1, BUF_B: 1, BUF_R: 1, BUF_LOGITS: 1}

# ── Model constants ───────────────────────────────────────────────────────────

N_MEL     = 64     # preprocessor: 64 mel features
FPS_IN    = 100    # window_stride 0.01 s
FPS_OUT   = 50     # C1 has stride 2

# NeMo `stt_en_quartznet15x5` has 28 *labels* (" ", a..z, "'") and ConvASRDecoder
# emits len(labels)+1 logits — the extra one is the CTC blank.  So the decoder is
# 1024 -> 29, and the blank is the LAST index.
LABELS      = [" "] + [chr(ord("a") + i) for i in range(26)] + ["'"]
N_LABELS    = len(LABELS)          # 28
N_CLASSES   = N_LABELS + 1         # 29  (28 labels + CTC blank)
BLANK_IDX   = N_CLASSES - 1        # 28
ENCODER_CH  = 1024                 # C3 output width

# Reference total, recomputed by param_count() and asserted below.
#   sum over layers of weight elements, biases excluded (NeMo convs are bias=False;
#   BN folds into the requantize scales).  28 classes would give 18,846,016 —
#   the +1024 is the CTC blank row of C4.
EXPECTED_PARAMS = 18_847_040


@dataclass(frozen=True)
class BlockSpec:
    """One entry of the QuartzNet block table."""
    name: str
    c_out: int
    k: int
    repeat: int
    residual: bool
    separable: bool
    stride: int = 1
    dilation: int = 1


# The verified QuartzNet 15x5 block table.  separable=True everywhere except C3/C4,
# which are kernel-1 pointwise convolutions and therefore separable by definition.
BLOCKS: list[BlockSpec] = [
    BlockSpec("C1", 256, 33, 1, False, True, stride=2),
    *[BlockSpec(f"B1.{i}", 256, 33, 5, True, True) for i in range(3)],
    *[BlockSpec(f"B2.{i}", 256, 39, 5, True, True) for i in range(3)],
    *[BlockSpec(f"B3.{i}", 512, 51, 5, True, True) for i in range(3)],
    *[BlockSpec(f"B4.{i}", 512, 63, 5, True, True) for i in range(3)],
    *[BlockSpec(f"B5.{i}", 512, 75, 5, True, True) for i in range(3)],
    BlockSpec("C2", 512, 87, 1, False, True, dilation=2),
    BlockSpec("C3", ENCODER_CH, 1, 1, False, False),
    BlockSpec("C4", N_CLASSES, 1, 1, False, False),
]


def same_padding(kernel: int, stride: int, dilation: int) -> int:
    """NeMo's get_same_padding(). Symmetric — QuartzNet is a non-causal, offline model."""
    if stride > 1 and dilation > 1:
        raise ValueError("stride > 1 with dilation > 1 is not supported by NeMo")
    return (dilation * (kernel - 1)) // 2


def conv_out_len(t_in: int, kernel: int, stride: int, dilation: int, pad: int) -> int:
    return (t_in + 2 * pad - dilation * (kernel - 1) - 1) // stride + 1


@dataclass
class LayerDesc:
    """One primitive op. This is what becomes a 64-byte binary descriptor record."""
    layer_id: int
    block_id: int
    block_name: str
    op: int
    c_in: int
    c_out: int
    k: int = 1
    stride: int = 1
    dilation: int = 1
    pad: int = 0
    relu: bool = False
    fuse_next: bool = False          # depthwise whose consumer is the very next pointwise
    in_buf: int = BUF_A
    out_buf: int = BUF_A
    res_buf: int = BUF_R
    # Channel slicing: c_in/c_out are what THIS descriptor computes; *_stride is the
    # full channel count of the underlying tensor (the row pitch).  c_out_base is the
    # first output channel of the slice.  Set by split_wide_layers().
    in_stride: int = 0
    out_stride: int = 0
    c_out_base: int = 0

    @property
    def name(self) -> str:
        return f"{self.block_name}.{OP_NAMES[self.op]}"

    @property
    def n_weights(self) -> int:
        """Weight elements owned by this descriptor."""
        if self.op == OP_DW:
            return self.c_out * self.k
        if self.op == OP_PW:
            return self.c_out * self.c_in * self.k
        return 0                      # ADD / REQUANT carry no weights

    @property
    def n_qch(self) -> int:
        """Number of (bias, q_mult, rshift) triples.

        Per output channel for the convolutions.  ADD is per-TENSOR and needs exactly
        three multipliers — {main operand, residual operand, output} — matching TFLite's
        ADD kernel, so it gets 3 regardless of channel count.
        """
        return 3 if self.op == OP_ADD else self.c_out


def expand(blocks: list[BlockSpec] = BLOCKS,
           n_mel: int = N_MEL) -> list[LayerDesc]:
    """Expand the block table into the flat primitive-op list.

    Buffer allocation is a strict ping-pong: every op that produces a new main
    activation writes the buffer it did not read.  The residual projection writes R
    and leaves the ping-pong pointer alone.
    """
    layers: list[LayerDesc] = []
    c_in = n_mel
    cur = BUF_IN
    lid = 0

    def other(b: int) -> int:
        return BUF_B if b == BUF_A else BUF_A

    for bid, spec in enumerate(blocks):
        is_last_block = bid == len(blocks) - 1

        # ── residual projection (runs on the BLOCK INPUT, before the main path
        #    overwrites it) ────────────────────────────────────────────────────
        if spec.residual:
            layers.append(LayerDesc(
                layer_id=lid, block_id=bid, block_name=f"{spec.name}/res",
                op=OP_PW, c_in=c_in, c_out=spec.c_out, k=1,
                relu=False, in_buf=cur, out_buf=BUF_R))
            lid += 1

        c = c_in
        for r in range(spec.repeat):
            last_repeat = (r == spec.repeat - 1)
            # ReLU is skipped on the last repeat of a residual block — it moves to
            # after the add.
            relu = not (spec.residual and last_repeat)

            if spec.separable:
                pad = same_padding(spec.k, spec.stride, spec.dilation)
                dst = other(cur)
                layers.append(LayerDesc(
                    layer_id=lid, block_id=bid, block_name=f"{spec.name}.r{r}",
                    op=OP_DW, c_in=c, c_out=c, k=spec.k,
                    stride=spec.stride, dilation=spec.dilation, pad=pad,
                    relu=False, fuse_next=True, in_buf=cur, out_buf=dst))
                lid += 1
                cur = dst

                dst = other(cur)
                out_buf = BUF_LOGITS if (is_last_block and last_repeat) else dst
                layers.append(LayerDesc(
                    layer_id=lid, block_id=bid, block_name=f"{spec.name}.r{r}",
                    op=OP_PW, c_in=c, c_out=spec.c_out, k=1,
                    relu=relu, in_buf=cur, out_buf=out_buf))
                lid += 1
                if out_buf != BUF_LOGITS:
                    cur = out_buf
            else:
                # Non-separable: a single conv.  For QuartzNet this is only ever
                # C3/C4, both K=1, so it is a plain pointwise.
                pad = same_padding(spec.k, spec.stride, spec.dilation)
                dst = other(cur)
                out_buf = BUF_LOGITS if (is_last_block and last_repeat) else dst
                layers.append(LayerDesc(
                    layer_id=lid, block_id=bid, block_name=f"{spec.name}.r{r}",
                    op=OP_PW, c_in=c, c_out=spec.c_out, k=spec.k,
                    stride=spec.stride, dilation=spec.dilation, pad=pad,
                    relu=relu, in_buf=cur, out_buf=out_buf))
                lid += 1
                if out_buf != BUF_LOGITS:
                    cur = out_buf

            c = spec.c_out

        # ── residual add ────────────────────────────────────────────────────────
        if spec.residual:
            dst = other(cur)
            layers.append(LayerDesc(
                layer_id=lid, block_id=bid, block_name=f"{spec.name}/add",
                op=OP_ADD, c_in=spec.c_out, c_out=spec.c_out, k=1,
                relu=True, in_buf=cur, out_buf=dst, res_buf=BUF_R))
            lid += 1
            cur = dst

        c_in = spec.c_out

    # A depthwise is fusable only if the immediately following descriptor is the
    # pointwise that consumes it.  (Always true for QuartzNet, but assert it rather
    # than assume it — the hardware relies on this flag to keep the DW intermediate
    # on chip.)
    for i, ld in enumerate(layers):
        if ld.fuse_next:
            nxt = layers[i + 1] if i + 1 < len(layers) else None
            ld.fuse_next = (nxt is not None and nxt.op == OP_PW
                            and nxt.in_buf == ld.out_buf)

    # Row pitch defaults to the descriptor's own channel counts; split_wide_layers()
    # overrides out_stride where a tensor is computed in channel slices.
    for ld in layers:
        ld.in_stride = ld.c_in
        ld.out_stride = ld.c_out

    return layers


def split_wide_layers(layers: list[LayerDesc], c_out_tile: int) -> list[LayerDesc]:
    """Split any descriptor whose c_out exceeds `c_out_tile` into channel slices.

    This is what bounds the on-chip output tile buffer.  It is expressible purely
    through (c_out, c_out_base, out_stride) — the tensor keeps its full row pitch,
    each slice writes a disjoint band of channels, and each slice takes a contiguous
    run of weight rows (weights are [out_ch, in_ch] row-major, so a channel slice is
    a contiguous byte range — no repacking).

    For QuartzNet 15x5 this fires exactly once: C3's 512 -> 1024 pointwise.
    """
    out: list[LayerDesc] = []
    lid = 0
    for ld in layers:
        if ld.op in (OP_DW,) or ld.c_out <= c_out_tile:
            # Depthwise is tiled over channels at run time by the interpreter (it is
            # channel-independent, so no descriptor split is needed).
            new = LayerDesc(**{**ld.__dict__})
            new.layer_id = lid
            out.append(new)
            lid += 1
            continue
        n_slices = (ld.c_out + c_out_tile - 1) // c_out_tile
        for s in range(n_slices):
            base = s * c_out_tile
            width = min(c_out_tile, ld.c_out - base)
            new = LayerDesc(**{**ld.__dict__})
            new.layer_id = lid
            new.c_out = width
            new.c_out_base = base
            new.out_stride = ld.c_out
            new.block_name = f"{ld.block_name}[{base}:{base + width}]"
            out.append(new)
            lid += 1
    return out


# ── Budgets ───────────────────────────────────────────────────────────────────

def param_count(layers: list[LayerDesc]) -> int:
    return sum(ld.n_weights for ld in layers)


def qparam_count(layers: list[LayerDesc]) -> int:
    """int32 entries in the requantize-parameter blob: 3 (bias, q_mult, rshift) per channel."""
    return sum(3 * ld.n_qch for ld in layers)


def buffer_channels(layers: list[LayerDesc]) -> dict[int, int]:
    """Max channel count each activation buffer must hold."""
    ch = {b: 0 for b in range(N_BUFFERS)}
    ch[BUF_IN] = N_MEL
    for ld in layers:
        ch[ld.in_buf] = max(ch[ld.in_buf], ld.in_stride)
        ch[ld.out_buf] = max(ch[ld.out_buf], ld.out_stride)
        if ld.op == OP_ADD:
            ch[ld.res_buf] = max(ch[ld.res_buf], ld.out_stride)
    return ch


def dw_input_span(layers: list[LayerDesc], t_tile: int) -> int:
    """Widest depthwise input window a single output tile needs.

    A tile of t_tile output frames reads
        (t_tile - 1) * stride + (K - 1) * dilation + 1
    input frames.  This is the halo, and it is the quantity that makes cross-layer
    tiling impossible and within-layer tiling cheap.
    """
    return max((t_tile - 1) * ld.stride + (ld.k - 1) * ld.dilation + 1
               for ld in layers if ld.op == OP_DW)


def receptive_field(layers: list[LayerDesc]) -> int:
    """Total receptive field in OUTPUT frames (50 fps), counting only the post-C1 stack.

    Every depthwise widens the receptive field by (K-1)*dilation.  The sum is what
    makes QuartzNet 15x5 an offline model: it cannot be streamed at bounded lag, and
    a cross-layer time tile would have to carry this many frames of halo.
    """
    total = 0
    for ld in layers:
        if ld.op == OP_DW and ld.stride == 1:
            total += (ld.k - 1) * ld.dilation
    return total


def activation_traffic(layers: list[LayerDesc], t_out: int, t_tile: int,
                       fuse_dw_pw: bool = True, retain_halo: bool = True) -> int:
    """Bytes of activation read+written to external memory for one utterance.

    Models the adopted 3-level dataflow: full-length activation tensors live
    off-chip, each descriptor streams through them in t_tile-frame tiles, and a
    fused depthwise's output never leaves the chip (so neither its write nor the
    pointwise's read of it is counted).

    Two knobs, because both cost real SRAM and both matter a lot:

    `fuse_dw_pw`   keep the depthwise intermediate tile on chip and feed the
                   pointwise directly.
    `retain_halo`  keep the tail of the previous tile's depthwise input window on
                   chip instead of re-reading the overlap.  Within one layer the
                   tiles are visited in time order, so the overlap is always
                   exactly the previous (span - t_tile) frames.  Without this the
                   depthwise reads are inflated by span/t_tile, which at t_tile=32
                   is 3-6x for the wide-kernel blocks.
    """
    total = 0
    for i, ld in enumerate(layers):
        t_out_l = t_out
        t_in_l = t_out * ld.stride
        n_tiles = (t_out_l + t_tile - 1) // t_tile
        if ld.op == OP_DW:
            if retain_halo:
                total += t_in_l * ld.in_stride
            else:
                span = (t_tile - 1) * ld.stride + (ld.k - 1) * ld.dilation + 1
                total += n_tiles * span * ld.c_in        # halo re-read every tile
            if not (fuse_dw_pw and ld.fuse_next):
                total += t_out_l * ld.out_stride         # DW intermediate write
        elif ld.op == OP_PW:
            prev = layers[i - 1] if i > 0 else None
            fused_in = (fuse_dw_pw and prev is not None
                        and prev.op == OP_DW and prev.fuse_next
                        and prev.out_buf == ld.in_buf)
            if not fused_in:
                total += t_in_l * ld.in_stride
            total += t_out_l * ld.c_out
        elif ld.op == OP_ADD:
            total += 2 * t_out_l * ld.c_out              # main + residual operands
            total += t_out_l * ld.c_out                  # result
        elif ld.op == OP_REQUANT:
            total += 2 * t_out_l * ld.c_out
    return total


# ── CLI report ────────────────────────────────────────────────────────────────

def _report() -> None:
    from quartznet_descriptors import C_OUT_TILE, T_TILE  # local import, avoids a cycle

    layers = expand()
    total = param_count(layers)

    print(f"{'#':>4} {'layer':<22} {'op':>7} {'C_in':>5} {'C_out':>6} {'K':>3} "
          f"{'s':>2} {'d':>2} {'pad':>4} {'relu':>4} {'in':>3}->{'out':<4} {'weights':>12}")
    print("-" * 108)
    for ld in layers:
        print(f"{ld.layer_id:>4} {ld.block_name:<22} {OP_NAMES[ld.op]:>7} "
              f"{ld.c_in:>5} {ld.c_out:>6} {ld.k:>3} {ld.stride:>2} {ld.dilation:>2} "
              f"{ld.pad:>4} {int(ld.relu):>4} "
              f"{BUF_NAMES[ld.in_buf]:>3}->{BUF_NAMES[ld.out_buf]:<4} {ld.n_weights:>12,}")

    n_by_op = {op: sum(1 for ld in layers if ld.op == op) for op in OP_NAMES}
    print("-" * 108)
    print(f"descriptors: {len(layers)}  ("
          + ", ".join(f"{OP_NAMES[o]}={n}" for o, n in n_by_op.items() if n) + ")")
    print(f"TOTAL weight elements: {total:,}   (expected {EXPECTED_PARAMS:,})")
    if total != EXPECTED_PARAMS:
        raise SystemExit(f"*** PARAMETER COUNT MISMATCH: {total:,} != {EXPECTED_PARAMS:,} ***")
    print("parameter count MATCHES the verified NeMo quartznet_15x5 budget.")

    qn = qparam_count(layers)
    print(f"\nrequantize-parameter blob: {qn:,} int32 = {qn * 4:,} B "
          f"({100.0 * qn * 4 / total:.1f}% on top of the weights)")
    print(f"  output channels needing a (bias, q_mult, rshift) triple: {qn // 3:,}")

    rf = receptive_field(layers)
    print(f"\nreceptive field (post-C1 depthwise stack): {rf:,} output frames "
          f"= {rf / FPS_OUT:.1f} s of audio")
    print("  -> QuartzNet 15x5 is an OFFLINE model. Symmetric padding means every")
    print("     depthwise also needs right-context, so it cannot be streamed at")
    print("     bounded lag, and a CROSS-LAYER time tile would need this much halo.")

    split = split_wide_layers(layers, C_OUT_TILE)
    print(f"\nafter channel-splitting at C_OUT_TILE={C_OUT_TILE}: "
          f"{len(split)} descriptors (was {len(layers)})")
    assert param_count(split) == total, "channel split changed the parameter count"

    ch = buffer_channels(split)
    print("\nactivation buffer widths (channels):")
    for b in range(N_BUFFERS):
        print(f"  {BUF_NAMES[b]:<4} {ch[b]:>5} ch  (rate x{BUF_RATE[b]})")
    per_frame = sum(ch[b] * BUF_RATE[b] for b in range(N_BUFFERS))
    print(f"  arena = {per_frame:,} B per output frame")

    span = dw_input_span(split, T_TILE)
    print(f"\ndepthwise input span at T_TILE={T_TILE}: {span} frames "
          f"(max halo {span - T_TILE})")

    print("\n=== external-memory traffic, adopted 3-level dataflow ===")
    wt_bytes = total + qn * 4
    print(f"  weights + qparams, read once per utterance (QSPI, read-only): "
          f"{wt_bytes / 1e6:.2f} MB")
    print(f"  {'utt':>4} {'T_out':>6} | {'act: naive':>11} {'+fuse':>11} "
          f"{'+fuse+halo':>11} | {'total':>9} {'MB/s':>7}")
    for secs in (5, 10, 20):
        t_out = secs * FPS_OUT
        naive = activation_traffic(split, t_out, T_TILE, False, False)
        fused = activation_traffic(split, t_out, T_TILE, True, False)
        best = activation_traffic(split, t_out, T_TILE, True, True)
        tot = wt_bytes + best
        print(f"  {secs:>3}s {t_out:>6} | {naive / 1e6:>10.2f}M {fused / 1e6:>10.2f}M "
              f"{best / 1e6:>10.2f}M | {tot / 1e6:>8.2f}M {tot / 1e6 / secs:>6.2f}")
    print("  (naive = neither optimization; the halo re-read alone costs more than")
    print("   DW->PW fusion saves, and it is the cheaper of the two to implement)")

    print(f"\n  T_TILE sensitivity of the depthwise halo re-read (10 s utterance,")
    print(f"  DW->PW fused, halo NOT retained):")
    t_out = 10 * FPS_OUT
    base = activation_traffic(split, t_out, T_TILE, True, True)
    for tt in (16, 32, 64, 128, 256):
        a = activation_traffic(split, t_out, tt, True, False)
        span = dw_input_span(split, tt)
        print(f"    T_TILE={tt:>4}  act {a / 1e6:7.2f} MB  ({a / base:.2f}x the "
              f"halo-retained floor)  max DW span {span} frames")

    print("\n=== accumulator width ===")
    c_in_max = max(ld.c_in for ld in split)
    worst = c_in_max * 255 * 127
    print(f"  max c_in = {c_in_max};  worst |acc| = {c_in_max} * 255 * 127 = {worst:,}")
    print(f"  int32 limit 2^31-1 = {2**31 - 1:,}  -> headroom {math.log2((2**31 - 1) / worst):.1f} bits  OK")
    print(f"  ACC_W=24 limit 2^23-1 = {2**23 - 1:,}  -> saturates for c_in >= "
          f"{(2**23 - 1) // (255 * 127) + 1}, i.e. EVERY layer from B1 onward")


if __name__ == "__main__":
    _report()
