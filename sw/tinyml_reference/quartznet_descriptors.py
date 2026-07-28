# quartznet_descriptors.py
#
# The binary layer-descriptor format that firmware/quartznet/quartznet_infer.c and
# (later) the accelerator's descriptor engine both consume, plus the emitter that
# turns quartznet_topology.expand() into a .bin and a human-readable dump.
#
# Run:  python sw/tinyml_reference/quartznet_descriptors.py [--out DIR]
#
# ── Why this format ───────────────────────────────────────────────────────────
#
# 1. 64-byte records, i.e. a POWER-OF-TWO stride.  A hardware descriptor fetch is
#    `addr = table_base + (idx << 6)` — a shift, never a multiplier.
# 2. Sixteen little-endian uint32 words per record, and NO FIELD STRADDLES A WORD.
#    A Verilog address generator therefore only ever does static bit-slicing of a
#    word it has already fetched (`w[15:0]`, `w[23:16]`, ...) — no barrel shifter,
#    no byte-enable logic, no cross-word assembly.
# 3. Every *address* gets a full 32-bit word to itself, because addresses are what
#    the DMA consumes and 18.85 MB of weights does not fit in 16 bits.  Small
#    enums and dimensions are packed into byte/halfword lanes, which costs the
#    hardware nothing.
# 4. `c_in`/`c_out` are the SLICE this descriptor computes; `in_stride`/`out_stride`
#    are the tensor's full channel count (its row pitch).  Splitting these apart is
#    what makes channel tiling expressible without repacking weights, and it is
#    exactly the (base, count, pitch) triple an address generator wants.
# 5. `acc_off` and the ACC_FIRST/ACC_LAST flags are defined but unused in v1.  They
#    are how the accelerator will split a layer along C_in when the input tile does
#    not fit in SRAM.  Reserving them now avoids a format churn later.
#
# The whole 15x5 table is 187 records = 11,968 B — small enough to sit in on-chip
# ROM next to the descriptor engine.
#
# ── Record layout (16 x uint32, little-endian) ────────────────────────────────
#
#  word  bits      field         notes
#   0    [7:0]     op            QN_OP_DW / PW / ADD / REQUANT
#        [15:8]    flags         bit0 RELU, bit1 ACC_FIRST, bit2 ACC_LAST, bit3 FUSE_NEXT
#        [23:16]   in_buf        activation buffer id
#        [31:24]   out_buf
#   1    [15:0]    c_in          input channels consumed by this descriptor
#        [31:16]   c_out         output channels produced by this descriptor
#   2    [15:0]    k             kernel taps
#        [23:16]   stride
#        [31:24]   dilation
#   3    [15:0]    pad           symmetric (NeMo get_same_padding)
#        [23:16]   res_buf       residual operand buffer id (ADD only)
#        [31:24]   --
#   4    [15:0]    in_stride     channels per input frame  (row pitch)
#        [31:16]   out_stride    channels per output frame (row pitch)
#   5    [7:0]     in_zp         int8, sign-extend
#        [15:8]    out_zp        int8
#        [23:16]   res_zp        int8
#        [31:24]   --
#   6    [31:0]    in_off        byte offset of (frame 0, channel 0) within in_buf
#   7    [31:0]    out_off       byte offset of (frame 0, first slice channel) in out_buf
#   8    [31:0]    res_off
#   9    [31:0]    w_off         byte offset into the weight blob
#  10    [31:0]    bias_off      byte offset into the qparam blob -> int32[n_qch]
#  11    [31:0]    qmult_off     int32[n_qch]   (ADD: n_qch == 3, per-tensor)
#  12    [31:0]    rshift_off    int32[n_qch]
#  13    [31:0]    acc_off       reserved (C_in-split partial sums); 0 in v1
#  14    [15:0]    layer_id
#        [31:16]   block_id
#  15    [31:0]    reserved
#
# ── Table file layout ─────────────────────────────────────────────────────────
#
#   [0x00]  64-byte header (16 x uint32, see HEADER_FIELDS)
#   [0x40]  n_buffers x 8-byte buffer records: (channels: u32, rate: u32)
#   [desc_off]  n_desc x 64-byte descriptor records
#
# ── Blob layouts referenced by the offsets ────────────────────────────────────
#
#   weights.bin  int8, per descriptor:
#       DW  w[c * K + k]                     length c_out * K   (c_out == c_in)
#       PW  w[oc * c_in + ic]                length c_out * c_in
#     The PW layout is the SAME [out, in, k] order as tiny_vad_infer.c with k == 1,
#     so the existing MATVEC / im2col reasoning carries over unchanged.  A channel
#     slice of a PW is a contiguous run of rows, hence a contiguous byte range.
#
#   qparams.bin  int32, per descriptor, three contiguous arrays of n_qch entries:
#       bias[n_qch], q_mult[n_qch], rshift[n_qch]
#     rshift is kept int32 rather than packed to int8 (which would save 209 KB of
#     the 837 KB blob, ~1% of total traffic).  Packing would force the address
#     generator to handle two element widths in one blob; hardware simplicity wins.

from __future__ import annotations

import argparse
import pathlib
import struct

from quartznet_topology import (
    BLOCKS, BUF_NAMES, BUF_RATE, N_BUFFERS, N_CLASSES, BLANK_IDX, N_MEL,
    OP_ADD, OP_DW, OP_NAMES, OP_PW, OP_REQUANT,
    LayerDesc, buffer_channels, expand, param_count, qparam_count,
    split_wide_layers,
)

# ── Format constants (must match QN_* in firmware/quartznet/quartznet_infer.h) ─

QN_DESC_MAGIC   = 0x54444E51        # 'QNDT' little-endian
QN_DESC_VERSION = 1
DESC_WORDS      = 16
DESC_BYTES      = DESC_WORDS * 4    # 64 — power of two on purpose
HEADER_BYTES    = 64
BUFREC_BYTES    = 8

# Flag bits
F_RELU      = 1 << 0
F_ACC_FIRST = 1 << 1                # reserved: first chunk of a C_in-split layer
F_ACC_LAST  = 1 << 2                # reserved: last chunk, requantize and store
F_FUSE_NEXT = 1 << 3                # DW whose output is consumed by the next PW;
                                    # the hardware keeps that tile on chip

# Tiling parameters baked into the emitted table.
T_TILE      = 32      # output frames per compute tile
C_OUT_TILE  = 512     # max output channels per descriptor -> bounds the output tile
DW_CH_TILE  = 64      # depthwise channels per tile (depthwise is channel-independent)

HEADER_FIELDS = [
    "magic", "version", "n_desc", "desc_stride",
    "weight_bytes", "qparam_bytes", "n_classes", "blank_idx",
    "in_ch", "in_zp", "t_tile", "c_out_tile",
    "dw_ch_tile", "dw_span_max", "n_buffers", "desc_off",
]


def _u8(x: int) -> int:
    """Two's-complement byte lane (accepts int8 or uint8 values)."""
    return x & 0xFF


def _u16(x: int) -> int:
    assert 0 <= x <= 0xFFFF, f"field {x} does not fit in 16 bits"
    return x & 0xFFFF


class DescriptorTable:
    """A packed descriptor table plus the blob offsets it hands out."""

    def __init__(self, layers: list[LayerDesc], in_zp: int = -20,
                 t_tile: int = T_TILE, dw_ch_tile: int = DW_CH_TILE,
                 c_out_tile: int = C_OUT_TILE):
        self.layers = layers
        self.in_zp = in_zp
        self.t_tile = t_tile
        self.dw_ch_tile = dw_ch_tile
        self.c_out_tile = c_out_tile
        self.buf_ch = buffer_channels(layers)

        # Assign blob offsets in descriptor order.  Weight rows for a channel slice
        # are contiguous, so a split layer just consumes its slice and moves on.
        self.w_off: list[int] = []
        self.q_off: list[int] = []
        w = 0
        q = 0
        for ld in layers:
            self.w_off.append(w)
            w += ld.n_weights                      # int8
            self.q_off.append(q)
            q += 3 * ld.n_qch * 4                  # int32 x 3 arrays
        self.weight_bytes = w
        self.qparam_bytes = q

        # Zero points.  ReLU outputs use zp = -128 so the activation occupies the
        # full [0, 255] unsigned range after (x - zp); everything else is symmetric.
        self.zp_out = [(-128 if ld.relu else 0) for ld in layers]
        self._resolve_zps()

    def _resolve_zps(self) -> None:
        """Propagate each buffer's current zero point to whoever reads it next."""
        cur_zp = {b: 0 for b in range(N_BUFFERS)}
        cur_zp[0] = self.in_zp                      # BUF_IN carries the mel zp
        self.zp_in: list[int] = []
        self.zp_res: list[int] = []
        for i, ld in enumerate(self.layers):
            self.zp_in.append(cur_zp[ld.in_buf])
            self.zp_res.append(cur_zp[ld.res_buf] if ld.op == OP_ADD else 0)
            cur_zp[ld.out_buf] = self.zp_out[i]

    # ── packing ────────────────────────────────────────────────────────────────

    def pack_record(self, i: int) -> bytes:
        ld = self.layers[i]
        flags = 0
        if ld.relu:
            flags |= F_RELU
        if ld.fuse_next:
            flags |= F_FUSE_NEXT

        w = [0] * DESC_WORDS
        w[0] = (_u8(ld.op) | (_u8(flags) << 8)
                | (_u8(ld.in_buf) << 16) | (_u8(ld.out_buf) << 24))
        w[1] = _u16(ld.c_in) | (_u16(ld.c_out) << 16)
        w[2] = _u16(ld.k) | (_u8(ld.stride) << 16) | (_u8(ld.dilation) << 24)
        w[3] = _u16(ld.pad) | (_u8(ld.res_buf) << 16)
        w[4] = _u16(ld.in_stride) | (_u16(ld.out_stride) << 16)
        w[5] = (_u8(self.zp_in[i]) | (_u8(self.zp_out[i]) << 8)
                | (_u8(self.zp_res[i]) << 16))
        w[6] = 0                                    # in_off: full-width tensors start at 0
        w[7] = ld.c_out_base                        # channel slice -> byte offset of ch 0
        w[8] = 0
        w[9] = self.w_off[i]
        w[10] = self.q_off[i]
        w[11] = self.q_off[i] + 4 * ld.n_qch
        w[12] = self.q_off[i] + 8 * ld.n_qch
        w[13] = 0                                   # acc_off, reserved
        w[14] = _u16(ld.layer_id) | (_u16(ld.block_id) << 16)
        w[15] = 0
        return struct.pack("<16I", *w)

    def dw_span_max(self) -> int:
        return max(((self.t_tile - 1) * ld.stride + (ld.k - 1) * ld.dilation + 1)
                   for ld in self.layers if ld.op == OP_DW)

    def header(self) -> bytes:
        desc_off = HEADER_BYTES + N_BUFFERS * BUFREC_BYTES
        vals = [
            QN_DESC_MAGIC, QN_DESC_VERSION, len(self.layers), DESC_BYTES,
            self.weight_bytes, self.qparam_bytes, N_CLASSES, BLANK_IDX,
            N_MEL, self.in_zp & 0xFFFFFFFF, self.t_tile, self.c_out_tile,
            self.dw_ch_tile, self.dw_span_max(), N_BUFFERS, desc_off,
        ]
        assert len(vals) == len(HEADER_FIELDS)
        return struct.pack("<16I", *vals)

    def to_bytes(self) -> bytes:
        out = bytearray(self.header())
        for b in range(N_BUFFERS):
            out += struct.pack("<2I", self.buf_ch[b], BUF_RATE[b])
        for i in range(len(self.layers)):
            out += self.pack_record(i)
        return bytes(out)

    # ── activation-arena geometry (mirrors qn_arena_layout() in the firmware) ───

    def arena_layout(self, t_out: int) -> tuple[list[int], int]:
        """Byte offset of each activation buffer and the total arena size."""
        offs = []
        cur = 0
        for b in range(N_BUFFERS):
            offs.append(cur)
            cur += self.buf_ch[b] * BUF_RATE[b] * t_out
        return offs, cur

    # ── human-readable dump ────────────────────────────────────────────────────

    def dump(self) -> str:
        L: list[str] = []
        A = L.append
        A("QuartzNet 15x5 layer-descriptor table")
        A("=" * 118)
        hdr = struct.unpack("<16I", self.header())
        for name, v in zip(HEADER_FIELDS, hdr):
            if name == "magic":
                A(f"  {name:<14} 0x{v:08X}  ('{struct.pack('<I', v).decode()}')")
            elif name == "in_zp":
                A(f"  {name:<14} {struct.unpack('<i', struct.pack('<I', v))[0]}")
            else:
                A(f"  {name:<14} {v}")
        A("")
        A("  activation buffers:")
        for b in range(N_BUFFERS):
            A(f"    [{b}] {BUF_NAMES[b]:<4} {self.buf_ch[b]:>5} ch  rate x{BUF_RATE[b]}")
        A("")
        A(f"{'idx':>4} {'name':<24} {'op':>7} {'C_in':>5} {'C_out':>6} "
          f"{'K':>3} {'s':>2} {'d':>2} {'pad':>4} {'flags':<14} "
          f"{'in':>3}>{'out':<4} {'inS':>5} {'outS':>5} {'zpi':>4} {'zpo':>4} "
          f"{'w_off':>10} {'w_len':>10} {'q_off':>9} {'nq':>5}")
        A("-" * 118)
        for i, ld in enumerate(self.layers):
            fl = []
            if ld.relu:
                fl.append("RELU")
            if ld.fuse_next:
                fl.append("FUSE")
            A(f"{i:>4} {ld.block_name:<24} {OP_NAMES[ld.op]:>7} "
              f"{ld.c_in:>5} {ld.c_out:>6} {ld.k:>3} {ld.stride:>2} {ld.dilation:>2} "
              f"{ld.pad:>4} {'|'.join(fl) or '-':<14} "
              f"{BUF_NAMES[ld.in_buf]:>3}>{BUF_NAMES[ld.out_buf]:<4} "
              f"{ld.in_stride:>5} {ld.out_stride:>5} "
              f"{self.zp_in[i]:>4} {self.zp_out[i]:>4} "
              f"{self.w_off[i]:>10,} {ld.n_weights:>10,} {self.q_off[i]:>9,} "
              f"{ld.n_qch:>5}")
        A("-" * 118)
        A(f"descriptors     {len(self.layers)}")
        A(f"table bytes     {len(self.to_bytes()):,}")
        A(f"weight blob     {self.weight_bytes:,} B")
        A(f"qparam blob     {self.qparam_bytes:,} B")
        A(f"DW input span   {self.dw_span_max()} frames at T_TILE={self.t_tile}")
        return "\n".join(L) + "\n"


def build_table(t_tile: int = T_TILE, c_out_tile: int = C_OUT_TILE,
                dw_ch_tile: int = DW_CH_TILE, in_zp: int = -20,
                blocks=None) -> DescriptorTable:
    layers = split_wide_layers(expand(blocks or BLOCKS), c_out_tile)
    return DescriptorTable(layers, in_zp=in_zp, t_tile=t_tile,
                           dw_ch_tile=dw_ch_tile, c_out_tile=c_out_tile)


def main() -> None:
    ap = argparse.ArgumentParser(description="Emit the QuartzNet descriptor table")
    ap.add_argument("--out", default=None,
                    help="output directory (default: <repo>/build/quartznet)")
    args = ap.parse_args()

    root = pathlib.Path(__file__).resolve().parent.parent.parent
    out = pathlib.Path(args.out) if args.out else root / "build" / "quartznet"
    out.mkdir(parents=True, exist_ok=True)

    tbl = build_table()
    blob = tbl.to_bytes()
    (out / "quartznet_desc.bin").write_bytes(blob)
    (out / "quartznet_desc.txt").write_text(tbl.dump())

    total = param_count(tbl.layers)
    print(f"descriptors : {len(tbl.layers)}")
    print(f"table       : {len(blob):,} B  -> {out / 'quartznet_desc.bin'}")
    print(f"dump        : {out / 'quartznet_desc.txt'}")
    print(f"weight blob : {tbl.weight_bytes:,} B")
    print(f"qparam blob : {tbl.qparam_bytes:,} B "
          f"({100.0 * tbl.qparam_bytes / tbl.weight_bytes:.1f}% of weights)")
    assert tbl.weight_bytes == total == qparam_ok(tbl), "blob sizing inconsistent"
    print(f"param count : {total:,}")


def qparam_ok(tbl: DescriptorTable) -> int:
    assert tbl.qparam_bytes == qparam_count(tbl.layers) * 4
    return param_count(tbl.layers)


if __name__ == "__main__":
    main()
