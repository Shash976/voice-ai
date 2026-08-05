`timescale 1 ns / 1 ps
/* addr_gen.v — compute-tile walker and address generator for one QN_DESC
 * descriptor.
 *
 * This is the piece `tinymac_accel.v` does not have: that core exposes raw FSM
 * counters (o_m / o_k_base) and lets the environment turn them into addresses.
 * QuartzNet has four op shapes, per-layer tiling and halo retention, so the
 * index->address mapping becomes real logic and gets its own module.
 *
 * ── What it must reproduce ──────────────────────────────────────────────────
 *
 * firmware/quartznet/quartznet_infer.c::walk_desc() is the authority.  Its tile
 * grid is:
 *
 *     for (t0 = 0; t0 < t_out; t0 += t_tile)          // OUTER
 *         for (c0 = 0; c0 < c_out; c0 += c_tile)      // INNER
 *
 * with c_tile = dw_ch_tile for OP_DW (depthwise is channel-independent, so it
 * tiles over channels too) and c_tile = c_out for every other op (c_out is
 * already bounded by c_out_tile at table-emission time).
 *
 * Halo retention (t_in0/t_in1 vs t_new0/t_new1) mirrors the same function and
 * the retain_halo traffic model in docs/07_quartznet_pivot.md.  It is pure
 * OBSERVABILITY here — the datapath re-reads whatever it needs — but it is
 * exported so the testbench can check the walk against the C interpreter, and
 * so a later increment can drive a real activation cache from it.
 *
 * ── Deliberate difference from the C, and why it is safe ────────────────────
 *
 * WITHIN a tile the C runs `for t { for c { ... } }`.  This module runs
 * `for c { for t { ... } }`.  Every output element is an independent reduction,
 * so element order is not observable in the result — and c-outer lets the
 * sequencer fetch each channel's (bias, qmult, rshift) triple ONCE per tile
 * instead of once per (t, c), which is the dominant qparam traffic term.  The
 * TILE grid itself, which is what tiling correctness actually depends on, is
 * identical to the C.
 *
 * ── Address arithmetic ──────────────────────────────────────────────────────
 *
 * Activation element (t, c) of a buffer lives at
 *     base + t * pitch + cbase + c
 * where `pitch` is the BUFFER's allocated channel count (not the descriptor's
 * in_stride/out_stride) — see quartznet_infer.h.  `cbase` is the descriptor's
 * in_off/out_off channel offset; the caller stages it already reduced modulo
 * the pitch, exactly as the C does with `in_off % pitch`, so no divider is
 * needed here.
 *
 * Reduction operands per chunk of LANES lanes:
 *
 *   OP_DW  lane l covers tap k = r_base + l, at input frame
 *              pos = t*stride + k*dilation - pad
 *          so consecutive lanes step `dilation * pitch` bytes.
 *          Lane enabled iff k < K AND 0 <= pos < t_in   (the C's zero-padding
 *          `continue`, which drops the whole product rather than substituting
 *          the zero point).
 *          weights: w_off + c*K + r_base, contiguous.
 *
 *   OP_PW  lane l covers input channel ic = r_base + l of frame t, so lanes are
 *          contiguous (stride 1).  Lane enabled iff ic < c_in.
 *          weights: w_off + oc*c_in + r_base, contiguous.
 *
 *   OP_REQUANT is a 1-tap reduction whose weight is a synthetic 1 (the caller
 *   supplies it; there is no weight blob for this op).  OP_ADD does not reduce
 *   at all and takes the elementwise operand addresses instead.
 *
 * Addresses are unsigned 32-bit and wrap on purpose: for OP_DW the frame index
 * of lane 0 may be negative (left padding), which makes the gather base address
 * meaningless — but every lane whose true address is in range still computes
 * correctly under two's-complement wraparound, and the out-of-range lanes are
 * masked off and never issued to memory.
 *
 * Yosys note: everything crossing an arithmetic operator is kept consistently
 * unsigned or consistently signed (no $signed() on unsigned whole wires, no
 * signed `integer` parameters inside unsigned expressions, no mixed-sign
 * ternaries) — the frontend assert described in CLAUDE.md.
 */

`default_nettype none

module addr_gen #(
    parameter integer LANES = 32
) (
    input  wire        clk,
    input  wire        rst_n,

    /* ── Control ────────────────────────────────────────────────────────── */
    input  wire        start,       /* 1-cycle pulse: begin this descriptor */
    input  wire        chunk_done,  /* 1-cycle pulse: current chunk consumed */
    input  wire        elem_done,   /* 1-cycle pulse: current element stored */

    /* ── Descriptor configuration (stable while busy) ───────────────────── */
    input  wire [1:0]  cfg_op,          /* 0 DW, 1 PW, 2 ADD, 3 REQUANT */
    input  wire [15:0] cfg_c_in,
    input  wire [15:0] cfg_c_out,
    input  wire [15:0] cfg_k,
    input  wire [7:0]  cfg_stride,
    input  wire [7:0]  cfg_dilation,
    input  wire [15:0] cfg_pad,
    input  wire [15:0] cfg_t_out,
    input  wire [15:0] cfg_t_tile,
    input  wire [15:0] cfg_dw_ch_tile,

    input  wire [31:0] cfg_in_base,     /* byte address of in_buf  */
    input  wire [15:0] cfg_in_pitch,    /* buffer channel count    */
    input  wire [15:0] cfg_in_cbase,    /* in_off % in_pitch       */
    input  wire [31:0] cfg_out_base,
    input  wire [15:0] cfg_out_pitch,
    input  wire [15:0] cfg_out_cbase,   /* out_off                 */
    input  wire [31:0] cfg_res_base,
    input  wire [15:0] cfg_res_pitch,
    input  wire [31:0] cfg_w_off,       /* byte offset of weights  */

    /* ── Tile-level observability (matches quartznet_infer.c's walk) ────── */
    output wire [15:0] o_t0,
    output wire [15:0] o_t1,
    output wire [15:0] o_c0,
    output wire [15:0] o_c1,
    output wire [15:0] o_in0,       /* input frame span read by this tile */
    output wire [15:0] o_in1,
    output wire [15:0] o_new0,      /* sub-span not retained as halo      */
    output wire [15:0] o_new1,
    output wire        o_first_tile,
    output wire        o_last_tile,
    output wire        o_tile_start, /* 1-cycle pulse at each new tile     */

    /* ── Element / chunk state ──────────────────────────────────────────── */
    output wire [15:0] o_t,          /* current output frame   */
    output wire [15:0] o_c,          /* current output channel */
    output wire        o_c_first,    /* first element of this channel in tile */
    output wire [15:0] o_r_len,      /* reduction length (K, c_in, or 1) */
    output wire [15:0] o_r_base,     /* current chunk's base index */
    output wire        o_last_chunk,

    /* ── Generated addresses ────────────────────────────────────────────── */
    output wire [31:0] o_in_addr,    /* gather base, activations   */
    output wire [31:0] o_in_stride,  /* gather stride in bytes     */
    output wire [LANES-1:0] o_lane_en,
    output wire [31:0] o_wt_addr,    /* gather base, weights (stride 1) */
    output wire [31:0] o_out_addr,   /* store address              */
    output wire [31:0] o_res_addr,   /* ADD residual operand       */

    /* ── Status ─────────────────────────────────────────────────────────── */
    output wire        o_valid,      /* an element is presented */
    output wire        busy,
    output reg         done          /* 1-cycle pulse when the walk finishes */
);

    localparam [1:0] OP_DW = 2'd0, OP_PW = 2'd1, OP_ADD = 2'd2, OP_REQ = 2'd3;

    localparam [31:0] LANES_U = LANES;

    /* ── Walk state ─────────────────────────────────────────────────────── */
    reg        running;
    reg [15:0] t0_q, c0_q;      /* tile origin        */
    reg [15:0] t_q,  c_q;       /* element within tile */
    reg [15:0] r_base_q;        /* reduction chunk base */
    reg [15:0] resident_end_q;  /* halo retention high-water mark */
    reg        first_tile_q;
    reg        tile_start_q;
    reg        c_first_q;       /* current element is this channel's first */

    assign busy    = running;
    assign o_valid = running;

    /* ── Latched configuration ──────────────────────────────────────────── */
    /* Held so the caller may re-point the descriptor registers freely. */
    reg [1:0]  op_q;
    reg [15:0] c_in_q, c_out_q, k_q, pad_q, t_out_q, t_tile_q, dw_ch_tile_q;
    reg [7:0]  stride_q, dil_q;
    reg [31:0] in_base_q, out_base_q, res_base_q, w_off_q;
    reg [15:0] in_pitch_q, in_cbase_q, out_pitch_q, out_cbase_q, res_pitch_q;

    /* Channel tile: depthwise tiles over channels, everything else does not. */
    wire [15:0] c_tile = (op_q == OP_DW) ? dw_ch_tile_q : c_out_q;

    /* Input frame count: depthwise consumes stride frames per output frame. */
    wire [31:0] t_in = (op_q == OP_DW)
                     ? ({16'd0, t_out_q} * {24'd0, stride_q})
                     : {16'd0, t_out_q};

    /* Reduction length per op. */
    wire [15:0] r_len = (op_q == OP_DW) ? k_q :
                        (op_q == OP_PW) ? c_in_q : 16'd1;

    /* Tile extents. */
    wire [15:0] t1_full = t0_q + t_tile_q;
    wire [15:0] t1_w    = (t1_full > t_out_q) ? t_out_q : t1_full;
    wire [15:0] c1_full = c0_q + c_tile;
    wire [15:0] c1_w    = (c1_full > c_out_q) ? c_out_q : c1_full;

    assign o_t0 = t0_q;
    assign o_t1 = t1_w;
    assign o_c0 = c0_q;
    assign o_c1 = c1_w;

    /* ── Signed working copies ──────────────────────────────────────────────
     * Every quantity below is a small non-negative index, but the padding
     * arithmetic goes transiently negative, so it has to happen in signed
     * 32-bit.  These are CONTINUOUS ASSIGNMENTS of unsigned bit patterns into
     * explicitly signed declarations — a bit copy, not a mixed-signedness
     * operator — so the RTLIL frontend never sees $signed() wrapped around an
     * unsigned expression (CLAUDE.md Yosys gotcha (b)).  All values are far
     * below 2^31, so zero-extension is exact. */
    wire signed [31:0] t0_s;      assign t0_s     = {16'd0, t0_q};
    wire signed [31:0] t_s;       assign t_s      = {16'd0, t_q};
    wire signed [31:0] t1m1_s;    assign t1m1_s   = {16'd0, (t1_w - 16'd1)};
    wire signed [31:0] km1_s;     assign km1_s    = {16'd0, (k_q  - 16'd1)};
    wire signed [31:0] rbase_s;   assign rbase_s  = {16'd0, r_base_q};
    wire signed [31:0] stride_s;  assign stride_s = {24'd0, stride_q};
    wire signed [31:0] dil_s;     assign dil_s    = {24'd0, dil_q};
    wire signed [31:0] pad_s;     assign pad_s    = {16'd0, pad_q};
    wire signed [31:0] t_in_s;    assign t_in_s   = t_in;

    /* ── Halo span for the current time tile (mirrors walk_desc) ────────────
     * DW: lo = t0*stride - pad, hi = (t1-1)*stride + (k-1)*dilation - pad,
     *     clamped into [0, t_in-1].  Everything else reads exactly [t0, t1). */
    wire signed [31:0] lo_raw = t0_s * stride_s - pad_s;
    wire signed [31:0] hi_raw = t1m1_s * stride_s + km1_s * dil_s - pad_s;

    wire signed [31:0] lo_c = (lo_raw < 32'sd0) ? 32'sd0 : lo_raw;
    wire signed [31:0] hi_c = (hi_raw > (t_in_s - 32'sd1)) ? (t_in_s - 32'sd1) : hi_raw;

    wire signed [31:0] in0_dw = lo_c;
    wire signed [31:0] in1_dw = (hi_c >= lo_c) ? (hi_c + 32'sd1) : lo_c;

    wire [15:0] in0_w = (op_q == OP_DW) ? in0_dw[15:0] : t0_q;
    wire [15:0] in1_w = (op_q == OP_DW) ? in1_dw[15:0] : t1_w;

    /* Both spans are clamped into [0, t_in], which is well under 16 bits; the
     * upper half of the signed intermediates is therefore always sign padding. */
    wire _unused_span = &{1'b0, in0_dw[31:16], in1_dw[31:16]};

    wire [15:0] new0_a = (in0_w > resident_end_q) ? in0_w : resident_end_q;
    wire [15:0] new0_w = (new0_a > in1_w) ? in1_w : new0_a;

    assign o_in0  = in0_w;
    assign o_in1  = in1_w;
    assign o_new0 = new0_w;
    assign o_new1 = in1_w;

    /* Last tile = last time tile AND last channel tile. */
    wire last_t_tile = (t1_w >= t_out_q);
    wire last_c_tile = (c1_w >= c_out_q);
    assign o_first_tile = first_tile_q;
    assign o_last_tile  = last_t_tile && last_c_tile;
    assign o_tile_start = tile_start_q;

    assign o_t       = t_q;
    assign o_c       = c_q;
    assign o_c_first = c_first_q;
    assign o_r_len   = r_len;
    assign o_r_base  = r_base_q;

    /* ── Chunk lane enables and gather addresses ────────────────────────── */

    /* Taps/channels still to cover in this reduction. */
    wire [15:0] r_rem   = r_len - r_base_q;
    wire [31:0] r_next  = {16'd0, r_base_q} + LANES_U;
    assign o_last_chunk = (r_next >= {16'd0, r_len});

    /* Base mask: low min(LANES, r_rem) lanes on.  Same unsigned construction as
     * tinymac_accel.v's lane_en (shift ones out to get an all-ones mask). */
    wire [LANES-1:0] mask_r = ~({LANES{1'b1}} << r_rem);

    /* Depthwise additionally masks taps whose input frame falls outside the
     * tensor — the C's `if (pos < 0 || pos >= t_in) continue;`. */
    wire signed [31:0] pos0 = t_s * stride_s + rbase_s * dil_s - pad_s;

    wire [LANES-1:0] mask_pos;
    genvar li;
    generate
        for (li = 0; li < LANES; li = li + 1) begin : gen_pos
            /* Explicitly sized signed localparam, NOT a signed `integer` param
             * (see the Yosys gotcha), so lane offset stays signed throughout. */
            localparam signed [31:0] LI_S = li;
            wire signed [31:0] pos_l = pos0 + LI_S * dil_s;
            assign mask_pos[li] = (pos_l >= 32'sd0) && (pos_l < t_in_s);
        end
    endgenerate

    assign o_lane_en = (op_q == OP_DW) ? (mask_r & mask_pos) : mask_r;

    /* Activation gather.
     *   DW: frame pos0, channel c, stepping dilation*pitch bytes per lane.
     *   PW: frame t,    channel r_base + lane, stepping 1 byte per lane.
     *   ADD/REQUANT: the single element (t, c); ADD reads from channel 0 (no
     *   cbase) per the golden model's read/write asymmetry, see below. */
    wire [31:0] in_row   = {16'd0, t_q} * {16'd0, in_pitch_q};
    /* Unsigned copy of the (possibly negative) DW frame index: two's-complement
     * wraparound is deliberate here — see the module header. */
    wire [31:0] pos0_u;  assign pos0_u = pos0;
    wire [31:0] dw_row   = pos0_u * {16'd0, in_pitch_q};

    wire [31:0] in_addr_dw = in_base_q + dw_row + {16'd0, in_cbase_q} + {16'd0, c_q};
    wire [31:0] in_addr_pw = in_base_q + in_row + {16'd0, in_cbase_q} + {16'd0, r_base_q};
    /* OP_ADD reads BOTH operands from channel 0 of their buffers even when the
     * descriptor writes an offset channel band (quartznet_infer.c header note 2
     * and CLAUDE.md gotcha (g)).  So: no in_cbase here, on purpose. */
    wire [31:0] in_addr_add = in_base_q + in_row + {16'd0, c_q};
    wire [31:0] in_addr_req = in_base_q + in_row + {16'd0, in_cbase_q} + {16'd0, c_q};

    assign o_in_addr = (op_q == OP_DW)  ? in_addr_dw  :
                       (op_q == OP_PW)  ? in_addr_pw  :
                       (op_q == OP_ADD) ? in_addr_add : in_addr_req;

    assign o_in_stride = (op_q == OP_DW) ? ({24'd0, dil_q} * {16'd0, in_pitch_q})
                                         : 32'd1;

    assign o_res_addr = res_base_q + ({16'd0, t_q} * {16'd0, res_pitch_q}) + {16'd0, c_q};

    /* Weight gather (always contiguous).
     *   DW: w_off + c * K + r_base
     *   PW: w_off + oc * c_in + r_base                                     */
    wire [31:0] wt_row = (op_q == OP_DW) ? ({16'd0, c_q} * {16'd0, k_q})
                                         : ({16'd0, c_q} * {16'd0, c_in_q});
    assign o_wt_addr = w_off_q + wt_row + {16'd0, r_base_q};

    /* Store address: out_base + t*out_pitch + out_cbase + c. */
    assign o_out_addr = out_base_q + ({16'd0, t_q} * {16'd0, out_pitch_q})
                      + {16'd0, out_cbase_q} + {16'd0, c_q};

    /* ── Sequencing ─────────────────────────────────────────────────────── */
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            running        <= 1'b0;
            done           <= 1'b0;
            t0_q           <= 16'd0;
            c0_q           <= 16'd0;
            t_q            <= 16'd0;
            c_q            <= 16'd0;
            r_base_q       <= 16'd0;
            resident_end_q <= 16'd0;
            first_tile_q   <= 1'b1;
            tile_start_q   <= 1'b0;
            c_first_q      <= 1'b1;
        end else begin
            done         <= 1'b0;
            tile_start_q <= 1'b0;

            if (start) begin
                /* Latch the descriptor. */
                op_q         <= cfg_op;
                c_in_q       <= cfg_c_in;
                c_out_q      <= cfg_c_out;
                k_q          <= cfg_k;
                stride_q     <= cfg_stride;
                dil_q        <= cfg_dilation;
                pad_q        <= cfg_pad;
                t_out_q      <= cfg_t_out;
                t_tile_q     <= cfg_t_tile;
                dw_ch_tile_q <= cfg_dw_ch_tile;
                in_base_q    <= cfg_in_base;
                in_pitch_q   <= cfg_in_pitch;
                in_cbase_q   <= cfg_in_cbase;
                out_base_q   <= cfg_out_base;
                out_pitch_q  <= cfg_out_pitch;
                out_cbase_q  <= cfg_out_cbase;
                res_base_q   <= cfg_res_base;
                res_pitch_q  <= cfg_res_pitch;
                w_off_q      <= cfg_w_off;

                t0_q           <= 16'd0;
                c0_q           <= 16'd0;
                t_q            <= 16'd0;
                c_q            <= 16'd0;
                r_base_q       <= 16'd0;
                resident_end_q <= 16'd0;
                first_tile_q   <= 1'b1;
                tile_start_q   <= 1'b1;
                c_first_q      <= 1'b1;
                /* A descriptor with no work at all still completes cleanly. */
                running        <= (cfg_c_out != 16'd0) && (cfg_t_out != 16'd0);
                done           <= (cfg_c_out == 16'd0) || (cfg_t_out == 16'd0);
            end else if (running) begin
                if (chunk_done) begin
                    r_base_q <= r_next[15:0];
                end
                if (elem_done) begin
                    r_base_q  <= 16'd0;
                    c_first_q <= 1'b0;
                    /* Element walk within the tile: t inner, c outer. */
                    if ((t_q + 16'd1) < t1_w) begin
                        t_q <= t_q + 16'd1;
                    end else begin
                        t_q       <= t0_q;
                        c_first_q <= 1'b1;
                        if ((c_q + 16'd1) < c1_w) begin
                            c_q <= c_q + 16'd1;
                        end else begin
                            /* Tile finished — advance the tile grid. */
                            first_tile_q <= 1'b0;
                            tile_start_q <= 1'b1;
                            if (!last_c_tile) begin
                                /* Next channel tile, same time tile. */
                                c0_q <= c1_w;
                                c_q  <= c1_w;
                                t_q  <= t0_q;
                            end else if (!last_t_tile) begin
                                /* Next time tile: channels restart, halo
                                 * high-water mark carries forward. */
                                resident_end_q <= in1_w;
                                t0_q <= t1_w;
                                t_q  <= t1_w;
                                c0_q <= 16'd0;
                                c_q  <= 16'd0;
                            end else begin
                                running      <= 1'b0;
                                done         <= 1'b1;
                                tile_start_q <= 1'b0;
                            end
                        end
                    end
                end
            end
        end
    end

endmodule

`default_nettype wire
