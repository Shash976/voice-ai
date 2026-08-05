`timescale 1 ns / 1 ps
/* requantize.v — fixed-point requantize + zero-point + ReLU + int8 clamp,
 * pipelined across 2 cycles.
 *
 * Bit-exact hardware mirror of the software path:
 *   - tiny_vad_infer.c  requantize()   (the Q31 multiply + signed shift)
 *   - sim_main.cpp       accel_execute() (the + out_zp, ReLU, clamp tail)
 *
 * Reference C (signed int64 throughout):
 *     int64_t val = (int64_t)q_mult * (int64_t)acc;
 *     val += (1<<30);                 // round before Q31 shift
 *     val >>= 31;                     // arithmetic
 *     if      (shift > 0) { val += (1 << (shift-1)); val >>= shift; }
 *     else if (shift < 0) {  val <<= (-shift); }
 *     int32_t r = (int32_t)val + out_zp;
 *     if (relu && r < out_zp) r = out_zp;
 *     out = clamp(r, -128, 127);
 *
 * All shifts are arithmetic (sign-preserving). `shift` is signed: a positive
 * value right-shifts (typical conv/dense), a negative value left-shifts
 * (e.g. the global-average-pool layer when sc_in > sc_out).
 *
 * ── PIPELINE ────────────────────────────────────────────────────────────────
 * The 32x32->64 signed multiply was measured as the whole accelerator's
 * critical path (nangate45: the design would not close faster than ~3.7 ns,
 * and that limit was independent of LANES).  It is now split:
 *
 *   stage 1 (registered):  prod = q_mult * acc, plus a cheap decode of the
 *                          shift controls that travel with the data
 *   stage 2 (combinational from the stage-1 registers):
 *                          Q31 round, variable shift, + out_zp, ReLU, clamp
 *
 * `out_q` is deliberately left combinational: the consumer already registers
 * it, so the datapath is exactly two register-to-register stages
 *   [multiply] -> prod_q -> [round/shift/zp/clamp] -> consumer's output reg.
 *
 * Handshake: pulse `in_valid` for one cycle with the operands; `out_valid`
 * pulses one cycle later with `out_q` valid alongside it.  The block is fully
 * pipelined (one result per cycle is legal), though the sequencer only
 * launches one per output channel.
 *
 * NOTE — the arithmetic below is character-for-character the same as the
 * previous combinational version; only the register boundary is new, so
 * results stay bit-identical.
 *
 * INTERFACE CONTRACT: `out_zp` and `relu` are NOT pipelined.  They are
 * whole-operation configuration in the parent (latched at `start`, held
 * constant for every channel of an op), so they are guaranteed stable while a
 * result is in flight.  Pipelining them would cost 33 flops for no functional
 * gain.  If this block is ever reused somewhere they can change per launch,
 * they must be registered alongside the shift controls.
 */

`default_nettype none

module requantize (
    input  wire               clk,
    input  wire               rst_n,

    /* ── Stage-1 launch ─────────────────────────────────────────────────── */
    input  wire               in_valid,  /* 1-cycle strobe: capture operands */
    input  wire signed [31:0] acc,       /* accumulated dot product (post-saturation) */
    input  wire signed [31:0] q_mult,    /* Q31 multiplier (per output channel) */
    input  wire signed [31:0] shift,     /* signed: >0 right, <0 left, 0 none */

    /* ── Whole-op configuration (must be stable while a result is in flight) */
    input  wire signed [31:0] out_zp,    /* output zero-point added after scaling */
    input  wire               relu,      /* 1 = clamp values below out_zp up to out_zp */

    /* ── Stage-2 result ─────────────────────────────────────────────────── */
    output reg                out_valid, /* 1 cycle after in_valid */
    output wire signed [7:0]  out_q,     /* requantized int8 result */

    /* Scaled result BEFORE +out_zp / ReLU / int8 clamp — i.e. exactly what the
     * software `requantize()` returns, truncated to int32 the same way its
     * `return (int32_t)val;` does.  Valid alongside out_valid.
     *
     * This exists for the TFLite ADD path (requantize_add.v), which pre-scales
     * two operands into a shared domain and must sum them at full int32 width
     * before the final requantize.  Purely additive: out_q and every existing
     * port are bit-identical to before. */
    output wire signed [31:0] out_raw
);

    /* ── Stage 1: the 64-bit signed multiply (the critical path) ────────── */
    wire signed [63:0] prod = $signed(q_mult) * $signed(acc);

    /* Shift-control decode, computed alongside the multiply and registered
     * with it so stage 2 never depends on the `shift` input — the parent
     * re-points that at the next output channel while a result is in flight.
     * Shift amounts are taken as explicitly *unsigned* slices so no signed/
     * unsigned operand mixing reaches the RTLIL generator. */
    wire       sh_pos_d = (shift > 0);            /* right shift */
    wire       sh_neg_d = (shift < 0);            /* left shift  */
    wire [5:0] rshift_d = shift[5:0];             /* right-shift magnitude */
    wire [5:0] lshift_d = (6'd0 - shift[5:0]);    /* left-shift magnitude  */

    /* Datapath registers carry no reset on purpose: they are qualified by
     * out_valid, so resetting 76 bits of datapath would only add reset pins
     * (area) for no observable behaviour. */
    reg signed [63:0] prod_q;
    reg               sh_pos_q, sh_neg_q;
    reg [5:0]         rshift_q, lshift_q;

    always @(posedge clk) begin
        if (in_valid) begin
            prod_q   <= prod;
            sh_pos_q <= sh_pos_d;
            sh_neg_q <= sh_neg_d;
            rshift_q <= rshift_d;
            lshift_q <= lshift_d;
        end
    end

    /* The valid strobe is the only state that needs a reset. */
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) out_valid <= 1'b0;
        else        out_valid <= in_valid;
    end

    /* ── Stage 2: Q31 round, variable shift, zero-point, ReLU, clamp ────── */

    /* Round and shift down by 31 (Q31).  +(1<<30) then arithmetic >> 31. */
    wire signed [63:0] q31 = (prod_q + 64'sd1073741824) >>> 31;

    /* Variable signed shift.  Shift magnitude is small (< 32) in practice. */
    reg signed [63:0] shifted;
    always @* begin
        if (sh_pos_q)
            /* round-to-nearest right shift: +(1 << (shift-1)) then >>> shift */
            shifted = (q31 + (64'sd1 <<< (rshift_q - 6'd1))) >>> rshift_q;
        else if (sh_neg_q)
            shifted = q31 <<< lshift_q;
        else
            shifted = q31;
    end

    /* Software `requantize()` returns int32; expose the same truncation. */
    assign out_raw = shifted[31:0];

    /* + out_zp, then ReLU floor at out_zp, then clamp to int8. */
    wire signed [63:0] out_zp_e = $signed({{32{out_zp[31]}}, out_zp});
    wire signed [63:0] biased   = shifted + out_zp_e;
    wire signed [63:0] reld     = (relu && (biased < out_zp_e)) ? out_zp_e : biased;

    /* int8 clamp. reld[7:0] wrapped in $signed so every ternary branch is
     * signed (strict Yosys frontends assert on mixed-signedness selects). */
    assign out_q = (reld >  64'sd127)  ?  8'sd127 :
                   (reld < -64'sd128)  ? -8'sd128 :
                   $signed(reld[7:0]);

endmodule

`default_nettype wire
