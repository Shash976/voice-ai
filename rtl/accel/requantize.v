`timescale 1 ns / 1 ps
/* requantize.v — fixed-point requantize + zero-point + ReLU + int8 clamp.
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
 * Combinational.  The 64-bit signed multiply here is expected to be the
 * critical path of the accelerator — exactly the datapath Stage 6 measures.
 */

`default_nettype none

module requantize (
    input  wire signed [31:0] acc,      /* accumulated dot product (post-saturation) */
    input  wire signed [31:0] q_mult,   /* Q31 multiplier (per output channel) */
    input  wire signed [31:0] shift,    /* signed: >0 right, <0 left, 0 none */
    input  wire signed [31:0] out_zp,   /* output zero-point added after scaling */
    input  wire               relu,     /* 1 = clamp values below out_zp up to out_zp */
    output wire signed [7:0]  out_q     /* requantized int8 result */
);

    /* val = q_mult * acc, full 64-bit signed product. */
    wire signed [63:0] prod = $signed(q_mult) * $signed(acc);

    /* Round and shift down by 31 (Q31).  +(1<<30) then arithmetic >> 31. */
    wire signed [63:0] q31 = (prod + 64'sd1073741824) >>> 31;

    /* Variable signed shift.  Shift magnitude is small (< 32) in practice. */
    reg signed [63:0] shifted;
    always @* begin
        if (shift > 0)
            /* round-to-nearest right shift: +(1 << (shift-1)) then >>> shift */
            shifted = (q31 + (64'sd1 <<< (shift[5:0] - 6'd1))) >>> shift[5:0];
        else if (shift < 0)
            shifted = q31 <<< ((-shift) & 32'h3F);
        else
            shifted = q31;
    end

    /* + out_zp, then ReLU floor at out_zp, then clamp to int8. */
    wire signed [63:0] out_zp_e = {{32{out_zp[31]}}, out_zp};
    wire signed [63:0] biased   = shifted + out_zp_e;
    wire signed [63:0] reld     = (relu && (biased < out_zp_e)) ? out_zp_e : biased;

    /* int8 clamp. reld[7:0] wrapped in $signed so every ternary branch is
     * signed (strict Yosys frontends assert on mixed-signedness selects). */
    assign out_q = (reld >  64'sd127)  ?  8'sd127 :
                   (reld < -64'sd128)  ? -8'sd128 :
                   $signed(reld[7:0]);

endmodule

`default_nettype wire
