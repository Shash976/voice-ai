`timescale 1 ns / 1 ps
/* int8_mac_array.v — parallel int8 multiply-accumulate lanes.
 *
 * Combinational. Computes the partial sum over LANES lanes:
 *
 *     psum = sum_{i : lane_en[i]} ( (in_i - in_zp) * wt_i )
 *
 * where in_i and wt_i are signed int8 operands packed little-endian into
 * in_bytes / wt_bytes (lane i occupies bits [8*i +: 8]).  in_zp is the
 * shared input zero-point that the software path subtracts before the
 * multiply (see tiny_vad_infer.c / sim_main.cpp accel_execute).
 *
 * lane_en masks off lanes in the final (tail) chunk when K is not a
 * multiple of LANES, so disabled lanes contribute zero.
 *
 * Operand range:  in_i - in_zp  spans roughly [-255, 255] (10-bit signed),
 * wt_i spans [-128, 127] (8-bit signed) → each product fits in 18 bits.
 * The sum of up to 16 such products fits comfortably in 32 bits.
 */

`default_nettype none

module int8_mac_array #(
    parameter integer LANES = 4
) (
    input  wire [LANES*8-1:0]  in_bytes,   /* LANES signed int8 inputs  */
    input  wire [LANES*8-1:0]  wt_bytes,   /* LANES signed int8 weights */
    input  wire signed [8:0]   in_zp,      /* input zero-point (subtracted) */
    input  wire [LANES-1:0]    lane_en,    /* 1 = lane participates */
    output wire signed [31:0]  psum        /* sum of enabled lane products */
);

    /* Per-lane products, sign-extended to 32 bits for the reduction. */
    wire signed [31:0] prod [0:LANES-1];

    genvar i;
    generate
        for (i = 0; i < LANES; i = i + 1) begin : gen_lane
            wire signed [7:0]  in_i = $signed(in_bytes[8*i +: 8]);
            wire signed [7:0]  wt_i = $signed(wt_bytes[8*i +: 8]);
            /* (in_i - in_zp): 9-bit operand widened so the subtraction
             * cannot overflow, then multiplied by the int8 weight. */
            wire signed [9:0]  in_adj = $signed({in_i[7], in_i}) - in_zp;
            wire signed [17:0] mul    = in_adj * wt_i;
            /* sign-extend to 32 bits; both ternary branches kept signed so
             * strict Yosys frontends don't assert on a signedness mismatch. */
            assign prod[i] = lane_en[i] ? $signed({{14{mul[17]}}, mul}) : 32'sd0;
        end
    endgenerate

    /* Balanced reduction of the lane products.  Synthesis builds an adder
     * tree from this; written as a simple accumulate for clarity. */
    integer j;
    reg signed [31:0] acc;
    always @* begin
        acc = 32'sd0;
        for (j = 0; j < LANES; j = j + 1)
            acc = acc + prod[j];
    end

    assign psum = acc;

endmodule

`default_nettype wire
