`timescale 1 ns / 1 ps
/* requantize_add.v — TFLite elementwise-ADD requantize (per-tensor, 3 mults).
 *
 * Bit-exact hardware mirror of quartznet_ref.py::RefRunner._acc_add() and
 * firmware/quartznet/quartznet_infer.c::tile_add() + qn_store():
 *
 *     mv = main[t][c] - in_zp;          // int32, caller supplies
 *     rv = res [t][c] - res_zp;         // int32, caller supplies
 *     a  = requantize(mv << 20, qmult[0], rshift[0]);
 *     b  = requantize(rv << 20, qmult[1], rshift[1]);
 *     r  = requantize(a + b,    qmult[2], rshift[2]) + out_zp;
 *     if (relu && r < out_zp) r = out_zp;
 *     out = clamp_i8(r);
 *
 * ADD carries NO bias and has exactly three qparam entries, per tensor rather
 * than per output channel — that is the whole reason this wrapper exists.
 *
 * ── Why a wrapper and not a mode bit ────────────────────────────────────────
 *
 * The base `requantize` block is already fully pipelined (one launch per cycle
 * is legal), so all three multiplies fit through ONE instance of it by simply
 * launching on three different cycles.  That keeps a single 32x32->64 multiplier
 * — the block's dominant area and its measured critical path — instead of the
 * three a mode bit would have inferred, and it leaves the per-channel path in
 * `requantize` completely untouched, so the existing TinyVAD core stays
 * bit-identical.
 *
 * ── Schedule (cycle 0 = the cycle `in_valid` is asserted) ───────────────────
 *
 *   cyc 0  launch #0: (mv << 20, qmult0, rshift0)
 *   cyc 1  launch #1: (rv << 20, qmult1, rshift1);  capture raw #0
 *   cyc 2  (idle);                                   capture raw #1
 *   cyc 3  launch #2: (raw0 + raw1, qmult2, rshift2)
 *   cyc 4  out_valid: out_q is the final int8
 *
 * The two pre-scales take `out_raw` (pre-zero-point, pre-clamp int32), so
 * out_zp/relu need no sequencing at all — they simply stay at their real values
 * for the whole operation, honouring the base block's "stable while a result is
 * in flight" contract rather than working around it.
 *
 * The << 20 cannot overflow int32: |mv| <= 255 and 255 << 20 = 267,386,880.
 */

`default_nettype none

module requantize_add (
    input  wire               clk,
    input  wire               rst_n,

    /* ── Launch ─────────────────────────────────────────────────────────── */
    input  wire               in_valid,  /* 1-cycle strobe */
    input  wire signed [31:0] mv,        /* main operand, zero-point removed */
    input  wire signed [31:0] rv,        /* residual operand, zero-point removed */

    /* Per-tensor qparams; stable for the whole operation. */
    input  wire signed [31:0] qmult0,
    input  wire signed [31:0] rshift0,
    input  wire signed [31:0] qmult1,
    input  wire signed [31:0] rshift1,
    input  wire signed [31:0] qmult2,
    input  wire signed [31:0] rshift2,

    input  wire signed [31:0] out_zp,
    input  wire               relu,

    /* ── Result ─────────────────────────────────────────────────────────── */
    output wire               out_valid, /* 4 cycles after in_valid */
    output wire signed [7:0]  out_q
);

    localparam [2:0] A_IDLE = 3'd0,
                     A_L1   = 3'd1,   /* launching #1, capturing raw #0 */
                     A_C1   = 3'd2,   /* capturing raw #1                */
                     A_L2   = 3'd3,   /* launching #2                    */
                     A_WAIT = 3'd4;   /* final result pops               */

    reg [2:0] state;

    /* Operands captured at launch so the caller may move on immediately. */
    reg signed [31:0] mv_q, rv_q;
    reg signed [31:0] raw0_q, raw1_q;

    /* ── Operand / control multiplexing into the single requantize core ──── */
    reg               rq_launch;
    reg signed [31:0] rq_acc;
    reg signed [31:0] rq_qmult;
    reg signed [31:0] rq_shift;

    /* QN_ADD_LEFT_SHIFT == 20 (quartznet_infer.h). Both ternary branches are
     * kept signed so strict Yosys frontends see no mixed-signedness select. */
    wire signed [31:0] mv_sh = mv_q <<< 20;
    wire signed [31:0] rv_sh = rv_q <<< 20;
    wire signed [31:0] sum   = raw0_q + raw1_q;

    always @* begin
        /* Defaults: no launch. */
        rq_launch = 1'b0;
        rq_acc    = 32'sd0;
        rq_qmult  = 32'sd0;
        rq_shift  = 32'sd0;
        if (in_valid && (state == A_IDLE)) begin
            /* Launch #0 straight off the input port — mv_q is not yet loaded. */
            rq_launch = 1'b1;
            rq_acc    = mv <<< 20;
            rq_qmult  = qmult0;
            rq_shift  = rshift0;
        end else if (state == A_L1) begin
            rq_launch = 1'b1;
            rq_acc    = rv_sh;
            rq_qmult  = qmult1;
            rq_shift  = rshift1;
        end else if (state == A_L2) begin
            rq_launch = 1'b1;
            rq_acc    = sum;
            rq_qmult  = qmult2;
            rq_shift  = rshift2;
        end
    end

    wire               rq_valid;
    wire signed [31:0] rq_raw;

    requantize u_rq (
        .clk      (clk),
        .rst_n    (rst_n),
        .in_valid (rq_launch),
        .acc      (rq_acc),
        .q_mult   (rq_qmult),
        .shift    (rq_shift),
        .out_zp   (out_zp),
        .relu     (relu),
        .out_valid(rq_valid),
        .out_q    (out_q),
        .out_raw  (rq_raw)
    );

    /* mv_sh is unused (launch #0 reads the port directly); keep lint quiet. */
    wire _unused_ok = &{1'b0, mv_sh};

    /* Aligned with out_q, which is combinational out of the base block's
     * stage-2 logic and is therefore only valid on the cycle rq_valid pulses. */
    assign out_valid = (state == A_WAIT) && rq_valid;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state     <= A_IDLE;
            mv_q      <= 32'sd0;
            rv_q      <= 32'sd0;
            raw0_q    <= 32'sd0;
            raw1_q    <= 32'sd0;
        end else begin
            case (state)
            A_IDLE: if (in_valid) begin
                mv_q  <= mv;
                rv_q  <= rv;
                state <= A_L1;
            end
            A_L1: begin
                /* Result of launch #0 pops now. */
                if (rq_valid) raw0_q <= rq_raw;
                state <= A_C1;
            end
            A_C1: begin
                /* Result of launch #1 pops now. */
                if (rq_valid) raw1_q <= rq_raw;
                state <= A_L2;
            end
            A_L2: begin
                state <= A_WAIT;
            end
            A_WAIT: begin
                /* Result of launch #2 pops now — the final int8 (see out_valid). */
                if (rq_valid) state <= A_IDLE;
            end
            default: state <= A_IDLE;
            endcase
        end
    end

endmodule

`default_nettype wire
