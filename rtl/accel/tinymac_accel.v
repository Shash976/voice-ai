`timescale 1 ns / 1 ps
/* tinymac_accel.v — synthesizable int8 matrix-vector accelerator core.
 *
 * This is the Stage-6 RTL realization of the Stage-4 behavioral model
 * (sim_main.cpp accel_execute, cmd=1 MATVEC).  It computes, output-stationary,
 * one output channel at a time:
 *
 *     out[m] = clamp( requantize( bias[m] + sum_{k} (in[k]-in_zp)*W[m][k],
 *                                 qmult[m], rshift[m] ) + out_zp )
 *
 * for m in [0, M).  Each output channel accumulates K products LANES at a
 * time, so it takes ceil(K/LANES) MAC cycles plus a few control cycles.
 *
 * SCOPE (Stage-6 decision: "accelerator core only, no memory mastering").
 * The core does NOT master the 256 KB system bus.  Instead it exposes the
 * indices it currently needs (o_m, o_k_base) and consumes operands provided
 * combinationally by the environment (0-wait-state, mirroring the sim's RAM
 * model).  A thin MMIO/DMA wrapper around this core — the full peripheral
 * with the 0x20000000 register file — is a separate, later concern; it adds
 * no new arithmetic, which is what physical design measures.
 *
 * CONV1D is intentionally out of scope for the first GDS: in hardware it is
 * lowered to matvec (im2col) by firmware and reuses this exact datapath.
 *
 * Parameters:
 *   LANES  — parallel int8 MAC lanes (Stage-5 optimum = 4)
 *   ACC_W  — accumulator width in bits; saturates per accumulate step.
 *            >=32 → no saturation (bit-exact with int32 software path).
 *            24   → safe for all TinyVAD layers (Stage-5 optimum).
 */

`default_nettype none

module tinymac_accel #(
    parameter integer LANES = 4,
    parameter integer ACC_W = 24
) (
    input  wire        clk,
    input  wire        rst_n,

    /* ── Control / configuration (latched on start) ─────────────────────── */
    input  wire        start,        /* pulse high for 1 cycle to begin */
    input  wire [15:0] cfg_m,        /* number of output channels (rows) */
    input  wire [15:0] cfg_k,        /* input vector length (cols) */
    input  wire [31:0] cfg_in_zp,    /* input zero-point (signed) */
    input  wire [31:0] cfg_out_zp,   /* output zero-point (signed) */
    input  wire        cfg_relu,     /* 1 = ReLU after requantize */

    /* ── Operand interface (combinational, 0-wait-state) ────────────────────
     * Environment supplies the LANES-wide input/weight chunk addressed by
     * (o_m, o_k_base), plus this channel's bias/qmult/rshift addressed by o_m.
     */
    output wire [15:0] o_m,          /* current output channel index */
    output wire [15:0] o_k_base,     /* base input index of the current chunk */
    input  wire [LANES*8-1:0] i_in_chunk,  /* LANES signed int8 inputs  */
    input  wire [LANES*8-1:0] i_wt_chunk,  /* LANES signed int8 weights */
    input  wire signed [31:0] i_bias,      /* bias[o_m] */
    input  wire signed [31:0] i_qmult,     /* qmult[o_m] */
    input  wire signed [31:0] i_rshift,    /* rshift[o_m] (signed) */

    /* ── Result stream ──────────────────────────────────────────────────── */
    output reg         o_out_valid,  /* 1-cycle strobe per output channel */
    output reg  [15:0] o_out_m,      /* channel index of o_out_data */
    output reg  signed [7:0] o_out_data,

    /* ── Status ─────────────────────────────────────────────────────────── */
    output wire        busy,
    output reg         done,         /* 1-cycle pulse when the op completes */
    output reg  [31:0] o_last_cycles /* cycles taken by the last op */
);

    /* ── FSM states ─────────────────────────────────────────────────────── */
    localparam [2:0] S_IDLE    = 3'd0,
                     S_INIT_CH = 3'd1,
                     S_MAC     = 3'd2,
                     S_REQ     = 3'd3,
                     S_DONE    = 3'd4;

    reg [2:0]  state;
    reg [15:0] m_cnt;       /* current output channel */
    reg [15:0] k_base;      /* current input chunk base index */
    reg [15:0] M_reg, K_reg;
    reg signed [8:0] in_zp_reg;     /* zero-points fit int8 range */
    reg [31:0] out_zp_reg;
    reg        relu_reg;
    reg signed [31:0] acc;
    reg [31:0] cyc_cnt;     /* cycles since start */

    assign o_m      = m_cnt;
    assign o_k_base = k_base;
    assign busy     = (state != S_IDLE);

    /* Zero-points fit the int8 range; upper config bits are intentionally
     * unused.  Tie them off so lint stays clean. */
    wire _unused_ok = &{1'b0, cfg_in_zp[31:9]};

    /* Unsigned copies of the (signed `integer`) parameters, so they never mix
     * signedness with unsigned signals in expressions below — strict Yosys
     * frontends assert on signed-integer/unsigned-signal operand mismatches. */
    localparam [31:0] LANES_U = LANES;

    /* ── Tail-chunk lane enable: enable min(LANES, K-k_base) low lanes ─────
     * Built as an unsigned mask (no signed loop variable): when fewer than
     * LANES inputs remain, only the low k_rem lanes are enabled. k_rem >= 1
     * whenever this is used (state S_MAC). */
    wire [15:0] k_rem  = K_reg - k_base;
    wire [31:0] k_next = {16'd0, k_base} + LANES_U;        /* next chunk base */
    /* Low min(LANES, k_rem) bits set, all-unsigned, exactly LANES wide:
     * shifting LANES ones left by k_rem and inverting leaves the low k_rem
     * bits set; when k_rem >= LANES everything shifts out → all lanes on. */
    wire [LANES-1:0] lane_en = ~({LANES{1'b1}} << k_rem);

    /* ── Combinational MAC array ────────────────────────────────────────── */
    wire signed [31:0] psum;
    int8_mac_array #(.LANES(LANES)) u_mac (
        .in_bytes (i_in_chunk),
        .wt_bytes (i_wt_chunk),
        .in_zp    (in_zp_reg),
        .lane_en  (lane_en),
        .psum     (psum)
    );

    /* ── Accumulator saturation to ACC_W bits ───────────────────────────── */
    wire signed [31:0] acc_sum = acc + psum;
    reg  signed [31:0] acc_sat;
    /* Saturation bounds. Only referenced when ACC_W < 32 (the always-block
     * below short-circuits the >=32 case), so the ACC_W=32 wraparound of these
     * constants is harmless. Kept in clean signed form to avoid frontend
     * signedness asserts. */
    localparam signed [31:0] ACC_HI = (32'sd1 <<< (ACC_W - 1)) - 32'sd1;
    localparam signed [31:0] ACC_LO = -(32'sd1 <<< (ACC_W - 1));
    always @* begin
        if (ACC_W >= 32)        acc_sat = acc_sum;
        else if (acc_sum > ACC_HI) acc_sat = ACC_HI;
        else if (acc_sum < ACC_LO) acc_sat = ACC_LO;
        else                       acc_sat = acc_sum;
    end

    /* ── Combinational requantize datapath ──────────────────────────────── */
    wire signed [7:0] out_q;
    requantize u_rq (
        .acc    (acc),
        .q_mult (i_qmult),
        .shift  (i_rshift),
        .out_zp ($signed(out_zp_reg)),
        .relu   (relu_reg),
        .out_q  (out_q)
    );

    /* ── Sequencer ──────────────────────────────────────────────────────── */
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state       <= S_IDLE;
            m_cnt       <= 16'd0;
            k_base      <= 16'd0;
            acc         <= 32'sd0;
            o_out_valid <= 1'b0;
            o_out_m     <= 16'd0;
            o_out_data  <= 8'sd0;
            done        <= 1'b0;
            cyc_cnt     <= 32'd0;
            o_last_cycles <= 32'd0;
        end else begin
            o_out_valid <= 1'b0;
            done        <= 1'b0;
            if (state != S_IDLE)
                cyc_cnt <= cyc_cnt + 32'd1;

            case (state)
            /* ---------------------------------------------------------- */
            S_IDLE: begin
                if (start) begin
                    M_reg      <= cfg_m;
                    K_reg      <= cfg_k;
                    in_zp_reg  <= $signed(cfg_in_zp[8:0]);
                    out_zp_reg <= cfg_out_zp;
                    relu_reg   <= cfg_relu;
                    m_cnt      <= 16'd0;
                    cyc_cnt    <= 32'd0;
                    state      <= S_INIT_CH;
                end
            end
            /* ---------------------------------------------------------- */
            S_INIT_CH: begin
                acc    <= i_bias;       /* seed accumulator with bias[m] */
                k_base <= 16'd0;
                /* zero-length input → straight to requantize on the bias */
                state  <= (K_reg == 16'd0) ? S_REQ : S_MAC;
            end
            /* ---------------------------------------------------------- */
            S_MAC: begin
                acc <= acc_sat;
                if (k_next >= {16'd0, K_reg})
                    state <= S_REQ;
                else
                    k_base <= k_next[15:0];
            end
            /* ---------------------------------------------------------- */
            S_REQ: begin
                o_out_valid <= 1'b1;
                o_out_m     <= m_cnt;
                o_out_data  <= out_q;
                if ((m_cnt + 16'd1) >= M_reg)
                    state <= S_DONE;
                else begin
                    m_cnt <= m_cnt + 16'd1;
                    state <= S_INIT_CH;
                end
            end
            /* ---------------------------------------------------------- */
            S_DONE: begin
                done          <= 1'b1;
                o_last_cycles <= cyc_cnt;
                state         <= S_IDLE;
            end
            default: state <= S_IDLE;
            endcase
        end
    end

endmodule

`default_nettype wire
