`timescale 1 ns / 1 ps
/* quartznet_accel.v — descriptor-driven QuartzNet inference core.
 *
 * Generalizes tinymac_accel.v's output-stationary FSM from one op shape (dense
 * matvec) to the four QN_DESC ops, and gives it the address generation and
 * memory mastering that core deliberately left to its environment.
 *
 *   OP_DW      depthwise:  out[t][c]  = bias[c] + sum_k (in[t*s + k*d - p][c] - in_zp) * w[c][k]
 *   OP_PW      pointwise:  out[t][oc] = bias[oc] + sum_ic (in[t][ic] - in_zp) * w[oc][ic]
 *   OP_ADD     residual:   per-tensor 3-multiplier TFLite ADD (see requantize_add.v)
 *   OP_REQUANT elementwise rescale: acc = (in[t][c] - in_zp) + bias[c]
 *
 * `int8_mac_array.v` is instantiated UNCHANGED — depthwise and pointwise differ
 * only in how addr_gen maps lanes to addresses, which is the reuse insight in
 * docs/07_quartznet_pivot.md.  OP_REQUANT rides the same array as a one-tap
 * reduction against a synthetic weight of 1, so it needs no separate datapath.
 *
 * ── Scope of this increment ─────────────────────────────────────────────────
 *
 * ONE DESCRIPTOR PER CMD.  Software stages a descriptor's full field set into
 * the register file, writes CTRL.START, and polls STATUS.BUSY — the same
 * staging convention firmware/tinyengine_port/accel.c already uses for the
 * TinyVAD core.  Autonomous in-ROM table walking (fetch descriptor i, execute,
 * i++) is a later increment; the register field set below is deliberately a
 * one-to-one image of the QN_DESC record so that increment can drive these same
 * registers from a fetched record without changing this interface.
 *
 * Not attempted here: fakeram45 macro wrapping (no behavioral model available),
 * and any performance-oriented overlap.  The sequencer blocks on each memory
 * response rather than pipelining prefetch against compute — correctness of the
 * addressing and the dataflow is what this increment is for, and the cycle cost
 * of the real machine is already modelled independently in
 * sim/quartznet_cycles/.
 *
 * ── MMIO register map (word addresses; byte address = index * 4) ────────────
 *
 * Style follows the field-layout comment at the top of
 * sw/tinyml_reference/quartznet_descriptors.py.
 *
 *  idx  name         acc  notes
 *   0   CTRL          W   bit0 START (self-clearing, ignored while busy)
 *   1   STATUS        R   bit0 BUSY, bit1 DONE (sticky; write 1 to clear)
 *   2   CYCLES        R   cycle count of the last completed operation
 *   3   OP            RW  0 DW, 1 PW, 2 ADD, 3 REQUANT      (QN_DESC w0[7:0])
 *   4   FLAGS         RW  bit0 RELU                         (QN_DESC w0[15:8])
 *   5   C_IN          RW                                    (QN_DESC w1[15:0])
 *   6   C_OUT         RW                                    (QN_DESC w1[31:16])
 *   7   K             RW                                    (QN_DESC w2[15:0])
 *   8   STRIDE        RW                                    (QN_DESC w2[23:16])
 *   9   DILATION      RW                                    (QN_DESC w2[31:24])
 *  10   PAD           RW                                    (QN_DESC w3[15:0])
 *  11   T_OUT         RW  output frames for this utterance  (table header)
 *  12   T_TILE        RW  time tile                         (table header)
 *  13   DW_CH_TILE    RW  depthwise channel tile            (table header)
 *  14   IN_ZP         RW  signed                            (QN_DESC w5[7:0])
 *  15   OUT_ZP        RW  signed                            (QN_DESC w5[15:8])
 *  16   RES_ZP        RW  signed                            (QN_DESC w5[23:16])
 *  17   IN_BASE       RW  byte address of in_buf in PSRAM
 *  18   IN_PITCH      RW  in_buf channel count (row pitch)
 *  19   IN_CBASE      RW  in_off reduced mod pitch          (QN_DESC w6)
 *  20   OUT_BASE      RW  byte address of out_buf in PSRAM
 *  21   OUT_PITCH     RW  out_buf channel count
 *  22   OUT_CBASE     RW  out_off                           (QN_DESC w7)
 *  23   RES_BASE      RW  byte address of res_buf in PSRAM
 *  24   RES_PITCH     RW  res_buf channel count
 *  25   W_OFF         RW  weight blob byte offset           (QN_DESC w9)
 *  26   BIAS_OFF      RW  qparam blob byte offset           (QN_DESC w10)
 *  27   QMULT_OFF     RW                                    (QN_DESC w11)
 *  28   RSHIFT_OFF    RW                                    (QN_DESC w12)
 *
 * The buffer BASE/PITCH registers are resolved by software from the table's
 * buffer records and the descriptor's in_buf/out_buf/res_buf ids; the hardware
 * never parses buffer records.  IN_CBASE is staged already reduced modulo the
 * pitch (the C's `in_off % pitch`), so there is no divider in the datapath.
 *
 * Parameters:
 *   LANES — parallel int8 MAC lanes; 32 is the QuartzNet design point.
 *   ACC_W — accumulator width.  MUST be 32 for QuartzNet: ACC_W=24 saturates
 *           for any pointwise with c_in >= 260, i.e. every layer from B1 on
 *           (CLAUDE.md gotcha (b)).  Kept parameterized only so the saturation
 *           logic stays structurally identical to tinymac_accel.v.
 */

`default_nettype none

module quartznet_accel #(
    parameter integer LANES        = 32,
    parameter integer ACC_W        = 32,
    parameter integer QSPI_BYTES   = 1 << 20,
    parameter integer PSRAM_BYTES  = 1 << 20,
    parameter integer QSPI_LAT     = 6,
    parameter integer PSRAM_LAT    = 3,
    parameter integer LAT_JITTER   = 1
) (
    input  wire        clk,
    input  wire        rst_n,

    /* ── MMIO ───────────────────────────────────────────────────────────── */
    input  wire        mmio_we,
    input  wire [7:0]  mmio_addr,     /* word index, see the map above */
    input  wire [31:0] mmio_wdata,
    output reg  [31:0] mmio_rdata,

    /* ── Simulation backdoor into the behavioral memory model ───────────── */
    input  wire        bd_en,
    input  wire        bd_sel,
    input  wire        bd_we,
    input  wire [31:0] bd_addr,
    input  wire [7:0]  bd_wdata,
    output wire [7:0]  bd_rdata,

    /* ── Tile-walk observability ────────────────────────────────────────────
     * The compute-tile grid addr_gen walks, pulsed on o_tile_start.  This is
     * how the testbench checks the walk against the authoritative schedule that
     * firmware/quartznet/qn_schedule dumps out of the C interpreter — including
     * the halo span, which is invisible in the final activations because
     * re-reading a retained frame yields the same answer.  A later increment
     * drives a real activation cache from these. */
    output wire        o_tile_start,
    output wire [15:0] o_t0,
    output wire [15:0] o_t1,
    output wire [15:0] o_c0,
    output wire [15:0] o_c1,
    output wire [15:0] o_in0,
    output wire [15:0] o_in1,
    output wire [15:0] o_new0,
    output wire [15:0] o_new1,
    output wire        o_first_tile,
    output wire        o_last_tile,

    /* ── Status ─────────────────────────────────────────────────────────── */
    output wire        busy,
    output wire        done_pulse
);

    localparam [1:0] OP_DW = 2'd0, OP_PW = 2'd1, OP_ADD = 2'd2, OP_REQ = 2'd3;

    /* ── Register file ──────────────────────────────────────────────────── */
    reg [31:0] r_op, r_flags, r_c_in, r_c_out, r_k, r_stride, r_dilation, r_pad;
    reg [31:0] r_t_out, r_t_tile, r_dw_ch_tile;
    reg [31:0] r_in_zp, r_out_zp, r_res_zp;
    reg [31:0] r_in_base, r_in_pitch, r_in_cbase;
    reg [31:0] r_out_base, r_out_pitch, r_out_cbase;
    reg [31:0] r_res_base, r_res_pitch;
    reg [31:0] r_w_off, r_bias_off, r_qmult_off, r_rshift_off;

    reg        start_pulse;
    reg        done_sticky;
    reg [31:0] cyc_cnt, last_cycles;

    wire ag_busy;
    assign busy = ag_busy;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            start_pulse  <= 1'b0;
            done_sticky  <= 1'b0;
            r_op         <= 32'd0; r_flags     <= 32'd0;
            r_c_in       <= 32'd0; r_c_out     <= 32'd0;
            r_k          <= 32'd1; r_stride    <= 32'd1;
            r_dilation   <= 32'd1; r_pad       <= 32'd0;
            r_t_out      <= 32'd0; r_t_tile    <= 32'd32;
            r_dw_ch_tile <= 32'd64;
            r_in_zp      <= 32'd0; r_out_zp    <= 32'd0; r_res_zp <= 32'd0;
            r_in_base    <= 32'd0; r_in_pitch  <= 32'd1; r_in_cbase  <= 32'd0;
            r_out_base   <= 32'd0; r_out_pitch <= 32'd1; r_out_cbase <= 32'd0;
            r_res_base   <= 32'd0; r_res_pitch <= 32'd1;
            r_w_off      <= 32'd0; r_bias_off  <= 32'd0;
            r_qmult_off  <= 32'd0; r_rshift_off<= 32'd0;
        end else begin
            start_pulse <= 1'b0;
            if (mmio_we) begin
                case (mmio_addr)
                8'd0:  if (mmio_wdata[0] && !ag_busy) start_pulse <= 1'b1;
                8'd1:  if (mmio_wdata[1]) done_sticky <= 1'b0;   /* W1C */
                8'd3:  r_op         <= mmio_wdata;
                8'd4:  r_flags      <= mmio_wdata;
                8'd5:  r_c_in       <= mmio_wdata;
                8'd6:  r_c_out      <= mmio_wdata;
                8'd7:  r_k          <= mmio_wdata;
                8'd8:  r_stride     <= mmio_wdata;
                8'd9:  r_dilation   <= mmio_wdata;
                8'd10: r_pad        <= mmio_wdata;
                8'd11: r_t_out      <= mmio_wdata;
                8'd12: r_t_tile     <= mmio_wdata;
                8'd13: r_dw_ch_tile <= mmio_wdata;
                8'd14: r_in_zp      <= mmio_wdata;
                8'd15: r_out_zp     <= mmio_wdata;
                8'd16: r_res_zp     <= mmio_wdata;
                8'd17: r_in_base    <= mmio_wdata;
                8'd18: r_in_pitch   <= mmio_wdata;
                8'd19: r_in_cbase   <= mmio_wdata;
                8'd20: r_out_base   <= mmio_wdata;
                8'd21: r_out_pitch  <= mmio_wdata;
                8'd22: r_out_cbase  <= mmio_wdata;
                8'd23: r_res_base   <= mmio_wdata;
                8'd24: r_res_pitch  <= mmio_wdata;
                8'd25: r_w_off      <= mmio_wdata;
                8'd26: r_bias_off   <= mmio_wdata;
                8'd27: r_qmult_off  <= mmio_wdata;
                8'd28: r_rshift_off <= mmio_wdata;
                default: ;
                endcase
            end
            if (done_pulse) done_sticky <= 1'b1;
        end
    end

    always @* begin
        case (mmio_addr)
        8'd1:  mmio_rdata = {30'd0, done_sticky, ag_busy};
        8'd2:  mmio_rdata = last_cycles;
        8'd3:  mmio_rdata = r_op;
        8'd4:  mmio_rdata = r_flags;
        8'd5:  mmio_rdata = r_c_in;
        8'd6:  mmio_rdata = r_c_out;
        8'd7:  mmio_rdata = r_k;
        default: mmio_rdata = 32'd0;
        endcase
    end

    wire [1:0] op = r_op[1:0];
    wire       relu = r_flags[0];

    /* The register file is a 32-bit-per-field image of the QN_DESC record, but
     * most fields are narrower than a word (a stride is 8 bits, a channel count
     * 16).  Tie the surplus bits off so lint stays clean without narrowing the
     * software-visible interface, which must keep matching the descriptor. */
    wire _unused_cfg = &{1'b0,
        r_stride[31:8], r_dilation[31:8], r_pad[31:16],
        r_t_out[31:16], r_t_tile[31:16], r_dw_ch_tile[31:16],
        r_in_zp[31:9], r_res_zp[31:9],
        r_in_pitch[31:16], r_in_cbase[31:16],
        r_out_pitch[31:16], r_out_cbase[31:16], r_res_pitch[31:16]};

    /* ── Address generator ──────────────────────────────────────────────── */
    wire        ag_chunk_done, ag_elem_done;
    wire [15:0] ag_t, ag_c, ag_r_len, ag_r_base;
    wire        ag_c_first, ag_last_chunk, ag_valid, ag_done;
    wire [31:0] ag_in_addr, ag_in_stride, ag_wt_addr, ag_out_addr, ag_res_addr;
    wire [LANES-1:0] ag_lane_en;
    wire [15:0] ag_t0, ag_t1, ag_c0, ag_c1, ag_in0, ag_in1, ag_new0, ag_new1;
    wire        ag_first_tile, ag_last_tile, ag_tile_start;

    addr_gen #(.LANES(LANES)) u_ag (
        .clk (clk), .rst_n (rst_n),
        .start (start_pulse),
        .chunk_done (ag_chunk_done),
        .elem_done  (ag_elem_done),
        .cfg_op         (op),
        .cfg_c_in       (r_c_in[15:0]),
        .cfg_c_out      (r_c_out[15:0]),
        .cfg_k          (r_k[15:0]),
        .cfg_stride     (r_stride[7:0]),
        .cfg_dilation   (r_dilation[7:0]),
        .cfg_pad        (r_pad[15:0]),
        .cfg_t_out      (r_t_out[15:0]),
        .cfg_t_tile     (r_t_tile[15:0]),
        .cfg_dw_ch_tile (r_dw_ch_tile[15:0]),
        .cfg_in_base    (r_in_base),
        .cfg_in_pitch   (r_in_pitch[15:0]),
        .cfg_in_cbase   (r_in_cbase[15:0]),
        .cfg_out_base   (r_out_base),
        .cfg_out_pitch  (r_out_pitch[15:0]),
        .cfg_out_cbase  (r_out_cbase[15:0]),
        .cfg_res_base   (r_res_base),
        .cfg_res_pitch  (r_res_pitch[15:0]),
        .cfg_w_off      (r_w_off),
        .o_t0 (ag_t0), .o_t1 (ag_t1), .o_c0 (ag_c0), .o_c1 (ag_c1),
        .o_in0 (ag_in0), .o_in1 (ag_in1), .o_new0 (ag_new0), .o_new1 (ag_new1),
        .o_first_tile (ag_first_tile), .o_last_tile (ag_last_tile),
        .o_tile_start (ag_tile_start),
        .o_t (ag_t), .o_c (ag_c), .o_c_first (ag_c_first),
        .o_r_len (ag_r_len), .o_r_base (ag_r_base), .o_last_chunk (ag_last_chunk),
        .o_in_addr (ag_in_addr), .o_in_stride (ag_in_stride),
        .o_lane_en (ag_lane_en), .o_wt_addr (ag_wt_addr),
        .o_out_addr (ag_out_addr), .o_res_addr (ag_res_addr),
        .o_valid (ag_valid), .busy (ag_busy), .done (ag_done)
    );

    /* Tile-level observability, exported on the top's ports. */
    assign o_tile_start = ag_tile_start;
    assign o_t0         = ag_t0;
    assign o_t1         = ag_t1;
    assign o_c0         = ag_c0;
    assign o_c1         = ag_c1;
    assign o_in0        = ag_in0;
    assign o_in1        = ag_in1;
    assign o_new0       = ag_new0;
    assign o_new1       = ag_new1;
    assign o_first_tile = ag_first_tile;
    assign o_last_tile  = ag_last_tile;

    wire _unused_tile = &{1'b0, ag_valid, ag_t, ag_r_len};

    /* ── Memory model ───────────────────────────────────────────────────── */
    reg         q_req_valid, q_req_word;
    reg  [31:0] q_req_addr;
    reg  [31:0] q_req_stride;
    reg  [LANES-1:0] q_req_mask;
    wire        q_req_ready, q_rsp_valid;
    wire [LANES*8-1:0] q_rsp_bytes;
    wire [31:0] q_rsp_word;

    reg         p_req_valid, p_req_we;
    reg  [31:0] p_req_addr, p_req_stride;
    reg  [LANES-1:0] p_req_mask;
    reg  [7:0]  p_req_wdata;
    wire        p_req_ready, p_rsp_valid;
    wire [LANES*8-1:0] p_rsp_bytes;

    ext_mem_if #(
        .LANES (LANES), .QSPI_BYTES (QSPI_BYTES), .PSRAM_BYTES (PSRAM_BYTES),
        .QSPI_LAT (QSPI_LAT), .PSRAM_LAT (PSRAM_LAT), .LAT_JITTER (LAT_JITTER)
    ) u_mem (
        .clk (clk), .rst_n (rst_n),
        .q_req_valid (q_req_valid), .q_req_ready (q_req_ready),
        .q_req_addr (q_req_addr), .q_req_stride (q_req_stride),
        .q_req_mask (q_req_mask), .q_req_word (q_req_word),
        .q_rsp_valid (q_rsp_valid), .q_rsp_bytes (q_rsp_bytes),
        .q_rsp_word (q_rsp_word),
        .p_req_valid (p_req_valid), .p_req_ready (p_req_ready),
        .p_req_we (p_req_we), .p_req_addr (p_req_addr),
        .p_req_stride (p_req_stride), .p_req_mask (p_req_mask),
        .p_req_wdata (p_req_wdata),
        .p_rsp_valid (p_rsp_valid), .p_rsp_bytes (p_rsp_bytes),
        .bd_en (bd_en), .bd_sel (bd_sel), .bd_we (bd_we),
        .bd_addr (bd_addr), .bd_wdata (bd_wdata), .bd_rdata (bd_rdata)
    );

    /* ── Operand registers ──────────────────────────────────────────────── */
    reg [LANES*8-1:0] in_chunk, wt_chunk;
    reg [LANES-1:0]   lane_en_q;
    reg signed [31:0] acc;
    reg signed [31:0] bias_q, qmult_q, rshift_q;
    /* ADD per-tensor qparams, fetched once per descriptor. */
    reg signed [31:0] qm0_q, qm1_q, qm2_q, rs0_q, rs1_q, rs2_q;
    reg signed [31:0] add_mv, add_rv;

    reg signed [8:0]  in_zp_q;
    reg signed [8:0]  res_zp_q;
    reg signed [31:0] out_zp_q;
    reg               relu_q;

    /* ── MAC array (unchanged module) ───────────────────────────────────── */
    wire signed [31:0] psum;
    int8_mac_array #(.LANES(LANES)) u_mac (
        .in_bytes (in_chunk),
        .wt_bytes (wt_chunk),
        .in_zp    (in_zp_q),
        .lane_en  (lane_en_q),
        .psum     (psum)
    );

    /* Accumulator saturation, structurally identical to tinymac_accel.v.  At
     * the mandatory ACC_W=32 this is a pass-through. */
    wire signed [31:0] acc_sum = acc + psum;
    reg  signed [31:0] acc_sat;
    localparam signed [31:0] ACC_HI = (32'sd1 <<< (ACC_W - 1)) - 32'sd1;
    localparam signed [31:0] ACC_LO = -(32'sd1 <<< (ACC_W - 1));
    always @* begin
        if (ACC_W >= 32)           acc_sat = acc_sum;
        else if (acc_sum > ACC_HI) acc_sat = ACC_HI;
        else if (acc_sum < ACC_LO) acc_sat = ACC_LO;
        else                       acc_sat = acc_sum;
    end

    /* ── Requantize datapaths ───────────────────────────────────────────── */
    reg                rq_launch;
    wire               rq_valid;
    wire signed [7:0]  rq_out;
    wire signed [31:0] rq_raw_unused;

    requantize u_rq (
        .clk (clk), .rst_n (rst_n),
        .in_valid (rq_launch),
        .acc      (acc),
        .q_mult   (qmult_q),
        .shift    (rshift_q),
        .out_zp   (out_zp_q),
        .relu     (relu_q),
        .out_valid(rq_valid),
        .out_q    (rq_out),
        .out_raw  (rq_raw_unused)
    );

    reg                rqa_launch;
    wire               rqa_valid;
    wire signed [7:0]  rqa_out;

    requantize_add u_rqa (
        .clk (clk), .rst_n (rst_n),
        .in_valid (rqa_launch),
        .mv (add_mv), .rv (add_rv),
        .qmult0 (qm0_q), .rshift0 (rs0_q),
        .qmult1 (qm1_q), .rshift1 (rs1_q),
        .qmult2 (qm2_q), .rshift2 (rs2_q),
        .out_zp (out_zp_q), .relu (relu_q),
        .out_valid (rqa_valid), .out_q (rqa_out)
    );

    wire _unused_raw = &{1'b0, rq_raw_unused};

    /* ── Sequencer ──────────────────────────────────────────────────────── */
    localparam [4:0]
        S_IDLE     = 5'd0,
        S_ADDQ     = 5'd1,   /* fetch the 6 per-tensor ADD qparams        */
        S_ADDQ_W   = 5'd2,
        S_ELEM     = 5'd3,   /* start an element                          */
        S_QP       = 5'd4,   /* fetch bias / qmult / rshift for a channel */
        S_QP_W     = 5'd5,
        S_CHUNK    = 5'd6,   /* issue operand gathers                     */
        S_CHUNK_W  = 5'd7,   /* await both responses                      */
        S_ACC      = 5'd8,   /* accumulate                                */
        S_ADD_M    = 5'd9,   /* ADD: read main operand byte               */
        S_ADD_M_W  = 5'd10,
        S_ADD_R    = 5'd11,  /* ADD: read residual operand byte           */
        S_ADD_R_W  = 5'd12,
        S_RQ       = 5'd13,  /* launch requantize                         */
        S_RQ_W     = 5'd14,
        S_ST       = 5'd15,  /* store the output byte                     */
        S_ST_W     = 5'd16,
        S_FIN      = 5'd17;

    reg [4:0] state;
    reg [2:0] qp_idx;        /* which qparam word is in flight */
    reg       in_got, wt_got;
    reg signed [7:0] out_byte;

    assign ag_chunk_done = (state == S_ACC);
    assign ag_elem_done  = (state == S_ST_W) && p_rsp_valid;
    assign done_pulse    = (state == S_FIN);

    /* qparam word address for the current fetch index. */
    reg [31:0] qp_addr;
    always @* begin
        case (qp_idx)
        3'd0:    qp_addr = r_bias_off   + {14'd0, ag_c, 2'd0};
        3'd1:    qp_addr = r_qmult_off  + {14'd0, ag_c, 2'd0};
        3'd2:    qp_addr = r_rshift_off + {14'd0, ag_c, 2'd0};
        3'd3:    qp_addr = r_qmult_off;              /* ADD: qmult[0]  */
        3'd4:    qp_addr = r_qmult_off  + 32'd4;     /* ADD: qmult[1]  */
        3'd5:    qp_addr = r_qmult_off  + 32'd8;     /* ADD: qmult[2]  */
        3'd6:    qp_addr = r_rshift_off;             /* ADD: rshift[0] */
        default: qp_addr = r_rshift_off + 32'd4;     /* ADD: rshift[1] */
        endcase
    end
    /* rshift[2] needs a 9th slot; handled with a separate flag below. */
    reg qp_last_add;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state       <= S_IDLE;
            q_req_valid <= 1'b0; q_req_word <= 1'b0;
            q_req_addr  <= 32'd0; q_req_stride <= 32'd1;
            q_req_mask  <= {LANES{1'b0}};
            p_req_valid <= 1'b0; p_req_we <= 1'b0;
            p_req_addr  <= 32'd0; p_req_stride <= 32'd1;
            p_req_mask  <= {LANES{1'b0}}; p_req_wdata <= 8'd0;
            rq_launch   <= 1'b0; rqa_launch <= 1'b0;
            acc         <= 32'sd0;
            in_got      <= 1'b0; wt_got <= 1'b0;
            qp_idx      <= 3'd0; qp_last_add <= 1'b0;
            in_chunk    <= {LANES*8{1'b0}};
            wt_chunk    <= {LANES*8{1'b0}};
            lane_en_q   <= {LANES{1'b0}};
            bias_q      <= 32'sd0; qmult_q <= 32'sd0; rshift_q <= 32'sd0;
            qm0_q <= 32'sd0; qm1_q <= 32'sd0; qm2_q <= 32'sd0;
            rs0_q <= 32'sd0; rs1_q <= 32'sd0; rs2_q <= 32'sd0;
            add_mv <= 32'sd0; add_rv <= 32'sd0;
            in_zp_q <= 9'sd0; res_zp_q <= 9'sd0; out_zp_q <= 32'sd0;
            relu_q  <= 1'b0;
            out_byte <= 8'sd0;
            cyc_cnt <= 32'd0; last_cycles <= 32'd0;
        end else begin
            rq_launch  <= 1'b0;
            rqa_launch <= 1'b0;

            /* Deassert a request once it has been accepted. */
            if (q_req_valid && q_req_ready) q_req_valid <= 1'b0;
            if (p_req_valid && p_req_ready) p_req_valid <= 1'b0;

            if (state != S_IDLE) cyc_cnt <= cyc_cnt + 32'd1;

            case (state)
            /* ------------------------------------------------------------ */
            S_IDLE: begin
                if (start_pulse) begin
                    /* Latch whole-op configuration.  out_zp / relu must stay
                     * stable for the entire op — the requantize blocks do not
                     * pipeline them (see requantize.v's interface contract). */
                    in_zp_q  <= r_in_zp[8:0];
                    res_zp_q <= r_res_zp[8:0];
                    out_zp_q <= r_out_zp;
                    relu_q   <= relu;
                    cyc_cnt  <= 32'd0;
                    qp_idx   <= (op == OP_ADD) ? 3'd3 : 3'd0;
                    qp_last_add <= 1'b0;
                    state    <= (op == OP_ADD) ? S_ADDQ : S_ELEM;
                end
            end
            /* ---- ADD: six per-tensor qparams, fetched once ------------- */
            S_ADDQ: begin
                if (!q_req_valid) begin
                    q_req_valid  <= 1'b1;
                    q_req_word   <= 1'b1;
                    q_req_addr   <= qp_last_add ? (r_rshift_off + 32'd8) : qp_addr;
                    q_req_stride <= 32'd1;
                    q_req_mask   <= {LANES{1'b0}};
                    state        <= S_ADDQ_W;
                end
            end
            S_ADDQ_W: begin
                if (q_rsp_valid) begin
                    /* qp_idx only counts 3..7; the seventh and final word
                     * (rshift[2]) is flagged by qp_last_add instead, so it must
                     * be decoded BEFORE the qp_idx case — otherwise qp_idx is
                     * still 7 and rshift[1] would be reloaded over it. */
                    if (qp_last_add) begin
                        rs2_q <= q_rsp_word;
                    end else begin
                        case (qp_idx)
                        3'd3:    qm0_q <= q_rsp_word;
                        3'd4:    qm1_q <= q_rsp_word;
                        3'd5:    qm2_q <= q_rsp_word;
                        3'd6:    rs0_q <= q_rsp_word;
                        default: rs1_q <= q_rsp_word;   /* qp_idx == 7 */
                        endcase
                    end
                    if (qp_last_add) begin
                        state <= S_ELEM;
                    end else if (qp_idx == 3'd7) begin
                        qp_last_add <= 1'b1;
                        state       <= S_ADDQ;
                    end else begin
                        qp_idx <= qp_idx + 3'd1;
                        state  <= S_ADDQ;
                    end
                end
            end
            /* ---- Element start ----------------------------------------- */
            S_ELEM: begin
                if (!ag_busy) begin
                    state <= S_FIN;
                end else if (op == OP_ADD) begin
                    state <= S_ADD_M;
                end else if (ag_c_first) begin
                    /* New output channel: refresh its qparam triple. */
                    qp_idx <= 3'd0;
                    state  <= S_QP;
                end else begin
                    acc    <= bias_q;
                    in_got <= 1'b0;
                    wt_got <= 1'b0;
                    state  <= S_CHUNK;
                end
            end
            /* ---- Per-channel bias / qmult / rshift --------------------- */
            S_QP: begin
                if (!q_req_valid) begin
                    q_req_valid  <= 1'b1;
                    q_req_word   <= 1'b1;
                    q_req_addr   <= qp_addr;
                    q_req_stride <= 32'd1;
                    q_req_mask   <= {LANES{1'b0}};
                    state        <= S_QP_W;
                end
            end
            S_QP_W: begin
                if (q_rsp_valid) begin
                    case (qp_idx)
                    3'd0:    bias_q   <= q_rsp_word;
                    3'd1:    qmult_q  <= q_rsp_word;
                    default: rshift_q <= q_rsp_word;
                    endcase
                    if (qp_idx == 3'd2) begin
                        in_got <= 1'b0;
                        wt_got <= 1'b0;
                        state  <= S_CHUNK;
                    end else begin
                        qp_idx <= qp_idx + 3'd1;
                        state  <= S_QP;
                    end
                end
            end
            /* ---- Reduction chunk --------------------------------------- */
            S_CHUNK: begin
                /* Both request ports are guaranteed idle on entry to this
                 * state: it is reached from S_ELEM / S_QP_W / S_ACC, each of
                 * which has already seen its outstanding request accepted (and
                 * S_ACC additionally waited for both responses).  So the
                 * requests below are issued unconditionally. */

                /* Seed the accumulator with bias on the first chunk. */
                if (ag_r_base == 16'd0) acc <= bias_q;
                lane_en_q <= ag_lane_en;

                /* Activation gather on PSRAM. */
                p_req_valid  <= 1'b1;
                p_req_we     <= 1'b0;
                p_req_addr   <= ag_in_addr;
                p_req_stride <= ag_in_stride;
                p_req_mask   <= ag_lane_en;

                /* Weight gather on QSPI.  OP_REQUANT owns no weights — its
                 * "weight" is a synthetic 1, which makes the MAC array compute
                 * exactly the C's bare (x - in_zp). */
                if (op == OP_REQ) begin
                    wt_chunk <= {{(LANES*8-8){1'b0}}, 8'd1};
                    wt_got   <= 1'b1;
                end else begin
                    q_req_valid  <= 1'b1;
                    q_req_word   <= 1'b0;
                    q_req_addr   <= ag_wt_addr;
                    q_req_stride <= 32'd1;
                    q_req_mask   <= ag_lane_en;
                end
                state <= S_CHUNK_W;
            end
            S_CHUNK_W: begin
                if (p_rsp_valid) begin
                    in_chunk <= p_rsp_bytes;
                    in_got   <= 1'b1;
                end
                if (q_rsp_valid) begin
                    wt_chunk <= q_rsp_bytes;
                    wt_got   <= 1'b1;
                end
                if ((in_got || p_rsp_valid) && (wt_got || q_rsp_valid))
                    state <= S_ACC;
            end
            S_ACC: begin
                /* in_chunk/wt_chunk/lane_en_q are settled, so psum is valid. */
                acc    <= acc_sat;
                in_got <= 1'b0;
                wt_got <= 1'b0;
                if (ag_last_chunk) state <= S_RQ;
                else               state <= S_CHUNK;
            end
            /* ---- ADD operands ------------------------------------------ */
            S_ADD_M: begin
                if (!p_req_valid) begin
                    p_req_valid  <= 1'b1;
                    p_req_we     <= 1'b0;
                    p_req_addr   <= ag_in_addr;
                    p_req_stride <= 32'd1;
                    p_req_mask   <= {{(LANES-1){1'b0}}, 1'b1};
                    state        <= S_ADD_M_W;
                end
            end
            S_ADD_M_W: begin
                if (p_rsp_valid) begin
                    /* (main - in_zp), sign-extended from int8. */
                    add_mv <= {{24{p_rsp_bytes[7]}}, p_rsp_bytes[7:0]}
                            - {{23{in_zp_q[8]}}, in_zp_q};
                    state  <= S_ADD_R;
                end
            end
            S_ADD_R: begin
                if (!p_req_valid) begin
                    p_req_valid  <= 1'b1;
                    p_req_we     <= 1'b0;
                    p_req_addr   <= ag_res_addr;
                    p_req_stride <= 32'd1;
                    p_req_mask   <= {{(LANES-1){1'b0}}, 1'b1};
                    state        <= S_ADD_R_W;
                end
            end
            S_ADD_R_W: begin
                if (p_rsp_valid) begin
                    add_rv <= {{24{p_rsp_bytes[7]}}, p_rsp_bytes[7:0]}
                            - {{23{res_zp_q[8]}}, res_zp_q};
                    rqa_launch <= 1'b1;
                    state      <= S_RQ_W;
                end
            end
            /* ---- Requantize -------------------------------------------- */
            S_RQ: begin
                rq_launch <= 1'b1;
                state     <= S_RQ_W;
            end
            S_RQ_W: begin
                if (op == OP_ADD) begin
                    if (rqa_valid) begin
                        out_byte <= rqa_out;
                        state    <= S_ST;
                    end
                end else if (rq_valid) begin
                    out_byte <= rq_out;
                    state    <= S_ST;
                end
            end
            /* ---- Store ------------------------------------------------- */
            S_ST: begin
                if (!p_req_valid) begin
                    p_req_valid  <= 1'b1;
                    p_req_we     <= 1'b1;
                    p_req_addr   <= ag_out_addr;
                    p_req_stride <= 32'd1;
                    p_req_mask   <= {LANES{1'b0}};
                    p_req_wdata  <= out_byte;
                    state        <= S_ST_W;
                end
            end
            S_ST_W: begin
                /* ag_elem_done fires on this condition, advancing addr_gen. */
                if (p_rsp_valid) state <= S_ELEM;
            end
            /* ---- Finish ------------------------------------------------ */
            S_FIN: begin
                last_cycles <= cyc_cnt;
                state       <= S_IDLE;
            end
            default: state <= S_IDLE;
            endcase
        end
    end

    wire _unused_agdone = &{1'b0, ag_done};

endmodule

`default_nettype wire
