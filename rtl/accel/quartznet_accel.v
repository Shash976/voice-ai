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
 * ── Increment 2: autonomous table walking ───────────────────────────────────
 *
 * CTRL bit1 (RUN_TABLE) starts an autonomous walk of the in-ROM descriptor
 * table at TABLE_BASE: fetch the 64B header, fetch the n_buffers x 8B
 * buffer-record table, fetch descriptor 0 (64B / 16 words), populate
 * registers 3-28 from it — deriving the five buffer-resolved fields (BASE x3,
 * PITCH x3, the four blob offsets) on-chip instead of via software — then
 * execute it through the SAME per-op states increment 1 already verified,
 * and repeat for descriptors 1..n_desc-1.  Nothing about int8_mac_array.v,
 * requantize.v/requantize_add.v, or addr_gen.v's per-op execution changes;
 * this increment only adds the fetch/dispatch loop around it.
 *
 * STATUS bit2 (TABLE_DONE) fires once, after the last descriptor, instead of
 * per-descriptor DONE (bit1), which still fires every descriptor, unchanged,
 * for observability. New registers:
 *
 *  idx  name          acc  notes
 *  29   TABLE_BASE     RW  QSPI byte address of the table image
 *  30   W_BLOB_BASE    RW  added to each descriptor's blob-relative W_OFF
 *  31   QP_BLOB_BASE   RW  added to each descriptor's blob-relative
 *                          BIAS_OFF/QMULT_OFF/RSHIFT_OFF
 *  32   DESC_IDX       R   current descriptor index (debug only)
 *
 * CTRL bit1 RUN_TABLE (self-clearing, ignored while busy — "busy" is now
 * `state != S_IDLE`, not just the address generator's own busy flag, since a
 * table walk is busy between descriptors too, while the next one is being
 * fetched).  STATUS bit3 is a sticky err_bad_in_off flag (read only, cleared
 * automatically at the next RUN_TABLE trigger): IN_CBASE still has no divider
 * (same reason as increment 1), and autonomous mode additionally REQUIRES
 * in_off == 0 for every descriptor it fetches — true of every table
 * quartznet_descriptors.py emits today (in_off is hardcoded 0, see its
 * pack_record()) — so a fetched descriptor with a nonzero in_off latches the
 * flag rather than silently computing a wrong address.
 *
 * ── Increment (Stage 7 Gap 1 D4): MEM_* host-access port + SINGLE_STEP ───────
 *
 * The simulation-only bd_* backdoor (used by rtl/tb) has no analogue on a real
 * bus: something must let firmware stage the mel input into PSRAM and read
 * logits back without a testbench. New registers replace it for that purpose,
 * while bd_* itself is UNCHANGED and still wins whenever the top-level bd_en
 * INPUT is asserted (so rtl/tb's existing 27/27 path is unaffected bit for
 * bit) — a real SoC ties bd_en=0, so the MEM_* port always drives ext_mem_if
 * there.
 *
 *  idx  name       acc  notes
 *  33   MEM_ADDR    RW  byte address within the selected bank
 *  34   MEM_CTRL    RW  bit0 SEL (0=QSPI,1=PSRAM), bit1 AUTOINC (write only)
 *  35   MEM_DATA    RW  single byte at MEM_ADDR (bits [7:0]; also latches the
 *                       MEM_FILL fill byte as a side effect of every write).
 *                       AUTOINC advances MEM_ADDR by 1 after a WRITE only —
 *                       this bus has no read-strobe/ready signal at all (every
 *                       register read is a stateless combinational lookup, as
 *                       STATUS/CYCLES/DESC_IDX already are), so there is no
 *                       edge to hang a read-side side-effect off of. Bulk
 *                       reads (qn_gather_output_hw/qn_read_logits_hw) restage
 *                       MEM_ADDR before every byte instead.
 *  36   MEM_FILL    W   write N: fill N bytes from MEM_ADDR with MEM_DATA's
 *                       low byte, advancing MEM_ADDR. Genuinely multi-cycle
 *                       (one bd_* write per cycle) — unlike every other
 *                       register here, this is NOT complete when the mmio_we
 *                       cycle ends: it asserts BUSY (state S_MEMFILL) until
 *                       done, polled the same way a compute op's STATUS.BUSY
 *                       already is. Ignored (N=0 no-op) while busy.
 *
 * A MEM_DATA write's bd_* request to ext_mem_if is REGISTERED one cycle
 * (r_mem_we_q/r_mem_sel_q/r_mem_addr_q/r_mem_wdata_q), not driven
 * combinationally in the same cycle as the mmio_we that triggered it — the
 * next MMIO transaction (staging MEM_ADDR for a readback, or the next
 * MEM_DATA write) already costs its own cycle, so this adds no visible
 * latency to any of the sequences above.
 *
 * CTRL bit3 (SINGLE_STEP, sampled only on the same RUN_TABLE-triggering write)
 * makes the table walker halt after each descriptor (state S_TWAIT, busy)
 * until firmware W1C-clears STATUS.DONE — the ack that "I have safely read
 * this descriptor's output before a later one's ping-pong reuse overwrites
 * it" (see quartznet_tb.cpp's own DESC_IDX-snapshot comment, and
 * firmware/quartznet/qn_accel.c's qn_run_hw_step()). Without this, CPU
 * polling (far slower than rtl/tb's per-cycle poll) cannot reliably catch
 * every intermediate descriptor before it is clobbered.
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
    localparam integer N_BUF_MAX = 8;   /* real tables use 5 (QN_N_BUFFERS) */

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

    /* ── Increment 2: table-walker registers ──────────────────────────────
     * TABLE_BASE/W_BLOB_BASE/QP_BLOB_BASE are software-staged once per table
     * walk (idx 29-31); everything else here is internal FSM state, not
     * software-visible except DESC_IDX (idx 32, read-only debug). */
    reg [31:0] r_table_base, r_w_blob_base, r_qp_blob_base;
    reg        run_table_pulse;
    reg        table_mode;
    reg        table_done_sticky;
    reg        err_bad_in_off;
    reg        tbl_start_pulse;      /* internal addr_gen kick, one per desc */
    reg        r_single_step;        /* latched from CTRL[3] at RUN_TABLE time */
    reg [31:0] desc_idx;

    /* ── D4: MEM_* host-access port state ──────────────────────────────── */
    reg [31:0] r_mem_addr;
    reg [31:0] r_mem_ctrl;           /* bit0 SEL, bit1 AUTOINC */
    reg [7:0]  r_mem_fill_byte;
    reg [31:0] r_fill_remaining;
    reg        fill_pulse;
    /* Registered one-cycle-delayed snapshot of a pending bd_* request --
     * see the "D4: bd_* mux" comment below for why this is registered
     * rather than combinational. */
    reg        r_mem_we_q, r_mem_sel_q;
    reg [31:0] r_mem_addr_q;
    reg [7:0]  r_mem_wdata_q;
    reg [31:0] hdr_n_desc, hdr_desc_off, hdr_n_buffers;
    reg [3:0]  word_i;               /* 0..15: header/descriptor word fetch */
    reg [3:0]  buf_i;                /* 0..N_BUF_MAX-1: buffer-record fetch */
    reg        buftbl_phase;         /* 0 = channels word, 1 = rate word    */
    reg [31:0] buf_cum;              /* running arena-offset accumulator    */
    reg [31:0] buf_ch_r   [0:N_BUF_MAX-1];   /* re-read per descriptor (PITCH) */
    reg [31:0] buf_off_r  [0:N_BUF_MAX-1];   /* re-read per descriptor (BASE)  */
    /* buf_rate is NOT kept resident: it only feeds the one-time buf_cum
     * running-offset update below, straight from q_rsp_word. */
    reg [31:0] desc_word_r[0:15];    /* the 16 words of the fetched record  */

    wire ag_busy;       /* addr_gen's own busy (this descriptor's walk)     */
    wire accel_busy;    /* whole-sequencer busy (spans an entire table walk) */
    assign busy = accel_busy;

    always @* begin
        case (mmio_addr)
        8'd1:  mmio_rdata = {28'd0, err_bad_in_off, table_done_sticky,
                              done_sticky, accel_busy};
        8'd2:  mmio_rdata = last_cycles;
        8'd3:  mmio_rdata = r_op;
        8'd4:  mmio_rdata = r_flags;
        8'd5:  mmio_rdata = r_c_in;
        8'd6:  mmio_rdata = r_c_out;
        8'd7:  mmio_rdata = r_k;
        8'd32: mmio_rdata = desc_idx;
        8'd33: mmio_rdata = r_mem_addr;
        8'd34: mmio_rdata = r_mem_ctrl;
        8'd35: mmio_rdata = {24'd0, bd_rdata};
        8'd36: mmio_rdata = r_fill_remaining;
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

    /* addr_gen's start is kicked either by the legacy CTRL.START pulse or, in
     * table-walk mode, by tbl_start_pulse once per fetched descriptor -- the
     * two are mutually exclusive in time (single-descriptor mode never
     * enters the table-fetch states that produce tbl_start_pulse). */
    wire ag_start = start_pulse | tbl_start_pulse;

    addr_gen #(.LANES(LANES)) u_ag (
        .clk (clk), .rst_n (rst_n),
        .start (ag_start),
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

    /* ── D4: bd_* mux ──────────────────────────────────────────────────────
     * The top-level bd_en INPUT (driven only by rtl/tb) wins whenever it is
     * asserted, leaving that regression path byte-for-byte unchanged. A real
     * SoC ties bd_en=0, so the internal MEM_* port (below) always drives
     * ext_mem_if there.
     *
     * The internal side is REGISTERED one cycle (r_mem_we_q/r_mem_sel_q/
     * r_mem_addr_q/r_mem_wdata_q below), not driven combinationally straight
     * from mmio_we/mmio_addr -- every other cross-module request in this file
     * (q_req_valid, p_req_valid) is a register for the same reason: it keeps
     * the address/data pair that reaches ext_mem_if a stable, already-settled
     * snapshot instead of a same-cycle combinational path that also happens
     * to depend on a register (r_mem_addr) this same write can update.
     *
     * Reads have no such hazard (no write, nothing else touches ext_mem_if's
     * state) and must see the CURRENT address the CPU just staged, not a
     * stale snapshot from whatever the last write happened to be -- so sel/
     * addr fall back to the live r_mem_ctrl[0]/r_mem_addr whenever a
     * registered write isn't in flight this exact cycle (r_mem_we_q == 0). */
    wire        u_mem_bd_en    = bd_en ? bd_en    : 1'b1;
    wire        u_mem_bd_sel   = bd_en ? bd_sel   : (r_mem_we_q ? r_mem_sel_q  : r_mem_ctrl[0]);
    wire        u_mem_bd_we    = bd_en ? bd_we    : r_mem_we_q;
    wire [31:0] u_mem_bd_addr  = bd_en ? bd_addr  : (r_mem_we_q ? r_mem_addr_q : r_mem_addr);
    wire [7:0]  u_mem_bd_wdata = bd_en ? bd_wdata : r_mem_wdata_q;

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
        .bd_en (u_mem_bd_en), .bd_sel (u_mem_bd_sel), .bd_we (u_mem_bd_we),
        .bd_addr (u_mem_bd_addr), .bd_wdata (u_mem_bd_wdata), .bd_rdata (bd_rdata)
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
        S_FIN      = 5'd17,
        /* ── Increment 2: table-walk fetch states ─────────────────────── */
        S_HDR      = 5'd18,  /* fetch the 64B/16-word table header        */
        S_HDR_W    = 5'd19,
        S_BUFTBL   = 5'd20,  /* fetch n_buffers x 8B buffer records       */
        S_BUFTBL_W = 5'd21,
        S_DESC     = 5'd22,  /* fetch descriptor desc_idx (64B/16 words)  */
        S_DESC_W   = 5'd23,
        S_POPULATE = 5'd24,  /* decode desc_word_r[] into regs 3-28       */
        S_KICK     = 5'd25,  /* one-cycle handoff to addr_gen (see below) */
        /* ── D4: MEM_FILL / SINGLE_STEP ──────────────────────────────── */
        S_MEMFILL  = 5'd26,  /* multi-cycle background bd_* fill loop     */
        S_TWAIT    = 5'd27;  /* single-step: hold until STATUS.DONE clears */

    reg [4:0] state;
    reg [2:0] qp_idx;        /* which qparam word is in flight */
    reg       in_got, wt_got;

    /* "Busy" for MMIO gating covers the whole table walk, not just whatever
     * addr_gen happens to be doing right now -- ag_busy alone drops between
     * descriptors while the next one is being fetched from ROM. */
    assign accel_busy = (state != S_IDLE);
    reg signed [7:0] out_byte;

    assign ag_chunk_done = (state == S_ACC);
    assign ag_elem_done  = (state == S_ST_W) && p_rsp_valid;
    assign done_pulse    = (state == S_FIN);

    /* Fires once, on the S_FIN that completes the LAST descriptor of a table
     * walk -- table_done_sticky (owned by the MMIO always block above, same
     * pattern as done_sticky/done_pulse) latches it. */
    wire table_done_pulse = (state == S_FIN) && table_mode
                          && ((desc_idx + 32'd1) >= hdr_n_desc);

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
            /* ── Register file + status (formerly a separate always block;
             * merged here because S_POPULATE below writes the same r_*
             * config registers software's mmio_we path writes, and Verilog
             * disallows two always blocks driving one reg) ──────────────── */
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
            r_table_base <= 32'd0; r_w_blob_base <= 32'd0; r_qp_blob_base <= 32'd0;
            run_table_pulse <= 1'b0; table_done_sticky <= 1'b0;
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
            /* ── Increment 2: table-walk FSM state (owned by this block) ── */
            table_mode      <= 1'b0;
            err_bad_in_off  <= 1'b0;
            tbl_start_pulse <= 1'b0;
            r_single_step   <= 1'b0;
            desc_idx        <= 32'd0;
            hdr_n_desc      <= 32'd0; hdr_desc_off <= 32'd0; hdr_n_buffers <= 32'd0;
            word_i          <= 4'd0;  buf_i        <= 4'd0;  buftbl_phase  <= 1'b0;
            buf_cum         <= 32'd0;
            /* ── D4: MEM_* port + fill state ──────────────────────────────── */
            r_mem_addr       <= 32'd0;
            r_mem_ctrl       <= 32'd0;
            r_mem_fill_byte  <= 8'd0;
            r_fill_remaining <= 32'd0;
            fill_pulse       <= 1'b0;
            r_mem_we_q       <= 1'b0;
            r_mem_sel_q      <= 1'b0;
            r_mem_addr_q     <= 32'd0;
            r_mem_wdata_q    <= 8'd0;
        end else begin
            start_pulse     <= 1'b0;
            run_table_pulse <= 1'b0;
            rq_launch       <= 1'b0;
            rqa_launch      <= 1'b0;
            tbl_start_pulse <= 1'b0;
            r_mem_we_q      <= 1'b0;
            fill_pulse      <= 1'b0;

            /* Software register writes.  Textually before the case(state)
             * block below, so if S_POPULATE and a same-cycle software write
             * ever targeted the same register (not a real scenario: software
             * does not poke config registers mid-table-walk), S_POPULATE's
             * NBA -- later in program order -- would be the one that takes
             * effect. */
            if (mmio_we) begin
                case (mmio_addr)
                8'd0:  begin
                    if (mmio_wdata[0] && !accel_busy) start_pulse     <= 1'b1;
                    if (mmio_wdata[1] && !accel_busy) begin
                        run_table_pulse <= 1'b1;
                        r_single_step   <= mmio_wdata[3];
                    end
                end
                8'd1:  begin
                    if (mmio_wdata[1]) done_sticky       <= 1'b0;   /* W1C */
                    if (mmio_wdata[2]) table_done_sticky <= 1'b0;   /* W1C */
                end
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
                8'd29: r_table_base   <= mmio_wdata;
                8'd30: r_w_blob_base  <= mmio_wdata;
                8'd31: r_qp_blob_base <= mmio_wdata;
                8'd33: r_mem_addr     <= mmio_wdata;
                8'd34: r_mem_ctrl     <= mmio_wdata;
                8'd35: begin
                    r_mem_fill_byte <= mmio_wdata[7:0];
                    r_mem_we_q      <= 1'b1;
                    r_mem_sel_q     <= r_mem_ctrl[0];
                    r_mem_addr_q    <= r_mem_addr;      /* pre-increment address */
                    r_mem_wdata_q   <= mmio_wdata[7:0];
                    if (r_mem_ctrl[1]) r_mem_addr <= r_mem_addr + 32'd1;   /* AUTOINC, write only */
                end
                8'd36: if (mmio_wdata != 32'd0 && !accel_busy) begin
                    r_fill_remaining <= mmio_wdata;
                    fill_pulse       <= 1'b1;
                end
                default: ;
                endcase
            end
            if (done_pulse)       done_sticky       <= 1'b1;
            if (table_done_pulse) table_done_sticky <= 1'b1;

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
                end else if (run_table_pulse) begin
                    desc_idx       <= 32'd0;
                    word_i         <= 4'd0;
                    table_mode     <= 1'b1;
                    err_bad_in_off <= 1'b0;
                    cyc_cnt        <= 32'd0;
                    state          <= S_HDR;
                end else if (fill_pulse) begin
                    state <= S_MEMFILL;
                end
            end
            /* ---- Increment 2: table header (16 words) ------------------ */
            S_HDR: begin
                if (!q_req_valid) begin
                    q_req_valid  <= 1'b1;
                    q_req_word   <= 1'b1;
                    q_req_addr   <= r_table_base + {26'd0, word_i, 2'd0};
                    q_req_stride <= 32'd1;
                    q_req_mask   <= {LANES{1'b0}};
                    state        <= S_HDR_W;
                end
            end
            S_HDR_W: begin
                if (q_rsp_valid) begin
                    case (word_i)
                    4'd2:    hdr_n_desc    <= q_rsp_word;   /* header word 2  */
                    4'd10:   r_t_tile      <= q_rsp_word;   /* header word 10 */
                    4'd12:   r_dw_ch_tile  <= q_rsp_word;   /* header word 12 */
                    4'd14:   hdr_n_buffers <= q_rsp_word;   /* header word 14 */
                    4'd15:   hdr_desc_off  <= q_rsp_word;   /* header word 15 */
                    default: ;
                    endcase
                    if (word_i == 4'd15) begin
                        buf_i        <= 4'd0;
                        buftbl_phase <= 1'b0;
                        buf_cum      <= 32'd0;
                        state        <= S_BUFTBL;
                    end else begin
                        word_i <= word_i + 4'd1;
                        state  <= S_HDR;
                    end
                end
            end
            /* ---- Increment 2: buffer-record table (n_buffers x 8B) ------ */
            S_BUFTBL: begin
                if (!q_req_valid) begin
                    q_req_valid  <= 1'b1;
                    q_req_word   <= 1'b1;
                    q_req_addr   <= r_table_base + 32'd64
                                  + {25'd0, buf_i[2:0], 3'd0}
                                  + (buftbl_phase ? 32'd4 : 32'd0);
                    q_req_stride <= 32'd1;
                    q_req_mask   <= {LANES{1'b0}};
                    state        <= S_BUFTBL_W;
                end
            end
            S_BUFTBL_W: begin
                if (q_rsp_valid) begin
                    if (!buftbl_phase) begin
                        buf_ch_r[buf_i[2:0]] <= q_rsp_word;
                        buftbl_phase         <= 1'b1;
                        state                <= S_BUFTBL;
                    end else begin
                        buf_off_r[buf_i[2:0]] <= buf_cum;
                        buf_cum <= buf_cum
                                 + buf_ch_r[buf_i[2:0]] * q_rsp_word * r_t_out;
                        buftbl_phase <= 1'b0;
                        if ({28'd0, buf_i} == hdr_n_buffers - 32'd1) begin
                            word_i <= 4'd0;
                            state  <= S_DESC;
                        end else begin
                            buf_i <= buf_i + 4'd1;
                            state <= S_BUFTBL;
                        end
                    end
                end
            end
            /* ---- Increment 2: one descriptor (16 words) ----------------- */
            S_DESC: begin
                if (!q_req_valid) begin
                    q_req_valid  <= 1'b1;
                    q_req_word   <= 1'b1;
                    q_req_addr   <= r_table_base + hdr_desc_off + (desc_idx << 6)
                                  + {26'd0, word_i, 2'd0};
                    q_req_stride <= 32'd1;
                    q_req_mask   <= {LANES{1'b0}};
                    state        <= S_DESC_W;
                end
            end
            S_DESC_W: begin
                if (q_rsp_valid) begin
                    desc_word_r[word_i] <= q_rsp_word;
                    if (word_i == 4'd15) begin
                        state <= S_POPULATE;
                    end else begin
                        word_i <= word_i + 4'd1;
                        state  <= S_DESC;
                    end
                end
            end
            /* ---- Increment 2: decode the fetched descriptor into regs
             * 3-28, deriving BASE/PITCH from the buffer table and the blob
             * offsets from the *_BLOB_BASE registers -- exactly what
             * software (rtl/tb/quartznet_tb.cpp) does today -- then dispatch
             * into the SAME S_ADDQ/S_ELEM path single-descriptor mode uses.
             * Decisions here read desc_word_r directly rather than the
             * (not-yet-updated) r_op/op wire, since those NBAs land on this
             * same clock edge and are not visible until next cycle. -------- */
            S_POPULATE: begin
                r_op         <= {24'd0, desc_word_r[0][7:0]};
                r_flags      <= {24'd0, desc_word_r[0][15:8]};
                r_c_in       <= {16'd0, desc_word_r[1][15:0]};
                r_c_out      <= {16'd0, desc_word_r[1][31:16]};
                r_k          <= {16'd0, desc_word_r[2][15:0]};
                r_stride     <= {24'd0, desc_word_r[2][23:16]};
                r_dilation   <= {24'd0, desc_word_r[2][31:24]};
                r_pad        <= {16'd0, desc_word_r[3][15:0]};
                r_in_zp      <= {{24{desc_word_r[5][7]}},  desc_word_r[5][7:0]};
                r_out_zp     <= {{24{desc_word_r[5][15]}}, desc_word_r[5][15:8]};
                r_res_zp     <= {{24{desc_word_r[5][23]}}, desc_word_r[5][23:16]};
                r_out_cbase  <= {16'd0, desc_word_r[7][15:0]};
                r_w_off      <= r_w_blob_base  + desc_word_r[9];
                r_bias_off   <= r_qp_blob_base + desc_word_r[10];
                r_qmult_off  <= r_qp_blob_base + desc_word_r[11];
                r_rshift_off <= r_qp_blob_base + desc_word_r[12];

                /* in_buf/out_buf/res_buf are full bytes in the descriptor
                 * (room for growth) but only their low 3 bits index the
                 * N_BUF_MAX=8 on-chip buffer table; real tables never use
                 * more than 5 (QN_N_BUFFERS). */
                r_in_base    <= buf_off_r[desc_word_r[0][18:16]];
                r_in_pitch   <= {16'd0, buf_ch_r[desc_word_r[0][18:16]][15:0]};
                r_out_base   <= buf_off_r[desc_word_r[0][26:24]];
                r_out_pitch  <= {16'd0, buf_ch_r[desc_word_r[0][26:24]][15:0]};
                r_res_base   <= buf_off_r[desc_word_r[3][18:16]];
                r_res_pitch  <= {16'd0, buf_ch_r[desc_word_r[3][18:16]][15:0]};

                /* Autonomous mode requires in_off == 0 (see the module
                 * header) -- no divider, so IN_CBASE is staged directly. */
                r_in_cbase <= {16'd0, desc_word_r[6][15:0]};
                if (desc_word_r[6] != 32'd0) err_bad_in_off <= 1'b1;

                in_zp_q  <= {desc_word_r[5][7], desc_word_r[5][7:0]};
                res_zp_q <= {desc_word_r[5][23], desc_word_r[5][23:16]};
                out_zp_q <= {{24{desc_word_r[5][15]}}, desc_word_r[5][15:8]};
                relu_q   <= desc_word_r[0][8];
                cyc_cnt  <= 32'd0;

                qp_idx      <= (desc_word_r[0][1:0] == OP_ADD) ? 3'd3 : 3'd0;
                qp_last_add <= 1'b0;
                tbl_start_pulse <= 1'b1;
                state <= S_KICK;
            end
            /* ---- Increment 2: addr_gen's own start must be high the SAME
             * cycle it is read, and it must latch cfg_* on the SAME edge
             * quartznet_accel's own state leaves S_ELEM's predecessor --
             * exactly the alignment start_pulse/S_IDLE already have (start
             * is set one state before the state that depends on it, not in
             * the same state that consumes it). tbl_start_pulse was set by
             * S_POPULATE's NBA (previous cycle) so it is already stable and
             * visible to addr_gen at this same edge; op/r_op are equally
             * already updated (same NBA batch), so reading `op` here (not
             * desc_word_r) is safe, unlike inside S_POPULATE itself. ------ */
            S_KICK: state <= (op == OP_ADD) ? S_ADDQ : S_ELEM;
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
                if (table_mode) begin
                    if ((desc_idx + 32'd1) < hdr_n_desc) begin
                        /* table_done_pulse (above) is false this cycle since
                         * it checks this same condition inverted -- next
                         * descriptor. In SINGLE_STEP mode, wait in S_TWAIT for
                         * firmware to W1C-clear STATUS.DONE (the ack that it
                         * has safely read this descriptor's output) before
                         * fetching the next one -- see the D4 header comment. */
                        desc_idx <= desc_idx + 32'd1;
                        word_i   <= 4'd0;
                        state    <= r_single_step ? S_TWAIT : S_DESC;
                    end else begin
                        /* Last descriptor: table_done_pulse fires THIS cycle
                         * (state==S_FIN && table_mode && desc_idx+1>=n_desc),
                         * latching table_done_sticky in the MMIO block. No
                         * next descriptor to wait for, so SINGLE_STEP doesn't
                         * apply here regardless of r_single_step. */
                        table_mode <= 1'b0;
                        state      <= S_IDLE;
                    end
                end else begin
                    state <= S_IDLE;
                end
            end
            /* ---- D4: SINGLE_STEP handoff -------------------------------- */
            S_TWAIT: begin
                if (!done_sticky) state <= S_DESC;
            end
            /* ---- D4: multi-cycle background fill (one bd_* write/cycle) - */
            S_MEMFILL: begin
                r_mem_we_q       <= 1'b1;
                r_mem_sel_q      <= r_mem_ctrl[0];
                r_mem_addr_q     <= r_mem_addr;
                r_mem_wdata_q    <= r_mem_fill_byte;
                r_mem_addr       <= r_mem_addr + 32'd1;
                r_fill_remaining <= r_fill_remaining - 32'd1;
                if (r_fill_remaining <= 32'd1) state <= S_IDLE;
            end
            default: state <= S_IDLE;
            endcase
        end
    end

    wire _unused_agdone = &{1'b0, ag_done};

endmodule

`default_nettype wire
