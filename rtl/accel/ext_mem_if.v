`timescale 1 ns / 1 ps
/* ext_mem_if.v — behavioral external-memory model with a real valid/ready
 * handshake, for two logical channels.
 *
 *   QSPI  read-only   weights + quantization parameters
 *   PSRAM read/write  activation arena
 *
 * ── Why this module exists ──────────────────────────────────────────────────
 *
 * `tinymac_accel.v` consumes operands combinationally from a 0-wait-state
 * environment and HAS NO STALL INPUT AT ALL.  That is fine for a matvec core
 * whose operands sit in on-chip SRAM, and it is exactly what breaks once real
 * memory has variable latency (docs/07_quartznet_pivot.md, "Components").  This
 * model supplies that latency so the sequencer's stall path is actually
 * exercised in simulation rather than assumed.
 *
 * ── Scope: NOT synthesizable, and deliberately so ───────────────────────────
 *
 * Storage is plain `reg [7:0] mem [0:N-1]` arrays.  This is the same choice the
 * repo already makes for the 256 KB main RAM in sim/verilator/sim_main.cpp — a
 * C++ array, not RTL.  Real fakeram45 macro wrapping (act_sram.v / wt_buf.v) is
 * explicitly out of scope for this increment: there is no fakeram45 behavioral
 * model to simulate against and no OpenROAD on this machine to synthesize
 * against, so wrapping macros here could not be verified either way.  On real
 * silicon this block becomes the QSPI controller + PSRAM PHY and the backdoor
 * port below disappears.
 *
 * ── Request model ───────────────────────────────────────────────────────────
 *
 * A request is a strided GATHER of up to LANES bytes:
 *
 *     byte[l] = mask[l] ? mem[addr + l * stride] : 0        for l < LANES
 *
 * With stride == 1 this is an ordinary contiguous burst (weights, pointwise
 * activations); with stride == dilation * pitch it is the depthwise tap walk.
 * A gather is the right granularity because the real part reads through paired
 * fakeram45 banks at 64 B/cycle (CLAUDE.md gotcha (f)), so per-byte handshaking
 * would model a machine nobody is building.
 *
 * Masked-off lanes are never fetched, so the caller may leave their addresses
 * garbage — which the depthwise left-padding case relies on.  Any address that
 * still lands outside the array reads as 0 rather than corrupting the model.
 *
 * The QSPI channel also serves 32-bit little-endian word reads (`req_word`) for
 * the bias / qmult / rshift blobs.
 *
 * Latency is `LAT_BASE + (addr[LAT_JIT_BITS-1:0] if LAT_JITTER)` cycles, so the
 * stall length varies from request to request instead of being a constant the
 * sequencer could accidentally be tuned to.  `req_ready` deasserts while a
 * request is in flight: one outstanding request per channel.
 */

`default_nettype none

module ext_mem_if #(
    parameter integer LANES        = 32,
    parameter integer QSPI_BYTES   = 1 << 20,  /* weight + qparam blobs */
    parameter integer PSRAM_BYTES  = 1 << 20,  /* activation arena      */
    parameter integer QSPI_LAT     = 6,
    parameter integer PSRAM_LAT    = 3,
    parameter integer LAT_JITTER   = 1,        /* 0 = fixed latency     */
    parameter integer LAT_JIT_BITS = 2         /* jitter = addr[1:0]    */
) (
    input  wire        clk,
    input  wire        rst_n,

    /* ── QSPI channel: read-only ────────────────────────────────────────── */
    input  wire        q_req_valid,
    output wire        q_req_ready,
    input  wire [31:0] q_req_addr,
    input  wire [31:0] q_req_stride,
    input  wire [LANES-1:0] q_req_mask,
    input  wire        q_req_word,        /* 1 = 32-bit LE word at q_req_addr */
    output reg         q_rsp_valid,
    output reg  [LANES*8-1:0] q_rsp_bytes,
    output reg  [31:0] q_rsp_word,

    /* ── PSRAM channel: read / write ────────────────────────────────────── */
    input  wire        p_req_valid,
    output wire        p_req_ready,
    input  wire        p_req_we,          /* 1 = single-byte write */
    input  wire [31:0] p_req_addr,
    input  wire [31:0] p_req_stride,
    input  wire [LANES-1:0] p_req_mask,
    input  wire [7:0]  p_req_wdata,
    output reg         p_rsp_valid,
    output reg  [LANES*8-1:0] p_rsp_bytes,

    /* ── Simulation backdoor ────────────────────────────────────────────────
     * Loads the blobs and reads results back without modelling a host bus.
     * Simulation-only, like the rest of this module. */
    input  wire        bd_en,
    input  wire        bd_sel,            /* 0 = QSPI, 1 = PSRAM */
    input  wire        bd_we,
    input  wire [31:0] bd_addr,
    input  wire [7:0]  bd_wdata,
    output wire [7:0]  bd_rdata
);

    reg [7:0] qmem [0:QSPI_BYTES-1];
    reg [7:0] pmem [0:PSRAM_BYTES-1];

    /* ── Backdoor ───────────────────────────────────────────────────────── */
    assign bd_rdata = (bd_addr < ((bd_sel == 1'b0) ? QSPI_BYTES[31:0] : PSRAM_BYTES[31:0]))
                    ? ((bd_sel == 1'b0) ? qmem[bd_addr] : pmem[bd_addr])
                    : 8'd0;

    /* ── Latency helper ─────────────────────────────────────────────────── */
    /* Latencies and jitter are single-digit cycle counts, so only the low byte
     * of each 32-bit intermediate can ever be significant. */
    /* verilator lint_off UNUSED */
    function [7:0] lat_of;
        input [31:0] base;
        input [31:0] addr;
        reg   [31:0] jit;
        begin
            jit    = (LAT_JITTER != 0) ? (addr & ((32'd1 << LAT_JIT_BITS) - 32'd1)) : 32'd0;
            lat_of = base[7:0] + jit[7:0];
        end
    endfunction
    /* verilator lint_on UNUSED */

    /* ── QSPI channel ───────────────────────────────────────────────────── */
    reg        q_busy;
    reg [7:0]  q_cnt;
    reg [LANES*8-1:0] q_data_q;
    reg [31:0] q_word_q;

    assign q_req_ready = !q_busy;

    integer ql;
    reg [31:0] qaddr;

    /* The gather result is assembled with BLOCKING assignments on purpose: it
     * is captured the moment the request is accepted, and the latency counter
     * below only models when it becomes observable.  This is behavioral model
     * code, not a synthesizable flop description. */
    /* verilator lint_off BLKSEQ */
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            q_busy      <= 1'b0;
            q_cnt       <= 8'd0;
            q_rsp_valid <= 1'b0;
            q_rsp_bytes <= {LANES*8{1'b0}};
            q_rsp_word  <= 32'd0;
        end else begin
            q_rsp_valid <= 1'b0;
            if (!q_busy) begin
                if (q_req_valid) begin
                    q_busy <= 1'b1;
                    q_cnt  <= lat_of(QSPI_LAT[31:0], q_req_addr);
                    /* Capture the data at accept time; the latency counter only
                     * models when it becomes observable. */
                    q_data_q = {LANES*8{1'b0}};
                    for (ql = 0; ql < LANES; ql = ql + 1) begin
                        if (q_req_mask[ql]) begin
                            qaddr = q_req_addr + q_req_stride * ql[31:0];
                            if (qaddr < QSPI_BYTES[31:0])
                                q_data_q[8*ql +: 8] = qmem[qaddr];
                        end
                    end
                    q_word_q = 32'd0;
                    if (q_req_word) begin
                        for (ql = 0; ql < 4; ql = ql + 1) begin
                            qaddr = q_req_addr + ql[31:0];
                            if (qaddr < QSPI_BYTES[31:0])
                                q_word_q[8*ql +: 8] = qmem[qaddr];
                        end
                    end
                end
            end else begin
                if (q_cnt == 8'd0) begin
                    q_busy      <= 1'b0;
                    q_rsp_valid <= 1'b1;
                    q_rsp_bytes <= q_data_q;
                    q_rsp_word  <= q_word_q;
                end else begin
                    q_cnt <= q_cnt - 8'd1;
                end
            end
            /* Backdoor lives in this block so qmem keeps a single driver. */
            if (bd_en && bd_we && (bd_sel == 1'b0) && (bd_addr < QSPI_BYTES[31:0]))
                qmem[bd_addr] <= bd_wdata;
        end
    end
    /* verilator lint_on BLKSEQ */

    /* ── PSRAM channel ──────────────────────────────────────────────────── */
    reg        p_busy;
    reg [7:0]  p_cnt;
    reg [LANES*8-1:0] p_data_q;

    assign p_req_ready = !p_busy;

    integer pl;
    reg [31:0] paddr;

    /* Blocking assignments here for the same reason as the QSPI channel. */
    /* verilator lint_off BLKSEQ */
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            p_busy      <= 1'b0;
            p_cnt       <= 8'd0;
            p_rsp_valid <= 1'b0;
            p_rsp_bytes <= {LANES*8{1'b0}};
        end else begin
            p_rsp_valid <= 1'b0;
            if (!p_busy) begin
                if (p_req_valid) begin
                    p_busy <= 1'b1;
                    p_cnt  <= lat_of(PSRAM_LAT[31:0], p_req_addr);
                    p_data_q = {LANES*8{1'b0}};
                    if (p_req_we) begin
                        if (p_req_addr < PSRAM_BYTES[31:0])
                            pmem[p_req_addr] <= p_req_wdata;
                    end else begin
                        for (pl = 0; pl < LANES; pl = pl + 1) begin
                            if (p_req_mask[pl]) begin
                                paddr = p_req_addr + p_req_stride * pl[31:0];
                                if (paddr < PSRAM_BYTES[31:0])
                                    p_data_q[8*pl +: 8] = pmem[paddr];
                            end
                        end
                    end
                end
            end else begin
                if (p_cnt == 8'd0) begin
                    p_busy      <= 1'b0;
                    p_rsp_valid <= 1'b1;
                    p_rsp_bytes <= p_data_q;
                end else begin
                    p_cnt <= p_cnt - 8'd1;
                end
            end
            /* Backdoor lives in this block so pmem keeps a single driver. */
            if (bd_en && bd_we && (bd_sel == 1'b1) && (bd_addr < PSRAM_BYTES[31:0]))
                pmem[bd_addr] <= bd_wdata;
        end
    end
    /* verilator lint_on BLKSEQ */

endmodule

`default_nettype wire
