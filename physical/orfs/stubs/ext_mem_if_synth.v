`timescale 1 ns / 1 ps
/* ext_mem_if_synth.v — SYNTHESIS-ONLY stand-in for rtl/accel/ext_mem_if.v.
 *
 * PHYSICAL DESIGN ONLY.  This file is never read by Verilator, never simulated,
 * and is not part of the verified RTL.  sim/verilator_qn/Makefile lists the real
 * rtl/accel/ext_mem_if.v explicitly and nothing in this repo globs *.v, so the
 * two coexist safely.  The module NAME must stay `ext_mem_if` -- that is what
 * quartznet_accel.v's `u_mem` instance binds to.  Only the FILE name differs, so
 * a config.mk file list can never be misread as pulling in the real model.
 *
 * -- Why a stub at all --------------------------------------------------------
 *
 * rtl/accel/ext_mem_if.v says so in its own header: "NOT synthesizable, and
 * deliberately so".  Its storage is `reg [7:0] qmem[0:QSPI_BYTES-1]` +
 * `pmem[0:PSRAM_BYTES-1]` -- 8 Mbit at the sim defaults, 208 Mbit at the real
 * 25 MB / 1 MB ones.  ORFS refuses that outright: scripts/synth.tcl dumps
 * mem.json and runs scripts/mem_dump.py --max-bits $(SYNTH_MEMORY_MAX_BITS)
 * (default 4096) purely to "fail early if this synthesis run is doomed".  On
 * real silicon this block becomes a QSPI controller + PSRAM PHY, which has not
 * been written.  Scoping it out mirrors exactly how tinymac_accel was scoped for
 * this repo's one existing GDS -- "just the accelerator compute core... not main
 * RAM... the safe choice for a first GDS" (physical/orfs/README.md).
 *
 * -- Why a stub and NOT a Yosys blackbox --------------------------------------
 *
 * A true `(* blackbox *)` ext_mem_if does elaborate and synthesize cleanly under
 * the ORFS Yosys 0.64 (measured).  But it survives into results/1_2_yosys.v as a
 * cell, and the very next ORFS step -- scripts/synth_odb.tcl -> scripts/load.tcl:
 * read_lef / read_verilog / link_design -- has no LEF master for it and cannot
 * link.  Supplying one means hand-writing an 805-pin LEF and .lib for
 * ADDITIONAL_LEFS / ADDITIONAL_LIBS.  Worse, Yosys mangles a parameterised
 * blackbox's name: the instance emitted is
 *   \$paramod$f781513705dc039e62151e995c6717a99693cdfe\ext_mem_if
 * so the LEF macro name could not even be written down in advance.
 *
 * -- What this stub does, and why it is written this way ---------------------
 *
 * It is NOT trying to be functionally correct -- no gate-level simulation depends
 * on it.  It has one job: present the same interface LOAD so the logic around it
 * is not dead-code-eliminated, at the least area that achieves that.
 *
 *   1. EVERY input bit is consumed.  Load-bearing, not tidiness.  An earlier
 *      draft folded only q_req_addr[7:0] into the response; Yosys then correctly
 *      pruned the upper 24 bits of every address and took a large part of
 *      addr_gen's 32-bit address arithmetic with it -- 79,698 cells fell to
 *      64,284.  fold32() XOR-folds all four bytes of each 32-bit port so no bit
 *      is danglable.  `shadow` exists only so p_req_wdata (and therefore
 *      out_byte, and therefore the whole requantize datapath) and the bd_* group
 *      have a real sink.
 *
 *   2. Responses are REGISTERED, matching the real module (q_rsp_* / p_rsp_* are
 *      `output reg` there too), so the boundary keeps the same flop-bounded
 *      timing character.  q_req_ready / p_req_ready / bd_rdata stay
 *      combinational, again as in the real module.
 *
 *   3. Latency is a flat 1 cycle instead of QSPI_LAT/PSRAM_LAT + jitter.  Static
 *      timing analysis does not care how many cycles a handshake takes; a
 *      counter comparison would be pure area for no timing information.
 *
 * All seven parameters are accepted and IGNORED.  quartznet_accel.v passes six
 * of them down at its u_mem instantiation (LANES, QSPI_BYTES, PSRAM_BYTES,
 * QSPI_LAT, PSRAM_LAT, LAT_JITTER), so the parameter list must still accept them
 * or elaboration fails.  Only LANES changes any geometry here; QSPI_BYTES and
 * PSRAM_BYTES sized arrays that no longer exist, so their values cannot affect
 * synthesized size at all.
 *
 * MEASURED COST (Yosys 0.64, nangate45, LANES=32, standalone):
 *   1,924 cells / 4,359.74 um^2, 67.9% of it sequential.
 * Subtract that from the top-level number to get the real accelerator's area.
 */

`default_nettype none

module ext_mem_if #(
    parameter integer LANES        = 32,
    parameter integer QSPI_BYTES   = 1 << 20,
    parameter integer PSRAM_BYTES  = 1 << 20,
    parameter integer QSPI_LAT     = 6,
    parameter integer PSRAM_LAT    = 3,
    parameter integer LAT_JITTER   = 1,
    parameter integer LAT_JIT_BITS = 2
) (
    input  wire        clk,
    input  wire        rst_n,

    /* -- QSPI channel: read-only ------------------------------------------ */
    input  wire        q_req_valid,
    output wire        q_req_ready,
    input  wire [31:0] q_req_addr,
    input  wire [31:0] q_req_stride,
    input  wire [LANES-1:0] q_req_mask,
    input  wire        q_req_word,
    output reg         q_rsp_valid,
    output reg  [LANES*8-1:0] q_rsp_bytes,
    output reg  [31:0] q_rsp_word,

    /* -- PSRAM channel: read / write --------------------------------------- */
    input  wire        p_req_valid,
    output wire        p_req_ready,
    input  wire        p_req_we,
    input  wire [31:0] p_req_addr,
    input  wire [31:0] p_req_stride,
    input  wire [LANES-1:0] p_req_mask,
    input  wire [7:0]  p_req_wdata,
    output reg         p_rsp_valid,
    output reg  [LANES*8-1:0] p_rsp_bytes,

    /* -- Backdoor (in a real SoC, quartznet_accel's MEM_* port drives it) -- */
    input  wire        bd_en,
    input  wire        bd_sel,
    input  wire        bd_we,
    input  wire [31:0] bd_addr,
    input  wire [7:0]  bd_wdata,
    output wire [7:0]  bd_rdata
);

    /* XOR-fold a 32-bit port to 8 bits.  Cheap (3 x XOR8 = 24 XOR2 cells) and,
     * critically, touches every bit so nothing upstream can be pruned.  Part
     * selects are unsigned in Verilog, so this never mixes signedness -- the
     * Yosys-0.64 genrtlil.cc:2214 assert of CLAUDE.md gotcha (b). */
    function [7:0] fold32;
        input [31:0] w;
        begin
            fold32 = w[7:0] ^ w[15:8] ^ w[23:16] ^ w[31:24];
        end
    endfunction

    /* The entire "storage" of this stub: one byte.  Its only purpose is to sink
     * the write-side ports (p_req_wdata, p_req_we, bd_*) and to be a source that
     * read responses depend on, so the store path stays live. */
    reg [7:0] shadow;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            shadow <= 8'd0;
        else if (bd_en && bd_we)
            shadow <= bd_wdata ^ fold32(bd_addr) ^ {7'd0, bd_sel};
        else if (p_req_valid && p_req_ready && p_req_we)
            shadow <= p_req_wdata ^ fold32(p_req_addr);
    end

    /* Combinational, like the real module's backdoor read. */
    assign bd_rdata = shadow ^ fold32(bd_addr);

    /* Per-lane "data": a deterministic function of (address, stride, lane, last
     * write).  Distinct per lane, so int8_mac_array sees LANES different bytes
     * and its adder tree cannot be collapsed.  `li` is an integer loop variable
     * in an always @* block, so the loop unrolls at elaboration and
     * q_gather[8*li +: 8] is a CONSTANT part select -- no address decoder. */
    wire [7:0] q_seed = fold32(q_req_addr) ^ fold32(q_req_stride) ^ shadow;
    wire [7:0] p_seed = fold32(p_req_addr) ^ fold32(p_req_stride) ^ shadow;

    integer li;
    reg [LANES*8-1:0] q_gather, p_gather;
    always @* begin
        q_gather = {LANES*8{1'b0}};
        p_gather = {LANES*8{1'b0}};
        for (li = 0; li < LANES; li = li + 1) begin
            q_gather[8*li +: 8] = q_req_mask[li] ? (q_seed ^ li[7:0]) : 8'd0;
            p_gather[8*li +: 8] = p_req_mask[li] ? (p_seed ^ li[7:0]) : 8'd0;
        end
    end

    /* One outstanding request per channel, fixed 1-cycle latency (see header). */
    reg q_busy, p_busy;
    assign q_req_ready = !q_busy;
    assign p_req_ready = !p_busy;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            q_busy      <= 1'b0;
            q_rsp_valid <= 1'b0;
            q_rsp_bytes <= {LANES*8{1'b0}};
            q_rsp_word  <= 32'd0;
            p_busy      <= 1'b0;
            p_rsp_valid <= 1'b0;
            p_rsp_bytes <= {LANES*8{1'b0}};
        end else begin
            q_rsp_valid <= 1'b0;
            p_rsp_valid <= 1'b0;

            if (!q_busy) begin
                if (q_req_valid) begin
                    q_busy      <= 1'b1;
                    q_rsp_bytes <= q_gather;
                    q_rsp_word  <= q_req_word ? (q_req_addr ^ {24'd0, shadow})
                                              : 32'd0;
                end
            end else begin
                q_busy      <= 1'b0;
                q_rsp_valid <= 1'b1;
            end

            if (!p_busy) begin
                if (p_req_valid) begin
                    p_busy      <= 1'b1;
                    p_rsp_bytes <= p_gather;
                end
            end else begin
                p_busy      <= 1'b0;
                p_rsp_valid <= 1'b1;
            end
        end
    end

endmodule

`default_nettype wire
