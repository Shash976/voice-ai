`timescale 1 ns / 1 ps
/* qn_soc.v — PicoRV32 + quartznet_accel SoC wrapper for Verilator.
 *
 * Stage 7 Gap 1 increment D5: the first real Verilog wiring of
 * quartznet_accel.v onto a PicoRV32 bus. Wraps the UNMODIFIED
 * rtl/soc/picorv32_soc.v (not a copy, not a modified fork -- the Stage 3/4
 * gate `make -C sim/verilator run` must stay byte-for-byte unaffected) and
 * adds the ~25 lines of address decode this needed: quartznet_accel.v's
 * mmio_* register file already does the hard part (a clean, already-verified
 * word-indexed interface -- see quartznet_tb.cpp), so wiring it onto a real
 * bus is a small mux, not a redesign.
 *
 * ── Address decode ───────────────────────────────────────────────────────
 *
 * quartznet_accel owns the 4 KB window 0x2000_0000-0x2000_0FFF, the same
 * base CLAUDE.md documents for the Stage 4 TinyMAC accelerator (a different
 * SoC/sim entirely -- sim/verilator's C++-only ACCEL_BASE -- this is the
 * first time that address range is backed by real RTL rather than a C++
 * behavioral model). mmio_addr is a WORD index (0-255, matching
 * quartznet_accel.v's own `case (mmio_addr)`), so cpu_mem_addr[9:2] is the
 * correct slice: byte addresses 0x000-0x3FC within the window map to word
 * indices 0-255.
 *
 * The register file is 0-wait-state (mem_ready asserted combinationally the
 * same cycle as the request), exactly like the flat RAM model the
 * C++ testbench already implements for sim/verilator's non-accelerator
 * address range -- so mmio_we naturally pulses for exactly the one cycle
 * PicoRV32 holds mem_valid for this transaction, matching how
 * quartznet_tb.cpp's mmio_write() drives it (one tick() per write).
 *
 * mmio_we additionally requires wstrb == 4'hF (a full 32-bit store): the
 * register file has no byte-strobe support, matching the existing
 * precedent for the Stage 4 TinyMAC accelerator in sim/verilator/sim_main.cpp
 * ("Accelerator registers are 32-bit word writes (strb=0xF)").
 *
 * Everything outside the accelerator's window (RAM/UART/EXIT) passes
 * straight through to the external mem_* bus, unchanged from what
 * picorv32_soc.v's own top-level ports already expose today -- the C++ side
 * (sim/verilator_qn's MODE=rtl) handles it exactly like sim_main.cpp does
 * for sim/verilator.
 *
 * ── What this is NOT ─────────────────────────────────────────────────────
 *
 * Not a redesign of quartznet_accel.v's QSPI/PSRAM memory model: the bd_*
 * simulation backdoor (used by rtl/tb) and the MEM_* host-access port
 * (Stage 7 Gap 1 D4, used by real firmware) both pass straight through
 * unchanged -- this module only decides which bus transactions are the
 * CPU talking to the accelerator's REGISTER FILE, not how the accelerator
 * itself reaches its own QSPI/PSRAM.
 *
 * Verilator flags a real (harmless) SYNCASYNCNET here: quartznet_accel.v
 * resets asynchronously (`posedge clk or negedge rst_n`) while picorv32
 * resets synchronously, and this module drives both from the same top-level
 * `resetn`. Both work correctly given a held-low reset pulse at start-of-day
 * (which every testbench/sim harness in this repo already does) -- suppress
 * with -Wno-SYNCASYNCNET at the build, don't "fix" by changing either IP's
 * reset style. -Wno-BLKSEQ/-Wno-DECLFILENAME are pre-existing picorv32.v
 * characteristics, not introduced here.
 */

`default_nettype none

module qn_soc #(
    parameter ENABLE_MUL      = 1,
    parameter ENABLE_FAST_MUL = 1,
    parameter ENABLE_DIV      = 1,
    parameter COMPRESSED_ISA  = 1,
    parameter ENABLE_COUNTERS = 1,

    parameter integer LANES       = 32,
    parameter integer ACC_W       = 32,
    parameter integer QSPI_BYTES  = 1 << 20,
    parameter integer PSRAM_BYTES = 1 << 20,
    parameter integer QSPI_LAT    = 6,
    parameter integer PSRAM_LAT   = 3,
    parameter integer LAT_JITTER  = 1
) (
    input  wire        clk,
    input  wire        resetn,

    /* ── External bus: RAM/UART/EXIT, handled by the C++ testbench ────────
     * Identical in shape and meaning to picorv32_soc.v's own top-level bus. */
    output wire        mem_valid,
    output wire        mem_instr,
    input  wire        mem_ready,
    output wire [31:0] mem_addr,
    output wire [31:0] mem_wdata,
    output wire  [3:0] mem_wstrb,
    input  wire [31:0] mem_rdata,

    output wire        trap,

    /* ── Simulation-only backdoor into quartznet_accel's QSPI/PSRAM model ──
     * Passed straight through to quartznet_accel.v's own bd_* port -- see
     * that file's header for the external-input-priority mux this feeds. */
    input  wire        bd_en,
    input  wire        bd_sel,
    input  wire        bd_we,
    input  wire [31:0] bd_addr,
    input  wire [7:0]  bd_wdata,
    output wire [7:0]  bd_rdata
);

    /* ── CPU ───────────────────────────────────────────────────────────── */
    wire        cpu_mem_valid, cpu_mem_instr;
    wire [31:0] cpu_mem_addr, cpu_mem_wdata;
    wire  [3:0] cpu_mem_wstrb;
    reg         cpu_mem_ready;
    reg  [31:0] cpu_mem_rdata;

    picorv32_soc #(
        .ENABLE_MUL      (ENABLE_MUL),
        .ENABLE_FAST_MUL (ENABLE_FAST_MUL),
        .ENABLE_DIV      (ENABLE_DIV),
        .COMPRESSED_ISA  (COMPRESSED_ISA),
        .ENABLE_COUNTERS (ENABLE_COUNTERS)
    ) u_cpu (
        .clk     (clk),
        .resetn  (resetn),
        .mem_valid (cpu_mem_valid),
        .mem_instr (cpu_mem_instr),
        .mem_ready (cpu_mem_ready),
        .mem_addr  (cpu_mem_addr),
        .mem_wdata (cpu_mem_wdata),
        .mem_wstrb (cpu_mem_wstrb),
        .mem_rdata (cpu_mem_rdata),
        .trap      (trap)
    );

    /* ── Address decode ────────────────────────────────────────────────── */
    wire accel_hit = cpu_mem_valid && (cpu_mem_addr[31:12] == 20'h20000);

    wire [31:0] accel_rdata;

    /* Tile-walk observability + busy/done_pulse are rtl/tb-only debug
     * outputs (see quartznet_accel.v) with no consumer here -- left
     * unconnected on purpose. */
    /* verilator lint_off PINMISSING */
    quartznet_accel #(
        .LANES       (LANES),
        .ACC_W       (ACC_W),
        .QSPI_BYTES  (QSPI_BYTES),
        .PSRAM_BYTES (PSRAM_BYTES),
        .QSPI_LAT    (QSPI_LAT),
        .PSRAM_LAT   (PSRAM_LAT),
        .LAT_JITTER  (LAT_JITTER)
    ) u_accel (
        .clk         (clk),
        .rst_n       (resetn),

        .mmio_we     (accel_hit && (cpu_mem_wstrb == 4'hF)),
        .mmio_addr   (cpu_mem_addr[9:2]),
        .mmio_wdata  (cpu_mem_wdata),
        .mmio_rdata  (accel_rdata),

        .bd_en       (bd_en),
        .bd_sel      (bd_sel),
        .bd_we       (bd_we),
        .bd_addr     (bd_addr),
        .bd_wdata    (bd_wdata),
        .bd_rdata    (bd_rdata)
    );
    /* verilator lint_on PINMISSING */

    always @* begin
        if (accel_hit) begin
            cpu_mem_ready = 1'b1;       /* 0-wait-state register file */
            cpu_mem_rdata = accel_rdata;
        end else begin
            cpu_mem_ready = mem_ready;  /* forwarded from the external bus */
            cpu_mem_rdata = mem_rdata;
        end
    end

    assign mem_valid = cpu_mem_valid && !accel_hit;
    assign mem_instr = cpu_mem_instr;
    assign mem_addr  = cpu_mem_addr;
    assign mem_wdata = cpu_mem_wdata;
    assign mem_wstrb = cpu_mem_wstrb;

endmodule

`default_nettype wire
