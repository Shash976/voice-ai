`timescale 1 ns / 1 ps
/* picorv32_soc.v — minimal SoC wrapper exposing the CPU memory bus to Verilator.
 *
 * All memory and I/O are handled by the C++ testbench (sim_main.cpp).
 * Optional PicoRV32 interfaces (look-ahead bus, co-processor, IRQ, trace)
 * are tied off here so they don't need to be wired in the testbench.
 */

`default_nettype none

module picorv32_soc #(
    parameter ENABLE_MUL      = 1,
    parameter ENABLE_FAST_MUL = 1,
    parameter ENABLE_DIV      = 1,
    parameter COMPRESSED_ISA  = 1,
    parameter ENABLE_COUNTERS = 1
) (
    input  wire        clk,
    input  wire        resetn,

    /* Native memory bus — handled by C++ testbench */
    output wire        mem_valid,
    output wire        mem_instr,
    input  wire        mem_ready,
    output wire [31:0] mem_addr,
    output wire [31:0] mem_wdata,
    output wire  [3:0] mem_wstrb,
    input  wire [31:0] mem_rdata,

    output wire        trap
);

/* Look-ahead memory interface outputs — unused, left open */
wire        mem_la_read;
wire        mem_la_write;
wire [31:0] mem_la_addr;
wire [31:0] mem_la_wdata;
wire  [3:0] mem_la_wstrb;

/* Co-processor interface — disabled (inputs tied low, outputs ignored) */
wire        pcpi_valid;
wire [31:0] pcpi_insn;
wire [31:0] pcpi_rs1;
wire [31:0] pcpi_rs2;

/* Trace interface — unused */
wire        trace_valid;
wire [35:0] trace_data;

picorv32 #(
    .ENABLE_MUL       (ENABLE_MUL),
    .ENABLE_FAST_MUL  (ENABLE_FAST_MUL),
    .ENABLE_DIV       (ENABLE_DIV),
    .COMPRESSED_ISA   (COMPRESSED_ISA),
    .ENABLE_COUNTERS  (ENABLE_COUNTERS),
    .REGS_INIT_ZERO   (1)
) cpu (
    .clk          (clk),
    .resetn       (resetn),
    .trap         (trap),

    /* Main memory bus */
    .mem_valid    (mem_valid),
    .mem_instr    (mem_instr),
    .mem_ready    (mem_ready),
    .mem_addr     (mem_addr),
    .mem_wdata    (mem_wdata),
    .mem_wstrb    (mem_wstrb),
    .mem_rdata    (mem_rdata),

    /* Look-ahead bus — wired to local wires, ignored by testbench */
    .mem_la_read  (mem_la_read),
    .mem_la_write (mem_la_write),
    .mem_la_addr  (mem_la_addr),
    .mem_la_wdata (mem_la_wdata),
    .mem_la_wstrb (mem_la_wstrb),

    /* Co-processor — tied off (no external co-processor) */
    .pcpi_valid   (pcpi_valid),
    .pcpi_insn    (pcpi_insn),
    .pcpi_rs1     (pcpi_rs1),
    .pcpi_rs2     (pcpi_rs2),
    .pcpi_wr      (1'b0),
    .pcpi_rd      (32'b0),
    .pcpi_wait    (1'b0),
    .pcpi_ready   (1'b0),

    /* Interrupts — all masked (no IRQ sources in simulation) */
    .irq          (32'b0),
    .eoi          (),

    /* Trace — ignored */
    .trace_valid  (trace_valid),
    .trace_data   (trace_data)
);

endmodule
`default_nettype wire
