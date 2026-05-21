/* picorv32_soc.v
 *
 * Minimal SoC wrapper around PicoRV32.
 * All memory and I/O are handled by the Verilator C++ testbench —
 * this module just exposes the CPU memory bus as top-level ports so
 * the testbench can intercept every access.
 *
 * Memory map (enforced by testbench, not this RTL):
 *   0x00000000 - 0x0003FFFF  RAM  256 KB  (code + data + stack)
 *   0x10000000               UART TX       (write = emit char)
 *   0x10000004               SIM_EXIT      (write = end simulation)
 *
 * Parameters you can tune for design-space exploration (Stage 5):
 *   ENABLE_MUL       1 = hardware multiplier (needed for int8 MAC)
 *   ENABLE_FAST_MUL  1 = single-cycle mul  (0 = shift-add, slower)
 *   ENABLE_COMPRESSED 1 = RV32C 16-bit insns (reduces code size)
 *   ENABLE_COUNTERS  1 = rdcycle CSR (needed for cycle measurement)
 */

`default_nettype none

module picorv32_soc #(
    parameter ENABLE_MUL       = 1,
    parameter ENABLE_FAST_MUL  = 1,
    parameter ENABLE_DIV       = 0,
    parameter ENABLE_COMPRESSED = 1,
    parameter ENABLE_COUNTERS  = 1
) (
    input  wire        clk,
    input  wire        resetn,

    /* Memory bus — handled by C++ testbench */
    output wire        mem_valid,
    output wire        mem_instr,
    input  wire        mem_ready,
    output wire [31:0] mem_addr,
    output wire [31:0] mem_wdata,
    output wire  [3:0] mem_wstrb,
    input  wire [31:0] mem_rdata,

    output wire        trap
);

picorv32 #(
    .ENABLE_MUL       (ENABLE_MUL),
    .ENABLE_FAST_MUL  (ENABLE_FAST_MUL),
    .ENABLE_DIV       (ENABLE_DIV),
    .ENABLE_COMPRESSED(ENABLE_COMPRESSED),
    .ENABLE_COUNTERS  (ENABLE_COUNTERS),
    .REGS_INIT_ZERO   (1)
) cpu (
    .clk      (clk),
    .resetn   (resetn),
    .trap     (trap),
    .mem_valid(mem_valid),
    .mem_instr(mem_instr),
    .mem_ready(mem_ready),
    .mem_addr (mem_addr),
    .mem_wdata(mem_wdata),
    .mem_wstrb(mem_wstrb),
    .mem_rdata(mem_rdata)
);

endmodule
`default_nettype wire
