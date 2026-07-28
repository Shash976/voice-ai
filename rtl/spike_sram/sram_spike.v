`timescale 1 ns / 1 ps
/* sram_spike.v — minimal fakeram45 macro-integration spike (nangate45).
 *
 * PURPOSE: this design computes nothing useful.  Its only job is to be the
 * smallest possible RTL that forces the ORFS flow to take real SRAM hard
 * macros all the way from synthesis to 6_final.gds, so that the QuartzNet
 * accelerator's ~18-macro memory system (docs/07_quartznet_pivot.md, Stage E)
 * is not the first time this repo ever places a macro.
 *
 * What it has to do to be a valid test:
 *   1. instantiate real fakeram45 macros (CLASS BLOCK in LEF)  → macro placement
 *   2. drive every macro input from registered logic           → real std cells
 *   3. consume every macro rd_out bit into a registered output → the router is
 *      forced to actually reach every pin; nothing can be optimised away
 *   4. share one clock across all macros                       → CTS must build
 *      a tree spanning the macro clock pins
 *
 * Structure: a free-running address counter walks the whole address space,
 * writing `din` (when `we`) and XOR-folding everything read back into `dout`.
 * The fold is deliberately a reduction so the output stays 32 bits wide no
 * matter how many macros are instantiated — top-level pin count stays flat as
 * N_512X64/N_1024X32 scale, which keeps the 4-vs-8 macro comparison about
 * macros rather than about I/O.
 *
 * Parameters (swept via ORFS VERILOG_TOP_PARAMS → yosys chparam):
 *   N_512X64  — number of fakeram45_512x64  instances (512 words × 64 bits, 4 KB)
 *   N_1024X32 — number of fakeram45_1024x32 instances (1024 words × 32 bits, 4 KB)
 *
 * Macro interface (from platforms/nangate45/lib/fakeram45_*.lib):
 *   clk, ce_in, we_in, addr_in[], wd_in[], w_mask_in[], rd_out[]
 * Control-pin polarity follows the upstream ORFS convention in
 * designs/nangate45/ariane133/macros.v, which drives ce_in/we_in ACTIVE LOW
 * (`csel_b = ~CSel_SI`).  fakeram45 has no behavioural model, so polarity is
 * unverifiable and physically irrelevant here — it only ever matters for
 * simulation, and there is nothing to simulate against.  Matching upstream
 * costs nothing and avoids inventing a third convention.
 *
 * Yosys 0.64 note (see CLAUDE.md): this file is deliberately 100% unsigned —
 * no $signed(), no signed `integer` in an unsigned expression, no mixed-sign
 * ?: — to stay clear of the genrtlil.cc:2214 assert.
 */

`default_nettype none

module sram_spike #(
    parameter integer N_512X64  = 2,
    parameter integer N_1024X32 = 2
) (
    input  wire        clk,
    input  wire        rst_n,

    input  wire        start,      /* pulse high to start a sweep */
    input  wire        we,         /* 1 = write din, 0 = read only */
    input  wire [31:0] din,        /* write data, broadcast to all macros */

    output reg  [31:0] dout,       /* XOR-fold of everything read back */
    output reg         busy        /* high while a sweep is in progress */
);

    /* ── Address / control sequencer ────────────────────────────────────────
     * 10-bit counter: full range for the 1024-deep macros, low 9 bits for the
     * 512-deep ones.  Wrapping back to 0 ends the sweep.
     */
    reg [9:0] addr_q;
    reg       run_q;

    wire [9:0] addr_next = addr_q + 10'd1;

    always @(posedge clk) begin
        if (!rst_n) begin
            addr_q <= 10'd0;
            run_q  <= 1'b0;
        end else begin
            if (run_q) begin
                addr_q <= addr_next;
                if (addr_next == 10'd0) begin
                    run_q <= 1'b0;      /* wrapped → sweep complete */
                end
            end else if (start) begin
                addr_q <= 10'd0;
                run_q  <= 1'b1;
            end
        end
    end

    /* Registered macro controls.  Active low, per the ariane133 convention. */
    reg        ce_b_q;
    reg        we_b_q;
    reg [31:0] wd_q;

    always @(posedge clk) begin
        if (!rst_n) begin
            ce_b_q <= 1'b1;
            we_b_q <= 1'b1;
            wd_q   <= 32'd0;
        end else begin
            ce_b_q <= ~run_q;
            we_b_q <= ~(run_q & we);
            wd_q   <= din;
        end
    end

    /* ── Macro array ────────────────────────────────────────────────────────
     * Each instance contributes one 32-bit slice to `fold`; the 64-bit macros
     * pre-fold their two halves.  `chain` is a ripple XOR across all slices,
     * built with genvars rather than a procedural loop over an `integer` index
     * (that index would be signed — exactly the Yosys 0.64 hazard).
     */
    localparam integer N_TOTAL = N_512X64 + N_1024X32;

    wire [31:0] fold  [0:N_TOTAL-1];
    wire [31:0] chain [0:N_TOTAL];

    assign chain[0] = 32'd0;

    genvar i;
    generate
        /* 512 × 64 macros: 9-bit address, 64-bit data folded to 32. */
        for (i = 0; i < N_512X64; i = i + 1) begin : g_ram512x64
            wire [63:0] rd;

            fakeram45_512x64 u_ram (
                .clk       (clk),
                .ce_in     (ce_b_q),
                .we_in     (we_b_q),
                .addr_in   (addr_q[8:0]),
                .wd_in     ({wd_q, wd_q}),
                .w_mask_in (64'hFFFF_FFFF_FFFF_FFFF),
                .rd_out    (rd)
            );

            assign fold[i] = rd[31:0] ^ rd[63:32];
        end

        /* 1024 × 32 macros: full 10-bit address, 32-bit data. */
        for (i = 0; i < N_1024X32; i = i + 1) begin : g_ram1024x32
            wire [31:0] rd;

            fakeram45_1024x32 u_ram (
                .clk       (clk),
                .ce_in     (ce_b_q),
                .we_in     (we_b_q),
                .addr_in   (addr_q),
                .wd_in     (wd_q),
                .w_mask_in (32'hFFFF_FFFF),
                .rd_out    (rd)
            );

            assign fold[N_512X64 + i] = rd;
        end

        /* Ripple XOR: every macro output bit reaches `dout`. */
        for (i = 0; i < N_TOTAL; i = i + 1) begin : g_fold
            assign chain[i+1] = chain[i] ^ fold[i];
        end
    endgenerate

    always @(posedge clk) begin
        if (!rst_n) begin
            dout <= 32'd0;
            busy <= 1'b0;
        end else begin
            dout <= chain[N_TOTAL];
            busy <= run_q;
        end
    end

endmodule

/* ── fakeram45 blackbox stubs ───────────────────────────────────────────────
 * NOT strictly required by the ORFS flow: flow/scripts/synth_stdcells.tcl runs
 *   read_liberty -overwrite -setattr liberty_cell -lib {*}$::env(LIB_FILES)
 * and nangate45/config.mk folds $(ADDITIONAL_LIBS) into LIB_FILES, so yosys
 * already knows these cells' ports from the .lib.  Upstream macro designs
 * (ariane133/macros.v) rely on exactly that and never declare the fakeram
 * modules themselves.
 *
 * They are kept here anyway so this file lints standalone under Verilator and
 * reads as self-documenting.  `read_liberty` runs before `read_verilog` and the
 * liberty view wins, so these are inert during synthesis — see the comment at
 * synth_preamble.tcl:66-74 ("competing Verilog definitions in the source files
 * are ignored in favor of the liberty view, consistent with the behavior of the
 * builtin Verilog frontend").
 */

(* blackbox *)
module fakeram45_512x64 (
    input  wire        clk,
    input  wire        ce_in,
    input  wire        we_in,
    input  wire [8:0]  addr_in,
    input  wire [63:0] wd_in,
    input  wire [63:0] w_mask_in,
    output wire [63:0] rd_out
);
endmodule

(* blackbox *)
module fakeram45_1024x32 (
    input  wire        clk,
    input  wire        ce_in,
    input  wire        we_in,
    input  wire [9:0]  addr_in,
    input  wire [31:0] wd_in,
    input  wire [31:0] w_mask_in,
    output wire [31:0] rd_out
);
endmodule

`default_nettype wire
