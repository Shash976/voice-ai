/* sim_main.cpp — Verilator testbench for PicoRV32 + TinyVAD firmware
 *
 * Simulates the PicoRV32 SoC with all memory and I/O handled in C++.
 * No Verilog UART or RAM modules needed — the testbench intercepts
 * every memory bus transaction directly.
 *
 * Memory map:
 *   0x00000000 - 0x0003FFFF  RAM 256 KB  (code + data + stack)
 *   0x10000000               UART TX     write byte → stdout
 *   0x10000004               SIM_EXIT    write → end simulation
 *
 * Usage:
 *   ./sim_picorv32 <firmware.bin> [--vcd <out.vcd>]
 *
 * Output:
 *   CSV printed to stdout (from firmware UART)
 *   Simulation stats printed to stderr
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cassert>
#include <string>

#include "Vpicorv32_soc.h"
#include "verilated.h"
#include "verilated_vcd_c.h"   /* for optional VCD waveform dump */

/* ── Configuration ───────────────────────────────────────────────────────── */

#define RAM_SIZE       (256 * 1024)
#define UART_ADDR      0x10000000u
#define EXIT_ADDR      0x10000004u
#define MAX_CYCLES     800000000ULL   /* 800M cycle timeout (64 vectors × ~11M cycles each) */

/* ── State ───────────────────────────────────────────────────────────────── */

static uint8_t  ram[RAM_SIZE];
static bool     sim_done  = false;
static int      exit_code = 0;
static uint64_t cycle_count = 0;

/* ── Firmware loader ─────────────────────────────────────────────────────── */

static void load_firmware(const char *path)
{
    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "ERROR: cannot open firmware: %s\n", path);
        exit(1);
    }
    memset(ram, 0, sizeof(ram));
    size_t n = fread(ram, 1, RAM_SIZE, f);
    fclose(f);
    fprintf(stderr, "[sim] Loaded %zu bytes from %s\n", n, path);
}

/* ── Memory access handlers ──────────────────────────────────────────────── */

static uint32_t mem_read(uint32_t addr)
{
    if (addr + 4 <= RAM_SIZE) {
        uint32_t v;
        memcpy(&v, &ram[addr], 4);
        return v;
    }
    /* Unmapped reads return 0 (UART status = always ready) */
    return 0u;
}

static void mem_write(uint32_t addr, uint32_t data, uint8_t strb)
{
    if (addr + 4 <= RAM_SIZE) {
        for (int i = 0; i < 4; i++)
            if (strb & (1u << i))
                ram[addr + i] = (uint8_t)(data >> (8 * i));
        return;
    }

    /* Memory-mapped I/O: only act on byte-lane 0 for single-byte writes */
    switch (addr) {
    case UART_ADDR:
        if (strb & 0x1) {
            char c = (char)(data & 0xFF);
            putchar(c);
            fflush(stdout);
        }
        break;

    case EXIT_ADDR:
        sim_done  = true;
        exit_code = (int)(data & 0xFF);
        break;

    default:
        fprintf(stderr, "[sim] WARNING: write to unmapped addr 0x%08x data=0x%08x strb=%x\n",
                addr, data, strb);
        break;
    }
}

/* ── Main ────────────────────────────────────────────────────────────────── */

int main(int argc, char **argv)
{
    /* Parse arguments */
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <firmware.bin> [--vcd <out.vcd>]\n", argv[0]);
        return 1;
    }
    const char *fw_path  = argv[1];
    const char *vcd_path = nullptr;
    for (int i = 2; i < argc - 1; i++) {
        if (std::string(argv[i]) == "--vcd")
            vcd_path = argv[i + 1];
    }

    load_firmware(fw_path);

    /* Set up Verilator */
    Verilated::commandArgs(argc, argv);
    Verilated::traceEverOn(vcd_path != nullptr);

    auto *top = new Vpicorv32_soc;

    /* Optional VCD trace */
    VerilatedVcdC *vcd = nullptr;
    if (vcd_path) {
        vcd = new VerilatedVcdC;
        top->trace(vcd, 99);
        vcd->open(vcd_path);
        fprintf(stderr, "[sim] VCD trace → %s\n", vcd_path);
    }

    /* Reset: hold resetn=0 for 8 cycles */
    top->clk    = 0;
    top->resetn = 0;
    top->mem_ready = 0;
    top->mem_rdata = 0;
    for (int i = 0; i < 16; i++) {
        top->clk = !top->clk;
        top->eval();
        if (vcd) vcd->dump((vluint64_t)(cycle_count * 10 + (top->clk ? 5 : 0)));
    }
    top->resetn = 1;

    fprintf(stderr, "[sim] Reset released — starting simulation\n");

    /* ── Main simulation loop ──────────────────────────────────────────────
     * Negedge: present combinatorial memory response (0-latency RAM).
     * Posedge: CPU latches the response.
     * This gives 1 clock per memory transaction, matching the firmware's
     * assumption that memory is fast (no wait states).
     */
    while (!sim_done && cycle_count < MAX_CYCLES) {

        /* ── Negedge ── */
        top->clk = 0;
        top->eval();
        if (vcd) vcd->dump((vluint64_t)(cycle_count * 10));

        /* Respond to any pending memory request before posedge */
        top->mem_ready = 0;
        if (top->mem_valid) {
            if (top->mem_wstrb) {
                mem_write(top->mem_addr, top->mem_wdata, (uint8_t)top->mem_wstrb);
            } else {
                top->mem_rdata = mem_read(top->mem_addr);
            }
            top->mem_ready = 1;
        }
        top->eval();   /* re-evaluate so CPU sees mem_ready on posedge */

        /* ── Posedge ── */
        top->clk = 1;
        top->eval();
        if (vcd) vcd->dump((vluint64_t)(cycle_count * 10 + 5));

        cycle_count++;

        if (top->trap) {
            fprintf(stderr, "[sim] CPU TRAP at cycle %llu — check firmware\n",
                    (unsigned long long)cycle_count);
            break;
        }
    }

    /* ── Report ── */
    if (cycle_count >= MAX_CYCLES)
        fprintf(stderr, "[sim] TIMEOUT after %llu cycles\n",
                (unsigned long long)cycle_count);
    else
        fprintf(stderr, "[sim] Done in %llu cycles (wall: ~%.1f ms at 100 MHz)\n",
                (unsigned long long)cycle_count,
                (double)cycle_count / 100000.0);

    if (vcd) { vcd->close(); delete vcd; }
    delete top;
    return exit_code;
}
