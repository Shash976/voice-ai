/* sim_main_qn_rtl.cpp — Verilator testbench for qn_soc (real quartznet_accel.v
 * wired onto the PicoRV32 bus), Stage 7 Gap 1 increment D5's MODE=rtl.
 *
 * Forked from sim_main_qn.cpp's RAM/UART/EXIT skeleton and main loop, but
 * with NO accelerator emulation at all: qn_soc.v's own address decode
 * (rtl/soc/qn_soc.v) makes real quartznet_accel.v answer every transaction
 * in the 0x2000_0000-0x2000_0FFF window directly, so this file's mem_read/
 * mem_write never even see those addresses -- qn_soc.v's own mem_valid stays
 * deasserted for them. This is the real fidelity claim MODE=shim's C++
 * emulation explicitly is NOT: qn_firmware.bin here talks to actual Verilog.
 *
 * QSPI/PSRAM preload goes through Vqn_soc's own top-level bd_* ports
 * (passed straight through from qn_soc.v to quartznet_accel.v's simulation
 * backdoor), one byte per cycle -- same mechanism rtl/tb/quartznet_tb.cpp
 * already uses, just driven here instead of from that testbench.
 *
 * Usage:
 *   ./sim_qn_rtl <firmware.bin> --qspi-weights W.bin --qspi-qparams Q.bin \
 *       --qspi-table T.bin [--qspi-bytes N] [--psram-bytes N] [--vcd out.vcd]
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "Vqn_soc.h"
#include "verilated.h"
#include "verilated_vcd_c.h"

#define RAM_SIZE    (256 * 1024)
#define UART_ADDR   0x10000000u
#define EXIT_ADDR   0x10000004u
// Cycle count scales roughly linearly with T_OUT (~7.9M cycles/output frame,
// measured: CONFIG=full T_OUT=70 -> 554,542,940 cycles), since ext_mem_if.v
// re-streams weights per T_TILE=32-frame chunk. The old 2,000,000,000 cap
// (fine for T_OUT<=70, the repo-wide default used elsewhere for golden/
// firmware generation) silently truncates any longer run with a TIMEOUT and
// no transcript once T_OUT exceeds ~250. Raised to cover the full range
// CONFIG=full's default 1MB PSRAM can hold (T_OUT up to 475, ~9.5s of
// audio, needing ~3.76B cycles by the same linear estimate) with headroom.
#define MAX_CYCLES  6000000000ULL

static uint8_t  ram[RAM_SIZE];
static bool     sim_done  = false;
static int      exit_code = 0;
static uint64_t cycle_count = 0;

static Vqn_soc *top;

static void tick()
{
    top->clk = 1; top->eval();
    top->clk = 0; top->eval();
}

static void bd_write(int sel, uint32_t addr, uint8_t data)
{
    top->bd_en = 1; top->bd_sel = (uint8_t)sel; top->bd_we = 1;
    top->bd_addr = addr; top->bd_wdata = data;
    tick();
    top->bd_en = 0; top->bd_we = 0;
}

static std::vector<uint8_t> read_file(const char *path)
{
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "ERROR: cannot open %s\n", path); exit(1); }
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    std::vector<uint8_t> v((size_t)n);
    if (n > 0 && fread(v.data(), 1, (size_t)n, f) != (size_t)n) {
        fprintf(stderr, "short read on %s\n", path); exit(1);
    }
    fclose(f);
    return v;
}

static void load_firmware(const char *path)
{
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "ERROR: cannot open firmware: %s\n", path); exit(1); }
    memset(ram, 0, sizeof(ram));
    size_t n = fread(ram, 1, RAM_SIZE, f);
    fclose(f);
    fprintf(stderr, "[sim] Loaded %zu bytes from %s\n", n, path);
}

/* Same weights/qparams/table QSPI layout convention as sim_main_qn.cpp and
 * firmware/quartznet/qn_main.c: weights at 0, qparams 4-byte-aligned right
 * after, table 4-byte-aligned right after that. */
static void load_qspi_image(const char *w_path, const char *q_path, const char *t_path,
                             uint32_t qspi_bytes)
{
    auto weights = read_file(w_path);
    auto qparams = read_file(q_path);
    auto table   = read_file(t_path);

    uint32_t w_base = 0;
    uint32_t q_base = (w_base + (uint32_t)weights.size() + 3u) & ~3u;
    uint32_t t_base = (q_base + (uint32_t)qparams.size() + 3u) & ~3u;
    if (t_base + table.size() > qspi_bytes) {
        fprintf(stderr, "ERROR: qspi image (%u B) exceeds --qspi-bytes=%u\n",
                (unsigned)(t_base + table.size()), qspi_bytes);
        exit(1);
    }
    for (size_t i = 0; i < weights.size(); i++) bd_write(0, w_base + (uint32_t)i, weights[i]);
    for (size_t i = 0; i < qparams.size(); i++) bd_write(0, q_base + (uint32_t)i, qparams[i]);
    for (size_t i = 0; i < table.size(); i++)   bd_write(0, t_base + (uint32_t)i, table[i]);
    fprintf(stderr, "[sim] QSPI image (RTL bd_* preload): weights %zu B @ 0x%x, "
                    "qparams %zu B @ 0x%x, table %zu B @ 0x%x\n",
            weights.size(), w_base, qparams.size(), q_base, table.size(), t_base);
}

static void zero_psram(uint32_t psram_bytes)
{
    for (uint32_t i = 0; i < psram_bytes; i++) bd_write(1, i, 0);
}

static uint32_t mem_read(uint32_t addr)
{
    if (addr + 4 <= RAM_SIZE) {
        uint32_t v;
        memcpy(&v, &ram[addr], 4);
        return v;
    }
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
    switch (addr) {
    case UART_ADDR:
        if (strb & 0x1) { putchar((int)(data & 0xFF)); fflush(stdout); }
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

int main(int argc, char **argv)
{
    if (argc < 2) {
        fprintf(stderr,
                "Usage: %s <firmware.bin> --qspi-weights W.bin --qspi-qparams Q.bin "
                "--qspi-table T.bin [--qspi-bytes N] [--psram-bytes N] [--vcd out.vcd]\n",
                argv[0]);
        return 1;
    }
    const char *fw_path  = argv[1];
    const char *vcd_path = nullptr;
    const char *w_path = nullptr, *q_path = nullptr, *t_path = nullptr;
    uint32_t qspi_bytes = 1u << 20, psram_bytes = 1u << 20;
    for (int i = 2; i < argc; i++) {
        std::string a = argv[i];
        if (a == "--vcd" && i + 1 < argc)               vcd_path = argv[++i];
        else if (a == "--qspi-bytes" && i + 1 < argc)   qspi_bytes = (uint32_t)std::strtoul(argv[++i], nullptr, 0);
        else if (a == "--psram-bytes" && i + 1 < argc)  psram_bytes = (uint32_t)std::strtoul(argv[++i], nullptr, 0);
        else if (a == "--qspi-weights" && i + 1 < argc) w_path = argv[++i];
        else if (a == "--qspi-qparams" && i + 1 < argc) q_path = argv[++i];
        else if (a == "--qspi-table" && i + 1 < argc)   t_path = argv[++i];
    }
    if (!w_path || !q_path || !t_path) {
        fprintf(stderr, "ERROR: --qspi-weights/--qspi-qparams/--qspi-table are required\n");
        return 1;
    }

    load_firmware(fw_path);

    Verilated::commandArgs(argc, argv);
    Verilated::traceEverOn(vcd_path != nullptr);

    top = new Vqn_soc;

    VerilatedVcdC *vcd = nullptr;
    if (vcd_path) {
        vcd = new VerilatedVcdC;
        top->trace(vcd, 99);
        vcd->open(vcd_path);
    }

    top->clk = 0;
    top->resetn = 0;
    top->mem_ready = 0;
    top->mem_rdata = 0;
    top->bd_en = 0; top->bd_sel = 0; top->bd_we = 0;
    top->bd_addr = 0; top->bd_wdata = 0;
    for (int i = 0; i < 16; i++) {
        top->clk = !top->clk;
        top->eval();
        if (vcd) vcd->dump((vluint64_t)(cycle_count * 10 + (top->clk ? 5 : 0)));
    }
    top->resetn = 1;
    tick();

    zero_psram(psram_bytes);
    load_qspi_image(w_path, q_path, t_path, qspi_bytes);

    fprintf(stderr, "[sim] Reset released -- starting simulation (RTL mode)\n");

    while (!sim_done && cycle_count < MAX_CYCLES) {
        top->clk = 0;
        top->eval();
        if (vcd) vcd->dump((vluint64_t)(cycle_count * 10));

        top->mem_ready = 0;
        if (top->mem_valid) {
            if (top->mem_wstrb)
                mem_write(top->mem_addr, top->mem_wdata, (uint8_t)top->mem_wstrb);
            else
                top->mem_rdata = mem_read(top->mem_addr);
            top->mem_ready = 1;
        }
        top->eval();

        top->clk = 1;
        top->eval();
        if (vcd) vcd->dump((vluint64_t)(cycle_count * 10 + 5));

        cycle_count++;

        if (top->trap) {
            fprintf(stderr, "[sim] CPU TRAP at cycle %llu\n", (unsigned long long)cycle_count);
            break;
        }
    }

    if (cycle_count >= MAX_CYCLES)
        fprintf(stderr, "[sim] TIMEOUT after %llu cycles\n", (unsigned long long)cycle_count);
    else
        fprintf(stderr, "[sim] Done in %llu cycles\n", (unsigned long long)cycle_count);

    if (vcd) { vcd->close(); delete vcd; }
    delete top;
    return exit_code;
}
