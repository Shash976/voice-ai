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
 *   0x20000000 - 0x20000FFF  TinyMAC accelerator registers (Stage 4)
 *
 * Usage:
 *   ./sim_picorv32 <firmware.bin> [--vcd <out.vcd>] [--mac-lanes N]
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
#define MAX_CYCLES     50000000ULL    /* 50M cycle timeout (accelerated: ~35K cycles/vector) */

/* Accelerator base address. MAC lanes are set at runtime via --mac-lanes N. */
#define ACCEL_BASE     0x20000000u

/* ── State ───────────────────────────────────────────────────────────────── */

static uint8_t  ram[RAM_SIZE];
static bool     sim_done  = false;
static int      exit_code = 0;
static uint64_t cycle_count = 0;
static int      mac_lanes = 8;   /* set by --mac-lanes at startup */

/* ── Accelerator emulation ───────────────────────────────────────────────── */

static struct {
    uint32_t in_ptr, wt_ptr, bias_ptr, mult_ptr, rshi_ptr, out_ptr;
    uint32_t dim_m, dim_k;
    uint32_t dim_kern, dim_inlen, dim_stride, dim_pad;
    uint32_t zp_in, zp_out, relu;
    uint32_t last_cyc;
} ar;

static uint64_t accel_done_at = 0;  /* simulation cycle when accel becomes idle */

/* Safe byte-level accessors — avoid strict-aliasing UB from casting ram (uint8_t[])
 * to int32_t* directly, which -O2 may miscompile. */
static inline int8_t  ram_r8 (uint32_t a)
{
    if (a >= RAM_SIZE) { fprintf(stderr, "[accel] OOB r8  0x%08x\n", a); return 0; }
    return (int8_t)ram[a];
}
static inline int32_t ram_r32(uint32_t a)
{
    if (a + 4 > RAM_SIZE) { fprintf(stderr, "[accel] OOB r32 0x%08x\n", a); return 0; }
    int32_t v; memcpy(&v, &ram[a], 4); return v;
}
static inline void ram_w8(uint32_t a, int8_t v)
{
    if (a >= RAM_SIZE) { fprintf(stderr, "[accel] OOB w8  0x%08x\n", a); return; }
    ram[a] = (uint8_t)v;
}

/* Mirrors tiny_vad_infer.c's requantize() exactly. */
static int32_t accel_requantize(int32_t x, int32_t q_mult, int32_t rshift_val)
{
    int64_t val = (int64_t)q_mult * (int64_t)x;
    val += (1LL << 30);
    val >>= 31;
    if (rshift_val > 0) {
        val += (1LL << (rshift_val - 1));
        val >>= rshift_val;
    } else if (rshift_val < 0) {
        val <<= (-rshift_val);
    }
    return (int32_t)val;
}

static int8_t accel_clamp(int32_t x)
{
    if (x >  127) return  127;
    if (x < -128) return -128;
    return (int8_t)x;
}

static void accel_execute(uint32_t cmd)
{
    int32_t in_zp  = (int32_t)ar.zp_in;
    int32_t out_zp = (int32_t)ar.zp_out;
    int     relu   = (int)ar.relu;
    uint64_t total_macs = 0;

    fprintf(stderr, "[accel] cmd=%u M=%u K=%u in=0x%x wt=0x%x bias=0x%x out=0x%x\n",
            cmd, ar.dim_m, ar.dim_k, ar.in_ptr, ar.wt_ptr, ar.bias_ptr, ar.out_ptr);

    if (cmd == 1u) {
        /* ── MATVEC (dense layer): out[M] = W[M][K] · in[K] ── */
        uint32_t M = ar.dim_m, K = ar.dim_k;
        for (uint32_t o = 0; o < M; o++) {
            int32_t acc = ram_r32(ar.bias_ptr + o * 4);
            for (uint32_t i = 0; i < K; i++) {
                int32_t x  = (int32_t)ram_r8(ar.in_ptr + i) - in_zp;
                int32_t wv = (int32_t)ram_r8(ar.wt_ptr + o * K + i);
                acc += x * wv;
            }
            int32_t qm = ram_r32(ar.mult_ptr + o * 4);
            int32_t qs = ram_r32(ar.rshi_ptr + o * 4);
            int32_t r  = accel_requantize(acc, qm, qs) + out_zp;
            if (relu && r < out_zp) r = out_zp;
            ram_w8(ar.out_ptr + o, accel_clamp(r));
        }
        total_macs = (uint64_t)M * K;

    } else if (cmd == 2u) {
        /* ── CONV1D: out[out_len][out_ch] ── */
        uint32_t out_ch = ar.dim_m,  in_ch  = ar.dim_k;
        uint32_t kern   = ar.dim_kern, inlen = ar.dim_inlen;
        uint32_t stride = ar.dim_stride, pad  = ar.dim_pad;
        if (stride == 0) { fprintf(stderr, "[accel] stride=0, skip\n"); return; }
        uint32_t out_len = (inlen + 2 * pad - kern) / stride + 1;

        for (uint32_t t = 0; t < out_len; t++) {
            for (uint32_t oc = 0; oc < out_ch; oc++) {
                int32_t acc = ram_r32(ar.bias_ptr + oc * 4);
                for (uint32_t ic = 0; ic < in_ch; ic++) {
                    for (uint32_t k = 0; k < kern; k++) {
                        int pos = (int)(t * stride + k) - (int)pad;
                        if (pos >= 0 && pos < (int)inlen) {
                            int32_t x  = (int32_t)ram_r8(ar.in_ptr  + (uint32_t)pos * in_ch + ic) - in_zp;
                            int32_t wv = (int32_t)ram_r8(ar.wt_ptr  + (oc * in_ch + ic) * kern + k);
                            acc += x * wv;
                        }
                    }
                }
                int32_t qm = ram_r32(ar.mult_ptr + oc * 4);
                int32_t qs = ram_r32(ar.rshi_ptr + oc * 4);
                int32_t r  = accel_requantize(acc, qm, qs) + out_zp;
                if (relu && r < out_zp) r = out_zp;
                ram_w8(ar.out_ptr + t * out_ch + oc, accel_clamp(r));
            }
        }
        total_macs = (uint64_t)out_len * out_ch * in_ch * kern;
    }

    uint64_t latency = (total_macs + (uint64_t)mac_lanes - 1) / (uint64_t)mac_lanes;
    ar.last_cyc   = (uint32_t)latency;
    accel_done_at = cycle_count + latency;
}

static uint32_t accel_read(uint32_t offset)
{
    switch (offset) {
    case 0x00: return (cycle_count < accel_done_at) ? 1u : 0u;  /* STATUS */
    case 0x44: return ar.last_cyc;                                /* LAST_CYC */
    default:   return 0u;
    }
}

static void accel_write(uint32_t offset, uint32_t val)
{
    switch (offset) {
    case 0x04: accel_execute(val); break;  /* CMD — triggers operation */
    case 0x08: ar.in_ptr    = val; break;
    case 0x0C: ar.wt_ptr    = val; break;
    case 0x10: ar.bias_ptr  = val; break;
    case 0x14: ar.mult_ptr  = val; break;
    case 0x18: ar.rshi_ptr  = val; break;
    case 0x1C: ar.out_ptr   = val; break;
    case 0x20: ar.dim_m     = val; break;
    case 0x24: ar.dim_k     = val; break;
    case 0x28: ar.dim_kern  = val; break;
    case 0x2C: ar.dim_inlen = val; break;
    case 0x30: ar.dim_stride = val; break;
    case 0x34: ar.dim_pad   = val; break;
    case 0x38: ar.zp_in     = val; break;
    case 0x3C: ar.zp_out    = val; break;
    case 0x40: ar.relu      = val; break;
    default: break;
    }
}

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
    if (addr >= ACCEL_BASE && addr < ACCEL_BASE + 0x1000u)
        return accel_read(addr - ACCEL_BASE);
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

    if (addr >= ACCEL_BASE && addr < ACCEL_BASE + 0x1000u) {
        /* Accelerator registers are 32-bit word writes (strb=0xF) */
        if (strb == 0xF)
            accel_write(addr - ACCEL_BASE, data);
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
        fprintf(stderr, "Usage: %s <firmware.bin> [--vcd <out.vcd>] [--mac-lanes N]\n", argv[0]);
        return 1;
    }
    const char *fw_path  = argv[1];
    const char *vcd_path = nullptr;
    for (int i = 2; i < argc; i++) {
        if (std::string(argv[i]) == "--vcd" && i + 1 < argc)
            vcd_path = argv[++i];
        else if (std::string(argv[i]) == "--mac-lanes" && i + 1 < argc)
            mac_lanes = std::atoi(argv[++i]);
    }
    if (mac_lanes < 1) mac_lanes = 1;
    fprintf(stderr, "[sim] mac_lanes=%d\n", mac_lanes);

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
        fprintf(stderr, "[sim] Done in %llu cycles (wall: ~%.1f ms at 100 MHz) mac_lanes=%d\n",
                (unsigned long long)cycle_count,
                (double)cycle_count / 100000.0,
                mac_lanes);

    if (vcd) { vcd->close(); delete vcd; }
    delete top;
    return exit_code;
}
