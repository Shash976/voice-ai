/* sim_main_qn.cpp — Verilator testbench for PicoRV32 + QuartzNet firmware.
 *
 * Forked from sim/verilator/sim_main.cpp: the RAM/UART/EXIT skeleton (256 KB
 * flat RAM, 0x10000000 UART TX, 0x10000004 SIM_EXIT) and the main simulation
 * loop are unchanged, Verilating the SAME unmodified rtl/soc/picorv32_soc.v --
 * see docs/07c and the Stage 7 plan's "never touch picorv32_soc.v" rule.
 *
 * What's different from sim_main.cpp: the Stage-4 TinyMAC accel_execute()
 * emulation is replaced by a QuartzNet table-walk emulation that DELEGATES
 * every actual computation to the real interpreter (firmware/quartznet/
 * quartznet_infer.c's qn_load/qn_set_arena/qn_run_desc/qn_gather_output),
 * so this shim cannot silently drift from the golden model the way a
 * hand-rolled re-implementation could. It exists to develop and debug
 * qn_accel.c/qn_main.c at normal C++ debugging speed before RTL exists
 * (Stage 7 Gap 1 increment D4/D5 add the real Verilog and a MODE=rtl
 * counterpart in this same directory).
 *
 * Two logical memories replace the single flat RAM the accelerator sees:
 *   QSPI  read-only (from the CPU's perspective)   weights + qparams + table
 *   PSRAM read/write                                activation arena
 * mirroring rtl/accel/ext_mem_if.v's two independent 1 MB-default spaces.
 * They are preloaded directly from files (--qspi-weights/--qspi-qparams/
 * --qspi-table), not through MMIO -- 19+ MB cannot pass through a register
 * file one word at a time, and on real silicon QSPI is pre-programmed flash
 * the CPU never writes anyway (see qn_accel.h).
 *
 * The register file (0x20000000, mirrors quartznet_accel.v's map and
 * firmware/quartznet/qn_accel.h exactly, including the MEM_ and SINGLE_STEP
 * additions that land in real Verilog in increment D4) is emulated with the
 * same lazy "compute progress on next read/write" style sim_main.cpp's
 * accel_done_at already uses -- accel_progress() is called at the top of
 * both accel_read() and accel_write().
 *
 * CYCLES is derived from this loop's real cycle_count (start-of-table to
 * TABLE_DONE), so it reflects actual simulated firmware polling overhead --
 * but per the Stage 7 plan, only a MODE=rtl run against real Verilog
 * (increment D5) is a timing claim; this shim's descriptor latency model
 * (estimate_desc_latency) is a coarse placeholder, not sim/quartznet_cycles.
 *
 * Usage:
 *   ./sim_qn_shim <firmware.bin> --qspi-weights W.bin --qspi-qparams Q.bin \
 *       --qspi-table T.bin [--lanes N] [--qspi-bytes N] [--psram-bytes N] [--vcd out.vcd]
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cassert>
#include <string>
#include <vector>

#include "Vpicorv32_soc.h"
#include "verilated.h"
#include "verilated_vcd_c.h"

extern "C" {
#include "quartznet_infer.h"
}

/* ── Configuration ───────────────────────────────────────────────────────── */

#define RAM_SIZE       (256 * 1024)
#define UART_ADDR      0x10000000u
#define EXIT_ADDR      0x10000004u
#define MAX_CYCLES     2000000000ULL

#define ACCEL_BASE     0x20000000u

/* Register word indices -- mirrors firmware/quartznet/qn_accel.h exactly. */
enum {
    R_CTRL = 0, R_STATUS = 1, R_CYCLES = 2,
    R_T_OUT = 11,
    R_TABLE_BASE = 29, R_W_BLOB_BASE = 30, R_QP_BLOB_BASE = 31, R_DESC_IDX = 32,
    R_MEM_ADDR = 33, R_MEM_CTRL = 34, R_MEM_DATA = 35, R_MEM_FILL = 36
};

#define CTRL_RUN_TABLE    0x2u
#define CTRL_SINGLE_STEP  0x8u

#define STATUS_BUSY           0x1u
#define STATUS_DONE           0x2u
#define STATUS_TABLE_DONE     0x4u
#define STATUS_ERR_BAD_IN_OFF 0x8u

#define MEM_CTRL_AUTOINC 0x2u
#define BANK_QSPI  0
#define BANK_PSRAM 1

/* ── State ───────────────────────────────────────────────────────────────── */

static uint8_t  ram[RAM_SIZE];
static bool     sim_done  = false;
static int      exit_code = 0;
static uint64_t cycle_count = 0;
static int      lanes = 32;   /* --lanes N: only feeds estimate_desc_latency() */

static uint32_t qspi_bytes  = 24u << 20;   /* full 15x5 needs ~19.7 MB; default generous */
static uint32_t psram_bytes = 2u << 20;    /* 10 s utterance needs ~1.1 MB; default generous */
static std::vector<uint8_t> qspi;
static std::vector<uint8_t> psram;

/* Table-walk state */
static qn_model_t qn_model;
static bool     table_active  = false;
static bool     single_step   = false;
static bool     waiting_ack   = false;
static uint32_t table_cur     = 0;
static uint64_t table_next_at = 0;
static uint64_t table_start_cycle = 0;
static uint32_t status_reg    = 0;   /* sticky DONE/TABLE_DONE/ERR_BAD_IN_OFF */
static uint32_t cycles_reg    = 0;

/* Staged config registers */
static uint32_t reg_t_out = 0, reg_table_base = 0, reg_w_blob_base = 0, reg_qp_blob_base = 0;

/* MEM_* port state */
static uint32_t mem_addr = 0;
static int      mem_bank = BANK_PSRAM;
static bool     mem_autoinc = false;
static uint8_t  mem_fill_byte = 0;

static inline uint8_t *bank_ptr(int bank) { return (bank == BANK_QSPI) ? qspi.data() : psram.data(); }
static inline uint32_t bank_size(int bank) { return (bank == BANK_QSPI) ? qspi_bytes : psram_bytes; }

/* ── Accelerator emulation ───────────────────────────────────────────────── *
 * Coarse per-descriptor latency placeholder -- NOT sim/quartznet_cycles.
 * Same shape as sim_main.cpp's TinyMAC accel_execute() cost formula: outputs
 * times (reduction chunks over `lanes` plus a fixed per-output overhead). */
static uint64_t estimate_desc_latency(uint32_t i)
{
    if (i >= qn_model.n_desc) return 1;
    const qn_desc_t &d = qn_model.desc[i];
    uint64_t reduction = (d.op == QN_OP_DW) ? (uint64_t)d.k
                       : (d.op == QN_OP_PW) ? (uint64_t)d.c_in : 1;
    uint64_t chunks    = (reduction + (uint64_t)lanes - 1) / (uint64_t)lanes;
    uint64_t n_outputs = (uint64_t)qn_model.t_out * (uint64_t)d.c_out;
    return n_outputs * (chunks + 2) + 1;
}

static void accel_progress()
{
    if (!table_active || waiting_ack) return;
    if (cycle_count < table_next_at) return;

    int rc = qn_run_desc(&qn_model, (int)table_cur);
    if (rc != QN_OK)
        fprintf(stderr, "[qn-shim] qn_run_desc(%u) failed rc=%d\n", table_cur, rc);

    status_reg |= STATUS_DONE;
    bool is_last = (table_cur + 1 >= qn_model.n_desc);
    if (is_last) {
        status_reg |= STATUS_TABLE_DONE;
        table_active = false;
        cycles_reg   = (uint32_t)(cycle_count - table_start_cycle);
        return;
    }
    table_cur++;
    if (single_step) {
        waiting_ack = true;
    } else {
        table_next_at = cycle_count + estimate_desc_latency(table_cur);
    }
}

static uint32_t accel_read(uint32_t offset)
{
    accel_progress();
    switch (offset >> 2) {
    case R_STATUS: return status_reg | (table_active ? STATUS_BUSY : 0u);
    case R_CYCLES: return cycles_reg;
    case R_DESC_IDX: return table_cur;
    case R_MEM_DATA: {
        uint32_t v = 0;
        if (mem_addr + 4 <= bank_size(mem_bank))
            memcpy(&v, bank_ptr(mem_bank) + mem_addr, 4);
        if (mem_autoinc) mem_addr += 4;
        return v;
    }
    default: return 0u;
    }
}

static void accel_write(uint32_t offset, uint32_t val)
{
    accel_progress();
    switch (offset >> 2) {
    case R_CTRL:
        if (val & CTRL_RUN_TABLE) {
            int rc = qn_load(&qn_model,
                              qspi.data() + reg_table_base, qspi_bytes - reg_table_base,
                              (const int8_t *)(qspi.data() + reg_w_blob_base), qspi_bytes - reg_w_blob_base,
                              qspi.data() + reg_qp_blob_base, qspi_bytes - reg_qp_blob_base,
                              (int)reg_t_out);
            if (rc != QN_OK)
                fprintf(stderr, "[qn-shim] qn_load failed rc=%d\n", rc);
            qn_set_arena((int8_t *)psram.data(), psram_bytes);

            table_active      = true;
            single_step       = (val & CTRL_SINGLE_STEP) != 0;
            table_cur         = 0;
            waiting_ack       = false;
            status_reg       &= ~(STATUS_DONE | STATUS_TABLE_DONE | STATUS_ERR_BAD_IN_OFF);
            table_start_cycle = cycle_count;
            table_next_at     = cycle_count + estimate_desc_latency(0);
        }
        break;
    case R_STATUS:
        if (val & STATUS_DONE)           status_reg &= ~STATUS_DONE;
        if (val & STATUS_TABLE_DONE)     status_reg &= ~STATUS_TABLE_DONE;
        if (val & STATUS_ERR_BAD_IN_OFF) status_reg &= ~STATUS_ERR_BAD_IN_OFF;
        if (waiting_ack && !(status_reg & STATUS_DONE)) {
            waiting_ack   = false;
            table_next_at = cycle_count + estimate_desc_latency(table_cur);
        }
        break;
    case R_T_OUT:        reg_t_out        = val; break;
    case R_TABLE_BASE:    reg_table_base   = val; break;
    case R_W_BLOB_BASE:   reg_w_blob_base  = val; break;
    case R_QP_BLOB_BASE:  reg_qp_blob_base = val; break;
    case R_MEM_ADDR: mem_addr = val; break;
    case R_MEM_CTRL:
        mem_bank    = (int)(val & 1u);
        mem_autoinc = (val & MEM_CTRL_AUTOINC) != 0;
        break;
    case R_MEM_DATA: {
        mem_fill_byte = (uint8_t)(val & 0xFFu);
        if (mem_addr + 4 <= bank_size(mem_bank))
            memcpy(bank_ptr(mem_bank) + mem_addr, &val, 4);
        if (mem_autoinc) mem_addr += 4;
        break;
    }
    case R_MEM_FILL: {
        uint8_t *b = bank_ptr(mem_bank);
        uint32_t sz = bank_size(mem_bank);
        for (uint32_t i = 0; i < val && mem_addr + i < sz; i++)
            b[mem_addr + i] = mem_fill_byte;
        break;
    }
    default: break;
    }
}

/* ── Firmware / QSPI image loaders ───────────────────────────────────────── */

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

/* Loads weights at byte 0, qparams 4-byte-aligned right after, table
 * 4-byte-aligned right after that -- the same convention rtl/tb/quartznet_tb.cpp
 * and firmware/quartznet/qn_main.c both use, derived here from the actual file
 * sizes (which must equal the table header's weight_bytes/qparam_bytes -- true
 * by construction, since quartznet_ref.py emits all three from one run). */
static void load_qspi_image(const char *w_path, const char *q_path, const char *t_path)
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
    memcpy(qspi.data() + w_base, weights.data(), weights.size());
    memcpy(qspi.data() + q_base, qparams.data(), qparams.size());
    memcpy(qspi.data() + t_base, table.data(), table.size());
    fprintf(stderr, "[sim] QSPI image: weights %zu B @ 0x%x, qparams %zu B @ 0x%x, "
                    "table %zu B @ 0x%x\n",
            weights.size(), w_base, qparams.size(), q_base, table.size(), t_base);
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
        if (strb == 0xF) accel_write(addr - ACCEL_BASE, data);
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

/* ── Main ────────────────────────────────────────────────────────────────── */

int main(int argc, char **argv)
{
    if (argc < 2) {
        fprintf(stderr,
                "Usage: %s <firmware.bin> --qspi-weights W.bin --qspi-qparams Q.bin "
                "--qspi-table T.bin [--lanes N] [--qspi-bytes N] [--psram-bytes N] "
                "[--vcd out.vcd]\n", argv[0]);
        return 1;
    }
    const char *fw_path  = argv[1];
    const char *vcd_path = nullptr;
    const char *w_path = nullptr, *q_path = nullptr, *t_path = nullptr;
    for (int i = 2; i < argc; i++) {
        std::string a = argv[i];
        if (a == "--vcd" && i + 1 < argc)               vcd_path = argv[++i];
        else if (a == "--lanes" && i + 1 < argc)        lanes = std::atoi(argv[++i]);
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
    if (lanes < 1) lanes = 1;

    qspi.assign(qspi_bytes, 0);
    psram.assign(psram_bytes, 0);
    load_qspi_image(w_path, q_path, t_path);
    load_firmware(fw_path);

    Verilated::commandArgs(argc, argv);
    Verilated::traceEverOn(vcd_path != nullptr);

    auto *top = new Vpicorv32_soc;

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
    for (int i = 0; i < 16; i++) {
        top->clk = !top->clk;
        top->eval();
        if (vcd) vcd->dump((vluint64_t)(cycle_count * 10 + (top->clk ? 5 : 0)));
    }
    top->resetn = 1;
    fprintf(stderr, "[sim] Reset released -- starting simulation\n");

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
