/* quartznet_tb.cpp — self-checking Verilator testbench for quartznet_accel.
 *
 * Unlike tinymac_tb.cpp, this testbench has NO hand-rolled golden model.  It
 * drives the accelerator from the real blobs emitted by
 *
 *     python3 sw/tinyml_reference/quartznet_ref.py --reduced
 *
 * and compares every output byte against quartznet_golden.bin.  That golden is
 * the untiled NumPy reference, which the C interpreter already reproduces
 * bit-exactly (27/27 + transcript), so a match here means the RTL agrees with
 * the model *and* that its tiling and address generation are exact — the same
 * argument quartznet_infer.c's header makes for the C tiling.
 *
 * The reduced config is the right target: 27 descriptors, all four ops, and it
 * is the only configuration that exercises OP_REQUANT at all.
 *
 * ── Flow ────────────────────────────────────────────────────────────────────
 *
 *   1. Parse quartznet_desc.bin (header + buffer records + 27 records), using
 *      the field layout documented at the top of quartznet_descriptors.py and
 *      implemented by quartznet_infer.c::unpack_record().
 *   2. Lay the activation arena out exactly as qn_load() does, and preload the
 *      weight/qparam blobs and the mel input through the memory model's
 *      simulation backdoor.
 *   3. For each descriptor in order: stage its fields into the MMIO register
 *      file, pulse CTRL.START, poll STATUS.DONE, then read the output slice
 *      back out of PSRAM and compare it byte for byte.
 *
 * Descriptors are run in sequence against one shared arena, exactly as
 * qn_run() does, so each one consumes its predecessors' real outputs rather
 * than a re-seeded buffer.
 *
 * Build/run via rtl/tb/Makefile:  make -f Makefile.quartznet LANES=32
 * Exit code 0 = every descriptor bit-exact.
 */

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "Vquartznet_accel.h"
#include "verilated.h"

#ifndef TB_LANES
#define TB_LANES 32
#endif
#ifndef TB_ACC_W
#define TB_ACC_W 32
#endif

/* ── Format constants (mirror quartznet_infer.h) ─────────────────────────── */

#define QN_DESC_MAGIC    0x54444E51u
#define QN_GOLDEN_MAGIC  0x4F474E51u
#define QN_DESC_BYTES    64
#define QN_HEADER_BYTES  64
#define QN_BUFREC_BYTES  8
#define QN_N_BUFFERS     5

#define QN_OP_DW       0
#define QN_OP_PW       1
#define QN_OP_ADD      2
#define QN_OP_REQUANT  3

#define QN_F_RELU      (1u << 0)

/* ── MMIO register map (mirrors the header comment in quartznet_accel.v) ─── */

enum {
    REG_CTRL = 0, REG_STATUS = 1, REG_CYCLES = 2,
    REG_OP = 3, REG_FLAGS = 4, REG_C_IN = 5, REG_C_OUT = 6, REG_K = 7,
    REG_STRIDE = 8, REG_DILATION = 9, REG_PAD = 10,
    REG_T_OUT = 11, REG_T_TILE = 12, REG_DW_CH_TILE = 13,
    REG_IN_ZP = 14, REG_OUT_ZP = 15, REG_RES_ZP = 16,
    REG_IN_BASE = 17, REG_IN_PITCH = 18, REG_IN_CBASE = 19,
    REG_OUT_BASE = 20, REG_OUT_PITCH = 21, REG_OUT_CBASE = 22,
    REG_RES_BASE = 23, REG_RES_PITCH = 24,
    REG_W_OFF = 25, REG_BIAS_OFF = 26, REG_QMULT_OFF = 27, REG_RSHIFT_OFF = 28,
    /* Increment 2: autonomous table walking. */
    REG_TABLE_BASE = 29, REG_W_BLOB_BASE = 30, REG_QP_BLOB_BASE = 31,
    REG_DESC_IDX = 32,
    /* D4: MEM_* host-access port (replaces bd_* for a real SoC). */
    REG_MEM_ADDR = 33, REG_MEM_CTRL = 34, REG_MEM_DATA = 35, REG_MEM_FILL = 36
};

/* CTRL bits */
#define CTRL_START        0x1u
#define CTRL_RUN_TABLE    0x2u
#define CTRL_SINGLE_STEP  0x8u
/* STATUS bits */
#define STATUS_BUSY           0x1u
#define STATUS_DONE           0x2u
#define STATUS_TABLE_DONE     0x4u
#define STATUS_ERR_BAD_IN_OFF 0x8u
/* MEM_CTRL bits */
#define MEM_CTRL_AUTOINC 0x2u

/* QSPI layout chosen by this testbench: the weight blob at 0, the qparam blob
 * after it.  The descriptor's w_off / *_off are offsets WITHIN their blob, so
 * the staged register value is base + offset. */
static uint32_t QSPI_W_BASE = 0;
static uint32_t QSPI_Q_BASE = 0;

/* ── Little-endian readers ───────────────────────────────────────────────── */

static uint32_t rd_u32(const uint8_t *p)
{
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8)
         | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}
static int s8(uint32_t v) { v &= 0xFFu; return (v >= 128u) ? (int)v - 256 : (int)v; }

struct Desc {
    int op, flags;
    int in_buf, out_buf, res_buf;
    int c_in, c_out, k, stride, dilation, pad;
    int in_stride, out_stride;
    int in_zp, out_zp, res_zp;
    uint32_t in_off, out_off, res_off;
    uint32_t w_off, bias_off, qmult_off, rshift_off;
    int layer_id, block_id;
};

/* Byte-for-byte the same unpack as quartznet_infer.c::unpack_record(). */
static Desc unpack(const uint8_t *rec)
{
    uint32_t w[16];
    for (int i = 0; i < 16; i++) w[i] = rd_u32(rec + i * 4);
    Desc d;
    d.op         = (int)( w[0]        & 0xFFu);
    d.flags      = (int)((w[0] >>  8) & 0xFFu);
    d.in_buf     = (int)((w[0] >> 16) & 0xFFu);
    d.out_buf    = (int)((w[0] >> 24) & 0xFFu);
    d.c_in       = (int)( w[1]        & 0xFFFFu);
    d.c_out      = (int)((w[1] >> 16) & 0xFFFFu);
    d.k          = (int)( w[2]        & 0xFFFFu);
    d.stride     = (int)((w[2] >> 16) & 0xFFu);
    d.dilation   = (int)((w[2] >> 24) & 0xFFu);
    d.pad        = (int)( w[3]        & 0xFFFFu);
    d.res_buf    = (int)((w[3] >> 16) & 0xFFu);
    d.in_stride  = (int)( w[4]        & 0xFFFFu);
    d.out_stride = (int)((w[4] >> 16) & 0xFFFFu);
    d.in_zp      = s8( w[5]);
    d.out_zp     = s8( w[5] >>  8);
    d.res_zp     = s8( w[5] >> 16);
    d.in_off     = w[6];
    d.out_off    = w[7];
    d.res_off    = w[8];
    d.w_off      = w[9];
    d.bias_off   = w[10];
    d.qmult_off  = w[11];
    d.rshift_off = w[12];
    d.layer_id   = (int)( w[14]        & 0xFFFFu);
    d.block_id   = (int)((w[14] >> 16) & 0xFFFFu);
    return d;
}

/* ── Reference tile schedule ──────────────────────────────────────────────
 * Produced by firmware/quartznet/qn_schedule, which drives the C interpreter's
 * own qn_walk() through qn_tile_hook.  Comparing against it means the RTL tile
 * grid is checked against the interpreter itself rather than a second copy of
 * the tiling rules — the same anti-drift argument sim/quartznet_cycles/ makes.
 *
 * The halo fields in particular CANNOT be validated from the activations: a
 * retained frame re-read gives the same value, so halo bookkeeping is invisible
 * downstream.  This is the only check that covers it. */
struct Tile {
    int desc, op, t0, t1, c0, c1, reduction, n_outputs;
    int in0, in1, new0, new1, first, last;
};

static std::vector<Tile> load_schedule(const std::string &path, bool &ok)
{
    std::vector<Tile> out;
    ok = false;
    FILE *f = fopen(path.c_str(), "r");
    if (!f) return out;
    char line[512];
    bool in_tiles = false, seen_hdr = false;
    while (fgets(line, sizeof line, f)) {
        if (line[0] == '#') { in_tiles = (strncmp(line, "#TILE", 5) == 0); seen_hdr = false; continue; }
        if (!in_tiles) continue;
        if (!seen_hdr) { seen_hdr = true; continue; }   /* column header row */
        Tile t;
        if (sscanf(line, "%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d",
                   &t.desc, &t.op, &t.t0, &t.t1, &t.c0, &t.c1,
                   &t.reduction, &t.n_outputs, &t.in0, &t.in1,
                   &t.new0, &t.new1, &t.first, &t.last) == 14)
            out.push_back(t);
    }
    fclose(f);
    ok = !out.empty();
    return out;
}

static std::vector<uint8_t> read_file(const std::string &path)
{
    FILE *f = fopen(path.c_str(), "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path.c_str()); exit(2); }
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    std::vector<uint8_t> v((size_t)n);
    if (n > 0 && fread(v.data(), 1, (size_t)n, f) != (size_t)n) {
        fprintf(stderr, "short read on %s\n", path.c_str()); exit(2);
    }
    fclose(f);
    return v;
}

/* ── DUT plumbing ────────────────────────────────────────────────────────── */

static Vquartznet_accel *dut;
static uint64_t g_cycles = 0;

static void tick()
{
    dut->clk = 1; dut->eval();
    dut->clk = 0; dut->eval();
    g_cycles++;
}

static void mmio_write(int addr, uint32_t val)
{
    dut->mmio_we    = 1;
    dut->mmio_addr  = (uint8_t)addr;
    dut->mmio_wdata = val;
    tick();
    dut->mmio_we = 0;
}

static uint32_t mmio_read(int addr)
{
    dut->mmio_addr = (uint8_t)addr;
    dut->eval();
    return dut->mmio_rdata;
}

static void bd_write(int sel, uint32_t addr, uint8_t data)
{
    dut->bd_en    = 1;
    dut->bd_sel   = (uint8_t)sel;
    dut->bd_we    = 1;
    dut->bd_addr  = addr;
    dut->bd_wdata = data;
    tick();
    dut->bd_en = 0;
    dut->bd_we = 0;
}

static uint8_t bd_read(int sel, uint32_t addr)
{
    dut->bd_en   = 1;
    dut->bd_sel  = (uint8_t)sel;
    dut->bd_we   = 0;
    dut->bd_addr = addr;
    dut->eval();
    uint8_t v = dut->bd_rdata;
    dut->bd_en = 0;
    return v;
}

/* ── Increment 2: autonomous table-walk test ─────────────────────────────────
 * Same blobs, same golden as the per-descriptor loop above, but instead of
 * staging each descriptor's fields over MMIO and pulsing CTRL.START once per
 * descriptor, this loads the raw descriptor-table bytes into QSPI and lets
 * quartznet_accel.v's own fetch FSM walk it autonomously: one TABLE_BASE /
 * W_BLOB_BASE / QP_BLOB_BASE / T_OUT write, one CTRL.RUN_TABLE pulse, one
 * STATUS.TABLE_DONE poll, then the same PSRAM-vs-golden compare per
 * descriptor. Runs on a fresh DUT instance so it starts from a clean reset,
 * independent of whatever state the per-descriptor loop left behind. */
static int run_table_test(uint32_t n_desc, uint32_t t_out, uint32_t in_ch,
                           uint32_t in_rows, uint32_t arena_bytes,
                           const std::vector<Desc> &desc,
                           const std::vector<uint8_t> &table,
                           const std::vector<uint8_t> &weights,
                           const std::vector<uint8_t> &qparams,
                           const std::vector<uint8_t> &input,
                           const uint8_t *g_data,
                           const std::vector<uint32_t> &g_off,
                           const std::vector<uint32_t> &g_size,
                           const uint32_t buf_ch[QN_N_BUFFERS],
                           const uint32_t buf_off[QN_N_BUFFERS])
{
    fprintf(stderr, "\n=== table-walk mode (LANES=%d ACC_W=%d) ===\n",
            TB_LANES, TB_ACC_W);

    uint32_t w_base = 0;
    uint32_t q_base = (w_base + (uint32_t)weights.size() + 3u) & ~3u;
    uint32_t t_base = (q_base + (uint32_t)qparams.size() + 3u) & ~3u;

    dut = new Vquartznet_accel;
    dut->clk = 0; dut->rst_n = 0;
    dut->mmio_we = 0; dut->mmio_addr = 0; dut->mmio_wdata = 0;
    dut->bd_en = 0; dut->bd_sel = 0; dut->bd_we = 0;
    dut->bd_addr = 0; dut->bd_wdata = 0;
    dut->eval();
    for (int i = 0; i < 8; i++) tick();
    dut->rst_n = 1;
    tick();

    for (size_t i = 0; i < weights.size(); i++)
        bd_write(0, w_base + (uint32_t)i, weights[i]);
    for (size_t i = 0; i < qparams.size(); i++)
        bd_write(0, q_base + (uint32_t)i, qparams[i]);
    for (size_t i = 0; i < table.size(); i++)
        bd_write(0, t_base + (uint32_t)i, table[i]);

    for (uint32_t i = 0; i < arena_bytes; i++) bd_write(1, i, 0);
    {
        uint32_t pitch = buf_ch[0];
        for (uint32_t t = 0; t < in_rows; t++)
            for (uint32_t c = 0; c < in_ch; c++)
                bd_write(1, buf_off[0] + t * pitch + c,
                         input[(size_t)t * in_ch + c]);
    }
    fprintf(stderr, "  preload done (table %zu B @ 0x%x)\n",
            table.size(), t_base);

    mmio_write(REG_T_OUT,        t_out);
    mmio_write(REG_TABLE_BASE,   t_base);
    mmio_write(REG_W_BLOB_BASE,  w_base);
    mmio_write(REG_QP_BLOB_BASE, q_base);

    /* BUF_A/BUF_B are ping-pong buffers that LATER descriptors legitimately
     * overwrite (that reuse is the whole point of the arena layout) -- so
     * unlike the single-descriptor loop above, we cannot wait for the whole
     * table to finish and then read every descriptor's slice back: by then
     * later descriptors have already clobbered earlier ones' output regions.
     * Snapshot and compare each descriptor's slice the moment DESC_IDX
     * shows it completed (BEFORE ticking again), same as the single-
     * descriptor loop does per-CTRL.START. */
    auto compare_one = [&](uint32_t i, int &pass, int &fail) {
        const Desc &d = desc[i];
        uint32_t out_pitch = buf_ch[d.out_buf];
        int bad = 0;
        int first_t = -1, first_c = -1, got_v = 0, exp_v = 0;
        if (g_size[i] != t_out * (uint32_t)d.c_out) {
            fprintf(stderr, "  [%2u] golden slice size %u != %u\n",
                    i, g_size[i], t_out * (uint32_t)d.c_out);
            bad++;
        } else {
            for (uint32_t t = 0; t < t_out && bad < 1000000; t++) {
                for (int c = 0; c < d.c_out; c++) {
                    uint32_t a = buf_off[d.out_buf] + t * out_pitch
                               + d.out_off + (uint32_t)c;
                    int got = (int8_t)bd_read(1, a);
                    int exp = (int8_t)g_data[g_off[i] + t * (uint32_t)d.c_out
                                             + (uint32_t)c];
                    if (got != exp) {
                        if (bad == 0) {
                            first_t = (int)t; first_c = c;
                            got_v = got; exp_v = exp;
                        }
                        bad++;
                    }
                }
            }
        }
        const char *opn = (d.op == QN_OP_DW) ? "dw" : (d.op == QN_OP_PW) ? "pw"
                        : (d.op == QN_OP_ADD) ? "add" : "requant";
        if (bad) {
            fail++;
            fprintf(stderr,
                    "  [%2u] %-7s MISMATCH %d/%u bytes "
                    "(first t=%d c=%d dut=%d golden=%d)\n",
                    i, opn, bad, g_size[i], first_t, first_c, got_v, exp_v);
        } else {
            pass++;
        }
    };

    uint64_t t_start = g_cycles;
    mmio_write(REG_CTRL, CTRL_RUN_TABLE);

    int pass = 0, fail = 0;
    uint32_t last_idx = 0;
    const uint64_t GUARD = 200000000ull;
    uint64_t guard = 0;
    bool timeout = false;
    for (;;) {
        uint32_t status = mmio_read(REG_STATUS);
        if (status & STATUS_TABLE_DONE) {
            compare_one(n_desc - 1, pass, fail);   /* the last descriptor */
            break;
        }
        uint32_t cur_idx = mmio_read(REG_DESC_IDX);
        if (cur_idx != last_idx) {
            compare_one(cur_idx - 1, pass, fail);  /* just-completed one */
            last_idx = cur_idx;
        }
        tick();
        if (++guard > GUARD) {
            fprintf(stderr, "  TABLE-WALK TIMEOUT after %llu cycles\n",
                    (unsigned long long)guard);
            timeout = true;
            break;
        }
    }
    if (timeout) fail++;
    uint64_t total_cycles = g_cycles - t_start;
    mmio_write(REG_STATUS, STATUS_TABLE_DONE);   /* W1C */

    if (mmio_read(REG_STATUS) & STATUS_ERR_BAD_IN_OFF)
        fprintf(stderr, "  err_bad_in_off latched -- a fetched descriptor had "
                         "in_off != 0, unsupported by the table walker\n");

    fprintf(stderr,
            "\n%u/%u descriptors bit-exact in table-walk mode "
            "(%llu total cycles%s)\n",
            (unsigned)pass, n_desc, (unsigned long long)total_cycles,
            timeout ? ", TIMEOUT" : "");

    delete dut;
    dut = nullptr;
    return fail ? 1 : 0;
}

/* ── D4: MEM_* host-access port + SINGLE_STEP regression ─────────────────────
 * Same blobs/golden as run_table_test() above, but preloads/reads back
 * through MEM_ADDR/MEM_CTRL/MEM_DATA/MEM_FILL (idx 33-36) instead of the
 * simulation-only bd_* backdoor, and drives the walk with CTRL.SINGLE_STEP
 * set -- exactly the sequence firmware/quartznet/qn_accel.c's
 * qn_run_hw_step() uses. Confirms MEM_* is a faithful bd_* replacement (every
 * other test in this file still uses bd_* unmodified) and that SINGLE_STEP
 * correctly gates the walker one descriptor at a time. Runs on a fresh DUT,
 * independent of the tests above. */

enum { MEMBANK_QSPI = 0, MEMBANK_PSRAM = 1 };

static void mem_write_byte(int bank, uint32_t addr, uint8_t val)
{
    mmio_write(REG_MEM_CTRL, (uint32_t)bank);
    mmio_write(REG_MEM_ADDR, addr);
    mmio_write(REG_MEM_DATA, val);
}

static uint8_t mem_read_byte(int bank, uint32_t addr)
{
    mmio_write(REG_MEM_CTRL, (uint32_t)bank);
    mmio_write(REG_MEM_ADDR, addr);
    return (uint8_t)mmio_read(REG_MEM_DATA);
}

/* Multi-cycle: MEM_FILL asserts STATUS.BUSY until the background bd_* loop
 * finishes (see quartznet_accel.v's S_MEMFILL). */
static void mem_fill(int bank, uint32_t addr, uint8_t val, uint32_t n)
{
    mmio_write(REG_MEM_CTRL, (uint32_t)bank);
    mmio_write(REG_MEM_ADDR, addr);
    mmio_write(REG_MEM_DATA, val);
    mmio_write(REG_MEM_FILL, n);
    /* fill_pulse (set this cycle) only reaches S_IDLE's dispatch on the NEXT
     * edge, so a status check with no intervening tick() would see stale
     * pre-transition BUSY=0 and return before the fill (or even its own
     * S_IDLE->S_MEMFILL transition) has happened -- do..while forces at
     * least one real edge before the first check. */
    do { tick(); } while (mmio_read(REG_STATUS) & STATUS_BUSY);
}

static int run_mem_port_test(uint32_t n_desc, uint32_t t_out, uint32_t in_ch,
                              uint32_t in_rows, uint32_t arena_bytes,
                              const std::vector<Desc> &desc,
                              const std::vector<uint8_t> &table,
                              const std::vector<uint8_t> &weights,
                              const std::vector<uint8_t> &qparams,
                              const std::vector<uint8_t> &input,
                              const uint8_t *g_data,
                              const std::vector<uint32_t> &g_off,
                              const std::vector<uint32_t> &g_size,
                              const uint32_t buf_ch[QN_N_BUFFERS],
                              const uint32_t buf_off[QN_N_BUFFERS])
{
    fprintf(stderr, "\n=== MEM_* port + SINGLE_STEP mode (LANES=%d ACC_W=%d) ===\n",
            TB_LANES, TB_ACC_W);

    uint32_t w_base = 0;
    uint32_t q_base = (w_base + (uint32_t)weights.size() + 3u) & ~3u;
    uint32_t t_base = (q_base + (uint32_t)qparams.size() + 3u) & ~3u;

    dut = new Vquartznet_accel;
    dut->clk = 0; dut->rst_n = 0;
    dut->mmio_we = 0; dut->mmio_addr = 0; dut->mmio_wdata = 0;
    dut->bd_en = 0; dut->bd_sel = 0; dut->bd_we = 0;
    dut->bd_addr = 0; dut->bd_wdata = 0;
    dut->eval();
    for (int i = 0; i < 8; i++) tick();
    dut->rst_n = 1;
    tick();

    /* ---- G1.4a: directed MMIO register read/write-back unit test ------- */
    mmio_write(REG_MEM_ADDR, 0x1234u);
    if (mmio_read(REG_MEM_ADDR) != 0x1234u) {
        fprintf(stderr, "  MEM_ADDR readback MISMATCH\n");
        delete dut; dut = nullptr; return 1;
    }
    mmio_write(REG_MEM_CTRL, 0x3u);
    if (mmio_read(REG_MEM_CTRL) != 0x3u) {
        fprintf(stderr, "  MEM_CTRL readback MISMATCH\n");
        delete dut; dut = nullptr; return 1;
    }
    mem_write_byte(MEMBANK_PSRAM, 100u, 0xABu);   /* no AUTOINC */
    if (mem_read_byte(MEMBANK_PSRAM, 100u) != 0xABu) {
        fprintf(stderr, "  MEM_DATA byte round-trip MISMATCH\n");
        delete dut; dut = nullptr; return 1;
    }
    /* AUTOINC: three consecutive MEM_DATA writes land at 200, 201, 202. */
    mmio_write(REG_MEM_CTRL, (uint32_t)MEMBANK_PSRAM | MEM_CTRL_AUTOINC);
    mmio_write(REG_MEM_ADDR, 200u);
    mmio_write(REG_MEM_DATA, 0x11u);
    mmio_write(REG_MEM_DATA, 0x22u);
    mmio_write(REG_MEM_DATA, 0x33u);
    bool autoinc_ok = (mem_read_byte(MEMBANK_PSRAM, 200u) == 0x11u)
                    && (mem_read_byte(MEMBANK_PSRAM, 201u) == 0x22u)
                    && (mem_read_byte(MEMBANK_PSRAM, 202u) == 0x33u);
    if (!autoinc_ok) {
        fprintf(stderr, "  MEM_DATA AUTOINC MISMATCH\n");
        delete dut; dut = nullptr; return 1;
    }
    fprintf(stderr, "  directed MMIO register unit test: ok\n");

    /* ---- Preload weights/qparams/table + mel input via MEM_*, not bd_* -- */
    mmio_write(REG_MEM_CTRL, (uint32_t)MEMBANK_QSPI | MEM_CTRL_AUTOINC);
    mmio_write(REG_MEM_ADDR, w_base);
    for (size_t i = 0; i < weights.size(); i++) mmio_write(REG_MEM_DATA, weights[i]);
    mmio_write(REG_MEM_CTRL, (uint32_t)MEMBANK_QSPI | MEM_CTRL_AUTOINC);
    mmio_write(REG_MEM_ADDR, q_base);
    for (size_t i = 0; i < qparams.size(); i++) mmio_write(REG_MEM_DATA, qparams[i]);
    mmio_write(REG_MEM_CTRL, (uint32_t)MEMBANK_QSPI | MEM_CTRL_AUTOINC);
    mmio_write(REG_MEM_ADDR, t_base);
    for (size_t i = 0; i < table.size(); i++) mmio_write(REG_MEM_DATA, table[i]);

    mem_fill(MEMBANK_PSRAM, 0, 0, arena_bytes);
    {
        uint32_t pitch = buf_ch[0];
        for (uint32_t t = 0; t < in_rows; t++) {
            /* AUTOINC only advances by 1 per write; a row may be narrower
             * than the buffer's pitch, so re-stage the address every row. */
            mmio_write(REG_MEM_CTRL, (uint32_t)MEMBANK_PSRAM | MEM_CTRL_AUTOINC);
            mmio_write(REG_MEM_ADDR, buf_off[0] + t * pitch);
            for (uint32_t c = 0; c < in_ch; c++)
                mmio_write(REG_MEM_DATA, input[(size_t)t * in_ch + c]);
        }
    }
    fprintf(stderr, "  preload done via MEM_* (table %zu B @ 0x%x)\n",
            table.size(), t_base);

    /* ---- SINGLE_STEP table walk, snapshotting each descriptor via MEM_* -- */
    mmio_write(REG_T_OUT,        t_out);
    mmio_write(REG_TABLE_BASE,   t_base);
    mmio_write(REG_W_BLOB_BASE,  w_base);
    mmio_write(REG_QP_BLOB_BASE, q_base);
    mmio_write(REG_CTRL, CTRL_RUN_TABLE | CTRL_SINGLE_STEP);

    int pass = 0, fail = 0;
    const uint64_t GUARD = 200000000ull;
    uint64_t guard = 0;
    bool timeout = false;

    for (uint32_t i = 0; i < n_desc && !timeout; i++) {
        while (!(mmio_read(REG_STATUS) & (STATUS_DONE | STATUS_TABLE_DONE))) {
            tick();
            if (++guard > GUARD) {
                fprintf(stderr, "  SINGLE_STEP TIMEOUT waiting on descriptor %u "
                                "(status=0x%x desc_idx=%u)\n",
                        i, mmio_read(REG_STATUS), mmio_read(REG_DESC_IDX));
                timeout = true; break;
            }
        }
        if (timeout) break;

        const Desc &d = desc[i];
        uint32_t out_pitch = buf_ch[d.out_buf];
        int bad = 0;
        if (g_size[i] != t_out * (uint32_t)d.c_out) {
            bad = 1;
        } else {
            for (uint32_t t = 0; t < t_out && bad < 1000000; t++)
                for (int c = 0; c < d.c_out; c++) {
                    uint32_t a = buf_off[d.out_buf] + t * out_pitch
                               + d.out_off + (uint32_t)c;
                    int got = (int8_t)mem_read_byte(MEMBANK_PSRAM, a);
                    int exp = (int8_t)g_data[g_off[i] + t * (uint32_t)d.c_out + (uint32_t)c];
                    if (got != exp) bad++;
                }
        }
        if (bad) { fail++; fprintf(stderr, "  [%2u] MEM_*+SINGLE_STEP MISMATCH %d bytes\n", i, bad); }
        else       pass++;

        /* W1C-clear DONE (and TABLE_DONE, harmless if unset): un-pauses the
         * walker for descriptor i+1, or is simply a no-op on the last one. */
        mmio_write(REG_STATUS, STATUS_DONE | STATUS_TABLE_DONE);
    }
    if (timeout) fail++;

    fprintf(stderr, "\n%u/%u descriptors bit-exact via MEM_*+SINGLE_STEP%s\n",
            (unsigned)pass, n_desc, timeout ? " (TIMEOUT)" : "");

    delete dut;
    dut = nullptr;
    return fail ? 1 : 0;
}

int main(int argc, char **argv)
{
    Verilated::commandArgs(argc, argv);

    std::string dir = "../../build/quartznet_reduced";
    std::string sched_path;
    for (int i = 1; i < argc; i++) {
        if (strncmp(argv[i], "--dir=", 6) == 0)   dir = argv[i] + 6;
        if (strncmp(argv[i], "--sched=", 8) == 0) sched_path = argv[i] + 8;
    }

    bool have_sched = false;
    std::vector<Tile> sched;
    if (!sched_path.empty()) {
        sched = load_schedule(sched_path, have_sched);
        if (!have_sched)
            fprintf(stderr, "warning: no tile schedule at %s — "
                            "tile-walk check skipped\n", sched_path.c_str());
    }

    std::vector<uint8_t> table   = read_file(dir + "/quartznet_desc.bin");
    std::vector<uint8_t> weights = read_file(dir + "/quartznet_weights.bin");
    std::vector<uint8_t> qparams = read_file(dir + "/quartznet_qparams.bin");
    std::vector<uint8_t> input   = read_file(dir + "/quartznet_input.bin");
    std::vector<uint8_t> golden  = read_file(dir + "/quartznet_golden.bin");

    /* ── Parse the descriptor table ─────────────────────────────────────── */
    uint32_t magic    = rd_u32(&table[0]);
    uint32_t n_desc   = rd_u32(&table[8]);
    uint32_t in_ch    = rd_u32(&table[32]);
    uint32_t t_tile   = rd_u32(&table[40]);
    uint32_t dw_ch_t  = rd_u32(&table[48]);
    uint32_t n_bufs   = rd_u32(&table[56]);
    uint32_t desc_off = rd_u32(&table[60]);
    if (magic != QN_DESC_MAGIC) { fprintf(stderr, "bad descriptor magic\n"); return 2; }
    if (n_bufs != QN_N_BUFFERS) { fprintf(stderr, "unexpected buffer count\n"); return 2; }

    uint32_t buf_ch[QN_N_BUFFERS], buf_rate[QN_N_BUFFERS];
    for (uint32_t b = 0; b < n_bufs; b++) {
        const uint8_t *p = &table[QN_HEADER_BYTES + b * QN_BUFREC_BYTES];
        buf_ch[b]   = rd_u32(p);
        buf_rate[b] = rd_u32(p + 4);
    }

    std::vector<Desc> desc;
    for (uint32_t i = 0; i < n_desc; i++)
        desc.push_back(unpack(&table[desc_off + i * QN_DESC_BYTES]));

    /* ── Parse the golden file (quartznet_ref.py::write_golden) ─────────── */
    uint32_t gmagic = rd_u32(&golden[0]);
    uint32_t g_n    = rd_u32(&golden[4]);
    uint32_t t_out  = rd_u32(&golden[8]);
    if (gmagic != QN_GOLDEN_MAGIC) { fprintf(stderr, "bad golden magic\n"); return 2; }
    if (g_n != n_desc) { fprintf(stderr, "golden/descriptor count mismatch\n"); return 2; }
    std::vector<uint32_t> g_off(g_n), g_size(g_n);
    for (uint32_t i = 0; i < g_n; i++) {
        g_off[i]  = rd_u32(&golden[32 + 8 * i]);
        g_size[i] = rd_u32(&golden[36 + 8 * i]);
    }
    const uint8_t *g_data = &golden[32 + 8 * g_n];

    /* ── Arena layout, exactly as qn_load() computes it ─────────────────── */
    uint32_t buf_off[QN_N_BUFFERS];
    uint32_t cur = 0;
    for (uint32_t b = 0; b < n_bufs; b++) {
        buf_off[b] = cur;
        cur += buf_ch[b] * buf_rate[b] * t_out;
    }
    uint32_t arena_bytes = cur;

    QSPI_W_BASE = 0;
    QSPI_Q_BASE = ((uint32_t)weights.size() + 3u) & ~3u;   /* keep words aligned */

    fprintf(stderr, "quartznet_accel TB  (LANES=%d ACC_W=%d)\n", TB_LANES, TB_ACC_W);
    fprintf(stderr, "  descriptors %u   T_out %u   T_TILE %u   DW_CH_TILE %u\n",
            n_desc, t_out, t_tile, dw_ch_t);
    fprintf(stderr, "  weights %zu B   qparams %zu B   arena %u B\n",
            weights.size(), qparams.size(), arena_bytes);

    /* ── Reset ──────────────────────────────────────────────────────────── */
    dut = new Vquartznet_accel;
    dut->clk = 0; dut->rst_n = 0;
    dut->mmio_we = 0; dut->mmio_addr = 0; dut->mmio_wdata = 0;
    dut->bd_en = 0; dut->bd_sel = 0; dut->bd_we = 0;
    dut->bd_addr = 0; dut->bd_wdata = 0;
    dut->eval();
    for (int i = 0; i < 8; i++) tick();
    dut->rst_n = 1;
    tick();

    /* ── Preload memory through the backdoor ────────────────────────────── */
    for (size_t i = 0; i < weights.size(); i++)
        bd_write(0, QSPI_W_BASE + (uint32_t)i, weights[i]);
    for (size_t i = 0; i < qparams.size(); i++)
        bd_write(0, QSPI_Q_BASE + (uint32_t)i, qparams[i]);

    /* Zero the arena, then load the mel input into BUF_IN with the buffer's
     * channel pitch — the copy qn_set_input() performs. */
    for (uint32_t i = 0; i < arena_bytes; i++) bd_write(1, i, 0);
    {
        uint32_t pitch = buf_ch[0];
        uint32_t rows  = buf_rate[0] * t_out;
        for (uint32_t t = 0; t < rows; t++)
            for (uint32_t c = 0; c < in_ch; c++)
                bd_write(1, buf_off[0] + t * pitch + c,
                         input[(size_t)t * in_ch + c]);
    }
    fprintf(stderr, "  preload done (%llu cycles)\n\n", (unsigned long long)g_cycles);

    /* ── Run every descriptor in order ──────────────────────────────────── */
    int pass = 0, fail = 0;
    uint64_t total_op_cycles = 0;
    size_t   total_tiles = 0;

    for (uint32_t i = 0; i < n_desc; i++) {
        const Desc &d = desc[i];

        uint32_t in_pitch  = buf_ch[d.in_buf];
        uint32_t out_pitch = buf_ch[d.out_buf];
        uint32_t res_pitch = buf_ch[d.res_buf];

        mmio_write(REG_OP,         (uint32_t)d.op);
        mmio_write(REG_FLAGS,      (uint32_t)d.flags);
        mmio_write(REG_C_IN,       (uint32_t)d.c_in);
        mmio_write(REG_C_OUT,      (uint32_t)d.c_out);
        mmio_write(REG_K,          (uint32_t)d.k);
        mmio_write(REG_STRIDE,     (uint32_t)d.stride);
        mmio_write(REG_DILATION,   (uint32_t)d.dilation);
        mmio_write(REG_PAD,        (uint32_t)d.pad);
        mmio_write(REG_T_OUT,      t_out);
        mmio_write(REG_T_TILE,     t_tile);
        mmio_write(REG_DW_CH_TILE, dw_ch_t);
        mmio_write(REG_IN_ZP,      (uint32_t)(int32_t)d.in_zp);
        mmio_write(REG_OUT_ZP,     (uint32_t)(int32_t)d.out_zp);
        mmio_write(REG_RES_ZP,     (uint32_t)(int32_t)d.res_zp);
        mmio_write(REG_IN_BASE,    buf_off[d.in_buf]);
        mmio_write(REG_IN_PITCH,   in_pitch);
        /* The C reads at `in_off % pitch`; stage it already reduced so the
         * hardware needs no divider. */
        mmio_write(REG_IN_CBASE,   d.in_off % in_pitch);
        mmio_write(REG_OUT_BASE,   buf_off[d.out_buf]);
        mmio_write(REG_OUT_PITCH,  out_pitch);
        mmio_write(REG_OUT_CBASE,  d.out_off);
        mmio_write(REG_RES_BASE,   buf_off[d.res_buf]);
        mmio_write(REG_RES_PITCH,  res_pitch);
        mmio_write(REG_W_OFF,      QSPI_W_BASE + d.w_off);
        mmio_write(REG_BIAS_OFF,   QSPI_Q_BASE + d.bias_off);
        mmio_write(REG_QMULT_OFF,  QSPI_Q_BASE + d.qmult_off);
        mmio_write(REG_RSHIFT_OFF, QSPI_Q_BASE + d.rshift_off);

        uint64_t t_start = g_cycles;
        mmio_write(REG_CTRL, 1u);

        /* Poll STATUS.DONE.  The bound is generous but finite so a hang is
         * reported as a failure rather than running forever. */
        const uint64_t GUARD = (uint64_t)t_out * d.c_out
                             * ((uint64_t)d.c_in + d.k + 64) * 64 + 1000000;
        uint64_t guard = 0;
        std::vector<Tile> seen_tiles;
        while (!(mmio_read(REG_STATUS) & 2u)) {
            tick();
            if (dut->o_tile_start) {
                Tile t;
                t.desc = (int)i;      t.op   = d.op;
                t.t0   = dut->o_t0;   t.t1   = dut->o_t1;
                t.c0   = dut->o_c0;   t.c1   = dut->o_c1;
                t.in0  = dut->o_in0;  t.in1  = dut->o_in1;
                t.new0 = dut->o_new0; t.new1 = dut->o_new1;
                t.first = dut->o_first_tile;
                t.last  = dut->o_last_tile;
                t.reduction = 0; t.n_outputs = 0;
                seen_tiles.push_back(t);
            }
            if (++guard > GUARD) {
                fprintf(stderr, "  [%2u] TIMEOUT after %llu cycles\n",
                        i, (unsigned long long)guard);
                break;
            }
        }
        bool timeout = (guard > GUARD);
        uint64_t op_cycles = g_cycles - t_start;
        total_op_cycles += op_cycles;
        mmio_write(REG_STATUS, 2u);   /* W1C the DONE flag */

        /* ── Compare the tile walk against the C interpreter's schedule ──── */
        int tile_bad = 0;
        if (have_sched) {
            std::vector<Tile> want;
            for (const Tile &t : sched)
                if (t.desc == (int)i) want.push_back(t);
            if (want.size() != seen_tiles.size()) {
                fprintf(stderr,
                        "  [%2u] TILE COUNT dut=%zu expected=%zu\n",
                        i, seen_tiles.size(), want.size());
                tile_bad++;
            } else {
                for (size_t j = 0; j < want.size(); j++) {
                    const Tile &a = seen_tiles[j], &b = want[j];
                    if (a.t0 != b.t0 || a.t1 != b.t1 || a.c0 != b.c0 ||
                        a.c1 != b.c1 || a.in0 != b.in0 || a.in1 != b.in1 ||
                        a.new0 != b.new0 || a.new1 != b.new1 ||
                        a.first != b.first || a.last != b.last) {
                        if (tile_bad == 0)
                            fprintf(stderr,
                                "  [%2u] TILE %zu dut=(t %d..%d c %d..%d in %d..%d "
                                "new %d..%d f%d l%d) exp=(t %d..%d c %d..%d "
                                "in %d..%d new %d..%d f%d l%d)\n",
                                i, j,
                                a.t0, a.t1, a.c0, a.c1, a.in0, a.in1,
                                a.new0, a.new1, a.first, a.last,
                                b.t0, b.t1, b.c0, b.c1, b.in0, b.in1,
                                b.new0, b.new1, b.first, b.last);
                        tile_bad++;
                    }
                }
            }
        }

        /* ── Compare the output slice against the golden ─────────────────── */
        int bad = 0;
        int first_t = -1, first_c = -1, got_v = 0, exp_v = 0;
        if (g_size[i] != t_out * (uint32_t)d.c_out) {
            fprintf(stderr, "  [%2u] golden slice size %u != %u\n",
                    i, g_size[i], t_out * (uint32_t)d.c_out);
            bad++;
        } else {
            for (uint32_t t = 0; t < t_out && bad < 1000000; t++) {
                for (int c = 0; c < d.c_out; c++) {
                    uint32_t a = buf_off[d.out_buf] + t * out_pitch
                               + d.out_off + (uint32_t)c;
                    int got = (int8_t)bd_read(1, a);
                    int exp = (int8_t)g_data[g_off[i] + t * (uint32_t)d.c_out
                                             + (uint32_t)c];
                    if (got != exp) {
                        if (bad == 0) {
                            first_t = (int)t; first_c = c;
                            got_v = got; exp_v = exp;
                        }
                        bad++;
                    }
                }
            }
        }

        const char *opn = (d.op == QN_OP_DW) ? "dw" : (d.op == QN_OP_PW) ? "pw"
                        : (d.op == QN_OP_ADD) ? "add" : "requant";
        if (bad || timeout || tile_bad) {
            fail++;
            fprintf(stderr,
                    "  [%2u] %-7s c_in=%-4d c_out=%-4d k=%-3d s=%d d=%d  "
                    "MISMATCH %d/%u bytes, %d bad tile(s) "
                    "(first t=%d c=%d dut=%d golden=%d)\n",
                    i, opn, d.c_in, d.c_out, d.k, d.stride, d.dilation,
                    bad, g_size[i], tile_bad, first_t, first_c, got_v, exp_v);
        } else {
            pass++;
            total_tiles += seen_tiles.size();
            fprintf(stderr,
                    "  [%2u] %-7s c_in=%-4d c_out=%-4d k=%-3d s=%d d=%d  ok  "
                    "(%2zu tiles, %llu cycles)\n",
                    i, opn, d.c_in, d.c_out, d.k, d.stride, d.dilation,
                    seen_tiles.size(), (unsigned long long)op_cycles);
        }
    }

    fprintf(stderr,
            "\n%u/%u descriptors bit-exact at LANES=%d  (%llu op cycles)\n",
            (unsigned)pass, n_desc, TB_LANES,
            (unsigned long long)total_op_cycles);
    if (have_sched)
        fprintf(stderr,
                "%zu/%zu compute tiles match quartznet_infer.c's walk "
                "(incl. halo spans)\n", total_tiles, sched.size());
    else
        fprintf(stderr, "tile-walk check SKIPPED (no --sched=)\n");
    fprintf(stderr, "==== %s : %d failing descriptor(s) (single-descriptor mode) ====\n",
            fail ? "FAIL" : "PASS", fail);

    delete dut;
    dut = nullptr;

    /* ── Increment 2: same blobs/golden, autonomous table-walk mode ──────── */
    uint32_t in_rows = buf_rate[0] * t_out;
    int table_fail = run_table_test(n_desc, t_out, in_ch, in_rows, arena_bytes,
                                     desc, table, weights, qparams, input,
                                     g_data, g_off, g_size, buf_ch, buf_off);
    fprintf(stderr, "==== %s : %d failing descriptor(s) (table-walk mode) ====\n",
            table_fail ? "FAIL" : "PASS", table_fail);

    /* ── D4: same blobs/golden, MEM_* port + SINGLE_STEP ─────────────────── */
    int mem_fail = run_mem_port_test(n_desc, t_out, in_ch, in_rows, arena_bytes,
                                      desc, table, weights, qparams, input,
                                      g_data, g_off, g_size, buf_ch, buf_off);
    fprintf(stderr, "==== %s : %d failing descriptor(s) (MEM_*+SINGLE_STEP mode) ====\n",
            mem_fail ? "FAIL" : "PASS", mem_fail);

    return (fail || table_fail || mem_fail) ? 1 : 0;
}
