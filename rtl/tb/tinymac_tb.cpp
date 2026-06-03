/* tinymac_tb.cpp — self-checking Verilator testbench for tinymac_accel.
 *
 * Drives the synthesizable int8 matvec core and compares its int8 outputs,
 * bit-for-bit, against a software golden model.  The golden mirrors the RTL's
 * accumulation order (LANES-wide chunks, saturate-per-chunk) and reuses the
 * exact fixed-point requantize math from tiny_vad_infer.c / sim_main.cpp.
 *
 * Operands are served combinationally from C++ arrays, addressed by the
 * indices the core exposes (o_m, o_k_base) — the same 0-wait-state model the
 * Stage-4 testbench uses for main RAM.
 *
 * Build/run via rtl/tb/Makefile (LANES and ACC_W must match the RTL build).
 *
 * Exit code 0 = all cases bit-exact; non-zero = mismatch.
 */

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <vector>
#include <random>

#include "Vtinymac_accel.h"
#include "verilated.h"

/* These MUST match the -GLANES / -GACC_W passed to Verilator (see Makefile). */
#ifndef TB_LANES
#define TB_LANES 4
#endif
#ifndef TB_ACC_W
#define TB_ACC_W 24
#endif

/* ── Golden software model (mirrors the RTL exactly) ─────────────────────── */

static int32_t g_requantize(int32_t x, int32_t q_mult, int32_t shift)
{
    int64_t val = (int64_t)q_mult * (int64_t)x;
    val += (1LL << 30);
    val >>= 31;
    if (shift > 0) { val += (1LL << (shift - 1)); val >>= shift; }
    else if (shift < 0) { val <<= (-shift); }
    return (int32_t)val;
}

static int32_t g_saturate(int32_t x)
{
    if (TB_ACC_W >= 32) return x;
    const int32_t hi = (1 << (TB_ACC_W - 1)) - 1;
    const int32_t lo = -(1 << (TB_ACC_W - 1));
    if (x > hi) return hi;
    if (x < lo) return lo;
    return x;
}

static int8_t g_clamp(int32_t x)
{
    if (x >  127) return  127;
    if (x < -128) return -128;
    return (int8_t)x;
}

/* Full matvec golden for one output channel, RTL accumulation order. */
static int8_t golden_channel(int m, int K,
                             const std::vector<int8_t> &in,
                             const std::vector<int8_t> &W,
                             const std::vector<int32_t> &bias,
                             const std::vector<int32_t> &qmult,
                             const std::vector<int32_t> &rshift,
                             int in_zp, int out_zp, int relu)
{
    int32_t acc = bias[m];
    for (int k_base = 0; k_base < K; k_base += TB_LANES) {
        int32_t psum = 0;
        for (int lane = 0; lane < TB_LANES; lane++) {
            int k = k_base + lane;
            if (k < K)
                psum += ((int32_t)in[k] - in_zp) * (int32_t)W[(size_t)m * K + k];
        }
        acc = g_saturate(acc + psum);
    }
    int32_t r = g_requantize(acc, qmult[m], rshift[m]) + out_zp;
    if (relu && r < out_zp) r = out_zp;
    return g_clamp(r);
}

/* ── Operand packing for the LANES-wide chunk ports ──────────────────────── */

/* Pack up to 8 lanes into a 64-bit word (Verilator scalar port type for
 * LANES<=8). LANES>8 needs the wide (VlWide) port API — not supported here. */
static uint64_t pack_chunk(const int8_t *base, int idx0, int K, int total)
{
    static_assert(TB_LANES <= 8, "tb supports LANES<=8 (64-bit chunk port)");
    uint64_t w = 0;
    for (int lane = 0; lane < TB_LANES; lane++) {
        int k = idx0 + lane;
        uint8_t b = 0;
        if (k < K && (idx0 + lane) < total) b = (uint8_t)base[k];
        w |= (uint64_t)b << (8 * lane);
    }
    return w;
}

/* ── Test harness ────────────────────────────────────────────────────────── */

static Vtinymac_accel *dut;

struct Case {
    int M, K, in_zp, out_zp, relu;
    std::vector<int8_t>  in;     /* K */
    std::vector<int8_t>  W;      /* M*K */
    std::vector<int32_t> bias;   /* M */
    std::vector<int32_t> qmult;  /* M */
    std::vector<int32_t> rshift; /* M */
};

/* Run one case on the DUT; return number of mismatches. */
static int run_case(const Case &c, const char *name)
{
    /* reset */
    dut->rst_n = 0; dut->start = 0;
    dut->clk = 0; dut->eval();
    for (int i = 0; i < 4; i++) { dut->clk = 1; dut->eval(); dut->clk = 0; dut->eval(); }
    dut->rst_n = 1;

    /* latch config + pulse start */
    dut->cfg_m     = c.M;
    dut->cfg_k     = c.K;
    dut->cfg_in_zp = (uint32_t)c.in_zp;
    dut->cfg_out_zp= (uint32_t)c.out_zp;
    dut->cfg_relu  = c.relu;
    dut->start     = 1;

    std::vector<int8_t> got(c.M, 0);
    std::vector<bool>   seen(c.M, false);
    int collected = 0;
    int guard = 0;
    int last_cyc = 0;
    bool done_seen = false;
    const int GUARD_MAX = c.M * (c.K + 8) + 100;

    /* combinational operand serving + clock stepping.
     * Run until the FSM pulses `done` (one cycle after the last output) so we
     * can capture o_last_cycles — the real RTL cycle count for the op. */
    bool started = true;
    while (!done_seen && guard++ < GUARD_MAX) {
        /* serve operands for the indices the core currently presents */
        int m  = dut->o_m;
        int kb = dut->o_k_base;
        if (m >= 0 && m < c.M) {
            dut->i_in_chunk = pack_chunk(c.in.data(), kb, c.K, c.K);
            dut->i_wt_chunk = pack_chunk(c.W.data() + (size_t)m * c.K, kb, c.K, c.K);
            dut->i_bias   = (uint32_t)c.bias[m];
            dut->i_qmult  = (uint32_t)c.qmult[m];
            dut->i_rshift = (uint32_t)c.rshift[m];
        }
        dut->eval();   /* settle combinational operand path before posedge */

        /* posedge */
        dut->clk = 1; dut->eval();

        if (dut->o_out_valid) {
            int om = dut->o_out_m;
            if (om >= 0 && om < c.M && !seen[om]) {
                got[om] = (int8_t)dut->o_out_data;
                seen[om] = true;
                collected++;
            }
        }
        if (dut->done) { done_seen = true; last_cyc = dut->o_last_cycles; }

        /* negedge */
        dut->clk = 0; dut->eval();

        if (started) { dut->start = 0; started = false; }
    }

    /* compare against golden */
    int mismatches = 0;
    for (int m = 0; m < c.M; m++) {
        int8_t gold = golden_channel(m, c.K, c.in, c.W, c.bias, c.qmult, c.rshift,
                                     c.in_zp, c.out_zp, c.relu);
        if (!seen[m] || got[m] != gold) {
            if (mismatches < 8)
                fprintf(stderr, "  [%s] m=%d: dut=%d golden=%d seen=%d\n",
                        name, m, (int)got[m], (int)gold, (int)seen[m]);
            mismatches++;
        }
    }
    fprintf(stderr, "  [%s] M=%d K=%d relu=%d  %s  (last_cycles=%d, guard=%d)\n",
            name, c.M, c.K, c.relu,
            mismatches ? "MISMATCH" : "ok", last_cyc, guard);
    return mismatches;
}

/* Build a randomized case with controlled ranges. */
static Case make_random(std::mt19937 &rng, int M, int K, int relu)
{
    std::uniform_int_distribution<int> i8(-128, 127);
    std::uniform_int_distribution<int> zp(-128, 127);
    std::uniform_int_distribution<int> sh(-2, 12);
    /* Q31 multipliers in [2^30, 2^31): realistic requantize range. */
    std::uniform_int_distribution<int64_t> qm(1LL << 30, (1LL << 31) - 1);
    std::uniform_int_distribution<int>  bd(-100000, 100000);

    Case c; c.M = M; c.K = K; c.in_zp = zp(rng); c.out_zp = zp(rng); c.relu = relu;
    c.in.resize(K); for (auto &v : c.in) v = (int8_t)i8(rng);
    c.W.resize((size_t)M * K); for (auto &v : c.W) v = (int8_t)i8(rng);
    c.bias.resize(M);  for (auto &v : c.bias)  v = bd(rng);
    c.qmult.resize(M); for (auto &v : c.qmult) v = (int32_t)qm(rng);
    c.rshift.resize(M);for (auto &v : c.rshift) v = sh(rng);
    return c;
}

int main(int argc, char **argv)
{
    Verilated::commandArgs(argc, argv);
    dut = new Vtinymac_accel;

    fprintf(stderr, "tinymac_accel TB  (LANES=%d ACC_W=%d)\n", TB_LANES, TB_ACC_W);

    int total_mismatch = 0;
    std::mt19937 rng(0xC0FFEE);

    /* Real TinyVAD dense-layer shapes (matvec): FC0 64->32, FC1 32->2. */
    total_mismatch += run_case(make_random(rng, 32, 64, 1), "FC0_relu");
    total_mismatch += run_case(make_random(rng, 2,  32, 0), "FC1");

    /* Randomized stress: assorted M, K, relu, shift signs. */
    for (int t = 0; t < 40; t++) {
        std::uniform_int_distribution<int> md(1, 40), kd(1, 70), rd(0, 1);
        Case c = make_random(rng, md(rng), kd(rng), rd(rng));
        char nm[32]; snprintf(nm, sizeof nm, "rand%02d", t);
        total_mismatch += run_case(c, nm);
    }

    /* Edge cases: K not a multiple of LANES, K=1, M=1. */
    total_mismatch += run_case(make_random(rng, 1, 1, 0),  "K1_M1");
    total_mismatch += run_case(make_random(rng, 5, 7, 1),  "K7");
    total_mismatch += run_case(make_random(rng, 3, 3, 0),  "K3");

    fprintf(stderr, "\n==== %s : %d mismatch(es) ====\n",
            total_mismatch ? "FAIL" : "PASS", total_mismatch);
    delete dut;
    return total_mismatch ? 1 : 0;
}
