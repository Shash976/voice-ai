/* test_quartznet_host.c
 *
 * Compares the C interpreter in quartznet_infer.c against the NumPy golden
 * model, per descriptor, BYTE FOR BYTE.
 *
 * Build:   make host
 * Run:     ./test_quartznet_host [dir ...]      (default: both build dirs)
 *
 * Regenerate the goldens first:
 *     python sw/tinyml_reference/quartznet_ref.py --reduced
 *     python sw/tinyml_reference/quartznet_ref.py
 *
 * ── Why this is bit-exact and test_infer_host.c is not ───────────────────────
 *
 * The TinyVAD host test allows +/-3 LSB.  That tolerance exists because it
 * compares against TFLITE, whose int32 accumulation runs in a different loop
 * order than tiny_vad_infer.c's (oc-first vs t-first); the drift is real but
 * benign (see the header comment of test_infer_host.c).
 *
 * Nothing like that applies here.  The comparison is C-vs-NumPy, both walking
 * the same descriptor list in the same order, both accumulating in exact
 * integer arithmetic (int32 in C, int64 in NumPy, and every accumulator is
 * proven to fit int32 — peak |acc| measured at 56,098,816, 2.6% of int32).
 * Tiling reorders nothing: each output element is an independent reduction.
 * So the only correct tolerance is ZERO.  A mismatch is a bug in the C, never
 * a reason to widen this file.
 */

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "quartznet_infer.h"

/* ── Static blobs (no dynamic allocation, matching house style) ─────────────── */

#define MAX_TABLE     (64u * 1024u)
#define MAX_WEIGHTS   (20u * 1024u * 1024u)
#define MAX_QPARAMS   ( 2u * 1024u * 1024u)
#define MAX_INPUT     ( 1u * 1024u * 1024u)
#define MAX_GOLDEN    ( 8u * 1024u * 1024u)
#define MAX_GATHER    ( 1u * 1024u * 1024u)
#define MAX_TEXT      ( 8u * 1024u)

static uint32_t table_blob32 [MAX_TABLE   / 4];
static int8_t   weight_blob  [MAX_WEIGHTS];
static uint32_t qparam_blob32[MAX_QPARAMS / 4];
static int8_t   input_blob   [MAX_INPUT];
static uint32_t golden_blob32[MAX_GOLDEN  / 4];
static int8_t   gather_buf   [MAX_GATHER];
static char     text_buf     [MAX_TEXT];

static qn_model_t model;

/* ── Helpers ────────────────────────────────────────────────────────────────── */

static long load_file(const char *dir, const char *name, void *dst, size_t cap)
{
    char path[1024];
    snprintf(path, sizeof path, "%s/%s", dir, name);
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "  cannot open %s\n", path); return -1; }
    long n = (long)fread(dst, 1, cap, f);
    int over = !feof(f);
    fclose(f);
    if (over) { fprintf(stderr, "  %s exceeds %zu B buffer\n", path, cap); return -1; }
    return n;
}

static uint32_t rd32(const uint8_t *p)
{
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8)
         | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

/* ── One configuration ──────────────────────────────────────────────────────── */

static int run_config(const char *dir)
{
    const uint8_t *table  = (const uint8_t *)table_blob32;
    const uint8_t *qparam = (const uint8_t *)qparam_blob32;
    const uint8_t *golden = (const uint8_t *)golden_blob32;

    printf("\n=== %s ===\n", dir);

    long tn = load_file(dir, "quartznet_desc.bin",    table_blob32,  MAX_TABLE);
    long wn = load_file(dir, "quartznet_weights.bin", weight_blob,   MAX_WEIGHTS);
    long qn = load_file(dir, "quartznet_qparams.bin", qparam_blob32, MAX_QPARAMS);
    long in = load_file(dir, "quartznet_input.bin",   input_blob,    MAX_INPUT);
    long gn = load_file(dir, "quartznet_golden.bin",  golden_blob32, MAX_GOLDEN);
    if (tn < 0 || wn < 0 || qn < 0 || in < 0 || gn < 0) return -1;

    /* golden header: magic, n_desc, t_out, t_in, n_classes, text_len, act_bytes, 0 */
    if (gn < 32 || rd32(golden) != QN_GOLDEN_MAGIC) {
        fprintf(stderr, "  bad golden magic\n"); return -1;
    }
    uint32_t g_ndesc = rd32(golden +  4);
    uint32_t g_tout  = rd32(golden +  8);
    uint32_t g_tin   = rd32(golden + 12);
    uint32_t g_tlen  = rd32(golden + 20);
    uint32_t g_abyte = rd32(golden + 24);
    const uint8_t *g_index = golden + 32;
    const uint8_t *g_data  = g_index + (size_t)g_ndesc * 8;
    const char    *g_text  = (const char *)(g_data + g_abyte);

    int rc = qn_load(&model, table, (size_t)tn,
                     weight_blob, (size_t)wn, qparam, (size_t)qn, (int)g_tout);
    if (rc != QN_OK) { fprintf(stderr, "  qn_load failed: %d\n", rc); return -1; }
    if (model.n_desc != g_ndesc) {
        fprintf(stderr, "  descriptor count %u != golden %u\n",
                model.n_desc, g_ndesc); return -1;
    }
    if ((uint32_t)((int)model.buf_rate[QN_BUF_IN] * model.t_out) != g_tin) {
        fprintf(stderr, "  input frame count mismatch\n"); return -1;
    }

    printf("  descriptors %u   T_out %u   T_in %u   T_TILE %u   DW_CH_TILE %u\n",
           model.n_desc, g_tout, g_tin, model.t_tile, model.dw_ch_tile);
    printf("  weights %u B   qparams %u B\n", model.weight_bytes, model.qparam_bytes);

    rc = qn_set_input(&model, input_blob);
    if (rc != QN_OK) { fprintf(stderr, "  qn_set_input failed: %d\n", rc); return -1; }

    int pass = 0, fail = 0;
    int op_pass[4] = {0, 0, 0, 0}, op_seen[4] = {0, 0, 0, 0};

    for (uint32_t i = 0; i < model.n_desc; i++) {
        rc = qn_run_desc(&model, (int)i);
        if (rc != QN_OK) { fprintf(stderr, "  desc %u failed: %d\n", i, rc); return -1; }

        uint32_t off = rd32(g_index + (size_t)i * 8);
        uint32_t len = rd32(g_index + (size_t)i * 8 + 4);
        uint32_t want = (uint32_t)model.t_out * (uint32_t)model.desc[i].c_out;
        if (len != want || (size_t)len > MAX_GATHER) {
            fprintf(stderr, "  desc %u golden size %u != expected %u\n", i, len, want);
            return -1;
        }
        qn_gather_output(&model, (int)i, gather_buf);

        int op = model.desc[i].op;
        if (op >= 0 && op < 4) op_seen[op]++;

        if (memcmp(gather_buf, g_data + off, len) == 0) {
            pass++;
            if (op >= 0 && op < 4) op_pass[op]++;
        } else {
            fail++;
            if (fail <= 5) {
                /* first differing element, for debugging */
                uint32_t j = 0;
                while (j < len && gather_buf[j] == (int8_t)g_data[off + j]) j++;
                int c_out = model.desc[i].c_out;
                fprintf(stderr,
                        "  MISMATCH desc %u (layer %d block %d op %d) at "
                        "t=%u c=%u : C %d vs golden %d\n",
                        i, model.desc[i].layer_id, model.desc[i].block_id,
                        model.desc[i].op, j / (unsigned)c_out, j % (unsigned)c_out,
                        (int)gather_buf[j], (int)(int8_t)g_data[off + j]);
            }
        }
    }

    int nsym = qn_ctc_greedy(&model, text_buf, (int)sizeof text_buf);
    int text_ok = (nsym >= 0 && (uint32_t)nsym == g_tlen
                   && memcmp(text_buf, g_text, g_tlen) == 0);

    static const char *opn[4] = {"dw", "pw", "add", "requant"};
    printf("  per-op   :");
    for (int o = 0; o < 4; o++)
        if (op_seen[o]) printf("  %s %d/%d", opn[o], op_pass[o], op_seen[o]);
    printf("\n");
    printf("  activations: %d/%u descriptors bit-exact\n", pass, model.n_desc);
    printf("  transcript : %s  (%d symbols)\n", text_ok ? "MATCH" : "MISMATCH", nsym);
    if (!text_ok) {
        fprintf(stderr, "    C      : \"%s\"\n", text_buf);
        fprintf(stderr, "    golden : \"%.*s\"\n", (int)g_tlen, g_text);
    }

    return (fail == 0 && text_ok) ? 0 : -1;
}

/* ── main ───────────────────────────────────────────────────────────────────── */

int main(int argc, char **argv)
{
    static const char *defaults[] = {
        "../../build/quartznet_reduced",
        "../../build/quartznet",
    };
    const char **dirs = defaults;
    int n = 2;
    if (argc > 1) { dirs = (const char **)(argv + 1); n = argc - 1; }

    printf("QuartzNet C interpreter — bit-exact test vs quartznet_ref.py\n");

    int bad = 0;
    for (int i = 0; i < n; i++)
        if (run_config(dirs[i]) != 0) bad++;

    printf("\n%s — %d/%d configurations bit-exact\n",
           bad ? "FAIL" : "PASS", n - bad, n);
    return bad ? 1 : 0;
}
