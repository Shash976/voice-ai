/* qn_transcribe.c
 *
 * mp3-to-text: loads a real (audio-derived) int8 input blob -- produced by
 * sw/tinyml_reference/mp3_to_text.py, NOT the golden's seeded-random input
 * -- runs it through the C interpreter against a chosen model's
 * weights/qparams, and prints the CTC-decoded transcript.
 *
 * With NO arguments (the sweep below), MODEL_DIR is one of the two
 * seeded-random build configs (`make goldens`): (random weights, qparams
 * calibrated against random input) vs real mp3 features is a numerics
 * mismatch by construction, so that transcript is expected to be gibberish
 * -- this only proves the plumbing: mp3 -> quartznet_audio.py ->
 * mp3_input.bin -> qn_load/qn_set_input/qn_run -> qn_ctc_greedy -> a real
 * string over the 29-class vocabulary.
 *
 * Pointed at REAL calibrated weights (explicit MODEL_DIR argv, e.g.
 * ../../build/quartznet_real from quartznet_export_int8.py -- Stage 7 Gap 2
 * A5/A6), the transcript is real English -- this is G2.6's actual
 * deliverable, `make transcribe-real MP3=...`.
 *
 * Build:  make transcribe MP3=path/to/clip.mp3        (seeded-random sweep)
 *         make transcribe-real MP3=path/to/clip.mp3   (real weights, A6)
 *         (or: make qn_transcribe, then run it directly -- see usage below)
 *
 * Usage:  ./qn_transcribe [MODEL_DIR INPUT_BLOB [T_OUT]]
 *   MODEL_DIR   directory with quartznet_desc.bin/quartznet_weights.bin/
 *               quartznet_qparams.bin (e.g. ../../build/quartznet[_reduced]
 *               or ../../build/quartznet_real)
 *   INPUT_BLOB  int8 [t_in, in_ch] blob, e.g. ../../build/quartznet/mp3_input.bin
 *   T_OUT       output frame count; if omitted, read from INPUT_BLOB's
 *               sibling ".t_out" sidecar (mp3_to_text.py writes one)
 *
 * With no arguments, sweeps the default synthetic-clip input against both
 * seeded-random build configs -- mirrors test_quartznet_host.c's
 * dual-config pattern. Every run also prints a "TRANSCRIPT\t<text>" line,
 * meant for a driver script to parse (Stage 7 Gap 2 A6's WER gate).
 */

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "quartznet_infer.h"

#define MAX_TABLE     (64u * 1024u)
#define MAX_WEIGHTS   (20u * 1024u * 1024u)
#define MAX_QPARAMS   ( 2u * 1024u * 1024u)
#define MAX_INPUT     ( 1u * 1024u * 1024u)
#define MAX_TEXT      ( 8u * 1024u)

static uint32_t table_blob32 [MAX_TABLE   / 4];
static int8_t   weight_blob  [MAX_WEIGHTS];
static uint32_t qparam_blob32[MAX_QPARAMS / 4];
static int8_t   input_blob   [MAX_INPUT];
static char     text_buf     [MAX_TEXT];

static qn_model_t model;

/* ── Helpers ────────────────────────────────────────────────────────────────── */

static long load_file(const char *path, void *dst, size_t cap)
{
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "  cannot open %s\n", path); return -1; }
    long n = (long)fread(dst, 1, cap, f);
    int over = !feof(f);
    fclose(f);
    if (over) { fprintf(stderr, "  %s exceeds %zu B buffer\n", path, cap); return -1; }
    return n;
}

/* mp3_to_text.py writes "<stem>.t_out" next to "<stem>.bin" -- derive it. */
static int read_tout_sidecar(const char *input_blob_path)
{
    char path[1024];
    snprintf(path, sizeof path, "%s", input_blob_path);
    size_t len = strlen(path);
    if (len < 4 || strcmp(path + len - 4, ".bin") != 0) {
        fprintf(stderr, "  cannot derive .t_out sidecar from %s (no .bin suffix)\n", path);
        return -1;
    }
    snprintf(path + len - 4, sizeof(path) - (len - 4), ".t_out");

    FILE *f = fopen(path, "r");
    if (!f) { fprintf(stderr, "  cannot open %s\n", path); return -1; }
    int t_out = -1;
    int got = fscanf(f, "%d", &t_out);
    fclose(f);
    if (got != 1 || t_out <= 0) {
        fprintf(stderr, "  bad t_out in %s\n", path);
        return -1;
    }
    return t_out;
}

/* ── One (model, input) pair ───────────────────────────────────────────────── */

static int run_one(const char *model_dir, const char *input_path, int t_out,
                   int expect_gibberish)
{
    char p[1024];

    snprintf(p, sizeof p, "%s/quartznet_desc.bin", model_dir);
    long tn = load_file(p, table_blob32, MAX_TABLE);
    snprintf(p, sizeof p, "%s/quartznet_weights.bin", model_dir);
    long wn = load_file(p, weight_blob, MAX_WEIGHTS);
    snprintf(p, sizeof p, "%s/quartznet_qparams.bin", model_dir);
    long qn = load_file(p, qparam_blob32, MAX_QPARAMS);
    long in = load_file(input_path, input_blob, MAX_INPUT);
    if (tn < 0 || wn < 0 || qn < 0 || in < 0) return -1;

    printf("\n=== model %s   input %s ===\n", model_dir, input_path);

    int rc = qn_load(&model, (const uint8_t *)table_blob32, (size_t)tn,
                     weight_blob, (size_t)wn,
                     (const uint8_t *)qparam_blob32, (size_t)qn, t_out);
    if (rc != QN_OK) { fprintf(stderr, "  qn_load failed: %d\n", rc); return -1; }
    if (!model.arena_ready) {
        fprintf(stderr, "  arena too small for t_out=%d (QN_MAX_T_OUT=%d) -- "
                "rerun mp3_to_text.py with a shorter --seconds\n",
                t_out, QN_MAX_T_OUT);
        return -1;
    }

    /* qn_set_input only reads the first buf_rate[IN]*t_out frames -- a longer
     * blob (odd trailing frame, per quartznet_audio.py) is fine, just unused. */
    long want_in = (long)model.buf_rate[QN_BUF_IN] * (long)t_out * (long)model.in_ch;
    if (in < want_in) {
        fprintf(stderr, "  input blob is %ld B, need at least %ld B for t_out=%d\n",
                in, want_in, t_out);
        return -1;
    }

    printf("  descriptors %u   T_out %d\n", model.n_desc, t_out);

    rc = qn_set_input(&model, input_blob);
    if (rc != QN_OK) { fprintf(stderr, "  qn_set_input failed: %d\n", rc); return -1; }

    rc = qn_run(&model);
    if (rc != QN_OK) { fprintf(stderr, "  qn_run failed: %d\n", rc); return -1; }

    int nsym = qn_ctc_greedy(&model, text_buf, (int)sizeof text_buf);
    if (nsym < 0) { fprintf(stderr, "  qn_ctc_greedy failed: %d\n", nsym); return -1; }

    printf("  transcript (%d symbols%s): \"%s\"\n", nsym,
           expect_gibberish ? ", gibberish expected -- seeded-random weights" : "",
           text_buf);
    printf("TRANSCRIPT\t%s\n", text_buf);  /* machine-readable, one line, tab-delimited */
    return 0;
}

/* ── main ───────────────────────────────────────────────────────────────────── */

int main(int argc, char **argv)
{
    printf("QuartzNet mp3-to-text mechanical smoke test\n");

    if (argc >= 2) {
        const char *model_dir  = argv[1];
        const char *input_path = (argc >= 3) ? argv[2]
            : "../../build/quartznet/mp3_input.bin";
        int t_out = (argc >= 4) ? atoi(argv[3]) : read_tout_sidecar(input_path);
        if (t_out <= 0) return 1;
        return run_one(model_dir, input_path, t_out, /*expect_gibberish=*/0) == 0 ? 0 : 1;
    }

    static const char *dirs[] = {
        "../../build/quartznet_reduced",
        "../../build/quartznet",
    };
    const char *input_path = "../../build/quartznet/mp3_input.bin";
    int t_out = read_tout_sidecar(input_path);
    if (t_out <= 0) {
        fprintf(stderr,
                "run: python3 ../../sw/tinyml_reference/mp3_to_text.py  first\n");
        return 1;
    }

    int bad = 0;
    for (int i = 0; i < 2; i++)
        if (run_one(dirs[i], input_path, t_out, /*expect_gibberish=*/1) != 0) bad++;

    printf("\n%s\n", bad
           ? "FAIL (plumbing broke)"
           : "PASS -- mp3 -> accelerator interpreter -> CTC transcript, end to end");
    return bad ? 1 : 0;
}
