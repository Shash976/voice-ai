/* test_im2col_host.c
 *
 * Proves the im2col conv lowering (conv_im2col.h) is BIT-EXACT with the direct
 * conv1d in tiny_vad_infer.c — i.e. that running TinyVAD's conv layers as
 * repeated MATVECs (the only op the synthesizable RTL accelerator implements)
 * gives identical results.
 *
 * Method: run full inference twice on every test vector —
 *   (1) pure software (both hooks NULL → direct conv1d),
 *   (2) conv hook = im2col lowering backed by the software matvec.
 * The two logit vectors must match exactly (not just within tolerance). This
 * isolates the conv→matvec lowering and needs no Verilator/MMIO, so it runs on
 * the host (gcc).
 *
 * Build:   make im2col
 * Run:     ./test_im2col_host    (exit 0 = all bit-exact)
 */

#include <stdio.h>
#include <stdint.h>

#include "tiny_vad_infer.h"
#include "tiny_vad_weights.h"
#include "tiny_vad_test_vectors.h"
#include "conv_im2col.h"

/* conv1d_fn wrapper: lower conv to im2col + MATVEC using the pure-software
 * matvec backend. This is the host stand-in for accel_conv1d_im2col, which uses
 * the hardware MATVEC driver instead. */
static void sw_im2col_conv(
    const int8_t *inp, const int8_t *w, const int32_t *b, int8_t *out,
    int in_ch, int in_len, int out_ch, int out_len,
    int kernel, int stride, int pad, int in_zp, int out_zp,
    const int32_t *q_mult, const int32_t *rshift, int relu)
{
    conv1d_im2col(tinyvad_dense_sw, inp, w, b, out,
                  in_ch, in_len, out_ch, out_len,
                  kernel, stride, pad, in_zp, out_zp,
                  q_mult, rshift, relu);
}

int main(void)
{
    int mismatches = 0;

    printf("im2col conv lowering — bit-exact check vs direct conv1d (%d vectors)\n",
           N_TEST_VECTORS);
    printf("----------------------------------------------------------------\n");

    for (int v = 0; v < N_TEST_VECTORS; v++) {
        int8_t ref[2], got[2];

        /* (1) reference: pure software, direct conv1d */
        tinyvad_conv1d_hook = NULL;
        tinyvad_dense_hook  = NULL;
        tiny_vad_infer(test_inputs[v], ref);

        /* (2) im2col conv lowering (dense path stays software) */
        tinyvad_conv1d_hook = sw_im2col_conv;
        tinyvad_dense_hook  = NULL;
        tiny_vad_infer(test_inputs[v], got);

        if (ref[0] != got[0] || ref[1] != got[1]) {
            mismatches++;
            printf("  MISMATCH v%02d: direct [%4d,%4d]  im2col [%4d,%4d]\n",
                   v, ref[0], ref[1], got[0], got[1]);
        }
    }

    /* leave hooks cleared */
    tinyvad_conv1d_hook = NULL;
    tinyvad_dense_hook  = NULL;

    printf("----------------------------------------------------------------\n");
    if (mismatches == 0) {
        printf("PASS — %d/%d vectors bit-exact. im2col+MATVEC == direct conv1d.\n",
               N_TEST_VECTORS, N_TEST_VECTORS);
        return 0;
    }
    printf("FAIL — %d/%d vectors differ. im2col lowering is NOT bit-exact.\n",
           mismatches, N_TEST_VECTORS);
    return 1;
}
