/* conv_im2col.h
 *
 * Lower a 1-D convolution to a sequence of matrix-vector products (im2col), so a
 * MATVEC-only datapath can execute it. This is what lets the synthesizable RTL
 * accelerator (rtl/accel/tinymac_accel.v — which implements MATVEC *only*) run
 * TinyVAD's two dominant conv layers without a dedicated conv engine.
 *
 * Why it is bit-exact with the direct conv1d() in tiny_vad_infer.c:
 *
 *   conv1d acc[t][oc] = b[oc] + Σ_ic Σ_k (inp[(t*stride+k-pad)*in_ch+ic] - in_zp)
 *                                        * w[(oc*in_ch+ic)*kernel + k]
 *
 *   Define K = in_ch*kernel and the flattened tap index j = ic*kernel + k. Then
 *   the conv weight is ALREADY a [out_ch, K] matrix: w[(oc*in_ch+ic)*kernel+k]
 *   == w[oc*K + j], no repacking needed. Build the patch vector
 *       patch[j] = inp[(t*stride+k-pad)*in_ch+ic]   (in range)
 *                = in_zp                              (out of range / padding)
 *   and a single MATVEC of M=out_ch, K=in_ch*kernel reproduces acc[t][:] exactly:
 *   the padding taps evaluate to (in_zp - in_zp)*w = 0, matching conv1d skipping
 *   out-of-range positions. The j-iteration order (ic-major, k-minor) is the same
 *   as conv1d's loop nesting, so per-MAC accumulator saturation (narrow ACC_W)
 *   also matches. Requantize / ReLU / zero-point / clamp are the matvec's job and
 *   are shared verbatim.
 *
 * The matvec backend is a caller-supplied tinyvad_dense_fn:
 *   - firmware  → accel_dense (the hardware MATVEC driver, cmd=1)
 *   - host test → tinyvad_dense_sw (pure software) for bit-exact verification
 */

#ifndef CONV_IM2COL_H
#define CONV_IM2COL_H

#include <stdint.h>
#include "tiny_vad_infer.h"   /* for tinyvad_dense_fn and the model dims */

/* Largest K = in_ch*kernel across TinyVAD convs: Conv0 = 40*5 = 200. */
#ifndef IM2COL_MAX_K
#define IM2COL_MAX_K 256
#endif

/* Run a conv1d layer as out_len matrix-vector products via `matvec`.
 * Output is written in [time, channel] layout: out[t*out_ch + oc]. */
static inline void conv1d_im2col(
    tinyvad_dense_fn matvec,
    const int8_t *inp, const int8_t *w, const int32_t *b, int8_t *out,
    int in_ch, int in_len, int out_ch, int out_len,
    int kernel, int stride, int pad, int in_zp, int out_zp,
    const int32_t *q_mult, const int32_t *rshift, int relu)
{
    int8_t patch[IM2COL_MAX_K];
    const int K = in_ch * kernel;

    for (int t = 0; t < out_len; t++) {
        /* im2col: gather this timestep's receptive field into a K-vector,
         * filling out-of-range (padding) taps with the input zero-point. */
        for (int ic = 0; ic < in_ch; ic++) {
            for (int k = 0; k < kernel; k++) {
                int pos = t * stride + k - pad;
                patch[ic * kernel + k] =
                    (pos >= 0 && pos < in_len) ? inp[pos * in_ch + ic]
                                               : (int8_t)in_zp;
            }
        }
        /* one MATVEC: out[t][0..out_ch-1] = W[out_ch][K] · (patch - in_zp) */
        matvec(patch, w, b, out + t * out_ch,
               K, out_ch, in_zp, out_zp, q_mult, rshift, relu);
    }
}

#endif /* CONV_IM2COL_H */
