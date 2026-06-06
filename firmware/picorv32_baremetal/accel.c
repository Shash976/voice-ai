/* accel.c — Firmware driver: writes accelerator registers, triggers, polls done. */

#include "accel.h"
#include "../tinyengine_port/conv_im2col.h"

static inline void accel_wait(void)
{
    while (ACCEL_STATUS != 0) {}
}

void accel_conv1d(
    const int8_t *inp, const int8_t *w, const int32_t *b, int8_t *out,
    int in_ch, int in_len, int out_ch, int out_len,
    int kernel, int stride, int pad, int in_zp, int out_zp,
    const int32_t *q_mult, const int32_t *rshift, int relu)
{
    (void)out_len;  /* hardware derives this from in_len/kernel/stride/pad */
    accel_wait();
    ACCEL_IN_PTR     = (uint32_t)inp;
    ACCEL_WT_PTR     = (uint32_t)w;
    ACCEL_BIAS_PTR   = (uint32_t)b;
    ACCEL_MULT_PTR   = (uint32_t)q_mult;
    ACCEL_RSHI_PTR   = (uint32_t)rshift;
    ACCEL_OUT_PTR    = (uint32_t)out;
    ACCEL_DIM_M      = (uint32_t)out_ch;
    ACCEL_DIM_K      = (uint32_t)in_ch;
    ACCEL_DIM_KERN   = (uint32_t)kernel;
    ACCEL_DIM_INLEN  = (uint32_t)in_len;
    ACCEL_DIM_STRIDE = (uint32_t)stride;
    ACCEL_DIM_PAD    = (uint32_t)pad;
    ACCEL_ZP_IN      = (uint32_t)(int32_t)in_zp;
    ACCEL_ZP_OUT     = (uint32_t)(int32_t)out_zp;
    ACCEL_RELU       = (uint32_t)relu;
    ACCEL_CMD        = ACCEL_CMD_CONV1D;  /* triggers operation */
    accel_wait();
}

/* conv1d via im2col → repeated MATVEC (cmd=1) on the MATVEC-only datapath.
 * Bit-identical to accel_conv1d / the software conv1d (see conv_im2col.h), but
 * uses ONLY the matvec operation — exactly what the synthesizable RTL supports.
 * Trades one cmd=2 op for out_len cmd=1 ops (more register programming), which
 * is the cost of dropping the dedicated conv engine from hardware. */
void accel_conv1d_im2col(
    const int8_t *inp, const int8_t *w, const int32_t *b, int8_t *out,
    int in_ch, int in_len, int out_ch, int out_len,
    int kernel, int stride, int pad, int in_zp, int out_zp,
    const int32_t *q_mult, const int32_t *rshift, int relu)
{
    conv1d_im2col(accel_dense, inp, w, b, out,
                  in_ch, in_len, out_ch, out_len,
                  kernel, stride, pad, in_zp, out_zp,
                  q_mult, rshift, relu);
}

void accel_dense(
    const int8_t *inp, const int8_t *w, const int32_t *b, int8_t *out,
    int in, int out_dim, int in_zp, int out_zp,
    const int32_t *q_mult, const int32_t *rshift, int relu)
{
    accel_wait();
    ACCEL_IN_PTR   = (uint32_t)inp;
    ACCEL_WT_PTR   = (uint32_t)w;
    ACCEL_BIAS_PTR = (uint32_t)b;
    ACCEL_MULT_PTR = (uint32_t)q_mult;
    ACCEL_RSHI_PTR = (uint32_t)rshift;
    ACCEL_OUT_PTR  = (uint32_t)out;
    ACCEL_DIM_M    = (uint32_t)out_dim;
    ACCEL_DIM_K    = (uint32_t)in;
    ACCEL_ZP_IN    = (uint32_t)(int32_t)in_zp;
    ACCEL_ZP_OUT   = (uint32_t)(int32_t)out_zp;
    ACCEL_RELU     = (uint32_t)relu;
    ACCEL_CMD      = ACCEL_CMD_MATVEC;   /* triggers operation */
    accel_wait();
}
