/* accel.h — Firmware driver for the TinyMAC behavioral accelerator.
 *
 * The accelerator lives at 0x20000000 and is emulated by sim_main.cpp.
 * It offloads conv1d and dense (matvec) layers using int8 MAC arrays.
 *
 * Usage:
 *   tinyvad_conv1d_hook = accel_conv1d;
 *   tinyvad_dense_hook  = accel_dense;
 *   tiny_vad_infer(...);   // hot layers now go to hardware
 */

#ifndef ACCEL_H
#define ACCEL_H

#include <stdint.h>
#include "../tinyengine_port/tiny_vad_infer.h"

/* ── Register map (offsets from ACCEL_BASE) ─────────────────────────────── */

#define ACCEL_BASE        0x20000000u

#define ACCEL_STATUS      (*(volatile uint32_t*)(ACCEL_BASE + 0x00)) /* R: 0=idle 1=busy */
#define ACCEL_CMD         (*(volatile uint32_t*)(ACCEL_BASE + 0x04)) /* W: trigger        */
#define ACCEL_IN_PTR      (*(volatile uint32_t*)(ACCEL_BASE + 0x08))
#define ACCEL_WT_PTR      (*(volatile uint32_t*)(ACCEL_BASE + 0x0C))
#define ACCEL_BIAS_PTR    (*(volatile uint32_t*)(ACCEL_BASE + 0x10))
#define ACCEL_MULT_PTR    (*(volatile uint32_t*)(ACCEL_BASE + 0x14))
#define ACCEL_RSHI_PTR    (*(volatile uint32_t*)(ACCEL_BASE + 0x18))
#define ACCEL_OUT_PTR     (*(volatile uint32_t*)(ACCEL_BASE + 0x1C))
#define ACCEL_DIM_M       (*(volatile uint32_t*)(ACCEL_BASE + 0x20)) /* out_ch / out_dim  */
#define ACCEL_DIM_K       (*(volatile uint32_t*)(ACCEL_BASE + 0x24)) /* in_ch  / in       */
#define ACCEL_DIM_KERN    (*(volatile uint32_t*)(ACCEL_BASE + 0x28)) /* kernel (conv)     */
#define ACCEL_DIM_INLEN   (*(volatile uint32_t*)(ACCEL_BASE + 0x2C)) /* in_len (conv)     */
#define ACCEL_DIM_STRIDE  (*(volatile uint32_t*)(ACCEL_BASE + 0x30)) /* stride (conv)     */
#define ACCEL_DIM_PAD     (*(volatile uint32_t*)(ACCEL_BASE + 0x34)) /* pad    (conv)     */
#define ACCEL_ZP_IN       (*(volatile uint32_t*)(ACCEL_BASE + 0x38))
#define ACCEL_ZP_OUT      (*(volatile uint32_t*)(ACCEL_BASE + 0x3C))
#define ACCEL_RELU        (*(volatile uint32_t*)(ACCEL_BASE + 0x40))
#define ACCEL_LAST_CYC    (*(volatile uint32_t*)(ACCEL_BASE + 0x44)) /* R: last op cycles */

#define ACCEL_CMD_MATVEC  1u
#define ACCEL_CMD_CONV1D  2u

/* ── Hook implementations ───────────────────────────────────────────────── */

void accel_conv1d(
    const int8_t *inp, const int8_t *w, const int32_t *b, int8_t *out,
    int in_ch, int in_len, int out_ch, int out_len,
    int kernel, int stride, int pad, int in_zp, int out_zp,
    const int32_t *q_mult, const int32_t *rshift, int relu);

/* Same as accel_conv1d but lowers the layer to im2col + repeated MATVEC, so the
 * MATVEC-only RTL accelerator can run conv layers. Drop-in tinyvad_conv1d_fn. */
void accel_conv1d_im2col(
    const int8_t *inp, const int8_t *w, const int32_t *b, int8_t *out,
    int in_ch, int in_len, int out_ch, int out_len,
    int kernel, int stride, int pad, int in_zp, int out_zp,
    const int32_t *q_mult, const int32_t *rshift, int relu);

void accel_dense(
    const int8_t *inp, const int8_t *w, const int32_t *b, int8_t *out,
    int in, int out_dim, int in_zp, int out_zp,
    const int32_t *q_mult, const int32_t *rshift, int relu);

#endif /* ACCEL_H */
