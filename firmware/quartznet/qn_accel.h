/* qn_accel.h — firmware driver for the QuartzNet table-walking accelerator.
 *
 * Register map mirrors quartznet_accel.v's header comment and
 * rtl/tb/quartznet_tb.cpp's REG_, CTRL_, STATUS_ names exactly (idx 0-32).
 * MEM_ADDR/MEM_CTRL/MEM_DATA/MEM_FILL (idx 33-36) and CTRL.SINGLE_STEP (bit3)
 * are new in this driver, added to replace rtl/tb's simulation-only bd_*
 * backdoor with a real MMIO path a CPU can use. Until Stage 7 Gap 1 increment
 * D4 lands in quartznet_accel.v, only the sim/verilator_qn C++ shim
 * (MODE=shim) implements them; D4 makes them real Verilog.
 *
 * This header only declares the hardware-dispatch API. qn_load()/
 * qn_ctc_greedy_buf() (quartznet_infer.h) are reused unchanged -- see
 * qn_main.c for the intended call sequence.
 */

#ifndef QN_ACCEL_H
#define QN_ACCEL_H

#include <stdint.h>
#include "quartznet_infer.h"

#define QN_ACCEL_BASE  0x20000000u
#define QN_REG(i)      (*(volatile uint32_t *)(QN_ACCEL_BASE + ((uint32_t)(i) << 2)))

enum {
    QN_R_CTRL = 0, QN_R_STATUS = 1, QN_R_CYCLES = 2,
    /* idx 3-28: legacy single-descriptor mode fields (quartznet_accel.v),
     * unused by the table-walk driver below. */
    QN_R_T_OUT = 11,
    QN_R_TABLE_BASE = 29, QN_R_W_BLOB_BASE = 30, QN_R_QP_BLOB_BASE = 31,
    QN_R_DESC_IDX = 32,
    /* new in this driver -- see the file header */
    QN_R_MEM_ADDR = 33, QN_R_MEM_CTRL = 34, QN_R_MEM_DATA = 35, QN_R_MEM_FILL = 36
};

#define QN_CTRL_START        0x1u
#define QN_CTRL_RUN_TABLE    0x2u
#define QN_CTRL_SINGLE_STEP  0x8u

#define QN_STATUS_BUSY           0x1u
#define QN_STATUS_DONE           0x2u
#define QN_STATUS_TABLE_DONE     0x4u
#define QN_STATUS_ERR_BAD_IN_OFF 0x8u

#define QN_MEM_CTRL_AUTOINC    0x2u

#define QN_BANK_QSPI  0
#define QN_BANK_PSRAM 1

#define QN_ERR_HW_TIMEOUT    -8
#define QN_ERR_HW_BAD_IN_OFF -9

/* QSPI/PSRAM byte addresses this table-walk uses. The caller (qn_main.c)
 * picks these to match how it laid the QSPI image out and how large the
 * PSRAM arena needs to be for m->t_out (quartznet_topology.py::arena_layout). */
typedef struct {
    uint32_t table_base;    /* QSPI byte addr of the descriptor-table image */
    uint32_t w_blob_base;   /* QSPI byte addr of the weight blob */
    uint32_t qp_blob_base;  /* QSPI byte addr of the qparam blob */
    uint32_t arena_base;    /* PSRAM byte addr of the activation arena */
} qn_hw_map_t;

/* ── Raw MEM_* port accessors ─────────────────────────────────────────────
 * addr is relative to the bank (QSPI or PSRAM), not to arena_base -- callers
 * add arena_base themselves, mirroring quartznet_tb.cpp's
 * bd_write(1, buf_off[b] + ..., ...) convention. */
void qn_mem_write(int bank, uint32_t addr, const void *src, uint32_t n);
void qn_mem_read (int bank, uint32_t addr, void *dst, uint32_t n);
void qn_mem_fill (int bank, uint32_t addr, uint8_t val, uint32_t n);

/* Zero the PSRAM arena footprint m->t_out needs and write the mel input into
 * BUF_IN at map->arena_base + m->buf_off[QN_BUF_IN]. Software analogue:
 * qn_set_input(). Call after qn_load(m, table, table_bytes, NULL, 0, NULL, 0, t_out). */
int qn_set_input_hw(const qn_model_t *m, const int8_t *input,
                     const qn_hw_map_t *map);

/* Stage T_OUT/TABLE_BASE/W_BLOB_BASE/QP_BLOB_BASE, pulse CTRL.RUN_TABLE, poll
 * STATUS.TABLE_DONE, W1C-clear it, and surface err_bad_in_off. *out_cycles
 * (if non-NULL) is set from REG_CYCLES on success -- informational only: on
 * the shim it reflects sim/quartznet_cycles' model, not real RTL timing; only
 * a run against real qn_soc.v RTL (D5) is a timing claim.
 * Software analogue: qn_run(). */
int qn_run_hw(const qn_model_t *m, const qn_hw_map_t *map, uint32_t *out_cycles);

/* Called by qn_run_hw_step() right after descriptor desc_idx completes and
 * before the walker is allowed to proceed -- the only point at which that
 * descriptor's output slice is guaranteed not yet overwritten by a later
 * descriptor reusing the same ping-pong buffer. */
typedef void (*qn_hw_step_fn)(const qn_model_t *m, int desc_idx,
                               const qn_hw_map_t *map);

/* Same as qn_run_hw() but walks with CTRL.SINGLE_STEP set: the accelerator
 * halts after each descriptor until firmware clears STATUS.DONE. Calls
 * on_desc() (if non-NULL) once per descriptor, in the safe window described
 * above -- typically qn_gather_output_hw() for a bit-exact regression check.
 * Slower than qn_run_hw() (one MMIO round trip per descriptor instead of one
 * for the whole table); use qn_run_hw() for production transcription. */
int qn_run_hw_step(const qn_model_t *m, const qn_hw_map_t *map,
                    qn_hw_step_fn on_desc, uint32_t *out_cycles);

/* Read descriptor i's output slice back out of PSRAM into dst, contiguous
 * [t_out, c_out] -- same layout as qn_gather_output(). Only safe to call
 * before a later descriptor has overwritten the same buffer region; see
 * qn_run_hw_step() above. */
int qn_gather_output_hw(const qn_model_t *m, int i, int8_t *dst,
                         const qn_hw_map_t *map);

/* Read BUF_LOGITS back out of PSRAM into dst, contiguous [t_out, pitch]
 * (pitch == m->buf_ch[QN_BUF_LOGITS]), ready for qn_ctc_greedy_buf(). */
int qn_read_logits_hw(const qn_model_t *m, int8_t *dst, const qn_hw_map_t *map);

#endif /* QN_ACCEL_H */
