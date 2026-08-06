/* qn_accel.c — firmware driver: stages MMIO registers, triggers, polls done.
 * See qn_accel.h for the register map and the contract each function follows.
 */

#include "qn_accel.h"

/* Real hangs are a hardware/firmware bug, not a scenario to design around --
 * this guard exists only so such a bug reports QN_ERR_HW_TIMEOUT instead of
 * wedging firmware forever, mirroring rtl/tb/quartznet_tb.cpp's own GUARD. */
#define QN_HW_POLL_GUARD 0x7FFFFFFFu

static inline void qn_mem_addr_ctrl(int bank, uint32_t addr, int autoinc)
{
    QN_REG(QN_R_MEM_CTRL) = (uint32_t)(bank & 1) | (autoinc ? QN_MEM_CTRL_AUTOINC : 0u);
    QN_REG(QN_R_MEM_ADDR) = addr;
}

/* MEM_DATA is a single byte, not 32-bit -- see quartznet_accel.v's D4 header
 * comment: this MMIO bus has no ready/wait-state signal, so a register write
 * must complete in the one cycle mmio_we is asserted, and a 32-bit transfer
 * through the byte-granular bd_* backdoor can't be atomic in one cycle. */
void qn_mem_write(int bank, uint32_t addr, const void *src, uint32_t n)
{
    const uint8_t *s = (const uint8_t *)src;
    qn_mem_addr_ctrl(bank, addr, 1);
    for (uint32_t i = 0; i < n; i++)
        QN_REG(QN_R_MEM_DATA) = s[i];   /* auto-increments MEM_ADDR by 1 */
}

/* No AUTOINC on read: this bus has no read-strobe, so there is no edge to
 * hang a read-side auto-increment off of (see quartznet_accel.v) -- MEM_ADDR
 * is restaged before every byte instead. */
void qn_mem_read(int bank, uint32_t addr, void *dst, uint32_t n)
{
    uint8_t *d = (uint8_t *)dst;
    QN_REG(QN_R_MEM_CTRL) = (uint32_t)(bank & 1);
    for (uint32_t i = 0; i < n; i++) {
        QN_REG(QN_R_MEM_ADDR) = addr + i;
        d[i] = (uint8_t)QN_REG(QN_R_MEM_DATA);
    }
}

void qn_mem_fill(int bank, uint32_t addr, uint8_t val, uint32_t n)
{
    QN_REG(QN_R_MEM_CTRL) = (uint32_t)(bank & 1);
    QN_REG(QN_R_MEM_ADDR) = addr;
    QN_REG(QN_R_MEM_DATA) = (uint32_t)val;
    QN_REG(QN_R_MEM_FILL) = n;
    /* Genuinely multi-cycle (one bd_* write per byte in hardware) -- must be
     * polled to completion before any other MEM_* access, or a subsequent
     * write races the still-running fill and corrupts both. */
    while (QN_REG(QN_R_STATUS) & QN_STATUS_BUSY) { }
}

int qn_set_input_hw(const qn_model_t *m, const int8_t *input, const qn_hw_map_t *map)
{
    if (m->t_out <= 0) return QN_ERR_ARENA;

    /* Zero the whole arena footprint this t_out needs -- ping-pong buffers
     * rely on a clean start, same as qn_set_input()'s zero-then-copy. */
    uint32_t used = m->buf_off[m->n_buffers - 1]
                  + m->buf_ch[m->n_buffers - 1] * m->buf_rate[m->n_buffers - 1]
                    * (uint32_t)m->t_out;
    qn_mem_fill(QN_BANK_PSRAM, map->arena_base, 0, used);

    uint32_t pitch = m->buf_ch[QN_BUF_IN];
    uint32_t rows  = m->buf_rate[QN_BUF_IN] * (uint32_t)m->t_out;
    uint32_t ch    = m->in_ch;
    for (uint32_t t = 0; t < rows; t++)
        qn_mem_write(QN_BANK_PSRAM,
                     map->arena_base + m->buf_off[QN_BUF_IN] + t * pitch,
                     input + (size_t)t * ch, ch);
    return QN_OK;
}

int qn_run_hw(const qn_model_t *m, const qn_hw_map_t *map, uint32_t *out_cycles)
{
    QN_REG(QN_R_T_OUT)        = (uint32_t)m->t_out;
    QN_REG(QN_R_TABLE_BASE)   = map->table_base;
    QN_REG(QN_R_W_BLOB_BASE)  = map->w_blob_base;
    QN_REG(QN_R_QP_BLOB_BASE) = map->qp_blob_base;

    QN_REG(QN_R_CTRL) = QN_CTRL_RUN_TABLE;

    uint32_t guard = 0;
    while (!(QN_REG(QN_R_STATUS) & QN_STATUS_TABLE_DONE)) {
        if (++guard > QN_HW_POLL_GUARD) return QN_ERR_HW_TIMEOUT;
    }
    uint32_t status = QN_REG(QN_R_STATUS);
    QN_REG(QN_R_STATUS) = QN_STATUS_TABLE_DONE;   /* W1C */

    if (out_cycles) *out_cycles = QN_REG(QN_R_CYCLES);
    return (status & QN_STATUS_ERR_BAD_IN_OFF) ? QN_ERR_HW_BAD_IN_OFF : QN_OK;
}

int qn_run_hw_step(const qn_model_t *m, const qn_hw_map_t *map,
                    qn_hw_step_fn on_desc, uint32_t *out_cycles)
{
    QN_REG(QN_R_T_OUT)        = (uint32_t)m->t_out;
    QN_REG(QN_R_TABLE_BASE)   = map->table_base;
    QN_REG(QN_R_W_BLOB_BASE)  = map->w_blob_base;
    QN_REG(QN_R_QP_BLOB_BASE) = map->qp_blob_base;

    QN_REG(QN_R_CTRL) = QN_CTRL_RUN_TABLE | QN_CTRL_SINGLE_STEP;

    uint32_t guard = 0;
    for (uint32_t i = 0; i < m->n_desc; i++) {
        while (!(QN_REG(QN_R_STATUS) & (QN_STATUS_DONE | QN_STATUS_TABLE_DONE))) {
            if (++guard > QN_HW_POLL_GUARD) return QN_ERR_HW_TIMEOUT;
        }
        if (on_desc) on_desc(m, (int)i, map);
        QN_REG(QN_R_STATUS) = QN_STATUS_DONE | QN_STATUS_TABLE_DONE;   /* W1C, lets the walker proceed */
    }
    uint32_t status = QN_REG(QN_R_STATUS);
    if (out_cycles) *out_cycles = QN_REG(QN_R_CYCLES);
    return (status & QN_STATUS_ERR_BAD_IN_OFF) ? QN_ERR_HW_BAD_IN_OFF : QN_OK;
}

int qn_gather_output_hw(const qn_model_t *m, int i, int8_t *dst, const qn_hw_map_t *map)
{
    if (i < 0 || i >= (int)m->n_desc) return QN_ERR_TRUNCATED;
    const qn_desc_t *d     = &m->desc[i];
    uint32_t         pitch = m->buf_ch[d->out_buf];
    uint32_t         base  = map->arena_base + m->buf_off[d->out_buf] + d->out_off;

    for (int t = 0; t < m->t_out; t++)
        qn_mem_read(QN_BANK_PSRAM, base + (uint32_t)t * pitch,
                    dst + (size_t)t * (uint32_t)d->c_out, (uint32_t)d->c_out);
    return QN_OK;
}

int qn_read_logits_hw(const qn_model_t *m, int8_t *dst, const qn_hw_map_t *map)
{
    uint32_t pitch = m->buf_ch[QN_BUF_LOGITS];
    uint32_t base  = map->arena_base + m->buf_off[QN_BUF_LOGITS];
    qn_mem_read(QN_BANK_PSRAM, base, dst, pitch * (uint32_t)m->t_out);
    return QN_OK;
}
