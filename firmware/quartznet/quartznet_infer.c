/* quartznet_infer.c
 *
 * Table-driven int8 interpreter for QuartzNet 15x5.  See quartznet_infer.h for
 * the contract and the memory layout.
 *
 * ── What this file must reproduce, bit for bit ───────────────────────────────
 *
 * sw/tinyml_reference/quartznet_ref.py is the golden model.  Three of its
 * conventions are load-bearing and easy to "fix" into a wrong answer:
 *
 *  1. The activation row pitch is the BUFFER's allocated channel count, not the
 *     descriptor's in_stride/out_stride.  RefRunner.buf is shaped
 *     [rate*t_out, buffer_channels]; a 48-channel layer writing into the
 *     128-channel buffer A still steps 128 per frame.
 *
 *  2. Every operand READ starts at channel 0 of its buffer.  Only WRITES are
 *     offset, by out_off (== the descriptor's c_out_base).  RefRunner._acc_pw
 *     takes src[:, :c_in], _acc_add takes buf[:, :c_out] — even for a
 *     channel-split ADD such as "B2.0/add[64:96]", which reads channels 0..32
 *     and writes channels 64..96.  That asymmetry is the reference's definition
 *     of the format; do not "correct" it here.
 *
 *  3. ADD carries no bias and only three qparam entries: [0] and [1] pre-scale
 *     the two operands after a fixed << 20, [2] requantizes the sum.  Every
 *     other op has one (bias, q_mult, rshift) triple per output channel and
 *     adds bias into the int32 accumulator.
 *
 * ── Tiling ───────────────────────────────────────────────────────────────────
 *
 * Time is tiled by header t_tile (32); depthwise additionally tiles channels by
 * header dw_ch_tile (64).  Output channels are already bounded by c_out_tile
 * (512) at emission time, so pointwise needs no channel tiling.  Cross-LAYER
 * tiling is invalid — the cumulative depthwise receptive field is 4,012 frames.
 *
 * Time tiles run OUTER and depthwise channel tiles INNER so that the whole
 * channel width of a time tile is finished before moving on: that is what makes
 * the depthwise->pointwise fusion flagged by QN_F_FUSE_NEXT expressible, and it
 * makes halo retention a property of the time loop alone.
 *
 * Tiling changes nothing numerically here (each output element is an
 * independent reduction), which is exactly why comparing against the untiled
 * NumPy reference proves the tiling exact.
 */

#include "quartznet_infer.h"

/* ── Acceleration / observation hook (NULL = nothing observes the schedule) ─── */

qn_tile_fn qn_tile_hook = NULL;

const char qn_labels[QN_N_LABELS + 1] = " abcdefghijklmnopqrstuvwxyz'";

/* ── Activation arena ──────────────────────────────────────────────────────────
 * QN_STATIC_ARENA=1 (default, host/x86 builds): a static QN_ARENA_BYTES array,
 * pointed to by qn_arena from the start -- behavior identical to before this
 * was a pointer. QN_STATIC_ARENA=0 (RV32 hardware-dispatch build): no array is
 * allocated; qn_arena stays NULL until qn_set_arena() is called, or forever if
 * only the hardware-dispatch API (qn_accel.h) is used, since that path never
 * touches qn_arena at all. */
#if QN_STATIC_ARENA
static int8_t qn_arena_storage[QN_ARENA_BYTES];
static int8_t *qn_arena = qn_arena_storage;
#else
static int8_t *qn_arena = NULL;
#endif

void qn_set_arena(int8_t *arena, size_t bytes)
{
    (void)bytes;
    qn_arena = arena;
}

/* ── Fixed-point requantize ───────────────────────────────────────────────────
 *
 * Ported verbatim from firmware/tinyengine_port/tiny_vad_infer.c:44-63 and
 * bit-identical to quartznet_ref.py::requantize().
 *
 * result = x * (q_mult * 2^-31) * 2^-shift
 * shift > 0 = right shift (round half up), shift < 0 = LEFT shift.
 */
static inline int32_t requantize(int32_t x, int32_t q_mult, int32_t shift)
{
    int64_t val = (int64_t)q_mult * (int64_t)x;
    val += (1LL << 30);   /* round before Q31 shift */
    val >>= 31;
    if (shift > 0) {
        val += (1LL << (shift - 1));
        val >>= shift;
    } else if (shift < 0) {
        val <<= (-shift);
    }
    return (int32_t)val;
}

static inline int8_t clamp_i8(int32_t x)
{
    if (x >  127) return  127;
    if (x < -128) return -128;
    return (int8_t)x;
}

/* ── Little-endian blob readers (no unaligned loads) ─────────────────────────── */

static uint32_t rd_u32(const uint8_t *p)
{
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8)
         | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

static int32_t rd_i32(const uint8_t *p)
{
    return (int32_t)rd_u32(p);
}

static int s8(uint32_t v)
{
    v &= 0xFFu;
    return (v >= 128u) ? (int)v - 256 : (int)v;
}

/* qparam blob accessors — offsets are 4-byte multiples by construction. */
static inline int32_t qp(const qn_model_t *m, uint32_t off, int i)
{
    return rd_i32(m->qparams + off + (uint32_t)i * 4u);
}

/* ── Loading ────────────────────────────────────────────────────────────────── */

static void unpack_record(qn_desc_t *d, const uint8_t *rec)
{
    uint32_t w[QN_DESC_WORDS];
    for (int i = 0; i < QN_DESC_WORDS; i++)
        w[i] = rd_u32(rec + i * 4);

    d->op         = (int)( w[0]        & 0xFFu);
    d->flags      = (int)((w[0] >>  8) & 0xFFu);
    d->in_buf     = (int)((w[0] >> 16) & 0xFFu);
    d->out_buf    = (int)((w[0] >> 24) & 0xFFu);
    d->c_in       = (int)( w[1]        & 0xFFFFu);
    d->c_out      = (int)((w[1] >> 16) & 0xFFFFu);
    d->k          = (int)( w[2]        & 0xFFFFu);
    d->stride     = (int)((w[2] >> 16) & 0xFFu);
    d->dilation   = (int)((w[2] >> 24) & 0xFFu);
    d->pad        = (int)( w[3]        & 0xFFFFu);
    d->res_buf    = (int)((w[3] >> 16) & 0xFFu);
    d->in_stride  = (int)( w[4]        & 0xFFFFu);
    d->out_stride = (int)((w[4] >> 16) & 0xFFFFu);
    d->in_zp      = s8( w[5]);
    d->out_zp     = s8( w[5] >>  8);
    d->res_zp     = s8( w[5] >> 16);
    d->in_off     = w[6];
    d->out_off    = w[7];
    d->res_off    = w[8];
    d->w_off      = w[9];
    d->bias_off   = w[10];
    d->qmult_off  = w[11];
    d->rshift_off = w[12];
    d->acc_off    = w[13];
    d->layer_id   = (int)( w[14]        & 0xFFFFu);
    d->block_id   = (int)((w[14] >> 16) & 0xFFFFu);

    d->n_qch  = (d->op == QN_OP_ADD) ? 3 : d->c_out;
    if (d->op == QN_OP_DW)
        d->w_bytes = (int32_t)d->c_out * d->k;
    else if (d->op == QN_OP_PW)
        d->w_bytes = (int32_t)d->c_out * d->c_in * d->k;
    else
        d->w_bytes = 0;
    d->q_bytes = (int32_t)(3 * d->n_qch * 4);
}

int qn_load(qn_model_t *m,
            const uint8_t *table, size_t table_bytes,
            const int8_t  *weights, size_t weight_bytes,
            const uint8_t *qparams, size_t qparam_bytes,
            int t_out)
{
    if (table_bytes < QN_HEADER_BYTES) return QN_ERR_TRUNCATED;

    m->magic        = rd_u32(table +  0);
    m->version      = rd_u32(table +  4);
    m->n_desc       = rd_u32(table +  8);
    m->desc_stride  = rd_u32(table + 12);
    m->weight_bytes = rd_u32(table + 16);
    m->qparam_bytes = rd_u32(table + 20);
    m->n_classes    = rd_u32(table + 24);
    m->blank_idx    = rd_u32(table + 28);
    m->in_ch        = rd_u32(table + 32);
    m->in_zp        = rd_i32(table + 36);
    m->t_tile       = rd_u32(table + 40);
    m->c_out_tile   = rd_u32(table + 44);
    m->dw_ch_tile   = rd_u32(table + 48);
    m->dw_span_max  = rd_u32(table + 52);
    m->n_buffers    = rd_u32(table + 56);
    m->desc_off     = rd_u32(table + 60);

    if (m->magic != QN_DESC_MAGIC)          return QN_ERR_MAGIC;
    if (m->version != QN_DESC_VERSION)      return QN_ERR_VERSION;
    if (m->desc_stride != QN_DESC_BYTES)    return QN_ERR_VERSION;
    if (m->n_desc > (uint32_t)QN_MAX_DESC)  return QN_ERR_TOO_MANY;
    if (m->n_buffers != QN_N_BUFFERS)       return QN_ERR_BUFFERS;
    if (table_bytes < m->desc_off + (size_t)m->n_desc * QN_DESC_BYTES)
        return QN_ERR_TRUNCATED;

    for (uint32_t b = 0; b < m->n_buffers; b++) {
        const uint8_t *p = table + QN_HEADER_BYTES + b * QN_BUFREC_BYTES;
        m->buf_ch[b]   = rd_u32(p);
        m->buf_rate[b] = rd_u32(p + 4);
    }

    for (uint32_t i = 0; i < m->n_desc; i++)
        unpack_record(&m->desc[i], table + m->desc_off + i * QN_DESC_BYTES);

    /* NULL blobs are legal: qn_walk() only needs the schedule, not the data. */
    if (weights && weight_bytes < m->weight_bytes) return QN_ERR_BLOB;
    if (qparams && qparam_bytes < m->qparam_bytes) return QN_ERR_BLOB;
    m->weights      = weights;
    m->qparams      = qparams;
    m->weight_avail = weight_bytes;
    m->qparam_avail = qparam_bytes;
    m->t_out        = t_out;

    /* Arena layout — mirrors DescriptorTable.arena_layout(). */
    size_t cur = 0;
    for (uint32_t b = 0; b < m->n_buffers; b++) {
        m->buf_off[b] = (uint32_t)cur;
        cur += (size_t)m->buf_ch[b] * m->buf_rate[b] * (size_t)t_out;
    }
    /* Not an error: qn_walk() needs no arena, so only qn_run() enforces this. */
    m->arena_ready = (t_out > 0 && cur <= (size_t)QN_ARENA_BYTES);

    return QN_OK;
}

int8_t *qn_buffer(const qn_model_t *m, int b)
{
    if (!m->arena_ready || b < 0 || b >= (int)m->n_buffers) return NULL;
    return qn_arena + m->buf_off[b];
}

int qn_set_input(qn_model_t *m, const int8_t *input)
{
    if (!m->arena_ready) return QN_ERR_ARENA;

    size_t used = (size_t)m->buf_off[m->n_buffers - 1]
                + (size_t)m->buf_ch[m->n_buffers - 1]
                  * m->buf_rate[m->n_buffers - 1] * (size_t)m->t_out;
    for (size_t i = 0; i < used; i++) qn_arena[i] = 0;

    int8_t *dst   = qn_arena + m->buf_off[QN_BUF_IN];
    int     pitch = (int)m->buf_ch[QN_BUF_IN];
    int     rows  = (int)m->buf_rate[QN_BUF_IN] * m->t_out;
    int     ch    = (int)m->in_ch;
    for (int t = 0; t < rows; t++)
        for (int c = 0; c < ch; c++)
            dst[(size_t)t * pitch + c] = input[(size_t)t * ch + c];
    return QN_OK;
}

/* ── Store: requantize -> +zp -> ReLU -> clamp -> write ──────────────────────── */

static inline void qn_store(const qn_model_t *m, const qn_desc_t *d,
                            int t, int c, int32_t acc)
{
    int32_t qm, rs;
    if (d->op == QN_OP_ADD) {
        /* n_qch == 3, per tensor: entry [2] requantizes the summed operands. */
        qm = qp(m, d->qmult_off, 2);
        rs = qp(m, d->rshift_off, 2);
    } else {
        qm = qp(m, d->qmult_off, c);
        rs = qp(m, d->rshift_off, c);
    }
    int32_t r = requantize(acc, qm, rs) + d->out_zp;
    if ((d->flags & QN_F_RELU) && r < d->out_zp) r = d->out_zp;

    int8_t *dst   = qn_arena + m->buf_off[d->out_buf];
    int     pitch = (int)m->buf_ch[d->out_buf];
    dst[(size_t)t * pitch + d->out_off + (uint32_t)c] = clamp_i8(r);
}

/* ── Per-op tile kernels ────────────────────────────────────────────────────── */

static void tile_dw(const qn_model_t *m, const qn_desc_t *d,
                    int t0, int t1, int c0, int c1, int t_in)
{
    const int8_t  *src   = qn_arena + m->buf_off[d->in_buf];
    const int      pitch = (int)m->buf_ch[d->in_buf];
    const int      base  = (int)(d->in_off % (uint32_t)pitch);
    const int8_t  *w     = m->weights + d->w_off;
    const int      K     = d->k;

    for (int t = t0; t < t1; t++) {
        for (int c = c0; c < c1; c++) {
            int32_t acc = qp(m, d->bias_off, c);
            for (int k = 0; k < K; k++) {
                int pos = t * d->stride + k * d->dilation - d->pad;
                if (pos < 0 || pos >= t_in) continue;   /* zero-point padding */
                int32_t x  = (int32_t)src[(size_t)pos * pitch + base + c] - d->in_zp;
                int32_t wv = (int32_t)w[(size_t)c * K + k];
                acc += x * wv;
            }
            qn_store(m, d, t, c, acc);
        }
    }
}

static void tile_pw(const qn_model_t *m, const qn_desc_t *d,
                    int t0, int t1, int c0, int c1)
{
    const int8_t  *src   = qn_arena + m->buf_off[d->in_buf];
    const int      pitch = (int)m->buf_ch[d->in_buf];
    const int      base  = (int)(d->in_off % (uint32_t)pitch);
    const int8_t  *w     = m->weights + d->w_off;
    const int      c_in  = d->c_in;

    for (int t = t0; t < t1; t++) {
        const int8_t *row = src + (size_t)t * pitch + base;
        for (int oc = c0; oc < c1; oc++) {
            const int8_t *wr = w + (size_t)oc * c_in;
            int32_t acc = qp(m, d->bias_off, oc);
            for (int ic = 0; ic < c_in; ic++)
                acc += ((int32_t)row[ic] - d->in_zp) * (int32_t)wr[ic];
            qn_store(m, d, t, oc, acc);
        }
    }
}

static void tile_add(const qn_model_t *m, const qn_desc_t *d,
                     int t0, int t1, int c0, int c1)
{
    /* TFLite ADD: pre-scale both operands into a shared domain, then sum.
     * Both operands are read from channel 0 of their buffers even when the
     * descriptor writes an offset channel band — see the file header. */
    const int8_t *ma = qn_arena + m->buf_off[d->in_buf];
    const int8_t *ra = qn_arena + m->buf_off[d->res_buf];
    const int     mp = (int)m->buf_ch[d->in_buf];
    const int     rp = (int)m->buf_ch[d->res_buf];
    const int32_t qm0 = qp(m, d->qmult_off, 0), rs0 = qp(m, d->rshift_off, 0);
    const int32_t qm1 = qp(m, d->qmult_off, 1), rs1 = qp(m, d->rshift_off, 1);

    for (int t = t0; t < t1; t++) {
        for (int c = c0; c < c1; c++) {
            int32_t mv = (int32_t)ma[(size_t)t * mp + c] - d->in_zp;
            int32_t rv = (int32_t)ra[(size_t)t * rp + c] - d->res_zp;
            int32_t a  = requantize(mv << QN_ADD_LEFT_SHIFT, qm0, rs0);
            int32_t b  = requantize(rv << QN_ADD_LEFT_SHIFT, qm1, rs1);
            qn_store(m, d, t, c, a + b);   /* ADD carries no bias */
        }
    }
}

static void tile_requant(const qn_model_t *m, const qn_desc_t *d,
                         int t0, int t1, int c0, int c1)
{
    const int8_t *src   = qn_arena + m->buf_off[d->in_buf];
    const int     pitch = (int)m->buf_ch[d->in_buf];
    const int     base  = (int)(d->in_off % (uint32_t)pitch);

    /* Element-wise, so in-place (in_buf == out_buf) is well defined. */
    for (int t = t0; t < t1; t++)
        for (int c = c0; c < c1; c++) {
            int32_t acc = (int32_t)src[(size_t)t * pitch + base + c] - d->in_zp;
            qn_store(m, d, t, c, acc + qp(m, d->bias_off, c));
        }
}

/* ── Tile walker: one loop nest, shared by qn_run_desc() and qn_walk() ───────── */

static int dw_in_len(const qn_model_t *m, const qn_desc_t *d)
{
    return m->t_out * ((d->op == QN_OP_DW) ? d->stride : 1);
}

static void walk_desc(const qn_model_t *m, int i, int execute)
{
    const qn_desc_t *d = &m->desc[i];
    const int t_out  = m->t_out;
    const int t_tile = (int)m->t_tile;
    const int t_in   = dw_in_len(m, d);

    /* Depthwise is channel-independent, so it tiles over channels too; every
     * other op already has c_out bounded by c_out_tile at emission time. */
    const int c_tile = (d->op == QN_OP_DW) ? (int)m->dw_ch_tile : d->c_out;

    int n_t = (t_out + t_tile - 1) / t_tile;
    int n_c = (d->c_out + c_tile - 1) / c_tile;
    int tile_idx = 0, n_tiles = n_t * n_c;

    /* Frames already resident from the previous time tile (halo retention). */
    int resident_end = 0;

    for (int t0 = 0; t0 < t_out; t0 += t_tile) {
        int t1 = t0 + t_tile; if (t1 > t_out) t1 = t_out;

        int in0, in1;
        if (d->op == QN_OP_DW) {
            int lo = t0 * d->stride - d->pad;
            int hi = (t1 - 1) * d->stride + (d->k - 1) * d->dilation - d->pad;
            if (lo < 0) lo = 0;
            if (hi > t_in - 1) hi = t_in - 1;
            in0 = lo; in1 = (hi >= lo) ? hi + 1 : lo;
        } else {
            in0 = t0; in1 = t1;
        }
        int new0 = in0 > resident_end ? in0 : resident_end;
        if (new0 > in1) new0 = in1;
        int new1 = in1;
        resident_end = in1;

        for (int c0 = 0; c0 < d->c_out; c0 += c_tile) {
            int c1 = c0 + c_tile; if (c1 > d->c_out) c1 = d->c_out;

            if (qn_tile_hook) {
                qn_tile_t tl;
                tl.desc_idx  = i;
                tl.op        = d->op;
                tl.t0 = t0; tl.t1 = t1;
                tl.c0 = c0; tl.c1 = c1;
                tl.reduction = (d->op == QN_OP_DW) ? d->k
                             : (d->op == QN_OP_PW) ? d->c_in : 1;
                tl.n_outputs = (t1 - t0) * (c1 - c0);
                tl.t_in0 = in0; tl.t_in1 = in1;
                tl.t_new0 = new0; tl.t_new1 = new1;
                tl.first_tile = (tile_idx == 0);
                tl.last_tile  = (tile_idx == n_tiles - 1);
                qn_tile_hook(&tl);
            }

            if (execute) {
                switch (d->op) {
                case QN_OP_DW:      tile_dw(m, d, t0, t1, c0, c1, t_in); break;
                case QN_OP_PW:      tile_pw(m, d, t0, t1, c0, c1);       break;
                case QN_OP_ADD:     tile_add(m, d, t0, t1, c0, c1);      break;
                case QN_OP_REQUANT: tile_requant(m, d, t0, t1, c0, c1);  break;
                default: break;
                }
            }
            tile_idx++;
        }
    }
}

int qn_run_desc(qn_model_t *m, int i)
{
    if (!m->arena_ready) return QN_ERR_ARENA;
    if (!m->weights || !m->qparams) return QN_ERR_BLOB;
    if (i < 0 || i >= (int)m->n_desc) return QN_ERR_TRUNCATED;
    walk_desc(m, i, 1);
    return QN_OK;
}

int qn_run(qn_model_t *m)
{
    if (!m->arena_ready) return QN_ERR_ARENA;
    if (!m->weights || !m->qparams) return QN_ERR_BLOB;
    for (uint32_t i = 0; i < m->n_desc; i++)
        walk_desc(m, (int)i, 1);
    return QN_OK;
}

void qn_walk(const qn_model_t *m)
{
    for (uint32_t i = 0; i < m->n_desc; i++)
        walk_desc(m, (int)i, 0);
}

/* ── Output extraction ──────────────────────────────────────────────────────── */

int qn_gather_output(const qn_model_t *m, int i, int8_t *dst)
{
    if (!m->arena_ready) return QN_ERR_ARENA;
    if (i < 0 || i >= (int)m->n_desc) return QN_ERR_TRUNCATED;
    const qn_desc_t *d = &m->desc[i];
    const int8_t *src  = qn_arena + m->buf_off[d->out_buf];
    const int pitch    = (int)m->buf_ch[d->out_buf];
    for (int t = 0; t < m->t_out; t++)
        for (int c = 0; c < d->c_out; c++)
            dst[(size_t)t * d->c_out + c] =
                src[(size_t)t * pitch + d->out_off + (uint32_t)c];
    return QN_OK;
}

/* ── CTC greedy decode (port of quartznet_ref.py::ctc_greedy) ────────────────── */

int qn_ctc_greedy_buf(const qn_model_t *m, const int8_t *logits, int pitch,
                      char *text, int text_cap)
{
    const int ncls  = (int)m->n_classes;
    const int blank = (int)m->blank_idx;

    int n = 0, prev = -1;
    for (int t = 0; t < m->t_out; t++) {
        const int8_t *row = logits + (size_t)t * pitch;
        int best = 0;
        for (int c = 1; c < ncls; c++)          /* argmax, first max wins */
            if ((int)row[c] > (int)row[best]) best = c;
        if (best != prev && best != blank) {
            if (n + 1 >= text_cap) return QN_ERR_TRUNCATED;
            text[n++] = (best < QN_N_LABELS) ? qn_labels[best] : '?';
        }
        prev = best;
    }
    if (n >= text_cap) return QN_ERR_TRUNCATED;
    text[n] = '\0';
    return n;
}

int qn_ctc_greedy(const qn_model_t *m, char *text, int text_cap)
{
    if (!m->arena_ready) return QN_ERR_ARENA;
    const int8_t *lg    = qn_arena + m->buf_off[QN_BUF_LOGITS];
    const int     pitch = (int)m->buf_ch[QN_BUF_LOGITS];
    return qn_ctc_greedy_buf(m, lg, pitch, text, text_cap);
}
