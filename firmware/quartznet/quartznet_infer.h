/* quartznet_infer.h
 *
 * Table-driven int8 interpreter for the QuartzNet 15x5 descriptor format emitted
 * by sw/tinyml_reference/quartznet_descriptors.py.
 *
 * Compiles for x86 (host tests) and for riscv32-unknown-elf (bare metal): no
 * libc beyond <stdint.h>/<stddef.h>, no dynamic allocation, one static
 * activation arena.  The weight and qparam blobs are NOT copied — the caller
 * hands in pointers, which on the real part are memory-mapped QSPI flash.
 *
 * Tensor layout: [time, channel] everywhere (TFLite NHWC), matching
 * tiny_vad_infer.c and quartznet_topology.py.  Element (t, c) of activation
 * buffer b lives at
 *
 *     arena[buf_off[b] + t * buf_ch[b] + c]
 *
 * NOTE the row pitch is the BUFFER's allocated channel count, not the
 * descriptor's in_stride/out_stride.  That is what quartznet_ref.py does
 * (RefRunner.buf is shaped [rate*t_out, buffer_channels]) and the golden model
 * is the authority.
 *
 * Numerics are bit-exact with quartznet_ref.py, which in turn is bit-exact with
 * firmware/tinyengine_port/tiny_vad_infer.c:44-63.
 */

#ifndef QUARTZNET_INFER_H
#define QUARTZNET_INFER_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ── Format constants (mirror quartznet_descriptors.py) ─────────────────────── */

#define QN_DESC_MAGIC    0x54444E51u   /* 'QNDT' */
#define QN_DESC_VERSION  1u
#define QN_DESC_WORDS    16
#define QN_DESC_BYTES    64            /* power-of-two stride, on purpose */
#define QN_HEADER_BYTES  64
#define QN_BUFREC_BYTES  8

/* Op codes — must match the OP_ constants in quartznet_topology.py */
#define QN_OP_DW       0
#define QN_OP_PW       1
#define QN_OP_ADD      2
#define QN_OP_REQUANT  3

/* Descriptor flag bits */
#define QN_F_RELU       (1u << 0)
#define QN_F_ACC_FIRST  (1u << 1)      /* reserved in v1 */
#define QN_F_ACC_LAST   (1u << 2)      /* reserved in v1 */
#define QN_F_FUSE_NEXT  (1u << 3)

/* Activation buffer ids — must match the BUF_ constants in quartznet_topology.py */
#define QN_BUF_IN      0
#define QN_BUF_A       1
#define QN_BUF_B       2
#define QN_BUF_R       3
#define QN_BUF_LOGITS  4
#define QN_N_BUFFERS   5

/* TFLite ADD's fixed pre-scale headroom (quartznet_ref.py:ADD_LEFT_SHIFT) */
#define QN_ADD_LEFT_SHIFT 20

/* Golden-file constants (quartznet_ref.py:write_golden) */
#define QN_GOLDEN_MAGIC 0x4F474E51u    /* 'QNGO' */

/* ── Static capacity limits ─────────────────────────────────────────────────── */

#ifndef QN_MAX_DESC
#define QN_MAX_DESC 256                /* 15x5 needs 187 */
#endif

#ifndef QN_MAX_T_OUT
#define QN_MAX_T_OUT 128               /* output frames the arena can hold */
#endif

/* Widest activation arena, in bytes per output frame:
 *   IN 64ch x rate2 + A 512 + B 1024 + R 512 + LOGITS 29 = 2205 for the 15x5
 *   table.  Rounded up to leave room for other configurations.             */
#ifndef QN_ARENA_PER_FRAME
#define QN_ARENA_PER_FRAME 2560
#endif

#define QN_ARENA_BYTES (QN_ARENA_PER_FRAME * QN_MAX_T_OUT)

/* QN_STATIC_ARENA=1 (default): the interpreter owns a static QN_ARENA_BYTES
 * array, sized for the host build's QN_MAX_T_OUT -- unchanged behavior.
 * QN_STATIC_ARENA=0: no array is allocated at all; the caller must point the
 * interpreter at external storage via qn_set_arena() before qn_set_input()/
 * qn_run(). This exists for the RV32 hardware-dispatch build, where the
 * activation arena lives in accelerator PSRAM and the software tile kernels
 * in this file are never exercised -- see firmware/quartznet/qn_accel.h. */
#ifndef QN_STATIC_ARENA
#define QN_STATIC_ARENA 1
#endif

/* Return codes */
#define QN_OK              0
#define QN_ERR_MAGIC      -1
#define QN_ERR_VERSION    -2
#define QN_ERR_TOO_MANY   -3
#define QN_ERR_TRUNCATED  -4
#define QN_ERR_BUFFERS    -5
#define QN_ERR_ARENA      -6
#define QN_ERR_BLOB       -7

/* ── Decoded descriptor ─────────────────────────────────────────────────────── */

typedef struct {
    int op;
    int flags;
    int in_buf, out_buf, res_buf;
    int c_in, c_out;
    int k, stride, dilation, pad;
    int in_stride, out_stride;
    int in_zp, out_zp, res_zp;
    uint32_t in_off, out_off, res_off;    /* channel offsets, in elements */
    uint32_t w_off, bias_off, qmult_off, rshift_off, acc_off;
    int layer_id, block_id;
    /* derived */
    int n_qch;          /* (bias, q_mult, rshift) triples: 3 for ADD, else c_out */
    int32_t w_bytes;    /* int8 weight elements owned by this descriptor */
    int32_t q_bytes;    /* bytes of qparam blob owned by this descriptor */
} qn_desc_t;

/* ── Model ──────────────────────────────────────────────────────────────────── */

typedef struct {
    /* header, in HEADER_FIELDS order */
    uint32_t magic, version, n_desc, desc_stride;
    uint32_t weight_bytes, qparam_bytes, n_classes, blank_idx;
    uint32_t in_ch;
    int32_t  in_zp;
    uint32_t t_tile, c_out_tile, dw_ch_tile, dw_span_max, n_buffers, desc_off;

    uint32_t buf_ch[QN_N_BUFFERS];      /* channels == row pitch */
    uint32_t buf_rate[QN_N_BUFFERS];    /* 2 for BUF_IN, else 1 */
    uint32_t buf_off[QN_N_BUFFERS];     /* byte offset within the arena */

    qn_desc_t desc[QN_MAX_DESC];

    const int8_t  *weights;             /* weight blob, not copied */
    const uint8_t *qparams;             /* qparam blob, not copied */
    size_t weight_avail, qparam_avail;

    int t_out;                          /* output frames for this utterance */
    int arena_ready;                    /* 0 if t_out exceeds QN_MAX_T_OUT */
} qn_model_t;

/* ── Tile schedule hook ─────────────────────────────────────────────────────── */
/*
 * Called once per compute tile, mirroring tinyvad_conv1d_hook in
 * tiny_vad_infer.c:28.  NULL by default; set it to observe the exact schedule
 * the interpreter walks (sim/quartznet_cycles/ drives its cost model from this).
 *
 * There is deliberately NO cycle counting in this file — the hook exports the
 * schedule and nothing more.
 *
 * Halo retention: t_in0/t_in1 is the full input frame span the tile reads.
 * t_new0/t_new1 is the sub-range NOT already resident from the previous time
 * tile of the same descriptor.  Summed over the tiles of a depthwise, the
 * t_new span is exactly the layer's input length — i.e. this is the
 * retain_halo=True traffic model of quartznet_topology.py::activation_traffic().
 */
typedef struct {
    int desc_idx;
    int op;
    int t0, t1;          /* output frames  [t0, t1) */
    int c0, c1;          /* output channels within this descriptor's slice */
    int reduction;       /* MAC reduction length: K for DW, c_in for PW, 1 else */
    int n_outputs;       /* (t1 - t0) * (c1 - c0) */
    int t_in0, t_in1;    /* input frame span read, clamped to the tensor */
    int t_new0, t_new1;  /* sub-span not retained from the previous time tile */
    int first_tile;      /* 1 on the first tile of this descriptor */
    int last_tile;       /* 1 on the last tile of this descriptor */
} qn_tile_t;

typedef void (*qn_tile_fn)(const qn_tile_t *tile);
extern qn_tile_fn qn_tile_hook;

/* ── API ────────────────────────────────────────────────────────────────────── */

/* Parse the descriptor table and bind the blobs.  Does not touch the arena. */
int qn_load(qn_model_t *m,
            const uint8_t *table, size_t table_bytes,
            const int8_t  *weights, size_t weight_bytes,
            const uint8_t *qparams, size_t qparam_bytes,
            int t_out);

/* Point the interpreter at external arena storage instead of the QN_STATIC_ARENA
 * default. Must be called (if at all) before qn_set_input()/qn_run()/qn_buffer().
 * `bytes` is not validated here -- the caller derives it from the same
 * arena-layout arithmetic qn_load() runs (buf_off/buf_ch/buf_rate/t_out). */
void qn_set_arena(int8_t *arena, size_t bytes);

/* Zero the arena and load the mel input into BUF_IN ([t_in, in_ch] int8). */
int qn_set_input(qn_model_t *m, const int8_t *input);

/* Execute descriptor i (all its tiles).  Requires qn_set_input() first. */
int qn_run_desc(qn_model_t *m, int i);

/* Execute every descriptor in order. */
int qn_run(qn_model_t *m);

/* Walk the tile schedule without computing anything (no arena, no blobs
 * needed).  Fires qn_tile_hook for every tile qn_run() would execute. */
void qn_walk(const qn_model_t *m);

/* Copy descriptor i's output slice out of the arena into `dst`, contiguous
 * [t_out, c_out] — the layout quartznet_ref.py::write_golden() stores. */
int qn_gather_output(const qn_model_t *m, int i, int8_t *dst);

/* Pointer to activation buffer b (NULL if the arena is not usable). */
int8_t *qn_buffer(const qn_model_t *m, int b);

/* CTC greedy decode of BUF_LOGITS: argmax per frame, collapse repeats, drop
 * blanks.  Port of quartznet_ref.py::ctc_greedy().  Writes a NUL-terminated
 * string to `text` and returns the symbol count (or a negative error). */
int qn_ctc_greedy(const qn_model_t *m, char *text, int text_cap);

/* Same decode, over caller-supplied logits (row-major [m->t_out, pitch], pitch
 * >= m->n_classes) instead of the arena's BUF_LOGITS. The hardware-dispatch
 * path uses this to decode logits read back from accelerator PSRAM without
 * an arena. qn_ctc_greedy() is a thin wrapper over this. */
int qn_ctc_greedy_buf(const qn_model_t *m, const int8_t *logits, int pitch,
                      char *text, int text_cap);

/* Label alphabet: " " + a..z + "'" (28 labels; class 28 is the CTC blank). */
extern const char qn_labels[];
#define QN_N_LABELS 28

#ifdef __cplusplus
}
#endif

#endif /* QUARTZNET_INFER_H */
