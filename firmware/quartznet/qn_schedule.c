/* qn_schedule.c
 *
 * Dumps the exact tile schedule quartznet_infer.c walks, as CSV, for an
 * arbitrary utterance length.  This is the bridge that lets
 * sim/quartznet_cycles/ cost-model the REAL schedule instead of a second,
 * drift-prone copy of the tiling logic: it drives qn_walk(), which is the same
 * loop nest qn_run() executes, via qn_tile_hook.
 *
 * Build:  make schedule
 * Run:    ./qn_schedule <desc.bin> <t_out>       > schedule.csv
 *
 * No weights, no qparams, no arena — qn_walk() needs none of them, so t_out can
 * far exceed QN_MAX_T_OUT (a 10 s utterance is 500 output frames).
 *
 * Output has three sections, each introduced by a '#' banner line:
 *   #MODEL   key,value            header fields + per-buffer geometry
 *   #DESC    one row per descriptor
 *   #TILE    one row per compute tile, in execution order
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#include "quartznet_infer.h"

#define MAX_TABLE (64u * 1024u)
static uint32_t table_blob32[MAX_TABLE / 4];
static qn_model_t model;

static void emit_tile(const qn_tile_t *t)
{
    printf("%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d\n",
           t->desc_idx, t->op, t->t0, t->t1, t->c0, t->c1,
           t->reduction, t->n_outputs,
           t->t_in0, t->t_in1, t->t_new0, t->t_new1,
           t->first_tile, t->last_tile);
}

int main(int argc, char **argv)
{
    if (argc != 3) {
        fprintf(stderr, "usage: %s <quartznet_desc.bin> <t_out>\n", argv[0]);
        return 2;
    }
    FILE *f = fopen(argv[1], "rb");
    if (!f) { perror(argv[1]); return 2; }
    long n = (long)fread(table_blob32, 1, MAX_TABLE, f);
    fclose(f);

    int t_out = atoi(argv[2]);
    if (t_out <= 0) { fprintf(stderr, "t_out must be > 0\n"); return 2; }

    int rc = qn_load(&model, (const uint8_t *)table_blob32, (size_t)n,
                     NULL, 0, NULL, 0, t_out);
    if (rc != QN_OK) { fprintf(stderr, "qn_load failed: %d\n", rc); return 2; }

    printf("#MODEL\n");
    printf("key,value\n");
    printf("n_desc,%u\n",      model.n_desc);
    printf("weight_bytes,%u\n", model.weight_bytes);
    printf("qparam_bytes,%u\n", model.qparam_bytes);
    printf("n_classes,%u\n",   model.n_classes);
    printf("t_tile,%u\n",      model.t_tile);
    printf("c_out_tile,%u\n",  model.c_out_tile);
    printf("dw_ch_tile,%u\n",  model.dw_ch_tile);
    printf("dw_span_max,%u\n", model.dw_span_max);
    printf("t_out,%d\n",       model.t_out);
    for (uint32_t b = 0; b < model.n_buffers; b++)
        printf("buf%u_ch_rate,%u:%u\n", b, model.buf_ch[b], model.buf_rate[b]);

    printf("#DESC\n");
    printf("idx,op,flags,c_in,c_out,k,stride,dilation,pad,"
           "in_buf,out_buf,res_buf,in_stride,out_stride,w_bytes,q_bytes\n");
    for (uint32_t i = 0; i < model.n_desc; i++) {
        const qn_desc_t *d = &model.desc[i];
        printf("%u,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d\n",
               i, d->op, d->flags, d->c_in, d->c_out, d->k, d->stride,
               d->dilation, d->pad, d->in_buf, d->out_buf, d->res_buf,
               d->in_stride, d->out_stride, d->w_bytes, d->q_bytes);
    }

    printf("#TILE\n");
    printf("desc,op,t0,t1,c0,c1,reduction,n_outputs,"
           "t_in0,t_in1,t_new0,t_new1,first,last\n");
    qn_tile_hook = emit_tile;
    qn_walk(&model);
    qn_tile_hook = NULL;

    return 0;
}
