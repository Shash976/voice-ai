/* main.c — TinyVAD inference benchmark firmware for PicoRV32
 *
 * Runs every test vector through tiny_vad_infer(), measures cycles,
 * and prints a CSV-style report over UART.
 *
 * Output format (one line per vector):
 *   vec,label,result,correct,logit0,logit1,cycles
 *
 * Stage 3 goal: establish the PicoRV32 software baseline cycle count.
 * Stage 4 will replace the hot loops with accelerator calls and compare.
 *
 * Build:  make  (in firmware/picorv32_baremetal/)
 * Run:    make -C ../../sim/verilator run
 */

#include <stdint.h>
#include "syscalls.h"
#include "accel.h"
#include "../tinyengine_port/tiny_vad_infer.h"
#include "../tinyengine_port/tiny_vad_weights.h"
#include "../tinyengine_port/tiny_vad_test_vectors.h"

int main(void)
{
    /* Route conv1d and dense layers to the hardware accelerator.
     * With -DACCEL_CONV_IM2COL (make CONV_IM2COL=1) conv layers are lowered to
     * im2col + MATVEC, exercising the MATVEC-only RTL datapath; otherwise the
     * behavioral CONV1D op (cmd=2) is used. */
#ifdef ACCEL_CONV_IM2COL
    tinyvad_conv1d_hook = accel_conv1d_im2col;
#else
    tinyvad_conv1d_hook = accel_conv1d;
#endif
    tinyvad_dense_hook  = accel_dense;

    uart_puts("vec,label,result,correct,logit0,logit1,cycles\n");

    int8_t  logits[2];
    int     n_correct = 0;
    int     n_total   = N_TEST_VECTORS;
    uint32_t total_cycles = 0;

    for (int v = 0; v < n_total; v++) {
        uint32_t t0     = rdcycle();
        int      result = tiny_vad_infer(test_inputs[v], logits);
        uint32_t cycles = rdcycle() - t0;

        int label   = (int)test_labels[v];
        int correct = (result == label) ? 1 : 0;
        if (correct) n_correct++;
        total_cycles += cycles;

        /* vec index */
        uart_putu32((uint32_t)v);      uart_putc(',');
        /* ground-truth label */
        uart_putu32((uint32_t)label);  uart_putc(',');
        /* model prediction */
        uart_putu32((uint32_t)result); uart_putc(',');
        /* correct flag */
        uart_putu32((uint32_t)correct);uart_putc(',');
        /* logit[0] and logit[1] */
        uart_puti32((int32_t)logits[0]); uart_putc(',');
        uart_puti32((int32_t)logits[1]); uart_putc(',');
        /* cycle count for this inference */
        uart_putu32(cycles);
        uart_putc('\n');
    }

    /* Summary */
    uart_puts("---\n");
    uart_puts("correct=");   uart_putu32((uint32_t)n_correct);
    uart_puts("/");          uart_putu32((uint32_t)n_total);
    uart_puts(" avg_cycles=");
    uart_putu32(total_cycles / (uint32_t)n_total);
    uart_putc('\n');

    /* Exit with number of failures (0 = all correct) */
    sim_exit(n_total - n_correct);
    return 0;
}
