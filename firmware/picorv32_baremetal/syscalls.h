/* syscalls.h — bare-metal I/O for PicoRV32 simulation
 *
 * Memory-mapped I/O (enforced by Verilator testbench):
 *   0x10000000  UART_TX   write a byte → testbench prints to stdout
 *   0x10000004  SIM_EXIT  write any value → end simulation
 *
 * CSRs:
 *   rdcycle → 64-bit hardware cycle counter (ENABLE_COUNTERS=1 in SoC)
 */

#ifndef SYSCALLS_H
#define SYSCALLS_H

#include <stdint.h>

#define UART_TX_REG  (*(volatile uint32_t *)0x10000000u)
#define SIM_EXIT_REG (*(volatile uint32_t *)0x10000004u)

/* ── UART ──────────────────────────────────────────────────────────────────── */

static inline void uart_putc(char c)
{
    UART_TX_REG = (uint32_t)(uint8_t)c;
}

static inline void uart_puts(const char *s)
{
    while (*s) uart_putc(*s++);
}

static void uart_putu32(uint32_t v)
{
    if (v == 0) { uart_putc('0'); return; }
    char buf[10];
    int  i = 0;
    while (v > 0) { buf[i++] = '0' + (int)(v % 10); v /= 10; }
    while (i > 0) uart_putc(buf[--i]);
}

static void uart_puti32(int32_t v)
{
    if (v < 0) { uart_putc('-'); uart_putu32((uint32_t)(-v)); }
    else        uart_putu32((uint32_t)v);
}

/* ── Cycle counter ─────────────────────────────────────────────────────────── */

/* rdcycle reads the lower 32 bits of the hardware cycle counter.
 * Wraps every ~4 billion cycles (~4s at 1 GHz).
 * For inference timing (a few million cycles) this is more than sufficient. */
static inline uint32_t rdcycle(void)
{
    uint32_t c;
    asm volatile ("rdcycle %0" : "=r"(c));
    return c;
}

/* ── Simulation exit ───────────────────────────────────────────────────────── */

static inline void sim_exit(int code)
{
    SIM_EXIT_REG = (uint32_t)code;
    /* Testbench stops simulation; this never returns */
    while (1) {}
}

#endif /* SYSCALLS_H */
