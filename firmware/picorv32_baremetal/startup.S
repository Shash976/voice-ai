/* startup.S — reset handler for PicoRV32 bare-metal firmware
 *
 * On reset, PicoRV32 fetches from address 0x00000000.
 * This file MUST be placed first in the linker output (.text.start).
 *
 * Responsibilities:
 *   1. Set stack pointer to top of RAM
 *   2. Zero the .bss section  (C assumes BSS is zero-initialised)
 *   3. Call main()
 *   4. If main returns, halt in an infinite loop
 */

.section .text.start
.global _start
_start:
    /* Stack grows downward from _stack_top (defined in linker.ld) */
    la   sp, _stack_top

    /* Zero .bss: from _bss_start to _bss_end (word-aligned by linker) */
    la   a0, _bss_start
    la   a1, _bss_end
.bss_loop:
    bge  a0, a1, .bss_done
    sw   zero, 0(a0)
    addi a0, a0, 4
    j    .bss_loop
.bss_done:

    call main

    /* main() returned — halt */
.halt:
    j .halt
