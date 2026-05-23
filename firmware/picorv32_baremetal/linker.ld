/* linker.ld — memory layout for PicoRV32 bare-metal firmware
 *
 * Single flat address space: code, data, and stack all in RAM.
 * PicoRV32 resets to address 0x00000000, so .text.start goes first.
 *
 * PHDRS is declared explicitly so the riscv64-linux-gnu linker does not
 * auto-insert a PT_PHDR entry that cannot be placed before address 0.
 */

ENTRY(_start)

MEMORY {
    RAM (rwx) : ORIGIN = 0x00000000, LENGTH = 256K
}

PHDRS {
    ram PT_LOAD FLAGS(7);   /* R + W + X — single LOAD segment covers everything */
}

SECTIONS {
    /* Reset vector must be at address 0 */
    .text 0x00000000 : {
        *(.text.start)      /* startup.S _start label */
        *(.text*)
        *(.rodata*)         /* const arrays: weights, test vectors */
    } > RAM :ram

    .data : {
        *(.data*)
        *(.sdata*)
    } > RAM :ram

    .bss : {
        _bss_start = .;
        *(.bss*)
        *(.sbss*)
        *(COMMON)
        _bss_end = .;
    } > RAM :ram

    /* Stack grows DOWN from the top of RAM */
    _stack_top = ORIGIN(RAM) + LENGTH(RAM);
}
