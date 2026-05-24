/* linker.ld — memory layout for PicoRV32 bare-metal firmware
 *
 * Single flat address space: code, data, and stack all in RAM.
 * PicoRV32 resets to address 0x00000000, so .text.start goes first.
 *
 * 256 KB is plenty for TinyVAD weights (~16 KB) +
 * test vectors (~82 KB) + code + stack.
 */

MEMORY {
    RAM (rwx) : ORIGIN = 0x00000000, LENGTH = 256K
}

SECTIONS {
    /* Reset vector must be at address 0 */
    .text 0x00000000 : {
        *(.text.start)      /* startup.S _start label */
        *(.text*)
        *(.rodata*)         /* const arrays: weights, test vectors */
    } > RAM

    .data : {
        *(.data*)
        *(.sdata*)
    } > RAM

    .bss : {
        _bss_start = .;
        *(.bss*)
        *(.sbss*)
        *(COMMON)
        _bss_end = .;
    } > RAM

    /* Stack grows DOWN from the top of RAM */
    _stack_top = ORIGIN(RAM) + LENGTH(RAM);
}
