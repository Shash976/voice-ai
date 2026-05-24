# Makefile — cross-compile TinyVAD firmware for PicoRV32 (RV32IMC bare-metal)
#
# Prerequisites:
#   riscv64-linux-gnu-gcc   (or riscv32-unknown-elf-gcc — change CC below)
#
# Before building, ensure these generated headers exist in ../tinyengine_port/:
#   tiny_vad_weights.h       (python sw/tinyml_reference/export_weights.py)
#   tiny_vad_test_vectors.h  (python sw/tinyml_reference/gen_test_vectors.py)
#
# Targets:
#   make             build firmware.bin
#   make size        show section sizes
#   make disasm      dump disassembly (check for illegal FP instructions)
#   make clean

CC      = riscv64-linux-gnu-gcc
OBJCOPY = riscv64-linux-gnu-objcopy
OBJDUMP = riscv64-linux-gnu-objdump
SIZE    = riscv64-linux-gnu-size

CFLAGS = \
    -march=rv32imc \
    -mabi=ilp32 \
    -O2 \
    -ffreestanding \
    -nostdlib \
    -fno-builtin \
    -fno-pic \
    -fno-pie \
    -ffunction-sections \
    -fdata-sections \
    -Wall \
    -I. \
    -I../tinyengine_port

LDFLAGS = -T linker.ld -Wl,--gc-sections -no-pie -Wl,--build-id=none

SRCS = startup.S \
       main.c \
       accel.c \
       ../tinyengine_port/tiny_vad_infer.c

OBJS = $(patsubst %.S,%.o,$(patsubst %.c,%.o,$(notdir $(SRCS))))

.PHONY: all size disasm clean

all: firmware.bin

firmware.elf: $(OBJS)
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $^

firmware.bin: firmware.elf
	$(OBJCOPY) -O binary $< $@
	@echo "firmware.bin: $$(wc -c < firmware.bin) bytes"

startup.o: startup.S
	$(CC) $(CFLAGS) -c -o $@ $<

main.o: main.c
	$(CC) $(CFLAGS) -c -o $@ $<

accel.o: accel.c accel.h
	$(CC) $(CFLAGS) -c -o $@ $<

tiny_vad_infer.o: ../tinyengine_port/tiny_vad_infer.c
	$(CC) $(CFLAGS) -c -o $@ $<

size: firmware.elf
	$(SIZE) firmware.elf

disasm: firmware.elf
	$(OBJDUMP) -d firmware.elf | head -300

clean:
	rm -f *.o *.elf *.bin
