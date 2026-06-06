# Makefile — TinyVAD C inference engine
#
# Targets:
#   make host     build and run x86 test (verifies C vs TFLite golden vectors)
#   make rv32     cross-compile for riscv32-unknown-elf (bare-metal)
#   make clean    remove build artifacts
#
# Prerequisites:
#   host:  gcc
#   rv32:  riscv32-unknown-elf-gcc  (or riscv64-unknown-elf-gcc -march=rv32imc)
#
# Before building, run (in repo root):
#   python sw/tinyml_reference/export_weights.py    → tiny_vad_weights.h
#   python sw/tinyml_reference/gen_test_vectors.py  → tiny_vad_test_vectors.h

CC_HOST = gcc
CC_RV32 = riscv32-unknown-elf-gcc

CFLAGS_COMMON = -O2 -Wall -Wextra -I.

# Host flags — allow int64 and standard libc
CFLAGS_HOST = $(CFLAGS_COMMON) -std=c11

# RV32 bare-metal flags
#   -march=rv32imc  : base integer + multiply + compressed (no FPU)
#   -mabi=ilp32     : 32-bit integer ABI, no float registers
#   -ffreestanding  : no implicit libc
#   -fno-builtin    : no compiler-inserted library calls
#   -Os             : optimize for size (important for on-chip SRAM)
CFLAGS_RV32 = $(CFLAGS_COMMON) -std=c11 \
              -march=rv32imc -mabi=ilp32 \
              -ffreestanding -fno-builtin \
              -nostdlib -Os \
              -ffunction-sections -fdata-sections

SRCS_INFER = tiny_vad_infer.c

# ── host test ────────────────────────────────────────────────────────────────

.PHONY: host
host: test_infer_host
	./test_infer_host

test_infer_host: $(SRCS_INFER) test_infer_host.c tiny_vad_weights.h tiny_vad_test_vectors.h
	$(CC_HOST) $(CFLAGS_HOST) -o $@ $(SRCS_INFER) test_infer_host.c

# ── im2col conv-lowering test ─────────────────────────────────────────────────
# Verifies the im2col conv→MATVEC path (conv_im2col.h) is bit-exact with the
# direct conv1d — i.e. the MATVEC-only RTL datapath can run the conv layers.

.PHONY: im2col
im2col: test_im2col_host
	./test_im2col_host

test_im2col_host: $(SRCS_INFER) test_im2col_host.c conv_im2col.h tiny_vad_weights.h tiny_vad_test_vectors.h
	$(CC_HOST) $(CFLAGS_HOST) -o $@ $(SRCS_INFER) test_im2col_host.c

# ── RV32 firmware object ─────────────────────────────────────────────────────
# Produces tiny_vad_infer.o ready to link into PicoRV32 firmware.

.PHONY: rv32
rv32: tiny_vad_infer_rv32.o
	@echo "Built tiny_vad_infer_rv32.o"
	$(CC_RV32) -print-file-name=libgcc.a
	size tiny_vad_infer_rv32.o

tiny_vad_infer_rv32.o: $(SRCS_INFER) tiny_vad_weights.h
	$(CC_RV32) $(CFLAGS_RV32) -c -o $@ tiny_vad_infer.c

# Check code / data section sizes without full link
.PHONY: rv32-size
rv32-size: tiny_vad_infer_rv32.o
	riscv32-unknown-elf-objdump -h tiny_vad_infer_rv32.o | grep -E "text|data|bss"

# Disassembly — useful to verify no FP instructions slipped in
.PHONY: rv32-disasm
rv32-disasm: tiny_vad_infer_rv32.o
	riscv32-unknown-elf-objdump -d tiny_vad_infer_rv32.o | head -200

# ── check headers exist ───────────────────────────────────────────────────────

tiny_vad_weights.h:
	@echo "ERROR: tiny_vad_weights.h not found."
	@echo "       Run:  python ../../sw/tinyml_reference/export_weights.py"
	@exit 1

tiny_vad_test_vectors.h:
	@echo "ERROR: tiny_vad_test_vectors.h not found."
	@echo "       Run:  python ../../sw/tinyml_reference/gen_test_vectors.py"
	@exit 1

# ── clean ─────────────────────────────────────────────────────────────────────

.PHONY: clean
clean:
	rm -f test_infer_host test_infer_host.exe \
	      test_im2col_host test_im2col_host.exe tiny_vad_infer_rv32.o
