# Makefile — QuartzNet 15x5 descriptor interpreter
#
# Targets:
#   make host       build and run the bit-exact host test (both configurations)
#   make goldens    regenerate the reference blobs both configurations need
#   make schedule   build ./qn_schedule (tile-schedule dumper for the cost model)
#   make transcribe mp3/wav -> features -> interpreter -> CTC transcript smoke
#                   test (seeded-random weights -- gibberish expected, proves
#                   the plumbing only). MP3=path/to/clip.mp3, or omit for a
#                   synthetic clip.
#   make rv32       cross-compile the interpreter alone for riscv32 (object only)
#   make gen-headers  regenerate qn_table.h/qn_input.h (linked-in const arrays,
#                     reduced-config table+input) for the rv32-fw build
#   make rv32-fw    build qn_firmware.bin: PicoRV32 firmware that drives the
#                   QuartzNet accelerator over MMIO (qn_accel.c) and prints a
#                   CTC transcript over UART -- see sim/verilator_qn to run it
#   make rv32-size, rv32-disasm
#   make clean
#
# Prerequisites:
#   host/schedule/rv32-fw: gcc, riscv64-linux-gnu-gcc (see CROSS_RV32 below)
#   goldens:                python3 + numpy  (system python is enough; no torch/onnx)
#
# The goldens are gitignored build products; regenerate them from the seed at
# any time, they are reproducible.

CC_HOST = gcc
PYTHON  = python3

ROOT    = ../..
REF     = $(ROOT)/sw/tinyml_reference/quartznet_ref.py
BUILD   = $(ROOT)/build
BM      = ../picorv32_baremetal

CFLAGS_COMMON = -O2 -Wall -Wextra -I.
CFLAGS_HOST   = $(CFLAGS_COMMON) -std=c11

# RV32 bare-metal toolchain. riscv32-unknown-elf-gcc is not installed on the
# machines this repo has been built on; riscv64-linux-gnu-gcc targeting
# rv32imc/ilp32 is proven to work (firmware/picorv32_baremetal builds and runs
# with it) -- see CROSS_RV32 below. -fno-pic -fno-pie is required: this
# toolchain defaults to PIE even with -nostdlib, which produces GOT-indirect
# loads for linker symbols like _stack_top and fails at link time (see
# CLAUDE.md's Firmware Build Flags section).
CROSS_RV32 ?= riscv64-linux-gnu
CC_RV32      = $(CROSS_RV32)-gcc
OBJCOPY_RV32 = $(CROSS_RV32)-objcopy
OBJDUMP_RV32 = $(CROSS_RV32)-objdump
SIZE_RV32    = $(CROSS_RV32)-size

# No -std= flag: defaults to gnu17/gnu11, matching picorv32_baremetal's own
# Makefile -- syscalls.h's rdcycle() uses the GNU `asm` keyword, which -std=c11
# (strict ISO) rejects (needs __asm__ instead).
CFLAGS_RV32 = $(CFLAGS_COMMON) \
              -march=rv32imc -mabi=ilp32 \
              -ffreestanding -fno-builtin \
              -nostdlib -Os \
              -fno-pic -fno-pie \
              -ffunction-sections -fdata-sections

CFLAGS_RV32_FW = $(CFLAGS_RV32) -I$(BM)

LDFLAGS_RV32_FW = -T $(BM)/linker.ld -Wl,--gc-sections -no-pie -Wl,--build-id=none

SRCS = quartznet_infer.c
SRCS_RV32_FW = $(BM)/startup.S qn_main.c qn_accel.c quartznet_infer.c

# ── goldens ───────────────────────────────────────────────────────────────────

.PHONY: goldens
goldens:
	cd $(ROOT)/sw/tinyml_reference && $(PYTHON) quartznet_ref.py --reduced
	cd $(ROOT)/sw/tinyml_reference && $(PYTHON) quartznet_ref.py

$(BUILD)/quartznet/quartznet_golden.bin $(BUILD)/quartznet_reduced/quartznet_golden.bin: $(REF)
	$(MAKE) goldens

# ── real-weights config (Stage 7 Gap 2 A5/G2.5) ────────────────────────────────
# Needs venv (onnx, for quartznet_export_int8.py) + build/quartznet_int8/
# (quartznet_calibrate.py's output, Stage 7 Gap 2 A4) + LibriSpeech -- none of
# which a bare gcc+numpy machine has, so this is a SEPARATE target from
# `goldens`/`host`, never folded into them (`make host` must keep working on
# a machine with only the host toolchain + system numpy).
VENV_PYTHON ?= $(ROOT)/venv/bin/python3
BUILD_REAL   = $(BUILD)/quartznet_real
REAL_AUDIO  ?= $(ROOT)/librispeech/LibriSpeech/dev-clean/1272/128104/1272-128104-0008.flac
REAL_T_OUT  ?= 70

.PHONY: goldens-real
goldens-real:
	cd $(ROOT)/sw/tinyml_reference && $(VENV_PYTHON) quartznet_export_int8.py \
	    --audio $(REAL_AUDIO) --t-out $(REAL_T_OUT) --out $(BUILD_REAL)
	cd $(ROOT)/sw/tinyml_reference && $(PYTHON) quartznet_ref.py --weights-from $(BUILD_REAL)

.PHONY: host-real
host-real: test_quartznet_host $(BUILD_REAL)/quartznet_golden.bin
	./test_quartznet_host $(BUILD_REAL)

$(BUILD_REAL)/quartznet_golden.bin:
	$(MAKE) goldens-real

# ── host test ─────────────────────────────────────────────────────────────────

.PHONY: host
host: test_quartznet_host \
      $(BUILD)/quartznet_reduced/quartznet_golden.bin \
      $(BUILD)/quartznet/quartznet_golden.bin
	./test_quartznet_host

test_quartznet_host: $(SRCS) test_quartznet_host.c quartznet_infer.h
	$(CC_HOST) $(CFLAGS_HOST) -o $@ $(SRCS) test_quartznet_host.c

# ── mp3-to-text mechanical smoke test ───────────────────────────────────────────
# Weights/qparams come from `make goldens`'s seeded-random blobs, unmodified --
# real trained weights are a separate later phase. This only proves mp3 ->
# quartznet_audio.py -> mp3_input.bin -> qn_load/qn_set_input/qn_run ->
# qn_ctc_greedy plumbing works end to end; the transcript is expected garbage.

.PHONY: transcribe
transcribe: qn_transcribe \
      $(BUILD)/quartznet_reduced/quartznet_golden.bin \
      $(BUILD)/quartznet/quartznet_golden.bin
	$(PYTHON) $(ROOT)/sw/tinyml_reference/mp3_to_text.py $(MP3)
	./qn_transcribe

# QN_MAX_T_OUT: the stock 128 caps the activation arena at t_out~=148
# (QN_ARENA_PER_FRAME=2560 B/frame headroom vs the real ~2205 B/frame used --
# arena_ready is `2205*t_out <= 2560*QN_MAX_T_OUT`). test-clean's longest
# utterance is t_out=1748 (34.96s); 2048 covers it with margin, 5.24 MB BSS,
# host-only (never touches the RV32 firmware build, which has its own much
# shorter T_OUT). QN_OPT: -O3 -march=native measured BIT-IDENTICAL to -O2
# over 200 real utterances (~74k output frames) -- pure speed (1.88x), not a
# behavior change -- but -O2 stays the canonical default since it's what
# `make host`/`host-real`'s bit-exactness gates already build with.
QN_MAX_T_OUT ?= 2048
QN_OPT       ?= -O2

qn_transcribe: $(SRCS) qn_transcribe.c quartznet_infer.h
	$(CC_HOST) $(CFLAGS_HOST) $(QN_OPT) -DQN_MAX_T_OUT=$(QN_MAX_T_OUT) \
	    -o $@ $(SRCS) qn_transcribe.c

# ── A6/G2.6: real weights, real audio, legible transcript ────────────────────
# mp3_to_text.py only needs numpy+soundfile ($(PYTHON) is enough, no venv);
# quartznet_export_int8.py (via goldens-real) is what needs $(VENV_PYTHON).
.PHONY: transcribe-real
transcribe-real: qn_transcribe $(BUILD_REAL)/quartznet_weights.bin
	$(PYTHON) $(ROOT)/sw/tinyml_reference/mp3_to_text.py $(MP3) \
	    --model-dir $(BUILD_REAL) --out-dir $(BUILD_REAL)
	./qn_transcribe $(BUILD_REAL) $(BUILD_REAL)/mp3_input.bin

# ── A6/G2.6: full-corpus WER through the firmware interpreter ────────────────
SPLIT ?= test-clean
.PHONY: wer-firmware
wer-firmware: qn_transcribe $(BUILD_REAL)/quartznet_weights.bin
	cd $(ROOT)/sw/tinyml_reference && $(VENV_PYTHON) quartznet_run_firmware.py \
	    --split $(SPLIT) --binary $(CURDIR)/qn_transcribe --model-dir $(BUILD_REAL)

# ── tile-schedule dumper (feeds sim/quartznet_cycles) ─────────────────────────

.PHONY: schedule
schedule: qn_schedule

qn_schedule: $(SRCS) qn_schedule.c quartznet_infer.h
	$(CC_HOST) $(CFLAGS_HOST) -o $@ $(SRCS) qn_schedule.c

# ── RV32 object (interpreter alone, no link -- back-compat) ──────────────────

.PHONY: rv32
rv32: quartznet_infer_rv32.o
	$(SIZE_RV32) quartznet_infer_rv32.o

quartznet_infer_rv32.o: $(SRCS) quartznet_infer.h
	$(CC_RV32) $(CFLAGS_RV32) -c -o $@ quartznet_infer.c

# ── RV32 firmware (links qn_main.c + qn_accel.c: drives the accelerator over
# MMIO and prints a transcript over UART -- see sim/verilator_qn to run it) ──

# t_out=70 / in_ch=64 matches quartznet_ref.py's fixed default (--t-out 70,
# both configs) -- the same table/input `make host` already verifies
# bit-exact. GOLDEN_DIR selects reduced (default, fast RTL iteration) vs full
# (make gen-headers GOLDEN_DIR=$(BUILD)/quartznet -- Stage 7 Gap 1 D6).
GOLDEN_DIR ?= $(BUILD)/quartznet_reduced
T_OUT      ?= 70

.PHONY: gen-headers
gen-headers: $(GOLDEN_DIR)/quartznet_golden.bin
	$(PYTHON) gen_fw_headers.py $(GOLDEN_DIR) $(T_OUT) 64

.PHONY: rv32-fw rv32-size rv32-disasm
rv32-fw: qn_firmware.bin

qn_firmware.elf: gen-headers $(SRCS_RV32_FW) qn_accel.h quartznet_infer.h qn_table.h qn_input.h
	$(CC_RV32) $(CFLAGS_RV32_FW) -DQN_STATIC_ARENA=0 $(LDFLAGS_RV32_FW) \
	    -o $@ $(SRCS_RV32_FW) -lgcc

qn_firmware.bin: qn_firmware.elf
	$(OBJCOPY_RV32) -O binary $< $@
	@echo "qn_firmware.bin: $$(wc -c < qn_firmware.bin) bytes"

rv32-size: qn_firmware.elf
	$(SIZE_RV32) qn_firmware.elf

rv32-disasm: qn_firmware.elf
	$(OBJDUMP_RV32) -d qn_firmware.elf

# ── clean ─────────────────────────────────────────────────────────────────────

.PHONY: clean
clean:
	rm -f test_quartznet_host test_quartznet_host.exe \
	      qn_schedule qn_schedule.exe \
	      qn_transcribe qn_transcribe.exe quartznet_infer_rv32.o \
	      qn_firmware.elf qn_firmware.bin qn_table.h qn_input.h
