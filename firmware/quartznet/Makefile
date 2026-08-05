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
#   make rv32       cross-compile the interpreter for riscv32-unknown-elf
#   make clean
#
# Prerequisites:
#   host/schedule: gcc
#   goldens:       python3 + numpy  (system python is enough; no torch/onnx)
#   rv32:          riscv32-unknown-elf-gcc     -- NOT installed on this machine
#
# The goldens are gitignored build products; regenerate them from the seed at
# any time, they are reproducible.

CC_HOST = gcc
CC_RV32 = riscv32-unknown-elf-gcc
PYTHON  = python3

ROOT    = ../..
REF     = $(ROOT)/sw/tinyml_reference/quartznet_ref.py
BUILD   = $(ROOT)/build

CFLAGS_COMMON = -O2 -Wall -Wextra -I.
CFLAGS_HOST   = $(CFLAGS_COMMON) -std=c11

# RV32 bare-metal flags, same set tiny_vad_infer.c uses.
CFLAGS_RV32 = $(CFLAGS_COMMON) -std=c11 \
              -march=rv32imc -mabi=ilp32 \
              -ffreestanding -fno-builtin \
              -nostdlib -Os \
              -ffunction-sections -fdata-sections

SRCS = quartznet_infer.c

# ── goldens ───────────────────────────────────────────────────────────────────

.PHONY: goldens
goldens:
	cd $(ROOT)/sw/tinyml_reference && $(PYTHON) quartznet_ref.py --reduced
	cd $(ROOT)/sw/tinyml_reference && $(PYTHON) quartznet_ref.py

$(BUILD)/quartznet/quartznet_golden.bin $(BUILD)/quartznet_reduced/quartznet_golden.bin: $(REF)
	$(MAKE) goldens

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

qn_transcribe: $(SRCS) qn_transcribe.c quartznet_infer.h
	$(CC_HOST) $(CFLAGS_HOST) -o $@ $(SRCS) qn_transcribe.c

# ── tile-schedule dumper (feeds sim/quartznet_cycles) ─────────────────────────

.PHONY: schedule
schedule: qn_schedule

qn_schedule: $(SRCS) qn_schedule.c quartznet_infer.h
	$(CC_HOST) $(CFLAGS_HOST) -o $@ $(SRCS) qn_schedule.c

# ── RV32 object ───────────────────────────────────────────────────────────────

.PHONY: rv32
rv32: quartznet_infer_rv32.o
	$(CC_RV32) -print-file-name=libgcc.a
	size quartznet_infer_rv32.o

quartznet_infer_rv32.o: $(SRCS) quartznet_infer.h
	$(CC_RV32) $(CFLAGS_RV32) -c -o $@ quartznet_infer.c

# ── clean ─────────────────────────────────────────────────────────────────────

.PHONY: clean
clean:
	rm -f test_quartznet_host test_quartznet_host.exe \
	      qn_schedule qn_schedule.exe \
	      qn_transcribe qn_transcribe.exe quartznet_infer_rv32.o
