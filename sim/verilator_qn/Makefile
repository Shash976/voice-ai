# Makefile — build and run the QuartzNet PicoRV32 Verilator simulation
#
# MODE=shim (default, this increment): Verilates the UNMODIFIED
#   rtl/soc/picorv32_soc.v; sim_main_qn.cpp emulates the accelerator's
#   register file in C++ but delegates every actual computation to the real
#   firmware/quartznet/quartznet_infer.c interpreter, so this mode cannot
#   silently drift from the golden model. Existed to develop/debug
#   qn_accel.c/qn_main.c fast, before RTL exists.
# MODE=rtl (Stage 7 Gap 1 increment D5): Verilates rtl/soc/qn_soc.v (real
#   quartznet_accel.v wired onto the bus) instead -- not yet added.
#
# Prerequisites: verilator, riscv64-linux-gnu-gcc (for the firmware), python3
# (for goldens -- see firmware/quartznet/Makefile).
#
# Targets:
#   make              build the shim simulator
#   make firmware     build firmware/quartznet's qn_firmware.bin + QSPI blobs
#   make run          build everything and run the reduced-config table walk
#   make vcd          same + VCD waveform dump -> sim_out.vcd
#   make clean

MODE ?= shim

REPO_ROOT     = ../..
PICORV32_V    = $(REPO_ROOT)/rtl/picorv32/picorv32.v
SOC_V         = $(REPO_ROOT)/rtl/soc/picorv32_soc.v
QN_DIR        = $(REPO_ROOT)/firmware/quartznet
QN_INFER_C    = $(QN_DIR)/quartznet_infer.c
FIRMWARE_BIN  = $(QN_DIR)/qn_firmware.bin
GOLDEN_DIR    = $(REPO_ROOT)/build/quartznet_reduced

# -I here must be absolute: verilator's --build invokes make from obj_dir/,
# two directories deeper than this Makefile, so a relative $(QN_DIR) would
# resolve wrong.
QN_DIR_ABS = $(abspath $(QN_DIR))

VERILATOR       = verilator
VERILATOR_FLAGS = \
    --cc \
    --exe \
    --build \
    --trace \
    -O2 \
    -CFLAGS "-O2 -std=c++17 -I$(QN_DIR_ABS)" \
    --Mdir obj_dir \
    --top-module picorv32_soc

SIM_BIN = sim_qn_shim

.PHONY: all firmware run vcd clean

all: $(SIM_BIN)

# ── Build simulator (MODE=shim only; MODE=rtl lands with qn_soc.v) ─────────────

ifeq ($(MODE),shim)
$(SIM_BIN): $(SOC_V) $(PICORV32_V) sim_main_qn.cpp $(QN_INFER_C)
	$(VERILATOR) $(VERILATOR_FLAGS) \
	    -o $(SIM_BIN) \
	    $(SOC_V) $(PICORV32_V) \
	    sim_main_qn.cpp $(QN_INFER_C)
	cp obj_dir/$(SIM_BIN) .
else
$(SIM_BIN):
	@echo "MODE=rtl not implemented yet -- Stage 7 Gap 1 increment D5 (qn_soc.v)" && exit 1
endif

# ── Build firmware + QSPI blobs ─────────────────────────────────────────────

firmware:
	$(MAKE) -C $(QN_DIR) rv32-fw

$(GOLDEN_DIR)/quartznet_desc.bin $(GOLDEN_DIR)/quartznet_weights.bin $(GOLDEN_DIR)/quartznet_qparams.bin:
	$(MAKE) -C $(QN_DIR) goldens

# ── Run ─────────────────────────────────────────────────────────────────────

run: $(SIM_BIN) firmware $(GOLDEN_DIR)/quartznet_desc.bin
	@echo "=== QuartzNet PicoRV32 table-walk simulation ($(MODE)) ==="
	./$(SIM_BIN) $(FIRMWARE_BIN) \
	    --qspi-weights $(GOLDEN_DIR)/quartznet_weights.bin \
	    --qspi-qparams $(GOLDEN_DIR)/quartznet_qparams.bin \
	    --qspi-table   $(GOLDEN_DIR)/quartznet_desc.bin \
	    --qspi-bytes 1048576 --psram-bytes 1048576

vcd: $(SIM_BIN) firmware $(GOLDEN_DIR)/quartznet_desc.bin
	./$(SIM_BIN) $(FIRMWARE_BIN) \
	    --qspi-weights $(GOLDEN_DIR)/quartznet_weights.bin \
	    --qspi-qparams $(GOLDEN_DIR)/quartznet_qparams.bin \
	    --qspi-table   $(GOLDEN_DIR)/quartznet_desc.bin \
	    --qspi-bytes 1048576 --psram-bytes 1048576 --vcd sim_out.vcd

# ── Clean ────────────────────────────────────────────────────────────────────

clean:
	rm -rf obj_dir $(SIM_BIN) sim_out.vcd
