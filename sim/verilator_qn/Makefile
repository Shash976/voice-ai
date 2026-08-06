# Makefile — build and run the QuartzNet PicoRV32 Verilator simulation
#
# MODE=shim (default): Verilates the UNMODIFIED rtl/soc/picorv32_soc.v;
#   sim_main_qn.cpp emulates the accelerator's register file in C++ but
#   delegates every actual computation to the real
#   firmware/quartznet/quartznet_infer.c interpreter, so this mode cannot
#   silently drift from the golden model. Existed to develop/debug
#   qn_accel.c/qn_main.c fast, before RTL existed.
# MODE=rtl (Stage 7 Gap 1 increment D5): Verilates rtl/soc/qn_soc.v -- real
#   quartznet_accel.v wired onto the bus, no C++ accelerator emulation at
#   all. This is the real fidelity claim; MODE=shim's is not.
#
# Prerequisites: verilator, riscv64-linux-gnu-gcc (for the firmware), python3
# (for goldens -- see firmware/quartznet/Makefile).
#
# Targets:
#   make              build the simulator for $(MODE)
#   make firmware     build firmware/quartznet's qn_firmware.bin + QSPI blobs
#   make run          build everything and run the table walk
#   make vcd          same + VCD waveform dump -> sim_out.vcd
#   make clean
#
# CONFIG=reduced (default, 27 descriptors, fast RTL iteration) or
# CONFIG=full (187 descriptors, Stage 7 Gap 1 D6 -- weights alone are ~18.85MB,
# so QSPI_BYTES/PSRAM_BYTES scale up accordingly). e.g.:
#   make run MODE=shim CONFIG=full

MODE   ?= shim
CONFIG ?= reduced

REPO_ROOT     = ../..
PICORV32_V    = $(REPO_ROOT)/rtl/picorv32/picorv32.v
SOC_V         = $(REPO_ROOT)/rtl/soc/picorv32_soc.v
QN_SOC_V      = $(REPO_ROOT)/rtl/soc/qn_soc.v
ACCEL_DIR     = $(REPO_ROOT)/rtl/accel
ACCEL_V       = $(ACCEL_DIR)/quartznet_accel.v $(ACCEL_DIR)/addr_gen.v \
                $(ACCEL_DIR)/ext_mem_if.v $(ACCEL_DIR)/int8_mac_array.v \
                $(ACCEL_DIR)/requantize.v $(ACCEL_DIR)/requantize_add.v
QN_DIR        = $(REPO_ROOT)/firmware/quartznet
QN_INFER_C    = $(QN_DIR)/quartznet_infer.c
FIRMWARE_BIN  = $(QN_DIR)/qn_firmware.bin

ifeq ($(CONFIG),full)
GOLDEN_DIR  = $(REPO_ROOT)/build/quartznet
QSPI_BYTES  = 25165824
PSRAM_BYTES = 1048576
else
GOLDEN_DIR  = $(REPO_ROOT)/build/quartznet_reduced
QSPI_BYTES  = 1048576
PSRAM_BYTES = 1048576
endif
T_OUT = 70

# -I here must be absolute: verilator's --build invokes make from obj_dir/,
# two directories deeper than this Makefile, so a relative $(QN_DIR) would
# resolve wrong.
QN_DIR_ABS = $(abspath $(QN_DIR))

VERILATOR = verilator

# No -Wall here, matching sim/verilator/Makefile's own precedent: picorv32.v
# has many pre-existing UNUSED-signal warnings (dbg_* debug taps, unused
# pcpi_insn bits) this project already tolerates without complaint for the
# exact same core. VFLAGS_RTL additionally needs -Wno-SYNCASYNCNET/-Wno-BLKSEQ/
# -Wno-DECLFILENAME -- see qn_soc.v's header for why those are expected, not
# bugs (only relevant here, which pulls in picorv32.v + quartznet_accel.v
# together for the first time).
VFLAGS_SHIM = --cc --exe --build --trace -O2 \
    -CFLAGS "-O2 -std=c++17 -I$(QN_DIR_ABS)" \
    --Mdir obj_dir --top-module picorv32_soc

VFLAGS_RTL = --cc --exe --build --trace -O2 \
    -CFLAGS "-O2 -std=c++17" \
    -Wno-SYNCASYNCNET -Wno-BLKSEQ -Wno-DECLFILENAME \
    --Mdir obj_dir --top-module qn_soc

SIM_BIN = sim_qn_$(MODE)

.PHONY: all firmware run vcd clean

all: $(SIM_BIN)

# ── Build simulator ──────────────────────────────────────────────────────────

ifeq ($(MODE),shim)
$(SIM_BIN): $(SOC_V) $(PICORV32_V) sim_main_qn.cpp $(QN_INFER_C)
	$(VERILATOR) $(VFLAGS_SHIM) \
	    -o $(SIM_BIN) \
	    $(SOC_V) $(PICORV32_V) \
	    sim_main_qn.cpp $(QN_INFER_C)
	cp obj_dir/$(SIM_BIN) .
else
$(SIM_BIN): $(QN_SOC_V) $(SOC_V) $(PICORV32_V) $(ACCEL_V) sim_main_qn_rtl.cpp
	$(VERILATOR) $(VFLAGS_RTL) \
	    -o $(SIM_BIN) \
	    $(QN_SOC_V) $(SOC_V) $(PICORV32_V) $(ACCEL_V) \
	    sim_main_qn_rtl.cpp
	cp obj_dir/$(SIM_BIN) .
endif

# ── Build firmware + QSPI blobs ──────────────────────────────────────────────

firmware:
	$(MAKE) -C $(QN_DIR) rv32-fw GOLDEN_DIR=$(GOLDEN_DIR) T_OUT=$(T_OUT)

$(GOLDEN_DIR)/quartznet_desc.bin $(GOLDEN_DIR)/quartznet_weights.bin $(GOLDEN_DIR)/quartznet_qparams.bin:
	$(MAKE) -C $(QN_DIR) goldens

# ── Run ──────────────────────────────────────────────────────────────────────

run: $(SIM_BIN) firmware $(GOLDEN_DIR)/quartznet_desc.bin
	@echo "=== QuartzNet PicoRV32 table-walk simulation (MODE=$(MODE) CONFIG=$(CONFIG)) ==="
	./$(SIM_BIN) $(FIRMWARE_BIN) \
	    --qspi-weights $(GOLDEN_DIR)/quartznet_weights.bin \
	    --qspi-qparams $(GOLDEN_DIR)/quartznet_qparams.bin \
	    --qspi-table   $(GOLDEN_DIR)/quartznet_desc.bin \
	    --qspi-bytes $(QSPI_BYTES) --psram-bytes $(PSRAM_BYTES)

vcd: $(SIM_BIN) firmware $(GOLDEN_DIR)/quartznet_desc.bin
	./$(SIM_BIN) $(FIRMWARE_BIN) \
	    --qspi-weights $(GOLDEN_DIR)/quartznet_weights.bin \
	    --qspi-qparams $(GOLDEN_DIR)/quartznet_qparams.bin \
	    --qspi-table   $(GOLDEN_DIR)/quartznet_desc.bin \
	    --qspi-bytes $(QSPI_BYTES) --psram-bytes $(PSRAM_BYTES) --vcd sim_out.vcd

# ── Clean ────────────────────────────────────────────────────────────────────

clean:
	rm -rf obj_dir sim_qn_shim sim_qn_rtl sim_out.vcd
