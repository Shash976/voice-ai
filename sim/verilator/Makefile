# Makefile — build and run PicoRV32 Verilator simulation
#
# Prerequisites (WSL):
#   verilator            sudo apt install verilator
#   riscv64-linux-gnu-*  sudo apt install gcc-riscv64-linux-gnu binutils-riscv64-linux-gnu
#
# Targets:
#   make              build simulator binary
#   make firmware     cross-compile firmware (in firmware/picorv32_baremetal/)
#   make run          build everything and run
#   make vcd          run with VCD waveform dump → sim_out.vcd  (open in GTKWave)
#   make clean

REPO_ROOT    = ../..
PICORV32_V   = $(REPO_ROOT)/rtl/picorv32/picorv32.v
SOC_V        = $(REPO_ROOT)/rtl/soc/picorv32_soc.v
FIRMWARE_BIN = $(REPO_ROOT)/firmware/picorv32_baremetal/firmware.bin
FIRMWARE_DIR = $(REPO_ROOT)/firmware/picorv32_baremetal

VERILATOR      = verilator
VERILATOR_FLAGS = \
    --cc \
    --exe \
    --build \
    --trace \
    -O2 \
    -CFLAGS "-O2 -std=c++17" \
    --Mdir obj_dir \
    --top-module picorv32_soc

SIM_BIN = sim_picorv32

.PHONY: all firmware run vcd clean check-deps

all: $(SIM_BIN)

# ── Check prerequisites ──────────────────────────────────────────────────────

check-deps:
	@which verilator > /dev/null 2>&1 || (echo "ERROR: verilator not found. Run: sudo apt install verilator" && exit 1)
	@which riscv64-linux-gnu-gcc > /dev/null 2>&1 || (echo "ERROR: riscv64-linux-gnu-gcc not found. Run: sudo apt install gcc-riscv64-linux-gnu" && exit 1)
	@test -f $(PICORV32_V) || (echo "ERROR: $(PICORV32_V) not found. Run: bash scripts/setup_picorv32.sh" && exit 1)
	@test -f $(REPO_ROOT)/firmware/tinyengine_port/tiny_vad_weights.h || (echo "ERROR: tiny_vad_weights.h missing. Run: python sw/tinyml_reference/export_weights.py" && exit 1)
	@test -f $(REPO_ROOT)/firmware/tinyengine_port/tiny_vad_test_vectors.h || (echo "ERROR: tiny_vad_test_vectors.h missing. Run: python sw/tinyml_reference/gen_test_vectors.py" && exit 1)
	@echo "All prerequisites OK"

# ── Build simulator ──────────────────────────────────────────────────────────

$(SIM_BIN): $(SOC_V) $(PICORV32_V) sim_main.cpp
	$(VERILATOR) $(VERILATOR_FLAGS) \
	    -o $(SIM_BIN) \
	    $(SOC_V) $(PICORV32_V) \
	    sim_main.cpp
	cp obj_dir/$(SIM_BIN) .

# ── Build firmware ───────────────────────────────────────────────────────────

firmware:
	$(MAKE) -C $(FIRMWARE_DIR)

# ── Run ─────────────────────────────────────────────────────────────────────

run: $(SIM_BIN) firmware
	@echo "=== PicoRV32 TinyVAD simulation ==="
	./$(SIM_BIN) $(FIRMWARE_BIN)

# Run with VCD waveform capture (opens GTKWave if available)
vcd: $(SIM_BIN) firmware
	./$(SIM_BIN) $(FIRMWARE_BIN) --vcd sim_out.vcd
	@which gtkwave > /dev/null 2>&1 && gtkwave sim_out.vcd & || echo "VCD saved to sim_out.vcd (open with GTKWave)"

# ── Clean ────────────────────────────────────────────────────────────────────

clean:
	rm -rf obj_dir $(SIM_BIN) sim_out.vcd
	$(MAKE) -C $(FIRMWARE_DIR) clean
