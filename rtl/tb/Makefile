# Makefile — Verilator unit test for the synthesizable tinymac_accel core.
#
#   make            build + run the self-checking testbench
#   make LANES=8    test a different lane count (must rebuild)
#   make ACC_W=32   test a different accumulator width
#   make clean
#
# The C++ golden model is told LANES/ACC_W via -DTB_* so it mirrors the RTL.

LANES ?= 4
ACC_W ?= 24

RTL_DIR := ../accel
RTL := $(RTL_DIR)/tinymac_accel.v $(RTL_DIR)/int8_mac_array.v $(RTL_DIR)/requantize.v

VERILATOR ?= verilator
TOP := tinymac_accel
BIN := V$(TOP)

VFLAGS := --cc --exe --build -Wall -Wno-DECLFILENAME \
          --top-module $(TOP) \
          -GLANES=$(LANES) -GACC_W=$(ACC_W) \
          -CFLAGS "-DTB_LANES=$(LANES) -DTB_ACC_W=$(ACC_W) -O2"

.PHONY: all run clean
all: run

run:
	$(VERILATOR) $(VFLAGS) $(RTL) tinymac_tb.cpp -o $(BIN)
	@echo "── running ──"
	./obj_dir/$(BIN)

clean:
	rm -rf obj_dir
