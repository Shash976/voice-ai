# config.mk — ORFS classic-flow config for tinymac_accel on sky130hd (130 nm).
# Same as the nangate45 config with PLATFORM swapped. Run with:
#   ./run.sh sky130hd

export DESIGN_HOME = .

export DESIGN_NAME = tinymac_accel
export PLATFORM    = sky130hd

export VERILOG_FILES = $(DESIGN_HOME)/src/$(DESIGN_NAME)/tinymac_accel.v \
                       $(DESIGN_HOME)/src/$(DESIGN_NAME)/int8_mac_array.v \
                       $(DESIGN_HOME)/src/$(DESIGN_NAME)/requantize.v

export SDC_FILE      = $(DESIGN_HOME)/$(PLATFORM)/$(DESIGN_NAME)/constraint.sdc

export CORE_UTILIZATION      ?= 40
export PLACE_DENSITY          ?= 0.60
export SYNTH_REPEATABLE_BUILD ?= 1
