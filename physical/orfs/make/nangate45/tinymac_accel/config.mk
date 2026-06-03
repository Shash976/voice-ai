# config.mk — ORFS classic-flow config for the TinyVAD int8 matvec accelerator.
#
# Mirrors the layout of /opt/OpenROAD-flow-scripts/flow/designs/nangate45/gcd.
# Per IT setup, `DESIGN_HOME = .` re-roots all paths to the working directory
# you run `make` from, so reads/writes stay in your workdir.
#
# Run (after assembling the workdir — see physical/orfs/make/run.sh):
#   source /opt/OpenROAD-flow-scripts/env.sh
#   make --file=/opt/OpenROAD-flow-scripts/flow/Makefile \
#        DESIGN_CONFIG=./nangate45/tinymac_accel/config.mk

export DESIGN_HOME = .

export DESIGN_NAME = tinymac_accel
export PLATFORM    = nangate45

export VERILOG_FILES = $(DESIGN_HOME)/src/$(DESIGN_NAME)/tinymac_accel.v \
                       $(DESIGN_HOME)/src/$(DESIGN_NAME)/int8_mac_array.v \
                       $(DESIGN_HOME)/src/$(DESIGN_NAME)/requantize.v

export SDC_FILE      = $(DESIGN_HOME)/$(PLATFORM)/$(DESIGN_NAME)/constraint.sdc

# Small synchronous block: pack reasonably, leave room for routing.
export CORE_UTILIZATION      ?= 40
export PLACE_DENSITY          ?= 0.60
export SYNTH_REPEATABLE_BUILD ?= 1
