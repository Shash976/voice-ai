# config.mk — ORFS classic-flow config for tinymac_accel on asap7 (7 nm).
#
# Same structure as the nangate45/sky130hd configs with PLATFORM swapped to
# asap7 — the plan's primary RTL-to-GDS target. Run with:
#   ./run.sh asap7              # full flow synth → … → GDS
#   ./run.sh asap7 synth        # stop after synthesis
#
# asap7 is the most fragile of the open PDKs (see the plan's "ASAP7 caveats"),
# so bring it up at default utilisation/density first; tighten only once a clean
# GDS exists. Parameter sweeps (LANES/ACC_W) work the same as the other PDKs:
#   ./sweep.sh asap7

export DESIGN_HOME = .

export DESIGN_NAME = tinymac_accel
export PLATFORM    = asap7

export VERILOG_FILES = $(DESIGN_HOME)/src/$(DESIGN_NAME)/tinymac_accel.v \
                       $(DESIGN_HOME)/src/$(DESIGN_NAME)/int8_mac_array.v \
                       $(DESIGN_HOME)/src/$(DESIGN_NAME)/requantize.v

export SDC_FILE      = $(DESIGN_HOME)/$(PLATFORM)/$(DESIGN_NAME)/constraint.sdc

# Small synchronous block: pack reasonably, leave room for routing.
# asap7 is dense; these match the other PDK configs and are a safe first cut.
export CORE_UTILIZATION      ?= 40
export PLACE_DENSITY          ?= 0.60
export SYNTH_REPEATABLE_BUILD ?= 1
