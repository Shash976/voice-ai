# config.mk — ORFS classic-flow config for the QuartzNet 15x5 inference core.
#
# Mirrors nangate45/tinymac_accel/config.mk (same DESIGN_HOME / util / density
# conventions). The one structural difference is VERILOG_FILES, which
# substitutes physical/orfs/stubs/ext_mem_if_synth.v for the real, deliberately
# non-synthesizable rtl/accel/ext_mem_if.v — see that stub's header for the full
# reasoning, including why a Yosys blackbox is NOT usable here.
#
# Per IT setup, `DESIGN_HOME = .` re-roots all design paths to the working
# directory you run `make` from. PLATFORM_DIR is NOT affected by that (ORFS
# derives it from FLOW_HOME) — same note as nangate45/sram_spike/config.mk.
#
# Run (see physical/orfs/make/run_quartznet.sh):
#   source ~/OpenROAD-flow-scripts/env.sh
#   make --file=$HOME/OpenROAD-flow-scripts/flow/Makefile \
#        DESIGN_CONFIG=./nangate45/quartznet_accel/config.mk

export DESIGN_HOME = .

export DESIGN_NAME = quartznet_accel
export PLATFORM    = nangate45

# Order is not significant (ORFS reads each with `read_verilog -defer` then
# elaborates from the top); the top is listed first for readability.
# NOTE the last entry: ext_mem_if_synth.v, NOT ext_mem_if.v. The module inside
# it is still named `ext_mem_if`, which is what quartznet_accel's u_mem binds to.
export VERILOG_FILES = $(DESIGN_HOME)/src/$(DESIGN_NAME)/quartznet_accel.v \
                       $(DESIGN_HOME)/src/$(DESIGN_NAME)/addr_gen.v \
                       $(DESIGN_HOME)/src/$(DESIGN_NAME)/int8_mac_array.v \
                       $(DESIGN_HOME)/src/$(DESIGN_NAME)/requantize.v \
                       $(DESIGN_HOME)/src/$(DESIGN_NAME)/requantize_add.v \
                       $(DESIGN_HOME)/src/$(DESIGN_NAME)/ext_mem_if_synth.v

export SDC_FILE      = $(DESIGN_HOME)/$(PLATFORM)/$(DESIGN_NAME)/constraint.sdc

# Pin the design point explicitly instead of inheriting the module's own
# defaults. They agree today (LANES=32, ACC_W=32); this is belt-and-braces so a
# future default change in rtl/accel/quartznet_accel.v cannot silently move the
# physical design point. ACC_W is NOT a swept axis for QuartzNet: 24 saturates
# for any pointwise with c_in >= 260 (CLAUDE.md Stage-7 gotcha (b)).
# Mechanism: ORFS parses this as a Tcl dict and calls `chparam -set` per pair
# (scripts/synth_preamble.tcl:124-127) — the same knob sweep.sh already uses.
export VERILOG_TOP_PARAMS = LANES 32 ACC_W 32

# Measured with the ORFS Yosys 0.64 before committing to P&R: 79,698 cells /
# 107,261 um^2 at this design point, of which 4,360 um^2 is the memory stub.
# That is 7.4x tinymac_accel's 14,518 um^2 and 1.6x the 4-macro sram_spike core.
# At 40% utilization it implies a ~268,000 um^2 core, i.e. roughly 518 x 518 um.
#
# Keeping tinymac's 40 / 0.60 rather than packing tighter: this is a first
# attempt at a brand-new design point, and 20 points of place-density headroom is
# what makes detailed routing likely to close on the first try. Tighten only
# after a clean run.
export CORE_UTILIZATION      ?= 40
export PLACE_DENSITY          ?= 0.60
export SYNTH_REPEATABLE_BUILD ?= 1

# First-pass bring-up only: ABC_AREA selects the much shorter abc_area.script
# (strash; dch; map -B 0.9) over the default abc_speed.script, which measured
# >35 minutes and still running on this ~130k-gate netlist. This validates the
# full synth->floorplan->place->cts->route->final pipeline mechanically before
# committing hours to a full-quality run -- drop this line for the real
# (tighter Fmax, smaller area) final numbers once the flow is proven clean.
export ABC_AREA = 1
