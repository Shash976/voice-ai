# config.mk — ORFS classic-flow config for the fakeram45 macro-integration spike.
#
# This is the repo's FIRST design with hard macros.  It exists to prove that
# fakeram45 SRAM macros survive synth → floorplan → macro place → place → CTS →
# route → GDS in this ORFS install, before the QuartzNet accelerator commits to
# a ~18-macro / ~69 KB memory system (docs/07_quartznet_pivot.md, Stage E).
# Findings are written up in docs/07a_sram_macro_notes.md.
#
# Follows the same conventions as nangate45/tinymac_accel/config.mk: per IT
# setup, `DESIGN_HOME = .` re-roots design paths to the working directory you
# run `make` from.  Note that PLATFORM_DIR is NOT affected by that — ORFS
# derives it from FLOW_HOME (scripts/variables.mk:29-35) — so the
# $(PLATFORM_DIR)/lef|lib references below resolve into the ORFS install.
#
# Run (see physical/orfs/make/run_spike.sh):
#   source /opt/OpenROAD-flow-scripts/env.sh
#   make --file=/opt/OpenROAD-flow-scripts/flow/Makefile \
#        DESIGN_CONFIG=./nangate45/sram_spike/config.mk

export DESIGN_HOME = .

export DESIGN_NAME = sram_spike
export PLATFORM    = nangate45

export VERILOG_FILES = $(DESIGN_HOME)/src/$(DESIGN_NAME)/sram_spike.v
export SDC_FILE      = $(DESIGN_HOME)/$(PLATFORM)/$(DESIGN_NAME)/constraint.sdc

# ── Macro views ─────────────────────────────────────────────────────────────
# ORDERING IS LOAD-BEARING: these MUST be set here, in the design config.
# platforms/nangate45/config.mk:10-11 composes
#     LIB_FILES = <NangateOpenCellLibrary_typical.lib> $(ADDITIONAL_LIBS)
# and scripts/variables.mk:40 includes that platform config AFTER the design
# config.  Setting ADDITIONAL_LIBS any later silently never reaches LIB_FILES,
# and synthesis then fails to find the macro cells with no obvious clue why.
#
# ADDITIONAL_LIBS  → timing/blackbox view.  Also how yosys learns the macro
#                    ports: scripts/synth_stdcells.tcl does
#                    `read_liberty -lib {*}$::env(LIB_FILES)`.
# ADDITIONAL_LEFS  → physical abstract (CLASS BLOCK) used by every stage.
export ADDITIONAL_LEFS = $(PLATFORM_DIR)/lef/fakeram45_512x64.lef \
                         $(PLATFORM_DIR)/lef/fakeram45_1024x32.lef
export ADDITIONAL_LIBS = $(PLATFORM_DIR)/lib/fakeram45_512x64.lib \
                         $(PLATFORM_DIR)/lib/fakeram45_1024x32.lib

# NOT set here on purpose:
#   GDS_ALLOW_EMPTY — platforms/nangate45/config.mk:91 already ships
#     `export GDS_ALLOW_EMPTY ?= fakeram.*`, which is what lets
#     util/def2stream.py:82-96 emit 6_final.gds despite fakeram45 having no
#     GDS.  Re-exporting it here would be redundant noise.
#   PDN_TCL — the platform default grid_strategy-M1-M4-M7.tcl already defines
#     CORE_macro_grid_1/2 (metal5/metal6 straps over macros, metal4<->metal5
#     connects onto the macros' metal4 VDD/VSS stripes).  No nangate45 macro
#     design upstream overrides it either.

# ── Floorplan ───────────────────────────────────────────────────────────────
# Explicit die rather than CORE_UTILIZATION: with macros dominating the area,
# utilization-derived sizing is fragile and hard to reason about.  This is what
# swerv_wrapper / black_parrot / bp_quad all do upstream.
#   4 macros = 2*17301.4 + 2*16406.1 = 67,415 um^2 of raw macro area.
#   Core 439.93 x 388.80 = 171,044 um^2  ->  39.4% macro utilization.
# Phase-3 (8-macro) runs keep that 39.4% constant so the comparison isolates
# macro COUNT rather than macro density — see run_spike.sh.
export DIE_AREA  = 0 0 460 410
export CORE_AREA = 10.07 11.2 450 400

# Upstream nangate45 macro designs use 10 10 (swerv, bp_quad, black_parrot) or
# 8 8 (ariane133); the platform default is a much looser 22.4 15.12.
export MACRO_PLACE_HALO = 10 10

export PLACE_DENSITY_LB_ADDON = 0.10
export SYNTH_REPEATABLE_BUILD = 1

# ── injected by run_spike.sh for variant 'outward' ──
export VERILOG_TOP_PARAMS = N_512X64 2 N_1024X32 2
export MACRO_PLACEMENT_TCL = $(DESIGN_HOME)/nangate45/sram_spike/macro_placement_outward.tcl
