#!/usr/bin/env bash
# run_quartznet.sh — drive the classic ORFS make flow for quartznet_accel.
#
# Modeled directly on run.sh / run_spike.sh (same staging, WORK_HOME and argument
# conventions). Three deliberate differences from run.sh, each explained inline:
#   1. ORFS_DIR defaults to ~/OpenROAD-flow-scripts, not /opt/... .
#   2. rtl/accel/ext_mem_if.v is replaced by physical/orfs/stubs/ext_mem_if_synth.v.
#   3. staging uses `cp -p`, which matters a great deal for this design.
#
# Usage:
#   ./run_quartznet.sh                       # full flow on nangate45 -> GDS
#   ./run_quartznet.sh nangate45 synth       # stop after synthesis
#   ./run_quartznet.sh nangate45 floorplan   # ... then place / cts / route / final
#   ./run_quartznet.sh nangate45 gui_final   # open the OpenROAD GUI on the result
#   ORFS_DIR=/path ./run_quartznet.sh ...    # override ORFS location
#
# Outputs land under ../runs/{results,reports,logs}/<platform>/quartznet_accel/base/.
set -euo pipefail

# This machine's ORFS lives in $HOME; /opt/OpenROAD-flow-scripts is the company
# VM's path and does not exist here. physical/orfs/synth_area.sh already assumes
# $HOME, so this is the consistent choice, not a new convention.
ORFS="${ORFS_DIR:-$HOME/OpenROAD-flow-scripts}"
PLATFORM="${1:-nangate45}"
TARGET="${2:-}"          # empty = full flow; or synth / floorplan / place / cts / route / final / gui_final
DESIGN="quartznet_accel"

HERE="$(cd "$(dirname "$0")" && pwd)"
RUNS="$(dirname "$HERE")/runs"
REPO="$(cd "$HERE/../../.." && pwd)"
STUB="$REPO/physical/orfs/stubs/ext_mem_if_synth.v"

CFG="$HERE/$PLATFORM/$DESIGN/config.mk"
[ -f "$CFG" ]            || { echo "ERROR: no config for platform '$PLATFORM' at $CFG"; exit 1; }
[ -f "$ORFS/env.sh" ]    || { echo "ERROR: ORFS not found at $ORFS (set ORFS_DIR)"; exit 1; }
[ -d "$REPO/rtl/accel" ] || { echo "ERROR: RTL not found at $REPO/rtl/accel"; exit 1; }
[ -f "$STUB" ]           || { echo "ERROR: memory stub not found at $STUB"; exit 1; }

# 1. stage the canonical RTL into the make/ work tree (src/ is gitignored).
#
#    `cp -p` preserves source mtimes. This is not cosmetic: ORFS puts
#    VERILOG_FILES in YOSYS_DEPENDENCIES (scripts/variables.mk:179), so a plain
#    `cp` bumps mtimes on every invocation and forces a full re-synthesis. For
#    tinymac that costs seconds; here ABC alone can run for tens of minutes, so a
#    staged sequence (synth, then floorplan, then route...) would pay it at every
#    step.
#
#    ext_mem_if.v is deliberately NOT staged. Delete any copy an earlier manual
#    run left behind so a stale real model can never be picked up.
mkdir -p "$HERE/src/$DESIGN"
rm -f "$HERE/src/$DESIGN/ext_mem_if.v"
cp -p "$REPO/rtl/accel/quartznet_accel.v" \
      "$REPO/rtl/accel/addr_gen.v" \
      "$REPO/rtl/accel/int8_mac_array.v" \
      "$REPO/rtl/accel/requantize.v" \
      "$REPO/rtl/accel/requantize_add.v" \
      "$HERE/src/$DESIGN/"
cp -p "$STUB" "$HERE/src/$DESIGN/"

# 2. run ORFS; WORK_HOME=runs/ keeps logs/objects/results out of make/
# shellcheck disable=SC1090
source "$ORFS/env.sh"
cd "$HERE"
mkdir -p "$RUNS"
echo "── ORFS $PLATFORM/$DESIGN  target='${TARGET:-<full flow>}' ──"
echo "   LANES=32 ACC_W=32; ext_mem_if is the SYNTHESIS STUB, not the sim model"
make --file="$ORFS/flow/Makefile" \
     FLOW_HOME="$ORFS/flow" \
     WORK_HOME="$RUNS" \
     DESIGN_CONFIG="./$PLATFORM/$DESIGN/config.mk" \
     $TARGET

echo
echo "Done. Key reports:"
echo "  $RUNS/reports/$PLATFORM/$DESIGN/base/synth_stat.txt  (expect ~79,700 cells / ~107,300 um^2)"
echo "  $RUNS/reports/$PLATFORM/$DESIGN/base/6_finish.rpt    (wns/tns, period_min, fmax, power)"
echo "  $RUNS/reports/$PLATFORM/$DESIGN/base/5_route_drc.rpt (MUST be 0 bytes)"
echo "  $RUNS/results/$PLATFORM/$DESIGN/base/6_final.gds     (open in klayout)"
