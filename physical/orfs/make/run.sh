#!/usr/bin/env bash
# run.sh — drive the classic ORFS make flow for tinymac_accel on the company VM.
#
# Source configs (<platform>/tinymac_accel/{config.mk,constraint.sdc}) live here
# in make/. ORFS build outputs (logs/, objects/, results/, reports/) go to
# ../runs/ (physical/orfs/runs/) so this directory stays clean.
#
# Usage:
#   ./run.sh                      # full flow on nangate45  (synth → … → GDS)
#   ./run.sh nangate45 gui_final  # full flow then open the OpenROAD GUI
#   ./run.sh sky130hd             # full flow on sky130hd
#   ./run.sh nangate45 synth      # stop after synthesis
#   ORFS_DIR=/path ./run.sh ...   # override ORFS location
#
# Outputs land under ../runs/results, ../runs/reports, ../runs/logs.
set -euo pipefail

ORFS="${ORFS_DIR:-/opt/OpenROAD-flow-scripts}"
PLATFORM="${1:-nangate45}"
TARGET="${2:-}"                 # empty = full flow; or synth / floorplan / route / final / gui_final
DESIGN="tinymac_accel"

HERE="$(cd "$(dirname "$0")" && pwd)"
RUNS="$(dirname "$HERE")/runs"
REPO="$(cd "$HERE/../../.." && pwd)"

CFG="$HERE/$PLATFORM/$DESIGN/config.mk"
[ -f "$CFG" ]            || { echo "ERROR: no config for platform '$PLATFORM' at $CFG"; exit 1; }
[ -f "$ORFS/env.sh" ]   || { echo "ERROR: ORFS not found at $ORFS (set ORFS_DIR)"; exit 1; }
[ -d "$REPO/rtl/accel" ]|| { echo "ERROR: RTL not found at $REPO/rtl/accel"; exit 1; }

# 1. stage the canonical RTL into the make/ work tree (src/ is gitignored)
mkdir -p "$HERE/src/$DESIGN"
cp "$REPO/rtl/accel/tinymac_accel.v" \
   "$REPO/rtl/accel/int8_mac_array.v" \
   "$REPO/rtl/accel/requantize.v" \
   "$HERE/src/$DESIGN/"

# 2. run ORFS; WORK_HOME=runs/ keeps logs/objects/results out of make/
# shellcheck disable=SC1090
source "$ORFS/env.sh"
cd "$HERE"
mkdir -p "$RUNS"
echo "── ORFS $PLATFORM/$DESIGN  target='${TARGET:-<full flow>}' ──"
make --file="$ORFS/flow/Makefile" \
     FLOW_HOME="$ORFS/flow" \
     WORK_HOME="$RUNS" \
     DESIGN_CONFIG="./$PLATFORM/$DESIGN/config.mk" \
     $TARGET

echo
echo "Done. Key reports:"
echo "  $RUNS/reports/$PLATFORM/$DESIGN/base/  (6_report.* — area, WNS/TNS, power)"
echo "  $RUNS/results/$PLATFORM/$DESIGN/base/6_final.gds  (open in klayout)"
