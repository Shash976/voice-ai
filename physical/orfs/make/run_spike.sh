#!/usr/bin/env bash
# run_spike.sh — drive the ORFS make flow for the fakeram45 macro spike.
#
# Modeled directly on run.sh (same staging / WORK_HOME / argument conventions),
# with one addition: SPIKE_VARIANT selects between the experiment variants used
# to de-risk the QuartzNet ~18-macro memory system.  Findings are written up in
# docs/07a_sram_macro_notes.md.
#
# Variants (SPIKE_VARIANT env, default `base`):
#   base    4 macros (2x512x64 + 2x1024x32), automatic rtl_macro_placer
#   r0grid  4 macros, forced 2x2 grid, ALL R0  — every macro's pin wall (west
#           edge, metal3-only) faces the same direction.  The naive arrangement
#           an 18-macro floorplan falls into by default.
#   mirror  4 macros, forced 2x2 grid, left column R180 / right column MX, so the
#           pin walls face each other across a central routing channel.
#   outward 4 macros, forced 2x2 grid, pins facing the die edges — worst case
#   m8      8 macros (4+4), automatic placement, die grown to hold macro
#           utilization at base's 39.4% so the 4-vs-8 comparison isolates macro
#           COUNT rather than density.
#
# r0grid/mirror/outward need a macro_placement_<variant>.tcl containing real post-synth
# instance names; generate it from the base run's
#   runs/results/nangate45/sram_spike/base/2_2_floorplan_macro.tcl
# (written by ORFS scripts/macro_place.tcl).  If the file is absent the variant
# falls back to automatic placement and says so.
#
# Usage:
#   ./run_spike.sh                          # full flow, base variant
#   ./run_spike.sh nangate45 synth          # stop after synthesis
#   SPIKE_VARIANT=mirror ./run_spike.sh     # full flow, mirrored macro columns
#   SPIKE_VARIANT=m8 ./run_spike.sh         # 8-macro scaling run
#   ./run_spike.sh nangate45 gui_final      # open the OpenROAD GUI
#   ORFS_DIR=/path ./run_spike.sh ...       # override ORFS location
#
# Outputs land under ../runs/{results,reports,logs}/<platform>/sram_spike/<variant>/
set -euo pipefail

ORFS="${ORFS_DIR:-/opt/OpenROAD-flow-scripts}"
PLATFORM="${1:-nangate45}"
TARGET="${2:-}"                 # empty = full flow; or synth / floorplan / route / final / gui_final
DESIGN="sram_spike"
VARIANT="${SPIKE_VARIANT:-base}"

HERE="$(cd "$(dirname "$0")" && pwd)"
RUNS="$(dirname "$HERE")/runs"
REPO="$(cd "$HERE/../../.." && pwd)"

CFGDIR="$HERE/$PLATFORM/$DESIGN"
BASE_CFG="$CFGDIR/config.mk"
[ -f "$BASE_CFG" ]        || { echo "ERROR: no config for platform '$PLATFORM' at $BASE_CFG"; exit 1; }
[ -f "$ORFS/env.sh" ]     || { echo "ERROR: ORFS not found at $ORFS (set ORFS_DIR)"; exit 1; }
[ -d "$REPO/rtl/spike_sram" ] || { echo "ERROR: RTL not found at $REPO/rtl/spike_sram"; exit 1; }

# ── Per-variant knobs ───────────────────────────────────────────────────────
# Die sizes hold macro utilization ~constant (39.4% of core area):
#   4 macros =  67,415 um^2 in a 439.93 x 388.80 core (171,044 um^2)
#   8 macros = 134,830 um^2 in a 619.93 x 552.80 core (342,700 um^2)
case "$VARIANT" in
    base|r0grid|mirror|outward)
        N512=2; N1024=2
        DIE="0 0 460 410"; CORE="10.07 11.2 450 400" ;;
    m8)
        N512=4; N1024=4
        DIE="0 0 640 575"; CORE="10.07 11.2 630 564" ;;
    *)
        echo "ERROR: unknown SPIKE_VARIANT '$VARIANT' (base|r0grid|mirror|outward|m8)"; exit 1 ;;
esac

# 1. stage the canonical RTL into the make/ work tree (src/ is gitignored)
mkdir -p "$HERE/src/$DESIGN"
cp "$REPO/rtl/spike_sram/sram_spike.v" "$HERE/src/$DESIGN/"

# 2. generate the per-variant config.mk (hard `=` assignments so the variant
#    values are authoritative and cannot be overridden from the environment)
GEN_CFG="$CFGDIR/config_${VARIANT}.mk"
if [ "$VARIANT" = "base" ]; then
    GEN_CFG="$BASE_CFG"
else
    MP_TCL="$CFGDIR/macro_placement_${VARIANT}.tcl"
    {
        sed -e "s|^export DIE_AREA .*|export DIE_AREA  = $DIE|" \
            -e "s|^export CORE_AREA .*|export CORE_AREA = $CORE|" \
            "$BASE_CFG"
        echo
        echo "# ── injected by run_spike.sh for variant '$VARIANT' ──"
        echo "export VERILOG_TOP_PARAMS = N_512X64 $N512 N_1024X32 $N1024"
        if [ -f "$MP_TCL" ]; then
            echo "export MACRO_PLACEMENT_TCL = \$(DESIGN_HOME)/$PLATFORM/$DESIGN/macro_placement_${VARIANT}.tcl"
        fi
    } > "$GEN_CFG"

    if [ ! -f "$MP_TCL" ]; then
        echo "NOTE: $MP_TCL not found — variant '$VARIANT' will use automatic"
        echo "      macro placement. Generate it from the base run's"
        echo "      $RUNS/results/$PLATFORM/$DESIGN/base/2_2_floorplan_macro.tcl"
    fi
fi

# 3. run ORFS; WORK_HOME=runs/ keeps logs/objects/results out of make/
# shellcheck disable=SC1090
source "$ORFS/env.sh"
cd "$HERE"
mkdir -p "$RUNS"
echo "── ORFS $PLATFORM/$DESIGN  variant='$VARIANT'  target='${TARGET:-<full flow>}' ──"
echo "   macros: ${N512}x fakeram45_512x64 + ${N1024}x fakeram45_1024x32   die: $DIE"

make --file="$ORFS/flow/Makefile" \
     FLOW_HOME="$ORFS/flow" \
     WORK_HOME="$RUNS" \
     DESIGN_CONFIG="$GEN_CFG" \
     FLOW_VARIANT="$VARIANT" \
     $TARGET

echo
echo "Done. Key reports:"
echo "  $RUNS/reports/$PLATFORM/$DESIGN/$VARIANT/  (6_report.* — area, WNS/TNS, power)"
echo "  $RUNS/results/$PLATFORM/$DESIGN/$VARIANT/6_final.gds  (open in klayout)"
echo "  $RUNS/results/$PLATFORM/$DESIGN/$VARIANT/2_2_floorplan_macro.tcl  (chosen macro placement)"
