#!/usr/bin/env bash
# sweep.sh — run a GRID of accelerator configs through the full ORFS RTL→GDS
# flow and collect a comparison table (area / timing / power), one GDS each.
#
# This is the "see different chip configurations" tool: it drives the same flow
# as run.sh, but once per config, overriding the RTL parameters (LANES, ACC_W)
# via ORFS VERILOG_TOP_PARAMS and the clock via a generated SDC. Each config
# gets its own FLOW_VARIANT, so results/<plat>/<design>/<variant>/6_final.gds
# can be opened independently in the GUI.
#
# Usage (on the VM, where /opt/OpenROAD-flow-scripts lives):
#   physical/orfs/make/sweep.sh                 # default grid on nangate45
#   physical/orfs/make/sweep.sh sky130hd        # a different PDK
#   CONFIGS_FILE=my.txt physical/orfs/make/sweep.sh   # custom grid
#
# Each line of the grid is "LANES ACC_W CLK_NS". Edit GRID below or supply a
# CONFIGS_FILE with the same format (one config per line, '#'=comment).
#
# Each full-flow run takes a few minutes — the default grid is intentionally
# small. Re-running skips configs whose 6_final.gds already exists.
set -uo pipefail

ORFS="${ORFS_DIR:-/opt/OpenROAD-flow-scripts}"
PLATFORM="${1:-nangate45}"
DESIGN="tinymac_accel"

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../../.." && pwd)"

# ── The grid: "LANES ACC_W CLK_NS" ──────────────────────────────────────────
# Default sweeps the lane count (the dominant area/speed knob) at the Stage-5
# accumulator width and a fixed clock. Add ACC_W or clock variations as rows.
GRID_DEFAULT=$'1 24 2.0\n2 24 2.0\n4 24 2.0\n8 24 2.0\n16 24 2.0'
if [ -n "${CONFIGS_FILE:-}" ] && [ -f "${CONFIGS_FILE}" ]; then
    GRID="$(grep -vE '^\s*#|^\s*$' "$CONFIGS_FILE")"
else
    GRID="$GRID_DEFAULT"
fi

[ -f "$ORFS/env.sh" ] || { echo "ERROR: ORFS not found at $ORFS (set ORFS_DIR)"; exit 1; }

# ── Stage the RTL (shared by every variant) ─────────────────────────────────
mkdir -p "$HERE/src/$DESIGN" "$HERE/$PLATFORM/$DESIGN"
cp "$REPO/rtl/accel/tinymac_accel.v" \
   "$REPO/rtl/accel/int8_mac_array.v" \
   "$REPO/rtl/accel/requantize.v" \
   "$HERE/src/$DESIGN/"

# shellcheck disable=SC1090
source "$ORFS/env.sh"
cd "$HERE"

# Platform time-unit multiplier: nangate45 SDC is in ns (×1), asap7 in ps (×1000).
# The optimizer always works in nanoseconds; multiply before writing the SDC.
declare -A PLATFORM_TIME_UNIT=( [nangate45]=1 [asap7]=1000 )
_TIME_UNIT="${PLATFORM_TIME_UNIT[$PLATFORM]:-1}"

RUN_ID="$(date +%Y%m%dT%H%M%S)"
CSV="$HERE/sweep_results.csv"
# Write header only when the file does not yet exist (append mode for all runs).
if [ ! -f "$CSV" ]; then
    echo "run_id,lanes,acc_w,clk_ns,variant,status,area_um2,util_pct,wns_ns,tns_ns,setup_viol,power_mw,fmax_mhz,timing_met,gds" > "$CSV"
fi

run_one() {
    local lanes="$1" acc="$2" clk="$3"
    # normalise clk to one decimal so the variant name matches the Python
    # optimizer's (optimizer/physical_runner.py), letting them share built GDS.
    local clkn; clkn="$(printf '%.1f' "$clk")"
    local variant="L${lanes}_A${acc}_c${clkn//./p}"
    local cfgdir="$HERE/$PLATFORM/$DESIGN"
    local gen_sdc="$cfgdir/constraint_${variant}.sdc"
    local gen_cfg="$cfgdir/config_${variant}.mk"
    local gds="$HERE/results/$PLATFORM/$DESIGN/$variant/6_final.gds"
    local rpt="$HERE/reports/$PLATFORM/$DESIGN/$variant/6_finish.rpt"
    local rlog="$HERE/logs/$PLATFORM/$DESIGN/$variant/6_report.log"

    echo
    echo "════════ $variant  (LANES=$lanes ACC_W=$acc clk=${clk}ns) ════════"

    # per-variant clock constraints (clone base SDC, swap the period).
    # asap7 SDC time unit is picoseconds; multiply ns→ps for that platform.
    local clk_sdc; clk_sdc="$(awk -v c="$clk" -v u="$_TIME_UNIT" 'BEGIN{printf "%g", c*u}')"
    sed "s/^set clk_period.*/set clk_period    ${clk_sdc}/" \
        "$cfgdir/constraint.sdc" > "$gen_sdc"

    # per-variant config.mk: use hard assignments (=, not ?=) so the swept values
    # are authoritative and cannot be overridden by environment variables.
    cat > "$gen_cfg" <<EOF
export DESIGN_HOME = .
export DESIGN_NAME = $DESIGN
export PLATFORM    = $PLATFORM
export VERILOG_FILES = \$(DESIGN_HOME)/src/\$(DESIGN_NAME)/tinymac_accel.v \\
                       \$(DESIGN_HOME)/src/\$(DESIGN_NAME)/int8_mac_array.v \\
                       \$(DESIGN_HOME)/src/\$(DESIGN_NAME)/requantize.v
export SDC_FILE      = \$(DESIGN_HOME)/$PLATFORM/\$(DESIGN_NAME)/constraint_${variant}.sdc
export CORE_UTILIZATION      = 40
export PLACE_DENSITY          = 0.60
export SYNTH_REPEATABLE_BUILD = 1
export VERILOG_TOP_PARAMS = LANES $lanes ACC_W $acc
EOF

    local make_status=0
    if [ -f "$gds" ]; then
        echo "  (already built — reusing $gds)"
    else
        timeout 2400 \
            make --file="$ORFS/flow/Makefile" \
                 FLOW_HOME="$ORFS/flow" WORK_HOME="$HERE" \
                 DESIGN_CONFIG="$gen_cfg" FLOW_VARIANT="$variant" \
                 > "$HERE/sweep_${variant}.log" 2>&1 \
            || make_status=$?
        if [ "$make_status" -ne 0 ]; then
            echo "  !! make exited with status $make_status — see sweep_${variant}.log"
        fi
    fi

    # ── parse metrics ────────────────────────────────────────────────────────
    # area/util live in the report LOG; timing/power in the finish RPT. Lines:
    #   (6_report.log) "Design area 19738 um^2 48% utilization."
    #   (6_finish.rpt) "wns max -1.72"   "tns max -25.86"  (value = last field)
    #                  "core_clock period_min = 3.72 fmax = 268.64"
    #                  "setup violation count 40"
    #                  report_power "Total ... <total_W> 100.0%"  (total = field 5)
    # Success requires BOTH make exit 0 AND a finish report (stale report + new
    # failure = FAIL; report-file existence alone is NOT sufficient).
    local status area util wns tns pw fmax setupv met
    if [ "$make_status" -ne 0 ]; then
        status="FAIL"; area=""; util=""; wns=""; tns=""; pw=""; fmax=""; setupv=""; met="?"
    elif [ -f "$rpt" ]; then
        status="ok"
        area=$(grep -m1 "Design area" "$rlog" 2>/dev/null | awk '{print $3}')
        util=$(grep -m1 "Design area" "$rlog" 2>/dev/null | awk '{print $5}' | tr -d '%')
        wns=$(grep -m1 -E '^wns ' "$rpt" | awk '{print $NF}')
        tns=$(grep -m1 -E '^tns ' "$rpt" | awk '{print $NF}')
        fmax=$(grep -m1 'period_min' "$rpt" | sed -nE 's/.*fmax *= *([0-9.]+).*/\1/p')
        setupv=$(grep -m1 'setup violation count' "$rpt" | awk '{print $NF}')
        local pw_w; pw_w=$(grep -m1 -E '^Total ' "$rpt" | awk '{print $5}')
        pw=$(awk -v w="$pw_w" 'BEGIN{ if(w=="")print ""; else printf "%.1f", w*1000 }')
        met=$(awk -v w="$wns" 'BEGIN{ if(w=="")print"?"; else if(w+0<0)print"NO"; else print"yes" }')
    else
        status="FAIL"; area=""; util=""; wns=""; tns=""; pw=""; fmax=""; setupv=""; met="?"
        echo "  !! no finish report — see sweep_${variant}.log"
    fi
    [ -f "$gds" ] || gds="(none)"

    echo "$RUN_ID,$lanes,$acc,$clk,$variant,$status,$area,$util,$wns,$tns,$setupv,$pw,$fmax,$met,$gds" >> "$CSV"
    printf "  area=%s um2  util=%s%%  WNS=%s ns (met=%s)  power=%s mW  Fmax=%s MHz\n" \
        "${area:-NA}" "${util:-NA}" "${wns:-NA}" "${met}" "${pw:-NA}" "${fmax:-NA}"
}

while IFS= read -r line; do
    [ -z "$line" ] && continue
    read -r L A C <<< "$line"
    run_one "$L" "$A" "$C"
done <<< "$GRID"

echo
echo "════════════════════════ SWEEP SUMMARY ════════════════════════"
column -t -s, "$CSV"
echo
echo "CSV: $CSV"
echo "Open any layout:  klayout results/$PLATFORM/$DESIGN/<variant>/6_final.gds"
echo "Or in the OpenROAD GUI per variant:"
echo "  make --file=$ORFS/flow/Makefile FLOW_HOME=$ORFS/flow WORK_HOME=$HERE \\"
echo "       DESIGN_CONFIG=$HERE/$PLATFORM/$DESIGN/config_<variant>.mk \\"
echo "       FLOW_VARIANT=<variant> gui_final"
