#!/usr/bin/env bash
# synth_area.sh — standalone Yosys synthesis of tinymac_accel against a PDK
# standard-cell liberty, for gate count + cell area. No bazel / OpenROAD / net
# needed. This is the *synthesis* half of Stage 6; full place-and-route → GDS
# needs an OpenROAD binary (see README.md).
#
# Usage (WSL):  physical/orfs/synth_area.sh [sky130hd|nangate45]
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
PLATFORM="${1:-sky130hd}"
PLAT="$HOME/OpenROAD-flow-scripts/flow/platforms"

case "$PLATFORM" in
  sky130hd)  LIB="$PLAT/sky130hd/lib/sky130_fd_sc_hd__tt_025C_1v80.lib" ;;
  nangate45) LIB="$PLAT/nangate45/lib/NangateOpenCellLibrary_typical.lib" ;;
  *) echo "unknown platform '$PLATFORM' (sky130hd|nangate45)"; exit 1 ;;
esac
[ -f "$LIB" ] || { echo "liberty not found: $LIB"; exit 1; }

RTL="$REPO/rtl/accel"
OUT="$HERE/reports"; mkdir -p "$OUT"

yosys -ql "$OUT/${PLATFORM}_synth.log" -p "
read_verilog $RTL/tinymac_accel.v $RTL/int8_mac_array.v $RTL/requantize.v;
hierarchy -check -top tinymac_accel;
synth -top tinymac_accel -flatten;
dfflibmap -liberty $LIB;
abc -liberty $LIB;
opt_clean -purge;
tee -o $OUT/${PLATFORM}_area.rpt stat -top tinymac_accel -liberty $LIB;
write_verilog -noattr $OUT/${PLATFORM}_netlist.v;
"
echo "── $PLATFORM ──"
grep -E "Number of cells|Number of cell|Chip area|Number of wires" "$OUT/${PLATFORM}_area.rpt" || true
echo "reports → $OUT/${PLATFORM}_{area.rpt,synth.log,netlist.v}"
