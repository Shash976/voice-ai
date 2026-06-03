#!/usr/bin/env bash
# sync.sh — copy the canonical ORFS design files into the bazel-orfs gallery
# workspace package where the flow actually runs (cached OpenROAD toolchain).
#
# Usage (WSL):  physical/orfs/sync.sh   [GALLERY_DIR]
# Default GALLERY_DIR = ~/bazel-orfs/gallery
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
GALLERY="${1:-$HOME/bazel-orfs/gallery}"
PKG="$GALLERY/tinymac"

mkdir -p "$PKG"
cp "$REPO/rtl/accel/tinymac_accel.v"  "$PKG/"
cp "$REPO/rtl/accel/int8_mac_array.v" "$PKG/"
cp "$REPO/rtl/accel/requantize.v"     "$PKG/"
cp "$HERE/BUILD.bazel"                "$PKG/"
cp "$HERE/constraints.sdc"            "$PKG/"

echo "Synced tinymac design -> $PKG"
echo "Run:  (cd '$GALLERY' && bazel build //tinymac:tinymac_accel_synth)"
