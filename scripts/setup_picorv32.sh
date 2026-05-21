#!/bin/bash
# setup_picorv32.sh — clone PicoRV32 RTL into rtl/picorv32/
# Run once from repo root: bash scripts/setup_picorv32.sh

set -e
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="$REPO_ROOT/rtl/picorv32"

if [ -f "$DEST/picorv32.v" ]; then
    echo "PicoRV32 already present at $DEST"
    exit 0
fi

mkdir -p "$DEST"
echo "Cloning PicoRV32..."
git clone --depth=1 https://github.com/YosysHQ/picorv32.git /tmp/picorv32_clone

# We only need the core Verilog file
cp /tmp/picorv32_clone/picorv32.v "$DEST/"
rm -rf /tmp/picorv32_clone

echo "Done — $DEST/picorv32.v ready"
