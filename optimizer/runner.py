"""runner.py — run one Verilator simulation trial and return parsed metrics."""

import re
import subprocess
from pathlib import Path

_REPO = Path(__file__).parent.parent
SIM_DIR  = _REPO / "sim" / "verilator"
SIM_BIN  = SIM_DIR / "sim_picorv32"
FIRMWARE = _REPO / "firmware" / "picorv32_baremetal" / "firmware.bin"

# Stage-3 pure-software baseline (no accelerator), used for speedup calculation.
SW_BASELINE_CYCLES = 175_324


def run_sim(mac_lanes: int, timeout: int = 120) -> dict:
    """
    Run the Verilator sim with the given mac_lanes and return a metrics dict:
      mac_lanes, avg_cycles, total_cycles, accuracy, correct, speedup
    Raises RuntimeError if the binary fails or output cannot be parsed.
    """
    if not SIM_BIN.exists():
        raise FileNotFoundError(
            f"sim binary not found at {SIM_BIN}. "
            "Run: cd sim/verilator && make CROSS=riscv-none-elf"
        )
    if not FIRMWARE.exists():
        raise FileNotFoundError(
            f"firmware.bin not found at {FIRMWARE}. "
            "Run: cd firmware/picorv32_baremetal && make CROSS=riscv-none-elf"
        )

    cmd = [str(SIM_BIN), str(FIRMWARE), "--mac-lanes", str(mac_lanes)]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)

    stdout = proc.stdout
    stderr = proc.stderr

    # Firmware prints: "correct=64/64 avg_cycles=911"
    m = re.search(r"correct=(\d+)/(\d+)\s+avg_cycles=(\d+)", stdout)
    if not m:
        raise RuntimeError(
            f"Could not parse sim output for mac_lanes={mac_lanes}.\n"
            f"stdout: {stdout[:400]}\nstderr: {stderr[:400]}"
        )

    correct   = int(m.group(1))
    total     = int(m.group(2))
    avg_cyc   = int(m.group(3))
    accuracy  = correct / total

    # Testbench prints: "Done in 58342 cycles"
    m2 = re.search(r"Done in (\d+) cycles", stderr)
    total_cyc = int(m2.group(1)) if m2 else None

    return {
        "mac_lanes":    mac_lanes,
        "avg_cycles":   avg_cyc,
        "total_cycles": total_cyc,
        "accuracy":     accuracy,
        "correct":      correct,
        "speedup":      SW_BASELINE_CYCLES / avg_cyc,
    }


if __name__ == "__main__":
    import sys
    lanes = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    result = run_sim(lanes)
    print(result)
