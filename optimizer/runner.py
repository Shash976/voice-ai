"""runner.py — run one Verilator simulation trial and return parsed metrics.

Key design decisions
--------------------
* run_sim() is memoised with @lru_cache.  The sim output depends only on
  (mac_lanes, acc_width) — both ints — so identical configs never re-launch
  the binary.  A 30-trial run over a 5-lane × 3-acc-width space (15 distinct
  sim inputs) now costs at most 15 subprocess calls, not 30.

* Firmware output format: main.c prints  "correct=N/N avg_cycles=A"  where A
  is  total_cycles / n_total  (per-inference average).  Older WSL binaries may
  print  total_cycles  without dividing.  We detect this by checking whether
  the reported value is close to the testbench's own total-cycle count (stderr:
  "Done in X cycles"), which is dominated by n_total × per_inference cycles
  — if they match within 5 %, the firmware printed the total and we divide.

* SW_BASELINE_CYCLES is the Stage-3 per-inference average measured from a
  no-accel sim run.  Update it by running: make run (no --mac-lanes flag)
  and reading avg_cycles from the firmware CSV summary line.
"""

import re
import subprocess
import warnings
from functools import lru_cache
from pathlib import Path

_REPO    = Path(__file__).parent.parent
SIM_DIR  = _REPO / "sim" / "verilator"
SIM_BIN  = SIM_DIR / "sim_picorv32"
FIRMWARE = _REPO / "firmware" / "picorv32_baremetal" / "firmware.bin"

SIM_TIMEOUT = 120   # seconds

# Stage-3 pure-SW per-inference baseline (no accelerator, 64 test vectors).
# EMPIRICALLY MEASURED 2026-06-01 by running the sim with hooks commented out:
#   vec0 cycles = 11,196,638  (11.2 M cycles / inference)
# The old value of 175,324 was stale — it was from an earlier smaller model.
# Re-measure whenever the model, test vectors, or firmware changes:
#   1. Comment out tinyvad_*_hook assignments in firmware/picorv32_baremetal/main.c
#   2. make -C firmware/picorv32_baremetal && make -C sim/verilator
#   3. ./sim_picorv32 firmware.bin | sed -n '2p' | cut -d, -f7
SW_BASELINE_CYCLES = 11_196_638


@lru_cache(maxsize=None)
def run_sim(mac_lanes: int, acc_width: int = 32) -> dict:
    """
    Launch the Verilator binary with the given (mac_lanes, acc_width) and
    return a metrics dict.  Results are cached — the function is deterministic
    for a fixed firmware binary and sim binary.

    Returns
    -------
    dict with keys:
        mac_lanes, acc_width, avg_cycles, total_cycles,
        accuracy, correct, n_total, speedup

    Raises
    ------
    FileNotFoundError   if the sim binary or firmware.bin are missing
    RuntimeError        if the sim exits non-zero or output can't be parsed
    """
    if not SIM_BIN.exists():
        raise FileNotFoundError(
            f"Sim binary not found: {SIM_BIN}\n"
            "Build it in WSL: cd sim/verilator && make"
        )
    if not FIRMWARE.exists():
        raise FileNotFoundError(
            f"firmware.bin not found: {FIRMWARE}\n"
            "Build it in WSL: cd firmware/picorv32_baremetal && make"
        )

    cmd = [
        str(SIM_BIN), str(FIRMWARE),
        "--mac-lanes", str(mac_lanes),
        "--acc-width",  str(acc_width),
    ]
    proc = subprocess.run(
        cmd, capture_output=True, text=True, timeout=SIM_TIMEOUT
    )

    stdout = proc.stdout
    stderr = proc.stderr

    # ── Parse firmware summary line ───────────────────────────────────────────
    # main.c prints: "correct=64/64 avg_cycles=911"
    # where avg_cycles = total_inference_cycles / n_total (per-inference avg).
    # Older WSL binaries may print total_cycles without dividing.
    m = re.search(r"correct=(\d+)/(\d+)\s+avg_cycles=(\d+)", stdout)
    if not m:
        raise RuntimeError(
            f"Could not parse sim output for mac_lanes={mac_lanes} acc_width={acc_width}.\n"
            f"stdout: {stdout[:400]}\nstderr: {stderr[:400]}"
        )

    correct   = int(m.group(1))
    n_total   = int(m.group(2))
    reported  = int(m.group(3))

    # ── Detect old firmware that prints total instead of per-inference avg ────
    m2          = re.search(r"Done in (\d+) cycles", stderr)
    total_cyc   = int(m2.group(1)) if m2 else None

    if total_cyc and n_total > 1:
        ratio = abs(reported - total_cyc) / max(total_cyc, 1)
        if ratio < 0.05:
            # reported ≈ total_sim_cycles  →  firmware printed total, not average
            warnings.warn(
                f"[runner] Firmware appears to print total_cycles ({reported}) "
                f"rather than per-inference average.  Dividing by n_total={n_total}. "
                "Rebuild firmware from current main.c to fix permanently.",
                stacklevel=2,
            )
            reported = reported // n_total

    avg_cyc  = reported
    accuracy = correct / n_total

    return {
        "mac_lanes":    mac_lanes,
        "acc_width":    acc_width,
        "avg_cycles":   avg_cyc,
        "total_cycles": total_cyc,
        "accuracy":     accuracy,
        "correct":      correct,
        "n_total":      n_total,
        "speedup":      SW_BASELINE_CYCLES / max(avg_cyc, 1),
    }


def cache_info() -> str:
    """Human-readable lru_cache hit/miss statistics."""
    ci = run_sim.cache_info()
    return f"sim cache: {ci.hits} hits / {ci.misses} misses / {ci.currsize} stored"


if __name__ == "__main__":
    import sys
    lanes = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    acc_w = int(sys.argv[2]) if len(sys.argv) > 2 else 32
    result = run_sim(lanes, acc_w)
    print(result)
    print(cache_info())
