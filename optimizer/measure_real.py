#!/usr/bin/env python3
"""
measure_real.py — run the Verilator sim with each (mac_lanes, acc_width) combo
and record the actual per-inference cycle counts.

Run in WSL:
    python3 optimizer/measure_real.py

Writes results to optimizer/sim_measurements.txt and updates
SW_BASELINE_CYCLES in optimizer/runner.py and max_speedup in
optimizer/search_space.yaml.
"""
import re
import subprocess
import sys
from pathlib import Path

REPO   = Path(__file__).parent.parent
SIM    = REPO / "sim" / "verilator" / "sim_picorv32"
FW     = REPO / "firmware" / "picorv32_baremetal" / "firmware.bin"
OUT    = Path(__file__).parent / "sim_measurements.txt"

# Sim timeout per run (one inference per process call = fast)
TIMEOUT = 90

MAC_LANES_VALS  = [1, 2, 4, 8, 16]
ACC_WIDTH_VALS  = [32, 24, 16]


def run_one(mac_lanes: int, acc_width: int) -> dict | None:
    """Run sim, capture stdout, return first-vector metrics dict or None on failure."""
    cmd = [str(SIM), str(FW), "--mac-lanes", str(mac_lanes), "--acc-width", str(acc_width)]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=TIMEOUT)
    except subprocess.TimeoutExpired:
        return {"error": "timeout", "mac_lanes": mac_lanes, "acc_width": acc_width}
    except Exception as e:
        return {"error": str(e), "mac_lanes": mac_lanes, "acc_width": acc_width}

    stdout, stderr = proc.stdout, proc.stderr

    # --- first data line: vec,label,result,correct,logit0,logit1,cycles ---
    lines = [ln for ln in stdout.splitlines() if ln and not ln.startswith("vec")]
    if not lines:
        return {"error": "no_output", "mac_lanes": mac_lanes, "acc_width": acc_width}

    first = lines[0].split(",")
    if len(first) < 7:
        return {"error": f"short_line: {lines[0]}", "mac_lanes": mac_lanes, "acc_width": acc_width}

    try:
        cycles  = int(first[6])
        correct = int(first[3])
    except ValueError:
        return {"error": f"parse_fail: {lines[0]}", "mac_lanes": mac_lanes, "acc_width": acc_width}

    # total sim cycles from stderr
    m = re.search(r"Done in (\d+) cycles", stderr)
    total_sim = int(m.group(1)) if m else None

    return {
        "mac_lanes":  mac_lanes,
        "acc_width":  acc_width,
        "cycles_v0":  cycles,     # first-vector per-inference cycles
        "total_sim":  total_sim,
        "correct_v0": correct,    # 1 = correct for first vector
    }


def main():
    if not SIM.exists():
        print(f"ERROR: sim binary not found at {SIM}")
        print("Rebuild: cd sim/verilator && make")
        sys.exit(1)
    if not FW.exists():
        print(f"ERROR: firmware.bin not found at {FW}")
        print("Rebuild: cd firmware/picorv32_baremetal && make")
        sys.exit(1)

    rows = []
    header = f"{'lanes':>5}  {'acc_w':>5}  {'cycles/inf':>12}  {'total_sim':>12}  {'correct_v0':>10}"
    print(header)
    print("-" * len(header))

    for lanes in MAC_LANES_VALS:
        for acc in ACC_WIDTH_VALS:
            r = run_one(lanes, acc)
            rows.append(r)
            if r.get("error"):
                print(f"{lanes:>5}  {acc:>5}  ERROR: {r['error']}")
            else:
                print(
                    f"{lanes:>5}  {acc:>5}  "
                    f"{r['cycles_v0']:>12,}  "
                    f"{str(r['total_sim'] or '?'):>12}  "
                    f"{r['correct_v0']:>10}"
                )
            sys.stdout.flush()

    # --- derive SW baseline (no hooks) ---
    # The SW baseline run should be done with hooks commented out.
    # Use 11_196_638 from the just-confirmed measurement if not overridden.
    SW_BASELINE = 11_196_638

    # --- find the best config (highest speedup, acc=32, correct_v0=1) ---
    valid = [r for r in rows if not r.get("error") and r["correct_v0"] == 1 and r["acc_width"] == 32]
    if valid:
        best = min(valid, key=lambda x: x["cycles_v0"])
        max_speedup = SW_BASELINE / best["cycles_v0"]
        print(f"\nSW baseline (no-accel):    {SW_BASELINE:,} cycles/inference")
        print(f"Best config:               lanes={best['mac_lanes']}  acc={best['acc_width']}b  "
              f"{best['cycles_v0']:,} cycles/inference")
        print(f"Max measured speedup:      {max_speedup:.1f}x")
        print(f"\n→ Update in search_space.yaml:  max_speedup: {round(max_speedup * 1.1)}.0"
              f"  (10% headroom above observed max)")
        print(f"→ Update in runner.py:  SW_BASELINE_CYCLES = {SW_BASELINE}")

    # --- save results ---
    with open(OUT, "w") as f:
        f.write(header + "\n")
        f.write("-" * len(header) + "\n")
        for r in rows:
            if r.get("error"):
                f.write(f"{r['mac_lanes']:>5}  {r['acc_width']:>5}  ERROR: {r['error']}\n")
            else:
                spd = SW_BASELINE / r["cycles_v0"]
                f.write(
                    f"{r['mac_lanes']:>5}  {r['acc_width']:>5}  "
                    f"{r['cycles_v0']:>12,}  "
                    f"{str(r['total_sim'] or '?'):>12}  "
                    f"{r['correct_v0']:>10}  "
                    f"speedup={spd:.1f}x\n"
                )
        f.write(f"\nSW baseline: {SW_BASELINE:,} cycles/inference\n")

    print(f"\nResults saved to: {OUT}")


if __name__ == "__main__":
    main()
