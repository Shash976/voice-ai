#!/usr/bin/env python3
"""
measure_real.py — run the Verilator sim with each (mac_lanes, acc_width) combo
and record the actual per-inference cycle counts.

Run in WSL:
    python3 optimizer/measure_real.py

Writes raw results to optimizer/sim_measurements.txt and PRINTS recommended
constants for SW_BASELINE_CYCLES (runner.py) and max_speedup (search_space.yaml).
It does NOT edit those files — copy the recommended values in by hand.

NOTE on max_speedup: the reward's speedup term is FREQUENCY-AWARE (real_speedup,
see reward.py), not the raw cycle ratio.  So the recommended max_speedup below is
derived from reward.real_speedup() over the measured grid × clock choices, NOT
from the cycle-based "speedup" (which is only reported for context).
"""
import math
import re
import subprocess
import sys
from pathlib import Path

# Bootstrap: make optimizer/ root importable (common/ is one level below it)
import pathlib as _pl
sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))
try:
    from gen1 import reward as _R
except Exception:  # pragma: no cover - reward import is optional for raw measurement
    _R = None

REPO   = Path(__file__).resolve().parent.parent.parent
SIM    = REPO / "sim" / "verilator" / "sim_picorv32"
FW     = REPO / "firmware" / "picorv32_baremetal" / "firmware.bin"
OUT    = Path(__file__).resolve().parent / "sim_measurements.txt"

# Sim timeout per run (one inference per process call = fast)
TIMEOUT = 90

MAC_LANES_VALS  = [1, 2, 4, 8, 16, 32]   # full sweep including lanes=32
ACC_WIDTH_VALS  = [32, 24, 16]


def run_one(mac_lanes: int, acc_width: int) -> dict | None:
    """Run sim, capture all 64 vectors, return averaged metrics dict or None on failure.

    Uses the mean of ALL 64 per-inference cycle values (column 7 of the CSV).
    The first-vector cycle count can differ slightly from steady state (firmware
    branch-prediction warm-up); averaging over all 64 gives a stable, reproducible
    number that matches the mode to within a fraction of a cycle.
    """
    cmd = [str(SIM), str(FW), "--mac-lanes", str(mac_lanes), "--acc-width", str(acc_width)]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=TIMEOUT)
    except subprocess.TimeoutExpired:
        return {"error": "timeout", "mac_lanes": mac_lanes, "acc_width": acc_width}
    except Exception as e:
        return {"error": str(e), "mac_lanes": mac_lanes, "acc_width": acc_width}

    stdout, stderr = proc.stdout, proc.stderr

    # --- parse all data lines: vec,label,result,correct,logit0,logit1,cycles ---
    cycles_all: list[int] = []
    correct_total = 0
    n_total = 0
    for ln in stdout.splitlines():
        if not ln or ln.startswith("vec") or ln.startswith("correct="):
            continue
        parts = ln.split(",")
        if len(parts) < 7:
            continue
        try:
            c_val = int(parts[6])
            is_correct = int(parts[3])
        except ValueError:
            continue
        cycles_all.append(c_val)
        correct_total += is_correct
        n_total += 1

    if not cycles_all:
        return {"error": "no_output", "mac_lanes": mac_lanes, "acc_width": acc_width}

    avg_cycles = sum(cycles_all) / len(cycles_all)
    # Round to nearest integer for use as a table constant.
    avg_cycles_int = round(avg_cycles)

    # total sim cycles from stderr
    m = re.search(r"Done in (\d+) cycles", stderr)
    total_sim = int(m.group(1)) if m else None

    return {
        "mac_lanes":    mac_lanes,
        "acc_width":    acc_width,
        "cycles_v0":    cycles_all[0] if cycles_all else None,   # kept for back-compat
        "avg_cycles":   avg_cycles,
        "avg_cycles_int": avg_cycles_int,
        "n_vectors":    n_total,
        "correct":      correct_total,
        "total_sim":    total_sim,
        "correct_v0":   1 if (cycles_all and correct_total > 0) else 0,  # back-compat
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
    header = (f"{'lanes':>5}  {'acc_w':>5}  {'avg_cyc':>12}  "
              f"{'correct':>10}  {'total_sim':>12}")
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
                    f"{r['avg_cycles_int']:>12,}  "
                    f"{r['correct']:>4}/{r['n_vectors']:<4}  "
                    f"{str(r['total_sim'] or '?'):>12}"
                )
            sys.stdout.flush()

    # --- derive SW baseline (no hooks) ---
    # The SW baseline run should be done with hooks commented out.
    # Use 11_196_638 from the just-confirmed measurement if not overridden.
    SW_BASELINE = 11_196_638

    # --- build AVG_CYCLES table (acc=32, correct rows only) ---
    acc32_rows = {r["mac_lanes"]: r for r in rows
                  if not r.get("error") and r["acc_width"] == 32}
    print(f"\nSW baseline (no-accel):    {SW_BASELINE:,} cycles/inference")

    if acc32_rows:
        # Show the AVG_CYCLES dict for constants.py
        sorted_lanes = sorted(acc32_rows.keys())
        avg_dict = {l: acc32_rows[l]["avg_cycles_int"] for l in sorted_lanes}
        print(f"\nAVG_CYCLES = {avg_dict!r}")

        # Fit a + b/lanes to the measured data
        import numpy as np
        X = np.array([[1, 1/l] for l in sorted_lanes], dtype=float)
        y = np.array([avg_dict[l] for l in sorted_lanes], dtype=float)
        (a_fit, b_fit), *_ = np.linalg.lstsq(X, y, rcond=None)
        print(f"\nbehavioral_cycles fit:  a + b/lanes  =  {a_fit:.1f} + {b_fit:.1f} / lanes")
        print("Residuals (measured - fitted):")
        for l in sorted_lanes:
            fitted = a_fit + b_fit / l
            print(f"  lanes={l:2d}: measured={avg_dict[l]:7d}  fitted={fitted:9.1f}  "
                  f"residual={avg_dict[l]-fitted:+7.1f}")

        # --- FREQUENCY-AWARE max_speedup recommendation (matches reward.py) -------
        if _R is not None:
            clock_choices = [5, 10, 20]
            # 45-grid lanes are {1,2,4,8,16} (search_space.yaml); evaluate only those.
            grid_lanes = [l for l in sorted_lanes if l in {1, 2, 4, 8, 16}]
            max_real = 0.0
            best_real = None
            for l in grid_lanes:
                cyc = avg_dict[l]
                for acc in [16, 24, 32]:
                    for clk in clock_choices:
                        cfg = {"mac_lanes": l,
                               "accumulator_width": acc,
                               "clock_period_ns": clk}
                        rs = _R.real_speedup(cfg, cyc)
                        if rs > max_real:
                            max_real, best_real = rs, (l, acc, clk)
            rec_grid = math.ceil(max_real * 1.11)
            # Round up to the nearest 64-boundary
            rec_nice = rec_grid + (64 - rec_grid % 64) % 64 if rec_grid % 64 else rec_grid
            print(f"\nMax REAL speedup (45-grid):  {max_real:.2f}x  at lanes={best_real[0]} "
                  f"acc={best_real[1]}b clk={best_real[2]}ns")
            print(f"→  ceil({max_real:.2f} × 1.11) = {rec_grid}  →  nearest-64 = {rec_nice}")
            print(f"   Current search_space.yaml max_speedup=576  →  "
                  f"{'KEEP 576 (still conservative)' if 576 >= max_real else 'UPDATE NEEDED'}")

            # MAX_SPEEDUP_FULL for cascade space (lanes up to 32)
            max_real_full = 0.0
            for l in sorted_lanes:
                cyc = avg_dict[l]
                for acc in [24, 32]:
                    for clk in clock_choices:
                        cfg = {"mac_lanes": l, "accumulator_width": acc, "clock_period_ns": clk}
                        rs = _R.real_speedup(cfg, cyc)
                        if rs > max_real_full:
                            max_real_full = rs
            print(f"\nMax REAL speedup (full space, lanes up to 32):  {max_real_full:.2f}x")
            print(f"MAX_SPEEDUP_FULL = 1024  (set by convention; covers {max_real_full:.2f}x with headroom)")

            best_cycles = min(avg_dict[l] for l in grid_lanes)
            cycle_speedup = SW_BASELINE / best_cycles
            print(f"\nMax CYCLE speedup (grid, acc=32):  {cycle_speedup:.1f}x  "
                  f"(at {best_cycles:,} cycles/inf; frequency-independent)")
        else:
            print("\n(reward.py not importable — skipping frequency-aware max_speedup "
                  "recommendation; see reward.py for the real_speedup model.)")

    # --- save results (append with date) ---
    import datetime
    stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    with open(OUT, "a") as f:
        f.write(f"\n### Sweep run {stamp} "
                f"(V13-fixed per-chunk saturation) ###\n")
        f.write(header + "\n")
        f.write("-" * len(header) + "\n")
        for r in rows:
            if r.get("error"):
                f.write(f"{r['mac_lanes']:>5}  {r['acc_width']:>5}  ERROR: {r['error']}\n")
            else:
                spd = SW_BASELINE / r["avg_cycles_int"]
                f.write(
                    f"{r['mac_lanes']:>5}  {r['acc_width']:>5}  "
                    f"{r['avg_cycles_int']:>12,}  "
                    f"{r['correct']:>4}/{r['n_vectors']:<4}  "
                    f"{str(r['total_sim'] or '?'):>12}  "
                    f"speedup={spd:.1f}x\n"
                )
        f.write(f"SW baseline: {SW_BASELINE:,} cycles/inference\n")

    print(f"\nResults appended to: {OUT}")


if __name__ == "__main__":
    main()
