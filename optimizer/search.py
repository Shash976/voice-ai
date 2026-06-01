"""search.py — DEPRECATED.  Use run_optimizer.py instead.

This file has been superseded by run_optimizer.py + env.py + agents/.
It is kept only as a reference.  DO NOT USE for new experiments:

  1. Its reward function (speedup / mac_lanes) is different from and
     incompatible with reward.py.  Mixing records from both scripts in
     results.jsonl produces a chart with incoherent reward axes.

  2. It does not pass acc_width to the sim, so accumulator_width is always
     treated as 32 regardless of what you configure.

  3. Its Bayesian mode uses Optuna over a 5-point categorical space — TPE
     over 5 discrete values is equivalent to random search.

Run this instead:
    python3 optimizer/run_optimizer.py --agent evo   --trials 40
    python3 optimizer/run_optimizer.py --agent bayesian --trials 30

Original search.py docstring follows:
-----------------------------------------------------------------------
Runs Verilator simulations across a parameter grid, logs every trial to
results.jsonl, and prints a ranked summary at the end.

Usage:
    python search.py              # grid search over mac_lanes [1,2,4,8,16]
    python search.py --bayesian   # Optuna Bayesian search (more params)
    python search.py --trials N   # override number of Optuna trials
"""

import argparse
import json
import sys
import time
from pathlib import Path

from runner import SW_BASELINE_CYCLES, run_sim

RESULTS_FILE = Path(__file__).parent / "results.jsonl"

# ── Search space ──────────────────────────────────────────────────────────────

MAC_LANES_GRID = [1, 2, 4, 8, 16]


# ── Reward function ───────────────────────────────────────────────────────────

def reward(result: dict) -> float:
    """
    Higher is better.
    - Accuracy must be 100 % (hard constraint, else large penalty).
    - Primary objective: speedup / area_proxy  (efficiency frontier).
    - Area proxy: mac_lanes (chip area scales roughly linearly with lane count).
    """
    if result["accuracy"] < 1.0:
        return -100.0
    speedup = result["speedup"]
    area    = result["mac_lanes"]
    return speedup / area


# ── Logging ───────────────────────────────────────────────────────────────────

def _log(record: dict) -> None:
    with open(RESULTS_FILE, "a") as f:
        f.write(json.dumps(record) + "\n")


def _run_and_log(mac_lanes: int, trial_num: int) -> dict:
    t0 = time.time()
    result = run_sim(mac_lanes)
    elapsed = time.time() - t0

    r = reward(result)
    record = {
        "trial":       trial_num,
        "timestamp":   time.time(),
        "mac_lanes":   mac_lanes,
        "avg_cycles":  result["avg_cycles"],
        "total_cycles":result["total_cycles"],
        "accuracy":    result["accuracy"],
        "speedup":     result["speedup"],
        "reward":      r,
        "elapsed_s":   round(elapsed, 2),
    }
    _log(record)
    print(
        f"  trial {trial_num:3d} | mac_lanes={mac_lanes:2d} | "
        f"avg_cycles={result['avg_cycles']:7,d} | "
        f"speedup={result['speedup']:6.1f}x | "
        f"reward={r:7.3f} | {elapsed:.1f}s"
    )
    return record


# ── Grid search ───────────────────────────────────────────────────────────────

def grid_search() -> list[dict]:
    print(f"Grid search  mac_lanes={MAC_LANES_GRID}")
    print(f"SW baseline  avg_cycles={SW_BASELINE_CYCLES:,}")
    print("-" * 72)
    RESULTS_FILE.unlink(missing_ok=True)

    records = []
    for i, ml in enumerate(MAC_LANES_GRID):
        rec = _run_and_log(ml, i)
        records.append(rec)

    _print_summary(records)
    return records


# ── Bayesian search (Optuna) ──────────────────────────────────────────────────

def bayesian_search(n_trials: int = 20) -> list[dict]:
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        print("ERROR: optuna not installed. Run: pip install --user optuna")
        sys.exit(1)

    print(f"Bayesian search  n_trials={n_trials}")
    print(f"SW baseline  avg_cycles={SW_BASELINE_CYCLES:,}")
    print("-" * 72)
    RESULTS_FILE.unlink(missing_ok=True)

    records = []
    trial_num = [0]

    def objective(trial: "optuna.Trial") -> float:
        mac_lanes = trial.suggest_categorical("mac_lanes", MAC_LANES_GRID)
        rec = _run_and_log(mac_lanes, trial_num[0])
        trial_num[0] += 1
        records.append(rec)
        return rec["reward"]

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)

    _print_summary(records)
    return records


# ── Summary ───────────────────────────────────────────────────────────────────

def _print_summary(records: list[dict]) -> None:
    print("\n" + "=" * 72)
    print("Results (sorted by reward):")
    print(f"  {'mac_lanes':>9}  {'avg_cycles':>10}  {'speedup':>8}  {'reward':>8}")
    for r in sorted(records, key=lambda x: x["reward"], reverse=True):
        print(f"  {r['mac_lanes']:>9}  {r['avg_cycles']:>10,}  "
              f"{r['speedup']:>8.1f}x  {r['reward']:>8.3f}")
    best = max(records, key=lambda x: x["reward"])
    print(f"\nBest config: mac_lanes={best['mac_lanes']}  "
          f"speedup={best['speedup']:.1f}x  reward={best['reward']:.3f}")
    print(f"Results saved to: {RESULTS_FILE}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bayesian", action="store_true",
                        help="Use Optuna Bayesian search instead of grid search")
    parser.add_argument("--trials", type=int, default=20,
                        help="Number of Optuna trials (default 20)")
    args = parser.parse_args()

    if args.bayesian:
        bayesian_search(n_trials=args.trials)
    else:
        grid_search()
