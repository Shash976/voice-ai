"""physical_env.py — PhysicalOptEnv: drives the real ORFS flow instead of the sim.

Subclasses OptEnv so the search-space helpers (sample_random, neighbors,
default_config) and the agent interface are reused verbatim — every existing
agent (random, evo, ucb, bayesian) works unchanged.  Only the evaluation in
step() changes: instead of a Verilator sim + analytical proxies, it runs the
actual RTL→GDS flow (physical_runner) and scores the measured area / timing /
power (physical_reward).

Config keys map to RTL/flow parameters:
    mac_lanes          → LANES        (chparam)
    accumulator_width  → ACC_W        (chparam)
    clock_period_ns    → SDC clock period (ns)

Results are logged to results_physical.jsonl (separate from the sim track's
results.jsonl) so the two can coexist.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from env import OptEnv
from physical_reward import behavioral_cycles, compute_physical_reward
from physical_runner import run_physical

PHYS_RESULTS_FILE = Path(__file__).parent / "results_physical.jsonl"


class PhysicalOptEnv(OptEnv):
    def __init__(self, search_space_path=None, platform: str = "nangate45") -> None:
        super().__init__(search_space_path)
        self.platform = platform
        self._results_file = PHYS_RESULTS_FILE

    # ── Evaluation ────────────────────────────────────────────────────────────

    def step(self, config: dict) -> tuple[dict, float, bool, dict]:
        t0 = time.time()

        lanes = int(config.get("mac_lanes", 4))
        acc_w = int(config.get("accumulator_width", 24))
        clk   = float(config.get("clock_period_ns", 5))

        metrics = run_physical(lanes, acc_w, clk, self.platform)  # cached per config
        cycles  = behavioral_cycles(lanes)
        scored  = compute_physical_reward(metrics, self._reward_cfg, cycles=cycles)
        reward  = scored["reward"]
        elapsed = round(time.time() - t0, 2)

        record = {
            "trial":     self._trial,
            "timestamp": time.time(),
            "config":    config,
            "metrics":   metrics,
            "scored":    scored,
            "cycles":    round(cycles, 1),
            "reward":    reward,
            "elapsed_s": elapsed,
            # flat fields for quick reading / dashboards
            "lanes":        lanes,
            "acc_w":        acc_w,
            "clk_ns":       clk,
            "area_um2":     metrics.get("area_um2"),
            "fmax_mhz":     metrics.get("fmax_mhz"),
            "power_mw":     metrics.get("power_mw"),
            "timing_met":   metrics.get("timing_met"),
            "real_speedup": scored.get("real_speedup"),
        }

        self._history.append(record)
        self._log(record)
        self._trial += 1
        return self._make_state(), reward, False, record

    # ── State (override: base reads sim-only keys) ────────────────────────────

    def _make_state(self) -> dict:
        if not self._history:
            return {
                "trial": 0, "best_reward": float("-inf"),
                "best_config": self.default_config(),
                "best_speedup": 0.0, "best_area_um2": None, "history_len": 0,
            }
        best = max(self._history, key=lambda r: r["reward"])
        return {
            "trial":         self._trial,
            "best_reward":   best["reward"],
            "best_config":   best["config"],
            "best_speedup":  best.get("real_speedup"),
            "best_area_um2": best.get("area_um2"),
            "history_len":   len(self._history),
        }

    # ── Logging (override OptEnv's module-global path) ────────────────────────

    def _log(self, record: dict) -> None:
        with open(self._results_file, "a") as f:
            f.write(json.dumps(record) + "\n")

    def clear_results(self) -> None:
        self._results_file.unlink(missing_ok=True)
        self._history.clear()
        self._trial = 0

    def load_existing_results(self) -> int:
        if not self._results_file.exists():
            return 0
        loaded = 0
        with open(self._results_file) as f:
            for line in f:
                s = line.strip()
                if not s:
                    continue
                try:
                    self._history.append(json.loads(s))
                    loaded += 1
                except json.JSONDecodeError:
                    pass
        self._trial = len(self._history)
        return loaded
