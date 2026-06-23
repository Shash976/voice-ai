"""cascade_env.py — CascadeOptEnv: gym-style env driving the evaluation funnel.

Subclasses OptEnv so the search-space helpers (sample_random, neighbors,
default_config) and the agent interface are reused verbatim — every existing
agent (random, evo, ucb, bayesian) works unchanged. step() pushes the config
through cascade.evaluate() (validate→elaborate→sim→proxy→full, short-circuiting)
and scores it with cascade_reward.

Results stream to results_cascade.jsonl (separate from the sim and physical
tracks). `max_stage` lets you cap the funnel: 'proxy' for fast search (seconds),
'full' for ground-truth PPA (minutes).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import yaml

from gen1.cascade import evaluate
from common.cascade_reward import compute_cascade_reward
from gen1.env import OptEnv

CASCADE_RESULTS_FILE = Path(__file__).resolve().parent.parent / "results" / "gen1" / "results_cascade.jsonl"
DEFAULT_SPACE = Path(__file__).resolve().parent / "search_space_full.yaml"


class CascadeOptEnv(OptEnv):
    #: UCBAgent normalisation window: cascade penalties reach −100 (invalid),
    #: so the default behavioral-track bounds (−12, 4.5) would clamp the entire
    #: penalty ladder to 0.0.  Setting a wider lower bound preserves the
    #: escalating-penalty signal (invalid < elaborate < sim < proxy < full-fail).
    reward_bounds: tuple[float, float] = (-100.0, 4.5)

    def __init__(self, search_space_path=None, platform: str = "nangate45",
                 max_stage: str = "full") -> None:
        if search_space_path is None:
            search_space_path = DEFAULT_SPACE
        super().__init__(search_space_path)
        # OptEnv ignores constraints/gates — load them here.
        with open(search_space_path, encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        self.constraints: list[str] = raw.get("constraints", []) or []
        self.gates: dict = raw.get("gates", {}) or {}
        self.platform = platform
        self.max_stage = max_stage
        self._results_file = CASCADE_RESULTS_FILE

    # ── Evaluation ────────────────────────────────────────────────────────────

    def step(self, config: dict) -> tuple[dict, float, bool, dict]:
        t0 = time.time()
        result = evaluate(
            config,
            space=self.search_space,
            constraints=self.constraints,
            gates=self.gates,
            platform=self.platform,
            max_stage=self.max_stage,
        )
        scored = compute_cascade_reward(result, self._reward_cfg)
        reward = scored["reward"]
        elapsed = round(time.time() - t0, 2)

        metrics = result.get("metrics") or {}
        record = {
            "trial":     self._trial,
            "timestamp": time.time(),
            "config":    config,
            "reached":   result.get("reached"),
            "failed_stage": result.get("failed_stage"),
            "reason":    result.get("reason", ""),
            "stages":    result.get("stages", {}),
            "sim":       result.get("sim"),
            "metrics":   metrics,
            "scored":    scored,
            "reward":    reward,
            "elapsed_s": elapsed,
            # flat fields for quick reading / dashboards
            "lanes":        config.get("mac_lanes"),
            "acc_w":        config.get("accumulator_width"),
            "clk_ns":       config.get("clock_period_ns"),
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
            return {"trial": 0, "best_reward": float("-inf"),
                    "best_config": self.default_config(), "history_len": 0}
        best = max(self._history, key=lambda r: r["reward"])
        n_full = sum(1 for r in self._history if r.get("reached") == "full")
        return {
            "trial":        self._trial,
            "best_reward":  best["reward"],
            "best_config":  best["config"],
            "best_area_um2": best.get("area_um2"),
            "reached_full": n_full,
            "history_len":  len(self._history),
        }

    # ── Logging (override OptEnv's module-global path) ────────────────────────

    def _log(self, record: dict) -> None:
        self._results_file.parent.mkdir(parents=True, exist_ok=True)
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
