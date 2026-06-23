"""env.py — OptEnv: gym-style environment wrapping Verilator sim + proxy models.

Interface mirrors OpenAI Gym:
    state, info = env.reset()
    state, reward, done, info = env.step(config)

State is a summary dict (not a flat tensor) for easy agent introspection.
Results are appended to results.jsonl for live dashboard consumption.
"""

from __future__ import annotations

import json
import random
import time
from pathlib import Path

import yaml

from gen1.reward import compute_proxies, compute_reward, real_speedup
from gen1.runner import SW_BASELINE_CYCLES, run_sim

_OPT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_FILE = _OPT_ROOT / "results" / "gen1" / "results.jsonl"


def _load_yaml(path: str | Path) -> dict:
    # encoding pinned: the search-space YAMLs contain non-ASCII (µm², ×, →) and
    # Windows' default cp1252 would crash on them (UnicodeDecodeError).
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


class OptEnv:
    """
    Design-space optimization environment for the TinyMAC accelerator.

    step(config) does three things:
      1. Calls the Verilator sim binary with sim_params from config.
      2. Computes proxy metrics for proxy_params (area, power, timing).
      3. Returns a scalar reward via the multi-objective reward function.

    History is written to results.jsonl so the Streamlit dashboard can read
    it live without any IPC.

    ``reward_bounds`` is read by UCBAgent at construction time to set the
    normalisation window that matches this track's actual reward range.
    The 45-config behavioral track's realistic range is roughly [−10.55, +4.01];
    bounds (−12.0, 4.5) give a little headroom on both ends.
    """

    #: UCBAgent normalisation window for the behavioral-sim track.
    reward_bounds: tuple[float, float] = (-12.0, 4.5)

    def __init__(self, search_space_path: str | Path | None = None) -> None:
        if search_space_path is None:
            search_space_path = Path(__file__).parent / "search_space.yaml"
        raw = _load_yaml(search_space_path)

        self._sim_specs:   dict = raw.get("sim_params",   {})
        self._proxy_specs: dict = raw.get("proxy_params", {})
        self._reward_cfg:  dict = raw.get("reward",       {})

        self._history: list[dict] = []
        self._trial: int = 0

    # ── Search space helpers ──────────────────────────────────────────────────

    @property
    def search_space(self) -> dict:
        """Combined sim + proxy parameter specs (name → spec dict)."""
        return {**self._sim_specs, **self._proxy_specs}

    def default_config(self) -> dict:
        cfg: dict = {}
        for name, spec in self.search_space.items():
            cfg[name] = spec.get("default", spec["choices"][0])
        return cfg

    def sample_random(self, rng: random.Random | None = None) -> dict:
        r = rng or random
        return {name: r.choice(spec["choices"]) for name, spec in self.search_space.items()}

    def neighbors(self, config: dict, n: int = 1) -> list[dict]:
        """Return n configs that differ in exactly one parameter from config."""
        import random as _rnd
        result = []
        params = list(self.search_space.keys())
        for _ in range(n):
            name = _rnd.choice(params)
            spec = self.search_space[name]
            others = [c for c in spec["choices"] if c != config[name]]
            if others:
                child = dict(config)
                child[name] = _rnd.choice(others)
                result.append(child)
        return result

    # ── Gym-style interface ───────────────────────────────────────────────────

    def reset(self) -> dict:
        """
        Return the initial state dict.  Does NOT clear results.jsonl or
        history — call clear_results() explicitly for a fresh run.

        Returns a single dict (not a tuple); callers should use:
            state = env.reset()
        """
        self._trial = len(self._history)
        return self._make_state()

    def step(self, config: dict) -> tuple[dict, float, bool, dict]:
        """
        Evaluate config dict against the simulator + proxy models.

        Returns
        -------
        state   : dict  — updated state visible to agents
        reward  : float — scalar reward (higher is better)
        done    : bool  — always False (optimizer controls termination)
        info    : dict  — full result record (logged to results.jsonl)
        """
        t0 = time.time()

        # --- sim run  (mac_lanes + acc_width are genuinely simulated) ---
        mac_lanes = int(config.get("mac_lanes",          8))
        acc_width = int(config.get("accumulator_width", 32))
        sim_metrics = run_sim(mac_lanes, acc_width)   # lru_cache: free on repeated configs

        elapsed = round(time.time() - t0, 2)

        # --- proxy metrics (analytical, instant) ---
        proxies = compute_proxies(config)

        # --- frequency-aware (real-time) speedup ---------------------------------
        # The sim's cycle-based "speedup" is frequency-independent.  Here we have
        # both avg_cycles and the full config (incl. clock_period_ns), so we
        # compute the real-time speedup the reward must use and merge it into
        # proxy_metrics.  The cycle-based sim_metrics["speedup"] is left untouched
        # for the dashboard / backward compatibility.
        avg_cycles = sim_metrics["avg_cycles"]
        r_speedup  = real_speedup(config, avg_cycles)
        proxies["real_speedup"] = round(r_speedup, 3)
        proxies["latency_ns"]   = round(avg_cycles * proxies["effective_clock_ns"], 1)

        # --- reward (wall-clock elapsed intentionally excluded) ---
        rew = compute_reward(sim_metrics, proxies, self._reward_cfg)

        # --- build record ---
        record: dict = {
            "trial":      self._trial,
            "timestamp":  time.time(),
            "config":     config,
            "sim_metrics":   sim_metrics,
            "proxy_metrics": proxies,
            "reward":     rew,
            "elapsed_s":  elapsed,
            # Flat fields kept for dashboard backward compatibility
            "mac_lanes":    mac_lanes,
            "avg_cycles":   sim_metrics["avg_cycles"],
            "total_cycles": sim_metrics.get("total_cycles"),
            "accuracy":     sim_metrics["accuracy"],
            "speedup":      sim_metrics["speedup"],
        }

        self._history.append(record)
        self._log(record)
        self._trial += 1

        return self._make_state(), rew, False, record

    # ── State representation ──────────────────────────────────────────────────

    def _make_state(self) -> dict:
        """
        Compact state dict visible to agents.
        Agents may use this to bias their search (e.g., best config so far).
        """
        if not self._history:
            return {
                "trial":       0,
                "best_reward": float("-inf"),
                "best_config": self.default_config(),
                "mean_speedup": 0.0,
                "mean_area":   1.0,
                "history_len": 0,
            }

        best = max(self._history, key=lambda r: r["reward"])
        speedups = [r["speedup"] for r in self._history]
        areas    = [r["proxy_metrics"]["area_proxy"] for r in self._history]
        return {
            "trial":        self._trial,
            "best_reward":  best["reward"],
            "best_config":  best["config"],
            "mean_speedup": sum(speedups) / len(speedups),
            "mean_area":    sum(areas)    / len(areas),
            "history_len":  len(self._history),
        }

    # ── History & logging ─────────────────────────────────────────────────────

    @property
    def history(self) -> list[dict]:
        return list(self._history)

    def _log(self, record: dict) -> None:
        RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(RESULTS_FILE, "a") as f:
            f.write(json.dumps(record) + "\n")

    def clear_results(self) -> None:
        """Delete results.jsonl and reset history (used at start of fresh run)."""
        RESULTS_FILE.unlink(missing_ok=True)
        self._history.clear()
        self._trial = 0

    def load_existing_results(self) -> int:
        """
        Load previously written results from results.jsonl into history.
        Call before reset() to resume an interrupted run.
        Returns the number of records loaded.
        """
        if not RESULTS_FILE.exists():
            return 0
        loaded = 0
        with open(RESULTS_FILE) as f:
            for line in f:
                s = line.strip()
                if not s:
                    continue
                try:
                    rec = json.loads(s)
                    self._history.append(rec)
                    loaded += 1
                except json.JSONDecodeError:
                    pass
        self._trial = len(self._history)
        return loaded
