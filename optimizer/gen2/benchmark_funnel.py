"""benchmark_funnel.py — table-simulator benchmark for funnel promotion policies.

Protocol:
    Load the results_funnel.jsonl table via funnel.load_table (or a synthetic
    table if the real one is absent/requested).  For each seed, run a search
    campaign under a simulated wall-clock budget B:
      - Shuffle the grid configs (seeded)
      - For each candidate config, run one FunnelEnv episode (agent acts at
        each funnel depth until done=True or kill/commit)
      - Track the best F3-level reward found vs simulated time spent
    Metric: wall-clock-simulated time to reach 95% of the table optimum.
    Report: per-agent median + p95 across seeds + final-best distribution.

Agents benchmarked:
    random    — RandomPromotionAgent (uniform over {kill, re-proxy, promote, commit})
    fixed     — FixedGateAgent (cascade.py hard-coded gates)
    linucb    — PromotionAgent (LinUCB contextual bandit; online updates during campaign)
    ppo       — (optional) stable-baselines3 PPO behind try/import guard

Honesty rules (from Phase 4 doc):
    - Table's F3 rewards define the optimum.
    - Configs with no F3 row in the table cannot commit; "commit" at F3 returns
      a failure-ladder reward of -20 (full-flow-fail level).  This is documented
      and logged.
    - The benchmark asserts linucb >= random is NOT required (honesty per Stage 5
      precedent).  We just run it and report.

Usage:
    python3 optimizer/benchmark_funnel.py [options]

Options:
    --seeds N          number of seeds                    (default: 20)
    --budget B         simulated budget in seconds        (default: 14400 = 4h)
    --out PATH         results jsonl                      (default: stdout only)
    --pretrain N       warm LinUCB for N throwaway campaigns before measuring
    --agents LIST      comma-sep subset e.g. random,fixed,linucb (default: all)
    --selftest         run synthetic-table selftest and exit
    --table PATH       explicit results_funnel.jsonl      (default: auto-find)
    --target-pct P     fraction of optimum for "found" metric (default: 0.95)

Set PHYSICAL_MOCK=1 if running without ORFS (FunnelEnv will use mock metrics).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import statistics
import sys
import time
from collections import defaultdict
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np

# ── CandidateGenerator (optional — absent before candidates.py lands) ─────────
_CAND_AVAILABLE = False
try:
    from gen2.candidates import CandidateGenerator, _fallback_space
    _CAND_AVAILABLE = True
except Exception:
    CandidateGenerator = None   # type: ignore[assignment,misc]
    _fallback_space = None      # type: ignore[assignment]

# ── path setup ────────────────────────────────────────────────────────────────
_THIS_DIR = Path(__file__).resolve().parent
_OPT_DIR  = Path(__file__).resolve().parents[1]
if str(_OPT_DIR) not in sys.path:
    sys.path.insert(0, str(_OPT_DIR))

# Force UTF-8 output
try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
except Exception:
    pass

# ── imports: FunnelEnv (primary), fallback to synthetic table mode ─────────────

_FUNNEL_AVAILABLE = False
_FUNNEL_IMPORT_ERROR: str = ""

try:
    from gen2.funnel import FunnelEnv, load_table  # type: ignore[import]
    _FUNNEL_AVAILABLE = True
except Exception as _e:
    _FUNNEL_IMPORT_ERROR = str(_e)

# ── promotion agent imports ────────────────────────────────────────────────────

from gen2.promotion_agent import (  # noqa: E402
    FixedGateAgent,
    PromotionAgent,
    RandomPromotionAgent,
    STATE_DIM,
    IDX_DEPTH_F0, IDX_DEPTH_F1, IDX_DEPTH_F2, IDX_DEPTH_F3,
    IDX_F0_ACC, IDX_F1_ACC, IDX_F2_WNS, IDX_SURR_MU, IDX_SURR_SIG,
    IDX_INCUMBENT, IDX_BUDGET_FRAC, IDX_F0_CYCLES, IDX_F1_CYCLES,
    IDX_F2_AREA, IDX_F2_FF, IDX_F2_CELLS, IDX_F2_LEVELS,
)

# ── constants ─────────────────────────────────────────────────────────────────

ACTIONS = ("kill", "re-proxy", "promote", "commit")

# Fidelity cost table (seconds simulated) — from FunnelEnv spec / Phase 4
FIDELITY_COST_S: dict[str, float] = {
    "F0": 0.0,
    "F1": 5.0,
    "F2": 45.0,
    "F3": 420.0,
    "F4": 420.0,
}

# Reward for "commit" when F3 row is absent from the table
MISSING_F3_COMMIT_REWARD: float = -20.0

# Depth labels in funnel order
DEPTH_ORDER = ["F0", "F1", "F2", "F3"]

# ── synthetic table generation ────────────────────────────────────────────────

def _make_synthetic_table(n: int = 200, seed: int = 42) -> dict:
    """Generate a synthetic table (~200 configs) for selftest.

    Structure: reward is correlated with proxy obs so there is *some* structure
    for an agent to exploit.  F3 rewards are present for all configs.  F2 proxy
    obs are a noisy version of the F3 reward.  F0/F1 obs are analytic.

    Returns a dict in the SAME format as funnel.load_table():
        { config_json_key: {"F0": row_dict, "F1": row_dict, "F2": row_dict, "F3": row_dict} }
    where config_json_key = json.dumps(sorted canonical config keys) and each
    row_dict has the full {"config", "fidelity", "obs", "cost_s", "platform", "status"}
    schema expected by FunnelEnv table mode.

    The F3 obs dict includes a "reward" key (pre-computed) so run_benchmark can
    find the table optimum without re-running the reward function.
    """
    import json as _json

    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)

    lanes_choices  = [1, 2, 4, 8, 16, 32]
    acc_w_choices  = [16, 24, 32]
    clk_choices    = [3.0, 4.0, 5.0, 6.0, 8.0]
    recipe_choices = ["orfs_speed", "orfs_area", "plain"]

    # True reward function: maximised near lanes=4, acc_w=24, clk=4-5, recipe=plain
    def _true_reward(lanes: int, acc_w: int, clk: float, recipe: str) -> float:
        # Component 1: speedup (higher lanes = faster, but area penalty)
        speedup_norm = math.log2(max(lanes, 1)) / math.log2(32)
        area_pen = lanes / 32.0
        # Component 2: clock pressure (clk near 4 ns is sweet spot)
        clk_score = 1.0 - abs(clk - 4.5) / 5.0
        # Component 3: acc_w (24+ is fine, 16 kills accuracy)
        acc_score = 1.0 if acc_w >= 24 else -2.0
        # Component 4: recipe
        recipe_score = {"plain": 0.1, "orfs_area": -0.1, "orfs_speed": 0.0}.get(recipe, 0.0)
        r = 2.0 * acc_score + 1.5 * speedup_norm - 0.8 * area_pen + clk_score + recipe_score
        return float(r)

    def _canonical_key(lanes: int, acc_w: int, clk: float, recipe: str) -> str:
        """Produce a config key matching funnel._config_key (canonical FunnelEnv names)."""
        cfg = {"mac_lanes": lanes, "accumulator_width": acc_w,
               "clock_period_ns": clk, "abc_recipe": recipe}
        return _json.dumps({k: cfg[k] for k in sorted(cfg)},
                           sort_keys=True, separators=(",", ":"))

    # Sample configs; allow duplicates to be removed naturally
    all_configs: list[tuple] = []
    # First: ensure representation from grid corners
    for lanes in [1, 4, 32]:
        for acc_w in [16, 24, 32]:
            for clk in [3.0, 5.0, 8.0]:
                for recipe in ["plain", "orfs_area"]:
                    all_configs.append((lanes, acc_w, clk, recipe))

    # Fill up to n with random samples
    while len(all_configs) < n:
        lanes  = rng.choice(lanes_choices)
        acc_w  = rng.choice(acc_w_choices)
        clk    = rng.choice(clk_choices)
        recipe = rng.choice(recipe_choices)
        all_configs.append((lanes, acc_w, clk, recipe))

    # Dedup
    all_configs = list(dict.fromkeys(all_configs))[:n]

    from common.constants import AVG_CYCLES, SW_BASELINE_CYCLES, behavioral_cycles
    table: dict[str, dict[str, dict]] = {}
    for (lanes, acc_w, clk, recipe) in all_configs:
        true_r = _true_reward(lanes, acc_w, clk, recipe)
        noise  = float(np_rng.normal(0, 0.1))

        cyc = float(AVG_CYCLES.get(lanes, behavioral_cycles(lanes)))

        # Canonical config dict (FunnelEnv key names)
        cfg = {"mac_lanes": int(lanes), "accumulator_width": int(acc_w),
               "clock_period_ns": float(clk), "abc_recipe": str(recipe)}

        # F3 obs: final reward + PPA metrics
        area_f3 = round(15000 + lanes * 500 + noise * 100, 1)
        fmax = round(260 + noise * 10, 1)
        f3_obs = {
            "reward":        round(true_r, 4),
            "area_um2":      area_f3,
            "fmax_mhz":      fmax,
            "timing_met":    clk >= 3.72,
            "power_mw":      round(900 + lanes * 30 + noise * 50, 1),
            "period_min_ns": round(1000.0 / max(fmax, 1.0), 3),
            "wns_ns":        round(clk - 3.72 + noise * 0.05, 3),
            "lanes":         int(lanes), "acc_w": int(acc_w),
            "clk_ns":        float(clk), "platform": "nangate45",
            "status":        "ok",
        }

        # F2 obs: proxy (noisy version of truth, stored with cell_count key for
        # FunnelEnv._run_f2 compatibility)
        wns_proxy = round((clk - 3.72) + noise * 0.3, 3)
        f2_obs = {
            "area_um2":    round(area_f3 * 0.75, 1),
            "wns_ns":      wns_proxy,
            "timing_met":  clk >= 3.5,
            "cell_count":  int(200 + lanes * 50),
            "ff_count":    int(50 + lanes * 15),
            "logic_levels": int(10 + lanes),
        }

        # F1 obs: behavioral sim
        f1_obs = {
            "avg_cycles": cyc,
            "accuracy":   1.0 if acc_w >= 24 else 47.0 / 64.0,
            "correct":    64 if acc_w >= 24 else 47,
            "n_total":    64,
        }

        # F0 obs: analytic
        f0_obs = {
            "cycles":         cyc,
            "accuracy":       1.0 if acc_w >= 24 else 47.0 / 64.0,
            "cycle_speedup":  SW_BASELINE_CYCLES / max(cyc, 1),
        }

        # Store using the canonical load_table format (same as what FunnelEnv._log_row writes)
        key = _canonical_key(lanes, acc_w, clk, recipe)
        table[key] = {
            "F0": {"config": cfg, "fidelity": "F0", "obs": f0_obs,
                   "cost_s": 0.0, "platform": "nangate45", "status": "ok"},
            "F1": {"config": cfg, "fidelity": "F1", "obs": f1_obs,
                   "cost_s": 5.0, "platform": "nangate45", "status": "ok"},
            "F2": {"config": cfg, "fidelity": "F2", "obs": f2_obs,
                   "cost_s": 45.0, "platform": "nangate45", "status": "ok"},
            "F3": {"config": cfg, "fidelity": "F3", "obs": f3_obs,
                   "cost_s": 420.0, "platform": "nangate45", "status": "ok"},
        }

    return table


# ── state builder from table row ──────────────────────────────────────────────

def _build_state(
    config: dict,
    depth: str,
    table_entry: dict | None,
    incumbent_reward: float | None,
    budget_remaining: float,
    budget_total: float,
) -> np.ndarray:
    """Construct the 22-dim state vector for a config at the given funnel depth.

    Uses table observations up to `depth` to fill in the relevant slots.
    Slots for un-run depths are filled with -1 (sentinel).
    """
    s = np.full(STATE_DIM, -1.0)

    lanes  = int(config.get("lanes",  4))
    acc_w  = int(config.get("acc_w",  24))
    clk    = float(config.get("clk",  5.0))
    recipe = str(config.get("recipe", "plain"))

    # [0–4] config encoding
    s[0] = lanes / 32.0
    s[1] = acc_w / 32.0
    s[2] = (clk - 3.0) / 5.0
    s[3] = 1.0 if recipe == "orfs_speed" else 0.0
    s[4] = 1.0 if recipe == "orfs_area"  else 0.0

    entry = table_entry or {}

    # [5–6] F0 obs (always available after F0 is run)
    depth_idx = DEPTH_ORDER.index(depth) if depth in DEPTH_ORDER else 0
    if depth_idx >= 0:
        f0 = entry.get("F0", {})
        s[IDX_F0_CYCLES] = f0.get("cycle_speedup", -1.0) / 600.0  # norm to ~[0,1]
        s[IDX_F0_ACC]    = float(f0.get("accuracy_flag", -1.0))

    # [7–8] F1 obs
    if depth_idx >= 1:
        f1 = entry.get("F1", {})
        max_cyc = 300_000.0
        cyc = f1.get("avg_cycles", -1.0)
        s[IDX_F1_CYCLES] = cyc / max_cyc if cyc > 0 else -1.0
        s[IDX_F1_ACC]    = float(f1.get("accuracy", -1.0))

    # [9–13] F2 obs
    if depth_idx >= 2:
        f2 = entry.get("F2", {})
        area = f2.get("area_um2", None)
        wns  = f2.get("wns_ns",   None)
        ff   = f2.get("ff_count",  f2.get("cells",  None))
        cells = f2.get("cells",   None)
        levels = f2.get("logic_levels", None)

        s[IDX_F2_AREA]   = (area / 50000.0) if area is not None else -1.0
        s[IDX_F2_WNS]    = float(np.clip(wns, -5.0, 5.0)) if wns is not None else -1.0
        s[IDX_F2_FF]     = (ff / 500.0) if ff is not None else -1.0
        s[IDX_F2_CELLS]  = (cells / 10000.0) if cells is not None else -1.0
        s[IDX_F2_LEVELS] = (levels / 30.0) if levels is not None else -1.0

    # [14–15] surrogate mu/sigma (not available without a fitted surrogate)
    s[IDX_SURR_MU]  = 0.0
    s[IDX_SURR_SIG] = 1.0

    # [16] incumbent
    s[IDX_INCUMBENT] = (incumbent_reward / 4.5) if incumbent_reward is not None else 0.0
    s[IDX_INCUMBENT] = float(np.clip(s[IDX_INCUMBENT], -2.0, 2.0))

    # [17] budget fraction remaining
    s[IDX_BUDGET_FRAC] = float(np.clip(budget_remaining / max(budget_total, 1.0), 0.0, 1.0))

    # [18–21] depth one-hot
    for i, d in enumerate(DEPTH_ORDER):
        s[IDX_DEPTH_F0 + i] = 1.0 if depth == d else 0.0

    return s.astype(np.float32)


# ── table-simulator episode ───────────────────────────────────────────────────

class TableSimEpisode:
    """Simulate one config's funnel episode against the pre-built table.

    Manages the depth progression and cost accounting.  The agent acts at each
    depth until it says "kill" (→ penalty reward), "commit" (→ table F3 reward
    or MISSING_F3_COMMIT_REWARD), or "promote" (→ advance to next depth).
    "re-proxy" at F2 depth re-runs F2 (costs another FIDELITY_COST_S["F2"]);
    elsewhere it is treated as "promote".

    Returns: (episode_reward, cost_s, depth_reached, info)
    """

    def __init__(
        self,
        config: dict,
        table: dict,       # from load_table / _make_synthetic_table
        budget_s: float,
        budget_total: float,
        incumbent_reward: float | None,
    ) -> None:
        self.config = config
        self.key = (
            int(config["lanes"]),
            int(config["acc_w"]),
            float(config["clk"]),
            str(config["recipe"]),
        )
        self.entry = table.get(self.key)
        self.budget_s = budget_s
        self.budget_total = budget_total
        self.incumbent_reward = incumbent_reward

        self.depth_idx = 0     # start at F0
        self.cost_s = 0.0
        self.done = False

    @property
    def depth(self) -> str:
        return DEPTH_ORDER[min(self.depth_idx, len(DEPTH_ORDER) - 1)]

    def _state(self) -> np.ndarray:
        return _build_state(
            self.config,
            self.depth,
            self.entry,
            self.incumbent_reward,
            self.budget_s - self.cost_s,
            self.budget_total,
        )

    def run(self, agent: Any) -> tuple[float, float, str, dict]:
        """Run to completion.  Returns (reward, cost_s, final_depth, info)."""
        # Accrue F0 cost (free)
        self.cost_s += FIDELITY_COST_S["F0"]

        while not self.done:
            if self.cost_s >= self.budget_s:
                # Budget exhausted mid-episode
                return (-20.0, self.cost_s, self.depth,
                        {"reason": "budget_exhausted"})

            s = self._state()
            action = agent.act(s)

            if action == "kill":
                # Penalty based on depth
                penalty_map = {"F0": -60.0, "F1": -40.0, "F2": -40.0, "F3": -20.0}
                reward = penalty_map.get(self.depth, -40.0)
                agent.update(s, action, reward)
                return (reward, self.cost_s, self.depth,
                        {"action": "kill", "depth": self.depth})

            elif action == "commit":
                # Can only commit meaningfully at F3 (when we have real data)
                if self.depth == "F3" or self.depth_idx >= 3:
                    if self.entry and "F3" in self.entry:
                        reward = float(self.entry["F3"].get("reward", MISSING_F3_COMMIT_REWARD))
                    else:
                        # F3 row absent from table: documented failure-ladder reward
                        reward = MISSING_F3_COMMIT_REWARD
                    agent.update(s, action, reward)
                    return (reward, self.cost_s, "F3",
                            {"action": "commit", "has_f3": self.entry and "F3" in self.entry})
                else:
                    # Premature commit: treated as promote (can't commit without data)
                    action = "promote"

            if action == "re-proxy":
                if self.depth == "F2":
                    # Re-run F2 at same depth (costs another cycle)
                    self.cost_s += FIDELITY_COST_S["F2"]
                    agent.update(s, action, 0.0)   # shaping reward = 0 for re-proxy
                    continue
                else:
                    # Re-proxy at other depths = promote
                    action = "promote"

            if action == "promote":
                if self.depth_idx < len(DEPTH_ORDER) - 1:
                    next_depth = DEPTH_ORDER[self.depth_idx + 1]
                    self.cost_s += FIDELITY_COST_S[next_depth]
                    agent.update(s, action, 0.0)   # shaping reward on promotion = 0
                    self.depth_idx += 1
                else:
                    # Already at F3: auto-commit
                    if self.entry and "F3" in self.entry:
                        reward = float(self.entry["F3"].get("reward", MISSING_F3_COMMIT_REWARD))
                    else:
                        reward = MISSING_F3_COMMIT_REWARD
                    agent.update(s, action, reward)
                    return (reward, self.cost_s, "F3",
                            {"action": "auto_commit_at_max_depth"})

        return (0.0, self.cost_s, self.depth, {"reason": "unexpected_done"})


# ── campaign simulation ────────────────────────────────────────────────────────

def _run_campaign(
    agent: Any,
    table: dict,
    grid_configs: list[dict],
    budget_s: float,
    seed: int,
    target_pct: float = 0.95,
    table_optimum: float = 4.5,
    funnel_env: Any = None,
    candidates: str = "shuffled",
) -> dict:
    """Run one search campaign (one seed) using FunnelEnv table mode and return metrics.

    This function drives the real FunnelEnv(table=...) so the agent sees the same
    22-dim state vector and same shaped rewards as the live deployment.

    Parameters
    ----------
    agent        : promotion agent (act/update interface)
    table        : pre-loaded table from load_table() or _make_synthetic_table()
                   Must be in funnel.load_table format: {config_key → {fidelity → row_dict}}
    grid_configs : list of config dicts to iterate over (canonical key names)
    budget_s     : simulated wall-clock budget (seconds)
    seed         : random seed for shuffling
    target_pct   : fraction of table optimum considered "found"
    table_optimum: the table's best F3 reward (used to define target)
    funnel_env   : optional pre-constructed FunnelEnv(table=...) to reuse across
                   campaigns (budget tracking is per-campaign; env.spent_s reset
                   indirectly by passing budget_s as the remaining budget)

    Returns:
        best_reward       : float (best F3-level reward found)
        time_to_target_s  : float (simulated wall-clock to reach target_pct of optimum,
                                   or budget_s if never reached)
        best_curve        : list[(simulated_time, best_reward)]  — for plotting
        n_killed          : int
        n_committed       : int
        n_budget_exhausted: int
    """
    if not _FUNNEL_AVAILABLE:
        raise RuntimeError(
            "FunnelEnv is required for _run_campaign but funnel.py is not available: "
            f"{_FUNNEL_IMPORT_ERROR}"
        )

    import tempfile
    rng = random.Random(seed)
    configs = list(grid_configs)
    rng.shuffle(configs)

    # ── CandidateGenerator (tpe / surrogate_ucb) wiring ──────────────────────
    # When candidates != "shuffled" and CandidateGenerator is available, we use
    # it as the config-ordering oracle instead of the pre-shuffled list.
    # The generator is seeded per-campaign (matches the seed argument) for
    # reproducibility.  The shuffled list still acts as the universe of valid
    # configs (configs not in the table are silently skipped by env.reset).
    _cand_gen: Any = None
    if candidates != "shuffled" and _CAND_AVAILABLE:
        # Build fallback space for the generator (matches the tinymac_accel space)
        _space = _fallback_space()
        _cand_gen = CandidateGenerator(
            space=_space,
            sampler=candidates,       # "tpe" or "surrogate_ucb"
            surrogate=None,           # no surrogate in benchmark (table-mode only)
            seed=seed,
            kappa=1.0,
            grid_snap=True,           # snap to table grid so lookups hit
        )

    # Create a per-campaign FunnelEnv in table mode with a fresh budget tracker.
    # We use a temporary results path so each campaign's JSONL doesn't accumulate.
    with tempfile.TemporaryDirectory() as tmpdir:
        env = FunnelEnv(
            table=table,
            budget_s=budget_s,
            results_path=Path(tmpdir) / f"campaign_{seed}.jsonl",
            seed=seed,
        )
        # Seed the incumbent from caller-provided best (for state slot [16])
        # We can't set env._incumbent directly so we pass None; the env starts fresh
        # per campaign which matches the protocol (each seed is an independent search).

        simulated_s = 0.0
        best_reward = float("-inf")
        target_reward = table_optimum * target_pct
        time_to_target_s = budget_s   # default: never found
        best_curve: list[tuple[float, float]] = [(0.0, float("-inf"))]

        n_killed = n_committed = n_budget_exhausted = n_configs = 0

        # Config iteration: use CandidateGenerator when available, else shuffled list.
        # For CandidateGenerator mode we iterate until budget is exhausted or the
        # generator has proposed all configs in the shuffled universe.
        def _config_iter():
            if _cand_gen is not None:
                # Propose configs from the generator up to len(grid_configs) total;
                # each call may repeat a config (grid small; generator has seen it
                # all after ~594 suggests).  We cap at 3× the grid size to prevent
                # infinite loops when the generator cycles.
                _seen: set[str] = set()
                _max_proposals = len(configs) * 3
                for _ in range(_max_proposals):
                    try:
                        cfg = _cand_gen.suggest()
                    except Exception:
                        break
                    # Snap to table-present configs: only yield if in the table
                    from gen2.funnel import _config_key as _ck
                    key = _ck(cfg)
                    if key in table and key not in _seen:
                        _seen.add(key)
                        yield cfg
                    # If not in table, still yield so env.reset can handle it
                    elif key not in _seen:
                        _seen.add(key)
                        yield cfg
            else:
                yield from configs

        for config in _config_iter():
            if simulated_s >= budget_s:
                break

            try:
                state = env.reset(config)
            except (ValueError, KeyError):
                # Config not in table or invalid; skip
                if _cand_gen is not None:
                    _cand_gen.update(config, reward=-100.0, fidelity="invalid")
                continue

            episode_done = False
            episode_reward = 0.0

            while not episode_done:
                if simulated_s + env._episode_spent_s >= budget_s:
                    n_budget_exhausted += 1
                    break

                action = agent.act(state)
                next_state, reward, episode_done, info = env.step(action)

                # For kill/commit/terminal: accumulate reward and update agent
                agent.update(state, action, reward)
                state = next_state
                episode_reward += reward

                if episode_done:
                    fid = info.get("fidelity", "?")
                    act = info.get("action", action)
                    if act == "kill":
                        n_killed += 1
                    elif fid == "F3":
                        n_committed += 1

            n_configs += 1
            # Update simulated time from env's episode cost
            episode_cost = env._episode_spent_s
            simulated_s += episode_cost

            # Only count terminal F3 rewards toward the incumbent.
            # episode_reward is the FunnelEnv terminal reward (including shaping);
            # for the benchmark metric we use it directly since this is the same
            # reward that agents are optimizing.
            fid_reached = info.get("fidelity", "?") if episode_done else "?"
            if fid_reached == "F3" and episode_done:
                final_reward = float(episode_reward)

                # Feed back to CandidateGenerator (F3-only tell rule)
                if _cand_gen is not None:
                    _cand_gen.update(config, final_reward, fidelity="F3")

                if final_reward > best_reward:
                    best_reward = final_reward
                    best_curve.append((simulated_s, best_reward))
                    if best_reward >= target_reward and time_to_target_s == budget_s:
                        time_to_target_s = simulated_s
            else:
                # Non-F3 result: feed as kill to CandidateGenerator
                if _cand_gen is not None:
                    _cand_gen.update(config, episode_reward, fidelity=fid_reached or "F0")

        return {
            "best_reward":        best_reward,
            "time_to_target_s":   time_to_target_s,
            "best_curve":         best_curve,
            "n_configs_tried":    n_configs,
            "n_killed":           n_killed,
            "n_committed":        n_committed,
            "n_budget_exhausted": n_budget_exhausted,
            "final_simulated_s":  simulated_s,
        }


# ── agent factory ─────────────────────────────────────────────────────────────

def _make_agent(name: str, seed: int) -> Any | None:
    """Construct and return an agent by name.  Returns None if unavailable."""
    if name == "random":
        return RandomPromotionAgent(seed=seed, actions=ACTIONS)
    elif name == "fixed":
        # _run_campaign uses FunnelEnv(table=...) which stores normalised WNS
        # (wns_ns / 5.0) in state[IDX_F2_WNS].  Pass the normalised threshold
        # -0.5 (= -2.5 ns / 5) so FixedGateAgent kills configs with WNS < -2.5 ns.
        return FixedGateAgent(seed=seed, actions=ACTIONS,
                               proxy_wns_kill_threshold=-0.5)
    elif name == "linucb":
        return PromotionAgent(dim=STATE_DIM, alpha=1.0, seed=seed, actions=ACTIONS)
    elif name == "ppo":
        try:
            from stable_baselines3 import PPO  # type: ignore[import]
            import gym  # type: ignore[import]
            # PPO hook: wrap in a thin adapter that satisfies our act/update interface
            # (not trained here — serves as a wiring test stub)
            class _PPOAdapter:
                def __init__(self) -> None:
                    self._rng = random.Random(seed)
                    self.actions = ACTIONS
                    # A real implementation would train PPO on table trajectories here
                    self._model = None

                def act(self, state: np.ndarray) -> str:
                    # Stub: random until trained
                    return self._rng.choice(self.actions)

                def update(self, state: np.ndarray, action: str, reward: float) -> None:
                    pass  # PPO trains episodically, not per-step online

            return _PPOAdapter()
        except ImportError:
            return None
    else:
        raise ValueError(f"unknown agent: {name!r}")


# ── benchmark runner ──────────────────────────────────────────────────────────

def run_benchmark(
    table: dict,
    agent_names: list[str],
    n_seeds: int,
    budget_s: float,
    target_pct: float,
    pretrain_campaigns: int,
    verbose: bool = True,
    candidates: str = "shuffled",
) -> dict[str, dict]:
    """Run the full benchmark.  Returns per-agent metric dict."""

    # Build the grid config list from the table.
    # Table keys are JSON strings of canonical config dicts (from load_table /
    # _make_synthetic_table).  Extract the actual config dict from each row.
    grid_configs: list[dict] = []
    for key, entry in table.items():
        # Get the config dict from any available fidelity row
        cfg = None
        for fid in ("F0", "F1", "F2", "F3"):
            if fid in entry:
                cfg = entry[fid].get("config")
                if cfg is not None:
                    break
        if cfg is None:
            # Fallback: try to parse the JSON key directly
            try:
                import json as _jj
                cfg = _jj.loads(key)
            except Exception:
                continue
        grid_configs.append(cfg)

    # Find table optimum: maximum F3 reward as computed by FunnelEnv.
    # We do a quick "commit" scan over all F3-populated configs using a throw-away
    # FunnelEnv (infinite budget, temp log) so the optimum uses the exact same
    # reward formula that the benchmark campaign will see.
    import tempfile as _tmpfile
    f3_rewards: list[float] = []

    with _tmpfile.TemporaryDirectory() as _tmpdir:
        _scan_env = FunnelEnv(
            table=table,
            budget_s=float("inf"),
            results_path=Path(_tmpdir) / "scan.jsonl",
        )
        for _cfg in grid_configs:
            try:
                _scan_env.reset(_cfg)
                _, _r, _, _ = _scan_env.step("commit")
                f3_rewards.append(float(_r))
            except Exception:
                pass

    if not f3_rewards:
        raise ValueError("Table has no F3 rows — cannot define the optimum.")
    table_optimum = max(f3_rewards)
    target_reward = table_optimum * target_pct

    if verbose:
        print(f"Table: {len(grid_configs)} configs, "
              f"optimum F3 reward = {table_optimum:.4f}, "
              f"target (={target_pct*100:.0f}%) = {target_reward:.4f}")
        f3_count = sum(1 for e in table.values() if "F3" in e)
        print(f"       {f3_count}/{len(table)} configs have F3 rows; "
              f"{len(table)-f3_count} table-miss commits return "
              f"{MISSING_F3_COMMIT_REWARD:.0f} via FunnelEnv (documented).")
        print()

    results: dict[str, dict] = {}
    skipped_agents: list[str] = []

    for name in agent_names:
        agent_check = _make_agent(name, seed=0)
        if agent_check is None:
            if verbose:
                print(f"NOTE: agent '{name}' unavailable (package not installed). Skipping.")
            skipped_agents.append(name)
            continue

        per_seed_ttt: list[float] = []      # time-to-target per seed
        per_seed_best: list[float] = []     # final best reward per seed

        for seed in range(n_seeds):
            agent = _make_agent(name, seed=seed)
            assert agent is not None

            # Optional LinUCB pre-training on throwaway campaigns
            if name == "linucb" and pretrain_campaigns > 0:
                for p in range(pretrain_campaigns):
                    _run_campaign(agent, table, grid_configs, budget_s,
                                  seed=seed * 1000 + p, target_pct=target_pct,
                                  table_optimum=table_optimum,
                                  candidates=candidates)
                # NOTE: pre-training modifies the agent in-place (it learns from
                # the throwaway campaigns); this is intended — the bandit IS the
                # trained policy after warm-up.

            res = _run_campaign(
                agent, table, grid_configs, budget_s, seed=seed,
                target_pct=target_pct, table_optimum=table_optimum,
                candidates=candidates,
            )
            per_seed_ttt.append(res["time_to_target_s"])
            per_seed_best.append(res["best_reward"])

        per_seed_ttt_h = [t / 3600.0 for t in per_seed_ttt]
        # Replace -inf sentinels with a finite floor before computing stats.
        # -inf occurs when the agent killed every config and earned no F3 reward.
        _FLOOR = -200.0
        per_seed_best_finite = [
            x if math.isfinite(x) else _FLOOR for x in per_seed_best
        ]
        # pstdev requires at least 2 data points (degenerate with 1 seed)
        std_best = (statistics.pstdev(per_seed_best_finite)
                    if len(per_seed_best_finite) > 1 else 0.0)
        results[name] = {
            "name":              name,
            "n_seeds":           n_seeds,
            "time_to_target_s":  per_seed_ttt,
            "final_best_reward": per_seed_best_finite,
            "median_ttt_h":      statistics.median(per_seed_ttt_h),
            "p95_ttt_h":         _percentile(per_seed_ttt_h, 95),
            "mean_best_reward":  statistics.mean(per_seed_best_finite),
            "std_best_reward":   std_best,
            "success_rate":      sum(1 for t in per_seed_ttt if t < budget_s) / n_seeds,
        }

    if verbose:
        _print_results(results, budget_s, table_optimum, target_pct, skipped_agents)

    return results


def _percentile(data: list[float], pct: float) -> float:
    """Compute the pct-th percentile of data (linear interpolation)."""
    if not data:
        return float("nan")
    sorted_d = sorted(data)
    idx = (pct / 100.0) * (len(sorted_d) - 1)
    lo, hi = int(idx), min(int(idx) + 1, len(sorted_d) - 1)
    frac = idx - lo
    return sorted_d[lo] * (1 - frac) + sorted_d[hi] * frac


def _print_results(
    results: dict[str, dict],
    budget_s: float,
    table_optimum: float,
    target_pct: float,
    skipped: list[str],
) -> None:
    budget_h = budget_s / 3600.0
    tgt_lbl = f"{target_pct*100:.0f}%"
    width = 78

    print("=" * width)
    print("HONEST BENCHMARK — Funnel promotion policy agents")
    print("Metric: simulated wall-clock to reach "
          f"{tgt_lbl} of table optimum ({table_optimum:.4f})")
    print("=" * width)

    if not results:
        print("No agents ran.")
        return

    hdr = (f"{'agent':<12} {'med_ttt(h)':>11} {'p95_ttt(h)':>11} "
           f"{'mean_best':>10} {'std_best':>10} {'success%':>9}")
    print(hdr)
    print("-" * width)
    for name, r in results.items():
        success_pct = r["success_rate"] * 100.0
        # Cap median/p95 at budget for display
        med = min(r["median_ttt_h"], budget_h)
        p95 = min(r["p95_ttt_h"], budget_h)
        print(f"{name:<12} "
              f"{med:>11.3f} "
              f"{p95:>11.3f} "
              f"{r['mean_best_reward']:>10.4f} "
              f"{r['std_best_reward']:>10.4f} "
              f"{success_pct:>8.1f}%")
    print("-" * width)
    print(f"(ttt = simulated time-to-{tgt_lbl}-optimum; "
          f"budget = {budget_h:.1f}h; "
          f"censored at budget if not found)")
    print()

    # Verdict (honest — no assertion that linucb beats random)
    print("VERDICT (honest, per Phase 4 / CP4 protocol):")
    print("-" * width)
    rand_r = results.get("random")
    fixed_r = results.get("fixed")
    linucb_r = results.get("linucb")

    lines: list[str] = []
    if rand_r and fixed_r and linucb_r:
        # Compare by median time-to-target (lower = better)
        r_med = rand_r["median_ttt_h"]
        f_med = fixed_r["median_ttt_h"]
        l_med = linucb_r["median_ttt_h"]
        best_name = min([("random", r_med), ("fixed", f_med), ("linucb", l_med)],
                        key=lambda x: x[1])[0]

        if linucb_r["median_ttt_h"] < rand_r["median_ttt_h"] * 0.95:
            lines.append(
                f"LinUCB ({l_med:.3f}h) beat random ({r_med:.3f}h) on median "
                f"time-to-{tgt_lbl}-optimum. The bandit learned useful promotion gates. "
                f"CP4 PASS criterion met.")
        else:
            lines.append(
                f"LinUCB ({l_med:.3f}h) did NOT beat random ({r_med:.3f}h) by "
                f"a clear margin on this table. This is an honest result per Stage 5 "
                f"precedent: on a small, deterministic table the bandit may not have "
                f"enough structure to exploit. CP4 threshold: ship with fixed gates if "
                f"this result persists on the real live campaign.")

        if fixed_r["median_ttt_h"] < rand_r["median_ttt_h"] * 0.95:
            lines.append(
                f"FixedGateAgent ({f_med:.3f}h) also beat random, confirming the gate "
                f"thresholds from Phase 5 are useful.")
    else:
        lines.append(
            f"Not all three agents (random, fixed, linucb) ran; "
            f"cannot render a full CP4 comparison.")

    import textwrap
    for line in lines:
        print(textwrap.fill(line, width=width))

    if skipped:
        print()
        print(f"Skipped agents: {skipped} (package not installed).")


# ── selftest ──────────────────────────────────────────────────────────────────

def _selftest(verbose: bool = True) -> None:
    """Selftest with a synthetic ~200-config table.

    Asserts:
    - Benchmark runs without error for 3 seeds × 3 agents
    - All agents produce valid numeric metrics
    - The "linucb >= random" condition is NOT asserted (honesty per the doc)
    - Selftest PASSES means: the code runs, produces valid output, and the
      harness is wired correctly.
    """
    if verbose:
        print("=" * 60)
        print("SELFTEST: synthetic table, 3 seeds × 3 agents")
        print("=" * 60)

    table = _make_synthetic_table(n=200, seed=42)
    assert len(table) >= 50, f"synthetic table too small: {len(table)}"
    f3_count = sum(1 for e in table.values() if "F3" in e)
    assert f3_count >= 50, f"too few F3 rows: {f3_count}"

    if verbose:
        print(f"Synthetic table: {len(table)} configs, {f3_count} with F3 rows")

    # Run with 3 seeds, small budget (2000 s simulated) so at least a few F3
    # evaluations can complete.  At cost F0+F1+F2+F3 = 0+5+45+420 = 470 s each,
    # this budget allows ~4 full F3 evaluations.  The test is fast because
    # everything is table-lookup (no real tool calls).
    budget_s = 2000.0  # seconds simulated
    results = run_benchmark(
        table=table,
        agent_names=["random", "fixed", "linucb"],
        n_seeds=3,
        budget_s=budget_s,
        target_pct=0.95,
        pretrain_campaigns=0,
        verbose=verbose,
    )

    # Assertions: runs without error, produces valid metrics
    assert "random" in results
    assert "fixed"  in results
    assert "linucb" in results

    for name, r in results.items():
        assert isinstance(r["median_ttt_h"], float), \
            f"{name}: median_ttt_h is not float: {r['median_ttt_h']!r}"
        assert isinstance(r["p95_ttt_h"], float), \
            f"{name}: p95_ttt_h is not float: {r['p95_ttt_h']!r}"
        assert not math.isnan(r["median_ttt_h"]), \
            f"{name}: median_ttt_h is NaN"
        assert r["n_seeds"] == 3
        # Sanity: median time is in [0, budget_s/3600] range
        assert 0.0 <= r["median_ttt_h"] <= budget_s / 3600.0 + 1e-6, \
            f"{name}: median_ttt_h {r['median_ttt_h']:.4f} out of range"

    # HONESTY NOTE: we explicitly do NOT assert linucb >= random
    # (the benchmark is the measurement, not a pass/fail on agent quality).

    if verbose:
        print()
        print("SELFTEST: PASS")
        print("  (linucb vs random comparison is printed above — not asserted)")
    else:
        print("benchmark_funnel.py selftest: PASS")


# ── FunnelEnv integration mode ─────────────────────────────────────────────────

def _run_with_funnel_env(
    table_path: Path,
    agent_names: list[str],
    n_seeds: int,
    budget_s: float,
    target_pct: float,
    pretrain_campaigns: int,
    candidates: str = "shuffled",
) -> None:
    """Run the benchmark using the real FunnelEnv + load_table."""
    if not _FUNNEL_AVAILABLE:
        print(f"WARNING: funnel.py not available ({_FUNNEL_IMPORT_ERROR}).")
        print("Falling back to synthetic table for --selftest or re-run after funnel.py lands.")
        return

    try:
        table = load_table(str(table_path))
    except Exception as exc:
        print(f"ERROR loading table from {table_path}: {exc}")
        print("Run build_table.py to populate the table first.")
        sys.exit(1)

    if not table:
        print(f"ERROR: table at {table_path} is empty.  "
              "Run build_table.py first.")
        sys.exit(1)

    run_benchmark(
        table=table,
        agent_names=agent_names,
        n_seeds=n_seeds,
        budget_s=budget_s,
        target_pct=target_pct,
        pretrain_campaigns=pretrain_campaigns,
        verbose=True,
        candidates=candidates,
    )


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark funnel promotion policy agents on the pre-built table.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--seeds", type=int, default=20,
                        help="Number of random seeds")
    parser.add_argument("--budget", type=float, default=14400.0,
                        help="Simulated budget in seconds (default: 4h)")
    parser.add_argument("--out", default=None,
                        help="Write results to this JSONL file")
    parser.add_argument("--pretrain-campaigns", type=int, default=0,
                        dest="pretrain_campaigns",
                        help="Warm LinUCB for N throwaway campaigns before measuring")
    parser.add_argument("--agents", default="random,fixed,linucb",
                        help="Comma-separated agent names to benchmark")
    parser.add_argument("--selftest", action="store_true",
                        help="Run synthetic-table selftest and exit")
    parser.add_argument("--table",
                        default=str(_THIS_DIR.parent / "results_funnel.jsonl"),
                        help="Path to results_funnel.jsonl table")
    parser.add_argument("--target-pct", type=float, default=0.95,
                        dest="target_pct",
                        help="Fraction of optimum considered 'found'")
    parser.add_argument("--candidates", default="shuffled",
                        choices=["shuffled", "tpe", "surrogate_ucb"],
                        help=(
                            "Candidate ordering: shuffled = current seeded-shuffle "
                            "(default, keeps existing results comparable); "
                            "tpe = Optuna TPE-backed CandidateGenerator; "
                            "surrogate_ucb = surrogate UCB (falls back to tpe without surrogate). "
                            "Non-shuffled modes require gen2/candidates.py."
                        ))

    args = parser.parse_args()

    if args.selftest:
        _selftest(verbose=True)
        return

    agent_names = [a.strip() for a in args.agents.split(",") if a.strip()]
    table_path = Path(args.table)

    if not table_path.exists():
        print(f"Table file not found: {table_path}")
        print("Run: python3 optimizer/build_table.py --subset strategic")
        print("Or use --selftest to run the synthetic-table selftest.")
        sys.exit(1)

    _run_with_funnel_env(
        table_path=table_path,
        agent_names=agent_names,
        n_seeds=args.seeds,
        budget_s=args.budget,
        target_pct=args.target_pct,
        pretrain_campaigns=args.pretrain_campaigns,
        candidates=getattr(args, "candidates", "shuffled"),
    )


if __name__ == "__main__":
    main()
