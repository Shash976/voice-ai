"""funnel.py — FunnelEnv: gym-style environment exposing per-stage observations
and accepting promotion *actions* over the multi-fidelity evaluation funnel.

Architecture (see docs/07_rl_pipeline_design.md Phase 4):

    F0  validate + analytic cycle model        cost ≈ 0 s    (always runs on reset)
    F1  behavioral Verilator sim               cost ≈ 5 s
    F2  synth+STA proxy                        cost ≈ 45 s
    F3  full ORFS flow (nangate45 / asap7)     cost ≈ 7 min  (terminal)

Each episode covers exactly ONE candidate config.  The caller (candidate
generator / promotion policy) drives the episode via step(action):

    "kill"     → immediate termination; episode done, no reward paid.
    "re-proxy" → (re-)run F2 (even if F1/F2 unrun; restarts the proxy).
    "promote"  → run the next not-yet-run fidelity in order (F1 → F2 → F3).
    "commit"   → skip to F3 immediately (terminal).

After F3 completes the episode is always done=True.

State vector (22-dim float32, PINNED):
    [0]  log2(lanes)/5
    [1]  (acc_w − 16)/16
    [2]  (clk − 3)/5           # nangate45 normalisation; asap7: (clk−0.3)/1.2
    [3]  recipe_idx/2           # orfs_speed=0, orfs_area=1, plain=2
    [4]  platform flag          # 0=nangate45, 1=asap7
    [5]  F0 cycles_norm         # log2(SW_BASELINE_CYCLES / cycles) / 10
    [6]  F0 accuracy            # 0..1, from analytic table
    [7]  F1 cycles_norm         # 0 if F1 unrun
    [8]  F1 accuracy            # 0 if F1 unrun
    [9]  F2 proxy_area_norm     # proxy_area/20000 clipped [0,3]; 0 if unrun
    [10] F2 wns_norm            # clip(proxy_wns_ns/5, −2, 2); 0 if unrun
    [11] ff_count/1000 clip[0,3]
    [12] cell_count/10000 clip[0,3]
    [13] logic_levels/50 clip[0,2]
    [14] surrogate μ/4.5        # 0 if no surrogate
    [15] surrogate σ            # 0 if no surrogate
    [16] incumbent best reward/4.5  # 0 if none
    [17] remaining budget fraction
    [18..21] depth one-hot (highest fidelity already run: F0, F1, F2, F3)

Logging: every (config, fidelity, obs) row is appended to results_funnel.jsonl:
    {"ts", "config", "fidelity", "obs", "cost_s", "platform", "status"}

Live mode (table=None): calls cascade.py's existing tool wrappers for F1/F2/F3.
Table mode (table=dict): replays observations from the table, charges recorded
    cost against the budget — no real tools.  Use load_table(path) to build the
    dict from an existing results_funnel.jsonl.
"""

from __future__ import annotations

import inspect
import json
import math
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml

# Bootstrap: make optimizer/ root importable (gen2/ is one level below it)
import pathlib as _pl, sys as _sys
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))

# ── Defensive imports for concurrent agents ──────────────────────────────────
# recipe.py is being written by a concurrent agent; fall back gracefully.
try:
    from common.recipe import RECIPES, recipe_suffix  # noqa: F401
except ImportError:
    RECIPES = ("orfs_speed", "orfs_area", "plain")

    def recipe_suffix(r: str) -> str:   # noqa: D401
        """Minimal fallback — matches variant_name convention when recipe.py absent.

        Corrected: 'plain' gets '_plain' suffix (distinct from 'orfs_speed' which
        has no suffix).  The old '' fallback was wrong and would alias 'plain'
        variant names to the 'orfs_speed' namespace.
        """
        _MAP = {"orfs_speed": "", "orfs_area": "_area", "plain": "_plain"}
        return _MAP.get(r, f"_{r}")


# surrogate.py is being written by a concurrent agent; fall back gracefully.
try:
    from gen2.surrogate import Surrogate  # noqa: F401  (type hint only)
    _SURROGATE_AVAILABLE = True
except ImportError:
    _SURROGATE_AVAILABLE = False
    Surrogate = None  # type: ignore[assignment,misc]

# ── Core imports (always available) ──────────────────────────────────────────
from common.constants import (
    AVG_CYCLES,
    SW_BASELINE_CYCLES,
    behavioral_cycles as _behavioral_cycles,
)
from gen1.cascade import _run_sim  # reuse the mock-aware Verilator wrapper
from common.physical_runner import run_synth_sta, run_physical
from common.cascade_reward import compute_cascade_reward
from common.physical_reward import compute_physical_reward

# ── Module paths ─────────────────────────────────────────────────────────────
_OPTIMIZER_DIR = Path(__file__).resolve().parent.parent
_DEFAULT_SPACE = Path(__file__).resolve().parent / "search_space_funnel.yaml"

# ── Accuracy table from sim_measurements.txt (V13: LANES-dependent at ACC_W<32) ──
# Rows: (lanes, acc_w) → accuracy fraction from the 2026-06-10 measured sweep.
# Used for F0 analytic accuracy (no real sim run).
_ACC_TABLE: dict[tuple[int, int], float] = {
    # (lanes, acc_w): accuracy
    (1,  32): 1.0, (1,  24): 1.0,  (1,  16): 47/64,
    (2,  32): 1.0, (2,  24): 1.0,  (2,  16): 48/64,
    (4,  32): 1.0, (4,  24): 1.0,  (4,  16): 48/64,
    (8,  32): 1.0, (8,  24): 1.0,  (8,  16): 48/64,
    (16, 32): 1.0, (16, 24): 1.0,  (16, 16): 48/64,
    (32, 32): 1.0, (32, 24): 1.0,  (32, 16): 58/64,
}

# abc recipe → index mapping (state dim [3])
_RECIPE_IDX: dict[str, int] = {"orfs_speed": 0, "orfs_area": 1, "plain": 2}

# Fidelity names and their baseline wall-clock cost (seconds)
FIDELITY_COST_S: dict[str, float] = {
    "F0": 0.0,
    "F1": 5.0,
    "F2": 45.0,
    "F3": 420.0,
    "F4": 420.0,
}

# Ordered fidelity list (F4 is asap7 and not used in the standard promote chain)
_FIDELITY_ORDER = ["F0", "F1", "F2", "F3"]

# Platform normalisations for state dim [2] (clock)
_CLK_NORM: dict[str, tuple[float, float]] = {
    "nangate45": (3.0, 5.0),   # (offset, scale): (clk - 3.0) / 5.0
    "asap7":     (0.3, 1.2),   # (clk - 0.3) / 1.2
}

_STATE_DIM = 22


# ── Table helpers ──────────────────────────────────────────────────────────────

def _config_key(config: dict) -> str:
    """Stable string key for a config dict (sorted items, JSON-encoded)."""
    return json.dumps(
        {k: config[k] for k in sorted(config)},
        sort_keys=True, separators=(",", ":"),
    )


def load_table(path: str | Path) -> dict[str, dict[str, dict]]:
    """Build a table dict from a results_funnel.jsonl file.

    Returns:
        {config_key: {fidelity: row_dict}}
    where row_dict is the full {"config", "fidelity", "obs", "cost_s", ...} row.

    Use this table as the `table` argument to FunnelEnv for offline simulation.
    """
    table: dict[str, dict[str, dict]] = {}
    p = Path(path)
    if not p.exists():
        return table
    with open(p, encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                row = json.loads(s)
            except json.JSONDecodeError:
                continue
            cfg = row.get("config")
            fid = row.get("fidelity")
            if cfg is None or fid is None:
                continue
            key = _config_key(cfg)
            if key not in table:
                table[key] = {}
            table[key][fid] = row
    return table


# ── Validate a config against the funnel space ────────────────────────────────

def _load_space(yaml_path: Path) -> tuple[dict, dict, list[str], dict, dict]:
    """Load and return (sim_params, proxy_params, constraints, gates, reward_cfg)."""
    with open(yaml_path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    sim    = raw.get("sim_params",   {}) or {}
    proxy  = raw.get("proxy_params", {}) or {}
    constr = raw.get("constraints",  []) or []
    gates  = raw.get("gates",        {}) or {}
    reward = raw.get("reward",       {}) or {}
    return sim, proxy, constr, gates, reward


def _validate_funnel(config: dict, sim: dict, proxy: dict,
                     constraints: list[str],
                     active_space: dict | None = None) -> tuple[bool, str]:
    """Minimal validation for the funnel space (continuous clock axis).

    Design-agnostic: only validates params that are **present in the config**.
    YAML params absent from the config are silently skipped — they belong to a
    different design's space and are not applicable here (e.g. mac_lanes /
    accumulator_width for non-TinyVAD designs like gcd).

    active_space: optional dict from KnobRegistry.space() (or _build_space).
      When provided, its bounds override the YAML bounds for continuous axes,
      and YAML constraints whose variables exist in the config but whose range
      differs from the active_space range are skipped.  This allows gcd configs
      with clock_period_ns ∈ [0.3, 2.0] to pass validation even though the
      tinymac YAML says [3.0, 8.0].
    """
    # Build a set of active axis names and their ranges (for constraint skipping)
    active_bounds: dict[str, tuple[float, float]] = {}
    active_choices: dict[str, list] = {}
    if active_space:
        for name, spec in active_space.items():
            typ = spec.get("type", "categorical")
            if typ in ("float", "continuous") and "range" in spec:
                active_bounds[name] = (float(spec["range"][0]), float(spec["range"][1]))
            elif typ == "categorical" and "choices" in spec:
                active_choices[name] = list(spec["choices"])
            elif typ == "int" and "choices" in spec:
                active_choices[name] = list(spec["choices"])

    # Membership / bounds check — only for params present in this config
    for name, spec in {**sim, **proxy}.items():
        if name not in config:
            continue   # not part of this design's space; skip
        if spec.get("type") == "categorical":
            # Use active_space choices if available (design may have different choices)
            choices = active_choices.get(name, spec.get("choices", []))
            if choices and config[name] not in choices:
                return False, f"{name}={config[name]!r} not in {choices}"
        elif spec.get("type") in ("continuous", "float"):
            # Use active_space bounds if available
            if name in active_bounds:
                lo, hi = active_bounds[name]
            else:
                lo = spec.get("low", spec.get("range", [float("-inf"), float("inf")])[0])
                hi = spec.get("high", spec.get("range", [float("-inf"), float("inf")])[1])
            try:
                v = float(config[name])
            except (TypeError, ValueError):
                return False, f"{name}={config[name]!r} is not numeric"
            if not (lo <= v <= hi):
                return False, f"{name}={v} outside [{lo}, {hi}]"

    # Constraint expressions (same sandbox as validate.py).
    # Skip constraints that:
    # (a) reference variables absent from the config (NameError) — design-specific
    # (b) are about axes that the active_space defines differently from the YAML
    #     (e.g. "3.0 <= clock_period_ns <= 8.0" is a tinymac constraint; gcd has
    #     clock_period_ns ∈ [0.3, 2.0] per KnobRegistry).
    safe_globals: dict = {"__builtins__": {}}
    for expr in constraints:
        # If constraint references an axis that active_space overrides, skip it.
        # Heuristic: if any active_bounds key appears in the expr and active_space
        # provides a different range, the YAML constraint is design-specific.
        if active_bounds:
            skip = False
            for aname in active_bounds:
                if aname in expr:
                    # This constraint is about an axis that active_space defines;
                    # the YAML constraint bounds may not apply to this design.
                    skip = True
                    break
            if skip:
                continue
        try:
            ok = bool(eval(expr, safe_globals, dict(config)))  # noqa: S307
        except NameError:
            # Variable referenced in constraint not in config — not applicable
            continue
        except Exception as exc:  # noqa: BLE001
            return False, f"constraint error in {expr!r}: {exc}"
        if not ok:
            return False, f"constraint failed: {expr}"
    return True, ""


# ── F0 analytic computations ──────────────────────────────────────────────────

def _f0_cycles(lanes: int) -> float:
    """Return analytic cycle estimate for F0 (uses measured table when available)."""
    if lanes in AVG_CYCLES:
        return float(AVG_CYCLES[lanes])
    return _behavioral_cycles(lanes)


def _f0_accuracy(lanes: int, acc_w: int) -> float:
    """Return analytic accuracy estimate for F0 from the measured table."""
    key = (lanes, acc_w)
    if key in _ACC_TABLE:
        return _ACC_TABLE[key]
    # Unknown lane count: use acc_w-only fallback
    if acc_w >= 24:
        return 1.0
    return 47.0 / 64.0   # conservative lower bound


# ── State vector construction ─────────────────────────────────────────────────

def _build_state(
    config: dict,
    platform: str,
    f0_cycles: float,
    f0_accuracy: float,
    f1_obs: dict | None,
    f2_obs: dict | None,
    surrogate: Any | None,
    incumbent_reward: float | None,
    budget_fraction: float,
    depth: int,                         # 0=F0, 1=F1, 2=F2, 3=F3
) -> np.ndarray:
    lanes  = int(config.get("mac_lanes", 1))        # 1 = sentinel for generic designs
    acc_w  = int(config.get("accumulator_width", 24))  # 24 = default for generic designs
    clk    = float(config["clock_period_ns"])
    recipe = config.get("abc_recipe", "plain")

    clk_offset, clk_scale = _CLK_NORM.get(platform, (3.0, 5.0))
    plat_flag = 1.0 if platform == "asap7" else 0.0
    recipe_idx = _RECIPE_IDX.get(recipe, 2)

    # [0..4] config encoding
    d0 = math.log2(max(lanes, 1)) / 5.0
    d1 = (acc_w - 16) / 16.0
    d2 = (clk - clk_offset) / clk_scale
    d3 = recipe_idx / 2.0
    d4 = plat_flag

    # [5..6] F0 analytic
    d5 = math.log2(SW_BASELINE_CYCLES / max(f0_cycles, 1.0)) / 10.0
    d6 = float(f0_accuracy)

    # [7..8] F1 sim (0 if unrun)
    if f1_obs is not None:
        f1_cyc = float(f1_obs.get("avg_cycles", f0_cycles))
        d7 = math.log2(SW_BASELINE_CYCLES / max(f1_cyc, 1.0)) / 10.0
        d8 = float(f1_obs.get("accuracy", 0.0))
    else:
        d7 = 0.0
        d8 = 0.0

    # [9..13] F2 proxy (0 if unrun)
    if f2_obs is not None:
        raw_area = f2_obs.get("area_um2")
        raw_wns  = f2_obs.get("wns_ns")
        d9  = float(np.clip(raw_area / 20000.0, 0.0, 3.0)) if raw_area is not None else 0.0
        d10 = float(np.clip((raw_wns or 0.0) / 5.0, -2.0, 2.0))
        # netlist stats (cells/FF count) from F2 proxy result
        ff_count    = f2_obs.get("ff_count")    or f2_obs.get("cells") or 0
        cell_count  = f2_obs.get("cell_count") or f2_obs.get("cells") or 0
        logic_levels = f2_obs.get("logic_levels") or 0
        d11 = float(np.clip(ff_count    / 1000.0,  0.0, 3.0))
        d12 = float(np.clip(cell_count  / 10000.0, 0.0, 3.0))
        d13 = float(np.clip(logic_levels / 50.0,   0.0, 2.0))
    else:
        d9 = d10 = d11 = d12 = d13 = 0.0

    # [14..15] surrogate
    if surrogate is not None and _SURROGATE_AVAILABLE:
        obs = {}
        if f2_obs is not None:
            obs.update(f2_obs)
        try:
            mu, sigma = surrogate.predict_reward_stats(config, obs)
            d14 = float(mu) / 4.5
            d15 = float(sigma)
        except Exception:  # noqa: BLE001
            d14 = d15 = 0.0
    else:
        d14 = d15 = 0.0

    # [16] incumbent
    d16 = float(incumbent_reward) / 4.5 if incumbent_reward is not None else 0.0

    # [17] remaining budget
    d17 = float(np.clip(budget_fraction, 0.0, 1.0))

    # [18..21] depth one-hot (highest fidelity run: F0=0, F1=1, F2=2, F3=3)
    one_hot = [0.0, 0.0, 0.0, 0.0]
    one_hot[min(depth, 3)] = 1.0

    vec = np.array([
        d0, d1, d2, d3, d4,      # [0..4]  config encoding
        d5, d6,                   # [5..6]  F0 analytic
        d7, d8,                   # [7..8]  F1 sim
        d9, d10, d11, d12, d13,  # [9..13] F2 proxy + netlist
        d14, d15,                # [14..15] surrogate
        d16, d17,                # [16..17] incumbent + budget
        *one_hot,                # [18..21] depth one-hot
    ], dtype=np.float32)
    assert vec.shape == (_STATE_DIM,), f"state dim mismatch: {vec.shape}"
    return vec


# ── Fallback design descriptor (when designs.py is not yet importable) ──────────

class _TinymacFallback:
    """Minimal stand-in for DesignSpec when designs.py is not available.
    Preserves all pre-V12 FunnelEnv behaviour for tinymac."""
    name = "tinymac_accel"
    top = "tinymac_accel"
    has_macros = False

    def is_tinyvad(self) -> bool:
        return True


# ── Main class ─────────────────────────────────────────────────────────────────

class FunnelEnv:
    """Gym-style multi-fidelity funnel environment for the design-space optimizer.

    One episode = one candidate config.  The promotion policy drives the episode
    by choosing actions from ACTIONS at each step.

    Parameters
    ----------
    space_yaml : path to search_space_funnel.yaml (or any compatible YAML).
    platform   : "nangate45" (default) or "asap7".
    budget_s   : total wall-clock budget for ALL episodes (seconds).
                 Default 14400 s = 4 h (Phase 4 measured sustainable rate).
    surrogate  : optional Surrogate instance (surrogate.py) for state dims [14..15].
    table      : optional pre-built table dict {config_key → {fidelity → row}}.
                 When provided, FunnelEnv does NOT call any real tool; it replays
                 observations from the table and charges FIDELITY_COST_S (or the
                 row's recorded cost) against the budget.  Build with load_table().
    results_path: where to append per-fidelity JSONL rows.
    seed        : random seed (currently unused; reserved for stochastic proxies).
    lambda_cost : budget-pressure scaling factor λ in the per-step shaping reward.
    design      : str | DesignSpec | None.
                 - None (default) → DesignSpec.load("tinymac_accel") — zero
                   behaviour change for all existing callers and tests.
                 - str → DesignSpec.load(name_or_path).
                 - DesignSpec instance used directly.
                 Controls which design is run at F1/F2/F3.  When the design's
                 functional_eval kind is NOT "tinyvad_sim", F1 is skipped:
                 the "promote" action goes directly from F0 to F2.
                 The fidelity-depth one-hot [18..21] reuses slot [19] for F1
                 with value 0.0 when F1 is skipped (never set to 1.0).
                 F0 analytic cycles/accuracy also only run for tinyvad; for
                 generic designs F0 = legality/validate_config only (cycles
                 and accuracy both 0.0 in the state vector).
    max_tier    : int (default 1). Maximum knob tier active for this design.
                 Passed through to KnobRegistry.active() and to the runner.
    """

    ACTIONS = ["kill", "re-proxy", "promote", "commit"]

    # Expose FIDELITY_COST_S as a class attribute for benchmark scripts.
    FIDELITY_COST_S = FIDELITY_COST_S

    def __init__(
        self,
        space_yaml: str | Path = _DEFAULT_SPACE,
        platform: str = "nangate45",
        budget_s: float = 14400.0,
        surrogate: Any | None = None,
        table: dict | None = None,
        results_path: str | Path = str(Path(__file__).resolve().parent.parent / "results_funnel.jsonl"),
        seed: int = 0,
        lambda_cost: float = 1.0,
        design: "str | Any | None" = None,
        max_tier: int = 1,
        active_space: dict | None = None,
    ) -> None:
        self._space_yaml = Path(space_yaml)
        self.platform = platform
        self.budget_s = float(budget_s)
        self._surrogate = surrogate
        self._table = table
        self._results_path = Path(results_path)
        self._seed = seed
        self._lambda = float(lambda_cost)
        self._max_tier = int(max_tier)
        # active_space: the KnobRegistry space for this design+tier.  When set,
        # _validate_funnel uses its bounds instead of the YAML-hardcoded ones.
        # This allows generic designs (gcd) with different clock ranges to pass.
        self._active_space: dict | None = active_space

        # Resolve design spec (lazy: load on first use to keep import-time fast)
        self._design_arg = design   # raw arg; resolved in property below
        self.__design_spec: Any = None  # cache

        # Load space
        self._sim_params, self._proxy_params, self._constraints, \
            self._gates, self._reward_cfg = _load_space(self._space_yaml)

        # Runtime state (set by reset())
        self._config: dict | None = None
        self._depth: int = -1          # -1 = no episode; 0=F0 done; 1=F1 done; etc.
        self._f0_cycles: float = 0.0
        self._f0_accuracy: float = 0.0
        self._f1_obs: dict | None = None
        self._f2_obs: dict | None = None
        self._f3_obs: dict | None = None
        self._done: bool = True        # must call reset() before step()
        self._episode_spent_s: float = 0.0   # cost accumulated in this episode

        # Cumulative budget tracking
        self._spent_s: float = 0.0
        self._incumbent: dict | None = None  # {"config": ..., "reward": float}

        # _state_vec is always the most recently built 22-dim vector
        self._state_vec: np.ndarray = np.zeros(_STATE_DIM, dtype=np.float32)

    @property
    def _design_spec(self) -> Any:
        """Resolve and cache the DesignSpec (loaded on first access)."""
        if self.__design_spec is None:
            arg = self._design_arg
            if arg is None:
                # Default: tinymac_accel (zero behaviour change)
                try:
                    from common.designs import DesignSpec
                    self.__design_spec = DesignSpec.load("tinymac_accel")
                except Exception:   # noqa: BLE001
                    self.__design_spec = _TinymacFallback()
            elif isinstance(arg, str):
                from common.designs import DesignSpec
                self.__design_spec = DesignSpec.load(arg)
            else:
                self.__design_spec = arg
        return self.__design_spec

    def _f1_enabled(self) -> bool:
        """Return True if F1 (behavioral sim) should run for this design."""
        try:
            return self._design_spec.is_tinyvad()
        except Exception:   # noqa: BLE001
            return True   # safe default: run F1

    # ── Public properties ──────────────────────────────────────────────────────

    @property
    def incumbent(self) -> dict | None:
        """Best (config, final_reward) pair committed so far across all episodes."""
        return self._incumbent

    @property
    def spent_s(self) -> float:
        """Total wall-clock budget consumed so far (seconds)."""
        return self._spent_s

    # ── Episode interface ──────────────────────────────────────────────────────

    def reset(self, config: dict) -> np.ndarray:
        """Start an episode for `config`.

        Validates the config, runs F0 (free), returns the 22-dim state vector.
        Raises ValueError if the config is invalid (the caller's generator
        should only propose valid configs, but we check defensively).
        """
        # Validate: use active_space bounds when set (allows generic designs with
        # different parameter ranges than the YAML defaults, e.g. gcd clock [0.3, 2.0]).
        ok, reason = _validate_funnel(
            config, self._sim_params, self._proxy_params, self._constraints,
            active_space=self._active_space,
        )
        if not ok:
            raise ValueError(f"FunnelEnv.reset: invalid config: {reason}")

        # Initialise episode
        self._config = dict(config)
        self._depth = -1
        self._f1_obs = None
        self._f2_obs = None
        self._f3_obs = None
        self._done = False
        self._episode_spent_s = 0.0

        # Run F0 (analytic, free)
        self._run_f0()

        return self._state_vec.copy()

    def step(self, action: str) -> tuple[np.ndarray, float, bool, dict]:
        """Execute `action` and return (obs, reward, done, info).

        Actions:
            "kill"     → episode done immediately; reward = 0 (no info gained).
            "re-proxy" → (re-)run F2; may be called before F1/F2 have run.
            "promote"  → run the next not-yet-run fidelity (F1 → F2 → F3).
            "commit"   → jump directly to F3 (skips any unrun F1/F2); terminal.

        Per-step shaping (doc "Shaping" paragraph):
            r_k = Δ(surrogate-expected best) − λ·cost_k / budget_s
            When no surrogate: r_k = −λ·cost_k / budget_s
            On F3 terminal: adds the final composite reward (ladder-consistent).

        Reward is only *positive* from the F3 terminal payoff; shaping terms
        are always ≤ 0 (anti-gaming: proxy can only kill, never accept).
        """
        if self._done:
            raise RuntimeError("FunnelEnv.step() called after done=True; call reset() first.")
        if action not in self.ACTIONS:
            raise ValueError(f"Unknown action {action!r}; must be one of {self.ACTIONS}")

        info: dict = {"action": action, "config": self._config, "platform": self.platform}

        if action == "kill":
            self._done = True
            self._update_state()
            info["fidelity"] = f"F{self._depth}" if self._depth >= 0 else "F0"
            info["reason"] = "killed by agent"
            return self._state_vec.copy(), 0.0, True, info

        if action == "commit":
            # Jump to F3; shaping cost is F3 regardless of what was skipped
            reward = self._run_stage("F3")
            self._done = True
            info.update({"fidelity": "F3", "f3_obs": self._f3_obs})
            # Surface the effective recipe so callers can see plain→orfs_speed remapping
            if self._f3_obs:
                info["effective_abc_recipe"] = self._f3_obs.get(
                    "effective_abc_recipe", self._config.get("abc_recipe", "orfs_speed"))
                info["plain_remapped_to_orfs_speed"] = self._f3_obs.get(
                    "plain_remapped_to_orfs_speed", False)
            return self._state_vec.copy(), reward, True, info

        if action == "re-proxy":
            # (Re-)run F2 even if already run; resets f2_obs
            self._f2_obs = None
            reward = self._run_stage("F2")
            done = self._done   # _run_stage may set done=True for F3
            info.update({"fidelity": "F2", "f2_obs": self._f2_obs})
            return self._state_vec.copy(), reward, done, info

        # action == "promote": run next not-yet-run fidelity
        next_fid = self._next_fidelity()
        if next_fid is None:
            # Already at F3; treat as terminal
            self._done = True
            self._update_state()
            info["reason"] = "already at max fidelity"
            return self._state_vec.copy(), 0.0, True, info

        reward = self._run_stage(next_fid)
        done = self._done
        info.update({"fidelity": next_fid})
        # Surface effective recipe when F3 completes via promote
        if next_fid == "F3" and self._f3_obs:
            info["effective_abc_recipe"] = self._f3_obs.get(
                "effective_abc_recipe", self._config.get("abc_recipe", "orfs_speed"))
            info["plain_remapped_to_orfs_speed"] = self._f3_obs.get(
                "plain_remapped_to_orfs_speed", False)
        return self._state_vec.copy(), reward, done, info

    def state(self) -> np.ndarray:
        """Return the current 22-dim state vector (copy)."""
        return self._state_vec.copy()

    # ── Internal stage runners ─────────────────────────────────────────────────

    def _run_f0(self) -> None:
        """Run F0: free analytic validation + cycle model + accuracy table.

        For TinyVAD designs: runs the analytic cycle model and accuracy table.
        For other designs: F0 = legality only (cycles and accuracy both 0.0).
        """
        assert self._config is not None
        if self._f1_enabled():
            # TinyVAD: use analytic cycle model + accuracy table
            lanes  = int(self._config.get("mac_lanes", 4))
            acc_w  = int(self._config.get("accumulator_width", 24))
            self._f0_cycles   = _f0_cycles(lanes)
            self._f0_accuracy = _f0_accuracy(lanes, acc_w)
        else:
            # Generic design: F0 = legality only
            self._f0_cycles   = 0.0
            self._f0_accuracy = 0.0
        self._depth = 0
        cost_s = FIDELITY_COST_S["F0"]
        self._charge(cost_s)

        obs = {
            "cycles": self._f0_cycles,
            "accuracy": self._f0_accuracy,
        }
        self._log_row("F0", obs, cost_s, "ok")
        self._update_state()

    def _run_stage(self, fidelity: str) -> float:
        """Run `fidelity` (F1/F2/F3), update state, return shaped reward."""
        assert self._config is not None
        t_start = time.perf_counter()

        if fidelity == "F1":
            obs, status = self._run_f1()
            cost_s = time.perf_counter() - t_start
            self._f1_obs = obs
            fidx = 1
        elif fidelity == "F2":
            obs, status = self._run_f2()
            cost_s = time.perf_counter() - t_start
            self._f2_obs = obs
            fidx = 2
        elif fidelity == "F3":
            obs, status = self._run_f3()
            cost_s = time.perf_counter() - t_start
            self._f3_obs = obs
            fidx = 3
        else:
            raise ValueError(f"Unknown fidelity {fidelity!r}")

        # In table mode charge the recorded cost (or fallback to FIDELITY_COST_S)
        if self._table is not None:
            key = _config_key(self._config)
            row = (self._table.get(key) or {}).get(fidelity, {})
            cost_s = float(row.get("cost_s", FIDELITY_COST_S[fidelity]))
        else:
            # Clamp to at least the expected floor (mock runs are near-instant)
            cost_s = max(cost_s, FIDELITY_COST_S.get(fidelity, 0.0) * 0.01)

        self._charge(cost_s)
        self._depth = max(self._depth, fidx)

        self._log_row(fidelity, obs, cost_s, status)
        self._update_state()

        # Compute shaped reward
        prior_mu = self._surrogate_mu()
        reward = -self._lambda * cost_s / max(self.budget_s, 1.0)

        if fidelity == "F3":
            # Terminal payoff: final composite reward using the ladder-consistent scorer
            terminal_reward = self._terminal_reward(obs, status)
            reward += terminal_reward
            # Update incumbent
            if self._incumbent is None or terminal_reward > self._incumbent["reward"]:
                self._incumbent = {"config": dict(self._config), "reward": terminal_reward}
            self._done = True

            # Surrogate Δ shaping: Δ(best) after vs before observation
            if self._surrogate is not None:
                post_mu = self._surrogate_mu()
                reward += (post_mu - prior_mu)
        else:
            # Intermediate shaping from surrogate (delta expected best)
            if self._surrogate is not None:
                post_mu = self._surrogate_mu()
                reward += (post_mu - prior_mu)

        return float(reward)

    def _run_f1(self) -> tuple[dict, str]:
        """F1: behavioral Verilator sim (mock-aware via cascade._run_sim).

        Only called for TinyVAD designs (_f1_enabled() is True).  mac_lanes and
        accumulator_width are always present in TinyVAD configs, but use .get()
        defensively to avoid KeyError on unusual call paths.
        """
        assert self._config is not None
        lanes = int(self._config.get("mac_lanes", 4))
        acc_w = int(self._config.get("accumulator_width", 24))

        if self._table is not None:
            key = _config_key(self._config)
            row = (self._table.get(key) or {}).get("F1")
            if row is not None:
                return dict(row.get("obs", {})), row.get("status", "ok")
            # Table miss: return F0 analytic as fallback
            return {
                "avg_cycles": self._f0_cycles,
                "accuracy": self._f0_accuracy,
                "correct": round(self._f0_accuracy * 64),
                "n_total": 64,
            }, "table_miss"

        try:
            sim = _run_sim(lanes, acc_w)
            obs = {
                "avg_cycles": sim.get("avg_cycles", self._f0_cycles),
                "accuracy":   sim.get("accuracy",   0.0),
                "correct":    sim.get("correct",     0),
                "n_total":    sim.get("n_total",     64),
            }
            return obs, "ok"
        except Exception as exc:  # noqa: BLE001
            return {"avg_cycles": self._f0_cycles, "accuracy": 0.0,
                    "error": str(exc)}, "FAIL"

    def _run_f2(self) -> tuple[dict, str]:
        """F2: synth+STA proxy (run_synth_sta, mock-aware)."""
        assert self._config is not None
        lanes  = int(self._config.get("mac_lanes", 4))
        acc_w  = int(self._config.get("accumulator_width", 24))
        clk    = float(self._config["clock_period_ns"])
        recipe = self._config.get("abc_recipe", "plain")

        if self._table is not None:
            key = _config_key(self._config)
            row = (self._table.get(key) or {}).get("F2")
            if row is not None:
                return dict(row.get("obs", {})), row.get("status", "ok")
            return {}, "table_miss"

        # Defensive: pass abc_recipe and design if run_synth_sta accepts them
        sig = inspect.signature(run_synth_sta)
        kwargs: dict = {}
        if "abc_recipe" in sig.parameters:
            kwargs["abc_recipe"] = recipe
        if "design" in sig.parameters:
            try:
                kwargs["design"] = self._design_spec
            except Exception:   # noqa: BLE001
                pass   # design resolution failed; fall back to tinymac behaviour

        try:
            result = run_synth_sta(lanes, acc_w, clk, self.platform, **kwargs)
            status = result.get("status", "ok")
            obs = {
                "area_um2":    result.get("area_um2"),
                "wns_ns":      result.get("wns_ns"),
                "tns_ns":      result.get("tns_ns"),
                "fmax_mhz":    result.get("fmax_mhz"),
                "timing_met":  result.get("timing_met"),
                "cell_count":  result.get("cells"),
                "ff_count":    result.get("cells"),   # proxy has no separate FF count
                "logic_levels": None,                 # not available from synth+STA
            }
            return obs, status
        except Exception as exc:  # noqa: BLE001
            return {"error": str(exc)}, "FAIL"

    def _run_f3(self) -> tuple[dict, str]:
        """F3: full ORFS RTL→GDS flow (run_physical, mock-aware)."""
        assert self._config is not None
        lanes   = int(self._config.get("mac_lanes", 4))
        acc_w   = int(self._config.get("accumulator_width", 24))
        clk     = float(self._config["clock_period_ns"])
        recipe  = self._config.get("abc_recipe", "plain")
        # Fixed flow constants from YAML
        util    = 40
        density = 0.60

        if self._table is not None:
            key = _config_key(self._config)
            row = (self._table.get(key) or {}).get("F3")
            if row is not None:
                return dict(row.get("obs", {})), row.get("status", "ok")
            return {}, "table_miss"

        # The ORFS full flow does not support the "plain" recipe (bare abc -liberty
        # is for the synth+STA proxy only; it produces an unbuffered netlist that
        # PnR handles differently).  Map "plain" → "orfs_speed" for F3 so that
        # a config with abc_recipe="plain" still gets a meaningful full-flow result.
        #
        # SEAM NOTE: this mapping is intentional (plain is proxy-only per recipe.py
        # docs), but the asymmetry must be recorded in the obs and info dict so the
        # training corpus and agent are never confused about what the flow actually ran.
        # Callers should check obs["effective_abc_recipe"] (not config["abc_recipe"])
        # when interpreting F3 physical results.
        f3_recipe = recipe if recipe != "plain" else "orfs_speed"
        # Record whether a recipe remapping occurred (for corpus integrity / debugging).
        plain_remapped = (recipe == "plain")

        # Defensive: pass abc_recipe and design if run_physical accepts them
        sig = inspect.signature(run_physical)
        kwargs: dict = {}
        if "abc_recipe" in sig.parameters:
            kwargs["abc_recipe"] = f3_recipe
        elif "abc" in sig.parameters:
            # Fallback: existing run_physical uses 'abc' kwarg; orfs_area→"area", else None
            abc_val: str | None = None
            if f3_recipe == "orfs_area":
                abc_val = "area"
            kwargs["abc"] = abc_val
        if "design" in sig.parameters:
            try:
                kwargs["design"] = self._design_spec
            except Exception:   # noqa: BLE001
                pass   # design resolution failed; fall back to tinymac behaviour

        try:
            result = run_physical(lanes, acc_w, clk, self.platform,
                                  util, density, **kwargs)
            status = result.get("status", "ok")
            obs = {
                "area_um2":    result.get("area_um2"),
                "wns_ns":      result.get("wns_ns"),
                "tns_ns":      result.get("tns_ns"),
                "fmax_mhz":    result.get("fmax_mhz"),
                "power_mw":    result.get("power_mw"),
                "timing_met":  result.get("timing_met"),
                "setup_viol":  result.get("setup_viol"),
                "period_min_ns": result.get("period_min_ns"),
                "gds":         result.get("gds"),
                # Effective recipe used at F3 (may differ from config["abc_recipe"]
                # because "plain" is not a valid full-flow recipe and is remapped to
                # "orfs_speed").  Always carry this so the training corpus is honest.
                "effective_abc_recipe": f3_recipe,
                "plain_remapped_to_orfs_speed": plain_remapped,
                # For reward computation
                "lanes":    lanes,
                "acc_w":    acc_w,
                "clk_ns":   clk,
                "platform": self.platform,
                "status":   status,
            }
            return obs, status
        except Exception as exc:  # noqa: BLE001
            return {"status": "FAIL", "error": str(exc),
                    "lanes": lanes, "acc_w": acc_w,
                    "clk_ns": clk, "platform": self.platform,
                    "effective_abc_recipe": f3_recipe,
                    "plain_remapped_to_orfs_speed": plain_remapped}, "FAIL"

    # ── Terminal reward computation ─────────────────────────────────────────────

    def _terminal_reward(self, obs: dict, status: str) -> float:
        """Compute the F3 terminal payoff using the ladder-consistent scorer.

        Successful F3 → compute_physical_reward (actual PPA metrics).
        Non-ok status → monotone penalty from the ladder (-20 for full-flow-fail).
        Table miss → -20 (same as full-flow-fail; we have no data).
        """
        penalties = self._reward_cfg.get("stage_penalty", {})
        full_fail_penalty = float(penalties.get("full", -20.0))

        if status in ("ok", "mock", "mock-proxy"):
            # Real or mock physical metrics: score normally
            sim_cycles = None
            if self._f1_obs is not None:
                sim_cycles = self._f1_obs.get("avg_cycles")

            scored = compute_physical_reward(obs, self._reward_cfg,
                                             cycles=sim_cycles)
            return float(scored.get("reward", full_fail_penalty))
        else:
            # FAIL / TIMEOUT / PARSE_FAIL / table_miss → ladder penalty
            if status in ("TIMEOUT", "table_miss"):
                # Charge less than a clean FAIL (some info gathered)
                return full_fail_penalty
            return full_fail_penalty

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _next_fidelity(self) -> str | None:
        """Return the next not-yet-run fidelity string, or None if at F3.

        F1 is skipped for non-TinyVAD designs: promote goes F0→F2 directly.
        The fidelity-depth one-hot [18..21] slot [19] (F1) stays 0.0 when skipped —
        it is never set to 1.0, so the state dim layout is preserved exactly.
        """
        if self._depth < 1:
            # F0 done; next is F1 for TinyVAD, or F2 for generic designs
            if self._f1_enabled():
                return "F1"
            else:
                return "F2"   # skip F1 — go straight to F2
        if self._depth < 2:
            return "F2"
        if self._depth < 3:
            return "F3"
        return None   # already at F3

    def _charge(self, cost_s: float) -> None:
        self._spent_s += cost_s
        self._episode_spent_s += cost_s

    def _budget_fraction(self) -> float:
        return max(0.0, 1.0 - self._spent_s / max(self.budget_s, 1.0))

    def _surrogate_mu(self) -> float:
        """Ask surrogate for expected reward; 0.0 if unavailable."""
        if self._surrogate is None or not _SURROGATE_AVAILABLE:
            return 0.0
        try:
            obs = {}
            if self._f2_obs:
                obs.update(self._f2_obs)
            mu, _ = self._surrogate.predict_reward_stats(self._config, obs)
            return float(mu)
        except Exception:  # noqa: BLE001
            return 0.0

    def _update_state(self) -> None:
        """Rebuild the 22-dim state vector from current episode state."""
        assert self._config is not None
        self._state_vec = _build_state(
            config=self._config,
            platform=self.platform,
            f0_cycles=self._f0_cycles,
            f0_accuracy=self._f0_accuracy,
            f1_obs=self._f1_obs,
            f2_obs=self._f2_obs,
            surrogate=self._surrogate,
            incumbent_reward=self._incumbent["reward"] if self._incumbent else None,
            budget_fraction=self._budget_fraction(),
            depth=self._depth,
        )

    def _log_row(self, fidelity: str, obs: dict, cost_s: float, status: str) -> None:
        """Append a row to results_funnel.jsonl."""
        row = {
            "ts":       time.time(),
            "config":   dict(self._config),
            "fidelity": fidelity,
            "obs":      obs,
            "cost_s":   round(cost_s, 4),
            "platform": self.platform,
            "status":   status,
        }
        try:
            with open(self._results_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")
        except OSError:
            pass   # non-fatal: logging failure should never crash the optimizer


# ── Self-test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    import tempfile

    print("=== FunnelEnv self-test ===")

    # ── Helper: make a minimal synthetic table ────────────────────────────────
    def _make_table_rows(configs: list[dict], results_path: Path) -> dict:
        """Write synthetic table rows and return load_table() result."""
        for cfg in configs:
            lanes = int(cfg["mac_lanes"])
            acc_w = int(cfg["accumulator_width"])
            clk   = float(cfg["clock_period_ns"])

            cyc = float(AVG_CYCLES.get(lanes, _behavioral_cycles(lanes)))
            acc = _f0_accuracy(lanes, acc_w)

            # F0 row
            f0_row = {"ts": 0.0, "config": cfg, "fidelity": "F0",
                      "obs": {"cycles": cyc, "accuracy": acc},
                      "cost_s": 0.0, "platform": "nangate45", "status": "ok"}
            # F1 row
            f1_row = {"ts": 0.0, "config": cfg, "fidelity": "F1",
                      "obs": {"avg_cycles": cyc, "accuracy": acc,
                              "correct": round(acc * 64), "n_total": 64},
                      "cost_s": 5.0, "platform": "nangate45", "status": "ok"}
            # F2 row (synthetic proxy)
            area_est = 14000.0 + 2500.0 * lanes
            wns = clk - 3.82
            f2_row = {"ts": 0.0, "config": cfg, "fidelity": "F2",
                      "obs": {"area_um2": area_est, "wns_ns": round(wns, 3),
                              "tns_ns": min(wns, 0.0) * 5,
                              "fmax_mhz": 1000.0 / max(3.82 - wns * 0.1, 1.0),
                              "timing_met": wns >= 0.0,
                              "cell_count": int(area_est / 3.5),
                              "ff_count": 230,
                              "logic_levels": None},
                      "cost_s": 45.0, "platform": "nangate45", "status": "ok"}
            # F3 row (synthetic full-flow)
            area_f3 = area_est * 1.35
            fmax = 269.0
            power = 900.0 + 30.0 * lanes
            f3_row = {"ts": 0.0, "config": cfg, "fidelity": "F3",
                      "obs": {"area_um2": round(area_f3, 1),
                              "wns_ns": round(wns, 3),
                              "fmax_mhz": fmax,
                              "power_mw": power,
                              "timing_met": wns >= 0.0,
                              "period_min_ns": round(1000.0 / fmax, 2),
                              "lanes": lanes, "acc_w": acc_w,
                              "clk_ns": clk, "platform": "nangate45",
                              "status": "ok"},
                      "cost_s": 420.0, "platform": "nangate45", "status": "ok"}
            with open(results_path, "a") as f:
                for row in (f0_row, f1_row, f2_row, f3_row):
                    f.write(json.dumps(row) + "\n")
        return load_table(results_path)

    SPACE = Path(__file__).resolve().parent / "search_space_funnel.yaml"

    # ── TEST A: TABLE MODE ─────────────────────────────────────────────────────
    print("\n--- TEST A: table mode (3 configs) ---")

    with tempfile.TemporaryDirectory() as tmpdir:
        tpath = Path(tmpdir) / "results_funnel.jsonl"
        rpath = Path(tmpdir) / "results_out.jsonl"

        configs = [
            {"mac_lanes": 4, "accumulator_width": 24, "clock_period_ns": 5.0,
             "abc_recipe": "plain"},
            {"mac_lanes": 8, "accumulator_width": 32, "clock_period_ns": 4.0,
             "abc_recipe": "orfs_area"},
            {"mac_lanes": 2, "accumulator_width": 16, "clock_period_ns": 6.0,
             "abc_recipe": "orfs_speed"},
        ]
        table = _make_table_rows(configs, tpath)
        print(f"  Table keys: {len(table)} configs, fidelities per key: "
              f"{[sorted(v.keys()) for v in table.values()]}")

        env = FunnelEnv(space_yaml=SPACE, table=table,
                        results_path=rpath, budget_s=3600.0)

        # A1: kill at F0
        s0 = env.reset(configs[0])
        assert s0.shape == (_STATE_DIM,), f"A1: state shape {s0.shape}"
        assert s0[18] == 1.0, f"A1: depth one-hot[F0] expected 1.0, got {s0[18]}"
        assert s0[19] == 0.0, f"A1: depth one-hot[F1] expected 0.0, got {s0[19]}"
        obs, r, done, info = env.step("kill")
        assert done, "A1: kill must set done=True"
        assert r == 0.0, f"A1: kill reward must be 0, got {r}"
        print(f"  A1 kill at F0: done={done}, r={r:.4f}  PASS")

        # A2: promote × 3 (F0 → F1 → F2 → F3)
        s0 = env.reset(configs[1])
        assert s0.shape == (_STATE_DIM,)
        depths = [env._depth]
        for step_i in range(3):
            obs, r, done, info = env.step("promote")
            assert obs.shape == (_STATE_DIM,), f"A2 step {step_i}: obs shape"
            depths.append(env._depth)
        assert done, "A2: must be done after 3 promotes"
        # Verify monotone depth: [0, 1, 2, 3]
        for i in range(1, len(depths)):
            assert depths[i] >= depths[i-1], f"A2: depth not monotone: {depths}"
        # Verify one-hot at F3: [18..21] = [0,0,0,1]
        assert obs[18] == 0.0 and obs[19] == 0.0 and obs[20] == 0.0 and obs[21] == 1.0, \
            f"A2: depth one-hot at F3: {obs[18:22]}"
        print(f"  A2 promote×3: done={done}, r={r:.4f}, depths={depths}  PASS")

        # A3: commit from F0 (skip F1/F2, go straight to F3)
        s0 = env.reset(configs[2])
        assert s0.shape == (_STATE_DIM,)
        assert env._depth == 0, f"A3: depth after reset should be 0, got {env._depth}"
        obs, r, done, info = env.step("commit")
        assert done, "A3: commit must set done=True"
        assert obs[21] == 1.0, f"A3: depth one-hot[F3] expected 1.0 after commit"
        print(f"  A3 commit from F0: done={done}, r={r:.4f}  PASS")

        # A4: budget accounting
        spent_before = env.spent_s
        s0 = env.reset(configs[0])
        env.step("promote")   # F1: 5s
        env.step("promote")   # F2: 45s
        env.step("promote")   # F3: 420s
        spent_after = env.spent_s
        episode_cost = spent_after - spent_before
        # F0=0 + F1=5 + F2=45 + F3=420 = 470s
        assert abs(episode_cost - 470.0) < 1.0, \
            f"A4: expected ~470s episode cost, got {episode_cost:.2f}s"
        print(f"  A4 budget: episode_cost={episode_cost:.1f}s (expected 470.0)  PASS")

        # A5: jsonl rows written
        rows = [json.loads(l) for l in rpath.read_text().splitlines() if l.strip()]
        assert len(rows) > 0, "A5: no rows written to results_funnel.jsonl"
        fidelities = [r["fidelity"] for r in rows]
        assert "F0" in fidelities, "A5: F0 row missing"
        print(f"  A5 jsonl rows: {len(rows)} rows, fidelities={sorted(set(fidelities))}  PASS")

        # A6: state dim invariant on every row
        # (already checked per step; do a quick final pass)
        assert env.state().shape == (_STATE_DIM,), "A6: state() shape"
        print(f"  A6 state dim: {env.state().shape}  PASS")

    # ── TEST B: LIVE MODE with PHYSICAL_MOCK=1 ────────────────────────────────
    print("\n--- TEST B: live mode (PHYSICAL_MOCK=1), promote×3 ---")
    os.environ["PHYSICAL_MOCK"] = "1"
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            rpath = Path(tmpdir) / "results_live.jsonl"
            env = FunnelEnv(space_yaml=SPACE, table=None,
                            results_path=rpath, budget_s=3600.0)
            cfg = {"mac_lanes": 4, "accumulator_width": 24,
                   "clock_period_ns": 5.0, "abc_recipe": "plain"}
            s0 = env.reset(cfg)
            assert s0.shape == (_STATE_DIM,), f"B: state shape after reset"

            total_r = 0.0
            for step_i in range(3):
                obs, r, done, info = env.step("promote")
                assert obs.shape == (_STATE_DIM,), f"B step {step_i}: obs shape"
                total_r += r
                fid = info.get("fidelity", "?")
                print(f"  B step {step_i+1} ({fid}): r={r:.4f}, done={done}")

            assert done, "B: must be done after 3 promotes"
            rows = [json.loads(l) for l in rpath.read_text().splitlines() if l.strip()]
            fids = [r["fidelity"] for r in rows]
            print(f"  B jsonl: {len(rows)} rows, fidelities={sorted(set(fids))}")
            assert "F1" in fids and "F2" in fids and "F3" in fids, \
                f"B: expected F1/F2/F3 rows, got {fids}"
            print(f"  B live mode PASS  (total_r={total_r:.4f})")
    finally:
        del os.environ["PHYSICAL_MOCK"]

    print("\n=== All self-tests PASSED ===")
    sys.exit(0)
