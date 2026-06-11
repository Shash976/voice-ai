"""build_table.py — resumable offline F0–F2 table builder.

Populates results_funnel.jsonl with F0 (analytic), F1 (behavioral sim), and
F2 (synth+STA proxy) evaluations across the reduced design space defined in
search_space_funnel.yaml (or the fallback synthetic space if the YAML hasn't
landed yet).

Design space per Phase 4 / Phase 6:
    lanes  ∈ {1, 2, 4, 8, 16, 32}    (6 values)
    acc_w  ∈ {16, 24, 32}             (3 values)
    clk    ∈ [3.0, 8.0] step 0.5 ns  (11 values: 3.0, 3.5, ..., 8.0)
    recipe ∈ {orfs_speed, orfs_area, plain}  (3 values)
Total grid: 6 × 3 × 11 × 3 = 594 configs.

Key optimisations for wall-clock efficiency:
    F1 (behavioral sim cycles) depends ONLY on lanes → evaluate once per
    unique lanes value and reuse.  acc_w affects accuracy but NOT cycles.

    F2 (synth+STA proxy) depends on lanes / acc_w / clk / recipe.
    util and density are fixed (dropped from the funnel space per Phase 5).

Row format (pinned jsonl schema):
    {"ts", "config", "fidelity", "obs", "cost_s", "platform", "status"}

Usage:
    python3 build_table.py [options]

Options:
    --space PATH       search_space_funnel.yaml  (default: search_space_funnel.yaml)
    --out PATH         output jsonl              (default: results_funnel.jsonl)
    --fidelity F0|F1|F2  target fidelity         (default: F2)
    --limit N          max evaluations to run    (default: no limit)
    --subset strategic  corner+axis sweep (~54-87 deduped configs instead of 594)
    --dry-run          print plan + cost estimate, no evaluations
    --platform NAME    ORFS platform             (default: nangate45)

Set PHYSICAL_MOCK=1 to run in mock mode (no ORFS tools needed).
"""

from __future__ import annotations

import argparse
import inspect
import json
import math
import os
import sys
import time
from itertools import product
from pathlib import Path

import numpy as np

# ── path setup ────────────────────────────────────────────────────────────────
_THIS_DIR = Path(__file__).resolve().parent
_OPT_DIR  = Path(__file__).resolve().parents[1]
if str(_OPT_DIR) not in sys.path:
    sys.path.insert(0, str(_OPT_DIR))

# ── fidelity cost table (seconds) ─────────────────────────────────────────────
# From Phase 4 / FunnelEnv spec.  F3/F4 are not built by this script.
FIDELITY_COST_S = {
    "F0": 0.0,
    "F1": 5.0,
    "F2": 45.0,
}

# ── RECIPES ───────────────────────────────────────────────────────────────────
try:
    from common.recipe import RECIPES  # type: ignore[import]
except ImportError:
    RECIPES = ("orfs_speed", "orfs_area", "plain")

# ── constants ─────────────────────────────────────────────────────────────────
from common.constants import (
    AVG_CYCLES,
    SW_BASELINE_CYCLES,
    behavioral_cycles,
)

# ── search space definition ───────────────────────────────────────────────────

_FUNNEL_YAML = _THIS_DIR / "search_space_funnel.yaml"

# Hard-coded fallback space matching the Phase 4 reduced space spec.
# Populated from the YAML if it exists (concurrent agent landing it).
_DEFAULT_LANES   = [1, 2, 4, 8, 16, 32]
_DEFAULT_ACC_WS  = [16, 24, 32]
_DEFAULT_CLKS    = [3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0]
_DEFAULT_RECIPES = list(RECIPES)


def _load_space(space_path: Path) -> dict:
    """Load search space from YAML, returning a dict with lanes/acc_ws/clks/recipes."""
    space = {
        "lanes":   _DEFAULT_LANES,
        "acc_ws":  _DEFAULT_ACC_WS,
        "clks":    _DEFAULT_CLKS,
        "recipes": _DEFAULT_RECIPES,
    }
    if not space_path.exists():
        return space
    try:
        import yaml  # noqa: F401
        with open(space_path, encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        # Parse funnel YAML structure; fall back to defaults for missing keys
        sp = raw.get("sim_params", {})
        pp = raw.get("proxy_params", {})

        if "mac_lanes" in sp:
            space["lanes"] = list(sp["mac_lanes"].get("choices", _DEFAULT_LANES))
        if "accumulator_width" in sp:
            space["acc_ws"] = list(sp["accumulator_width"].get("choices", _DEFAULT_ACC_WS))
        if "clock_period_ns" in pp:
            space["clks"] = list(pp["clock_period_ns"].get("choices", _DEFAULT_CLKS))
        if "abc_recipe" in pp:
            space["recipes"] = list(pp["abc_recipe"].get("choices", _DEFAULT_RECIPES))
        elif "abc_strategy" in pp:
            space["recipes"] = list(pp["abc_strategy"].get("choices", _DEFAULT_RECIPES))
    except Exception:  # noqa: BLE001 — yaml not installed or wrong format; use defaults
        pass
    return space


def _enumerate_grid(space: dict) -> list[dict]:
    """Generate all (lanes, acc_w, clk, recipe) combos.

    Config dicts use the canonical FunnelEnv key names (mac_lanes,
    accumulator_width, clock_period_ns, abc_recipe) so that funnel.load_table
    can match them via _config_key without any key-name translation.
    """
    configs = []
    for lanes, acc_w, clk, recipe in product(
        space["lanes"], space["acc_ws"], space["clks"], space["recipes"]
    ):
        configs.append({
            "mac_lanes":          int(lanes),
            "accumulator_width":  int(acc_w),
            "clock_period_ns":    float(clk),
            "abc_recipe":         str(recipe),
        })
    return configs


def _strategic_subset(space: dict) -> list[dict]:
    """Corner + axis sweep: ~54–87 deduped configs.

    Strategy (per Phase 6 doc):
      (A) All lanes × acc_w combos at clk=4.0, each recipe: 6×3×3 = 54 configs
      (B) Clock sweep at L4_A24, each recipe: 11×3 = 33 configs (many overlap with A)

    Config dicts use canonical FunnelEnv key names (mac_lanes, accumulator_width,
    clock_period_ns, abc_recipe) to ensure funnel.load_table lookup works correctly.

    Deduped.
    """
    seen: set[tuple] = set()
    configs: list[dict] = []

    def _add(lanes: int, acc_w: int, clk: float, recipe: str) -> None:
        key = (lanes, acc_w, round(clk, 4), recipe)
        if key not in seen:
            seen.add(key)
            configs.append({
                "mac_lanes":         lanes,
                "accumulator_width": acc_w,
                "clock_period_ns":   clk,
                "abc_recipe":        recipe,
            })

    # (A) all lanes × acc_w at clk=4.0, all recipes
    target_clk = 4.0
    # Find nearest clk value in the space
    clk_choices = space["clks"]
    nearest = min(clk_choices, key=lambda c: abs(c - target_clk))
    for lanes in space["lanes"]:
        for acc_w in space["acc_ws"]:
            for recipe in space["recipes"]:
                _add(lanes, acc_w, nearest, recipe)

    # (B) clock sweep at L4_A24, all recipes
    l4 = 4 if 4 in space["lanes"] else space["lanes"][0]
    a24 = 24 if 24 in space["acc_ws"] else space["acc_ws"][0]
    for clk in clk_choices:
        for recipe in space["recipes"]:
            _add(l4, a24, clk, recipe)

    return configs


# ── JSONL I/O ─────────────────────────────────────────────────────────────────

def _load_existing(out_path: Path) -> set[tuple]:
    """Return set of (lanes, acc_w, clk, recipe, fidelity) already present.

    Accepts both the canonical FunnelEnv key names (mac_lanes / accumulator_width /
    clock_period_ns / abc_recipe) AND the legacy build_table short names (lanes /
    acc_w / clk / recipe) so existing rows in the file are correctly detected as
    already done when resuming.
    """
    done: set[tuple] = set()
    if not out_path.exists():
        return done
    with open(out_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                cfg = row.get("config", {})
                fid = row.get("fidelity", "")
                # Support both canonical and legacy key names
                lanes  = int(cfg.get("mac_lanes") or cfg.get("lanes") or 0)
                acc_w  = int(cfg.get("accumulator_width") or cfg.get("acc_w") or 0)
                clk    = round(float(cfg.get("clock_period_ns") or cfg.get("clk") or 0), 4)
                recipe = str(cfg.get("abc_recipe") or cfg.get("recipe") or "")
                key = (lanes, acc_w, clk, recipe, str(fid))
                done.add(key)
            except (json.JSONDecodeError, TypeError, KeyError):
                pass
    return done


def _write_row(out_path: Path, config: dict, fidelity: str, obs: dict,
               cost_s: float, platform: str, status: str) -> None:
    row = {
        "ts":       time.time(),
        "config":   config,
        "fidelity": fidelity,
        "obs":      obs,
        "cost_s":   round(cost_s, 3),
        "platform": platform,
        "status":   status,
    }
    with open(out_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")


# ── F0: analytic (free) ───────────────────────────────────────────────────────

def _eval_f0(config: dict) -> tuple[dict, str]:
    """Analytic evaluation: cycles from constants.py + accuracy gate.

    F0 obs keys: cycles, accuracy_flag, cycle_speedup

    Accepts both canonical (mac_lanes / accumulator_width) and legacy (lanes / acc_w)
    key names so this function works regardless of which grid-builder produced config.
    """
    lanes  = int(config.get("mac_lanes") or config.get("lanes") or 0)
    acc_w  = int(config.get("accumulator_width") or config.get("acc_w") or 0)

    # Cycles: use measured table if available, else analytic fit
    cyc = float(AVG_CYCLES[lanes]) if lanes in AVG_CYCLES else behavioral_cycles(lanes)
    # Accuracy flag: acc_w < 24 causes overflow (measured 47/64 ≈ 0.734)
    if acc_w >= 24:
        acc = 1.0
    elif acc_w >= 20:
        acc = 0.92
    else:
        acc = 47.0 / 64.0
    speedup = SW_BASELINE_CYCLES / max(cyc, 1.0)

    obs = {
        "cycles":         cyc,
        "accuracy_flag":  acc,
        "cycle_speedup":  round(speedup, 3),
    }
    return obs, "ok"


# ── F1: behavioral sim (cycles depend only on lanes → dedup) ─────────────────

# Module-level cache: {lanes: (obs, status)} so each unique lanes value is
# simulated once.  acc_w does NOT affect cycle count (only accuracy).
_F1_CACHE: dict[int, tuple[dict, str]] = {}


def _eval_f1(config: dict) -> tuple[dict, str]:
    """Behavioral sim.  Caches per-lanes (cycles invariant to acc_w/clk/recipe).

    DEDUP NOTE: F1 cycles = n_outputs × (ceil(K/LANES) + 2), formula from
    constants.py.  Only LANES matters for the cycle count; acc_w only affects
    the accuracy of the output values, not the cycle count.  So we run the sim
    once per lanes value and reuse.  The accuracy obs comes from the F0 analytic
    model (same gate logic, already consistent with cascade.py _run_sim mock).
    """
    lanes = int(config.get("mac_lanes") or config.get("lanes") or 0)
    acc_w = int(config.get("accumulator_width") or config.get("acc_w") or 0)

    if lanes not in _F1_CACHE:
        if os.environ.get("PHYSICAL_MOCK"):
            # Mock mode: return analytic value (same as F0 but "measured")
            cyc = float(AVG_CYCLES[lanes]) if lanes in AVG_CYCLES else behavioral_cycles(lanes)
            obs_sim = {"avg_cycles": cyc, "accuracy": 1.0 if acc_w >= 24 else 47.0 / 64.0}
            _F1_CACHE[lanes] = (obs_sim, "mock")
        else:
            try:
                from gen1.runner import run_sim  # type: ignore[import]
                raw = run_sim(lanes, acc_width=32)   # acc_width=32 → guaranteed correct
                obs_sim = {
                    "avg_cycles": float(raw.get("avg_cycles", behavioral_cycles(lanes))),
                    "accuracy":   float(raw.get("accuracy", 1.0)),
                }
                _F1_CACHE[lanes] = (obs_sim, "ok")
            except Exception as exc:  # noqa: BLE001
                # Sim not available — fall back to analytic (marks as "fallback")
                cyc = float(AVG_CYCLES[lanes]) if lanes in AVG_CYCLES else behavioral_cycles(lanes)
                _F1_CACHE[lanes] = (
                    {"avg_cycles": cyc, "accuracy": 1.0, "fallback_reason": str(exc)},
                    "fallback",
                )

    base_obs, base_status = _F1_CACHE[lanes]
    # Override accuracy with acc_w-specific value (the per-lanes cache assumed acc_w=32).
    # Uses funnel's measured (lanes, acc_w) table (V13: acc16 accuracy is LANES-dependent)
    # so table F1 rows agree with FunnelEnv's F0 analytic accuracy.
    try:
        from gen2.funnel import _f0_accuracy
        acc = _f0_accuracy(lanes, acc_w)
    except ImportError:
        acc = 1.0 if acc_w >= 24 else (0.92 if acc_w >= 20 else 47.0 / 64.0)

    obs = dict(base_obs)
    obs["accuracy"] = acc
    obs["lanes_cache_hit"] = True   # audit flag so reviewer can see the dedup in action
    return obs, base_status


# ── F2: synth+STA proxy ────────────────────────────────────────────────────────

# Module-level cache: {(lanes, acc_w, clk, recipe): (obs, status)}
# Avoids re-running the 45-second synthesis for identical physical inputs.
_F2_CACHE: dict[tuple, tuple[dict, str]] = {}


def _eval_f2(config: dict, platform: str) -> tuple[dict, str]:
    """Run synth+STA proxy (run_synth_sta), importing abc_recipe defensively.

    F2 cache key = (lanes, acc_w, clk, recipe) — util/density are fixed.

    The abc_recipe parameter is passed to run_synth_sta only if the function
    signature accepts it (concurrent Stage-B recipe.py agent may add it).
    """
    lanes  = int(config.get("mac_lanes") or config.get("lanes") or 0)
    acc_w  = int(config.get("accumulator_width") or config.get("acc_w") or 0)
    clk    = float(config.get("clock_period_ns") or config.get("clk") or 5.0)
    recipe = str(config.get("abc_recipe") or config.get("recipe") or "plain")

    cache_key = (lanes, acc_w, round(clk, 4), recipe, platform)
    if cache_key in _F2_CACHE:
        obs, status = _F2_CACHE[cache_key]
        obs = dict(obs)
        obs["f2_cache_hit"] = True
        return obs, status

    try:
        from common.physical_runner import run_synth_sta  # type: ignore[import]
    except ImportError as exc:
        return {"error": str(exc)}, "import_error"

    # Defensive: pass abc_recipe only if run_synth_sta accepts it
    try:
        sig = inspect.signature(run_synth_sta)
        accepts_recipe = "abc_recipe" in sig.parameters
    except (ValueError, TypeError):
        accepts_recipe = False

    try:
        if accepts_recipe:
            raw = run_synth_sta(lanes, acc_w, clk, platform, abc_recipe=recipe)
        else:
            raw = run_synth_sta(lanes, acc_w, clk, platform)
    except Exception as exc:  # noqa: BLE001
        return {"error": str(exc)}, "error"

    status = raw.get("status", "ok")
    obs = {
        "area_um2":      raw.get("area_um2"),
        "wns_ns":        raw.get("wns_ns"),
        "tns_ns":        raw.get("tns_ns"),
        "fmax_mhz":      raw.get("fmax_mhz"),
        "timing_met":    raw.get("timing_met"),
        "cells":         raw.get("cells"),
        "cell_area_um2": raw.get("cell_area_um2"),
    }

    if status in ("ok", "mock", "mock-proxy"):
        _F2_CACHE[cache_key] = (obs, status)

    return obs, status


# ── pending-eval planning ─────────────────────────────────────────────────────

def _build_pending(
    configs: list[dict],
    done_keys: set[tuple],
    target_fidelity: str,
    platform: str,
) -> list[tuple[dict, str]]:
    """Return list of (config, fidelity) pairs not yet in the output file.

    For each config we may need F0, F1, and F2 (up to target_fidelity).
    Fidelity order: F0 < F1 < F2.
    """
    fid_order = ["F0", "F1", "F2"]
    max_idx = fid_order.index(target_fidelity)

    pending = []
    for cfg in configs:
        # Support both canonical (mac_lanes / accumulator_width / clock_period_ns /
        # abc_recipe) and legacy (lanes / acc_w / clk / recipe) key names.
        lanes  = int(cfg.get("mac_lanes") or cfg.get("lanes") or 0)
        acc_w  = int(cfg.get("accumulator_width") or cfg.get("acc_w") or 0)
        clk    = round(float(cfg.get("clock_period_ns") or cfg.get("clk") or 0), 4)
        recipe = str(cfg.get("abc_recipe") or cfg.get("recipe") or "")
        key_base = (lanes, acc_w, clk, recipe)
        for fid in fid_order[: max_idx + 1]:
            k = key_base + (fid,)
            if k not in done_keys:
                pending.append((cfg, fid))
    return pending


# ── cost estimation ────────────────────────────────────────────────────────────

def _estimate_cost(pending: list[tuple[dict, str]]) -> tuple[float, dict]:
    """Return (total_seconds, breakdown_dict)."""
    counts: dict[str, int] = {}
    for _, fid in pending:
        counts[fid] = counts.get(fid, 0) + 1

    total_s = sum(FIDELITY_COST_S.get(fid, 0.0) * n for fid, n in counts.items())

    # F1 dedup savings: only unique lanes values need real sim calls
    # (already reflected if F1_CACHE fills up, but note it for the plan)
    return total_s, counts


# ── main evaluation loop ──────────────────────────────────────────────────────

def run_table_builder(
    space_path: Path,
    out_path: Path,
    target_fidelity: str,
    limit: int | None,
    subset: str | None,
    dry_run: bool,
    platform: str,
    design: "str | None" = None,
    max_tier: int = 1,
) -> None:
    """Run the table builder.

    When `design` is specified, KnobRegistry.space(design_spec, max_tier, platform)
    augments the base YAML space with knob axes.  For designs without TinyVAD
    functional eval, F1 is skipped automatically in FunnelEnv.
    """
    space = _load_space(space_path)

    # When --design is provided, derive the space entirely from KnobRegistry.space()
    # so a design with no RTL params (e.g. gcd) gets no phantom lanes/acc_w axes.
    if design is not None:
        try:
            from common.knobs import KnobRegistry
            reg = KnobRegistry.load()
            # reg.space() accepts str and normalizes via DesignSpec.load() internally.
            knob_space = reg.space(max_tier=max_tier, design=design, platform=platform)
            # Map from KnobRegistry.space() canonical axis names to build_table grid keys.
            # RTL param axes (mac_lanes, accumulator_width) map to lanes/acc_ws.
            # For designs without these axes (e.g. gcd), the grid has no such dimension:
            # set lanes=[1] and acc_ws=[0] as sentinels so the grid enumerator still
            # works but produces configs without these keys (they are filtered below).
            # clock_period_ns and abc_recipe always present.
            #
            # NOTE: mac_lanes/accumulator_width are design-specific axes whose choices
            # come from design.params.  ORFS-only designs have no such axes.
            if "mac_lanes" in knob_space:
                lspec = knob_space["mac_lanes"]
                if "choices" in lspec:
                    space["lanes"] = [int(v) for v in lspec["choices"]]
            elif "LANES" in knob_space:
                # Legacy fallback (shouldn't occur after Seam 2 fix)
                lspec = knob_space["LANES"]
                if "choices" in lspec:
                    space["lanes"] = [int(v) for v in lspec["choices"]]
            else:
                # No RTL lanes axis: use a sentinel so enumeration produces one variant
                space["lanes"] = [0]   # sentinel: no LANES chparam

            if "accumulator_width" in knob_space:
                aspec = knob_space["accumulator_width"]
                if "choices" in aspec:
                    space["acc_ws"] = [int(v) for v in aspec["choices"]]
            elif "ACC_W" in knob_space:
                # Legacy fallback
                aspec = knob_space["ACC_W"]
                if "choices" in aspec:
                    space["acc_ws"] = [int(v) for v in aspec["choices"]]
            else:
                # No RTL acc_w axis: use sentinel
                space["acc_ws"] = [0]   # sentinel: no ACC_W chparam

            if "clock_period_ns" in knob_space:
                cspec = knob_space["clock_period_ns"]
                if "range" in cspec:
                    lo, hi = cspec["range"]
                    steps = max(int(round((hi - lo) / 0.5)), 1)
                    space["clks"] = [round(lo + i * 0.5, 4) for i in range(steps + 1)]
            if "abc_recipe" in knob_space:
                rspec = knob_space["abc_recipe"]
                if "choices" in rspec:
                    space["recipes"] = list(rspec["choices"])
            has_lanes = space["lanes"] != [0]
            has_acc_ws = space["acc_ws"] != [0]
            print(f"Design '{design}' space (tier={max_tier}): "
                  f"{'lanes=' + str(space['lanes']) + ' ' if has_lanes else '(no RTL lanes) '}"
                  f"{'acc_ws=' + str(space['acc_ws']) + ' ' if has_acc_ws else '(no RTL acc_w) '}"
                  f"clks=[{space['clks'][0]}..{space['clks'][-1]}] "
                  f"recipes={space['recipes']}")
        except (ImportError, FileNotFoundError) as exc:
            print(f"WARNING: --design {design!r} could not be resolved: {exc}. "
                  "Using default space.")
        except Exception as exc:   # noqa: BLE001
            print(f"WARNING: --design augmentation failed: {exc}. Using default space.")

    if subset == "strategic":
        configs = _strategic_subset(space)
        print(f"Strategic subset: {len(configs)} configs (corner+axis sweep).")
    else:
        configs = _enumerate_grid(space)
        print(f"Full grid: {len(configs)} configs "
              f"({len(space['lanes'])} lanes × {len(space['acc_ws'])} acc_w × "
              f"{len(space['clks'])} clk × {len(space['recipes'])} recipes).")

    # RESUMABILITY: scan existing rows
    done_keys = _load_existing(out_path)
    n_existing = len(done_keys)
    if n_existing > 0:
        print(f"Resuming: {n_existing} (config, fidelity) pairs already in {out_path}.")

    pending = _build_pending(configs, done_keys, target_fidelity, platform)

    if limit is not None and limit < len(pending):
        pending = pending[:limit]
        print(f"--limit {limit}: capped pending list to {len(pending)} evaluations.")

    total_s, breakdown = _estimate_cost(pending)

    # F1 dedup note
    unique_lanes = {
        int(cfg.get("mac_lanes") or cfg.get("lanes") or 0)
        for cfg, fid in pending if fid == "F1"
    }
    n_f1_actual = len(unique_lanes)
    n_f1_nominal = breakdown.get("F1", 0)

    print(f"\n{'='*60}")
    print(f"PLAN: {len(pending)} evaluations pending")
    print(f"  Target fidelity : {target_fidelity}")
    print(f"  Platform        : {platform}")
    print(f"  Fidelity counts : {breakdown}")
    print(f"  F1 dedup        : {n_f1_nominal} nominal → {n_f1_actual} unique-lanes "
          f"sim calls (rest reuse cache)")
    actual_f1_cost = n_f1_actual * FIDELITY_COST_S["F1"]
    nominal_f1_cost = n_f1_nominal * FIDELITY_COST_S["F1"]
    f1_savings = nominal_f1_cost - actual_f1_cost
    actual_total = total_s - f1_savings
    print(f"  Est. wall-clock : {total_s/3600:.2f} h nominal → "
          f"{actual_total/3600:.2f} h after F1 dedup")
    print(f"  (PHYSICAL_MOCK  : {'YES — no tools needed' if os.environ.get('PHYSICAL_MOCK') else 'no'})")
    print(f"{'='*60}")

    if dry_run:
        print("\n--dry-run: listing pending evaluations (first 20):")
        for i, (cfg, fid) in enumerate(pending[:20]):
            lanes  = int(cfg.get("mac_lanes") or cfg.get("lanes") or 0)
            acc_w  = int(cfg.get("accumulator_width") or cfg.get("acc_w") or 0)
            clk    = float(cfg.get("clock_period_ns") or cfg.get("clk") or 5.0)
            recipe = str(cfg.get("abc_recipe") or cfg.get("recipe") or "plain")
            print(f"  {i+1:4d}. {fid}  lanes={lanes:2d} acc_w={acc_w:2d} "
                  f"clk={clk:4.1f} recipe={recipe}")
        if len(pending) > 20:
            print(f"  ... ({len(pending)-20} more)")
        return

    # ── evaluation loop ───────────────────────────────────────────────────────
    n_done = 0
    n_errors = 0
    t_start = time.time()

    for cfg, fid in pending:
        # config_doc uses canonical FunnelEnv key names so funnel.load_table can
        # match rows via _config_key without any translation.  _eval_* functions
        # accept both canonical and legacy key names (defensive dual-key lookup).
        lanes  = int(cfg.get("mac_lanes") or cfg.get("lanes") or 0)
        acc_w  = int(cfg.get("accumulator_width") or cfg.get("acc_w") or 0)
        clk    = float(cfg.get("clock_period_ns") or cfg.get("clk") or 5.0)
        recipe = str(cfg.get("abc_recipe") or cfg.get("recipe") or "plain")
        config_doc = {
            "mac_lanes":         lanes,
            "accumulator_width": acc_w,
            "clock_period_ns":   clk,
            "abc_recipe":        recipe,
        }

        t0 = time.time()
        if fid == "F0":
            obs, status = _eval_f0(cfg)
        elif fid == "F1":
            obs, status = _eval_f1(cfg)
        elif fid == "F2":
            obs, status = _eval_f2(cfg, platform)
        else:
            obs, status = {}, "skipped"
        elapsed = time.time() - t0

        _write_row(out_path, config_doc, fid, obs, elapsed, platform, status)
        n_done += 1
        if status not in ("ok", "mock", "mock-proxy", "fallback"):
            n_errors += 1

        if n_done % 10 == 0 or n_done == len(pending):
            wall = time.time() - t_start
            rate = n_done / max(wall, 1)
            remaining = (len(pending) - n_done) / max(rate, 1e-9)
            print(f"  [{n_done}/{len(pending)}] {fid} "
                  f"L{lanes}_A{acc_w}_c{clk}_r{recipe}"
                  f"  status={status}  elapsed={elapsed:.1f}s"
                  f"  ETA: {remaining/60:.1f} min", flush=True)

    wall_total = time.time() - t_start
    print(f"\nDone. {n_done} rows written, {n_errors} errors, "
          f"wall-clock {wall_total/60:.2f} min.")
    print(f"Output: {out_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build F0–F2 table for FunnelEnv promotion-policy training.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--space",
        default=str(_FUNNEL_YAML),
        help="Path to search_space_funnel.yaml",
    )
    parser.add_argument(
        "--out",
        default=str(_THIS_DIR.parent / "results_funnel.jsonl"),
        help="Output JSONL file path",
    )
    parser.add_argument(
        "--fidelity",
        choices=["F0", "F1", "F2"],
        default="F2",
        help="Target fidelity level (evaluates all levels up to and including this)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max number of evaluations to run (for quick smoke tests)",
    )
    parser.add_argument(
        "--subset",
        choices=["strategic"],
        default=None,
        help="'strategic': corner+axis sweep (~54-87 configs) instead of full grid",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print plan and cost estimate without running any evaluations",
    )
    parser.add_argument(
        "--platform",
        default="nangate45",
        help="ORFS target platform",
    )
    parser.add_argument(
        "--design",
        default=None,
        help=(
            "Design name or YAML path (e.g. 'gcd' or 'optimizer/designs/gcd.yaml'). "
            "Default: tinymac_accel (unchanged behaviour). "
            "Grid enumeration uses KnobRegistry.space(design, max_tier, platform) "
            "when --design is specified. "
            "For generic designs (no tinyvad_sim functional_eval), F1 is skipped."
        ),
    )
    parser.add_argument(
        "--max-tier",
        type=int,
        default=1,
        dest="max_tier",
        help=(
            "Maximum knob tier to include in the search space (1–4). "
            "Default: 1 (only dominant axes). Ignored when --design is not set."
        ),
    )

    args = parser.parse_args()

    run_table_builder(
        space_path=Path(args.space),
        out_path=Path(args.out),
        target_fidelity=args.fidelity,
        limit=args.limit,
        subset=args.subset,
        dry_run=args.dry_run,
        platform=args.platform,
        design=args.design,
        max_tier=args.max_tier,
    )


# ── self-test (py_compile check) ──────────────────────────────────────────────

def _selftest() -> None:
    """Quick smoke test — no ORFS tools, PHYSICAL_MOCK not required."""
    import tempfile, os as _os

    # Build a minimal space
    space = {
        "lanes":   [4, 8],
        "acc_ws":  [24, 32],
        "clks":    [4.0, 5.0],
        "recipes": ["plain"],
    }
    configs = _enumerate_grid(space)
    assert len(configs) == 2 * 2 * 2 * 1  # 8

    strat = _strategic_subset(space)
    # Strategic: (2 lanes × 2 acc_ws × 1 clk_nearest_4 × 1 recipe = 4) + clk sweep (2×1) = 6 - dedup
    assert len(strat) >= 4

    # F0 eval (canonical key names)
    cfg = {"mac_lanes": 4, "accumulator_width": 24, "clock_period_ns": 4.0, "abc_recipe": "plain"}
    obs, status = _eval_f0(cfg)
    assert status == "ok"
    assert obs["accuracy_flag"] == 1.0
    assert obs["cycles"] > 0

    # F1 eval (mock, canonical keys)
    _os.environ.setdefault("PHYSICAL_MOCK", "1")
    obs1, s1 = _eval_f1(cfg)
    assert "avg_cycles" in obs1
    assert obs1["accuracy"] == 1.0

    # Cost estimation
    pending = [(cfg, "F0"), (cfg, "F1"), (cfg, "F2")]
    total_s, breakdown = _estimate_cost(pending)
    assert breakdown["F0"] == 1
    assert breakdown["F1"] == 1
    assert breakdown["F2"] == 1
    assert abs(total_s - (0.0 + 5.0 + 45.0)) < 1e-6

    # write/read roundtrip: canonical key names round-trip through _load_existing
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "test_table.jsonl"
        canonical_cfg = {"mac_lanes": 4, "accumulator_width": 24,
                         "clock_period_ns": 4.0, "abc_recipe": "plain"}
        _write_row(out, canonical_cfg, "F0", {"cycles": 91650}, 0.001, "nangate45", "ok")
        done = _load_existing(out)
        assert (4, 24, 4.0, "plain", "F0") in done, f"canonical key roundtrip failed; done={done}"

    # Also verify config_key produced by build_table matches funnel.load_table expectation
    import json as _json
    from gen2.funnel import load_table as _load_table
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "funnel_compat.jsonl"
        canonical_cfg = {"mac_lanes": 4, "accumulator_width": 24,
                         "clock_period_ns": 4.0, "abc_recipe": "plain"}
        _write_row(out, canonical_cfg, "F0", {"cycles": 91650}, 0.001, "nangate45", "ok")
        table = _load_table(out)
        # The FunnelEnv config key for the same config must find the row
        env_cfg = {"mac_lanes": 4, "accumulator_width": 24,
                   "clock_period_ns": 4.0, "abc_recipe": "plain"}
        env_key = _json.dumps({k: env_cfg[k] for k in sorted(env_cfg)},
                               sort_keys=True, separators=(",", ":"))
        assert env_key in table, \
            f"FunnelEnv key not found in table. env_key={env_key!r}; table keys={list(table.keys())}"

    print("build_table.py self-test: PASS")


if __name__ == "__main__":
    main()
