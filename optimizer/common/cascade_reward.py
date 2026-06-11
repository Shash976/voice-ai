"""cascade_reward.py — turn a funnel result into a scalar reward.

Configs that die early get a flat stage penalty (escalating: the cheaper the
rejection, the more negative — so the agent prefers configs that at least make
progress, while a wasted full-P&R failure is scored on its real poor metrics).
Configs that survive to the proxy or full stage are scored with the SAME
multi-objective reward as the other tracks (physical_reward.compute_physical_reward),
so numbers are comparable across optimizers.

All weights AND stage penalties come from the search space's `reward:` block, so
retuning the funnel is data-driven — edit the YAML, no code change.
"""

from __future__ import annotations

from common.physical_reward import compute_physical_reward

# V4: monotone penalty ladder — more information gained = less negative penalty.
# invalid −100 < elaborate −80 < sim −60 < proxy −40 < full-flow-fail −20
# 'full' failure (deepest stage reached) must be LESS negative than proxy failure
# so the agent learns that dying at P&R is better than dying at elaboration.
_DEFAULT_PENALTY = {"invalid": -100.0, "elaborate": -80.0, "sim": -60.0,
                    "proxy": -40.0, "full": -20.0}


def compute_cascade_reward(result: dict, weights: dict | None = None) -> dict:
    """Return {'reward': float, 'reached': str, ...} for a cascade.evaluate result."""
    w = weights or {}
    penalty = {**_DEFAULT_PENALTY, **(w.get("stage_penalty") or {})}
    reached = result.get("reached")
    failed = result.get("failed_stage")

    # ── Early rejections: flat penalty by the gate that killed it ──────────────
    if failed == "validate":
        return _early(penalty["invalid"], "validate", result)
    if failed == "elaborate":
        return _early(penalty["elaborate"], "elaborate", result)
    if failed == "sim":
        return _early(penalty["sim"], "sim", result)
    if failed == "proxy":
        return _early(penalty["proxy"], "proxy", result)
    # V4: full-flow failure (reached P&R but the flow/parse failed) scores −20
    # — better than proxy failure (−40) because more information was gathered.
    # This prevents the fall-through to compute_physical_reward which would score
    # −100 on missing metrics, inverting the penalty ladder.
    if failed == "full":
        return _early(penalty["full"], "full", result)

    # ── Survived to proxy/full (or full failed): score on real metrics ─────────
    metrics = dict(result.get("metrics") or {})
    sim = result.get("sim") or {}
    cycles = sim.get("avg_cycles")

    # Ensure compute_physical_reward has the keys it needs even for the rare
    # debug modes that stop at sim/elaborate (no physical metrics gathered).
    if "acc_w" not in metrics:
        cfg = result.get("config") or {}
        metrics.setdefault("lanes", cfg.get("mac_lanes"))
        metrics.setdefault("acc_w", cfg.get("accumulator_width"))
        metrics.setdefault("clk_ns", cfg.get("clock_period_ns"))
        metrics.setdefault("status", "ok")

    scored = compute_physical_reward(metrics, w, cycles=cycles)
    scored["reached"] = reached
    scored["sim_accuracy"] = sim.get("accuracy")
    return scored


def _early(reward: float, stage: str, result: dict) -> dict:
    return {
        "reward": float(reward),
        "reached": stage,
        "rejected_at": stage,
        "reason": result.get("reason", ""),
        "real_speedup": 0.0,
        "norm_speedup": -1.0,
        "accuracy": (result.get("sim") or {}).get("accuracy", 0.0),
        "timing_violation": True,
        "infeasible": True,
    }
