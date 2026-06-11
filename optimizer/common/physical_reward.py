"""physical_reward.py — multi-objective reward over REAL ORFS physical metrics.

The Stage-6 analogue of reward.py's compute_reward.  Same shape and the same
weights (from search_space.yaml's `reward:` block) so results are comparable —
but the speedup, area and power terms are driven by *measured* numbers from the
OpenROAD flow instead of analytical proxies:

    speedup : behavioral cycle count (lanes) × the REAL achieved clock period
              (the slower of the requested clock and 1000/Fmax — you can't beat
              the routed critical path).  Fmax comes from ORFS.
    area    : real Design area (µm²), normalised to the LANES=4 reference.
    power   : real total power (mW), normalised to the LANES=4 reference.
    timing  : real timing_met flag (setup violation count == 0).

accuracy is not measured by the physical flow — the RTL is already bit-exact
verified (rtl/tb).  The only way a config is functionally wrong is an
accumulator too narrow for TinyVAD, which reward.acc_overflows() detects
analytically (acc_width=16 overflows); we reuse it so acc=16 is penalised.
"""

from __future__ import annotations

import math

from gen1.reward import SW_BASELINE_LATENCY_NS, acc_overflows
# Single source of truth: cycle model constants come from constants.py.
# Measured 2026-06-10 after V13 saturation-order fix.
# Old values: _CYCLE_OVERHEAD=28000, _CYCLE_MAC_WORK=242000 (synthetic fit).
# New values: fit to measured AVG_CYCLES over lanes {1,2,4,8,16,32}.
from common.constants import _CYCLE_OVERHEAD, _CYCLE_MAC_WORK

# Real nangate45 LANES=4 ACC_W=24 anchors (the first full GDS): area/power terms
# are normalised to ~1.0 here so the YAML weights carry over from the proxy reward.
AREA_REF_UM2 = 19_738.0
POWER_REF_MW = 1_020.0


def behavioral_cycles(lanes: int) -> float:
    return _CYCLE_OVERHEAD + _CYCLE_MAC_WORK / max(lanes, 1)


def achieved_period_ns(clk_ns: float, fmax_mhz: float | None) -> float:
    """Period the silicon actually runs at: the slower of the requested clock
    and the routed critical-path period (1000/Fmax).  An over-aggressive clock
    request gives no free speed — exactly reward.effective_clock_ns, but using
    the MEASURED Fmax instead of the analytical critical-path estimate.

    V2: if fmax_mhz is None (parse failed or flow didn't produce it), return a
    very large period so the speedup term is essentially 0, not a fallback to the
    requested clock which would silently award speed from zero physical data."""
    if fmax_mhz is None:
        return float("inf")          # no measurement → zero speedup; never award free speed
    crit = 1000.0 / fmax_mhz
    return max(float(clk_ns), crit)


def physical_real_speedup(metrics: dict, cycles: float | None = None) -> float:
    """Frequency-aware speedup over the Stage-3 SW baseline, using real Fmax."""
    cyc = cycles if cycles is not None else behavioral_cycles(metrics["lanes"])
    period = achieved_period_ns(metrics["clk_ns"], metrics.get("fmax_mhz"))
    latency_ns = max(cyc, 1.0) * period
    return SW_BASELINE_LATENCY_NS / max(latency_ns, 1e-9)


def compute_physical_reward(
    metrics: dict,
    weights: dict | None = None,
    cycles: float | None = None,
) -> dict:
    """Return {'reward': float, ...derived fields} for a physical metrics dict.

    Mirrors reward.compute_reward's weighting so the two tracks are comparable.
    Returns a dict (not a bare float) so the env can log the derived speedup /
    normalised terms without recomputing them.
    """
    w = weights or {}
    w_acc    = w.get("w_accuracy",          2.0)
    w_spd    = w.get("w_speedup",           3.0)
    w_area   = w.get("w_area",             -0.4)
    w_pwr    = w.get("w_power",            -0.4)
    w_tv     = w.get("w_timing_violation", -3.0)
    max_spd  = w.get("max_speedup",        576.0)
    min_spd  = w.get("min_useful_speedup",  10.0)
    perf_pen = w.get("perf_floor_penalty",  -8.0)

    # V2: ANY non-ok status (FAIL, PARSE_FAIL, TIMEOUT, mock-proxy-fail, …) is
    # treated as the worst-case outcome.  We never substitute reference values
    # for missing measurements — that would silently award reward from no data.
    status = metrics.get("status", "ok")
    if status not in ("ok", "mock", "mock-proxy"):
        return {"reward": -100.0, "real_speedup": 0.0, "norm_speedup": -1.0,
                "accuracy": 0.0, "timing_violation": True, "infeasible": True,
                "status": status}

    accuracy = 0.0 if acc_overflows({"accumulator_width": metrics["acc_w"]}) else 1.0
    spd      = physical_real_speedup(metrics, cycles)

    if spd > 0 and max_spd > 1:
        norm_spd = math.log2(max(spd, 1e-3)) / math.log2(max_spd)
        norm_spd = max(-1.0, min(1.0, norm_spd))
    else:
        norm_spd = -1.0

    # V2: do NOT substitute AREA_REF / POWER_REF when measurements are missing.
    # Missing area/power means the flow didn't complete — treat as infeasible.
    area_um2 = metrics.get("area_um2")
    power_mw = metrics.get("power_mw")
    if area_um2 is None or power_mw is None:
        return {"reward": -100.0, "real_speedup": 0.0, "norm_speedup": -1.0,
                "accuracy": 0.0, "timing_violation": True, "infeasible": True,
                "status": "PARSE_FAIL"}

    area  = area_um2 / AREA_REF_UM2
    power = power_mw / POWER_REF_MW
    t_viol = 0.0 if metrics.get("timing_met", True) else 1.0

    floor_penalty = perf_pen if spd < min_spd else 0.0
    correctness   = -50.0 * (1.0 - accuracy)

    reward = (
        w_acc * accuracy
        + w_spd * norm_spd
        + w_area * area
        + w_pwr * power
        + w_tv * t_viol
        + correctness
        + floor_penalty
    )
    return {
        "reward":           round(reward, 4),
        "real_speedup":     round(spd, 3),
        "norm_speedup":     round(norm_spd, 4),
        "accuracy":         accuracy,
        "area_norm":        round(area, 4),
        "power_norm":       round(power, 4),
        "timing_violation": bool(t_viol),
        "infeasible":       False,
    }
