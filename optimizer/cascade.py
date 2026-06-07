"""cascade.py — the multi-fidelity evaluation FUNNEL.

A config is pushed through gates ordered cheapest → most expensive. Each gate can
reject the config; rejection short-circuits the rest, so expensive place-and-route
only ever runs on configs that already survived everything cheaper.

    validate   (µs)   pure-Python legality + declarative constraints   → validate.py
       │ pass
    elaborate  (~s)   Yosys reads the parameterised RTL & checks it     → run_elaborate
       │ pass
    sim        (~s)   Verilator behavioural run: real correctness +     → runner.run_sim
       │ pass         cycle count (kills functionally-wrong configs)
    proxy      (s–m)  Yosys synth + OpenROAD STA: gate area + Fmax      → run_synth_sta
       │ pass         (no P&R) — coarse area/speed screen
    full       (min)  full RTL→GDS: real area / routed timing / power   → run_physical
       ▼
    metrics + reached_stage  →  cascade_reward.compute_cascade_reward

Every tool stage is wrapped so a tool error becomes a clean gate failure (the
agent learns to avoid that region) instead of crashing the optimizer. Set
PHYSICAL_MOCK=1 to run the whole funnel with synthetic-but-plausible numbers
(no tools needed) — used by the offline tests and on machines without OpenROAD.
"""

from __future__ import annotations

import os

from physical_runner import run_elaborate, run_physical, run_synth_sta
from validate import validate

STAGE_ORDER = ["validate", "elaborate", "sim", "proxy", "full"]


# ── sim stage (mock-aware wrapper around the Verilator runner) ─────────────────

def _run_sim(lanes: int, acc_w: int) -> dict:
    """Behavioural Verilator run → {accuracy, correct, n_total, avg_cycles}.
    Under PHYSICAL_MOCK, returns numbers shaped like the real measured sweep:
    acc_width < 24 loses accuracy (int16 ≈ 47/64), matching the empirical finding.
    """
    if os.environ.get("PHYSICAL_MOCK"):
        if acc_w >= 24:
            acc = 1.0
        elif acc_w >= 20:
            acc = 0.92
        else:                       # int16 — overflow, real measured 47/64
            acc = 47.0 / 64.0
        cyc = 28_000.0 + 242_000.0 / max(lanes, 1)
        return {"accuracy": acc, "correct": round(acc * 64), "n_total": 64,
                "avg_cycles": cyc}
    from runner import run_sim       # imported lazily: needs the Verilator binary
    return run_sim(lanes, acc_w)


# ── the funnel ────────────────────────────────────────────────────────────────

def _fail(result: dict, stage: str, reason: str) -> dict:
    result["reached"] = stage
    result["failed_stage"] = stage
    result["ok"] = False
    result["reason"] = reason
    return result


def evaluate(
    config: dict,
    *,
    space: dict,
    constraints: list[str],
    gates: dict,
    platform: str = "nangate45",
    max_stage: str = "full",
) -> dict:
    """Push one config through the funnel. Returns a result dict:

        reached       : the furthest stage attempted (the stage it died at, or
                        'full'/max_stage if it survived)
        ok            : True iff it passed every gate up to max_stage
        failed_stage  : the gate it failed, or None
        reason        : human-readable failure reason ('' on success)
        stages        : {stage: {ok, ...}} per-stage detail
        sim           : the sim metrics dict (avg_cycles drives the reward), or None
        metrics       : merged physical metrics (area/fmax/power/timing/gds)
    """
    lanes   = int(config["mac_lanes"])
    acc_w   = int(config["accumulator_width"])
    clk     = float(config["clock_period_ns"])
    util    = int(config.get("core_utilization", 40))
    density = float(config.get("place_density", 0.60))
    abc     = config.get("abc_strategy")
    if max_stage not in STAGE_ORDER:
        raise ValueError(f"max_stage must be one of {STAGE_ORDER}, got {max_stage!r}")
    stop_at = STAGE_ORDER.index(max_stage)

    result: dict = {
        "reached": None, "ok": False, "failed_stage": None, "reason": "",
        "stages": {}, "sim": None, "metrics": {},
        "config": dict(config), "platform": platform,
    }

    # ── 1. validate ────────────────────────────────────────────────────────────
    ok, reason = validate(config, space, constraints)
    result["stages"]["validate"] = {"ok": ok, "reason": reason}
    if not ok:
        return _fail(result, "validate", reason)
    result["reached"] = "validate"
    if stop_at == STAGE_ORDER.index("validate"):
        result["ok"] = True
        return result

    # ── 2. elaborate ───────────────────────────────────────────────────────────
    try:
        el = run_elaborate(lanes, acc_w, platform)
    except Exception as exc:  # noqa: BLE001 — tool error = gate failure, not a crash
        result["stages"]["elaborate"] = {"ok": False, "error": str(exc)}
        return _fail(result, "elaborate", f"elaborate error: {exc}")
    result["stages"]["elaborate"] = el
    if not el.get("ok"):
        return _fail(result, "elaborate", "RTL did not elaborate")
    result["reached"] = "elaborate"
    if stop_at == STAGE_ORDER.index("elaborate"):
        result["ok"] = True
        return result

    # ── 3. sim (functional correctness + cycles) ───────────────────────────────
    try:
        sim = _run_sim(lanes, acc_w)
    except Exception as exc:  # noqa: BLE001
        result["stages"]["sim"] = {"ok": False, "error": str(exc)}
        return _fail(result, "sim", f"sim error: {exc}")
    result["sim"] = sim
    min_acc = float(gates.get("sim", {}).get("min_accuracy", 0.95))
    acc_ok = sim.get("accuracy", 0.0) >= min_acc
    result["stages"]["sim"] = {"ok": acc_ok, "accuracy": sim.get("accuracy"),
                               "avg_cycles": sim.get("avg_cycles")}
    if not acc_ok:
        return _fail(result, "sim",
                     f"accuracy {sim.get('accuracy'):.3f} < {min_acc} (acc_width too narrow)")
    result["reached"] = "sim"
    if stop_at == STAGE_ORDER.index("sim"):
        result["ok"] = True
        return result

    # ── 4. proxy (synth + STA: gate area / Fmax) ───────────────────────────────
    try:
        prox = run_synth_sta(lanes, acc_w, clk, platform)
    except ValueError as exc:
        # No proxy lib wired for this platform (e.g. asap7) — skip the screen and
        # let the full flow be the judge, rather than failing a valid config.
        result["stages"]["proxy"] = {"ok": True, "skipped": True, "note": str(exc)}
        prox = None
    except Exception as exc:  # noqa: BLE001
        result["stages"]["proxy"] = {"ok": False, "error": str(exc)}
        return _fail(result, "proxy", f"proxy error: {exc}")

    if prox is not None:
        if prox.get("status") == "FAIL":
            result["stages"]["proxy"] = {"ok": False, "reason": "synthesis failed"}
            return _fail(result, "proxy", "synthesis failed")
        result["metrics"].update(prox)
        gcfg = gates.get("proxy", {})
        max_area = gcfg.get("max_area_um2")
        area = prox.get("area_um2")
        if max_area is not None and area is not None and area > float(max_area):
            result["stages"]["proxy"] = {"ok": False, "area_um2": area,
                                         "reason": f"area {area:.0f} > {max_area}"}
            return _fail(result, "proxy", f"proxy area {area:.0f} > {max_area} µm²")
        if gcfg.get("require_timing_met") and prox.get("timing_met") is False:
            result["stages"]["proxy"] = {"ok": False, "reason": "pre-layout timing failed"}
            return _fail(result, "proxy", "pre-layout timing failed")
        result["stages"]["proxy"] = {"ok": True, "area_um2": area,
                                     "fmax_mhz": prox.get("fmax_mhz")}
    result["reached"] = "proxy"
    if stop_at == STAGE_ORDER.index("proxy"):
        result["ok"] = True
        return result

    # ── 5. full RTL→GDS ────────────────────────────────────────────────────────
    try:
        full = run_physical(lanes, acc_w, clk, platform, util, density, abc)
    except Exception as exc:  # noqa: BLE001
        result["stages"]["full"] = {"ok": False, "error": str(exc)}
        return _fail(result, "full", f"full-flow error: {exc}")
    result["metrics"].update(full)
    result["reached"] = "full"
    if full.get("status") == "FAIL":
        result["stages"]["full"] = {"ok": False, "reason": "flow failed / no report"}
        result["failed_stage"] = "full"
        result["ok"] = False
        result["reason"] = "full flow failed"
        return result
    result["stages"]["full"] = {"ok": True, "area_um2": full.get("area_um2"),
                                "fmax_mhz": full.get("fmax_mhz"),
                                "power_mw": full.get("power_mw"),
                                "timing_met": full.get("timing_met")}
    result["ok"] = True
    return result
