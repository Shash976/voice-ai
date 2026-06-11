"""test_cascade.py — offline (mock) tests for the multi-fidelity funnel.

Runs with PHYSICAL_MOCK so NO tools (Verilator/Yosys/OpenROAD) are needed — it
exercises the funnel control flow, the validator, the gate thresholds, the
stage-aware reward, and a short end-to-end agent loop. Real metrics are checked
on the VM; this proves the logic.

Run:  PHYSICAL_MOCK=1 python optimizer/test_cascade.py
      (the script sets PHYSICAL_MOCK itself if unset)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("PHYSICAL_MOCK", "1")
import pathlib as _pl
sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))

import yaml

from gen1.cascade import STAGE_ORDER, evaluate
from common.cascade_reward import compute_cascade_reward
from common.validate import validate

SPACE_PATH = Path(__file__).parent / "search_space_full.yaml"
with open(SPACE_PATH, encoding="utf-8") as f:
    _RAW = yaml.safe_load(f)
SPACE = {**_RAW.get("sim_params", {}), **_RAW.get("proxy_params", {})}
CONSTRAINTS = _RAW.get("constraints", [])
GATES = _RAW.get("gates", {})
REWARD = _RAW.get("reward", {})

_n_pass = 0
_n_fail = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global _n_pass, _n_fail
    if cond:
        _n_pass += 1
        print(f"  PASS  {name}")
    else:
        _n_fail += 1
        print(f"  FAIL  {name}  {detail}")


def good_config(**over) -> dict:
    cfg = {"mac_lanes": 4, "accumulator_width": 24, "clock_period_ns": 2.0,
           "core_utilization": 40, "place_density": 0.60, "abc_strategy": "speed"}
    cfg.update(over)
    return cfg


def run(cfg, max_stage="full"):
    return evaluate(cfg, space=SPACE, constraints=CONSTRAINTS, gates=GATES,
                    platform="nangate45", max_stage=max_stage)


def reward(res):
    return compute_cascade_reward(res, REWARD)["reward"]


# ── 1. validator ──────────────────────────────────────────────────────────────
print("\n[1] validator")
ok, _ = validate(good_config(), SPACE, CONSTRAINTS)
check("valid config accepted", ok)

ok, why = validate(good_config(place_density=0.99), SPACE, CONSTRAINTS)
check("illegal value rejected (place_density not in choices)", not ok, why)

# cross-param infeasible corner: util=70 AND density=0.75
ok, why = validate(good_config(core_utilization=70, place_density=0.75), SPACE, CONSTRAINTS)
check("infeasible util x density corner rejected by constraint", not ok, why)

ok, why = validate({"mac_lanes": 4}, SPACE, CONSTRAINTS)
check("missing params rejected", not ok, why)


# ── 2. funnel short-circuits at the right gate ─────────────────────────────────
print("\n[2] funnel short-circuit")
r = run(good_config(place_density=0.99))
check("invalid config dies at validate", r["reached"] == "validate" and r["failed_stage"] == "validate")

# acc_width=16 → mock sim accuracy 47/64 < 0.95 → dies at sim gate
r = run(good_config(accumulator_width=16))
check("acc_width=16 dies at sim gate", r["failed_stage"] == "sim", r.get("reason"))
check("  ...elaborate passed before sim", r["stages"]["elaborate"]["ok"])

# good config reaches full
r = run(good_config())
check("good config reaches full", r["reached"] == "full" and r["ok"], r.get("reason"))
check("  ...full produced area+fmax", r["metrics"].get("area_um2") and r["metrics"].get("fmax_mhz"))

# proxy area gate: force tiny cap to prove the proxy gate fires
tight_gates = {**GATES, "proxy": {"max_area_um2": 1.0}}
r = evaluate(good_config(mac_lanes=16), space=SPACE, constraints=CONSTRAINTS,
             gates=tight_gates, platform="nangate45", max_stage="full")
check("config rejected at proxy when area cap is tiny", r["failed_stage"] == "proxy", r.get("reason"))


# ── 3. max_stage cap ───────────────────────────────────────────────────────────
print("\n[3] max_stage cap")
r = run(good_config(), max_stage="proxy")
check("max_stage=proxy stops at proxy", r["reached"] == "proxy" and r["ok"])
check("  ...no full metrics gathered (gds None)", not r["metrics"].get("gds"))


# ── 4. reward ordering: deeper success scores higher than earlier death ────────
print("\n[4] reward ordering")
r_full    = reward(run(good_config()))
r_proxy   = reward(evaluate(good_config(mac_lanes=16), space=SPACE, constraints=CONSTRAINTS,
                            gates={**GATES, "proxy": {"max_area_um2": 1.0}},
                            platform="nangate45", max_stage="full"))
r_sim     = reward(run(good_config(accumulator_width=16)))
r_invalid = reward(run(good_config(place_density=0.99)))
print(f"    full={r_full:.2f}  proxy-death={r_proxy:.2f}  sim-death={r_sim:.2f}  invalid={r_invalid:.2f}")
check("full success > proxy death", r_full > r_proxy)
check("proxy death > sim death", r_proxy > r_sim)
check("sim death > invalid", r_sim > r_invalid)
check("full success reward is positive", r_full > 0)


# ── 5. end-to-end agent loop (evo) over the big space ──────────────────────────
print("\n[5] end-to-end optimizer loop (mock, evo)")
from gen1.agents.evo_agent import EvoAgent
from gen1.cascade_env import CascadeOptEnv

env = CascadeOptEnv(platform="nangate45", max_stage="full")
env.clear_results()
agent = EvoAgent(env.search_space)
state = env.reset()
n_full = 0
for _ in range(25):
    cfg = agent.suggest(state, env.history)
    state, rew, _done, info = env.step(cfg)
    agent.update(cfg, rew, info)
    if info.get("reached") == "full" and not info.get("failed_stage"):
        n_full += 1
check("loop completed 25 trials", len(env.history) == 25)
check("at least one config reached full P&R", n_full >= 1, f"n_full={n_full}")
best = max(env.history, key=lambda r: r["reward"])
check("best config reached full", best.get("reached") == "full", f"best reached {best.get('reached')}")
check("results file written", env._results_file.exists())
env.clear_results()  # cleanup mock artifacts


# ── summary ────────────────────────────────────────────────────────────────────
print("\n" + "-" * 50)
print(f"  {_n_pass} passed, {_n_fail} failed")
sys.exit(1 if _n_fail else 0)
