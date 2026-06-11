#!/usr/bin/env python3
"""test_reward_sanity.py — standalone conceptual checks for the frequency-coupled
reward rewrite (Stage-5 fix, 2026-06-02).

Runs on WINDOWS without the Verilator sim: it uses a SYNTHETIC sim model that
matches the real sim's behaviour (avg_cycles depends only on lanes; accuracy
only on acc_width).  It exercises reward.py / search_space.yaml end-to-end and
asserts the conceptual invariants the rewrite must satisfy.

Usage:
    python optimizer/test_reward_sanity.py

Dependencies: stdlib + pyyaml + the optimizer's reward.py.
"""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

import yaml

# Make optimizer/ importable when run as: python optimizer/test_reward_sanity.py
import pathlib as _pl
sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))

from gen1 import reward as R  # noqa: E402


# ── Synthetic sim model (matches real sim behaviour) ──────────────────────────
# avg_cycles decreases with lanes, INDEPENDENT of clock/buffers.
# accuracy depends only on accumulator width (int16 overflows TinyVAD).

def synth_avg_cycles(lanes: int) -> float:
    return 28000 + 242000 / lanes


def synth_accuracy(acc_width: int) -> float:
    return 1.0 if acc_width >= 24 else 0.734


def synth_sim_metrics(lanes: int, acc_width: int) -> dict:
    avg = synth_avg_cycles(lanes)
    return {
        "mac_lanes":  lanes,
        "acc_width":  acc_width,
        "avg_cycles": avg,
        "accuracy":   synth_accuracy(acc_width),
        # cycle-based speedup (frequency-INDEPENDENT) — what the sim still returns
        "speedup":    R.SW_BASELINE_CYCLES / max(avg, 1),
    }


def evaluate(config: dict, reward_cfg: dict) -> dict:
    """Mirror env.step: sim → proxies → merge real_speedup → reward."""
    lanes = int(config["mac_lanes"])
    acc_w = int(config.get("accumulator_width", 32))

    sim = synth_sim_metrics(lanes, acc_w)
    proxies = R.compute_proxies(config)

    avg_cycles = sim["avg_cycles"]
    proxies["real_speedup"] = round(R.real_speedup(config, avg_cycles), 3)
    proxies["latency_ns"]   = round(avg_cycles * proxies["effective_clock_ns"], 1)

    rew = R.compute_reward(sim, proxies, reward_cfg)
    return {"config": config, "sim": sim, "proxies": proxies, "reward": rew}


# ── Search space ──────────────────────────────────────────────────────────────

def load_space() -> tuple[list[dict], dict]:
    raw = yaml.safe_load(
        (Path(__file__).parent / "search_space.yaml").read_text(encoding="utf-8"))
    sim_specs   = raw.get("sim_params", {})
    proxy_specs = raw.get("proxy_params", {})
    reward_cfg  = raw.get("reward", {})

    # Buffer axes must be gone from the active search.
    assert "input_buffer_bytes"  not in proxy_specs, "input_buffer_bytes still active!"
    assert "weight_buffer_bytes" not in proxy_specs, "weight_buffer_bytes still active!"

    space = {**sim_specs, **proxy_specs}
    names = list(space.keys())
    choices = [space[n]["choices"] for n in names]
    configs = [dict(zip(names, combo)) for combo in itertools.product(*choices)]
    return configs, reward_cfg


# ── Assertions ────────────────────────────────────────────────────────────────

_passes = 0
_fails  = 0


def check(label: str, cond: bool, detail: str = "") -> None:
    global _passes, _fails
    status = "PASS" if cond else "FAIL"
    if cond:
        _passes += 1
    else:
        _fails += 1
    line = f"  [{status}] {label}"
    if detail:
        line += f"\n         {detail}"
    print(line)


def main() -> int:
    configs, reward_cfg = load_space()
    print(f"Loaded {len(configs)} configs from search_space.yaml "
          f"(params: {', '.join(configs[0].keys())})")
    print(f"max_speedup={reward_cfg.get('max_speedup')}  "
          f"min_useful_speedup={reward_cfg.get('min_useful_speedup')}")
    print()

    # ── (a) Faster clock helps; sub-critical-path clock is capped ─────────────
    print("(a) Fixed lanes=8, acc=24 — faster clock raises real_speedup, capped below crit:")
    base = {"mac_lanes": 8, "accumulator_width": 24}
    r5  = evaluate({**base, "clock_period_ns": 5},  reward_cfg)
    r10 = evaluate({**base, "clock_period_ns": 10}, reward_cfg)
    r20 = evaluate({**base, "clock_period_ns": 20}, reward_cfg)
    rs5, rs10, rs20 = (r5["proxies"]["real_speedup"],
                       r10["proxies"]["real_speedup"],
                       r20["proxies"]["real_speedup"])
    check("real_speedup(clk=5) > real_speedup(clk=10) > real_speedup(clk=20)",
          rs5 > rs10 > rs20,
          f"clk5={rs5}  clk10={rs10}  clk20={rs20}")
    check("reward(clk=5) > reward(clk=10) > reward(clk=20)  [faster clock now wins]",
          r5["reward"] > r10["reward"] > r20["reward"],
          f"clk5={r5['reward']}  clk10={r10['reward']}  clk20={r20['reward']}")

    # The GDS-calibrated path is LANES-independent ≈3.72 ns (at 24b).  The sim
    # grid's fastest clock is 5 ns > crit, so nothing is capped on-grid — which
    # is correct (200 MHz < 269 MHz Fmax).  Verify capping off-grid where it
    # bites: an aggressive 2 ns request (< crit) must clamp up to the crit path.
    cap_base = {"mac_lanes": 16, "accumulator_width": 24}
    crit_16  = R.critical_path_ns(cap_base)
    rc2 = evaluate({**cap_base, "clock_period_ns": 2},  reward_cfg)
    eff2 = rc2["proxies"]["effective_clock_ns"]
    check("clk below critical path is capped: effective_clock == critical_path",
          abs(eff2 - round(crit_16, 3)) < 1e-6,
          f"requested clk=2  crit={crit_16:.3f}  effective={eff2}")

    print()

    # ── (b) clk < crit does NOT out-reward clk == crit (same config) ──────────
    print("(b) Requesting clk < critical_path never beats clk = critical_path:")
    # crit(16,24) ≈ 3.72 ns.  Compare an aggressive clk=2 ns (< crit → violation)
    # against clk = crit (the fastest legal clock).  Both are off the {5,10,20}
    # grid; we synthesise them only to exercise the invariant.
    viol = evaluate({"mac_lanes": 16, "accumulator_width": 24,
                     "clock_period_ns": 2}, reward_cfg)
    at_crit = evaluate({"mac_lanes": 16, "accumulator_width": 24,
                        "clock_period_ns": round(crit_16, 3)}, reward_cfg)
    check("reward(clk<crit) <= reward(clk=crit)  for same lanes/acc",
          viol["reward"] <= at_crit["reward"],
          f"clk=2(violation)={viol['reward']}  clk=crit({crit_16:.3f})={at_crit['reward']}")
    check("clk<crit flagged as timing_violation; clk=crit is not",
          viol["proxies"]["timing_violation"] and not at_crit["proxies"]["timing_violation"],
          f"viol.flag={viol['proxies']['timing_violation']}  "
          f"crit.flag={at_crit['proxies']['timing_violation']}")
    check("real_speedup identical (effective clock capped to crit either way)",
          abs(viol["proxies"]["real_speedup"] - at_crit["proxies"]["real_speedup"]) < 1e-3,
          f"viol.rs={viol['proxies']['real_speedup']}  crit.rs={at_crit['proxies']['real_speedup']}")

    print()

    # ── (c) acc_width=16 reward far below 24/32 (overflow penalty intact) ─────
    print("(c) acc_width=16 overflow penalty intact:")
    a16 = evaluate({"mac_lanes": 8, "accumulator_width": 16, "clock_period_ns": 10}, reward_cfg)
    a24 = evaluate({"mac_lanes": 8, "accumulator_width": 24, "clock_period_ns": 10}, reward_cfg)
    a32 = evaluate({"mac_lanes": 8, "accumulator_width": 32, "clock_period_ns": 10}, reward_cfg)
    check("reward(acc=16) << reward(acc=24) and << reward(acc=32)  (gap > 8)",
          (a24["reward"] - a16["reward"] > 8.0) and (a32["reward"] - a16["reward"] > 8.0),
          f"acc16={a16['reward']}  acc24={a24['reward']}  acc32={a32['reward']}")
    check("acc=16 flagged acc_overflow; acc>=24 not",
          a16["proxies"]["acc_overflow"]
          and not a24["proxies"]["acc_overflow"]
          and not a32["proxies"]["acc_overflow"])

    print()

    # ── (d) argmax is a real trade-off, not all-extremes-degenerate ───────────
    print("(d) Global argmax over the full space is a genuine trade-off:")
    results = [evaluate(c, reward_cfg) for c in configs]
    best = max(results, key=lambda r: r["reward"])
    bcfg = best["config"]
    print(f"      argmax config = {bcfg}")
    print(f"      reward = {best['reward']}  real_speedup = {best['proxies']['real_speedup']}  "
          f"area = {best['proxies']['area_proxy']}  power = {best['proxies']['power_proxy']}")
    # The old bug forced clk=20 (slowest) unconditionally.  Assert it is NOT 20.
    check("optimum does NOT force the slowest clock (clk != 20)",
          bcfg["clock_period_ns"] != 20,
          f"chosen clk = {bcfg['clock_period_ns']} ns")
    # Optimum must be correct (acc >= 24) — never overflow.
    check("optimum uses a non-overflowing accumulator (acc >= 24)",
          bcfg["accumulator_width"] >= 24)
    # Not degenerate-maximal on every axis (real trade-off): a real Pareto pick
    # should not simultaneously max lanes AND min clock (that's the timing-viol
    # corner) — at minimum it must not be the all-max-throughput-illegal corner.
    degenerate = (bcfg["mac_lanes"] == 16 and bcfg["clock_period_ns"] == 5)
    check("optimum is not the degenerate max-lanes + min-clock timing-violation corner",
          not degenerate,
          f"lanes={bcfg['mac_lanes']} clk={bcfg['clock_period_ns']}")

    # Reward bounds sanity vs UCB normalisation window.
    all_rewards = [r["reward"] for r in results]
    rmin, rmax = min(all_rewards), max(all_rewards)
    print(f"      reward range over space: [{rmin}, {rmax}]")
    try:
        from gen1.agents.ucb_agent import _REWARD_LO, _REWARD_HI
        check("UCB bounds bracket the realistic reward range "
              "(_REWARD_LO < rmin and rmax < _REWARD_HI)",
              _REWARD_LO < rmin and rmax < _REWARD_HI,
              f"_REWARD_LO={_REWARD_LO}  rmin={rmin}  rmax={rmax}  _REWARD_HI={_REWARD_HI}")
        # normalised optimum strictly inside (0,1) — no clamp
        rng = _REWARD_HI - _REWARD_LO
        norm_best = (rmax - _REWARD_LO) / rng
        check("normalised optimum strictly inside (0,1) — no clamp",
              0.0 < norm_best < 1.0,
              f"normalised best = {norm_best:.4f}")
    except Exception as exc:  # pragma: no cover
        check("UCB bounds importable", False, f"import failed: {exc}")

    print()

    # ── Bayesian dangling-trial fix (only if optuna installed) ────────────────
    print("(e) Bayesian dangling-trial leak fix:")
    try:
        import optuna  # noqa: F401
    except ImportError:
        print("      [SKIP] optuna not installed — skipping Bayesian test gracefully.")
    else:
        from gen1.agents.bayesian_agent import BayesianAgent
        # Build proper search_space dict for the agent.
        raw = yaml.safe_load(
            (Path(__file__).parent / "search_space.yaml").read_text(encoding="utf-8"))
        ss = {**raw.get("sim_params", {}), **raw.get("proxy_params", {})}
        agent = BayesianAgent(ss, seed=0)

        # Normal path: suggest → update works.
        c1 = agent.suggest({}, [])
        agent.update(c1, 1.23, {})
        check("normal suggest→update path works (no exception)", True)

        # Simulate a SKIPPED trial: suggest WITHOUT update (sim RuntimeError).
        _c2 = agent.suggest({}, [])         # asks a trial, never told
        # Next suggest() must resolve the dangling trial as FAILED first.
        _c3 = agent.suggest({}, [])
        from optuna.trial import TrialState
        states = [t.state for t in agent._study.get_trials(deepcopy=False)]
        n_running = sum(1 for s in states if s == TrialState.RUNNING)
        check("at most one RUNNING trial remains after a skipped trial "
              "(no leak of the un-told trial)",
              n_running <= 1,
              f"trial states = {[s.name for s in states]}")
        # And telling the latest still works.
        agent.update(_c3, 0.5, {})
        states2 = [t.state for t in agent._study.get_trials(deepcopy=False)]
        n_running2 = sum(1 for s in states2 if s == TrialState.RUNNING)
        check("after final update no RUNNING trial remains",
              n_running2 == 0,
              f"trial states = {[s.name for s in states2]}")

    print()
    print("-" * 70)
    print(f"  {_passes} passed, {_fails} failed")
    return 0 if _fails == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
