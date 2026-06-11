#!/usr/bin/env python3
"""run_physical_optimizer.py — agentic optimizer over the REAL ORFS RTL→GDS flow.

Same agents as run_optimizer.py (random / evo / ucb / bayesian), but each trial
runs the actual OpenROAD flow for the proposed {LANES, ACC_W, clock} config and
scores the measured area / Fmax / power / timing.  This is the Stage-5↔Stage-6
loop: the AI proposes a chip configuration, the tools build it, the real metrics
feed the reward, the agent picks the next one.

Run on a machine with a real OpenROAD (the VM, /opt/OpenROAD-flow-scripts):

  python3 optimizer/run_physical_optimizer.py                 # evo, 12 trials, nangate45
  python3 optimizer/run_physical_optimizer.py --agent random --trials 8
  python3 optimizer/run_physical_optimizer.py --platform sky130hd
  python3 optimizer/run_physical_optimizer.py --resume
  python3 optimizer/run_physical_optimizer.py --dry-run        # print configs, no flow

  PHYSICAL_MOCK=1 python3 optimizer/run_physical_optimizer.py  # offline self-test

⚠ Each non-cached trial runs a full place-and-route (minutes).  Distinct configs
are cached and already-built variants are reused, so keep trial counts modest.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from agents.bayesian_agent import BayesianAgent
from agents.evo_agent import EvoAgent
from agents.random_agent import RandomAgent
from agents.ucb_agent import UCBAgent
from physical_env import PHYS_RESULTS_FILE, PhysicalOptEnv

AGENTS = {
    "random":   RandomAgent,
    "evo":      EvoAgent,
    "ucb":      UCBAgent,
    "bayesian": BayesianAgent,
}


def _fmt(v, spec="{}", na="—"):
    return spec.format(v) if v is not None else na


def _print_trial(rec: dict) -> None:
    cfg, m, s = rec["config"], rec["metrics"], rec["scored"]
    flags = ""
    if m.get("timing_met") is False:
        flags += " \033[91mTIMING!\033[0m"
    if s.get("accuracy", 1.0) < 1.0:
        flags += " \033[91mOVERFLOW!\033[0m"
    if m.get("status") == "FAIL":
        flags += " \033[91mFLOW-FAIL\033[0m"
    print(
        f"  [{rec['trial']:3d}]  "
        f"lanes={cfg.get('mac_lanes', 4):2d}  "
        f"acc={cfg.get('accumulator_width', 24):2d}b  "
        f"clk={cfg.get('clock_period_ns', 5)}ns  "
        f"| area={_fmt(m.get('area_um2'), '{:7.0f}')}µm²  "
        f"Fmax={_fmt(m.get('fmax_mhz'), '{:6.1f}')}MHz  "
        f"pwr={_fmt(m.get('power_mw'), '{:6.0f}')}mW  "
        f"| spd={_fmt(s.get('real_speedup'), '{:6.1f}')}×  "
        f"rew={rec['reward']:7.3f}  ({rec['elapsed_s']:.0f}s){flags}"
    )


def _print_summary(history: list[dict]) -> None:
    if not history:
        print("No completed trials.")
        return
    print("\n" + "═" * 78)
    print("  Top configurations (by reward):")
    print(f"  {'lanes':>5} {'acc':>4} {'clk':>4} {'area_µm²':>9} {'Fmax':>7} "
          f"{'pwr_mW':>7} {'speedup':>8} {'met':>4} {'reward':>8}")
    seen, deduped = set(), []
    for r in sorted(history, key=lambda x: x["reward"], reverse=True):
        key = str(sorted(r["config"].items()))
        if key not in seen:
            seen.add(key)
            deduped.append(r)
    for r in deduped[:10]:
        c, m = r["config"], r["metrics"]
        print(f"  {c.get('mac_lanes', 4):>5} {c.get('accumulator_width', 24):>4} "
              f"{c.get('clock_period_ns', 5):>4} {_fmt(m.get('area_um2'), '{:9.0f}')} "
              f"{_fmt(m.get('fmax_mhz'), '{:7.1f}')} {_fmt(m.get('power_mw'), '{:7.0f}')} "
              f"{_fmt(r.get('real_speedup'), '{:7.1f}×')} "
              f"{('yes' if m.get('timing_met') else 'NO'):>4} {r['reward']:>8.3f}")
    best = max(history, key=lambda x: x["reward"])
    bc = best["config"]
    print(f"\n  Best: lanes={bc.get('mac_lanes')} acc={bc.get('accumulator_width')}b "
          f"clk={bc.get('clock_period_ns')}ns → reward={best['reward']:.3f}, "
          f"area={_fmt(best['metrics'].get('area_um2'), '{:.0f}')}µm², "
          f"Fmax={_fmt(best['metrics'].get('fmax_mhz'), '{:.0f}')}MHz")
    print(f"  Results: {PHYS_RESULTS_FILE}")


def main() -> None:
    p = argparse.ArgumentParser(description="Physical (ORFS) design-space optimizer")
    p.add_argument("--agent", choices=list(AGENTS), default="evo")
    p.add_argument("--trials", type=int, default=12)
    p.add_argument("--platform", default="nangate45")
    p.add_argument("--mode", choices=["full", "proxy"], default="full",
                   help="full = RTL→GDS (minutes); proxy = synth+STA (seconds, for fast search)")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--space", default=None)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    env = PhysicalOptEnv(search_space_path=args.space, platform=args.platform, mode=args.mode)
    if args.resume:
        n = env.load_existing_results()
        if n:
            print(f"Resumed: loaded {n} existing results.")
    else:
        env.clear_results()

    # UCB bounds wiring: pass reward_bounds from the env if available so the UCB
    # agent's normalisation preserves the penalty ladder structure (V9 / Stage C).
    agent_kwargs = {}
    if args.agent == "ucb":
        agent_kwargs["reward_bounds"] = getattr(env, "reward_bounds", None)
    agent = AGENTS[args.agent](env.search_space, **agent_kwargs)
    if args.resume and env.history:
        agent.warm_start(env.history)
        print(f"Agent warm-started from {len(env.history)} records.")

    print(f"\n  Physical optimizer | agent={args.agent} trials={args.trials} "
          f"platform={args.platform} mode={args.mode}")
    print(f"  params: {', '.join(env.search_space.keys())}")
    print(f"  results: {PHYS_RESULTS_FILE}\n")

    state = env.reset()
    n_ok = n_skip = 0
    t0 = time.time()
    for _ in range(args.trials):
        config = agent.suggest(state, env.history)
        if args.dry_run:
            print(f"  [dry-run] would evaluate: {config}")
            continue
        try:
            state, reward, _done, info = env.step(config)
            agent.update(config, reward, info)
            _print_trial(info)
            # V10: TIMEOUT trials are logged by env.step and returned like any
            # other trial (with their penalty reward) so the agent can avoid
            # re-proposing the same config.
            if info.get("metrics", {}).get("status") == "TIMEOUT":
                print(f"  [TIMEOUT] trial {info.get('trial')} timed out; logged with penalty reward.")
            n_ok += 1
        except FileNotFoundError as exc:
            print(f"\n  [ERROR] {exc}\n")
            sys.exit(1)
        except Exception as exc:  # noqa: BLE001 — one bad trial shouldn't kill the sweep
            print(f"  [SKIP] trial failed: {exc}")
            n_skip += 1

    if not args.dry_run:
        print(f"\n  Completed {n_ok} trials ({n_skip} skipped) in {time.time() - t0:.0f}s")
        _print_summary(env.history)


if __name__ == "__main__":
    main()
