#!/usr/bin/env python3
"""run_cascade_optimizer.py — agentic optimizer over the multi-fidelity FUNNEL.

Same agents as the other tracks (random / evo / ucb / bayesian), but each trial
is pushed through cascade.evaluate(): validate → elaborate → sim → proxy → full,
short-circuiting at the first gate it fails. Cheap gates reject most bad configs
in milliseconds, so expensive place-and-route only runs on survivors. The reward
(stage penalty for early death, full multi-objective PPA for survivors) feeds the
agent, which steers toward configs that make it deep into the funnel.

Run on a machine with the real tools (the VM, /opt/OpenROAD-flow-scripts):

  python3 optimizer/run_cascade_optimizer.py                       # evo, 30 trials, nangate45
  python3 optimizer/run_cascade_optimizer.py --agent random --trials 50
  python3 optimizer/run_cascade_optimizer.py --max-stage proxy     # fast: stop before P&R
  python3 optimizer/run_cascade_optimizer.py --platform asap7
  python3 optimizer/run_cascade_optimizer.py --resume

  PHYSICAL_MOCK=1 python3 optimizer/run_cascade_optimizer.py       # offline self-test (no tools)
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import Counter
from pathlib import Path

import pathlib as _pl
sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))

# The summary uses box-drawing/colour glyphs; make stdout UTF-8 so a Windows
# cp1252 console (e.g. for --dry-run) doesn't crash on them.
try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
except (AttributeError, ValueError):
    pass

from gen1.agents.bayesian_agent import BayesianAgent
from gen1.agents.evo_agent import EvoAgent
from gen1.agents.random_agent import RandomAgent
from gen1.agents.ucb_agent import UCBAgent
from gen1.cascade import STAGE_ORDER
from gen1.cascade_env import CASCADE_RESULTS_FILE, CascadeOptEnv

AGENTS = {"random": RandomAgent, "evo": EvoAgent, "ucb": UCBAgent, "bayesian": BayesianAgent}

# colour the furthest stage reached: deeper = greener
_STAGE_COLOR = {"validate": "\033[91m", "elaborate": "\033[93m", "sim": "\033[93m",
                "proxy": "\033[96m", "full": "\033[92m"}
_RESET = "\033[0m"


def _fmt(v, spec="{}", na="—"):
    return spec.format(v) if v is not None else na


def _print_trial(rec: dict) -> None:
    reached = rec.get("reached") or "—"
    col = _STAGE_COLOR.get(reached, "")
    tag = f"{col}{reached:<9}{_RESET}"
    m = rec.get("metrics") or {}
    extra = ""
    if reached == "full" and not rec.get("failed_stage"):
        extra = (f" area={_fmt(m.get('area_um2'), '{:7.0f}')}µm²"
                 f" Fmax={_fmt(m.get('fmax_mhz'), '{:6.1f}')}MHz"
                 f" pwr={_fmt(m.get('power_mw'), '{:5.0f}')}mW")
    elif rec.get("reason"):
        extra = f" \033[90m({rec['reason']})\033[0m"
    print(
        f"  [{rec['trial']:3d}] reached={tag}"
        f" lanes={_fmt(rec.get('lanes'), '{:2}')}"
        f" acc={_fmt(rec.get('acc_w'), '{:2}')}"
        f" clk={_fmt(rec.get('clk_ns'))}ns"
        f" util={_fmt((rec.get('config') or {}).get('core_utilization'))}"
        f" dens={_fmt((rec.get('config') or {}).get('place_density'))}"
        f" | rew={rec['reward']:8.3f}{extra} ({rec['elapsed_s']:.0f}s)"
    )


def _print_summary(history: list[dict]) -> None:
    if not history:
        print("No completed trials.")
        return
    # funnel attrition: how many configs reached each stage
    reached = Counter(r.get("reached") for r in history)
    print("\n" + "═" * 78)
    print("  Funnel attrition (furthest stage reached):")
    for stage in STAGE_ORDER:
        n = reached.get(stage, 0)
        bar = "█" * n
        print(f"    {stage:<10} {n:3d}  {bar}")

    full_ok = [r for r in history if r.get("reached") == "full" and not r.get("failed_stage")]
    print(f"\n  Reached full P&R successfully: {len(full_ok)}/{len(history)}")
    if full_ok:
        print("\n  Top configs that completed P&R (by reward):")
        print(f"  {'lanes':>5} {'acc':>4} {'clk':>5} {'util':>4} {'dens':>5} "
              f"{'area_µm²':>9} {'Fmax':>7} {'pwr_mW':>7} {'reward':>8}")
        seen, deduped = set(), []
        for r in sorted(full_ok, key=lambda x: x["reward"], reverse=True):
            key = str(sorted(r["config"].items()))
            if key not in seen:
                seen.add(key)
                deduped.append(r)
        for r in deduped[:10]:
            c, m = r["config"], r.get("metrics") or {}
            # Stage C: use _fmt for all config fields that may be None to avoid
            # a crash when formatting None with a width spec (e.g. :>5).
            print(f"  {_fmt(c.get('mac_lanes'), '{:>5}')} {_fmt(c.get('accumulator_width'), '{:>4}')} "
                  f"{_fmt(c.get('clock_period_ns'), '{:>5}')} {_fmt(c.get('core_utilization'), '{:>4}')} "
                  f"{_fmt(c.get('place_density'), '{:>5}')} {_fmt(m.get('area_um2'), '{:9.0f}')} "
                  f"{_fmt(m.get('fmax_mhz'), '{:7.1f}')} {_fmt(m.get('power_mw'), '{:7.0f}')} "
                  f"{r['reward']:>8.3f}")
    best = max(history, key=lambda x: x["reward"])
    print(f"\n  Best reward: {best['reward']:.3f}  (reached {best.get('reached')}, "
          f"config {best['config']})")
    print(f"  Results: {CASCADE_RESULTS_FILE}")


def main() -> None:
    p = argparse.ArgumentParser(description="Cascade (multi-fidelity) design-space optimizer")
    p.add_argument("--agent", choices=list(AGENTS), default="evo")
    p.add_argument("--trials", type=int, default=30)
    p.add_argument("--platform", default="nangate45")
    p.add_argument("--max-stage", choices=STAGE_ORDER, default="full",
                   help="cap the funnel: 'proxy' = fast (no P&R), 'full' = ground-truth PPA")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--space", default=None)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    env = CascadeOptEnv(search_space_path=args.space, platform=args.platform,
                        max_stage=args.max_stage)
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

    space_size = 1
    for spec in env.search_space.values():
        space_size *= len(spec["choices"])
    print(f"\n  Cascade optimizer | agent={args.agent} trials={args.trials} "
          f"platform={args.platform} max_stage={args.max_stage}")
    print(f"  params ({len(env.search_space)}): {', '.join(env.search_space.keys())}")
    print(f"  space size: {space_size} configs | results: {CASCADE_RESULTS_FILE}\n")

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
            n_ok += 1
        except FileNotFoundError as exc:
            print(f"\n  [ERROR] {exc}\n")
            sys.exit(1)
        except Exception as exc:  # noqa: BLE001 — one bad trial shouldn't kill the run
            print(f"  [SKIP] trial failed: {exc}")
            n_skip += 1

    if not args.dry_run:
        print(f"\n  Completed {n_ok} trials ({n_skip} skipped) in {time.time() - t0:.0f}s")
        _print_summary(env.history)


if __name__ == "__main__":
    main()
