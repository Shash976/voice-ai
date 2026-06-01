#!/usr/bin/env python3
"""run_optimizer.py — TinyMAC RL / agentic design-space optimizer.

Usage
-----
  # Run from the repo root (WSL):
  python3 optimizer/run_optimizer.py                        # default: evo, 30 trials
  python3 optimizer/run_optimizer.py --agent random         # random search baseline
  python3 optimizer/run_optimizer.py --agent ucb            # UCB1 bandit (RL)
  python3 optimizer/run_optimizer.py --agent bayesian       # Optuna TPE
  python3 optimizer/run_optimizer.py --agent evo --trials 50
  python3 optimizer/run_optimizer.py --resume               # continue previous run
  python3 optimizer/run_optimizer.py --dry-run              # print configs, skip sim

Prerequisites
-------------
  - Verilator sim built:  cd sim/verilator && make  (in WSL)
  - Firmware built:       cd firmware/picorv32_baremetal && make  (in WSL)
  - Python deps:          pip install --user pyyaml  (optuna only for --agent bayesian)

Dashboard (separate terminal):
  streamlit run optimizer/dashboard.py
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Make optimizer/ importable when run as: python3 optimizer/run_optimizer.py
sys.path.insert(0, str(Path(__file__).parent))

from agents.bayesian_agent import BayesianAgent
from agents.evo_agent import EvoAgent
from agents.random_agent import RandomAgent
from agents.ucb_agent import UCBAgent
from env import RESULTS_FILE, OptEnv
from runner import SW_BASELINE_CYCLES

AGENTS = {
    "random":   RandomAgent,
    "evo":      EvoAgent,
    "ucb":      UCBAgent,
    "bayesian": BayesianAgent,
}

_AGENT_DESCRIPTIONS = {
    "random":   "uniform random sampling (baseline)",
    "evo":      "(mu+lambda) evolutionary strategy — recommended default",
    "ucb":      "factored UCB1 bandit — good for small spaces",
    "bayesian": "Optuna TPE — best sample efficiency (requires: pip install optuna)",
}


# ── Formatting helpers ────────────────────────────────────────────────────────

def _flags(prx: dict) -> str:
    f = []
    if prx.get("timing_violation"):
        f.append("\033[91mTIMING!\033[0m")
    if prx.get("acc_overflow"):
        f.append("\033[91mOVERFLOW!\033[0m")
    return " ".join(f)


def _print_trial(rec: dict) -> None:
    cfg = rec["config"]
    sim = rec["sim_metrics"]
    prx = rec["proxy_metrics"]
    flag_str = _flags(prx)

    print(
        f"  [{rec['trial']:3d}]  "
        f"lanes={cfg['mac_lanes']:2d}  "
        f"acc={cfg.get('accumulator_width', 32):2d}b  "
        f"clk={cfg.get('clock_period_ns', 10):2d}ns  "
        f"buf={cfg.get('input_buffer_bytes', 1024):4d}B  "
        f"| cycles={sim['avg_cycles']:7,d}  speedup={sim['speedup']:6.1f}×  "
        f"| area={prx['area_proxy']:.2f}  pwr={prx['power_proxy']:.2f}  "
        f"slack={prx['timing_slack_ns']:+.1f}ns  "
        f"| rew={rec['reward']:7.3f}  ({rec['elapsed_s']:.1f}s)  {flag_str}"
    )


def _print_header(agent_name: str, n_trials: int, space: dict) -> None:
    print()
    print("┌─────────────────────────────────────────────────────────────────────────┐")
    print("│ TinyMAC RL / Agentic Design-Space Optimizer                             │")
    print("└─────────────────────────────────────────────────────────────────────────┘")
    print(f"  agent    : {agent_name}  —  {_AGENT_DESCRIPTIONS.get(agent_name, '')}")
    print(f"  trials   : {n_trials}")
    print(f"  params   : {', '.join(space.keys())}")
    print(f"  baseline : {SW_BASELINE_CYCLES:,} cycles/inference (Stage 3, no accel)")
    print(f"  results  : {RESULTS_FILE}")
    print()
    print(
        f"  {'[t]':>4}  {'lanes':>5}  {'acc':>4}  {'clk':>4}  {'buf':>5}  "
        "│ cycles   speedup  │ area  pwr  slack    │ reward"
    )
    print("  " + "─" * 88)


def _print_summary(history: list[dict]) -> None:
    if not history:
        print("No completed trials.")
        return

    print()
    print("═" * 90)
    print("  Top configurations (by reward):")
    print(
        f"  {'trial':>5}  {'lanes':>5}  {'acc':>4}  {'clk':>4}  "
        f"{'cycles':>8}  {'speedup':>8}  {'area':>6}  {'pwr':>6}  "
        f"{'slack':>7}  {'reward':>8}"
    )

    seen: set = set()
    deduped = []
    for r in sorted(history, key=lambda x: x["reward"], reverse=True):
        key = str(sorted(r["config"].items()))
        if key not in seen:
            seen.add(key)
            deduped.append(r)

    for r in deduped[:10]:
        cfg = r["config"]
        prx = r["proxy_metrics"]
        flags = ""
        if prx.get("timing_violation"):
            flags += " ⚠T"
        if prx.get("acc_overflow"):
            flags += " ⚠O"
        print(
            f"  {r['trial']:>5}  {cfg['mac_lanes']:>5}  "
            f"{cfg.get('accumulator_width', 32):>4}  "
            f"{cfg.get('clock_period_ns', 10):>4}  "
            f"{r['sim_metrics']['avg_cycles']:>8,}  "
            f"{r['sim_metrics']['speedup']:>8.1f}×  "
            f"{prx['area_proxy']:>6.2f}  "
            f"{prx['power_proxy']:>6.2f}  "
            f"{prx['timing_slack_ns']:>+7.1f}  "
            f"{r['reward']:>8.3f}{flags}"
        )

    best = max(history, key=lambda x: x["reward"])
    b_cfg = best["config"]
    b_sim = best["sim_metrics"]
    b_prx = best["proxy_metrics"]
    print()
    print(
        f"  Best: lanes={b_cfg['mac_lanes']}  "
        f"acc={b_cfg.get('accumulator_width', 32)}b  "
        f"clk={b_cfg.get('clock_period_ns', 10)}ns  "
        f"speedup={b_sim['speedup']:.1f}×  "
        f"area={b_prx['area_proxy']:.2f}  "
        f"reward={best['reward']:.3f}"
    )
    print(f"  Results written to: {RESULTS_FILE}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="TinyMAC accelerator design-space optimizer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="\n".join(
            f"  {k:10s}  {v}" for k, v in _AGENT_DESCRIPTIONS.items()
        ),
    )
    parser.add_argument(
        "--agent", choices=list(AGENTS), default="evo",
        help="Search strategy (default: evo)",
    )
    parser.add_argument(
        "--trials", type=int, default=30,
        help="Number of optimizer trials (default: 30)",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Append to existing results.jsonl instead of clearing it",
    )
    parser.add_argument(
        "--space", default=None, metavar="YAML",
        help="Path to search_space.yaml (default: optimizer/search_space.yaml)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print proposed configs but skip the simulator (useful for testing)",
    )
    args = parser.parse_args()

    env = OptEnv(search_space_path=args.space)

    if args.resume:
        n_loaded = env.load_existing_results()
        if n_loaded:
            print(f"Resumed: loaded {n_loaded} existing results.")
    else:
        env.clear_results()

    agent = AGENTS[args.agent](env.search_space)

    # Warm-start the agent with previously observed results so it continues
    # learning rather than starting cold.  (Fixes the broken --resume behaviour
    # where agents ignored all history because update() was never replayed.)
    if args.resume and env.history:
        agent.warm_start(env.history)
        print(f"Agent warm-started from {len(env.history)} historical records.")

    _print_header(args.agent, args.trials, env.search_space)

    state = env.reset()
    n_ok = n_skip = 0
    t_start = time.time()

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
            print(f"\n  [ERROR] {exc}")
            print("  Build the sim first: cd sim/verilator && make  (in WSL)\n")
            sys.exit(1)
        except RuntimeError as exc:
            print(f"  [SKIP] sim failed: {exc}")
            n_skip += 1

    elapsed = time.time() - t_start
    from runner import cache_info
    print(f"\n  Completed {n_ok} trials ({n_skip} skipped) in {elapsed:.1f}s  |  {cache_info()}")
    _print_summary(env.history)


if __name__ == "__main__":
    main()
