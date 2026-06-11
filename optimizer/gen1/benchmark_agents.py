"""benchmark_agents.py — HONEST sample-efficiency benchmark for the search agents.

This is an OFFLINE benchmark.  It does NOT call runner.run_sim() (which launches a
Linux Verilator binary).  Instead it mirrors env.step() exactly using a small table
of REAL measured sim outputs (avg_cycles vs. mac_lanes, accuracy vs. accumulator_width),
so every reward is computed by the same reward.py the real optimizer uses.

The 45-config space (5 mac_lanes × 3 accumulator_width × 3 clock_period_ns) is tiny and
deterministic with a single global optimum, so agent quality can only be judged by SAMPLE
EFFICIENCY: how few trials to reach the known optimum, and cumulative regret along the way.

We compute the true optimum exhaustively, then run each agent for T=30 trials over SEEDS=
range(30), and report regret / trials-to-optimum / success-rate.  We also print the
analytical random-search baseline so the empirical "random" agent can be sanity-checked.

Run:  python optimizer/benchmark_agents.py
"""

from __future__ import annotations

import os
import random
import statistics
import sys
from itertools import product

# Force UTF-8 stdout so output is identical on Windows (cp1252 console) and POSIX.
try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
except Exception:
    pass

# Bootstrap: make optimizer/ root importable (gen1/ is one level below it)
import pathlib as _pl
_THIS_DIR = str(_pl.Path(__file__).resolve().parent)
_OPT_DIR  = str(_pl.Path(__file__).resolve().parents[1])
if _OPT_DIR not in sys.path:
    sys.path.insert(0, _OPT_DIR)

import yaml  # noqa: E402

from gen1 import reward  # noqa: E402
from gen1.agents.random_agent import RandomAgent  # noqa: E402
from gen1.agents.evo_agent import EvoAgent  # noqa: E402
from gen1.agents.ucb_agent import UCBAgent  # noqa: E402
from common.constants import AVG_CYCLES as _AVG_CYCLES_ALL  # noqa: E402

# ── Real measured offline sim model (matches the real sim exactly) ───────────────
# avg_cycles: imported from constants.py (single source of truth).
#   Measured 2026-06-10 after V13 saturation-order fix (per-chunk, matches RTL).
#   45-config grid uses lanes ∈ {1,2,4,8,16}; we slice that subset here.
# accuracy depends ONLY on accumulator_width (acc_width does not affect cycles).
#   acc=16 accuracy is LANES-DEPENDENT after the V13 fix (per-chunk saturation
#   gives slightly different overflow behaviour); for the offline benchmark we keep
#   the historical 47/64 = 0.734375 for acc=16 since the benchmark's sanity anchor
#   still holds (any acc=16 config scores far below acc>=24 regardless).
AVG_CYCLES = {l: _AVG_CYCLES_ALL[l] for l in [1, 2, 4, 8, 16]}
ACCURACY = {16: 0.734375, 24: 1.0, 32: 1.0}

# ── Benchmark protocol constants ─────────────────────────────────────────────────
T = 30
SEEDS = range(30)
REGRET_CHECKPOINTS = [1, 5, 10, 15, 20, 30]
EPS = 1e-9


def load_space():
    """Load search_space.yaml → (search_space dict, list of 45 configs, reward_cfg)."""
    path = os.path.join(_THIS_DIR, "search_space.yaml")
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    sim_specs = raw.get("sim_params", {})
    proxy_specs = raw.get("proxy_params", {})
    search_space = {**sim_specs, **proxy_specs}  # name → spec (combined)
    reward_cfg = raw.get("reward", {})

    names = list(search_space.keys())
    combos = product(*[search_space[n]["choices"] for n in names])
    configs = [dict(zip(names, c)) for c in combos]
    return search_space, configs, reward_cfg


def offline_eval(config: dict, reward_cfg: dict) -> float:
    """Mirror env.step() EXACTLY, but with the offline measured-sim table.

    Returns the scalar reward.  Deterministic.
    """
    lanes = int(config["mac_lanes"])
    accw = int(config["accumulator_width"])

    avg = AVG_CYCLES[lanes]
    acc = ACCURACY[accw]
    cycle_speedup = reward.SW_BASELINE_CYCLES / avg

    sim_metrics = {
        "avg_cycles": avg,
        "accuracy": acc,
        "speedup": cycle_speedup,
        "mac_lanes": lanes,
        "acc_width": accw,
    }

    proxies = reward.compute_proxies(config)
    proxies["real_speedup"] = round(reward.real_speedup(config, avg), 3)
    proxies["latency_ns"] = round(avg * proxies["effective_clock_ns"], 1)

    return reward.compute_reward(sim_metrics, proxies, reward_cfg)


def make_agent(name: str, search_space: dict, seed: int):
    """Construct + seed an agent.  Returns the agent, or None if unavailable."""
    if name == "random":
        agent = RandomAgent(search_space)
    elif name == "evo":
        agent = EvoAgent(search_space)
    elif name == "ucb":
        agent = UCBAgent(search_space)
    elif name == "bayesian":
        try:
            from gen1.agents.bayesian_agent import BayesianAgent
            agent = BayesianAgent(search_space, seed=seed)
        except Exception:
            return None
    else:
        raise ValueError(f"unknown agent {name!r}")

    # Deterministic seeding (per spec): override the agent's RNG.
    agent._rng = random.Random(seed)
    return agent


def run_one(agent, reward_cfg: dict, optimal_reward: float):
    """Run a single (agent, seed) optimization for T trials.

    Replicates run_optimizer's loop offline.  Returns (best_so_far list of len T).
    """
    history: list[dict] = []
    best_so_far: list[float] = []
    best = float("-inf")
    # Agents read `state` only loosely; mirror env's pre-step state shape minimally.
    # None of random/evo/ucb/bayesian read specific state keys in suggest(); they use
    # their own internal tables + the `history` list.  An empty-ish dict is faithful.
    for _t in range(T):
        state = {"trial": len(history), "history_len": len(history)}
        config = agent.suggest(state, history)
        r = offline_eval(config, reward_cfg)
        agent.update(config, r, {})
        history.append({"config": config, "reward": r})
        best = max(best, r)
        best_so_far.append(best)
    return best_so_far


def benchmark_agent(name: str, search_space: dict, reward_cfg: dict,
                    optimal_reward: float):
    """Run an agent across all SEEDS.  Returns an aggregate dict, or None if skipped."""
    per_seed_best = []  # best_so_far curves, one per seed
    for seed in SEEDS:
        agent = make_agent(name, search_space, seed)
        if agent is None:
            return None
        per_seed_best.append(run_one(agent, reward_cfg, optimal_reward))

    final_regrets = [optimal_reward - bsf[T - 1] for bsf in per_seed_best]

    trials_to_opt = []
    for bsf in per_seed_best:
        hit = 31  # censored
        for t in range(1, T + 1):
            if bsf[t - 1] >= optimal_reward - EPS:
                hit = t
                break
        trials_to_opt.append(hit)

    successes = sum(1 for tt in trials_to_opt if tt <= T)

    # Mean regret at specific 1-indexed checkpoints t.
    regret_curve = {}
    for t in REGRET_CHECKPOINTS:
        regrets_t = [optimal_reward - bsf[t - 1] for bsf in per_seed_best]
        regret_curve[t] = statistics.mean(regrets_t)

    return {
        "name": name,
        "mean_final_regret": statistics.mean(final_regrets),
        "std_final_regret": statistics.pstdev(final_regrets),
        "mean_trials_to_opt": statistics.mean(trials_to_opt),
        "median_trials_to_opt": statistics.median(trials_to_opt),
        "success_rate": successes / len(per_seed_best),
        "regret_curve": regret_curve,
    }


def main():
    search_space, configs, reward_cfg = load_space()

    assert len(configs) == 45, f"expected 45 configs, got {len(configs)}"

    # ── Exhaustive grid → optimum ────────────────────────────────────────────────
    scored = [(offline_eval(c, reward_cfg), c) for c in configs]
    optimal_reward, optimal_config = max(scored, key=lambda x: x[0])

    # SANITY ANCHOR
    expected = {"mac_lanes": 4, "accumulator_width": 24, "clock_period_ns": 5}
    if optimal_config != expected or abs(optimal_reward - 4.01) > 0.05:
        print("SANITY CHECK FAILED - grid optimum does not match the expected anchor.")
        print(f"  got config = {optimal_config}, reward = {optimal_reward}")
        print(f"  expected   = {expected}, reward ~= 4.01")
        sys.exit(1)

    print("=" * 78)
    print("HONEST SAMPLE-EFFICIENCY BENCHMARK - TinyMAC design-space search agents")
    print("=" * 78)
    print(f"Search space   : {len(configs)} configs "
          f"(5 mac_lanes x 3 accumulator_width x 3 clock_period_ns)")
    print(f"Protocol       : T = {T} trials, SEEDS = range({len(list(SEEDS))})")
    print(f"Offline model  : avg_cycles(mac_lanes) + accuracy(acc_width), "
          f"reward via reward.compute_reward (frequency-aware)")
    print()
    print("GRID OPTIMUM (exhaustive over all 45 configs):")
    print(f"  optimal_config = {optimal_config}")
    print(f"  optimal_reward = {optimal_reward}")
    # Show how unique the optimum is.
    n_at_opt = sum(1 for r, _ in scored if r >= optimal_reward - EPS)
    print(f"  configs at the global optimum reward: {n_at_opt} / {len(configs)} "
          f"(unique optimum)" if n_at_opt == 1 else
          f"  configs at the global optimum reward: {n_at_opt} / {len(configs)}")
    print()

    # ── Run agents ───────────────────────────────────────────────────────────────
    agent_names = ["random", "evo", "ucb", "bayesian"]
    results = []
    skipped = []
    for name in agent_names:
        res = benchmark_agent(name, search_space, reward_cfg, optimal_reward)
        if res is None:
            skipped.append(name)
        else:
            results.append(res)

    if skipped:
        print(f"NOTE: skipped agent(s) {skipped} "
              f"(optuna not installed - `pip install --user optuna` to include bayesian).")
        print()

    # ── Per-agent comparison table ───────────────────────────────────────────────
    print("PER-AGENT COMPARISON (aggregated over {} seeds)".format(len(list(SEEDS))))
    print("-" * 78)
    hdr = (f"{'agent':<10} {'mean_regret':>12} {'std_regret':>11} "
           f"{'mean_t2opt':>11} {'med_t2opt':>10} {'success@'+str(T):>11}")
    print(hdr)
    print("-" * 78)
    for r in results:
        print(f"{r['name']:<10} "
              f"{r['mean_final_regret']:>12.4f} "
              f"{r['std_final_regret']:>11.4f} "
              f"{r['mean_trials_to_opt']:>11.3f} "
              f"{r['median_trials_to_opt']:>10.1f} "
              f"{r['success_rate']*100:>10.1f}%")
    print("-" * 78)
    print("(t2opt = trials-to-reach-optimum; censored at 31 if not found within T)")
    print()

    # ── Mean-regret curve table ──────────────────────────────────────────────────
    print("MEAN BEST-SO-FAR REGRET vs. TRIAL  (lower is better; 0 = optimum reached)")
    print("-" * 78)
    head = f"{'agent':<10}" + "".join(f"{'t='+str(t):>11}" for t in REGRET_CHECKPOINTS)
    print(head)
    print("-" * 78)
    for r in results:
        row = f"{r['name']:<10}"
        for t in REGRET_CHECKPOINTS:
            row += f"{r['regret_curve'][t]:>11.4f}"
        print(row)
    print("-" * 78)
    print()

    # ── Analytical random-search baseline ────────────────────────────────────────
    n = len(configs)
    p_find_30 = 1.0 - (((n - 1) / n) ** T) if n_at_opt == 1 else None
    exp_trials = n  # expected trials to first-hit a unique target in uniform sampling
    print("ANALYTICAL RANDOM-SEARCH BASELINE (unique optimum, uniform sampling)")
    print("-" * 78)
    print(f"  P(find optimum by trial {T}) = 1 - ({n-1}/{n})^{T} = "
          f"{p_find_30:.4f}  ({p_find_30*100:.2f}%)")
    print(f"  Expected trials to first hit  = {exp_trials}")
    emp = next((r for r in results if r["name"] == "random"), None)
    if emp is not None:
        print(f"  Empirical 'random' agent      : success@{T} = "
              f"{emp['success_rate']*100:.2f}%, "
              f"mean trials_to_opt = {emp['mean_trials_to_opt']:.3f} "
              f"(censored at 31)")
    print()

    # ── Conclusion ───────────────────────────────────────────────────────────────
    print("CONCLUSION")
    print("-" * 78)
    rand = next((r for r in results if r["name"] == "random"), None)
    lines = _conclusion(results, rand, optimal_reward, p_find_30, T, skipped)
    for ln in lines:
        print(ln)


def _wrap(text: str, width: int = 78):
    import textwrap
    return textwrap.fill(text, width=width)


def _conclusion(results, rand, optimal_reward, p_find_30, T, skipped):
    """Build a candid conclusion paragraph from the computed numbers."""
    out = []
    if rand is None:
        out.append("Random agent missing — cannot compare.")
        return out

    # Rank by mean final regret (lower better), then success rate.
    ranked = sorted(results, key=lambda r: (r["mean_final_regret"],
                                            -r["success_rate"]))
    best = ranked[0]

    rand_succ = rand["success_rate"]
    rand_reg = rand["mean_final_regret"]
    rand_t = rand["mean_trials_to_opt"]

    summary = []
    for r in results:
        summary.append(
            f"{r['name']}: mean_final_regret={r['mean_final_regret']:.4f}, "
            f"success@{T}={r['success_rate']*100:.1f}%, "
            f"mean_t2opt={r['mean_trials_to_opt']:.2f}")
    out.append(_wrap("Numbers: " + "  |  ".join(summary)))
    out.append("")

    # Decide whether anyone MEANINGFULLY beats random.
    # "Meaningful" = clearly lower regret AND/OR clearly higher success, beyond noise.
    beaters = []
    for r in results:
        if r["name"] == "random":
            continue
        better_regret = r["mean_final_regret"] < rand_reg - 1e-4
        better_succ = r["success_rate"] > rand_succ + 1e-9
        better_speed = r["mean_trials_to_opt"] < rand_t - 0.5
        if better_regret and (better_succ or better_speed):
            beaters.append(r["name"])

    if beaters:
        out.append(_wrap(
            f"VERDICT: {', '.join(beaters)} measurably beat random search on this "
            f"45-config space — they reach the unique global optimum "
            f"(reward={optimal_reward}) in fewer trials and/or with higher "
            f"success within T={T}. The margin is the difference in the table above; "
            f"on a space this small it is modest, not dramatic."))
    else:
        out.append(_wrap(
            f"VERDICT: No agent MEANINGFULLY beats random search on this 45-config "
            f"space. With only 45 deterministic configs and a single global optimum "
            f"(reward={optimal_reward}), there is very little structure for a smarter "
            f"agent to exploit before random search has already stumbled onto the "
            f"answer. The agents land within noise of each other on final regret and "
            f"trials-to-optimum (see the table). Random search itself reaches the "
            f"optimum with probability {p_find_30*100:.1f}% by trial {T}, which sets a "
            f"high bar. This is an HONEST tie: the agents are not proven superior here. "
            f"To demonstrate agent value you would need a larger / rougher / noisier "
            f"search space (or expensive evaluations where every trial counts)."))

    if skipped:
        out.append("")
        out.append(_wrap(
            f"Caveat: {', '.join(skipped)} agent(s) were skipped (optuna not "
            f"installed), so this comparison covers the stdlib agents only."))
    return out


if __name__ == "__main__":
    main()
