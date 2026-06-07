# 04 — The Design-Space Optimizer (Stage 5)

Now that trying a hardware config is cheap (just re-run the behavioral sim with
different flags), we can *automatically search* for the best accelerator design.
**Runs on Windows** (pure Python; it shells out to the WSL-built sim, or uses an
offline table for the benchmark). Lives in [`../optimizer/`](../optimizer/).

---

## What "optimize" means here

The accelerator has knobs. Each combination (a *config*) has trade-offs: more MAC
lanes = faster but bigger area; narrower accumulator = smaller but may overflow;
faster clock = more performance but may violate timing. The optimizer's job: try
configs, **score each with a reward function**, and find the best.

### The search space — [`../optimizer/search_space.yaml`](../optimizer/search_space.yaml)

Only **2 axes are genuinely simulated**, plus 1 evaluated by formula:

| Param | Values | How it's evaluated |
|-------|--------|--------------------|
| `mac_lanes` | 1, 2, 4, 8, 16 | **Simulated** (`--mac-lanes`) — sets real cycle count |
| `accumulator_width` | 16, 24, 32 | **Simulated** (`--acc-width`) — sets real accuracy (16 overflows) |
| `clock_period_ns` | 5, 10, 20 | **Proxy** — analytic timing/power formulas in `reward.py` |

That's **5 × 3 × 3 = 45 configs** total. Small and fully enumerable — which matters
for the honest conclusion below.

> **Design principle (from the plan):** *score real results, don't guess
> performance.* That's why buffer-size and dataflow axes from the original plan were
> **dropped** — the behavioral sim has no buffer/cache model, so optimizing them
> would be guessing. They belong in Stage 6 (ORFS) where SRAM area/timing is real.
> `dataflow` is also dropped because output- vs weight-stationary have *identical*
> MAC counts here; only physical routing differs (again, Stage 6).

---

## The reward function — [`../optimizer/reward.py`](../optimizer/reward.py)

A single scalar (higher = better):

```
reward = 2.0 * accuracy
       + 3.0 * log2_norm_speedup        # the primary objective
       − 0.4 * area_proxy               # 1.0 at baseline (8 lanes, 32b)
       − 0.4 * power_proxy
       − 3.0 * timing_violation         # 0 or 1
       − 8.0 * perf_floor_penalty       # if real_speedup < 10×
       − 50  * (1 − accuracy)           # hard penalty for wrong outputs
```

### The frequency-aware speedup fix (the key Stage-5 insight)

The reward's speedup term uses a **real-time** speedup, *not* a raw cycle ratio:

```python
effective_clock_ns = max(clock_period_ns, critical_path_ns)   # can't beat the critical path
latency_ns         = avg_cycles * effective_clock_ns
real_speedup       = SW_BASELINE_LATENCY_NS / latency_ns
```

**Why this matters:** the old reward used `SW_cycles / accel_cycles`, which is
*frequency-independent*. A 20 ns (slow) clock scored identically to a 5 ns (fast)
clock — even though the slow clock makes the actual chip slower. So the optimizer
kept picking the slowest clock (a degeneracy). Converting cycles → nanoseconds fixes
it: a faster clock that still meets timing earns more reward, while requesting a clock
*faster than the critical path* gets **capped** (no free reward) — and still pays the
higher power cost and trips the timing-violation penalty. See the long comment at
`reward.py:14` and `reward.py:253`.

### The proxy formulas (analytic, no sim)

`compute_proxies()` returns area/power/timing estimates from simple calibrated models:
- `area_proxy` — MAC array (∝ lanes × acc_width) is 80%, SRAM is a fixed 20%.
  Normalized to **1.0 at the baseline** (8 lanes, 32b).
- `power_proxy` — `area × clock_frequency`, also 1.0 at baseline.
- `critical_path_ns` — `2.5 + 0.15·lanes + 0.05·(acc_width/8)` (ASAP7-calibrated
  guess). E.g. 16 lanes/32b = 5.10 ns → can't run at a 5 ns clock → timing violation.

These are **estimates**, clearly labeled as such, and become *real* numbers in
Stage 6 ORFS. The constants (`SW_BASELINE_CYCLES = 11,196,638`, `max_speedup = 576`)
were pinned empirically — see [`../optimizer/measure_real.py`](../optimizer/measure_real.py)
and the comments in the YAML/reward derivations.

---

## The agents — [`../optimizer/agents/`](../optimizer/agents/)

All agents share one interface (`BaseAgent`):

```python
config = agent.suggest(state, history)   # propose next config
agent.update(config, reward, info)        # learn from the result
```

| Agent | File | Strategy |
|-------|------|----------|
| `random` | `random_agent.py` | Uniform random sampling — the baseline |
| `evo` | `evo_agent.py` | (μ+λ) evolutionary search (mutate the best) — the default |
| `ucb` | `ucb_agent.py` | Factored UCB1 bandit — treats each axis as an independent multi-armed bandit |
| `bayesian` | `bayesian_agent.py` | Optuna TPE — needs `pip install optuna` (optional) |

The UCB agent is well-documented (`ucb_agent.py:1`) — note its rewards are normalized
to `[0,1]` with *fixed* bounds (`[−12, +4.5]`), never a running min (which would
silently re-interpret history).

---

## ⚠️ The honest finding (read this — it's the whole point of Stage 5)

[`../optimizer/benchmark_agents.py`](../optimizer/benchmark_agents.py) runs an
**offline** benchmark (uses a table of real measured sim outputs, not live sim) to
compare agents by *sample efficiency*: how few trials to reach the known optimum, and
cumulative regret. The exhaustive grid optimum is:

```
best config = {mac_lanes: 4, accumulator_width: 24, clock_period_ns: 5}   reward ≈ 4.01
```

**The verdict: no agent meaningfully beats random search on this 45-config space.**
With only 45 deterministic configs and a single global optimum, there's almost no
structure for a "smart" agent to exploit before random search has already stumbled
onto the answer (random finds the optimum with high probability within 30 trials).
UCB actually does slightly *worse*.

This is an **honest, deliberately-reported tie** — not a failure. The lesson: agent
cleverness only pays off on larger, rougher, noisier, or expensive-to-evaluate
spaces. On a tiny deterministic grid, just enumerate it. Documenting that honestly is
the deliverable.

There's also [`../optimizer/test_reward_sanity.py`](../optimizer/test_reward_sanity.py)
— 13 offline invariant checks on the reward function (e.g. "a timing-violating config
never out-scores the same config clocked legally"). Runs on Windows, no sim needed.

---

## How to run

### The offline benchmark + sanity tests (Windows, no sim needed)

```powershell
python optimizer/benchmark_agents.py    # agent comparison + the honest verdict
python optimizer/test_reward_sanity.py  # 13/13 reward invariants
```

These are the quickest way to see Stage 5 work — they need only Python + pyyaml.

### The live optimizer (needs the WSL-built sim + firmware)

```bash
# Prereqs (WSL): build firmware.bin and sim_picorv32 first (see doc 02)
python3 optimizer/run_optimizer.py                  # default: evo, 30 trials
python3 optimizer/run_optimizer.py --agent random
python3 optimizer/run_optimizer.py --agent ucb
python3 optimizer/run_optimizer.py --agent bayesian # needs optuna
python3 optimizer/run_optimizer.py --agent evo --trials 50
python3 optimizer/run_optimizer.py --resume         # continue a previous run
python3 optimizer/run_optimizer.py --dry-run        # print configs, skip the sim
```

This drives `OptEnv` ([`../optimizer/env.py`](../optimizer/env.py)), which for each
config calls the sim via [`../optimizer/runner.py`](../optimizer/runner.py)
(`run_sim()` is `@lru_cache`'d on `(mac_lanes, acc_width)`, so the 45-config space
costs at most 15 subprocess launches, not 45), computes proxies + reward, and appends
each result to `results.jsonl`.

### Live dashboard (optional)

```bash
streamlit run optimizer/dashboard.py    # reads results.jsonl live
```

---

## How the pieces connect

```
run_optimizer.py  (CLI loop)
    │  agent.suggest(state, history) → config
    ▼
env.py  OptEnv.step(config)
    │  ├─ runner.run_sim(lanes, acc_width)  ──> shells out to ./sim_picorv32 (WSL)
    │  ├─ reward.compute_proxies(config)    ──> area/power/timing estimates
    │  ├─ reward.real_speedup(...)          ──> frequency-aware speedup
    │  └─ reward.compute_reward(...)        ──> scalar
    ▼
results.jsonl  (one record per trial)  ──> dashboard.py / run summary table
```

---

## Mental model / what to remember

- Only `mac_lanes` and `accumulator_width` are *really* simulated; `clock_period_ns`
  is analytic. Buffers/dataflow were honestly dropped (no sim model → can't measure).
- The reward's headline insight: use **frequency-aware real_speedup**, not raw cycle
  ratio, or the optimizer cheats by picking the slowest clock.
- Grid optimum: `{lanes:4, acc:24, clk:5}`, reward ≈ 4.01.
- **No agent beats random on 45 configs** — and that's reported honestly. Agent value
  needs bigger/noisier spaces.
- Quickest demo: `benchmark_agents.py` + `test_reward_sanity.py` (Windows, no sim).

---

## The cascade optimizer — a bigger space + a screening funnel

The 45-config space above is fully enumerable, so search can't beat brute force.
The **cascade optimizer** ([`../optimizer/run_cascade_optimizer.py`](../optimizer/run_cascade_optimizer.py))
exists for the *opposite* regime: a much larger space where most configs are bad
and full evaluation is expensive.

### Bigger space — [`../optimizer/search_space_full.yaml`](../optimizer/search_space_full.yaml)

Six real, measurable axes (~27,000 configs):

| Param | Values | Wired to |
|-------|--------|----------|
| `mac_lanes` | 1,2,3,4,6,8,12,16,32 | RTL chparam LANES (sim + synth) |
| `accumulator_width` | 16,20,24,28,32 | RTL chparam ACC_W (sim correctness) |
| `clock_period_ns` | 0.5 … 10.0 (10 values) | SDC clock |
| `core_utilization` | 20 … 70 | ORFS `CORE_UTILIZATION` |
| `place_density` | 0.45 … 0.75 | ORFS `PLACE_DENSITY` |
| `abc_strategy` | speed, area | ORFS `ABC_AREA` |

Adding a knob still requires wiring it to a real tool (every axis above is) —
the project rule "score real results, don't guess" still holds. The flow knobs
are plumbed through `physical_runner._config_mk`; `validate.py` enforces the
declarative `constraints:` block, and gate thresholds live under `gates:`.

### The funnel (multi-fidelity, early rejection)

Each config is pushed through gates cheapest → most expensive
([`../optimizer/cascade.py`](../optimizer/cascade.py)); the first failure
short-circuits the rest, so a full place-and-route only runs on survivors:

```
validate  (µs)   legality + declarative constraints        validate.py
elaborate (~s)   Yosys reads the parameterised RTL          run_elaborate
sim       (~s)   Verilator: real correctness + cycles       runner.run_sim
proxy     (s–m)  Yosys synth + OpenROAD STA: area + Fmax     run_synth_sta
full      (min)  full RTL→GDS: real area/timing/power        run_physical
```

The reward ([`../optimizer/cascade_reward.py`](../optimizer/cascade_reward.py))
gives an escalating penalty for early death (`stage_penalty` in the YAML) and the
full multi-objective PPA reward for survivors — fed to the **same** agents
(random/evo/ucb/bayesian), so they steer toward configs that reach deep.

```bash
python optimizer/run_cascade_optimizer.py --agent evo --trials 30        # full funnel
python optimizer/run_cascade_optimizer.py --max-stage proxy --trials 80  # fast: no P&R
python optimizer/run_cascade_optimizer.py --platform asap7
PHYSICAL_MOCK=1 python optimizer/test_cascade.py                         # offline self-test (20 checks)
```

The run prints **funnel attrition** (how many configs reached each stage) so you
see where bad configs die — e.g. `acc_width<24` is killed at the cheap `sim` gate,
never wasting a P&R.

---

Next: [05_commands_cheatsheet.md](05_commands_cheatsheet.md).
