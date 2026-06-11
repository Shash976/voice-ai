# 04 — The Design-Space Optimizer (Stage 5)

> **This doc covers the first-generation optimizer** (the 45-config grid, the
> agents, the cascade funnel with fixed gates). It is still accurate and all of
> it still runs. The second generation — a multi-fidelity funnel with a learned
> surrogate and a trainable promotion policy, plus the repairs that motivated
> it — is documented in [08_funnel_optimizer.md](08_funnel_optimizer.md).

> **Package note:** the gen-1 code now lives under `optimizer/gen1/` (moved
> there when gen2 and common packages were added). Thin shims at the
> `optimizer/` root re-export everything, so **every command in this document
> works unchanged**. If you need to import directly, use e.g.
> `from optimizer.gen1.env import OptEnv` instead of `from optimizer.env import
> OptEnv`. The common plumbing (physical_runner, constants, recipe, validate)
> moved to `optimizer/common/`; import paths there are also shimmed.

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
- `critical_path_ns` — `3.72 + 0.02·(acc_width − 24)` ns, **calibrated to the real
  Stage-6 GDS** (was an uncalibrated `2.5 + 0.15·lanes + …` guess). The first
  nangate45 layout (docs/06) measured the critical path as the requantize Q31
  multiply at **3.72 ns (Fmax ≈ 269 MHz)**, and crucially **independent of lanes** —
  the MAC adder tree is shallower than the 32×32 multiply. So more lanes buys
  throughput without hurting Fmax, and a 5 ns clock (200 MHz) *meets* timing at every
  lane count. The old guess wrongly grew the path with lanes (16 lanes → 5.10 ns) and
  fictitiously flagged high-lane configs as timing violations.

The area/power terms remain **estimates**, clearly labeled as such, and become
*real* numbers in Stage 6 ORFS (the timing term now already is one). All measured
constants (`SW_BASELINE_CYCLES = 11,196,638`, the per-lane cycle table, the
`behavioral_cycles(lanes)` fit, `max_speedup = 576`) live in one place —
[`../optimizer/constants.py`](../optimizer/constants.py) — pinned empirically by
[`../optimizer/measure_real.py`](../optimizer/measure_real.py), which sweeps the
real Verilator sim over all lane counts. Every consumer (reward, runner,
benchmark, cascade, dashboard) imports from there; nothing duplicates the numbers.

---

## The agents — [`../optimizer/agents/`](../optimizer/agents/)

> **This is design-space exploration (DSE), not reinforcement learning.** Each
> "agent" picks one full hardware config per trial; the env scores it in a single
> step (`OptEnv.step` always returns `done=False` — there is no episode, trajectory,
> or MDP state the agents learn over). The strategies are classic black-box search,
> not policies. We keep the word "agent" only for the shared `suggest/update`
> interface; nothing here is RL.

All strategies share one interface (`BaseAgent`):

```python
config = agent.suggest(state, history)   # propose next config
agent.update(config, reward, info)        # record the result
```

| Strategy | File | What it does |
|----------|------|--------------|
| `enumerate` | `enumerate_agent.py` | **Exhaustive grid sweep — the correct tool for this 45-config space.** Evaluates every config once, reports the true optimum, zero sampling variance. |
| `random` | `random_agent.py` | Uniform random sampling — the baseline |
| `evo` | `evo_agent.py` | (μ+λ) evolutionary search (mutate the best) — the CLI default |
| `ucb` | `ucb_agent.py` | Factored UCB1 bandit — treats each axis as an independent multi-armed bandit |
| `bayesian` | `bayesian_agent.py` | Optuna TPE — needs `pip install optuna` (optional) |

The UCB agent is well-documented (`ucb_agent.py:1`) — note its rewards are normalized
to `[0,1]` with *fixed* bounds, never a running min (which would silently
re-interpret history). The bounds are **per-track**: each env exposes a
`reward_bounds` attribute (`(−12, 4.5)` for the behavioral track, `(−100, 4.5)`
for the cascade/physical tracks whose failure-penalty ladder reaches −100) and the
run scripts pass it through — otherwise the escalating penalties would all clamp
to the same normalized value and be invisible to the bandit.

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
spaces. On a tiny deterministic grid, just enumerate it — which is exactly what
`--agent enumerate` does (one pass over all 45 configs, true global optimum, no
variance). The learning strategies are there for the larger cascade space below.

There's also [`../optimizer/test_reward_sanity.py`](../optimizer/test_reward_sanity.py)
— 16 offline invariant checks on the reward function (e.g. "a timing-violating config
never out-scores the same config clocked legally"). Runs on Windows, no sim needed.

---

## How to run

### The offline benchmark + sanity tests (Windows, no sim needed)

```powershell
python optimizer/benchmark_agents.py    # agent comparison + the honest verdict
python optimizer/test_reward_sanity.py  # 16/16 reward invariants
```

These are the quickest way to see Stage 5 work — they need only Python + pyyaml.

### The live optimizer (needs the WSL-built sim + firmware)

```bash
# Prereqs (WSL): build firmware.bin and sim_picorv32 first (see doc 02)
python3 optimizer/run_optimizer.py --agent enumerate # exhaustive 45-config sweep (recommended here)
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
- Grid optimum: `{lanes:4, acc:24, clk:5}`, reward ≈ 4.0.
- This is **DSE, not RL** — single-step black-box search, no MDP. On 45 configs the
  honest tool is `--agent enumerate`; **no learning strategy beats random** here.
  Agent value needs bigger/noisier spaces (the cascade track).
- Quickest demo: `benchmark_agents.py` + `test_reward_sanity.py` (Windows, no sim).
- The behavioral sim (`sim_main.cpp`) matches the RTL on both counts that matter:
  the cycle model (latency = `n_outputs × (ceil(K/LANES) + 2)`, the `+2` being
  per-channel bias-load + requantize overhead the old `ceil(M·K/LANES)` formula
  missed) **and** accumulator saturation order (per `LANES`-chunk, not per MAC —
  which makes int16-accumulator accuracy lane-count-dependent: 47–58/64). The sim
  is rebuilt and the measured constants are pinned in `optimizer/constants.py`
  (8 lanes = 61,400 cycles/inf; 16 lanes = 46,670). Grid optimum unchanged.

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
(random/evo/ucb/bayesian), so they steer toward configs that reach deep. The
penalty ladder is **monotone in information gained**: invalid −100 < elaborate
−80 < sim −60 < proxy −40 < full-flow-fail −20. (It originally scored a failure
at full place-and-route as −100, the same as an invalid config — which taught
agents to *avoid* deep progress.)

```bash
python optimizer/run_cascade_optimizer.py --agent evo --trials 30        # full funnel
python optimizer/run_cascade_optimizer.py --max-stage proxy --trials 80  # fast: no P&R
python optimizer/run_cascade_optimizer.py --platform asap7
PHYSICAL_MOCK=1 python optimizer/test_cascade.py                         # offline self-test (20 checks)
```

The run prints **funnel attrition** (how many configs reached each stage) so you
see where bad configs die — e.g. `acc_width<24` is killed at the cheap `sim` gate,
never wasting a P&R.

The cascade's promote/kill decisions are **hard-coded gates**. The second-generation
optimizer ([doc 08](08_funnel_optimizer.md)) turns exactly those decisions into the
actions of a trainable promotion policy, adds a learned surrogate between the
fidelities, and reduces the space to the axes that measurably matter.

---

Next: [05_commands_cheatsheet.md](05_commands_cheatsheet.md).
