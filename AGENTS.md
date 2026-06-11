# AGENTS.md — AI Agent Orientation

This file is for AI coding assistants working on the optimizer subsystem of the
TinyMAC accelerator project. Read it before touching anything in `optimizer/`.

---

## What this project is (one paragraph)

We are designing a custom int8 MAC accelerator chip — a RISC-V CPU (PicoRV32) plus
a hardware matrix-multiply unit (TinyMAC) — using open-source EDA tools (Verilator,
Yosys, OpenROAD). The hardware has tunable parameters (MAC lanes, accumulator width,
clock period, place-and-route density…) and we want to find the best combination for
the TinyVAD inference workload. The optimizer (`optimizer/`) is the system that
drives that search. Full context: `CLAUDE.md`, `docs/04_optimizer.md`,
`docs/06_rtl_to_gds.md`.

---

## Current state — honest summary

The existing agents (`random`, `evo`, `ucb`, `bayesian`, `enumerate`) are
**black-box design-space exploration (DSE) strategies, not reinforcement learning**:

- Each trial proposes one complete hardware config.
- The environment scores it in a single step (`OptEnv.step` always returns
  `done=False` — no episode, no MDP state, no trajectory).
- The agents have no memory of *how* they arrived at a config, only what rewards
  past configs produced.
- On the 45-config sim grid, `--agent enumerate` is provably optimal: exhaustive,
  zero variance. No learning agent beats random here (documented honestly in
  `docs/04_optimizer.md`).

The cascade optimizer (`run_cascade_optimizer.py`) extends this to a ~27 K-config
space with a multi-fidelity funnel (validate → elaborate → sim → proxy → full), but
the agents driving it are still the same black-box strategies — they receive higher
reward for configs that reach deeper funnel stages, but they do not learn a *policy*
over trajectories.

**The first genuinely RL-shaped component now exists**: `optimizer/funnel.py`
(`FunnelEnv`) turns the funnel's promote/kill decisions into per-step *actions*
(`kill / re-proxy / promote / commit`) over a 22-dim observation, and
`optimizer/agents/promotion_agent.py` provides a LinUCB contextual bandit policy
plus the fixed-gate and random baselines it must beat. The env runs live (real
tools) or replays a logged table (`build_table.py` → `results_funnel.jsonl`) so
policies train offline at zero tool cost. `optimizer/benchmark_funnel.py` is the
honest adjudicator (wall-clock-to-95%-of-optimum, ≥20 seeds). Design rationale:
`docs/07_rl_pipeline_design.md`; operator guide: `docs/08_funnel_optimizer.md`.

**Do not re-label the existing DSE agents as RL.** The codebase documents this
distinction deliberately and carefully. Introducing misleading terminology will be
reverted. The promotion policy is the only component with an MDP behind it — and
even there, the benchmark decides whether learning earns its keep (current honest
result: a *cold-start* bandit loses to fixed gates; the pretrained-on-real-table
test is open).

---

## The direction: toward true RL

The goal is to evolve the optimizer from black-box DSE toward genuine reinforcement
learning — an agent that learns a *policy* (state → action) from experience, improves
over episodes, and ideally generalises across chip design problems.

### Why the cascade space is the natural MDP

The multi-fidelity cascade already has the structure of a sequential decision problem:

```
State    what we know so far: configs tried, rewards observed, budget remaining,
         partial metrics from earlier funnel stages (e.g. Yosys cell count at
         the "elaborate" stage before committing to full P&R)
Action   which config to evaluate next, and at what fidelity
Reward   improvement over the current best (sparse), or weighted PPA gain,
         or funnel depth reached (denser signal for the cascade)
Episode  one optimizer run with a fixed trial budget (e.g. 30 or 60 trials)
Policy   a function: observation → distribution over configs
```

An RL policy can exploit things black-box search cannot:
1. **Correlations across axes** — e.g. if configs with `acc_width=16` always die at
   the `sim` gate (overflow → wrong outputs), a policy should stop proposing them.
   UCB treats each axis independently; a neural policy learns joint structure.
2. **Gating decisions** — given marginal timing slack at the `proxy` stage, is it
   worth the minutes of full P&R? A policy that sees partial metrics can decide.
3. **Transfer** — train on nangate45 proxy data, deploy on asap7. If the policy
   learns "more lanes = faster but bigger" as a principle, it may need fewer trials
   on the new target.

### Concrete RL formulation to build toward

| Component | Definition |
|-----------|-----------|
| **Observation** | `[trials_remaining / budget, best_reward_so_far, last_k_(config,reward) pairs, partial_metrics_if_available]` — a fixed-size float vector |
| **Action** | index into the config grid (discrete, ~27 K actions for the cascade space; or a delta over the best-seen config) |
| **Reward** | `r_t = reward(config_t) - max(rewards_{0..t-1})` — marginal improvement; 0 or negative for non-improvements |
| **Episode** | fixed budget of N trials; terminal when budget exhausted |
| **Policy** | small MLP (~2 hidden layers, 64 units) outputting a softmax or Gaussian over actions |
| **Training** | policy gradient (PPO or REINFORCE) over rollouts; rollouts can be simulated offline against a pre-built table of (config, reward) pairs |

### Build status against this formulation

The plan above has largely been **built**, with one deliberate redirection: the
trainable policy operates over *fidelity promotion* (the genuinely sequential
decision this project has) rather than over raw config selection (where Bayesian
optimization over a surrogate is the better tool for a ≤10-dim mixed space — a
config-selection PPO would re-learn what a GP prior gets for free).

1. ~~Episode-aware env wrapper~~ — **built** as `optimizer/funnel.py` (`FunnelEnv`,
   gym-style `reset/step/done`, 22-dim observation, table-replay mode for offline
   training).
2. ~~Offline dataset~~ — **built**: `optimizer/build_table.py` (resumable) writes
   per-fidelity observation rows to `optimizer/results_funnel.jsonl`; an 84-config
   strategic F0–F2 subset is already populated from real tool runs.
3. ~~Policy~~ — **built** as `optimizer/agents/promotion_agent.py`: LinUCB
   contextual bandit first (hundreds of logged traces suffice); a PPO upgrade is
   warranted only if table-simulation shows myopia measurably losing to lookahead.
4. ~~Honest benchmark~~ — **built**: `optimizer/benchmark_funnel.py`, metric =
   wall-clock-to-95%-of-table-optimum over ≥20 seeds vs random and fixed gates.
   Current honest result: cold-start LinUCB **loses** to fixed gates; the fair
   test (pretrained on the real table, F3 rows present) is the open experiment.
5. **Generalization test (still open)** — train the surrogate on nangate45,
   evaluate sample-efficiency on asap7 (now unblocked: the asap7 flow runs
   correctly and the first 7 nm GDS exists).

Supporting pieces that did not exist when this roadmap was written:
`optimizer/surrogate.py` (multi-fidelity quantile-GBT, CV Spearman ρ ≈ 0.9 on
area), `optimizer/recipe.py` (ABC synthesis recipe as a search axis at both proxy
and full-flow fidelity), `optimizer/constants.py` (measured cycle constants,
single source of truth), `optimizer/search_space_funnel.yaml` (the evidence-reduced
594-config space — utilization/density measured at <1.4% effect and frozen).

---

## Key files

| File | Role |
|------|------|
| `optimizer/env.py` | `OptEnv` — the base environment; `step(config)` → `(state, reward, done, info)` |
| `optimizer/cascade.py` | multi-fidelity funnel (fixed gates); `run_cascade(config)` → stage reached + metrics |
| `optimizer/cascade_env.py` | `CascadeOptEnv` — wraps the fixed-gate funnel as an env |
| `optimizer/funnel.py` | `FunnelEnv` — the funnel with promotion *actions*; live or table-replay mode |
| `optimizer/surrogate.py` | multi-fidelity surrogate: `(config, cheap obs) → (μ, σ)` per metric |
| `optimizer/recipe.py` | ABC synthesis-recipe axis, shared between proxy and full flow |
| `optimizer/build_table.py` | resumable offline F0–F2 table builder → `results_funnel.jsonl` |
| `optimizer/benchmark_funnel.py` | promotion-policy benchmark on the table simulator |
| `optimizer/constants.py` | measured constants (SW baseline, per-lane cycles, speedup caps) — import, never duplicate |
| `optimizer/reward.py` | reward computation: `compute_reward()`, `real_speedup()`, `critical_path_ns()` |
| `optimizer/physical_runner.py` | one config through real ORFS; unit conversion, RTL-hash variant names, timeout discipline |
| `optimizer/search_space.yaml` | 45-config sim space (3 axes) |
| `optimizer/search_space_full.yaml` | ~27 K cascade space (6 axes, all wired to real tools) |
| `optimizer/search_space_funnel.yaml` | 594-config reduced funnel space (lanes × acc_w × clk × recipe) |
| `optimizer/agents/base_agent.py` | `BaseAgent` interface: `suggest(state, history)` + `update(config, reward, info)` |
| `optimizer/agents/enumerate_agent.py` | exhaustive grid sweep — the ground truth for comparison |
| `optimizer/agents/promotion_agent.py` | LinUCB promotion policy + fixed-gate and random baselines |
| `optimizer/benchmark_agents.py` | offline benchmark (uses a table of real measured results, no live sim) |
| `optimizer/measure_real.py` | pins SW baseline + per-lane cycle counts from real Verilator sweeps |

---

## Rules — what NOT to do

- **Don't add RL framing to the existing DSE agents.** `ucb`, `evo`, `random`,
  `bayesian`, `enumerate` are correctly labeled black-box search. Only new
  trajectory-learning agents qualify as RL.
- **Don't claim the cascade funnel is RL** just because it has stages. Each trial
  is still single-step scoring per config.
- **Don't invent axes or reward terms** that aren't wired to a real tool or sim.
  The project rule is "score real results, don't guess performance."
- **Don't add heavy ML framework dependencies** (PyTorch, JAX, etc.) without
  confirming they're available in both Windows (Python optimizer) and WSL (sim
  runner). Prefer stdlib + numpy; use torch only if it's already in the venv.
- **Don't re-run the full P&R flow** (`run_physical_optimizer.py`) in tests.
  `PHYSICAL_MOCK=1` exists for exactly this reason. Mock mode is the CI path.

---

## Quick orientation commands

```bash
# See the honest DSE baseline — no sim needed (Windows):
python optimizer/benchmark_agents.py
python optimizer/test_reward_sanity.py

# Exhaustive sweep of the 45-config sim space (WSL, needs firmware + sim built):
python3 optimizer/run_optimizer.py --agent enumerate

# Run the cascade funnel in mock mode (no ORFS, no sim):
PHYSICAL_MOCK=1 python optimizer/test_cascade.py

# Explore the full cascade space (proxy stage only, no P&R):
python3 optimizer/run_cascade_optimizer.py --max-stage proxy --agent random --trials 20

# FunnelEnv + promotion-policy track (no tools needed for these):
PHYSICAL_MOCK=1 python3 optimizer/funnel.py            # env self-test
python3 optimizer/benchmark_funnel.py --selftest       # policy benchmark, synthetic table
python3 optimizer/build_table.py --dry-run             # what a real table build would run

# Understand the reward function interactively (91,650 = measured cycles at 4 lanes):
python -c "from optimizer.reward import compute_reward, critical_path_ns; \
           c={'mac_lanes':4,'accumulator_width':24,'clock_period_ns':5}; \
           print(compute_reward(c, 1.0, 91650))"
```
