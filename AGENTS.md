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

**Do not re-label the existing agents as RL.** The codebase documents this distinction
deliberately and carefully. Introducing misleading terminology will be reverted.

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

### What needs to be built (in order)

1. **Episode-aware env wrapper** (`optimizer/rl_env.py`).
   Today `OptEnv.step` always returns `done=False`. Wrap it as a proper `gym.Env`:
   - observation space: the fixed-size vector above
   - action space: `Discrete(len(all_configs))`
   - episode termination: `trials_remaining == 0`
   - `reset()`: fresh episode, optional warm-start from a prior result file

2. **Offline dataset** of (config, cascade_reward) pairs covering the cascade space
   at `--max-stage proxy` (no full P&R; takes hours once on the VM, then cached).
   Store as `results_proxy_full.jsonl`. This is the RL training corpus — rollouts
   are simulated against it without re-running synthesis.

3. **Policy net + training loop** (`optimizer/agents/rl_agent.py`).
   - `RLAgent(BaseAgent)` satisfies the existing `suggest/update` interface so it
     can be dropped in as `--agent rl` with zero changes to `run_optimizer.py`.
   - Internally maintains a replay buffer of (observation, action, reward) tuples
     accumulated across trials, and runs gradient updates after each episode.
   - Start with REINFORCE (simpler); upgrade to PPO if variance is a problem.

4. **Honest benchmark** against the existing bar.
   `benchmark_agents.py` already measures regret and trials-to-optimum. Add `rl`
   to the comparison. The bar to beat is `--agent random` (which no current agent
   clears on the 45-config space). Report sample efficiency: trials to reach 95%
   of the grid-optimum reward, averaged over N random seeds.

5. **Generalization test** (stretch).
   Train on nangate45 proxy data → evaluate on asap7 proxy data. If the policy
   needs fewer trials than random to find the asap7 optimum, that is a meaningful
   transfer result.

---

## Key files

| File | Role |
|------|------|
| `optimizer/env.py` | `OptEnv` — the base environment; `step(config)` → `(state, reward, done, info)` |
| `optimizer/cascade.py` | multi-fidelity funnel; `run_cascade(config)` → stage reached + metrics |
| `optimizer/cascade_env.py` | `CascadeOptEnv` — wraps the funnel as an env |
| `optimizer/reward.py` | reward computation: `compute_reward()`, `real_speedup()`, `critical_path_ns()` |
| `optimizer/search_space.yaml` | 45-config sim space (active; 3 axes) |
| `optimizer/search_space_full.yaml` | ~27 K cascade space (6 axes, all wired to real tools) |
| `optimizer/agents/base_agent.py` | `BaseAgent` interface: `suggest(state, history)` + `update(config, reward, info)` |
| `optimizer/agents/enumerate_agent.py` | exhaustive grid sweep — the ground truth for comparison |
| `optimizer/benchmark_agents.py` | offline benchmark (uses a table of real measured results, no live sim) |
| `optimizer/measure_real.py` | pins SW baseline + per-config cycle counts from real Verilator runs |

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

# Understand the reward function interactively:
python -c "from optimizer.reward import compute_reward, critical_path_ns; \
           c={'mac_lanes':4,'accumulator_width':24,'clock_period_ns':5}; \
           print(compute_reward(c, 0.95, 43125))"
```
