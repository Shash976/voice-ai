# 08 — The Multi-Fidelity Funnel Optimizer (Stage 5, second generation)

This doc describes the **as-implemented** second generation of the optimizer: a
multi-fidelity evaluation funnel with a learned surrogate and a promotion policy —
the first component of this project that is structured as a genuine sequential
decision problem (and therefore trainable with RL), as opposed to the single-step
black-box search of [doc 04](04_optimizer.md).

The design rationale, measurements, and the audit that motivated all of this are in
[07_rl_pipeline_design.md](07_rl_pipeline_design.md). This doc is the operator's
view: what exists, why, and how to run it.

---

## Why a second generation?

The first-generation optimizer (doc 04) was honest about being design-space
exploration, but an audit of the whole pipeline found two kinds of problems:

**1. Correctness bugs that poisoned results.** The worst three:

| What was wrong | Consequence | What it is now |
|---|---|---|
| asap7 SDC files are in **picoseconds**; the optimizer wrote its nanosecond clock values straight in | every asap7 run was constrained to 5–20 *ps* — all 12 logged asap7 results were garbage (wns "−1861 ns") yet logged `status=ok` | `PLATFORM_TIME_UNIT` table in `physical_runner.py` converts ns→ps on SDC write and ps→ns on report parse; the poisoned log was quarantined to `results_physical_INVALID_psbug.jsonl`; verified by a fresh asap7 run (wns −0.965 ns — sane) |
| if every report regex missed, the reward silently substituted reference area/power and the **requested** clock | a parse failure scored **+1.2 reward from zero physical data** — report-format drift would crown wrong winners | `_parse_metrics` returns `PARSE_FAIL` when nothing parses; any non-ok status pays the failure penalty; missing Fmax ⇒ zero speedup. Reward only ever pays on parsed measurements |
| built-GDS results were cached by parameter values only | after any RTL edit, all previously built variants would silently return **old-RTL metrics** as fresh results | variant names embed an 8-hex hash of the RTL sources (`L4_A24_c5_r3fa2b1c9`) — an RTL edit automatically invalidates every cached build |

Plus a tail of smaller ones, all fixed: a failure-penalty ladder that punished
dying at full place-and-route *worse* than dying instantly (now monotone:
invalid −100 < elaborate −80 < sim −60 < proxy −40 < full-flow-fail −20);
the physical env silently dropping three of six config axes; ORFS timeouts
orphaning yosys/openroad processes and never being logged (now: process-group
kill, `TIMEOUT` status, logged and fed to the agent); UCB reward normalization
clamping all penalties to the same value; clock values 1.25/1.2 colliding to the
same variant name; the behavioral sim saturating the accumulator per-MAC while
the RTL saturates per-chunk (sim now matches RTL exactly — a consequence is that
int16-accumulator accuracy is lane-count-dependent, 47–58/64); and stale cycle
constants (re-measured, see below).

**2. The action space didn't match the levers that matter.** Measured on 46 real
nangate45 builds:

| Knob | Measured effect on the design |
|---|---|
| `LANES` (RTL) | area ×2.6, cycles ×5.8 — dominant |
| clock constraint | Fmax 113→307 MHz, area ±18%, power ×3 — the tool spends effort proportional to pressure |
| ABC synthesis recipe | **±43% synthesis area** — and it wasn't even wired as a working axis |
| `ACC_W` (RTL) | ±5% area, accuracy cliff at 16 |
| `CORE_UTILIZATION` / `PLACE_DENSITY` | **<0.3% / ~1.4%** — noise |

So: utilization and density were demoted to fixed constants, the ABC recipe was
promoted to a first-class axis, and the clock became a continuous design axis
whose response surface is *learned*, not assumed (the old `max(clk, 3.72ns)`
analytic cap contradicted the measurements).

---

## Architecture

```
            ┌──────────────────────────────────────────────────────┐
            │                  SEARCH CONTROLLER                   │
            │  candidate generator (TPE/BO over the config space)  │
            │  PROMOTION POLICY π(a|s) ── the RL component         │
            │  actions: {kill, re-proxy, promote, commit}          │
            └───────────────┬──────────────────────▲───────────────┘
                            │ config, action       │ observations
            ┌───────────────▼──────────────────────┴───────────────┐
            │               MULTI-FIDELITY FUNNEL                  │
            │  F0  validate + analytic cycle model      ~0 s       │
            │  F1  behavioral Verilator sim             ~5 s       │
            │  F2  yosys synth + pre-layout STA proxy   ~45 s      │
            │  F3  full ORFS RTL→GDS flow               ~7 min     │
            │  every (config, fidelity, obs) row → results_funnel.jsonl │
            └───────────────────────┬──────────────────────────────┘
                                    │ all observations
            ┌───────────────────────▼──────────────────────────────┐
            │  SURROGATE ĝ: (x, obs) → (μ, σ) per metric           │
            │  gradient-boosted quantile trees;                    │
            │  F3 prediction conditions on F2 observables          │
            └──────────────────────────────────────────────────────┘
```

The one place where this project has a *genuine* sequential decision problem is
budget allocation across fidelities: given what the cheap stages revealed about a
config, is it worth 45 s of proxy or 7 min of place-and-route? The first
generation hard-coded that decision as fixed gates. Here it is an *action*, taken
per step by a promotion policy that can be trained offline against logged funnel
traces — without spending a single new tool-minute.

Two honesty rules are built into the reward:

- **Reward only pays on F3 measurement.** Proxy results can kill a config but
  never accept one — the proxy's measured bias is pessimistic on timing, so the
  policy cannot inflate its score through cheap stages.
- **Failures pay the monotone ladder.** Deeper progress before failing is
  strictly less bad, reflecting information gained.

## The modules

| File | What it is |
|---|---|
| [`../optimizer/funnel.py`](../optimizer/funnel.py) | `FunnelEnv` — gym-style environment over the fidelity ladder. `reset(config)` runs F0 and returns a 22-dim state; `step(action)` with `kill / re-proxy / promote / commit`; terminal on kill or after F3. Runs **live** (real tools) or in **table mode** (replays logged observations, charging recorded costs against a simulated wall-clock budget — the offline training simulator). Logs every row to `results_funnel.jsonl`. |
| [`../optimizer/search_space_funnel.yaml`](../optimizer/search_space_funnel.yaml) | The evidence-reduced space: `mac_lanes` {1,2,4,8,16,32} × `accumulator_width` {16,24,32} × `clock_period_ns` continuous [3.0, 8.0] (0.5 ns grid for offline tabling) × `abc_recipe` {orfs_speed, orfs_area, plain} = **594 grid configs**; utilization/density fixed at 40 / 0.60. |
| [`../optimizer/recipe.py`](../optimizer/recipe.py) | The ABC recipe axis. `orfs_speed`/`orfs_area` map to ORFS's own abc scripts at *both* the F2 proxy and the F3 full flow (previously the proxy synthesized with a recipe the full flow never used). `plain` (bare `abc -liberty`) is proxy-only — ORFS hard-codes its script selection — and F3 records the effective recipe when a plain config is committed. |
| [`../optimizer/surrogate.py`](../optimizer/surrogate.py) | Per-metric quantile-GBT surrogate: `fit(rows)`, `predict(x, obs) → (μ, σ)` for area/period/power, plus `predict_reward_stats` for the composite reward. Multi-fidelity: F2 observables (proxy area, proxy WNS, FF/cell counts) enter as conditioning features with missing-indicators, so one model serves both "config only" and "config + proxy results" queries. `fit_surrogate.py` mines the existing ORFS report tree and validates by cross-validation. |
| [`../optimizer/agents/promotion_agent.py`](../optimizer/agents/promotion_agent.py) | The promotion policies: `PromotionAgent` (LinUCB contextual bandit over the 22-dim state — the doc-07 analysis shows a bandit is the right starting point, with PPO as a later upgrade *only if* lookahead measurably beats myopia), `FixedGateAgent` (the first-generation hard gates expressed as a policy — the baseline to beat), `RandomPromotionAgent`. |
| [`../optimizer/build_table.py`](../optimizer/build_table.py) | Resumable offline table builder over the reduced space at F0–F2. Dedupes what physics allows (cycles depend only on lanes → one sim per lane count; the proxy is keyed by lanes/acc_w/clk/recipe). `--subset strategic` = 84-config corner+axis sweep (~1.2 h); the full 594-config table is ~7 h. |
| [`../optimizer/benchmark_funnel.py`](../optimizer/benchmark_funnel.py) | The benchmark that adjudicates whether learned promotion beats fixed gates: random vs fixed-gate vs LinUCB driving the *real* `FunnelEnv` in table mode, ≥20 seeds, metric = simulated wall-clock to 95% of the table optimum (median and p95). |
| [`../optimizer/constants.py`](../optimizer/constants.py) | Single source of truth for measured constants — SW baseline (11,196,638 cycles/inference), the per-lane cycle table, the `behavioral_cycles(lanes)` fit, speedup normalization caps. Everything that previously duplicated these numbers imports them from here. |

### The 22-dim promotion-policy state

`[config encoding (5) | F0 cycles+accuracy (2) | F1 cycles+accuracy (2) |
F2 proxy area, WNS, FFs, cells, logic levels (5) | surrogate μ,σ (2) |
incumbent best (1) | remaining budget fraction (1) | fidelity-depth one-hot (4)]`

A GNN over the netlist was considered and rejected: for one 5K-cell design family,
a dozen scalar netlist statistics carry the same signal (revisit if the design
family diversifies).

### Re-measured cycle constants

The behavioral sim's cycle model (`latency = n_outputs × (ceil(K/LANES) + 2)`,
backported from the RTL FSM) and the per-chunk saturation fix shifted the measured
cycles/inference. Now pinned in `constants.py` from real Verilator sweeps:

| lanes | 1 | 2 | 4 | 8 | 16 | 32 |
|---|---|---|---|---|---|---|
| cycles/inf | 273,130 | 152,140 | 91,650 | 61,400 | 46,670 | 39,310 |

The 45-config grid optimum is unchanged: `{lanes:4, acc:24, clk:5}` (reward 3.995).

---

## How to run

```bash
# Build the offline table (resumable; rows append to optimizer/results_funnel.jsonl)
python3 optimizer/build_table.py --subset strategic        # 84 configs, ~1.2 h
python3 optimizer/build_table.py                           # full 594-config grid, ~7 h
python3 optimizer/build_table.py --dry-run                 # show the plan + cost estimate

# Fit / validate the surrogate on everything built so far
python3 optimizer/fit_surrogate.py                         # prints per-metric CV correlation
                                                           # writes optimizer/surrogate_n45.joblib

# Benchmark promotion policies on the table simulator
python3 optimizer/benchmark_funnel.py --seeds 20
python3 optimizer/benchmark_funnel.py --selftest           # synthetic table, fast

# Self-tests (no real tools needed)
PHYSICAL_MOCK=1 python3 optimizer/funnel.py                # FunnelEnv self-test
PHYSICAL_MOCK=1 python3 optimizer/build_table.py --subset strategic --limit 5
```

Live mode is just `FunnelEnv(table=None)` — same env, real tools. Sustainable
F3 throughput on the VM is ~8 serial full flows/hour (~14/h with 2 concurrent).

---

## Validation status (all reproduced, none aspirational)

- **Units fix, end-to-end**: fresh asap7 full flow at 1.0 ns → `6_final.gds`,
  area 1433 µm², Fmax 509 MHz, wns −0.965 ns. (7 nm block is ~14× smaller than
  the 19.7K µm² nangate45 layout — sensible scaling.)
- **No reward without data**: an all-None parse now yields `PARSE_FAIL` → −100,
  never +1.2.
- **Recipe axis is real**: at L4/A24/4.0 ns the three recipes measure 14,456 /
  17,187 / 20,675 µm² synthesis cell area (plain / orfs_area / orfs_speed) — a
  43% spread, reproducing the doc-07 experiment exactly.
- **Surrogate** (fit on 44 fully-built variants): 5-fold Spearman ρ = **0.895
  area / 0.865 period / 0.95 power** (power is relative-only — ORFS power is
  activity-factor fiction, 99.8% combinational).
- **Strategic table built**: 252 rows (84 configs × F0/F1/F2), zero errors,
  72.6 min wall-clock, in `optimizer/results_funnel.jsonl`.
- **Benchmark, honestly reported**: on a synthetic table (20 seeds), fixed gates
  reach 95%-of-optimum in median 0.59 h @ 90% success vs random 1.09 h @ 85%;
  **cold-start LinUCB loses (50% success)**. Same finding pattern as the
  first-generation benchmark: learning must earn its keep. The bandit's fair
  test — pretrained on the real table, with F3 rows present — is the next
  experiment, and if it cannot beat fixed gates there, the funnel ships with
  fixed gates + BO and the negative result gets recorded like doc 04 did.

## What remains

1. **F3 rows for the table** — commit top F2-screened candidates through the full
   flow (~7 min each) so the table simulator has real terminal rewards; then the
   real LinUCB-vs-fixed-gates benchmark, with `--pretrain-campaigns`.
2. **Full 594-config F2 table** (~7 h, resumable).
3. **asap7 co-search / transfer test** — the funnel's F4; unblocked by the units
   fix. Train the surrogate on nangate45, measure sample-efficiency on asap7.
4. **Requantize pipelining** (RTL) — the measured ~3.7 ns critical-path wall that
   no synthesis recipe moves; expected ~2× Fmax for <1% throughput cost. The RTL
   hash in variant names means the existing result corpus survives the edit
   correctly (old variants become unreachable instead of stale).
5. **PPO upgrade of the promotion policy** — only if table-simulation shows the
   bandit's myopia measurably loses to lookahead.
