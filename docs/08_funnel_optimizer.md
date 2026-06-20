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

## Code layout (gen1 / gen2 / common)

The optimizer is now three packages under `optimizer/`. Every documented command works unchanged — thin shims at the `optimizer/` root re-export the real modules.

| Package | What lives there |
|---|---|
| `optimizer/gen1/` | Single-step black-box DSE: 45-config sim track, fixed-gate cascade funnel, physical track. `env.py`, `cascade.py`, `cascade_env.py`, `physical_env.py`, `reward.py`, `runner.py`, `dashboard.py`, `agents/` (random/evo/ucb/bayesian/enumerate). |
| `optimizer/gen2/` | The funnel: `funnel.py`, `surrogate.py`, `promotion_agent.py`, `candidates.py`, `build_table.py`, `benchmark_funnel.py`, `fit_surrogate.py`, `search_space_funnel.yaml`. |
| `optimizer/common/` | Shared plumbing: `physical_runner.py`, `physical_reward.py`, `cascade_reward.py`, `recipe.py`, `constants.py`, `validate.py`, `measure_real.py`, **`designs.py`**, **`knobs.py`**. |
| `optimizer/designs/` | Per-design YAML specs: `tinymac_accel.yaml`, `gcd.yaml`. |

Root-level shims (`optimizer/run_funnel_optimizer.py`, `optimizer/build_table.py`, etc.) call `runpy.run_path` into the real module — callers do not need to know about the package split.

## The modules

| File | What it is |
|---|---|
| [`../optimizer/gen2/funnel.py`](../optimizer/gen2/funnel.py) | `FunnelEnv` — gym-style environment over the fidelity ladder. `reset(config)` runs F0 and returns a 22-dim state; `step(action)` with `kill / re-proxy / promote / commit`; terminal on kill or after F3. Accepts `design`, `max_tier`, and `active_space` params so it is design-agnostic. For designs without a `tinyvad_sim` functional-eval hook, the F1 stage is skipped (depth goes F0→F2; the two F1 state slots stay zero). Runs **live** (real tools) or in **table mode** (replays logged observations, charging recorded costs against a simulated wall-clock budget). Logs every row to `results_funnel.jsonl`. |
| [`../optimizer/gen2/search_space_funnel.yaml`](../optimizer/gen2/search_space_funnel.yaml) | The evidence-reduced tinymac space: `mac_lanes` {1,2,4,8,16,32} × `accumulator_width` {16,24,32} × `clock_period_ns` continuous [3.0, 8.0] (0.5 ns grid for offline tabling) × `abc_recipe` {orfs_speed, orfs_area, plain} = **594 grid configs**; utilization/density fixed at 40 / 0.60. (For other designs the space is built dynamically from the YAML spec + KnobRegistry.) |
| [`../optimizer/common/recipe.py`](../optimizer/common/recipe.py) | The ABC recipe axis. `orfs_speed`/`orfs_area` map to ORFS's own abc scripts at *both* the F2 proxy and the F3 full flow (previously the proxy synthesized with a recipe the full flow never used). `plain` (bare `abc -liberty`) is proxy-only — ORFS hard-codes its script selection — and F3 records the effective recipe when a plain config is committed. |
| [`../optimizer/gen2/surrogate.py`](../optimizer/gen2/surrogate.py) | Per-metric quantile-GBT surrogate: `fit(rows)`, `predict(x, obs) → (μ, σ)` for area/period/power, plus `predict_reward_stats` for the composite reward. Multi-fidelity: F2 observables (proxy area, proxy WNS, FF/cell counts) enter as conditioning features with missing-indicators, so one model serves both "config only" and "config + proxy results" queries. `fit_surrogate.py` mines the existing ORFS report tree and validates by cross-validation. |
| [`../optimizer/gen2/promotion_agent.py`](../optimizer/gen2/promotion_agent.py) | The promotion policies: `PromotionAgent` (LinUCB contextual bandit over the 22-dim state — the doc-07 analysis shows a bandit is the right starting point, with PPO as a later upgrade *only if* lookahead measurably beats myopia), `FixedGateAgent` (the first-generation hard gates expressed as a policy — the baseline to beat), `RandomPromotionAgent`. |
| [`../optimizer/gen2/candidates.py`](../optimizer/gen2/candidates.py) | `CandidateGenerator` — Optuna-backed next-config proposer (see "Candidate generation" section below). |
| [`../optimizer/gen2/build_table.py`](../optimizer/gen2/build_table.py) | Resumable offline table builder over the reduced space at F0–F2. Dedupes what physics allows (cycles depend only on lanes → one sim per lane count; the proxy is keyed by lanes/acc_w/clk/recipe). `--subset strategic` = 84-config corner+axis sweep (~1.2 h); the full 594-config table is ~7 h. Accepts `--design` and `--max-tier` for non-tinymac designs. |
| [`../optimizer/gen2/benchmark_funnel.py`](../optimizer/gen2/benchmark_funnel.py) | The benchmark that adjudicates whether learned promotion beats fixed gates: random vs fixed-gate vs LinUCB driving the *real* `FunnelEnv` in table mode, ≥20 seeds, metric = simulated wall-clock to 95% of the table optimum (median and p95). Accepts `--candidates shuffled|tpe|surrogate_ucb`. |
| [`../optimizer/gen2/run_funnel_optimizer.py`](../optimizer/gen2/run_funnel_optimizer.py) | Live campaign driver — see "How to run" below. Shim at `optimizer/run_funnel_optimizer.py`. |
| [`../optimizer/common/constants.py`](../optimizer/common/constants.py) | Single source of truth for measured constants — SW baseline (11,196,638 cycles/inference), the per-lane cycle table, the `behavioral_cycles(lanes)` fit, speedup normalization caps. Everything that previously duplicated these numbers imports them from here. |
| [`../optimizer/common/designs.py`](../optimizer/common/designs.py) | `DesignSpec` dataclass + `DesignSpec.load(name_or_path)` — see "Bring your own design" below. |
| [`../optimizer/common/knobs.py`](../optimizer/common/knobs.py) | `KnobRegistry` with 24 ORFS variables in 4 tiers — see "The knob tiers" below. |

---

## Bring your own design

`DesignSpec` (`optimizer/common/designs.py`) decouples the optimizer from
tinymac. A design = an RTL file list, a top module name, a clock port, optional
RTL chparam axes, and per-platform clock ranges. Everything else (knob space,
candidate generation, FunnelEnv, physical_runner) derives from the spec at
runtime.

A new design takes roughly 10 lines of YAML in `optimizer/designs/<name>.yaml`:

```yaml
name: my_design
top:  my_top_module
rtl_files:
  - rtl/my_design/my_top.v        # relative to repo root or absolute
clock_port: clk
params: {}                         # RTL chparam axes; omit or {} if none
platforms:
  nangate45:
    clock_range_ns: [3.0, 10.0]
    default_clock_ns: 5.0
has_macros: false                  # or true, or omit for auto-detect at F2
functional_eval:
  kind: none                       # use tinyvad_sim for the TinyVAD evaluator
```

`DesignSpec.load("my_design")` resolves the YAML from `optimizer/designs/`,
resolves RTL paths (relative to repo root), computes an 8-hex RTL content hash
for variant-name invalidation, and generates SDC text with the correct platform
time unit.

**`tinymac_accel.yaml`** reproduces historical behavior exactly: canonical
search-axis names (`mac_lanes`, `accumulator_width`) with `rtl_param_name`
fields (`LANES`, `ACC_W`) for VERILOG_TOP_PARAMS emission; same RTL hash as the
legacy hard-coded path; `functional_eval.kind = tinyvad_sim` to enable F1.

**`gcd.yaml`** wraps ORFS's shipped gcd design and was run through the real
full flow as proof of generality:

| Clock (ns) | Area (µm²) | WNS (ns) | Fmax (MHz) | Timing |
|---|---|---|---|---|
| 0.8 | 684 | — | — | met |
| 0.6 | 892 | — | 1465 | 26 violations |

gcd has no RTL params (`params: {}`) and no functional eval hook, so F1 is
skipped automatically (F0→F2 only; the two F1 state slots in the 22-dim vector
remain zero).

---

## The knob tiers

`KnobRegistry` (`optimizer/common/knobs.py`) holds 24 ORFS variables in four
importance tiers, each with a verified emit line, range, and evidence note.
`--max-tier N` caps the search to tiers ≤ N everywhere (build_table,
run_funnel_optimizer). Tier-4 knobs are suppressed automatically when
`design.has_macros` is False.

| Tier | Count | Knobs | Tier label |
|------|-------|-------|------------|
| 1 | 4 | `VERILOG_TOP_PARAMS` (RTL chparams), `CLOCK_PERIOD`, `ABC_AREA` (synthesis recipe), `CORE_UTILIZATION` | Dominant |
| 2 | 6 | `CORE_ASPECT_RATIO`, `CORE_MARGIN`, `PLACE_DENSITY`, `PLACE_DENSITY_LB_ADDON`, `CELL_PAD_IN_SITES_GLOBAL_PLACEMENT`, `CELL_PAD_IN_SITES_DETAIL_PLACEMENT` | Floorplan/placement |
| 3 | 9 | `CTS_CLUSTER_SIZE`, `CTS_CLUSTER_DIAMETER`, `TNS_END_PERCENT`, `SETUP_SLACK_MARGIN`, `ROUTING_LAYER_ADJUSTMENT`, `RECOVER_POWER`, `DETAILED_ROUTE_END_ITERATION`, `MIN_PLACE_STEP_COEF`, `MAX_PLACE_STEP_COEF` | CTS/route fine-tuning |
| 4 | 5 | `MACRO_PLACE_HALO`, `MACRO_BLOCKAGE_HALO`, `RTLMP_MAX_LEVEL`, `RTLMP_WIRELENGTH_WT`, `RTLMP_BOUNDARY_WT` | Macro-only (suppressed when `has_macros=False`) |

Evidence notes for tier-1 selection: `LANES` dominates (area ×2.6 L1→L32,
cycles ×5.8); `CLOCK_PERIOD` is the strongest flow-level coupling on 46 real
builds (Fmax 113→307 MHz, area ±18%); `ABC_AREA` (i.e. abc_recipe) produced
43% synthesis-area spread across 3 recipes at fixed geometry; `CORE_UTILIZATION`
is AutoTuner's canonical #1 lever for general designs (measured <0.3% on
tinymac — a very sparse design — but dominant for denser designs like gcd).

`validate_config()` blocks known flow-crashing combinations:
`CORE_UTILIZATION > 60` with `CELL_PAD > 2` triggers a placer abort; `PLACE_DENSITY > 0.80` aborts the placer; `MIN_PLACE_STEP_COEF > MAX_PLACE_STEP_COEF` crashes. These are checked before any tool is invoked.

When ORFS knobs beyond tier 1 are active, variant names gain a knob-hash suffix
(`L4_A24_c5_r3fa2b1c9_k7f2e`) so configurations differing only in ORFS knobs
never alias in the cache.

---

## Designs with macros

The gen2 funnel works with macro-containing designs without any architectural
change. When `design.has_macros` is True (or auto-detected as True at first F2
synthesis), tier-4 knobs (`MACRO_PLACE_HALO`, `MACRO_BLOCKAGE_HALO`,
`RTLMP_MAX_LEVEL`, `RTLMP_WIRELENGTH_WT`, `RTLMP_BOUNDARY_WT`) become active
and enter the search space; otherwise they are suppressed.

The adopted macro-placement strategy is OpenROAD's own hierarchical macro
placer (RTL-MP), steered through the tier-4 knobs. At the 2–4 macro counts
this project will see (SRAM weight/activation buffers), learned
placement (AlphaChip-style) was evaluated and rejected as over-engineering:
it requires thousands of macro-placement examples to train, provides no
improvement over RTL-MP + simple knob tuning for small macro counts, and adds
a heavyweight dependency. The funnel architecture is unchanged — macros just
activate one more knob group and trigger the `MACRO_PLACEMENT_TCL` path in ORFS.

---

## Candidate generation (Optuna)

`CandidateGenerator` (`optimizer/gen2/candidates.py`) sits above the FunnelEnv
and proposes which config to evaluate next. It wraps an Optuna study and
optionally consults the fitted surrogate for UCB acquisition.

Three sampler modes:

| Sampler | What it does |
|---|---|
| `"tpe"` | Optuna TPESampler (Tree-structured Parzen Estimator) via ask/tell API. Default. Handles mixed discrete/continuous/categorical spaces with cold-start warmup (10 random trials before switching to TPE). |
| `"surrogate_ucb"` | Ranks candidates by `μ + κ·σ` from `surrogate.predict_reward_stats(x)`. Pool = full grid enumeration (when all axes are finite) or 512 random draws + one TPE ask. Re-ranked on every `update()`. Falls back to TPE without a surrogate. |
| `"random"` | Seeded uniform sampling. The offline baseline. |

**Honesty rule (F3-only tell):** only terminal F3 rewards are fed to the Optuna
study via `study.tell()`. Killed configs and proxy-only results go to a
skip-memo so they are avoided on subsequent `suggest()` calls, but they are
marked FAIL in the study rather than carrying a proxy reward value. This
ensures TPE learns the true F3 objective, not a cheaper proxy signal.

`grid_snap=True` snaps continuous axes (clock_period_ns) to 0.5 ns increments
so table-mode FunnelEnv lookups hit stored rows. `warm_start(history)` injects
historical F3 records into the study before a campaign (for transfer from a
previous run).

---

## How to run

### 22-dim promotion-policy state

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

```bash
# Build the offline table (resumable; rows append to optimizer/results_funnel.jsonl)
python3 optimizer/build_table.py --subset strategic        # 84 configs, ~1.2 h
python3 optimizer/build_table.py                           # full 594-config grid, ~7 h
python3 optimizer/build_table.py --dry-run                 # show the plan + cost estimate

# Table for a different design (gcd, tier-2 knob space):
python3 optimizer/build_table.py --design gcd --max-tier 2

# Fit / validate the surrogate on everything built so far
python3 optimizer/fit_surrogate.py                         # prints per-metric CV correlation
                                                           # writes optimizer/surrogate_n45.joblib

# Benchmark promotion policies on the table simulator
python3 optimizer/benchmark_funnel.py --seeds 20
python3 optimizer/benchmark_funnel.py --selftest           # synthetic table, fast
python3 optimizer/benchmark_funnel.py --candidates tpe     # use Optuna TPE candidate ordering
python3 optimizer/benchmark_funnel.py --candidates surrogate_ucb   # surrogate UCB ordering

# Live campaign (tinymac, default 4-axis tier-1 space):
python3 optimizer/run_funnel_optimizer.py \
    --design tinymac_accel --platform nangate45 \
    --budget-hours 4 --max-tier 1 --sampler tpe --promotion fixed

# Live campaign (gcd, tier-2 knob space, Optuna TPE):
python3 optimizer/run_funnel_optimizer.py \
    --design gcd --platform nangate45 \
    --budget-hours 4 --max-tier 2 --sampler tpe --promotion fixed

# Table-mode campaign (replay logged observations, no real ORFS):
python3 optimizer/run_funnel_optimizer.py \
    --design tinymac_accel --platform nangate45 \
    --budget-hours 4 --sampler surrogate_ucb --promotion linucb \
    --table optimizer/results_funnel.jsonl

# Self-tests (no real tools needed)
PHYSICAL_MOCK=1 python3 optimizer/funnel.py                # FunnelEnv self-test
PHYSICAL_MOCK=1 python3 optimizer/build_table.py --subset strategic --limit 5
PHYSICAL_MOCK=1 python3 optimizer/run_funnel_optimizer.py \
    --design tinymac_accel --budget-hours 0.01 --sampler tpe --promotion fixed \
    --table optimizer/results_funnel.jsonl
```

All `optimizer/run_funnel_optimizer.py` commands also work via the gen2 path:
`python3 optimizer/gen2/run_funnel_optimizer.py`.

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

## Campaign output path

Each run of `run_funnel_optimizer.py` appends one JSONL row per episode to:

```
optimizer/campaigns/<design>/<platform>/results_funnel_campaigns.jsonl
```

e.g. `optimizer/campaigns/tinymac_accel/nangate45/results_funnel_campaigns.jsonl`.
The path is printed at the end of every campaign (`Results → ...`). Override with
`--out /your/path.jsonl`. A per-campaign trace is also written alongside as
`funnel_campaign_<seed>_<ts>.jsonl`.

---

## Visualizing a campaign (`optimizer/viz/`)

Every `run_funnel_optimizer.py` campaign appends one JSONL row per episode
(`config`, `fidelity`, `f3_reward`, `episode_reward`, `best_reward`, `spent_s`).
`optimizer/viz/` turns those logs into graphs. Needs `optuna-dashboard`
(`pip install optuna-dashboard`); plotly/pandas are already present.

**Static HTML report** — reward-vs-each-parameter scatters (coloured by the
fidelity the episode died at), optimization history vs episode and vs wall-clock,
the fidelity funnel, the F3 reward distribution, plus Optuna param-importance /
slice / parallel-coordinate / contour:

```bash
# auto-finds the most-recently-modified results_funnel_campaigns.jsonl under campaigns/
python3 optimizer/viz/report.py

# point at a specific log
python3 optimizer/viz/report.py \
    --log optimizer/campaigns/tinymac_accel/nangate45/results_funnel_campaigns.jsonl --open

python3 optimizer/viz/report.py --campaign all   # pool every campaign in the file
```

Writes a single self-contained `optimizer/report_<campaign_id>.html` (Plotly via
CDN, no server). `--campaign` takes a `campaign_id`, `latest`, or `all`.

**Live Optuna dashboard** — reconstructs an Optuna study (direction=maximize,
value = `f3_reward` if reached else the `episode_reward` penalty) into a
`JournalStorage` file and launches `optuna-dashboard`. With `--live` it tails the
log and appends new episodes; the dashboard auto-refreshes, so you watch
history / importances update as the optimizer runs:

```bash
LOG=optimizer/campaigns/tinymac_accel/nangate45/results_funnel_campaigns.jsonl

# follow a live run (open http://127.0.0.1:8080/ in browser)
python3 optimizer/viz/dashboard.py --live --log $LOG

# one-shot snapshot of the latest campaign across all designs
python3 optimizer/viz/dashboard.py

# rebuild the JournalStorage file without launching the server
python3 optimizer/viz/dashboard.py --no-serve --log $LOG
```

Objective convention: killed/aborted episodes are kept (not dropped) so the
plots show *where in parameter space designs die* — that is part of "how reward
changed with parameters", and the failure-ladder penalties (−20…−100) read as
the low tail of the objective. `campaign_data.py` is the shared loader both tools
use, so the static report and the live dashboard never disagree.

Items now closed (were listed as gaps in earlier versions of this doc):
- **Optuna candidate generation** — built and validated (`gen2/candidates.py`,
  three samplers: tpe/surrogate_ucb/random, F3-only tell rule). See section above.
- **Design-agnostic input** — built (`common/designs.py`, `optimizer/designs/`
  YAML registry, proven on gcd as a second real design).
