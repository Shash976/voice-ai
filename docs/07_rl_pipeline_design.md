# RL Pipeline Architecture for TinyMAC Physical Design

*Produced 2026-06-10. Every number in this document was measured on this machine
(8 cores, 30 GB RAM, ORFS at `/opt/OpenROAD-flow-scripts`, yosys 0.64, OpenROAD 26Q2)
or cited to file:line. Phase-0/1 investigation used parallel sub-agents; the
Phase-4/5 experiments were run inline after a session-limit interruption killed
the second agent fan-out.*

---

## Phase 0 — Repository Orientation (synthesis)

Three parallel sub-agents read all markdown, all of `optimizer/`, and all of
`physical/orfs/` + the ORFS install. Consolidated picture:

- **Three environments**: `OptEnv` (behavioral sim + analytic proxies, 45-config
  grid), `PhysicalOptEnv` (real ORFS full flow), `CascadeOptEnv` (multi-fidelity
  funnel: validate → elaborate → sim → proxy → full; ~27–32K configs over 6 axes).
- **Five agents** (`random`, `evo`, `ucb`, `bayesian`/Optuna, `enumerate`) — all
  single-step black-box DSE; `step()` never returns `done=True` (env.py).
  `AGENTS.md` correctly distinguishes this from RL and proposes an episode-based
  MDP (state = search history + budget, action = config index, reward = marginal
  improvement, train REINFORCE/PPO on an offline proxy table).
- **Reward** (`reward.py`): `real_speedup = SW_BASELINE / (cycles × max(clk, crit_path))`,
  crit_path GDS-calibrated to 3.72 ns + 0.02·(ACC_W−24), LANES-independent;
  area/power proxies; weights from YAML.
- **ORFS interface** (`physical_runner.py`): per-variant `config_<v>.mk` +
  `constraint_<v>.sdc`, `VERILOG_TOP_PARAMS` chparam, `FLOW_VARIANT` isolation,
  regex report parsing, `PHYSICAL_MOCK=1` offline mode, plus a fast
  `run_synth_sta` proxy (yosys + pre-layout OpenROAD STA).
- **Existing data**: 46 fully-built nangate45 variants with complete reports;
  40 behavioral trials; 12 asap7 physical trials (all invalid — see V1);
  30 cascade trials.
- **Platforms installed**: nangate45, asap7, sky130hd (+gf180/ihp, unused).

Doc claims audit: the headline numbers (64/64, 191×/258×, Fmax 269 MHz,
19,738 µm², grid optimum {L4, A24, clk5}) all trace to real artifacts, with
three exceptions found and quantified in Phase 1 (host test now 63/64; FF count
is 230 not 231; cycle-model constants stale).

---

## Phase 1 — Functional Vulnerability Audit (merged & ranked)

Three parallel audit agents (optimizer / ORFS flow / firmware+sim+RTL), findings
verified by execution where possible. Deduplicated and ranked:

### blocks_correctness

| # | Finding | Location | Root cause |
|---|---------|----------|------------|
| **V1** | **asap7 SDC unit bug (ps vs ns)**: asap7 SDCs are in *picoseconds* (ORFS asap7 gcd uses `clk_period 310`); the repo writes the optimizer's ns values (5.0–20.0) straight in, constraining every asap7 run to 5–20 **ps**. All 12 records in `results_physical.jsonl` are garbage (wns −1861 "ns" is really ps; 865 mW for a 1309 µm² block), logged `status=ok`, `timing_met=false`. Parser stores ps values in `*_ns` keys with no conversion. | `physical/orfs/make/asap7/tinymac_accel/constraint.sdc:13`, `physical_runner.py:125–131, 196–199`, `sweep.sh:73–74` | ns assumed everywhere; asap7 liberty time unit is ps |
| **V2** | **Silent all-None parse scores +1.2 reward**: `run_physical` flips to FAIL only if the report *file* is missing, not if every regex misses (`physical_runner.py:262`). With all-None metrics, `physical_reward.py:97–98` substitutes reference area/power and line 50 falls back to the *requested* clock with no Fmax cap → verified reward **+1.2, real_speedup 630×, from zero physical data**. A report-format drift would silently crown wrong winners. | `physical_runner.py:103`, `physical_reward.py:50,97–98` | No "parsed anything" assertion; `or AREA_REF` defaults |
| **V3** | **Stale-results cache survives RTL edits**: flow is skipped when `results/<plat>/<design>/<variant>/6_final.gds` exists (`physical_runner.py:242`, `sweep.sh:91`). The variant name encodes parameters only — after any `rtl/accel/*.v` edit (requantize pipelining is on the roadmap!), all 46 built variants silently return old-RTL metrics as `ok`. | `physical_runner.py:66–80,242`, `sweep.sh:91–93` | Cache key = params, no RTL content hash, no clean step |

### degrades_results (top 10, ordered by impact)

| # | Finding | Location |
|---|---------|----------|
| V4 | **Cascade penalty inversion**: failing at *full P&R* (deepest stage) scores −100 — equal to "invalid", *worse* than dying at elaborate (−80)/sim (−60)/proxy (−40). Verified numerically. Teaches the agent to avoid deep progress. | `cascade_reward.py:39`, `physical_reward.py:84–86` |
| V5 | **`max_speedup=576` saturates the cascade space**: derived for the 45-config grid; in the 27K space, 16 lanes → 698× and 32 lanes → 847×, both clamp the log₂ norm to exactly 1.0 — the speedup term is flat across the entire fast frontier; ranking degenerates to area/power. | `search_space_full.yaml:109` |
| V6 | **Stale cycle-model constants**: `sim_main.cpp` now implements `latency = n_outputs×(ceil(K/LANES)+2)` (verified against the RTL FSM: S_INIT_CH + MAC chunks + S_REQ, bit-exact in the TB) but `AVG_CYCLES`, the `behavioral_cycles` fit (`28000+242000/lanes`), and `max_speedup` are pinned to the old model. Rebuilt the sim here: 8 lanes 58,577 → **61,399** (+4.8%); 16 lanes 43,447 → **46,669** (+7.4%); whole-inference error +1.2% (L1) to +8.6% (L16), 10.7% at L32. Grid optimum unchanged ({4,24,5}, reward 4.0117 → 3.9926). The "~12.5%" in CLAUDE.md is the FC0-layer-only figure. | `benchmark_agents.py:47`, `physical_reward.py:32`, `cascade.py:49` |
| V7 | **`PhysicalOptEnv.step` drops 3 of 6 axes**: only lanes/acc_w/clk are forwarded; util/density/abc silently evaluate the same default build (and `lru_cache` aliases them) — agents learn noise on those axes. Only the cascade path forwards them. | `physical_env.py:53` |
| V8 | **`abc_strategy='speed'` is a no-op that forks the namespace**: config.mk only emits `ABC_AREA=1` for `'area'`, but the variant name gets a `_speed` suffix → flow-identical builds under two names, defeating GDS reuse and polluting the space ('speed' is the YAML default!). Verified: `config_L4_A28_c5p0_d0p45_speed.mk` contains no ABC line. | `physical_runner.py:78–79,162` |
| V9 | **UCB normalisation destroys penalty structure**: fixed bounds [−12, 4.5] clamp −40/−60/−80/−100 all to 0.0 — the escalating-penalty ladder is invisible to the UCB agent on physical/cascade tracks. | `agents/ucb_agent.py:58` |
| V10 | **Timeout handling**: `subprocess.run(timeout=…)` kills only the bash wrapper (no `start_new_session`/process-group kill) → orphaned yosys/openroad keep burning CPU; `TimeoutExpired` propagates, the per-variant log is never written, the trial is neither logged nor `agent.update()`d, and the agent can re-propose the same config for another 40-min timeout. `lru_cache` also memoises transient FAILs forever. | `physical_runner.py:249`, `run_physical_optimizer.py:149–151` |
| V11 | **Variant-name float collision**: clk formatted to one decimal → `variant_name(4,24,1.25) == variant_name(4,24,1.2)`; colliding config silently reuses the other's GDS. Current grids are collision-free but nothing validates membership outside the cascade. | `physical_runner.py:66` |
| V12 | **Host test regression**: `test_infer_host` now fails 63/64 (v04 logits [−39,42] vs [−42,44], diff 3 > ±2 LSB tolerance; labels still 64/64). Predates the im2col commit — header/claim drift. | `firmware/tinyengine_port/` artifacts |
| V13 | **Behavioral/RTL saturation-order mismatch at ACC_W<32**: behavioral sim saturates per-MAC (`sim_main.cpp:141,169`); RTL saturates once per LANES-chunk (`tinymac_accel.v:124`, mirrored by the TB golden). The accuracy table `{16: 0.734}` is measured under the wrong semantics and RTL accuracy at A16 is LANES-dependent. | `sim_main.cpp:141`, `tinymac_accel.v:124` |
| V14 | **`sweep.sh` ignores make's exit status** (success = report-file existence; stale report + new failure = recorded ok); no timeout; CSV truncated per run. `?=` in sweep.sh config (vs `=` in physical_runner) is environment-overridable. | `sweep.sh:21,54,85,94–108` |

Cosmetic (recorded, not blocking): proxy/elaborate log paths omit platform;
mock metrics internally inconsistent (crit 3.82 vs fmax 269→3.717); 230 vs 231
FFs; "42/42" stale string in CLAUDE.md; evo agent's memory pruning freezes
elites; `timing_met` dead-default; duplicate YAML constraint; doc drift items.

---

## Phase 2 — Fix Plan (prioritized, effort vs impact)

Order of operations (S = <30 min, M = ~2 h, L = ~1 day):

1. **[S] V2 — parse-fail → FAIL**: in `_parse_metrics`, if `area_um2` *or*
   (`fmax_mhz` and `wns_ns`) are None after parsing, return `status="PARSE_FAIL"`;
   in `physical_reward.py` treat any non-ok status as failure, remove the
   `or AREA_REF` / requested-clock fallbacks. *Do this first: it converts every
   other silent failure mode into a visible one.*
2. **[S] V1 — asap7 units**: single source of truth `PLATFORM_TIME_UNIT = {"nangate45": 1.0, "asap7": 1000.0}`
   in `physical_runner.py`; multiply the optimizer's ns clock when writing the
   asap7 SDC (`5.0 ns → 5000`); divide parsed wns/tns/period by 1000 for asap7
   before storing in `*_ns` keys. Fix `asap7/tinymac_accel/constraint.sdc` default
   to `1000` (1 ns). **Quarantine `results_physical.jsonl`** (rename to
   `results_physical_INVALID_psbug.jsonl`) so resume/warm-start never feeds it
   back.
3. **[S] V4 — penalty ladder**: in `cascade_reward.py`, intercept
   `failed_stage == 'full'` and score −20 (deeper than proxy's −40, reflecting
   information gained), not the −100 fall-through.
4. **[S] V5 — re-derive `max_speedup`** for the cascade space from the *new*
   cycle model at lanes=32, clk=crit-path: ~847× → set 1024 (log₂ headroom);
   keep the 45-space YAML at 576.
5. **[S] V8 — abc aliasing**: treat `'speed'` as default in `variant_name`
   (no suffix) or emit `export ABC_AREA = 0` explicitly and keep the suffix —
   pick one; the two notions of default must agree.
6. **[M] V3 — RTL content hash in the cache key**: `variant_name` gains an
   8-hex digest of the three RTL files' contents (e.g. `L4_A24_c5p0_r3fa2b1c9`);
   old undigested dirs become unreachable naturally. Also fixes V11 by
   formatting clk with `{:.4g}` and validating against the space's choices.
7. **[M] V6 — re-pin constants**: run the rebuilt `sim_picorv32` (done here,
   binary now current) via `measure_real.py` extended to *actually* sweep and
   write: `AVG_CYCLES` (8→61,399; 16→46,669; measure 1/2/4/32), the
   `behavioral_cycles(lanes)` fit, `max_speedup`. Make `SW_BASELINE_CYCLES` a
   single constant imported everywhere (currently duplicated ×3).
8. **[M] V7 — forward all knobs**: `PhysicalOptEnv.step` builds kwargs from
   `config.get(...)` for util/density/abc and passes them to `run_physical`.
9. **[M] V10 — timeout discipline**: `start_new_session=True` +
   `os.killpg` on `TimeoutExpired`; convert to `status="TIMEOUT"` result (logged,
   `agent.update()`d, written to the variant log); replace `lru_cache` with an
   explicit dict that does **not** memoise TIMEOUT/transient failures.
10. **[M] V9 — per-track UCB bounds** (or tanh-squash): pass reward bounds from
    the env (`OptEnv` [−12, 4.5]; cascade [−100, 4.5]).
11. **[M] V12/V13 — model truth**: regenerate headers from the canonical
    `tiny_vad_int8.tflite` and re-verify (decide whether v04 means tolerance →
    ±3 LSB or a real regression); change behavioral saturation to per-chunk to
    match RTL (or document A16 accuracy as RTL-undefined and gate A16 out).
12. **[S] V14 — sweep.sh**: `|| status=FAIL` on make, `timeout(1)` wrapper,
    append CSV with a run-id column, `=` not `?=`.

No conflicts among these; items 1–5 are independent and can land in one PR.

---

## Phase 3 — Current RL State Assessment

**What exists**: a clean DSE harness with the right *interfaces* (suggest/update
agents, JSONL logging, YAML spaces, mock mode, a genuine multi-fidelity funnel)
and the right epistemics (`AGENTS.md` refuses to call it RL; benchmark proves
no learner beats random on 45 configs). What does **not** exist: any trajectory,
any policy, any state beyond "history of (config, reward) pairs", any surrogate
model, any budget-awareness.

**The fundamental architectural limitation is not the algorithm — it is that
the action space the agents see does not match the levers that matter, and the
reward model contradicts measured tool behavior:**

1. **Wrong levers.** Measured: util and density move final area by **<0.3% and
   ~1.4%** respectively (matched triples: L2_A24_c5p0 u30/50/60 → 15,787/15,756/15,753 µm²;
   L4_A28_c5p0 d0.45→0.65 → 17,789→17,542 µm²), while the ABC recipe — currently
   *broken* as an axis (V8) and absent from the proxy — moves synthesis area by
   **43%** (14,426→20,675 µm²). Two of six axes are noise; the highest-variance
   synthesis lever isn't wired.
2. **Wrong clock model.** The reward assumes `achieved = max(clk, 3.72)`. Measured
   across 46 builds: achieved Fmax **tracks the constraint** — 113 MHz at a 20 ns
   constraint, 205–218 at 5 ns, 260–270 at 2 ns, up to **307 MHz at 0.5 ns**
   (L2_A16_c0p5). The tool spends effort proportional to pressure; there is no
   single "critical path" constant. The clock constraint is a *design input*, and
   the area/power/Fmax response to it is the single strongest flow-level coupling
   in the data (L4_A24: area 16,685→19,738 µm² and power 330→1020 mW as clk goes
   10→2 ns).
3. **Missing surrogate.** Every evaluation is either free (analytic) or 45 s–7 min
   (real); nothing learns the mapping in between, so every agent restarts from
   zero structure.
4. **Episode structure exists in the funnel but the agent can't see it** — the
   cascade makes promote/kill decisions with fixed gates; the agent only sees the
   final scalar. The one genuinely sequential decision problem in this project
   (allocate budget across fidelities) is hard-coded.

**Baseline anchor (Experiment 1, run fresh here)** — `L4_A24_c4p0`, nangate45,
full flow, no cache:

| Stage | Wall-clock |
|---|---|
| synth (yosys) | 32 s |
| floorplan (+pdn/tapcell) | 23 s |
| place (gp+resize+dp) | 37 s |
| cts | 67 s |
| route (grt 84 s + droute 112 s) | 196 s |
| finish/report/gds | 24 s |
| **Total** | **6 m 51 s** (user 22 m 36 s across 8 cores) |

Final: area **18,769 µm²** (45% util), WNS **−0.16 ns** @ 4.0 ns, TNS −1.26,
8 setup violations, period_min 4.16 ns (Fmax 240 MHz), power 825 mW
(99.8% combinational — default activity factors, treat as relative only).
Note Fmax *fell* vs the 2 ns-constrained build (269 MHz): effort follows
constraint, confirming point 2.

---

## Phase 4 — Paradigm Evidence and Architecture

### Paradigm evidence (gathered inline; agents were killed by session limits)

**P1 — Logic-synthesis RL (PrefixRL / ABC-RL / ERL-LS).**
Experiment (11 recipes, L4_A24 flattened, nangate45, pre-layout STA @ 4 ns SDC):

| Recipe | Area µm² | reg2reg min period (pre-layout) | WNS @4ns (incl. IO paths) | yosys time |
|---|---|---|---|---|
| plain `abc -liberty` (= current proxy) | 14,456 | 1.15 ns | −3.23 | 31 s |
| `abc -fast` | 16,280 | 1.46 ns | — | 19 s |
| ORFS `abc_speed.script` −D 1/2/4/8 | 20,675 (identical all D) | 1.24 ns | −0.23 | 60–64 s |
| ORFS `abc_area.script` −D 4 | 17,187 | 0.92 ns | −0.75 | 25 s |
| heavy resyn2-style custom | 18,418 | 1.08 ns | — | 25 s |
| balance-only | 16,250 | 1.46 ns | — | 18 s |
| hierarchical (no flatten) | 14,427 | 0.79 ns | −2.97 | 43 s |

Findings: (a) **43% area / 59% delay spread** — the recipe is a real lever;
(b) **`-D` is a no-op with the ORFS speed script in yosys 0.64** (bit-identical
stats for D ∈ {1,2,4,8}) — delay-targeted mapping is not actually reachable
through the standard knob; (c) plain/hier netlists are unbuffered → their small
area is partly fictitious (PnR resizer adds it back) and their WNS is fanout-
distorted — exactly the proxy's recipe, see Exp 3 bias; (d) the Q31 requantize
multiply is mapped wholesale by ABC in every recipe; recipe choice does not move
the *post-route* 3.7 ns wall — only RTL restructuring (pipelining) does.
**Verdict**: sequential AIG-op RL (state = AIG stats, action = next abc op) is
*technically* wireable (yosys `abc -script +cmd;cmd;…` accepts arbitrary
sequences; ~15-op vocabulary: `b, rw, rwz, rf, rfz, resub, dch, st, &syn2, &nf, map, …`)
but unjustified at 5K cells with 3–5 recipes spanning the reachable space:
**expose the recipe as a categorical axis** {orfs_speed, orfs_area, plain}, and
get delay through RTL pipelining, not AIG surgery. Revisit PrefixRL-granularity
only if a custom MAC/adder-tree generator (LANES ≥ 16) becomes a design axis.

**P2 — Macro placement RL (AlphaChip / EfficientPlace).**
`synth_stat.txt`: **0 memory bits, no macros** — operands stream through ports
(`tinymac_accel.v:38–69`); the design is pure standard-cell at 14–44K µm².
Floorplan DOF is {CORE_UTILIZATION, ASPECT_RATIO, pin placement} and measured
sensitivity of util is <0.3%. **Verdict: not applicable today.** It becomes
relevant exactly when the roadmap adds SRAM weight/activation buffers
(~17 KB weights + ~2 KB activations → 2–4 fakeram/SRAM macros): at 2–4 macros,
placement is enumerable/analytic — even then AlphaChip-style learned placement
is over-engineering; OpenROAD's macro placer + a coordinate axis in the config
space suffices.

**P3 — Routing / joint place-route RL (DeepPCB / DeepPR).**
Census over all 46 built variants: detailed route converges to **0 DRC
violations in every single build**, including L32 @ density 0.75 and util 70
(starts at 1.5–3.3K violations, converges within the 64-iteration budget).
Routing is a **non-binding constraint** for this family on nangate45.
**Verdict: routing belongs in the reward as a failure gate (DRC count > 0 after
final iteration = fail), nothing more.** Re-examine after asap7 retarget (7 nm
pin access is genuinely harder) and after util targets rise — the trigger is
the first build where droute fails to converge.

**P4 — Agentic tool-tuning (Auto-PPA / Self-Evolved-ABC).**
ORFS exposes **242 documented variables** (synth 47, floorplan 69, place 40,
CTS 29, grt/route 49, final 15 — `flow/scripts/variables.yaml`). Measured
sensitivity on this design concentrates in exactly four:

| Knob | Measured effect |
|---|---|
| LANES (RTL) | area ×2.6 (14.3K → 43.8K µm², L1→L32); cycles ×5.8 |
| clock constraint | Fmax 113→307 MHz; area ±18%; power ×3 at fixed RTL |
| ABC recipe | area ±43% at synthesis (post-route effect smaller but real) |
| ACC_W (RTL) | area ±5%, accuracy cliff at 16 |
| CORE_UTILIZATION / PLACE_DENSITY | **<0.3% / ~1.4%** — fix at defaults |

DDPG-style continuous control over hundreds of knobs is solving a problem this
design does not have; the effective dimensionality is ~5 mixed knobs.
LLM-rewrites-ABC: rebuilding yosys/ABC takes minutes and the validation oracle
(this very funnel) exists, but the expected gain against a 43%-spread script
space that is barely explored is negligible — **not appropriate at this scope.**

### Architecture decision

The design space is small-mixed (4 discrete axes + 1 continuous clock), the
oracle is 7 min, a 45 s proxy ranks it at ρ≈0.9, routing never fails, and there
are no macros. No single paradigm fits whole; the right composition takes
*multi-fidelity gating* (the one structural idea that pays everywhere), a
*learned surrogate* (the missing middle), the *recipe-as-action* reduction of
P1, and — the only place genuine RL is defensible here — a **learned promotion
policy over the funnel**, which is the sequential decision problem the project
actually has. This also subsumes the AGENTS.md roadmap (its offline-table
REINFORCE milestone becomes the training harness for the promotion policy).

```
                        ┌────────────────────────────────────────────────────────┐
                        │                      SEARCH CONTROLLER                  │
                        │                                                        │
                        │   candidate generator: TPE/BO over surrogate UCB       │
                        │   (Optuna, mixed space) — proposes config x            │
                        │                                                        │
                        │   PROMOTION POLICY π(a | s)   ← the RL component       │
                        │   actions: {kill, re-proxy, promote, commit-full}      │
                        │   trained offline on logged funnel traces (PPO /       │
                        │   contextual bandit), reward = quality of final        │
                        │   incumbent under wall-clock budget                    │
                        └────────────┬───────────────────────────▲───────────────┘
                                     │ x, action                 │ observations o_k
        ┌────────────────────────────▼───────────────────────────┴──────────────┐
        │                        MULTI-FIDELITY FUNNEL                           │
        │                                                                        │
        │  F0 validate+analytic      cost ≈ 0 s     o_0: legality, cycle model   │
        │      cycles(x) from re-pinned table; accuracy gate (ACC_W)             │
        │  F1 behavioral sim         cost ≈ 5 s     o_1: exact cycles, accuracy  │
        │      (Verilator, 64 vec)                                               │
        │  F2 synth+STA proxy        cost ≈ 45 s    o_2: cell area, pre-layout   │
        │      (yosys recipe-aware,        WNS, netlist stats (FF/cell/lvl)      │
        │       + ORFS abc scripts)        gate: kill if wns < −2.5·margin       │
        │  F3 full ORFS flow         cost ≈ 7 min   o_3: area, Fmax, power,      │
        │      (nangate45)                 DRC-converged, timing_met             │
        │  F4 asap7 full flow        cost ≈ 7 min   o_4: same @ 7 nm (transfer)  │
        │                                                                        │
        │  every (x, k, o_k) row → results_funnel.jsonl  (training corpus)       │
        └────────────────────────────────┬───────────────────────────────────────┘
                                         │ all observations, all fidelities
                        ┌────────────────▼───────────────────────┐
                        │     SURROGATE  ĝ: (x, o_{≤k}) → (μ,σ)  │
                        │  gradient-boosted trees / GP per       │
                        │  output {area, period, power};         │
                        │  multi-fidelity: F3 prediction         │
                        │  conditions on F2 observables          │
                        │  (proxy area, proxy wns, FF count)     │
                        └────────────────────────────────────────┘
```

**Action space** (post-fix, evidence-justified):

| Parameter | Type | Range | Stage | Sensitivity | Acted on by |
|---|---|---|---|---|---|
| LANES | discrete | {1,2,4,8,16,32} | RTL chparam | dominant (×2.6 area, ×5.8 cycles) | candidate generator |
| ACC_W | discrete | {16,24,32} | RTL chparam | ±5% area, accuracy cliff | candidate generator (F0 accuracy gate kills 16 unless re-validated per V13) |
| clk constraint | continuous | [3.0, 8.0] ns n45 / [0.3, 1.5] ns asap7 | SDC | Fmax ±2.7×, area ±18%, power ×3 | candidate generator |
| ABC recipe | categorical | {orfs_speed, orfs_area, plain} | synth | ±43% synth area | candidate generator; **proxy must use the same recipe** (today it doesn't) |
| requantize pipelining | bool (once written) | {0,1} | RTL | ~2× Fmax expected | candidate generator |
| CORE_UTILIZATION | — | **fixed 40** | floorplan | <0.3% | dropped |
| PLACE_DENSITY | — | **fixed 0.60** | place | ~1.4% | dropped |
| fidelity action | discrete | {kill, re-proxy, promote, commit} | funnel | — | **promotion policy (RL)** |

**State** (promotion policy observation, fixed-size vector — a GNN over a 5K-cell
netlist of one design family adds nothing a dozen scalar netlist stats don't;
revisit if the design *family* diversifies):
`s = [x normalized (5), o_F0: cycles_norm/accuracy (2), o_F1 if run (2),
o_F2 if run: proxy_area_norm, proxy_wns, FF count, cell count, logic levels (5),
surrogate μ/σ for final reward (2), incumbent best reward (1),
remaining wall-clock budget fraction (1), funnel depth one-hot (4)]` ≈ 22 dims.

**Reward.** Terminal, per episode (one search campaign under budget B):
`R = best_final_reward_found(B)` where the per-config final reward keeps the
existing composite but with two repairs: `real_speedup` uses **measured Fmax
from F3** (never the requested clock — V2) with the re-pinned cycle model (V6);
and stage failures follow the *monotone* information ladder
(invalid −100 < elaborate −80 < sim −60 < proxy −40 < full-flow-fail −20 — V4).
Shaping for the promotion policy: per-step `r_k = Δ(surrogate-expected best) −
λ·cost_k/B` — the marginal expected improvement of the information bought minus
normalized wall-clock spent. This is computable offline from logged traces, so
the policy trains in *simulation against the table* (AGENTS.md's exact
mechanism) without spending a single new ORFS minute. Anti-gaming: the proxy's
measured bias is *pessimistic* on timing (15/18 agreement, all 3 misses
proxy-says-fail/full-flow-meets), so gates only ever **kill** on proxy timing,
never **accept** — the agent cannot inflate reward via the proxy because reward
only pays on F3 measurements.

**Algorithm.** Candidate generation: Optuna TPE (already integrated) over the
5-axis space, sampling from the surrogate's UCB — BO is the right tool for
≤10-dim mixed spaces with expensive oracles; PPO over *configs* would re-learn
what a GP prior gets free. Promotion policy: start as a contextual bandit
(LinUCB / Thompson over the 22-dim state; ~hundreds of logged traces suffice),
graduate to small-MLP PPO (2×64, as AGENTS.md specifies) only if the bandit's
myopia measurably loses to lookahead in table-simulation. MCTS is unjustified:
the funnel is depth-4 with branching 4 — exhaustive lookahead over the
*fidelity* tree is trivial; the hard part is the value estimate, which is the
surrogate's job. Benchmark bar (unchanged from AGENTS.md, it's the honest one):
**beat `--agent random` and the fixed-gate cascade on wall-clock-to-95%-optimum
over ≥20 seeds, on the table simulator, then once live.**

**Credit assignment & hardest problems, honestly stated:**
1. *Long-horizon RTL→route credit*: mostly **absent here** — routing never
   fails, so the dreaded "RTL choice causes route failure 3 stages later" path
   doesn't exist on this design at 45 nm. It may appear on asap7; the funnel
   logs (x, stage, failure) tuples precisely so the surrogate learns the
   boundary when it emerges. This is the principled answer: don't solve credit
   assignment for failure modes that don't occur; instrument so you see them
   when they do.
2. *Surrogate reliability is worst on power* (reported power is
   activity-factor-fiction, 99.8% combinational; treat as relative within
   matched clk only) and *on the clk→effort coupling* (the response surface
   achieved-period(requested-clk) is tool-version-dependent). Mitigation: the
   surrogate conditions on F2 observables, not just x; σ gates fall back to
   real evaluation.
3. *Non-stationarity*: every RTL edit shifts the table. The RTL content hash
   (fix V3) partitions the corpus; the surrogate takes the hash as a context
   feature and transfers via the unchanged axes — this is also exactly the
   nangate45→asap7 transfer test AGENTS.md wants (train on F3, evaluate
   sample-efficiency on F4).
4. *Action space growth*: adding buffer-SRAM axes later multiplies the space —
   the funnel architecture is unchanged; only the candidate generator's space
   definition grows. That's the point of separating proposal from promotion.

**Against the alternatives:**
- vs naive random over 2 params: random needs ~23 full flows (49% success@30
  measured on the 45-grid) ≈ 2.7 h to find the 45-grid optimum, and *cannot*
  exploit cheap fidelities — the funnel evaluates ~9 F2-screened candidates per
  F3 it spends.
- vs pure PrefixRL: months of GPU training to shave a multiplier that
  pipelining beats by 2× for one RTL edit.
- vs pure AlphaChip: there is literally nothing to place.
- vs pure Auto-PPA: 242-knob DDPG explores axes measured at <1.4% effect.
- The composition's marginal value over plain multi-fidelity BO (the strongest
  baseline, and the fallback if RL loses the benchmark): the learned promotion
  policy adapts gate thresholds to budget pressure (early: explore-promote;
  late: only commit near-incumbent configs) — fixed gates can't, and that is a
  measurable, falsifiable claim the benchmark will adjudicate.

---

## Phase 5 — Empirical Calibration (all measured today)

**Exp 1 — Full-flow baseline**: table in Phase 3. Budget math: serial ≈ 8
full flows/hour; route+cts = 65% of wall-clock; 2 concurrent flows fit in
30 GB (peak 1.5 GB each) and ~8 cores → **~14 F3 evals/hour sustainable**.

**Exp 2 — Parameter sensitivity** (synth-level swept; PnR-level mined from the
46 builds): table in Phase 4/P4. Headline: LANES ≫ clk ≫ recipe ≫ ACC_W ≫
density (1.4%) > util (0.3%). Interaction found: clk×everything (effort
coupling); recipe×`-D` interaction is *zero* (broken knob, see P1-b).

**Exp 3 — Proxy fidelity correlation** (18 configs, proxy run fresh here vs
existing full-flow results, matched default util/density):
- Spearman proxy-area vs final-area: **0.904** (n=18; 0.891 on 11 unique (L,A))
- Spearman proxy-WNS vs final-WNS: **0.868**; proxy-period vs final-period: −0.863*
- timing_met agreement **15/18**; all 3 misses proxy-pessimistic (proxy −2.3…−2.5
  at 5 ns where the full flow met) — calibrated kill-gate: `proxy_wns < −2.5 ns`
  at the target clock loses zero true positives on this data.
- *The negative period correlation is an artifact worth recording: proxy
  `report_clock_min_period` is reg2reg-only on an unbuffered netlist, while
  final period tracks the constraint (effort coupling) — rank them within
  matched clk groups, or use proxy WNS, which is the stronger signal.*
- **>0.7 threshold met → synth+STA stands as the F2 gate.**

**Exp 4 — Synthesis recipe variance**: table in Phase 4/P1. 43% area, 59%
pre-layout delay spread, `-D` no-op. **High variance → recipe enters the action
space; sequential AIG-op RL still rejected** (3 recipes span the space; the
post-route critical path is RTL-bound).

**Architecture updates forced by the data** (vs the pre-experiment draft):
(1) clock promoted from "evaluation setting" to first-class continuous design
axis with the effort-coupling response surface learned by the surrogate — the
`max(clk, 3.72)` analytic cap is retired; (2) proxy must be made *recipe-aware*
(it currently synthesizes with `plain`, a recipe the full flow never uses —
the ρ=0.9 will improve further once F2 and F3 share recipes); (3) util/density
demoted to constants, shrinking the cascade space from ~27K to ~2.2K configs
per RTL variant — small enough that the *offline table* for promotion-policy
training is buildable at F2 level in ~28 h, or over the strategic subsets in
under a workday.

---

## Phase 6 — Implementation Plan

**Stage A — repairs (1–2 days, ordered; items independent unless noted)**
1. V2 parse-fail status (S) → 2. V1 asap7 units + quarantine results (S) →
3. V4 penalty ladder (S) → 4. V5 max_speedup (S) → 5. V8 abc default (S) →
6. V3+V11 RTL-hash variant names (M, after 5) → 7. V6 re-pin via extended
`measure_real.py` (M; sim already rebuilt) → 8. V7 env knob forwarding (S) →
9. V10 timeout/process-group/no-FAIL-memoise (M) → 10. V9 UCB bounds (S) →
11. V14 sweep.sh hardening (S) → 12. V12/V13 model-truth decisions (M, can
parallel-track).

**Stage B — new infrastructure**
- `optimizer/funnel.py`: `FunnelEnv` — refactor of `CascadeOptEnv` exposing
  per-stage observations and accepting promotion *actions* (gym-style step per
  fidelity transition; `done` at kill/commit). Depends on A.1–A.6.
- `optimizer/surrogate.py`: `Surrogate.fit(rows) / predict(x, obs) → (μ,σ)`
  per metric; sklearn `GradientBoostingRegressor` baseline, swap-in GP later.
- `optimizer/recipe.py`: ABC recipe axis — extends `_yosys_synth_script` to
  accept `{orfs_speed, orfs_area, plain}` (reusing
  `$SCRIPTS_DIR/abc_{speed,area}.script` + the `-constr`/driver constants
  measured here), and adds recipe to `variant_name`/`config.mk` (`ABC_AREA`).
- `optimizer/build_table.py`: offline F0–F2 table over the ~2.2K-config reduced
  space (resumable; ~28 h F2 budget or strategic subset ~6 h).
- `optimizer/agents/promotion_agent.py`: LinUCB contextual bandit first; PPO
  (stable-baselines3) behind the same interface.
- `optimizer/benchmark_funnel.py`: table-simulator benchmark — random vs
  fixed-gate cascade vs bandit vs PPO; metric = wall-clock-to-95%-optimum,
  ≥20 seeds, report p95.

**Stage C — refactors**: `physical_runner.py` (units table, hash, recipe,
timeout, explicit cache) — interface change: `run_physical(..., abc_recipe=,
rtl_hash=)`; `physical_reward.py` (no fallbacks, ladder); `run_*_optimizer.py`
(log TIMEOUT trials, fix None-format crash at `run_cascade_optimizer.py:109`).

**Dependency DAG**: A.{1,2,3,4,5} → A.6 → B.funnel → B.table → B.promotion →
B.benchmark; A.7 → B.table (cycle constants feed F0); B.recipe → B.table
(recipe must be in the table's axes); B.surrogate ∥ B.table (trains on whatever
exists); asap7 F4 unblocked by A.2 — first asap7 sweep (~12 configs @ realistic
0.8–1.2 ns) lands right after A.2 and is the transfer test set.

**Packages**: `optuna>=3.6` (present in plan, not installed here — install),
`scikit-learn>=1.4`, `scipy>=1.11`; only if/when PPO: `stable-baselines3>=2.3`,
`torch>=2.2` (CPU). No torch-geometric — no GNN (justified in Phase 4).

**Validation checkpoints**:
- CP1 (after A.1–A.6): re-run `run_physical_optimizer` 5 trials nangate45 —
  zero positive rewards from None-metrics; asap7 single run at 1.0 ns shows
  |wns| < 1 ns scale (not −1861). *Fail = any all-None 'ok'.*
- CP2 (after A.7): `benchmark_agents.py` reproduces optimum {4,24,5} with
  re-pinned table; sanity anchor re-centered (margin was 0.018 — re-derive).
- CP3 (after B.table subset, ~50 rows): surrogate 5-fold ρ ≥ 0.8 on F3 area;
  ≥ 0.7 on period. *Fail = add F2 observables or more rows before proceeding.*
- CP4 (benchmark): bandit ≥ fixed-gate cascade on p95 wall-clock-to-95%.
  *Fail = ship the funnel with fixed gates + BO; record the negative result
  honestly like Stage 5 did.*
- CP5 (live): one 4-hour live campaign ≈ table-simulated trajectory within
  noise.

**Expected outcomes** (from measured costs and the 46-build distribution):
within a **100-F3-config budget** (~7 h live): the reduced space is ~2.2K
configs/RTL-variant; F2 screening at ρ=0.9 with a 9:1 screen ratio effectively
covers ~900 configs — expect to locate the global optimum of the fixed-RTL
space with high confidence, and the realistic Pareto gain over the current
incumbent ({L4,A24,clk5} at the old reward) comes from the clk axis near
4.0–4.5 ns and the area recipe: order **−10–15% area at iso-throughput**
(area recipe −17% at synth, ~half survives PnR) plus correct (smaller but
honest) speedup accounting after V6. A **1000-config budget** only pays if the
space grows — requantize pipelining (the 2× Fmax lever, which moves real
throughput, not %), SRAM buffers, or asap7 co-search; the knee is at "every
fixed-RTL config F2-screened once" ≈ 350 F3-equivalents. Past that, spend on
RTL variants, not flow configs.
