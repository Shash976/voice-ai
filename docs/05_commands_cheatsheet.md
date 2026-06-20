# 05 — Commands Cheat Sheet

Every command, in one place. **Mind which machine each runs on.**

- 🪟 **Windows** = PowerShell, Python venv active, repo at `C:\Users\shash\Desktop\Code\voiceAI`
- 🐧 **WSL** = Ubuntu shell, repo at `~/voiceAI` (separate clone — sync via git!)

---

## 0. First-time setup

🐧 **WSL — install toolchain** (once):
```bash
sudo apt install verilator gcc-riscv64-linux-gnu binutils-riscv64-linux-gnu
bash scripts/setup_picorv32.sh      # fetches rtl/picorv32/picorv32.v
```

🪟 **Windows — Python deps** (once):
```powershell
pip install torch torchaudio tensorflow pyyaml
pip install optuna        # optional, only for --agent bayesian
```

---

## 1. Stage 1–2 — model → int8 → C headers  🪟 Windows

```powershell
python train_tiny_vad.py                          # → tiny_vad_best.pt, tiny_vad.onnx  (downloads dataset 1st run)
python convert_to_tflite.py                        # → tiny_vad_int8.tflite
python sw/tinyml_reference/export_weights.py       # → firmware/tinyengine_port/tiny_vad_weights.h
python sw/tinyml_reference/gen_test_vectors.py     # → firmware/tinyengine_port/tiny_vad_test_vectors.h
```
> Only needed if you change the model. The checked-in headers are already current.
> The two `.h` files are **auto-generated — never hand-edit**.

---

## 2. Sanity-check the C engine on x86  🐧 WSL (fast, no RISC-V)

```bash
cd firmware/tinyengine_port
make host                # gcc → ./test_infer_host
./test_infer_host        # expect: 64/64 pass, max logit error ≤ 3 LSB
make clean
```
**This is the fastest "is it alive?" check.** Do this first.

---

## 3. Stage 3 + 4 — firmware + Verilator simulation  🐧 WSL

```bash
cd sim/verilator
make check-deps          # verify toolchain + generated headers present
make run                 # cross-compile fw → Verilate → build testbench → run
make vcd                 # same + dump sim_out.vcd (open in GTKWave)
make clean
```

Run the built sim directly with different accelerator knobs (no rebuild):
```bash
cd sim/verilator
./sim_picorv32 firmware.bin                          # defaults: 8 lanes, 32-bit acc
./sim_picorv32 firmware.bin --mac-lanes 16           # 16 lanes  → ~240× speedup
./sim_picorv32 firmware.bin --mac-lanes 1            # 1 lane    → slowest accel
./sim_picorv32 firmware.bin --acc-width 16           # 16-bit acc → accuracy drops to 47–58/64
                                                     #   (lane-dependent: saturation is per-chunk, like the RTL)
./sim_picorv32 firmware.bin --vcd out.vcd            # waveform dump
```

Build firmware by itself:
```bash
cd firmware/picorv32_baremetal
make            # → firmware.bin
make size       # section sizes
make disasm     # disassembly (grep for unexpected FP instructions)
make clean
```

**Measure the pure-SW Stage-3 baseline** (~11.2M cycles/inf): comment out the two
`tinyvad_*_hook = ...` lines in `firmware/picorv32_baremetal/main.c`, then
`make` (firmware) and `make run` (sim). Read cycles from vec 0 (the full sweep
times out at the 50M-cycle cap in pure SW — that's expected).

---

## 4. Stage 5 — optimizer

There are **two tracks** (see `docs/04_optimizer.md`):

- **`run_optimizer.py`** — the small 45-config space. Runs the Verilator sim for
  cycles/accuracy and **analytic proxy formulas** for area/power/timing. **Never
  runs Yosys or OpenROAD — no PDK, no GDS.** Fast.
- **`run_cascade_optimizer.py`** — the big ~27 K space. A **multi-fidelity funnel**:
  it simulates (and synth-screens) each config *first*, and only configs that pass
  every cheap gate reach the expensive full RTL→GDS place-and-route. So a broken or
  oversized config is thrown out in seconds instead of wasting a multi-minute PDK run.

🪟 **Windows — offline, no sim needed** (quickest Stage-5 demo):
```powershell
python optimizer/benchmark_agents.py     # agent comparison + honest "no agent beats random" verdict
python optimizer/test_reward_sanity.py   # 16/16 reward-function invariants
```

🐧 **WSL — live optimizer** (needs sim_picorv32 + firmware.bin built first):
```bash
python3 optimizer/run_optimizer.py --agent enumerate # exhaustive 45-config sweep — recommended
python3 optimizer/run_optimizer.py                   # evo, 30 trials (default)
python3 optimizer/run_optimizer.py --agent random    # random baseline
python3 optimizer/run_optimizer.py --agent ucb       # UCB1 bandit
python3 optimizer/run_optimizer.py --agent bayesian  # Optuna TPE (needs: pip install optuna)
python3 optimizer/run_optimizer.py --agent evo --trials 50
python3 optimizer/run_optimizer.py --resume          # continue previous run (appends to results.jsonl)
python3 optimizer/run_optimizer.py --dry-run         # print configs, skip the sim
python3 optimizer/runner.py 16 32                    # run one sim config directly (lanes acc_width)
```

**Cascade optimizer** — larger ~27 K-config space with the multi-fidelity funnel.
Each config is pushed through gates cheapest → most expensive, and the **first
failure short-circuits the rest** so the full PDK flow only runs on survivors:

```
validate  (µs)   legality + YAML constraints — no tools
elaborate (~s)   Yosys reads the parameterised RTL
sim       (~s)   Verilator: correctness + cycles   → reject if accuracy < 0.95 (e.g. acc_w<24)
proxy     (s–m)  Yosys synth + OpenROAD STA        → reject if synth area > 80,000 µm²
full      (min)  full RTL→GDS with the PDK         → real area / timing / power → reward
```

```bash
python3 optimizer/run_cascade_optimizer.py --agent evo --trials 30        # full funnel (slow — reaches GDS)
python3 optimizer/run_cascade_optimizer.py --max-stage proxy --trials 80  # stop at synth+STA (no P&R/GDS)
python3 optimizer/run_cascade_optimizer.py --max-stage sim --trials 200   # stop after the sim gate (fastest screen)
python3 optimizer/run_cascade_optimizer.py --platform asap7               # switch target PDK
PHYSICAL_MOCK=1 python optimizer/test_cascade.py                          # offline self-test, 20 checks
```

The run prints **funnel attrition** (how many configs reached each gate), so you can
see where bad configs die — e.g. `acc_width<24` is killed at the cheap `sim` gate and
never wastes a place-and-route. `--max-stage` caps how deep the funnel goes (handy to
screen many configs without paying for GDS).

Live dashboard (separate terminal):
```bash
streamlit run optimizer/dashboard.py
```

**Second-generation funnel optimizer** (see `docs/08_funnel_optimizer.md`) —
promotion decisions become trainable actions, a surrogate model sits between the
fidelities, and the space is either the reduced 4-axis tinymac space or a
design-agnostic space built from a YAML spec + tiered ORFS knob registry.

> All commands below work via the shims at `optimizer/`; the real code lives
> under `optimizer/gen2/` and `optimizer/common/`.

```bash
# Offline F0–F2 table (tinymac, default):
python3 optimizer/build_table.py --subset strategic    # 84 configs, ~1.2 h (resumable)
python3 optimizer/build_table.py                       # full 594-config grid, ~7 h
python3 optimizer/build_table.py --dry-run             # show plan + cost estimate first

# Table for a different design (gcd, tier-2 ORFS knobs active):
python3 optimizer/build_table.py --design gcd --max-tier 2

# Fit + cross-validate the surrogate on all built data:
python3 optimizer/fit_surrogate.py                     # writes optimizer/surrogate_n45.joblib

# Benchmark promotion policies on the table simulator:
python3 optimizer/benchmark_funnel.py --seeds 20       # random vs fixed-gate vs LinUCB
python3 optimizer/benchmark_funnel.py --candidates tpe         # Optuna TPE candidate ordering
python3 optimizer/benchmark_funnel.py --candidates surrogate_ucb  # surrogate UCB ordering
python3 optimizer/benchmark_funnel.py --candidates shuffled    # default (seeded shuffle)

# Live campaign driver — tinymac, 4-axis tier-1 space, TPE candidates:
python3 optimizer/run_funnel_optimizer.py \
    --design tinymac_accel --platform nangate45 \
    --budget-hours 4 --max-tier 1 --sampler tpe --promotion fixed

# Live campaign — gcd, tier-2 knob space:
python3 optimizer/run_funnel_optimizer.py \
    --design gcd --platform nangate45 \
    --budget-hours 4 --max-tier 2 --sampler tpe --promotion fixed

# Table-mode campaign (replay without real ORFS calls):
python3 optimizer/run_funnel_optimizer.py \
    --design tinymac_accel --budget-hours 4 \
    --sampler surrogate_ucb --promotion linucb \
    --table optimizer/results_funnel.jsonl

# Self-tests (no real tools needed):
PHYSICAL_MOCK=1 python3 optimizer/funnel.py            # FunnelEnv self-test
PHYSICAL_MOCK=1 python3 optimizer/run_funnel_optimizer.py \
    --design tinymac_accel --budget-hours 0.01 \
    --sampler tpe --promotion fixed \
    --table optimizer/results_funnel.jsonl
```

Campaign logs are written to
`optimizer/campaigns/<design>/<platform>/results_funnel_campaigns.jsonl`
(printed as `Results →` at run end). Use `--out /path.jsonl` to override.

**Visualize a campaign** (`pip install optuna-dashboard plotly` once):
```bash
LOG=optimizer/campaigns/tinymac_accel/nangate45/results_funnel_campaigns.jsonl

# Static self-contained HTML report (reward vs params, history, funnel, importances):
python3 optimizer/viz/report.py                        # auto-finds latest log
python3 optimizer/viz/report.py --log $LOG --open      # specific log, open in browser
python3 optimizer/viz/report.py --log $LOG --campaign all  # pool all campaigns in file

# Live Optuna dashboard (auto-refreshes while the optimizer runs):
python3 optimizer/viz/dashboard.py --live --log $LOG   # http://127.0.0.1:8080/
python3 optimizer/viz/dashboard.py                     # one-shot snapshot, latest campaign
python3 optimizer/viz/dashboard.py --no-serve --log $LOG  # rebuild JournalStorage only
```

---

## 5. Stage 6 — RTL → GDS  🚧 GDS produced

RTL is written (`rtl/accel/`), bit-exact-verified, and a full nangate45 GDS has been
produced via the classic ORFS make flow on the company VM
(`/opt/OpenROAD-flow-scripts`). Numbers: LANES=4 ACC_W=24 → ~19,738 µm², ~269 MHz
Fmax, 230 FFs. See `docs/06_rtl_to_gds.md` for details.

**RTL unit tests** (Verilator, runs anywhere):
```bash
cd rtl/tb
make               # LANES=4 ACC_W=24 — expect 45/45 PASS
make ACC_W=32      # also bit-exact
make LANES=8       # more parallelism
make clean
```

**Synthesis-only area sweep** (Yosys, no OpenROAD needed):
```bash
bash physical/orfs/synth_area.sh nangate45   # sweeps LANES 1–16, prints cell area table
bash physical/orfs/synth_area.sh sky130hd    # same for sky130
```

**Full RTL→GDS** (company VM only — needs `/opt/OpenROAD-flow-scripts`):
```bash
physical/orfs/make/run.sh                    # nangate45, LANES=4, single config
physical/orfs/make/run.sh nangate45 gui_final  # + open OpenROAD GUI after route
physical/orfs/make/sweep.sh                  # sweeps LANES={1,2,4,8,16}, → sweep_results.csv
```

**Physical optimizer** — agents driving the real ORFS flow:
```bash
# On the VM (real OpenROAD):
python3 optimizer/run_physical_optimizer.py --agent evo --trials 12
# Offline self-test (no OpenROAD, PHYSICAL_MOCK=1):
PHYSICAL_MOCK=1 python3 optimizer/run_physical_optimizer.py --agent random --trials 6
```

---

## Quick reference: expected outputs

| Command | Healthy result |
|---------|----------------|
| `./test_infer_host` | 64/64 pass, max logit error ≤ 3 LSB |
| `make run` (accel on, 8 lanes) | `correct=64/64 avg_cycles≈61.4K` |
| `./sim_picorv32 ... --mac-lanes 16` | `correct=64/64`, ~46.7K cycles |
| `./sim_picorv32 ... --acc-width 16` | `correct=47–58/64` (overflow; lane-dependent, expected) |
| `benchmark_agents.py` | "No agent meaningfully beats random search" |
| `test_reward_sanity.py` | 16/16 checks pass |
| `cd rtl/tb && make` | `45/45 PASS  0 mismatches` |
| `synth_area.sh nangate45` | LANES=1: ~12.3K µm² → LANES=16: ~22.9K µm² |
| `PHYSICAL_MOCK=1 python3 optimizer/funnel.py` | "All self-tests PASSED" |
| `python3 optimizer/fit_surrogate.py` | CV Spearman ρ ≥ 0.8 area, ≥ 0.7 period |

(Per-lane cycle counts are pinned in `optimizer/constants.py` from real sweeps —
`measure_real.py` regenerates them after any sim/RTL cycle-model change.)

---

## Common pitfalls

1. **Edited a file on Windows, WSL doesn't see it** → the two repos are separate
   clones. Commit + pull, or edit in the WSL copy.
2. **Sim "times out" / never finishes** → you're running the pure-SW path (hooks
   commented out). ~11.2M cyc/inf × 64 > the 50M cap. Expected; read per-inference.
3. **`make run` shows huge speedup "for free"** → hooks are ON by default; you're
   measuring the *accelerated* path, not the baseline.
4. **16-bit accumulator "passes" on vec 0** → it doesn't overflow on vec 0. Always
   check the full 64-vector run (16-bit → 47–58/64, lane-dependent).
5. **Firmware traps at reset** → PIE/GOT or build-id flags missing/changed. See
   doc 02's build-flag section.
6. **`runner.py` can't find the sim** → build it in WSL first:
   `cd sim/verilator && make`.
7. **Yosys assertion `genrtlil.cc:2214`** → signed/unsigned mixing in RTL. No
   `$signed()` on unsigned whole-wires, no signed `integer` params in unsigned
   exprs, no mixed-sign `?:` branches. Verilator lint won't catch this.
8. **`run.sh` / `sweep.sh` fails to find OpenROAD** → these scripts require the
   company VM at `/opt/OpenROAD-flow-scripts`. Synthesis-only (`synth_area.sh`)
   needs only Yosys and runs anywhere.
9. **Physical optimizer runs real P&R in tests** → set `PHYSICAL_MOCK=1`; never
   invoke `run_physical_optimizer.py` without it in a test or CI context.
