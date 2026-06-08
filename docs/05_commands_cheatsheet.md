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
./test_infer_host        # expect: all vectors pass, max error ≤ 2 LSB
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
./sim_picorv32 firmware.bin --mac-lanes 16           # 16 lanes  → ~258× speedup
./sim_picorv32 firmware.bin --mac-lanes 1            # 1 lane    → slowest accel
./sim_picorv32 firmware.bin --acc-width 16           # 16-bit acc → accuracy drops to 47/64
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

🪟 **Windows — offline, no sim needed** (quickest Stage-5 demo):
```powershell
python optimizer/benchmark_agents.py     # agent comparison + honest "no agent beats random" verdict
python optimizer/test_reward_sanity.py   # 13/13 reward-function invariants
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

**Cascade optimizer** — larger ~27 K-config space with multi-fidelity funnel:
```bash
python3 optimizer/run_cascade_optimizer.py --agent evo --trials 30        # full funnel (slow — runs P&R)
python3 optimizer/run_cascade_optimizer.py --max-stage proxy --trials 80  # proxy only (Yosys + STA, no P&R)
python3 optimizer/run_cascade_optimizer.py --platform asap7               # switch target PDK
PHYSICAL_MOCK=1 python optimizer/test_cascade.py                          # offline self-test, 20 checks
```

Live dashboard (separate terminal):
```bash
streamlit run optimizer/dashboard.py
```

---

## 5. Stage 6 — RTL → GDS  🚧 GDS produced

RTL is written (`rtl/accel/`), bit-exact-verified, and a full nangate45 GDS has been
produced via the classic ORFS make flow on the company VM
(`/opt/OpenROAD-flow-scripts`). Numbers: LANES=4 ACC_W=24 → ~19,738 µm², ~269 MHz
Fmax, 231 FFs. See `docs/06_rtl_to_gds.md` for details.

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
| `./test_infer_host` | all vectors pass, max error ≤ 2 LSB |
| `make run` (accel on, 8 lanes) | `correct=64/64 avg_cycles≈58–66K`¹ |
| `./sim_picorv32 ... --mac-lanes 16` | `correct=64/64`, ~43–49K cycles¹ |
| `./sim_picorv32 ... --acc-width 16` | `correct=47/64` (overflow, expected) |
| `benchmark_agents.py` | "No agent meaningfully beats random search" |
| `test_reward_sanity.py` | 13/13 checks pass |
| `cd rtl/tb && make` | `45/45 PASS  0 mismatches` |
| `synth_area.sh nangate45` | LANES=1: ~12.3K µm² → LANES=16: ~22.9K µm² |

¹ Cycle counts pending WSL rebuild with the updated cycle model (`ACCEL_CH_OVERHEAD=2`,
~12.5% higher than old `ceil(M·K/LANES)` formula). Re-pin with `measure_real.py` after.

---

## Common pitfalls

1. **Edited a file on Windows, WSL doesn't see it** → the two repos are separate
   clones. Commit + pull, or edit in the WSL copy.
2. **Sim "times out" / never finishes** → you're running the pure-SW path (hooks
   commented out). ~11.2M cyc/inf × 64 > the 50M cap. Expected; read per-inference.
3. **`make run` shows huge speedup "for free"** → hooks are ON by default; you're
   measuring the *accelerated* path, not the baseline.
4. **16-bit accumulator "passes" on vec 0** → it doesn't overflow on vec 0. Always
   check the full 64-vector run (16-bit → 47/64).
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
