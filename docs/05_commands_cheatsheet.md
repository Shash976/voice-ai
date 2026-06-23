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

## 4. Stage 5 — Design-space optimization → [eda-rl](https://github.com/Shash976/eda-rl)

The optimizer is now a standalone tool in its own repo. Install it and point it at
this accelerator's DesignSpec:

```bash
pip install -e <eda-rl checkout>
EDA_RL_DESIGN_ROOT=$(pwd) \
  eda-rl optimize --design <eda-rl>/eda_rl/designs/tinymac_accel.yaml \
    --platform nangate45 --budget-hours 4
eda-rl report  --campaign latest --open    # graphical HTML dashboard
eda-rl collect --campaign latest --open    # best configs + their GDS
```

See the [eda-rl README](https://github.com/Shash976/eda-rl) for the full command set
and options.

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
