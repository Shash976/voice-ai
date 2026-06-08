# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Pocket AI Voice Recorder with RISC-V TinyML Accelerator** — a 6-stage project culminating in a custom int8 MAC accelerator chip. The work flows from Pi software → TinyML C inference → PicoRV32 Verilator simulation → behavioral accelerator → RTL accelerator → RTL-to-GDS via OpenROAD-flow-scripts (ASAP7).

Full plan: `pocket_ai_voice_recorder_riscv_tinyml_plan.md`

### Current status
- ✅ Stage 1–2: TinyVAD trained, int8 TFLite quantized, C inference matches TFLite within ±2 LSB (64/64 vectors pass)
- ✅ Stage 3: Verilator simulation working; 64/64 correct; SW baseline = ~11.2M cycles/inference (~112ms @ 100 MHz)
- ✅ Stage 4: Behavioral TinyMAC accelerator working; 64/64 correct; ~58.6K cycles/inference (~0.6ms @ 100MHz); ~191× speedup vs SW baseline (8 lanes). 16 lanes → ~43.4K cycles, ~258×.
- ✅ Stage 5: Design-space optimizer (`optimizer/`) — frequency-aware reward (`real_speedup` = cycles × clock, caps clock at critical path; kills the slow-clock degeneracy), buffer axes dropped (no sim model), 45-config space. `benchmark_agents.py` measures regret/trials-to-optimum vs an exhaustive grid optimum `{lanes=4, acc=24, clk=5}`; honest finding: **no agent beats random** on this small deterministic space (UCB worse). `test_reward_sanity.py` = offline invariant checks (13/13). Constants pinned via `measure_real.py` (SW baseline 11,196,638 cycles/inf; max real_speedup ≈519 → `max_speedup=576` with 11% headroom). `enumerate` agent added (`optimizer/agents/enumerate_agent.py`) — exhaustive grid sweep, the correct tool here. **These are DSE (black-box search), not RL** — see `AGENTS.md` for the path toward real RL.
- 🚧 Stage 6: Synthesizable accelerator RTL written (`rtl/accel/{int8_mac_array,requantize,tinymac_accel}.v`) + Verilator unit TB (`rtl/tb/`) bit-exact vs SW golden (45/45, LANES∈{2,4,8}, ACC_W∈{24,32}). **Full nangate45 GDS produced** via classic ORFS make flow on the company VM (`/opt/OpenROAD-flow-scripts`). LANES=4 ACC_W=24: ~19,738µm² (48% util), 231 FFs, **Fmax ≈269 MHz** (period_min 3.72ns); critical path = requantize Q31 multiply, **independent of LANES** → clean area↑/Fmax-flat Pareto. Synth-only area sweep (`physical/orfs/synth_area.sh`): nangate45 L1=12.3K→L16=22.9K µm² (16× MACs, only 1.86× area). Flow files: `physical/orfs/make/{run.sh,sweep.sh,<plat>/tinymac_accel/{config.mk,constraint.sdc}}`. **Gotchas:** (a) bazel-orfs route abandoned (PyPI fetch times out); use classic make flow. (b) Yosys 0.64 asserts `genrtlil.cc:2214` on signed/unsigned mixing — NO `$signed()` on unsigned whole wires, NO signed `integer` params in unsigned exprs, NO mixed-sign `?:` branches (yosys 0.9 + Verilator lint miss these). (c) param sweep via ORFS `VERILOG_TOP_PARAMS="LANES n ACC_W w"` (chparam) + `FLOW_VARIANT` per config. **Stage-5↔6 loop wired:** `optimizer/{physical_runner,physical_reward,physical_env,run_physical_optimizer}.py` make the existing agents drive the real ORFS flow and score measured area/Fmax/power (reward = behavioral cycles × `max(clk,1000/Fmax)`, real area/power, real timing_met; same YAML weights). `PHYSICAL_MOCK=1` for offline tests; logs to `results_physical.jsonl`. Reviewed for hallucinations (report parsers checked vs real VM strings). **Behavioral sim cycle model backported from RTL** (`sim_main.cpp` `ACCEL_CH_OVERHEAD=2`): latency = `n_outputs×(ceil(K/LANES)+2)` vs old `ceil(M·K/LANES)`; ~12.5% higher; WSL rebuild + `measure_real.py` re-pin needed. Remaining: realistic-clock re-sweep, asap7 retarget, optional requantize pipelining.

---

## Environment Split

| Task | Machine |
|------|---------|
| Python ML (train, convert, export) | **Windows** (has GPU, PyTorch, TFLite) |
| Hardware (Verilator, RV32 cross-compile, ORFS) | **WSL** (Ubuntu on the same machine) |

The repo lives on Windows at `C:\Users\shash\Desktop\Code\voiceAI`. WSL has a **separate copy** at `~/voiceAI` (`/home/shashg/voiceAI`) — NOT a symlink to `/mnt/c/...`. Always edit files in the WSL copy when making hardware changes; sync back to Windows manually (or via git).

---

## Build Commands

All hardware/firmware commands run **in WSL**.

### Generated headers (run once after model changes)
```bash
# Windows (Python venv active)
python sw/tinyml_reference/export_weights.py     # → firmware/tinyengine_port/tiny_vad_weights.h
python sw/tinyml_reference/gen_test_vectors.py   # → firmware/tinyengine_port/tiny_vad_test_vectors.h
```

### Firmware (cross-compile for RV32)
```bash
cd firmware/picorv32_baremetal
make              # → firmware.bin
make size         # section sizes
make disasm       # disassembly (grep for FP instructions)
make clean
```

### Host-side C inference test (x86, fast sanity check)
```bash
cd firmware/tinyengine_port
make host         # gcc x86 binary
./test_infer_host # should print "42/42 passed"
```

### Verilator simulation (Stage 3)
```bash
cd sim/verilator
make check-deps   # verify prerequisites
make run          # build + compile firmware + run PicoRV32 simulation
make vcd          # same + VCD waveform dump → sim_out.vcd
make clean
```

Simulation prints CSV to stdout, stats to stderr. Expected output columns: `vec,label,result,correct,logit0,logit1,cycles`.

### ML training & conversion (Windows)
```bash
python train_tiny_vad.py         # → tiny_vad_best.pt, tiny_vad.onnx
python convert_to_tflite.py      # → tiny_vad_int8.tflite
```

---

## Architecture

### Data flow (end-to-end)
```
Audio (16 kHz mono)
  → extract_logmel() [speech_simulator.py]
  → int8[49×40] log-mel tensor
  → TinyVAD (speech/silence) → prob[1] > 0.5 → speech detected
  → if speech: whisper.cpp → transcript
```

### Quantization scheme
- **Input**: `float = INPUT_SCALE * (int8 − INPUT_ZP)`
- **Weights**: per-channel int8, scale extracted from TFLite `quantization_parameters.scales`
- **Requantization**: `real_mult = scale_in × weight_scale / scale_out` decomposed to Q31 `(q_mult, rshift)` pair where `shift` can be negative (left shift)
- `requantize(x, q_mult, shift)`: int64 accumulation, handles `shift < 0` via `val <<= (−shift)`

### Tensor layout throughout
All tensors use **[time, channel]** order (TFLite NHWC convention), not PyTorch's [channel, time]. This is critical — past layout bugs caused completely wrong outputs.

### TinyVAD model dimensions
| Layer | Input | Output |
|-------|-------|--------|
| Conv0 (k=5,s=2,p=2) | [49, 40] | [25, 32] |
| Conv1 (k=3,s=2,p=1) | [25, 32] | [13, 64] |
| GlobalAvgPool | [13, 64] | [64] |
| FC0 | [64] | [32] |
| FC1 | [32] | [2] (logits) |

Static scratch buffers: buf0[800], buf1[832], buf2[64], buf3[32] — ~2 KB total, no dynamic allocation.

### Memory map (Verilator sim)
| Address | Purpose |
|---------|---------|
| `0x00000000–0x0003FFFF` | 256 KB RAM (code + data + stack) |
| `0x10000000` | UART TX (write byte → stdout) |
| `0x10000004` | SIM_EXIT (write → halt sim) |
| `0x20000000–0x20000FFF` | TinyMAC accelerator registers (Stage 4) |

PicoRV32 resets to `0x00000000`. Stack grows down from `0x00040000`.

### PicoRV32 parameter names
The correct parameter name is `COMPRESSED_ISA` (not `ENABLE_COMPRESSED`). Other used params: `ENABLE_MUL`, `ENABLE_FAST_MUL`, `ENABLE_DIV`, `ENABLE_COUNTERS`, `REGS_INIT_ZERO`. `ENABLE_DIV` must be **1** — `global_avg_pool` uses a `div` instruction.

### Firmware build flags (critical)
The riscv64-linux-gnu toolchain defaults to PIE mode even with `-nostdlib`, causing GOT-indirect loads for linker symbols like `_stack_top`. Required flags to prevent this:
- `-fno-pic -fno-pie` in CFLAGS — forces direct `auipc+addi` addressing, no GOT
- `-no-pie -Wl,--build-id=none` in LDFLAGS — suppresses PT_PHDR and `.note.gnu.build-id` sections that would push `.text` away from address 0

### Verilator simulation loop
Combinatorial memory (0-wait-state): on negedge, present `mem_rdata` and assert `mem_ready`; CPU latches on posedge. This means 1 clock per memory transaction.

---

## Artifact Dependency Chain

```
train_tiny_vad.py
  → tiny_vad_best.pt
      → convert_to_tflite.py
          → tiny_vad_int8.tflite
              → export_weights.py → tiny_vad_weights.h
              → gen_test_vectors.py → tiny_vad_test_vectors.h
                  → firmware/picorv32_baremetal/ (Makefile)
                      → firmware.bin
                          → sim/verilator/sim_main.cpp → simulation
```

Both `tiny_vad_weights.h` and `tiny_vad_test_vectors.h` are **auto-generated** — do not edit by hand.

---

## Open work / next steps

### Immediate (hardware)
1. **WSL sim rebuild** — sync `sim/verilator/sim_main.cpp` to WSL and rebuild `sim_picorv32`; the cycle model changed (per-channel overhead). Then re-run `optimizer/measure_real.py` to re-pin `SW_BASELINE_CYCLES` and per-lane cycle tables.
2. **Realistic-clock re-sweep** — re-run `physical/orfs/make/sweep.sh` at a clock ≈4 ns (close to the 3.72 ns critical path) so all LANES configs meet timing; collect updated area/Fmax/power into `sweep_results.csv`.
3. **ASAP7 retarget** — copy `physical/orfs/make/nangate45/tinymac_accel/` → `asap7/`, set `PLATFORM = asap7`, pick a ~700 ps clock (7 nm). See if the critical path scales as expected.
4. **Optional: requantize pipelining** — split the Q31 64-bit multiply across 2 cycles; should roughly double Fmax for <1% throughput loss (requantize runs once per output channel, not per MAC).

### Optimizer / RL direction
The optimizer currently does **design-space exploration (DSE)** — black-box single-step search. The next milestone is evolving it toward genuine **reinforcement learning**: a trajectory-based policy that improves from experience and generalizes. See `AGENTS.md` for the full formulation, the MDP definition, and what needs to be built.
