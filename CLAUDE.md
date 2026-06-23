# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Pocket AI Voice Recorder with RISC-V TinyML Accelerator** — a 6-stage project culminating in a custom int8 MAC accelerator chip. The work flows from Pi software → TinyML C inference → PicoRV32 Verilator simulation → behavioral accelerator → RTL accelerator → RTL-to-GDS via OpenROAD-flow-scripts (ASAP7).

Full plan: `pocket_ai_voice_recorder_riscv_tinyml_plan.md`

> **Design-space optimization lives in a separate repo.** The RL/DSE engine that
> searches the ORFS flow for area/Fmax/power-optimal configs is now its own
> standalone tool: **[eda-rl](https://github.com/Shash976/eda-rl)**. To optimize
> this accelerator with it, install eda-rl and point it at a tinymac DesignSpec
> YAML with `EDA_RL_DESIGN_ROOT` set to this checkout:
> ```bash
> pip install -e <eda-rl checkout>
> EDA_RL_DESIGN_ROOT=$(pwd) \
>   eda-rl optimize --design <eda-rl>/eda_rl/designs/tinymac_accel.yaml \
>     --platform nangate45 --budget-hours 4
> ```

### Current status
- ✅ Stage 1–2: TinyVAD trained, int8 TFLite quantized, C inference matches TFLite within ±3 LSB (64/64 vectors pass)
- ✅ Stage 3: Verilator simulation working; 64/64 correct; SW baseline = ~11.2M cycles/inference (~112ms @ 100 MHz)
- ✅ Stage 4: Behavioral TinyMAC accelerator working; 64/64 correct; ~61.4K cycles/inference (~0.6ms @ 100MHz); ~182× speedup vs SW baseline (8 lanes). 16 lanes → ~46.7K cycles, ~240×.
- ✅ Stage 5: Design-space optimization — extracted to the standalone **[eda-rl](https://github.com/Shash976/eda-rl)** repo (multi-fidelity funnel optimizer over the ORFS flow; design-agnostic via a `DesignSpec` YAML). See that repo's README for the full pipeline; use the pointer above to run it against this accelerator.
- 🚧 Stage 6: Synthesizable accelerator RTL written (`rtl/accel/{int8_mac_array,requantize,tinymac_accel}.v`) + Verilator unit TB (`rtl/tb/`) bit-exact vs SW golden (45/45, LANES∈{2,4,8}, ACC_W∈{24,32}). **Full nangate45 GDS produced** via classic ORFS make flow on the company VM (`/opt/OpenROAD-flow-scripts`). LANES=4 ACC_W=24: ~19,738µm² (48% util), 230 FFs, **Fmax ≈269 MHz** (period_min 3.72ns); critical path = requantize Q31 multiply, **independent of LANES** → clean area↑/Fmax-flat Pareto. **First asap7 GDS produced** (L4_A24 @ 1.0ns: 1433µm², Fmax 509 MHz, wns −0.96ns). Synth-only area sweep (`physical/orfs/synth_area.sh`): nangate45 L1=12.3K→L16=22.9K µm² (16× MACs, only 1.86× area). Flow files: `physical/orfs/make/{run.sh,sweep.sh,<plat>/tinymac_accel/{config.mk,constraint.sdc}}`. **Gotchas:** (a) bazel-orfs route abandoned (PyPI fetch times out); use classic make flow. (b) Yosys 0.64 asserts `genrtlil.cc:2214` on signed/unsigned mixing — NO `$signed()` on unsigned whole wires, NO signed `integer` params in unsigned exprs, NO mixed-sign `?:` branches (yosys 0.9 + Verilator lint miss these). (c) param sweep via ORFS `VERILOG_TOP_PARAMS="LANES n ACC_W w"` (chparam) + `FLOW_VARIANT` per config. **Behavioral sim matches RTL** on cycle model (`ACCEL_CH_OVERHEAD=2`: latency = `n_outputs×(ceil(K/LANES)+2)`) **and saturation order** (per-LANES-chunk, not per-MAC → acc16 accuracy is lanes-dependent, 47–58/64). Measured AVG_CYCLES: L8=61,400, L16=46,670. Remaining: realistic-clock re-sweep, asap7 sweep (first GDS done), optional requantize pipelining (the ~3.7ns wall no synth recipe moves). (Automated search over these configs is driven externally by the [eda-rl](https://github.com/Shash976/eda-rl) engine, which calls the same ORFS make flow.)

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
./test_infer_host # should print "64/64 passed"
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
1. **Realistic-clock re-sweep** — re-run `physical/orfs/make/sweep.sh` at a clock ≈4 ns (close to the 3.72 ns critical path) so all LANES configs meet timing; collect updated area/Fmax/power into `sweep_results.csv`.
2. **ASAP7 sweep** — the first asap7 GDS exists (L4_A24 @ 1.0 ns); next is a ~12-config sweep at 0.8–1.2 ns. Doubles as the surrogate transfer-test set (train on nangate45, evaluate sample-efficiency on asap7).
3. **Optional: requantize pipelining** — split the Q31 64-bit multiply across 2 cycles; should roughly double Fmax for <1% throughput loss (requantize runs once per output channel, not per MAC). Measured: no synthesis recipe moves this path — only the RTL change does.

### Optimizer
The design-space optimizer now lives in **[eda-rl](https://github.com/Shash976/eda-rl)**; its roadmap is tracked there.
