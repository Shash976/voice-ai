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
- 🚧 Stage 6: Synthesizable accelerator RTL written (`rtl/accel/{int8_mac_array,requantize,tinymac_accel}.v`) + Verilator unit TB (`rtl/tb/`) bit-exact vs SW golden (45/45, LANES∈{2,4,8}, ACC_W∈{24,32}). **Full nangate45 GDS produced** via classic ORFS make flow on the company VM (`/opt/OpenROAD-flow-scripts`). LANES=4 ACC_W=24: ~19,738µm² (48% util), 230 FFs, **Fmax ≈269 MHz** (period_min 3.72ns); critical path = requantize Q31 multiply, **independent of LANES** → clean area↑/Fmax-flat Pareto. ⚠️ **Both of those facts changed in Stage 7** — requantize is now pipelined (Fmax 414.9 MHz) and the critical path moved to the LANES-*dependent* MAC accumulate path, so the Fmax-flat conclusion no longer holds and the LANES sweep must be re-run. **First asap7 GDS produced** (L4_A24 @ 1.0ns: 1433µm², Fmax 509 MHz, wns −0.96ns). Synth-only area sweep (`physical/orfs/synth_area.sh`): nangate45 L1=12.3K→L16=22.9K µm² (16× MACs, only 1.86× area). Flow files: `physical/orfs/make/{run.sh,sweep.sh,<plat>/tinymac_accel/{config.mk,constraint.sdc}}`. **Gotchas:** (a) bazel-orfs route abandoned (PyPI fetch times out); use classic make flow. (b) Yosys 0.64 asserts `genrtlil.cc:2214` on signed/unsigned mixing — NO `$signed()` on unsigned whole wires, NO signed `integer` params in unsigned exprs, NO mixed-sign `?:` branches (yosys 0.9 + Verilator lint miss these). (c) param sweep via ORFS `VERILOG_TOP_PARAMS="LANES n ACC_W w"` (chparam) + `FLOW_VARIANT` per config. **Behavioral sim matches RTL** on cycle model (`ACCEL_CH_OVERHEAD=2`: latency = `n_outputs×(ceil(K/LANES)+2)`) **and saturation order** (per-LANES-chunk, not per-MAC → acc16 accuracy is lanes-dependent, 47–58/64). Measured AVG_CYCLES: L8=61,400, L16=46,670. Remaining: realistic-clock re-sweep, asap7 sweep (first GDS done). ~~requantize pipelining~~ **done in Stage 7**. (Automated search over these configs is driven externally by the [eda-rl](https://github.com/Shash976/eda-rl) engine, which calls the same ORFS make flow.)
- 🚧 **Stage 7 — QuartzNet 15x5 ASR pivot** (branch `feat/quartznet-asr-accel`). Product change: the chip transcribes speech itself, replacing Whisper on the Pi. Design doc `docs/07_quartznet_pivot.md`; macro reference `docs/07a_sram_macro_notes.md`; setup `docs/07b_machine_setup.md`. **Done:** (a) **requantize pipelined over 2 cycles → 1.54× Fmax (268.8→414.9 MHz) for +3.3% area**, both variants measured on the same machine (reports committed in `physical/orfs/measured/`); critical path moved from the Q31 multiply to `i_in_chunk→acc`, which **is** LANES-dependent. (b) **TB widened past 8 lanes** (it had `static_assert(TB_LANES<=8)`; Verilator uses `VlWide` above 64-bit ports) — **12/12 pass, LANES∈{1,2,4,8,16,32} × ACC_W∈{24,32}**; LANES=32 is the design point. (c) **fakeram45 SRAM macro spike: full GDS, 0 DRC, WNS 0.00** — first macro integration in this repo. (d) **Software foundation: 187/187 descriptors reproduce** (`sw/tinyml_reference/quartznet_{topology,descriptors,ref}.py`). (e) ~~`firmware/quartznet/` C interpreter~~ **written and bit-exact** — table-driven over the QN_DESC format, four ops, static arena, zero tolerance: **187/187 descriptors + transcript byte-identical** on the full 15x5 and **27/27 + transcript** on the reduced config (the only one that exercises `OP_REQUANT`). Tiled T_TILE=32 (time) × DW_CH_TILE=64 (depthwise channels) with halo retention, which is 1,785 tiles at T_out=70 — so bit-exactness against the *untiled* NumPy golden is what proves the tiling. (f) ~~Stage C cycle model~~ **done — and the gate PASSES** (`sim/quartznet_cycles/quartznet_cycles.py`), driven off the interpreter's real schedule via `qn_tile_hook` → `firmware/quartznet/qn_schedule`, so cost model and interpreter cannot drift. Reproduces every committed figure (18,847,040 MACs/frame, 19.68 MB weights+qparams, 48.08 MB activations, 6.78 MB/s at 10 s, 90.65% PW). **Key numbers:** 18,847,040 params = 18.85 MB int8 = MACs *per output frame*; 942 MMAC/s for real-time; 90.6% of MACs are plain GEMM; layer-sequential dataflow with activations in external PSRAM ≈ 6.78 MB/s for a 10 s utterance. **Stage C results @ 414.9 MHz, 10 s utterance:** real-time at *every* LANES ∈ {8,16,32,64} — worst case 1.84× with a 64 KB weight buffer (QSPI-bound, flat across LANES), 3.3×→13.0× once weights stop re-streaming. LANES=32: 372.0 M cycles, 0.897 s, 10,511 MMAC/s, 79.2% array utilisation, 11.2× real-time. **Gotchas:** (a) cross-layer frame tiling is INVALID — cumulative depthwise receptive field is 4,012 frames (80.2 s); tile *within* each layer instead. (b) `ACC_W=32` is mandatory, not a swept axis (ACC_W=24 saturates for any pointwise with c_in≥260, i.e. every layer from B1 on). (c) `fakeram45_1024x32` is the densest macro at 4.005 µm²/byte — the larger `2048x39` is 12.1% *worse* per byte. (d) SRAM will be ~90% of die area, so the dominant area knob is SRAM capacity, not LANES. (e) **the on-chip WEIGHT buffer, not LANES, decides whether the part is QSPI-bound.** At 64 KB, 93/187 descriptors get re-streamed once per time tile → 282 MB/utterance, 14.3× the 19.68 MB floor, and QSPI sets the wall time at every LANES. The widest descriptor is 268,288 B, so **262 KB removes all re-streaming** and QSPI collapses to the floor (0.38 s); the design turns compute-bound. (f) SRAM ports: 8 paired `fakeram45_1024x32` bank pairs = 64 B/cycle, and a MAC eats 2 B (one activation + one weight), so 8 pairs sustain exactly LANES=32 — LANES=64 needs 16 pairs or utilisation falls to 46.3%. (g) the golden model reads every operand from channel 0 but writes at `out_off`, even for a channel-split ADD; that asymmetry *is* the format — matching it is required for bit-exactness. **Remaining:** real int8 weights via NeMo→ONNX→ORT PTQ; LANES resweep and macro orientation (both need OpenROAD, not built here).

---

## Environment Split

| Task | Machine |
|------|---------|
| Python ML (train, convert, export) | **Windows** (has GPU, PyTorch, TFLite) |
| Hardware (Verilator, RV32 cross-compile, ORFS) | **WSL** (Ubuntu on the same machine) |

The repo lives on Windows at `C:\Users\shash\Desktop\Code\voiceAI`. WSL has a **separate copy** at `~/voiceAI` (`/home/shashg/voiceAI`) — NOT a symlink to `/mnt/c/...`. Always edit files in the WSL copy when making hardware changes; sync back to Windows manually (or via git).

> **The split above describes the original dev box.** Stage 7 was done on a plain Linux
> machine with **no sudo**, where everything (Python *and* hardware tooling) runs in one
> place. See `docs/07b_machine_setup.md` for the portable setup. Two things to know before
> starting work anywhere new:
> - **Python goes in a conda env**: `conda env create -f environment.yml && conda activate voiceai`.
> - **A RISC-V toolchain may be missing.** It is needed only for firmware and the
>   full-system PicoRV32 sim — the unit TB, ORFS flows, and the whole Python chain do not
>   need it. The `sim/verilator/sim_main.cpp` latency change from Stage 7 (+1 drain cycle per
>   op) is verified against the RTL by `rtl/tb`, **and as of 2026-08 also end-to-end**: on a
>   machine with `riscv64-linux-gnu-gcc` present, `make -C sim/verilator run` completes
>   64/64 correct, `avg_cycles=61769` (LANES=8, ACC_W=32). That is 369 cycles above the
>   pre-Stage-7 documented `L8=61,400` baseline — far more than the `+1`/call change alone
>   predicts (4 accel calls/inference → ~+4 cycles) — but not investigated further; likely a
>   different toolchain built the original baseline binary, not a drain-cycle bug (`rtl/tb`
>   already proves that formula bit-exact against real RTL independently). Re-confirm on a
>   machine without this toolchain if the gap needs to be closed.

> **Toolchain state of the current machine (measured 2026-08, Stage 7 software work).**
> Probe before assuming — this box differs from both boxes described above.
> - **No conda.** System `python3` 3.10.12 + `numpy` 1.21.5 is enough for the entire pure
>   software chain: `quartznet_{topology,descriptors,ref}.py`, the goldens, and
>   `sim/quartznet_cycles/`. Only the NeMo/ONNX PTQ work needs the conda env.
> - **`riscv64-linux-gnu-gcc` 11 IS present** and targets `-march=rv32imc -mabi=ilp32`;
>   `firmware/picorv32_baremetal` builds clean (firmware.bin, 144,324 B). This contradicts
>   the "currently missing" note in `docs/07_quartznet_pivot.md` — and the full-system sim
>   **is** runnable here: `make -C sim/verilator run` completes 64/64 correct (see the
>   drain-cycle note above). `riscv32-unknown-elf-gcc` is absent; use the
>   `CROSS ?= riscv64-linux-gnu` default.
> - **No OpenROAD binary.** `~/OpenROAD-flow-scripts` is checked out but its build stopped
>   at 81% (`build_openroad.log`) and no `openroad` executable exists, so every ORFS item is
>   blocked here. System `yosys` is **0.9**, which is exactly the version that *misses* the
>   signed/unsigned assert 0.64 catches — do not treat a clean 0.9 run as synthesis passing.
> - **Verilator is 4.038**, not the **5.048** `docs/07b_machine_setup.md` documents; `rtl/tb`
>   and `sim/verilator` were not re-run here.

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

### QuartzNet interpreter + Stage C cost model (Stage 7, x86 host — no RV32/Verilator/ORFS)
```bash
cd firmware/quartznet
make goldens      # regenerate both reference configs into build/ (python3 + numpy)
make host         # build + run the bit-exact test → "PASS — 2/2 configurations bit-exact"
make schedule     # build ./qn_schedule, the tile-schedule dumper

python3 sim/quartznet_cycles/quartznet_cycles.py            # 10 s utterance, LANES 8/16/32/64
python3 sim/quartznet_cycles/quartznet_cycles.py --help     # --seconds/--lanes/--banks/--wt-buf-kb
```
The goldens under `build/` are gitignored and reproducible from the seed — regenerate, never
commit. The cost model shells out to `qn_schedule`, which drives the *real* interpreter's tile
walker, so the schedule it prices cannot drift from the one the C executes.

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

### Stage 7 (current — branch `feat/quartznet-asr-accel`)
1. **Real int8 weights** — NeMo `stt_en_quartznet15x5` → ONNX → ONNX Runtime static per-channel PTQ. *Deferred:* needs torch/onnx/onnxruntime/NeMo, none installed here, and it changes no tensor shape or integer path that the interpreter and cost model have already validated — everything so far runs on *seeded random* weights at the true shapes, which proves format and arithmetic but says nothing about accuracy. Q-ASR measured only +0.29% WER for W8A8, so this is expected to be low-risk.
2. **Re-run the LANES sweep** — the pre-Stage-7 conclusion "area rises with LANES, Fmax stays flat" is now void; the critical path moved onto the LANES-dependent MAC accumulate path. *Deferred:* needs OpenROAD, which is not built on this machine (see Environment Split).
3. **Macro orientation experiment** — configs staged at `physical/orfs/make/nangate45/sram_spike/config_{mirror,outward,r0grid}.mk`. 4 macros routed with 0 DRC but that is too few to expose the west-edge-pin problem; test before committing to an ~18-macro floorplan. *Deferred:* same, needs OpenROAD.

### Carried over from Stage 6
4. **Realistic-clock re-sweep** at a clock near the (now 2.41 ns) critical path.
5. **ASAP7 sweep** — first GDS exists (L4_A24 @ 1.0 ns); next is ~12 configs at 0.8–1.2 ns.
6. **`physical/orfs/synth_area.sh` is not portable** — hardcodes `$HOME/OpenROAD-flow-scripts` and a bare `yosys`. Use `run.sh` (honours `ORFS_DIR`) meanwhile.

### Optimizer
The design-space optimizer lives in **[eda-rl](https://github.com/Shash976/eda-rl)**; its roadmap is tracked there. Note Stage 7 changes its search space: SRAM capacity, not LANES, is now the dominant area knob.
