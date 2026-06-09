# 00 — Onboarding Overview

Welcome. This doc is the map. Read it first, then dive into the numbered docs for
each subsystem. Everything here is verified against the actual code as of June 2026.

---

## What are we building?

A **privacy-first pocket voice recorder** that is really *two chips*:

1. **A Raspberry Pi** — captures audio and runs full speech-to-text (`whisper.cpp`)
   *only when there is actually speech*.
2. **A custom always-on chip** (the thing this repo designs) — listens continuously
   at very low power, runs a tiny neural net called **TinyVAD** (Voice Activity
   Detection), and just answers one question: *"is this audio speech or silence?"*.
   When it says "speech", it wakes the Pi.

Running whisper.cpp on every second of audio is expensive. The tiny chip is the
power-saving gatekeeper. **This repo is about designing, simulating, and physically
laying out that gatekeeper chip.**

> ⚠️ The Raspberry Pi half is *aspirational/context* in this repo — there's no Pi
> code checked in yet (`benchmark_whisper.py` is the only Pi-adjacent script). The
> real, working content is the **custom chip pipeline**: train → quantize → compile
> → simulate → accelerate → optimize → (eventually) lay out.

---

## The 6 stages

```
Stage 1–2  TinyVAD: train in PyTorch, quantize to int8, export to C    ✅ done
Stage 3    Run it on a simulated RISC-V CPU (PicoRV32 in Verilator)    ✅ done
Stage 4    Add a hardware MAC accelerator (behavioral C++ model)       ✅ done
Stage 5    Auto-search the accelerator's design knobs (the optimizer)  ✅ done
Stage 6    RTL → GDS: turn it into a real chip layout (OpenROAD/ASAP7)  🚧 GDS produced
```

Each stage feeds the next. The doc map below tells you which file to read for each.

| Stage | What it is | Doc |
|-------|-----------|-----|
| 1–2 | The TinyVAD model + int8 quantization + the C inference engine | [01_model_and_quantization.md](01_model_and_quantization.md) |
| 3 | PicoRV32 CPU, bare-metal firmware, the Verilator simulator | [02_firmware_and_simulation.md](02_firmware_and_simulation.md) |
| 4 | The memory-mapped MAC accelerator + its firmware driver | [03_accelerator.md](03_accelerator.md) |
| 5 | The Python design-space optimizer (agents + reward + benchmark) | [04_optimizer.md](04_optimizer.md) |
| 6 | Synthesizable RTL, Verilator correctness gate, ORFS → GDS, physical metrics | [06_rtl_to_gds.md](06_rtl_to_gds.md) |
| — | Every command in one place | [05_commands_cheatsheet.md](05_commands_cheatsheet.md) |

The authoritative long-form plan is [`../pocket_ai_voice_recorder_riscv_tinyml_plan.md`](../pocket_ai_voice_recorder_riscv_tinyml_plan.md).
The top-level [`../README.md`](../README.md) is a good narrative intro but is slightly
less precise than these docs — when they disagree, trust the code and these docs.

---

## The single most important thing: the two-machine split

This project lives on **one physical Windows 11 machine** but uses **two
environments**:

| You are doing… | Run it in… | Why |
|----------------|-----------|-----|
| ML training, TFLite conversion, weight/test-vector export, the optimizer | **Windows** (PowerShell + Python venv) | Has the GPU, PyTorch, TensorFlow |
| Cross-compiling firmware, running Verilator, (later) OpenROAD | **WSL** (Ubuntu) | The RISC-V toolchain + Verilator live here |

**Critical gotcha:** the repo exists *twice*.
- Windows: `C:\Users\shash\Desktop\Code\voiceAI`
- WSL: `~/voiceAI` (i.e. `/home/shashg/voiceAI`) — **a separate git clone, NOT a
  symlink** to the Windows path.

So if you edit a Verilog/C firmware file on Windows, WSL won't see it until you
sync (via git, or by editing directly in the WSL copy). When you do hardware work,
edit in the WSL copy. Keep them in sync with git.

```
┌── Windows ──────────────────┐        ┌── WSL (Ubuntu) ─────────────────┐
│ C:\...\voiceAI              │  git   │ ~/voiceAI  (separate clone)     │
│  • train_tiny_vad.py        │ <────> │  • firmware cross-compile       │
│  • convert_to_tflite.py     │  sync  │  • sim/verilator (Verilator)    │
│  • export_weights.py        │        │  • rtl/tb (RTL unit tests)      │
│  • optimizer/ (Python)      │        │  • physical/orfs (OpenROAD/GDS) │
└─────────────────────────────┘        └─────────────────────────────────┘
```

(If you were on a Mac/Linux machine, you wouldn't need the WSL split — it's only
because the primary box is Windows.)

---

## End-to-end data flow (how a single audio chunk becomes a decision)

```
1s of audio @ 16 kHz
    │  extract_logmel()  (speech_simulator.py)
    ▼
int8 log-mel spectrogram  [49 time frames × 40 mel bins]
    │  tiny_vad_infer()   (the int8 C engine)
    ▼
2 logits  [silence_score, speech_score]
    │  argmax
    ▼
speech (1) or silence (0)
```

The exact same int8 math runs in three places, and they must agree bit-for-bit:
- **Python/TFLite** (the golden reference)
- **The C engine** on x86 (host sanity test) and on RISC-V (in the sim)
- **The accelerator** (C++ behavioral model in the sim)

That agreement is the whole correctness story. Test vectors (64 of them) are baked
into the firmware and checked every run: the target is **64/64 correct**.

---

## The artifact chain (what generates what)

```
train_tiny_vad.py
  └─> tiny_vad_best.pt        (trained PyTorch weights)
        convert_to_tflite.py
          └─> tiny_vad_int8.tflite    (quantized model — the source of truth)
                ├─ export_weights.py    ─> firmware/tinyengine_port/tiny_vad_weights.h
                └─ gen_test_vectors.py  ─> firmware/tinyengine_port/tiny_vad_test_vectors.h
                      │  (both headers are AUTO-GENERATED — never hand-edit)
                      ▼
                firmware/picorv32_baremetal/  (cross-compile → firmware.bin)
                      ▼
                sim/verilator/sim_main.cpp     (Verilator testbench runs firmware.bin)
                      ▼
                CSV results + cycle counts
                      ▼
                optimizer/  (sweeps accelerator configs, ranks them)
                      ▼
                rtl/accel/  (synthesizable Verilog: int8_mac_array, requantize, tinymac_accel)
                      │  rtl/tb/ Verilator unit TB  (45/45 bit-exact vs SW golden)
                      ▼
                physical/orfs/make/  → OpenROAD-flow-scripts → GDS
                      (LANES=4 ACC_W=24: ~19,738 µm², ~269 MHz Fmax, 231 FFs)
```

Two headers are **generated** and marked "do not edit":
`tiny_vad_weights.h` and `tiny_vad_test_vectors.h`. If you change the model, you
must regenerate them (see doc 01).

---

## Fastest possible "is it alive?" check

If you just want to confirm the toolchain works before reading anything else:

```bash
# In WSL:
cd ~/voiceAI/firmware/tinyengine_port
make host          # compile the int8 C engine for x86
./test_infer_host  # expect: all test vectors pass, max error ≤ 2 LSB
```

That runs the inference engine on your laptop CPU (no RISC-V, no Verilator) and
checks it against the baked-in vectors. If that passes, the model + C engine are
healthy. Then move on to doc 02 to run the full RISC-V simulation.

See [05_commands_cheatsheet.md](05_commands_cheatsheet.md) for every command.

---

## Key results to know (so the numbers mean something)

**Behavioral sim figures** (per inference, one audio chunk):

| Configuration | Cycles / inference | Time @ 100 MHz | Speedup | Correct |
|---|---|---|---|---|
| Stage 3 — pure software (no accelerator) | ~11.2 M | ~112 ms | 1× (baseline) | 64/64 |
| Stage 4 — accelerator, 8 MAC lanes | ~58–66 K¹ | ~0.6 ms | **~170–191×** ¹ | 64/64 |
| Stage 4 — accelerator, 16 MAC lanes | ~43–49 K¹ | ~0.4 ms | **~230–258×** ¹ | 64/64 |

¹ Cycle model updated to match RTL (per-channel overhead; WSL rebuild pending — re-pin
with `measure_real.py` to get exact current numbers).

**Stage 6 physical results** (nangate45, synthesized + placed + routed, LANES=4 ACC_W=24):

| Metric | Value |
|--------|-------|
| Die area | ~19,738 µm² (48% utilization) |
| Flip-flops | 231 |
| Fmax | **~269 MHz** (period_min 3.72 ns) |
| Critical path | requantize Q31 multiply — independent of LANES |
| Area vs lanes | 1×→16× MACs costs only 1.86× area (fixed overhead dominates) |

The ~191× behavioral speedup plus the physical numbers together answer the key
question: a dedicated MAC array is fast (few hundred µs/inference) and small
(~20 K µm² at 45 nm), making the always-on chip feasible.

---

## Glossary (terms you'll hit immediately)

- **TinyVAD** — the small 1D-CNN that classifies speech vs. silence. Our workload.
- **int8 quantization** — storing weights/activations as 8-bit integers instead of
  32-bit floats. The chip has no floating-point unit, so *everything* is integers.
- **PicoRV32** — a small open-source RISC-V CPU (~2000 lines of Verilog). Runs our
  firmware.
- **RTL (Register-Transfer Level)** — Verilog/SystemVerilog code that describes a
  digital *circuit* (not software). What chip designers write.
- **Verilator** — compiles RTL into a fast C++ simulation you run on your laptop.
  *Not* an FPGA — it's software simulating hardware.
- **Behavioral model** — C++ that *acts like* a hardware block without being real
  RTL yet. Our Stage-4 accelerator is behavioral (lives in `sim_main.cpp`).
- **MAC** — Multiply-ACcumulate, the `a*b + acc` operation that dominates neural nets.
- **Requantization** — after accumulating int8×int8 into int32, scaling the result
  back down to int8 for the next layer (fixed-point multiply + shift).
- **GDS / GDSII** — the final chip-layout file a fab uses to make masks. Stage 6's output.
- **ORFS** — OpenROAD-flow-scripts; runs the whole RTL→GDS pipeline. Stage 6.
- **ASAP7** — an academic 7nm process design kit (cell library) for realistic
  area/timing estimates. The project's target PDK; nangate45 is used first (easier).
- **nangate45** — an open 45 nm PDK used for development/validation before asap7.
- **Fmax** — maximum operating frequency; the reciprocal of the critical path delay.
- **WNS / TNS** — worst/total negative slack: how much timing is violated at a given
  clock target. Negative = design won't work at that clock.
- **DSE** — design-space exploration: searching over hardware configs to find the best
  tradeoff. What the optimizer currently does. Distinct from RL (see `AGENTS.md`).
