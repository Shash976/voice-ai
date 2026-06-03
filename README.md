# Pocket AI Voice Recorder — RISC-V TinyML Accelerator

A privacy-preserving, local-first voice recorder. The end goal is a two-chip system: a Raspberry Pi for audio capture and full speech transcription, and a custom silicon chip (designed here from scratch) that handles always-on voice activity detection at low power. This repo covers designing, simulating, and physically synthesizing that custom chip.

Full project plan: [`pocket_ai_voice_recorder_riscv_tinyml_plan.md`](pocket_ai_voice_recorder_riscv_tinyml_plan.md)

---

## The Big Picture — Why Two Chips?

Running whisper.cpp (full speech-to-text) on every 1-second audio chunk is expensive — it takes hundreds of milliseconds and significant CPU power. You don't want to do that when someone is just breathing or there's background noise.

The solution: a tiny always-on chip that listens continuously and only wakes up the Pi when it hears actual speech. That chip runs **TinyVAD** — a small neural network (voice activity detector) that classifies each audio chunk as *speech* or *silence*. Because TinyVAD is tiny and runs on a minimal RISC-V CPU, it can run continuously at very low power.

```
┌─────────────────────────────────────────────────────────────────┐
│ Raspberry Pi (main host)                                        │
│                                                                 │
│  Mic → audio frontend  ──▶  Custom Chip       │
│                                               (PicoRV32 +       │
│                                                AI accelerator)  │
│                                                    │            │
│           ◀──────────────── speech/silence ────────┘            │
│           │                                                     │
│           └─ if speech: run whisper.cpp → transcript            │
└─────────────────────────────────────────────────────────────────┘
```

**The Pi's job:** capture audio, send to the custom chip, and when the chip reports "speech", run whisper.cpp for full transcription.

**The custom chip's job:** run TinyVAD on the received features and return a yes/no speech decision. That's it — small, fast, always on.

**This repo** is about building and verifying that custom chip: training TinyVAD, compiling it to run on a tiny RISC-V CPU, simulating that CPU in software, adding a hardware matrix-multiply accelerator to make it faster, and eventually physically synthesizing it to get a real chip layout.

---

## Stage Progress

```
Stage 1–2  TinyVAD: train, quantize to int8, export to C         ✅ complete
Stage 3    PicoRV32 software baseline in Verilator simulation     ✅ complete
Stage 4    Behavioral int8 AI accelerator                         ✅ complete
Stage 5    Design-space optimization (agents + reward + benchmark) ✅ complete
Stage 6    RTL synthesis → place-and-route → GDS (real chip)      🔜 next
```

---

## Environment Setup

| Task | Where to run |
|------|---------|
| ML training, TFLite conversion, weight/vector export | **Windows** (Python venv, has GPU) |
| Firmware cross-compilation, Verilator simulation | **WSL** (Ubuntu, `~/voiceAI`) |

The WSL copy at `~/voiceAI` is a **separate git clone**, not a symlink. Edit hardware files there; sync back to Windows manually via git.
 
No need for a WSL if you have a Mac/Linux-based OS.
---

## Stage 1–2: TinyVAD Model

### What it does

Trains a small 1D CNN on the Google Speech Commands v2 dataset (35 word classes + silence/background). The model's job is binary: *speech* vs. *silence*. After training in PyTorch, it gets quantized to int8 — all weights and activations become 8-bit integers instead of 32-bit floats.

**Why int8?** The custom chip has no floating-point unit. Everything must be integers. int8 also makes weights 4× smaller than float32, fitting the whole model in a few kilobytes of RAM.

**Model architecture** — input is a log-mel spectrogram of 1 second of audio: 49 time frames × 40 mel frequency bins, encoded as int8.

| Layer | What it does | Output shape |
|-------|-------------|--------------|
| Conv0 (k=5, s=2, pad=2) | Slides a 5-frame filter over time, extracts 32 features | [25 × 32] |
| Conv1 (k=3, s=2, pad=1) | Slides a 3-frame filter, extracts 64 features | [13 × 64] |
| GlobalAvgPool | Averages across the time axis → single vector | [64] |
| FC0 (dense) | Matrix-vector multiply, 64→32 | [32] |
| FC1 (dense) | Matrix-vector multiply, 32→2 | [2] logits |

The output is two scores (logits). Whichever is larger wins: logit[0] = silence score, logit[1] = speech score.

All tensors are stored in `[time, channel]` order — time steps in the outer dimension, feature channels in the inner dimension. This matches TFLite's layout and is important: past bugs caused completely wrong outputs when this was swapped.

### Files

```
train_tiny_vad.py                         PyTorch training
convert_to_tflite.py                      Converts trained PTH → ONNX → int8 TFLite
sw/tinyml_reference/export_weights.py     Reads TFLite, writes tiny_vad_weights.h
sw/tinyml_reference/gen_test_vectors.py   Runs TFLite on samples, writes tiny_vad_test_vectors.h

firmware/tinyengine_port/
  tiny_vad_infer.c / .h      Hand-written int8 C inference engine (no stdlib, no malloc)
  tiny_vad_weights.h         Auto-generated weight tables — do not edit
  tiny_vad_test_vectors.h    Auto-generated test inputs + expected labels — do not edit
  test_infer_host.c          x86 harness to check the C engine against TFLite
```

### How to run (Windows)

```powershell
# Step 1: train (downloads ~2.3 GB Speech Commands dataset on first run)
python train_tiny_vad.py
# Outputs: tiny_vad_best.pt  tiny_vad.onnx

# Step 2: quantize to int8 TFLite
python convert_to_tflite.py
# Output: tiny_vad_int8.tflite

# Step 3: export weights and test vectors to C headers (re-run if model changes)
python sw/tinyml_reference/export_weights.py
python sw/tinyml_reference/gen_test_vectors.py
```

### Sanity check (WSL)

```bash
cd firmware/tinyengine_port
make host          # compiles the C inference engine for x86
./test_infer_host  # runs against baked-in test vectors
```

**Expected output:**
```
Running 42 test vectors...
42/42 passed
Max logit error vs TFLite: 2 LSB
```

"2 LSB" means the int8 C engine's output differs from the TFLite float reference by at most 2 counts — normal quantization rounding noise. This confirms the C inference engine is correct before it gets compiled for the RISC-V chip.

---

## Background: What is Verilator and What is RTL?

Skip this if you already know chip design basics.

### RTL (Register-Transfer Level)

RTL is a style of hardware description where you write code (in Verilog or SystemVerilog) that describes exactly what happens in a digital circuit on every clock cycle — which registers store what values, how signals flow between them. RTL is what chip designers write. It is not software that runs on a CPU; it is a description of the circuit itself.

Example: a hardware adder in RTL is just `assign out = a + b;`. A register that holds a value until the next clock: `always @(posedge clk) reg_out <= reg_in;`.

RTL is "synthesizable" if a tool (like Yosys) can automatically convert it to an actual circuit made of logic gates. That synthesis step is what Stage 6 does.

### Verilator

Verilator is a tool that compiles RTL Verilog into a C++ class. You then write a C++ "testbench" that instantiates that class and drives it clock-by-clock, just like the real hardware would be driven. The result is a fast software simulation of the chip.

This is different from an FPGA: an FPGA is real hardware that is reconfigured to implement the circuit. Verilator is a software simulation running on your laptop.

**Why Verilator instead of just running C code?** Because we want to simulate the actual RISC-V CPU (PicoRV32) executing real machine code — fetch, decode, execute — counting real clock cycles. A plain C program wouldn't tell us how many hardware clock cycles the chip would need.

### What "behavioral model" means

A behavioral model is C++ code that *acts like* a hardware block without being actual RTL. The accelerator in Stage 4 is behavioral: the testbench simulates what the accelerator would do (run matrix multiplies) but the accelerator itself is not yet written in Verilog. This lets us test the firmware interface and measure cycle counts before committing to RTL. Stage 6 replaces the behavioral model with real synthesizable Verilog.

---

## Stage 3: PicoRV32 Software Baseline

### What is PicoRV32?

PicoRV32 is an open-source RISC-V CPU written in Verilog (~2000 lines). It implements the RV32IMC instruction set — 32-bit base integer instructions, multiply/divide, and compressed (16-bit) instructions. It is small enough to eventually fit on a custom chip alongside our accelerator.

In this stage, PicoRV32 runs the TinyVAD firmware **in pure software** — every multiply-accumulate in every Conv/FC layer executes as individual RISC-V instructions. No accelerator. The point is to measure how slow it is so we have a baseline to compare against.

### The SoC wrapper

`rtl/soc/picorv32_soc.v` is a tiny Verilog wrapper around the PicoRV32 core. Its only job is to expose the CPU's memory bus signals to the outside world so the Verilator testbench can handle memory and I/O in C++. The wrapper itself has no RAM or UART — those live in the testbench.

```
┌─────────────────────────────────────────────────────┐
│  Verilator testbench (sim_main.cpp)                 │
│                                                     │
│  uint8_t ram[256KB]    ← holds firmware + data      │
│  putchar() for UART    ← firmware output to stdout  │
│  accel emulation       ← Stage 4 only               │
│                │                                    │
│         memory bus signals                          │
│         (addr, data, wstrb, valid, ready)           │
│                │                                    │
│  ┌─────────────▼──────────────┐                     │
│  │  picorv32_soc.v (Verilog)  │                     │
│  │  └── picorv32 CPU core     │                     │
│  └────────────────────────────┘                     │
└─────────────────────────────────────────────────────┘
```

### The memory bus

PicoRV32 uses a simple handshake protocol for all memory accesses (instruction fetches, loads, stores):

1. CPU asserts `mem_valid=1` and puts the address on `mem_addr`
2. Testbench responds: for a read, it puts data on `mem_rdata` and asserts `mem_ready=1`; for a write, it receives `mem_wdata`
3. CPU latches the response and deasserts `mem_valid`

Every single memory transaction — including reading the next instruction — goes through this bus. The testbench serves all of them from a plain `uint8_t ram[256*1024]` array. This is 0-wait-state memory: every access takes exactly 1 clock cycle.

### Memory map

The firmware uses different address ranges for different purposes. The CPU itself doesn't know what's at each address; the testbench decides:

| Address | What it is | How it works |
|---------|-----------|--------------|
| `0x00000000–0x0003FFFF` | 256 KB RAM | Testbench reads/writes `ram[]` array |
| `0x10000000` | UART TX | Testbench calls `putchar()` — firmware output appears on stdout |
| `0x10000004` | SIM_EXIT | Testbench sets `sim_done = true` — ends the simulation loop |
| `0x20000000–0x20000FFF` | AI accelerator registers | Stage 4 only |

The firmware writes to `0x10000000` to print a character — there is no real UART hardware, just a testbench side-effect.

### The firmware

`firmware/picorv32_baremetal/` contains the bare-metal firmware that runs on PicoRV32:

- **`startup.S`** — the very first code that runs after reset (address 0). Sets the stack pointer, zeroes `.bss`, then calls `main()`. Must be at address 0 because PicoRV32 always boots from `0x00000000`.
- **`linker.ld`** — tells the linker where to place code (at 0x0), where the stack starts (top of 256 KB RAM), and how to lay out sections.
- **`syscalls.h`** — implements `uart_puts()`, `uart_putu32()`, etc. by writing individual bytes to address `0x10000000`. Also provides `rdcycle()` which reads PicoRV32's hardware cycle counter register.
- **`main.c`** — loops over 64 test vectors, calls `tiny_vad_infer()` for each, measures cycles with `rdcycle()`, and prints results as CSV over UART.

**Why cross-compile?** The firmware runs on a 32-bit RISC-V CPU, not x86. `riscv64-linux-gnu-gcc` with `-march=rv32imc` produces machine code for RV32, which Verilator then executes instruction-by-instruction inside the simulated PicoRV32.

**Build flag note:** The toolchain defaults to PIE (Position-Independent Executable) mode, which generates GOT-indirect addressing for linker symbols like `_stack_top`. On bare-metal with no dynamic linker, this crashes at startup. `-fno-pic -fno-pie` forces direct addressing. `ENABLE_DIV=1` is required because the global average pool layer uses a hardware divide instruction.

### How to run (WSL)

```bash
cd sim/verilator
make check-deps   # verify riscv toolchain and verilator are installed
make run          # compile firmware → compile Verilog → link testbench → simulate
make vcd          # same + dump sim_out.vcd for GTKWave waveform inspection
make clean
```

What `make run` does internally:
1. Cross-compiles firmware to `firmware.bin` (raw RV32 machine code)
2. Runs Verilator on `picorv32_soc.v` to generate a C++ model
3. Compiles the testbench `sim_main.cpp` with the generated model
4. Runs the resulting binary with `firmware.bin` as input

### Expected output

**stdout** — CSV printed by the firmware over the simulated UART, one row per test vector:
```
vec,label,result,correct,logit0,logit1,cycles
0,1,1,1,-42,18,11196638
1,0,0,1,23,-9,11201027
2,1,1,1,-38,15,11193880
...
```

**stderr** — from the testbench itself (not firmware):
```
[sim] Loaded 12480 bytes from firmware.bin
[sim] Reset released — starting simulation
[sim] mac_lanes=8 acc_width=32          ← (ignored in pure-SW; no hooks active)
[sim] TIMEOUT after 50000000 cycles
```

> **Note:** in pure software each inference costs ~11.2 M cycles, so a full
> 64-vector sweep (~716 M cycles) exceeds the simulator's 50 M-cycle cap and
> times out. The Stage-3 baseline is therefore taken **per inference** (vec 0 ≈
> 11,196,638 cycles) — which is exactly why the accelerator in Stage 4 exists.

**Column meanings:**

| Column | Meaning |
|--------|---------|
| `vec` | Test vector index 0–63 |
| `label` | Ground-truth class: 0 = silence, 1 = speech |
| `result` | TinyVAD prediction |
| `correct` | 1 if prediction = label |
| `logit0` | Silence score (int8). If this > logit1, predicted silence |
| `logit1` | Speech score (int8). If this > logit0, predicted speech |
| `cycles` | Clock cycles for this inference (from `rdcycle()` before and after) |

**What it means:** ~11.2 M cycles **per inference**. At a hypothetical 100 MHz clock that is ~112 ms per chunk — far too slow for always-on use. This is the cost of doing everything in scalar RISC-V software with no hardware assistance: every multiply-accumulate in every Conv/FC layer (~242 K MACs) runs as individual instructions, and PicoRV32's multi-cycle multiply makes each MAC expensive. It is the baseline the accelerator must beat.

---

## Stage 4: Behavioral AI Accelerator

### The idea

A dedicated hardware unit for multiply-accumulate (MAC) operations can do in parallel what the CPU does serially. A single RISC-V multiply instruction takes 1–3 cycles. An 8-lane MAC array can do 8 multiplications and accumulate the results in 1 cycle. For a matrix-vector multiply with 64×32 = 2048 MACs, that is a potential 8× speedup just from parallelism — on top of not spending cycles on loop control, pointer arithmetic, etc.

The accelerator is a **memory-mapped peripheral**: the firmware programs it by writing values to specific addresses (its registers), then triggers it by writing a command. From the CPU's perspective, it looks identical to writing to RAM — same bus, same address space, just a different address range.

### What the accelerator does

It implements two operations that cover the hot layers in TinyVAD:

- **MATVEC (cmd=1):** dense layer — output[M] = W[M][K] · input[K] + bias[M], fully requantized to int8
- **CONV1D (cmd=2):** 1D convolution — slides filters of shape [out_ch, in_ch, kernel] over a [in_len, in_ch] input tensor

Both operations read operands directly from the firmware's RAM (via pointer registers), compute in int32 accumulators, apply per-channel requantization (the same fixed-point arithmetic as the software path), and write int8 results back to RAM.

### Accelerator register map (base `0x20000000`)

| Offset | Register | R/W | Purpose |
|--------|----------|-----|---------|
| `0x00` | STATUS | R | 0 = idle, 1 = busy — firmware polls this |
| `0x04` | CMD | W | Write 1 for MATVEC, 2 for CONV1D — triggers execution |
| `0x08` | IN_PTR | W | RAM address of input tensor |
| `0x0C` | WT_PTR | W | RAM address of weight tensor |
| `0x10` | BIAS_PTR | W | RAM address of int32 bias array |
| `0x14` | MULT_PTR | W | RAM address of per-channel Q-multiplier array |
| `0x18` | RSHI_PTR | W | RAM address of per-channel right-shift array |
| `0x1C` | OUT_PTR | W | RAM address where output is written |
| `0x20` | DIM_M | W | Output dimension (rows for MATVEC, out_ch for CONV1D) |
| `0x24` | DIM_K | W | Input dimension (cols for MATVEC, in_ch for CONV1D) |
| `0x28` | DIM_KERN | W | Kernel size (CONV1D only) |
| `0x2C` | DIM_INLEN | W | Input sequence length (CONV1D only) |
| `0x30` | DIM_STRIDE | W | Convolution stride (CONV1D only) |
| `0x34` | DIM_PAD | W | Zero-padding (CONV1D only) |
| `0x38` | ZP_IN | W | Input quantization zero-point |
| `0x3C` | ZP_OUT | W | Output quantization zero-point |
| `0x40` | RELU | W | 1 = apply ReLU after requantization |
| `0x44` | LAST_CYC | R | How many cycles the last operation took |

### What requantization means

After accumulating int8 inputs × int8 weights into an int32 accumulator, the result needs to be scaled back to int8 for the next layer. The scale factor is `(input_scale × weight_scale) / output_scale`, decomposed into a fixed-point multiply + shift: `result ≈ (accumulator × q_mult) >> 31 >> rshift`. This is done per output channel because TFLite uses per-channel weight quantization. The accelerator implements the same formula as the software path so outputs are bitwise identical.

### How the firmware uses it

`firmware/picorv32_baremetal/accel.c` — the driver. For a dense layer call:

```c
// 1. Wait for any previous operation to finish
while (ACCEL_STATUS != 0) {}

// 2. Write all operand pointers and dimensions
ACCEL_IN_PTR  = (uint32_t)input_ptr;
ACCEL_WT_PTR  = (uint32_t)weight_ptr;
ACCEL_BIAS_PTR= (uint32_t)bias_ptr;
ACCEL_DIM_M   = out_dim;
ACCEL_DIM_K   = in_dim;
// ... (zero-points, Q-mult/shift pointers, ReLU flag)

// 3. Trigger — writing CMD causes the accelerator to start immediately
ACCEL_CMD = ACCEL_CMD_MATVEC;   // = 1

// 4. Poll until done (busy-wait)
while (ACCEL_STATUS != 0) {}
```

The hook system in `tiny_vad_infer.c` makes this transparent — at startup, `main.c` sets:
```c
tinyvad_conv1d_hook = accel_conv1d;
tinyvad_dense_hook  = accel_dense;
```
From then on, every Conv and FC layer call goes through the accelerator instead of the software path. The inference loop itself doesn't change.

### Where the accelerator lives (Stage 4)

Right now, the accelerator is **not RTL** — it is a C++ function inside `sim_main.cpp`. When the testbench sees a write to `0x20000000–0x20000FFF`, it calls `accel_write()` which runs the C++ MAC logic directly. This is the "behavioral model."

Simulated latency: `ceil(total_MACs / MAC_LANES)` cycles, where `MAC_LANES = 8` (configurable at line 43 of `sim_main.cpp`). The testbench sets `accel_done_at = current_cycle + latency` and reports STATUS=busy until then. This models what real pipelined hardware would do without needing actual RTL.

### How to run (WSL)

```bash
cd sim/verilator
make run
```

Same command as Stage 3 — the firmware already has the hooks enabled. To try different parallelism:

```bash
# Edit MAC_LANES in sim/verilator/sim_main.cpp line 43, then:
make run
```

### Expected output

**stdout** — same format as Stage 3, same 64/64 correct, drastically fewer cycles:
```
vec,label,result,correct,logit0,logit1,cycles
0,1,1,1,-42,18,58577
1,0,0,1,23,-9,58572
...
63,1,1,1,-38,21,58581
---
correct=64/64 avg_cycles=58577
```

**stderr** — shows the accelerator being invoked for each layer of each inference:
```
[sim] Loaded 12480 bytes from firmware.bin
[sim] Reset released — starting simulation
[accel] cmd=2 M=32 K=40 in=0x0000c1a0 wt=0x... bias=0x... out=0x...   ← Conv0 for vec 0
[accel] cmd=2 M=64 K=32 in=0x... wt=0x... bias=0x... out=0x...        ← Conv1
[accel] cmd=1 M=32 K=64 in=0x... wt=0x... bias=0x... out=0x...        ← FC0
[accel] cmd=1 M=2  K=32 in=0x... wt=0x... bias=0x... out=0x...        ← FC1
...  (repeats for all 64 vectors)
[sim] Done in 3854228 cycles (wall: ~38.5 ms at 100 MHz) mac_lanes=8 acc_width=32
```

### Results

All cycle figures are **per inference**. Speedup is vs. the Stage-3 SW baseline
(11,196,638 cycles/inference), measured empirically (`optimizer/measure_real.py`).

| | Stage 3 — software only | Stage 4 — accel, 8 lanes | Stage 4 — accel, 16 lanes |
|---|---|---|---|
| Cycles / inference | ~11.2 M | ~58.6 K | ~43.4 K |
| Time at 100 MHz | ~112 ms | ~0.59 ms | ~0.43 ms |
| Accuracy | 64/64 | 64/64 | 64/64 |
| **Speedup** | 1× | **~191×** | **~258×** |

At `acc_width=32` the accelerator's output is bitwise identical to the software
path — its C++ requantization matches the firmware's exactly. (At `acc_width=16`
the accelerator deliberately reproduces hardware-accurate overflow, dropping to
47/64 — see Stage 5.)

---

## Stage 5 (Next): Design-Space Optimization

The behavioral model makes it cheap to try different hardware configurations. `MAC_LANES` is the most obvious knob: 1 lane = sequential, 16 lanes = 16 MACs per cycle. But there are others: accumulator width (int16 vs int32), dataflow strategy (weight-stationary vs output-stationary), buffer sizes.

The plan: a Python script sweeps these parameters, re-runs the Verilator simulation for each configuration, and records cycle count as a proxy for performance. Area (gate count) can be estimated proportionally to lane count. The result is a Pareto frontier of latency vs. area, which feeds directly into what RTL to actually write for Stage 6.

Candidate parameters to sweep (from the project plan):
```yaml
mac_lanes:           [1, 2, 4, 8, 16]
accumulator_width:   [16, 24, 32]
dataflow:            [output_stationary, weight_stationary]
```

---

## Stage 6 (Next): RTL to GDS — Making a Real Chip

### What this stage is

"RTL to GDS" means taking the hardware description (Verilog RTL) all the way to a GDSII file — the final photomask layout that a fab would use to manufacture the chip. This is done entirely with open-source tools.

### The toolchain

```
AI accelerator in Verilog  (write this in Stage 6)
        │
        ▼  Yosys (synthesis)
Logic gates (AND, OR, flip-flops, etc.) from the ASAP7 cell library
        │
        ▼  OpenROAD (floorplan + placement)
Gates placed on a 2D canvas with assigned X,Y coordinates
        │
        ▼  OpenROAD (clock tree synthesis)
Clock signal routed to all flip-flops with balanced delays
        │
        ▼  OpenROAD (routing)
Metal wires connecting all gates
        │
        ▼  GDS file
Physical layout — what the chip actually looks like under a microscope
```

**Yosys** converts RTL Verilog into a netlist of logic gates chosen from the target cell library (ASAP7 in our case).

**OpenROAD** handles everything from placement through routing — it figures out where to put each gate on the chip canvas and how to wire them together while meeting timing constraints.

**ASAP7** is an academic 7nm-node process design kit (PDK). It is not a real commercial process, but it models realistic 7nm-class cell sizes and timing, making it useful for research-grade area and speed estimates.

**OpenROAD-flow-scripts (ORFS)** is a wrapper that runs the entire Yosys → OpenROAD pipeline with a single command, driven by a configuration file that specifies the target platform (ASAP7), clock period, and which RTL files to compile.

### What gets synthesized

Not the full system — just the AI accelerator block. The PicoRV32 CPU core is already a known quantity (it is a widely-used open-source core). The custom part is the int8 multiplier array, its register file, and the requantization logic. That is what needs to be measured for area and timing to know if the design is viable.

### What we get out

- **Cell area** — how many square microns the circuit occupies
- **Worst Negative Slack (WNS)** — how many nanoseconds of timing margin remain at the target clock frequency; negative means it fails timing
- **Total Negative Slack (TNS)** — aggregate timing violations across all paths
- **GDS file** — the actual layout, viewable in KLayout

These numbers let us compare different accelerator configurations (from Stage 5's sweep) with real physical constraints, not just cycle count estimates.

---

## How the Stages Connect

```
Python (Windows)               WSL
─────────────────────          ──────────────────────────────────────────
train_tiny_vad.py
  → tiny_vad_best.pt
      convert_to_tflite.py
        → tiny_vad_int8.tflite
            export_weights.py  →  firmware/tinyengine_port/tiny_vad_weights.h
            gen_test_vectors.py → firmware/tinyengine_port/tiny_vad_test_vectors.h
                                       │
                                  firmware/picorv32_baremetal/  (cross-compile for RV32)
                                       → firmware.bin
                                           │
                                      sim/verilator/sim_main.cpp
                                           → Verilator simulation
                                               → CSV results + cycle counts
                                                   │
                                              Stage 5: sweep configs
                                                   → best config
                                                       │
                                                  Stage 6: write RTL
                                                   → ORFS / ASAP7
                                                       → GDS
```

---

## Repository Layout

```
train_tiny_vad.py                     Stage 1: PyTorch VAD training
convert_to_tflite.py                  Stage 2: PTH → int8 TFLite

sw/tinyml_reference/
  export_weights.py                   TFLite → C weight headers
  gen_test_vectors.py                 TFLite → C test vector headers

firmware/
  tinyengine_port/
    tiny_vad_infer.c / .h             int8 C inference engine (portable C, no stdlib)
    tiny_vad_weights.h                auto-generated weights (do not edit)
    tiny_vad_test_vectors.h           auto-generated test vectors (do not edit)
    test_infer_host.c                 x86 correctness check
  picorv32_baremetal/
    startup.S                         reset handler — sets stack, zeroes BSS, calls main
    linker.ld                         memory layout for RV32 bare-metal
    syscalls.h                        UART print helpers + rdcycle()
    main.c                            benchmark firmware: runs 64 vectors, prints CSV
    accel.h / accel.c                 accelerator firmware driver (Stage 4)

rtl/
  picorv32/                           PicoRV32 CPU core (upstream, unmodified Verilog)
  soc/picorv32_soc.v                  thin wrapper exposing CPU memory bus to testbench

sim/verilator/
  sim_main.cpp                        Verilator C++ testbench: RAM, UART, accelerator emulation

speech_commands/                      Google Speech Commands v2 training data
```

---

## Toolchain

| Tool | Purpose | Where |
|------|---------|-------|
| PyTorch + torchaudio | Model training | Windows |
| TensorFlow / tflite-runtime | int8 quantization + TFLite reference | Windows |
| `riscv64-linux-gnu-gcc` | Cross-compile firmware for RV32IMC | WSL |
| Verilator | Compile Verilog RTL into a C++ simulation model | WSL |
| GTKWave | View `.vcd` waveform dumps from `make vcd` | WSL |
| Yosys | RTL → gate-level netlist (Stage 6) | WSL |
| OpenROAD | Placement, routing, GDS (Stage 6) | WSL |
| OpenROAD-flow-scripts | Orchestrates the full RTL-to-GDS flow (Stage 6) | WSL (already cloned) |
| ASAP7 PDK | 7nm-class cell library for academic synthesis (Stage 6) | WSL |
