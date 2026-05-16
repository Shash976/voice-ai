# Pocket AI Voice Recorder with RISC-V TinyML Accelerator

## Goal

Build a local-first pocket AI voice recorder prototype that starts as a Raspberry Pi software system, evolves into a simulated PicoRV32 + TinyML hardware accelerator platform, and ends with an RTL-to-GDS flow using OpenROAD/OpenROAD-flow-scripts in ASAP7. The project uses open-source blocks wherever possible and applies reinforcement learning or agentic optimization loops at each stage.

## One-line project pitch

> A privacy-preserving pocket AI voice recorder that uses Raspberry Pi for local speech AI, PicoRV32 + TinyML acceleration for low-power always-on inference, Verilator for hardware/software co-simulation, and OpenROAD-flow-scripts for RTL-to-GDS exploration in ASAP7.

---

## Recommended staged plan

```text
Stage 1: Raspberry Pi software prototype
    ↓
Stage 2: TinyML model/runtime path
    ↓
Stage 3: PicoRV32 + accelerator behavioral simulation
    ↓
Stage 4: Full hardware/software Verilator co-simulation
    ↓
Stage 5: RL/agentic design-space optimization
    ↓
Stage 6: RTL synthesis, place-and-route, and GDS with ORFS + ASAP7
```

---

## System block diagram

```text
                   ┌────────────────────────────┐
                   │      Pocket Recorder        │
                   └────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│ Raspberry Pi 5 / Pi 4                                           │
│                                                                 │
│  ┌──────────────┐   ┌───────────────┐   ┌────────────────────┐ │
│  │ Mic Capture  │ → │ Audio Frontend │ → │ Local AI Pipeline  │ │
│  │ ALSA/Pipewire│   │ MFCC/VAD       │   │ whisper.cpp/TFLite │ │
│  └──────────────┘   └───────────────┘   └────────────────────┘ │
│             │                  │                    │           │
│             │                  ▼                    ▼           │
│             │        ┌────────────────┐    ┌─────────────────┐ │
│             └──────→ │ ROS 2 Nodes     │ →  │ Local UI/Logger │ │
│                      └────────────────┘    └─────────────────┘ │
│                               │                                 │
│                               ▼                                 │
│                    ┌────────────────────┐                       │
│                    │ FPGA / Verilator    │                       │
│                    │ PicoRV32 + AI Accel │                       │
│                    └────────────────────┘                       │
└─────────────────────────────────────────────────────────────────┘
```

---

## Final target hardware/software architecture

```text
                         Raspberry Pi Host
┌──────────────────────────────────────────────────────────────────┐
│                                                                  │
│  ROS 2 / Python Controller                                       │
│  ┌───────────────┐   ┌──────────────────┐   ┌────────────────┐ │
│  │ Audio Node    │ → │ Feature Node      │ → │ FPGA Driver    │ │
│  │ WAV/Mic input │   │ MFCC/log-mel/VAD  │   │ UART/SPI/USB   │ │
│  └───────────────┘   └──────────────────┘   └────────────────┘ │
│            │                    │                    │           │
│            ▼                    ▼                    ▼           │
│  ┌────────────────┐  ┌──────────────────┐   ┌────────────────┐ │
│  │ whisper.cpp     │  │ TFLite Runtime   │   │ RL Optimizer   │ │
│  │ local ASR       │  │ software baseline│   │ Claude-assisted│ │
│  └────────────────┘  └──────────────────┘   └────────────────┘ │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
                              │
                              │ UART/SPI/USB or Verilator DPI/socket
                              ▼
                 Simulated or FPGA RISC-V TinyML SoC
┌──────────────────────────────────────────────────────────────────┐
│                                                                  │
│  ┌──────────────┐       ┌───────────────┐       ┌────────────┐  │
│  │ PicoRV32     │ ←──→  │ SRAM / BRAM   │ ←──→  │ UART/SPI   │  │
│  │ RV32IMC core │       │ code + data   │       │ interface  │  │
│  └──────────────┘       └───────────────┘       └────────────┘  │
│          │                         │                             │
│          ▼                         ▼                             │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │ TinyML Accelerator                                         │  │
│  │ - int8 MAC array                                           │  │
│  │ - depthwise/pointwise conv option                          │  │
│  │ - activation / quantization support                        │  │
│  │ - cycle counter and performance registers                  │  │
│  └────────────────────────────────────────────────────────────┘  │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

---

# Stage 1 — Raspberry Pi software-only voice AI prototype

## Objective

Prove the pocket voice AI concept before building custom hardware.

## Hardware

- Raspberry Pi 5 preferred; Raspberry Pi 4 is acceptable.
- USB microphone or I2S MEMS microphone board.
- Optional small display or web dashboard.
- Optional battery pack if you want a portable prototype.

## Software stack

- Ubuntu Server or Raspberry Pi OS 64-bit.
- Python 3.
- ROS 2 Humble or Jazzy, depending on OS support.
- `whisper.cpp` for local offline speech-to-text.
- TFLite Runtime for lightweight local inference.
- Optional TinyEngine build environment for generated C inference code.
- SQLite or flat files for local transcript storage.

## Pipeline

```text
Mic input
  ↓
Audio chunks, e.g. 16 kHz mono PCM
  ↓
VAD / keyword spotting / audio event classifier
  ↓
If speech detected: run whisper.cpp locally
  ↓
Transcript
  ↓
Optional local summarizer or structured note extractor
  ↓
Store transcript, timestamps, keywords, and metadata
```

## First working demo

Create a Raspberry Pi app that can:

1. Record short audio clips.
2. Run local transcription with `whisper.cpp`.
3. Run a tiny classifier using TFLite Runtime, such as:
   - voice activity detection,
   - wake word,
   - speech/no-speech,
   - noise class,
   - simple command classifier.
4. Log latency, CPU usage, and memory use.

## Metrics to collect

| Metric | Why it matters |
|---|---|
| Audio chunk latency | Determines responsiveness |
| Transcription latency | Shows local ASR feasibility |
| TinyML inference latency | Baseline for accelerator speedup |
| CPU utilization | Shows power/performance pressure |
| Memory footprint | Helps size future embedded design |
| Battery/runtime estimate | Relevant to pocket recorder use case |

## Recommended Stage 1 milestone

> Raspberry Pi records audio, detects speech locally, transcribes locally with whisper.cpp, and logs performance data.

---

# Stage 2 — TinyML model/runtime path

## Objective

Create a small model that can later run on PicoRV32 and be accelerated in hardware.

## Recommended TinyML tasks

Do **not** try to run full Whisper on PicoRV32. Use PicoRV32 for always-on low-power pre-filtering.

Best workloads:

1. Voice activity detection.
2. Keyword spotting.
3. Simple command recognition: `start`, `stop`, `save`, `delete`, `marker`.
4. Environmental audio classification: speech, silence, music, noise.
5. Tiny speaker verification or speaker-change trigger.

## Suggested model

Start with one of these:

```text
Option A: Tiny 1D CNN
Input: 1 second of audio features
Features: MFCC or log-mel bins
Output: speech / no-speech or command class

Option B: DS-CNN keyword spotter
Input: MFCC feature matrix
Output: command class

Option C: Tiny MLP
Input: compact handcrafted features
Output: speech / no-speech / noise
```

## TinyEngine usage

TinyEngine is a good fit for this stage because it is designed for memory-constrained microcontroller inference. Use it to generate or adapt C inference code, then compile that C code for RV32.

Expected adaptations for PicoRV32/RV32:

- Remove or replace ARM-specific intrinsics.
- Use plain C kernels first.
- Add RV32 custom accelerator calls later.
- Fix static memory layout.
- Avoid dynamic allocation.
- Use int8 quantized weights and activations.

## TinyML software layers

```text
Model training, probably on laptop or Pi
  ↓
Quantization to int8
  ↓
TinyEngine or TFLite Micro style C code
  ↓
Compile for RV32 bare-metal firmware
  ↓
Run first in emulator/simulator
  ↓
Replace selected kernels with accelerator calls
```

## Stage 2 milestone

> A quantized TinyML model runs on the Raspberry Pi as a software reference and has a C inference path suitable for RV32 firmware.

---

# Stage 3 — PicoRV32 + AI accelerator behavioral simulation

## Objective

Build a behavioral simulation of the hardware stack before committing to detailed RTL.

## Baseline SoC

```text
PicoRV32
  + instruction memory
  + data memory
  + UART or simulation console
  + cycle counter
  + memory-mapped accelerator stub
```

## Accelerator v1: behavioral model

Start with a simple memory-mapped accelerator:

```text
Registers:
0x00 CONTROL      start/done/reset
0x04 STATUS       busy/done/error
0x08 INPUT_PTR    input feature address
0x0C WEIGHT_PTR   weight address
0x10 OUTPUT_PTR   output address
0x14 LENGTH       vector length
0x18 CONFIG       lanes, mode, quantization options
0x1C CYCLES       accelerator cycle count
```

## Accelerator operation modes

Start with one or two modes only:

```text
Mode 0: int8 dot product
Mode 1: int8 matrix-vector multiply
Mode 2: optional 1D convolution
Mode 3: optional depthwise convolution
```

## Behavioral simulation strategy

First, implement the accelerator as high-level Verilog/SystemVerilog behavior or even a C++ model connected through the Verilator testbench. The goal is to test architecture and firmware flow before optimizing gates.

```text
RV32 firmware calls accelerator
  ↓
Memory-mapped register writes
  ↓
Behavioral accelerator reads simulated memory
  ↓
Computes result
  ↓
Writes output
  ↓
Firmware validates result against golden output
```

## Stage 3 milestone

> PicoRV32 firmware calls a behavioral TinyML accelerator in Verilator and produces correct inference outputs for small test vectors.

---

# Stage 4 — Combined software/hardware Verilator stack

## Objective

Run the whole system together: Raspberry Pi host software + Verilated PicoRV32 SoC + TinyML firmware + accelerator model.

## Co-simulation architecture

```text
Raspberry Pi Python/ROS 2 host
  ↓ socket/file/pipe/UART emulation
Verilator C++ testbench
  ↓
PicoRV32 SoC
  ↓
TinyML firmware
  ↓
Behavioral or RTL accelerator
```

## Recommended interfaces

Use the simplest first:

1. File-based test vectors.
2. C++ testbench loads memory images.
3. Simulated UART output.
4. Socket bridge for interactive ROS 2 integration.
5. SPI/UART bridge later for real FPGA.

## Firmware responsibilities

The RV32 firmware should:

- initialize memory,
- load or receive feature tensors,
- call TinyEngine/TinyML kernels,
- offload selected kernels to accelerator,
- verify outputs,
- expose cycle counts,
- return classification result.

## Host responsibilities

The Raspberry Pi host should:

- capture audio,
- compute features,
- send feature vectors to simulation,
- collect prediction and cycle count,
- compare against TFLite Runtime reference,
- log results for optimizer.

## Stage 4 milestone

> Raspberry Pi sends audio-derived features into a Verilated PicoRV32 TinyML SoC and receives classification results plus cycle counts.

---

# Stage 5 — RL and Claude-assisted optimization loops

## Important note

Use RL/agentic loops to optimize parameters and generate candidate code/configurations, but keep the measurement loop deterministic and reproducible. The optimizer should not guess performance; it should run synthesis/simulation/profiling and score real results.

## Optimization hierarchy

```text
Level 0: Software parameters
Level 1: TinyML model parameters
Level 2: Firmware/kernel mapping
Level 3: Accelerator microarchitecture
Level 4: SoC memory/interconnect parameters
Level 5: Physical design constraints
```

## RL/agentic loop diagram

```text
┌─────────────────────────────────────────────────────────────┐
│ RL / Agentic Optimizer                                      │
│                                                             │
│  State:                                                     │
│  - model size, accuracy, latency                            │
│  - accelerator config                                       │
│  - area/timing/power estimates                              │
│  - previous experiment results                              │
│                                                             │
│  Action:                                                    │
│  - choose MAC lanes                                         │
│  - choose buffer sizes                                      │
│  - choose tiling strategy                                   │
│  - choose quantization                                      │
│  - choose kernel offload mapping                            │
│  - choose ORFS constraints                                  │
│                                                             │
│  Reward:                                                    │
│  + accuracy                                                 │
│  + speedup                                                  │
│  - area                                                     │
│  - power proxy                                              │
│  - timing violations                                        │
│  - build/runtime cost                                       │
└─────────────────────────────────────────────────────────────┘
                      │
                      ▼
          Generate config / code / constraints
                      │
                      ▼
           Run simulation / synthesis / PnR
                      │
                      ▼
             Parse metrics and update policy
```

## Where Claude fits

Claude or another LLM agent can be used as an assistant inside the loop, not as the sole optimizer.

Suggested roles:

| Stage | Claude role | Deterministic checker |
|---|---|---|
| Stage 1 | Generate ROS 2/Python code | Unit tests, audio tests |
| Stage 2 | Suggest model/kernel simplifications | Accuracy and latency tests |
| Stage 3 | Generate accelerator RTL variants | Verilator tests |
| Stage 4 | Debug firmware/harness issues | Golden vector comparison |
| Stage 5 | Propose new design-space candidates | Measured reward function |
| Stage 6 | Suggest constraints/floorplan changes | ORFS timing/area reports |

## RL algorithms to start with

Begin simple. You probably do not need deep RL at first.

Recommended order:

1. Grid search for basic sanity.
2. Random search.
3. Bayesian optimization.
4. Evolutionary search.
5. Reinforcement learning after the environment is stable.

## Candidate design parameters

```yaml
accelerator:
  mac_lanes: [1, 2, 4, 8, 16]
  accumulator_width: [16, 24, 32]
  input_buffer_bytes: [256, 512, 1024, 2048, 4096]
  weight_buffer_bytes: [256, 512, 1024, 2048, 4096]
  dataflow: [output_stationary, weight_stationary]
  kernel_modes: [dot, matvec, conv1d, depthwise]

firmware:
  offload_threshold: [16, 32, 64, 128]
  loop_unroll: [1, 2, 4, 8]
  quantization: [int8, int4_experimental]

soc:
  cpu_core: [picorv32_rv32i, picorv32_rv32im]
  bus_width: [32]
  memory_size_kb: [16, 32, 64, 128]

physical_design:
  clock_period_ns: [5, 10, 20]
  core_utilization: [30, 40, 50, 60]
  placement_density: [0.45, 0.55, 0.65]
```

## Example reward function

```text
reward =
  2.0 * normalized_accuracy
+ 1.5 * normalized_speedup
- 1.0 * normalized_area
- 1.0 * normalized_power_proxy
- 3.0 * timing_violation_penalty
- 0.5 * simulation_runtime_penalty
```

For early behavioral simulation, use cycle count and resource estimates. For ORFS, use actual timing, area, and power reports if available.

## Stage 5 milestone

> Automated loop generates multiple accelerator configurations, runs Verilator and/or ORFS, ranks designs, and produces a Pareto frontier of latency versus area.

---

# Stage 6 — RTL-to-GDS with OpenROAD-flow-scripts and ASAP7

## Objective

Take the final RTL candidate through synthesis, floorplanning, placement, clock tree synthesis, routing, and GDS generation.

## Suggested RTL scope for first GDS

Do not push the full recorder stack through GDS. Use a clean digital block:

```text
Option A: TinyML accelerator alone
Option B: PicoRV32 + accelerator + small SRAM/register-file model
Option C: PicoRV32 + accelerator + simple memory-mapped bus, no large SRAM macro
```

For the first ASAP7 flow, Option A or B is safer.

## OpenROAD-flow-scripts target

Use ORFS for:

```text
RTL
  ↓
Yosys synthesis
  ↓
OpenROAD floorplan/place/CTS/route
  ↓
Timing/area/power reports
  ↓
GDS
```

## ASAP7 caveats

ASAP7 is useful for academic/nanoscale exploration, but it can be more fragile than mature open PDK flows such as Sky130 or GF180. Keep an alternate path with Sky130/GF180 for robustness.

Recommended approach:

1. Bring up the design in a simple ORFS platform first, such as Nangate45 if available.
2. Then try ASAP7.
3. Keep the accelerator block small and synchronous.
4. Avoid complex generated memories until the basic flow is stable.
5. Treat SRAM macros separately if the platform support is incomplete.

## Physical design metrics

Collect:

- cell area,
- utilization,
- worst negative slack,
- total negative slack,
- routed wirelength,
- congestion reports,
- clock frequency estimate,
- power estimate if available.

## Stage 6 milestone

> Final accelerator RTL produces an ORFS ASAP7 GDS candidate with area/timing reports and a documented comparison against simulation-level metrics.

---

# Repository structure

```text
pocket-ai-riscv-recorder/
├── README.md
├── docs/
│   ├── architecture.md
│   ├── experiments.md
│   ├── rtl_to_gds.md
│   └── ros2_pipeline.md
├── sw/
│   ├── rpi_voice_app/
│   │   ├── audio_capture.py
│   │   ├── vad_tflite.py
│   │   ├── whisper_local.py
│   │   └── logger.py
│   ├── ros2_ws/
│   │   └── src/
│   └── tinyml_reference/
│       ├── train/
│       ├── export_tflite/
│       └── test_vectors/
├── firmware/
│   ├── picorv32_baremetal/
│   ├── tinyengine_port/
│   ├── linker.ld
│   └── startup.S
├── rtl/
│   ├── soc/
│   ├── picorv32/
│   ├── accel/
│   │   ├── int8_mac_array.v
│   │   ├── matvec_accel.v
│   │   └── accel_regs.v
│   └── tb/
├── sim/
│   ├── verilator/
│   ├── cpp_harness/
│   ├── golden_vectors/
│   └── run_sim.py
├── optimizer/
│   ├── env.py
│   ├── search_space.yaml
│   ├── reward.py
│   ├── agents/
│   └── results/
├── physical/
│   ├── orfs/
│   ├── asap7/
│   ├── constraints/
│   └── reports/
└── scripts/
    ├── setup_rpi.sh
    ├── build_verilator.sh
    ├── run_experiment.sh
    └── parse_reports.py
```

---

# Development milestones

## Milestone 0 — Setup

Deliverables:

- Git repository.
- Raspberry Pi development environment.
- Verilator build environment.
- RV32 cross compiler.
- Basic CI or reproducible scripts.

Success criteria:

- Can build a hello-world PicoRV32 simulation.
- Can run whisper.cpp on a WAV file.
- Can run a TFLite model on the Pi.

---

## Milestone 1 — Pi-only pocket recorder

Deliverables:

- Local audio recording.
- Local speech-to-text using whisper.cpp.
- TinyML classifier using TFLite Runtime.
- Latency logger.

Success criteria:

- Records audio and generates local transcript.
- Classifier runs locally with measured latency.

---

## Milestone 2 — TinyML C inference path

Deliverables:

- Quantized model.
- Golden test vectors.
- TinyEngine or C inference code.
- RV32-compatible build.

Success criteria:

- C inference matches Python/TFLite reference within quantization tolerance.

---

## Milestone 3 — PicoRV32 baseline simulation

Deliverables:

- PicoRV32 SoC in Verilator.
- Bare-metal firmware.
- Cycle counter.
- UART/log output.

Success criteria:

- TinyML inference runs on simulated PicoRV32 without accelerator.
- Cycle count is reported.

---

## Milestone 4 — Behavioral accelerator

Deliverables:

- Memory-mapped accelerator model.
- Firmware driver.
- Golden vector tests.

Success criteria:

- Accelerator produces correct dot/matvec outputs.
- Speedup is visible in simulation cycle counts.

---

## Milestone 5 — RTL accelerator

Deliverables:

- Synthesizable RTL for accelerator.
- Verilator testbench.
- Firmware integration.

Success criteria:

- RTL accelerator matches behavioral model.
- TinyML kernel offload works.

---

## Milestone 6 — RL/agentic optimizer

Deliverables:

- Search-space YAML.
- Experiment runner.
- Result parser.
- Pareto plots.

Success criteria:

- Optimizer evaluates at least 20 design candidates.
- Produces best design under chosen area/latency constraints.

---

## Milestone 7 — ORFS/ASAP7 flow

Deliverables:

- ORFS design configuration.
- Synthesis reports.
- Place-and-route reports.
- GDS output.

Success criteria:

- Accelerator or small SoC block completes RTL-to-GDS.
- Final report compares area/timing against architecture choices.

---

# Toolchain map

| Function | Recommended tool |
|---|---|
| Local ASR | whisper.cpp |
| TinyML software baseline | TFLite Runtime |
| Microcontroller-style inference | TinyEngine or TFLite Micro |
| RISC-V CPU | PicoRV32 |
| RTL simulation | Verilator |
| Waveform debug | GTKWave |
| RTL synthesis | Yosys through ORFS |
| Place and route | OpenROAD through ORFS |
| GDS flow | OpenROAD-flow-scripts |
| Process target | ASAP7, with fallback to Sky130/GF180/Nangate45 |
| Automation | Python, Make, YAML configs |
| Optimization | random search, Bayesian optimization, evolutionary search, RL |
| LLM assistance | Claude for code generation/debugging/design suggestions |

---

# Suggested experiment table

| Experiment | Purpose | Output |
|---|---|---|
| Pi whisper.cpp baseline | Check local transcription feasibility | ASR latency, CPU use |
| Pi TFLite TinyML baseline | Establish software TinyML speed | Inference latency |
| PicoRV32 TinyML baseline | Establish RV32 bottleneck | Cycle count |
| Behavioral accelerator | Test architecture quickly | Correctness, idealized speedup |
| RTL accelerator | Validate synthesizable design | Cycle count, waveform, resource estimate |
| Verilator design search | Optimize microarchitecture | Pareto frontier |
| ORFS synthesis sweep | Estimate area/timing | Area, WNS, frequency |
| ORFS PnR sweep | Physical design feasibility | GDS, routed timing |

---

# Initial hardware accelerator recommendation

Start with an `int8_matvec_accel`, not a full CNN accelerator.

Why:

- Easier to verify.
- Useful for dense layers and 1x1 convolutions.
- Can be extended to convolution later.
- Small enough for ASAP7 flow.
- Easy to parameterize for optimization.

## Accelerator v1 datapath

```text
input SRAM/register window
        │
        ▼
┌──────────────────┐
│ int8 input lanes  │
└──────────────────┘
        │
        ▼
┌──────────────────┐
│ int8 weight lanes │
└──────────────────┘
        │
        ▼
┌─────────────────────────────┐
│ parallel multiply-accumulate │
│ lanes = 1/2/4/8/16           │
└─────────────────────────────┘
        │
        ▼
┌──────────────────┐
│ int32 accumulator │
└──────────────────┘
        │
        ▼
┌─────────────────────────────┐
│ requantize / clamp / output  │
└─────────────────────────────┘
```

---

# Risk register

| Risk | Mitigation |
|---|---|
| Whisper is too slow on Pi | Use tiny/base models, chunked audio, or only transcribe after VAD |
| TinyEngine has ARM-specific assumptions | Start with plain C kernels, port incrementally |
| PicoRV32 is too slow for full model | Offload only hot kernels; keep model tiny |
| Verilator co-sim gets complex | Start with file-based test vectors before live ROS 2 socket bridge |
| RL loop wastes time | Start with random/Bayesian search before deep RL |
| ASAP7 flow is fragile | First validate with smaller block and fallback platform |
| SRAM macros complicate GDS | Use register-file/small inferred memory first; macro integration later |
| LLM-generated RTL has bugs | Require lint, Verilator tests, golden vectors, and formal checks where possible |

---

# Practical first 30-day plan

## Week 1

- Set up Raspberry Pi.
- Build whisper.cpp.
- Run local WAV transcription.
- Install TFLite Runtime.
- Run a tiny audio classifier.

## Week 2

- Build ROS 2 or simple Python pipeline.
- Implement audio chunking and logging.
- Train or select tiny VAD/keyword model.
- Export int8 model.

## Week 3

- Bring up PicoRV32 in Verilator.
- Compile RV32 bare-metal hello world.
- Add memory map, UART, cycle counter.
- Generate golden test vectors.

## Week 4

- Implement behavioral int8 dot/matvec accelerator.
- Add firmware driver.
- Compare PicoRV32 baseline versus accelerator.
- Start simple design-space search over MAC lanes and buffer sizes.

---

# Final demonstration script

```text
1. User speaks into Raspberry Pi microphone.
2. Raspberry Pi performs audio frontend and VAD.
3. TinyML feature vector is sent to Verilated or FPGA PicoRV32 SoC.
4. PicoRV32 runs TinyEngine-style firmware.
5. Firmware offloads matrix/vector kernel to custom AI accelerator.
6. Accelerator returns classification result and cycle count.
7. Raspberry Pi runs whisper.cpp locally if speech is detected.
8. Optimizer compares hardware candidate results.
9. Best accelerator design is pushed through ORFS ASAP7.
10. Final report shows software baseline, RV32 baseline, accelerated simulation, and GDS metrics.
```

---

# Source notes

Useful upstream projects and references:

- PicoRV32: https://github.com/YosysHQ/picorv32
- Verilator: https://github.com/verilator/verilator
- whisper.cpp: https://github.com/ggml-org/whisper.cpp
- TinyEngine: https://github.com/mit-han-lab/tinyengine
- OpenROAD-flow-scripts documentation: https://openroad-flow-scripts.readthedocs.io/
- OpenROAD-flow-scripts GitHub: https://github.com/The-OpenROAD-Project/OpenROAD-flow-scripts

---

# Recommended MVP definition

The most realistic minimum viable project is:

> Raspberry Pi runs local audio capture and whisper.cpp. A tiny VAD or keyword model is compiled into PicoRV32 firmware. Verilator simulates PicoRV32 plus a memory-mapped int8 matrix-vector accelerator. A Python optimizer sweeps accelerator parameters and chooses the best latency/area candidate. The final accelerator RTL is pushed through OpenROAD-flow-scripts targeting ASAP7 or a fallback open platform.

This MVP is achievable, technically coherent, and strong enough to demonstrate AI hardware, TinyML, RISC-V, robotics/edge software, simulation, and physical design skills in one integrated project.
