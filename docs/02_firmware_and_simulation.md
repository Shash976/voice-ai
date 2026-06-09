# 02 — Firmware & Simulation (Stage 3)

This is where the int8 C engine runs on an actual (simulated) RISC-V CPU, and we
measure how many clock cycles it takes. **Runs in WSL.**

---

## The big picture of this stage

We take the TinyVAD C engine, cross-compile it into RISC-V machine code, and run it
on **PicoRV32** — a real RISC-V CPU described in Verilog — inside the **Verilator**
simulator. No accelerator yet. The point is to get a *baseline*: how slow is pure
software? Answer: **~11.2 million cycles per inference** (~112 ms at 100 MHz). That
slowness is the motivation for Stage 4.

```
firmware (C) ──cross-compile──> firmware.bin (RV32 machine code)
                                      │
                                      ▼
        ┌─────────────────────────────────────────────────┐
        │ Verilator testbench  (sim_main.cpp, C++)         │
        │   • ram[256 KB]   ← holds firmware + data + stack│
        │   • UART          ← firmware printf → stdout     │
        │   • accelerator   ← Stage 4 only                 │
        │            │ memory bus signals                  │
        │   ┌────────▼──────────┐                          │
        │   │ picorv32_soc.v    │  ← Verilog, compiled by  │
        │   │  └ picorv32 core  │    Verilator into C++    │
        │   └───────────────────┘                          │
        └─────────────────────────────────────────────────┘
```

---

## Concepts (skip if you know chip design)

- **PicoRV32** — a small open-source RISC-V CPU (~2000 lines of Verilog). Implements
  **RV32IMC**: 32-bit integer base + Multiply/divide + Compressed 16-bit instructions.
- **Verilator** — compiles Verilog RTL into a C++ class. You write a C++ "testbench"
  that drives that class clock-by-clock. Result: a fast *software* simulation of the
  hardware. (Different from an FPGA, which is real reconfigurable hardware.)
- **Why simulate the CPU at all, instead of just running the C engine on x86?**
  Because we want *real clock-cycle counts* for the actual RISC-V machine code —
  fetch/decode/execute, multi-cycle multiplies, the works. x86 timing tells us
  nothing about how the eventual chip performs.

---

## The pieces

### The SoC wrapper — [`../rtl/soc/picorv32_soc.v`](../rtl/soc/picorv32_soc.v)

A thin Verilog wrapper around the PicoRV32 core. Its *only* job is to expose the
CPU's memory-bus signals (`mem_valid`, `mem_addr`, `mem_wdata`, `mem_wstrb`,
`mem_ready`, `mem_rdata`) to the outside. It contains **no RAM and no UART** — those
live in the C++ testbench. All the optional PicoRV32 interfaces (look-ahead bus,
co-processor, IRQ, trace) are tied off here.

Notable parameters set in the wrapper (and *why they matter*):
- `ENABLE_DIV = 1` — **required**, because `global_avg_pool` uses a hardware `div`
  instruction. Turn it off and the model traps.
- `COMPRESSED_ISA = 1` — note the correct name is `COMPRESSED_ISA`, *not*
  `ENABLE_COMPRESSED` (an easy mistake).
- `ENABLE_MUL`, `ENABLE_FAST_MUL`, `ENABLE_COUNTERS`, `REGS_INIT_ZERO`.

### The testbench — [`../sim/verilator/sim_main.cpp`](../sim/verilator/sim_main.cpp)

The heart of the simulation. It:
1. Loads `firmware.bin` into a `uint8_t ram[256*1024]` array.
2. Drives the CPU clock in a loop (negedge presents memory data, posedge the CPU
   latches it → **1 clock per memory transaction**, zero wait states).
3. Decodes every memory access by address (the memory map below).
4. Prints firmware UART output to stdout, simulation stats to stderr.
5. Also contains the Stage-4 accelerator emulation (covered in doc 03).

### The firmware — [`../firmware/picorv32_baremetal/`](../firmware/picorv32_baremetal/)

| File | Role |
|------|------|
| [`startup.S`](../firmware/picorv32_baremetal/startup.S) | First code after reset (must be at address 0). Sets stack pointer, zeroes `.bss`, calls `main()`. |
| [`linker.ld`](../firmware/picorv32_baremetal/linker.ld) | Memory layout: code at `0x0`, stack at top of 256 KB RAM. |
| [`syscalls.h`](../firmware/picorv32_baremetal/syscalls.h) | `uart_puts()` / `uart_putu32()` etc. (write bytes to `0x10000000`), and `rdcycle()` (reads PicoRV32's hardware cycle counter). |
| [`main.c`](../firmware/picorv32_baremetal/main.c) | The benchmark: loops over 64 test vectors, times each with `rdcycle()`, prints CSV over UART. |
| `accel.h` / `accel.c` | Stage-4 accelerator driver (doc 03). |

---

## The memory map (the testbench decides what each address means)

| Address range | Purpose | How the testbench handles it |
|---------------|---------|------------------------------|
| `0x00000000–0x0003FFFF` | 256 KB RAM (code + data + stack) | read/write the `ram[]` array |
| `0x10000000` | UART TX | `putchar()` the byte → stdout |
| `0x10000004` | SIM_EXIT | set `sim_done = true`, end the loop |
| `0x20000000–0x20000FFF` | Accelerator registers (Stage 4) | call `accel_read/write()` |

The CPU resets to `0x00000000`; the stack grows down from `0x00040000`. There is no
*real* UART — writing a byte to `0x10000000` is just a testbench side-effect that
calls `putchar`.

---

## The two firmware build-flag gotchas (you WILL hit these)

The `riscv64-linux-gnu` toolchain defaults to **PIE** (Position-Independent
Executable) mode even with `-nostdlib`. On bare metal with no dynamic linker, this
breaks startup. The Makefile fixes it:

- **`-fno-pic -fno-pie`** in CFLAGS — forces direct `auipc+addi` addressing for
  linker symbols like `_stack_top` (otherwise it emits GOT-indirect loads that crash).
- **`-no-pie -Wl,--build-id=none`** in LDFLAGS — suppresses the `PT_PHDR` and
  `.note.gnu.build-id` sections that would otherwise push `.text` away from address 0
  (the CPU *must* boot from `0x0`).

If you ever see the firmware trap immediately at reset, these flags are the first
suspect.

---

## How to run (WSL)

```bash
cd sim/verilator
make check-deps   # verifies: verilator, riscv gcc, picorv32.v, generated headers
make run          # cross-compile firmware → Verilate → compile testbench → simulate
make vcd          # same, but also dump sim_out.vcd (open in GTKWave)
make clean
```

What `make run` does internally:
1. Cross-compiles the firmware to `firmware.bin` (raw RV32 machine code).
2. Runs Verilator on `picorv32_soc.v` + `picorv32.v` → a C++ model.
3. Compiles `sim_main.cpp` against that model → `sim_picorv32` binary.
4. Runs `./sim_picorv32 firmware.bin`.

> **Note:** by default the firmware already enables the Stage-4 accelerator hooks
> (see `main.c` lines 26–27). So a plain `make run` actually measures the
> *accelerated* path. To measure the **pure-software Stage-3 baseline**, comment out
> the two `tinyvad_*_hook = ...` lines in `main.c`, rebuild, and re-run.

---

## Reading the output

### stdout — CSV from the firmware (one row per vector)

```
vec,label,result,correct,logit0,logit1,cycles
0,1,1,1,-42,18,11196638
1,0,0,1,23,-9,11201027
...
---
correct=64/64 avg_cycles=...
```

| Column | Meaning |
|--------|---------|
| `vec` | Test-vector index 0–63 |
| `label` | Ground truth: 0 = silence, 1 = speech |
| `result` | TinyVAD's prediction |
| `correct` | 1 if `result == label` |
| `logit0` | Silence score (int8) |
| `logit1` | Speech score (int8) — if `> logit0`, predicted speech |
| `cycles` | Clock cycles for this one inference (`rdcycle` delta) |

### stderr — from the testbench itself

```
[sim] Loaded 12480 bytes from firmware.bin
[sim] Reset released — starting simulation
[sim] mac_lanes=8 acc_width=32
[sim] Done in <N> cycles (wall: ~X ms at 100 MHz) ...
```

### The timeout gotcha (Stage-3 specifically)

The simulator caps at **50 million cycles** (`MAX_CYCLES` in `sim_main.cpp:36`). In
*pure software* each inference is ~11.2M cycles, so a full 64-vector sweep
(~716M cycles) **blows past the cap and times out**. That's expected. The Stage-3
baseline is therefore read **per inference** — e.g. vec 0 ≈ `11,196,638` cycles —
not from a completed full run. With the accelerator on (Stage 4), each inference is
~58–66 K cycles (exact figure pending WSL rebuild with updated cycle model), so the
full sweep finishes comfortably within the cap.

---

## Why ~11.2M cycles is so slow

Every one of TinyVAD's ~242K multiply-accumulates runs as **individual RISC-V
instructions** — load input, load weight, multiply (PicoRV32's multiply is
multi-cycle), add, loop bookkeeping, repeat. No parallelism, no special hardware.
That's the baseline the accelerator must beat. It does — by ~170–191× depending on lane count (doc 03).

---

## Mental model / what to remember

- `picorv32_soc.v` is a *thin shell*; the testbench (`sim_main.cpp`) is where all the
  RAM/UART/accelerator behavior actually lives.
- 0-wait-state memory → 1 clock per memory transaction.
- `make run` measures the **accelerated** path by default; comment out the hooks in
  `main.c` for the true SW baseline.
- The 50M-cycle timeout is why the pure-SW full sweep "fails" — read per-inference.
- PIE/GOT and build-id flags are non-negotiable for bare-metal boot.

Next: [03_accelerator.md](03_accelerator.md) — adding the MAC accelerator.
