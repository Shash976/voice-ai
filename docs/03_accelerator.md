# 03 — The MAC Accelerator (Stage 4)

The accelerator is the payoff: a dedicated hardware unit that does multiply-accumulate
in parallel, turning ~11.2M cycles/inference into ~58K. **Runs in WSL** (same sim).

---

## The idea in one paragraph

A single RISC-V multiply takes 1–3 cycles and does *one* multiply. An **8-lane MAC
array** does 8 multiplies and accumulates them in **1 cycle**. For a matrix-vector
multiply with 64×32 = 2048 MACs, that's an 8× win just from parallelism — plus you
skip all the loop/pointer overhead the CPU was paying. Across the whole model that's
~182× faster (8 lanes).

The accelerator is a **memory-mapped peripheral**: the firmware programs it by
*writing to specific addresses* (its registers), then triggers it. To the CPU it
looks exactly like writing to RAM — same bus, just a different address range
(`0x20000000`).

---

## Where the accelerator actually lives

The Verilator sim still uses a **C++ behavioral model** inside the testbench
([`../sim/verilator/sim_main.cpp`](../sim/verilator/sim_main.cpp),
`accel_execute()` at line 118): when the testbench sees a memory write into
`0x20000000–0x20000FFF`, it runs the MAC math directly in C++ and reports a
*simulated* cycle latency. This is how Stages 3–5 measure cycle counts.

**Stage 6 has since written the real synthesizable RTL** —
[`../rtl/accel/int8_mac_array.v`](../rtl/accel/int8_mac_array.v),
[`../rtl/accel/requantize.v`](../rtl/accel/requantize.v), and
[`../rtl/accel/tinymac_accel.v`](../rtl/accel/tinymac_accel.v) — and verified it
bit-exact against the behavioral model (45/45, see `rtl/tb/`). A full nangate45 GDS
has been produced. The behavioral model in `sim_main.cpp` *remains* the cycle-count
source for the optimizer; the RTL is the ground truth for area/timing/power.
See [doc 06](06_rtl_to_gds.md) for RTL details and physical results.

---

## What the accelerator computes

Two operations, covering TinyVAD's hot layers:

- **MATVEC (cmd=1)** — dense layer: `out[M] = W[M][K] · in[K] + bias[M]`, fully
  requantized to int8.
- **CONV1D (cmd=2)** — 1D convolution: slides `[out_ch, in_ch, kernel]` filters over
  a `[in_len, in_ch]` input.

Both read operands straight from the firmware's RAM (via pointer registers), compute
in int32 accumulators, apply **per-channel requantization** (the *exact same*
fixed-point math as the software path — see `accel_requantize()` at
`sim_main.cpp:81`, a mirror of `tiny_vad_infer.c`'s `requantize()`), and write int8
results back to RAM.

> At `acc_width=32` the accelerator output is **bitwise identical** to the software
> path. That identity is the correctness guarantee — same 64/64 vectors pass.

---

## The register map (base `0x20000000`)

Defined in both [`../firmware/picorv32_baremetal/accel.h`](../firmware/picorv32_baremetal/accel.h)
(firmware side) and `sim_main.cpp`'s `accel_write()`/`accel_read()` (sim side) — they
must agree.

| Offset | Register | R/W | Purpose |
|--------|----------|-----|---------|
| `0x00` | STATUS | R | 0 = idle, 1 = busy (firmware polls this) |
| `0x04` | CMD | W | write 1 = MATVEC, 2 = CONV1D — **triggers execution** |
| `0x08` | IN_PTR | W | RAM address of input tensor |
| `0x0C` | WT_PTR | W | RAM address of weights |
| `0x10` | BIAS_PTR | W | RAM address of int32 bias array |
| `0x14` | MULT_PTR | W | RAM address of per-channel Q-multipliers |
| `0x18` | RSHI_PTR | W | RAM address of per-channel right-shifts |
| `0x1C` | OUT_PTR | W | RAM address to write output |
| `0x20` | DIM_M | W | output dim (rows for MATVEC, out_ch for CONV1D) |
| `0x24` | DIM_K | W | input dim (cols for MATVEC, in_ch for CONV1D) |
| `0x28` | DIM_KERN | W | kernel size (CONV1D only) |
| `0x2C` | DIM_INLEN | W | input sequence length (CONV1D only) |
| `0x30` | DIM_STRIDE | W | conv stride (CONV1D only) |
| `0x34` | DIM_PAD | W | zero-padding (CONV1D only) |
| `0x38` | ZP_IN | W | input zero-point |
| `0x3C` | ZP_OUT | W | output zero-point |
| `0x40` | RELU | W | 1 = apply ReLU after requantize |
| `0x44` | LAST_CYC | R | cycles the last op took |

---

## How the firmware drives it

The driver is [`../firmware/picorv32_baremetal/accel.c`](../firmware/picorv32_baremetal/accel.c).
The protocol for one layer:

```c
1. while (ACCEL_STATUS != 0) {}      // wait for idle
2. ACCEL_IN_PTR  = ...;              // write all pointers + dims + zero-points
   ACCEL_WT_PTR  = ...;
   ACCEL_DIM_M   = ...;  // etc.
3. ACCEL_CMD = ACCEL_CMD_MATVEC;     // writing CMD triggers it
4. while (ACCEL_STATUS != 0) {}      // busy-wait until done
```

### The hook that makes it transparent

In [`main.c`](../firmware/picorv32_baremetal/main.c) (lines 26–27):

```c
tinyvad_conv1d_hook = accel_conv1d;
tinyvad_dense_hook  = accel_dense;
```

Recall from doc 01: `conv1d()` and `dense()` check these pointers first. Setting them
means *every* conv/dense call in `tiny_vad_infer()` routes to the accelerator —
**the inference loop itself never changes**. Comment these two lines out and you're
back to the pure-SW Stage-3 path. That's the only switch.

---

## How latency is modeled

The behavioral model doesn't cycle-step the MAC array; it computes the answer
instantly in C++ and then *pretends* it took:

```c
// Output-stationary model, calibrated to real RTL (sim_main.cpp):
chunks  = ceil(reduction / mac_lanes)        // reduction = K (MATVEC) or in_ch*kern (CONV1D)
latency = n_outputs * (chunks + ACCEL_CH_OVERHEAD)   // ACCEL_CH_OVERHEAD = 2
accel_done_at = current_cycle + latency
```

The `+2` overhead per output channel accounts for bias load + requantize cycles — the
same per-channel overhead measured in the Stage-6 RTL (FC0: 512 idealized → 576 real).
The old formula was `ceil(total_MACs / mac_lanes)` which ignored this overhead and
understated hardware latency by ~12.5%. STATUS reads as busy (1) until
`accel_done_at`. The sim is built with this model and the measured per-lane cycle
counts are pinned in `optimizer/constants.py` (`measure_real.py` regenerates them).

- `mac_lanes` is set at runtime via `--mac-lanes N` (default 8). See `sim_main.cpp:47`.
- `acc_width` is set via `--acc-width N` (16/24/32, default 32). See below.

### The accumulator-width subtlety (a real overflow model)

`accel_saturate()` (`sim_main.cpp:108`) clips the running accumulator to the
configured bit-width *after every MAC*, modeling real hardware overflow:

- **32-bit:** no-op, always correct → **64/64**.
- **24-bit:** safe for all TinyVAD layers → **64/64**.
- **16-bit:** genuinely overflows → **47/64** (73%) — hardware-accurate wrongness.

> 🐛 Gotcha verified 2026-06-01: `vec 0` alone does *not* trigger the 16-bit
> overflow. You must run the **full 64-vector suite** to catch it. Never validate
> accumulator width on a single vector.

---

## How to run (WSL)

```bash
cd sim/verilator
make run                              # default: 8 lanes, 32-bit acc (hooks already on)
./sim_picorv32 firmware.bin --mac-lanes 16 --acc-width 32   # try 16 lanes directly
./sim_picorv32 firmware.bin --mac-lanes 8  --acc-width 16   # watch accuracy drop to 47/64
```

(`make run` builds `firmware.bin` + `sim_picorv32` then runs with defaults. Once
built, call `./sim_picorv32` directly to sweep flags without rebuilding.)

### Output

stdout is the same CSV as Stage 3 but with far fewer cycles and a real summary line
(the full sweep now finishes within the 50M cap):

```
vec,label,result,correct,logit0,logit1,cycles
0,1,1,1,-42,18,61399
...
---
correct=64/64 avg_cycles=61399
```

stderr logs each accelerator invocation (`[accel] cmd=2 M=32 K=40 ...`) — one per
conv/dense layer per vector — and a final `[sim] Done in <N> cycles`.

---

## Results

Per inference, vs. the Stage-3 SW baseline (11,196,638 cycles):

| | SW only | accel, 8 lanes | accel, 16 lanes |
|---|---|---|---|
| Cycles / inference | ~11.2 M | ~61.4 K | ~46.7 K |
| Time @ 100 MHz | ~112 ms | ~0.61 ms | ~0.47 ms |
| Accuracy | 64/64 | 64/64 | 64/64 |
| **Speedup** | 1× | **~182×** | **~240×** |

Measured with the RTL-matched cycle model (`ACCEL_CH_OVERHEAD=2`: per-channel
bias-load + requantize overhead). The full per-lane table (1→32 lanes:
273,130 → 39,310 cycles) is pinned in `optimizer/constants.py`.

Notice 16 lanes is *not* 2× faster than 8: the convolution/dense work doesn't divide
evenly, and fixed per-channel overhead dominates at high lane counts. This diminishing
return is exactly what the Stage-5 optimizer quantifies — and the Stage-6 RTL
confirmed it by measuring 576 cycles (FC0) vs the 512-cycle idealized estimate.

---

## Mental model / what to remember

- The Verilator sim uses a **C++ behavioral model** for cycle counting — the
  synthesizable RTL lives in `rtl/accel/` and is verified separately (Stage 6).
- Memory-mapped peripheral: program registers → write CMD to trigger → poll STATUS.
- The hook pointers in `main.c` are the on/off switch between SW and accelerated.
- Requantize math is duplicated and must stay identical across SW/accel; `acc=32`
  gives bitwise-identical output.
- `mac_lanes` scales modeled latency; `acc_width` models real overflow (test all 64
  vectors).

Next: [04_optimizer.md](04_optimizer.md) — automatically searching the design knobs.
