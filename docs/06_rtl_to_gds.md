# 06 — RTL to GDS (Stage 6)

Stage 6 turns the accelerator from a **C++ behavioral model** (Stage 4) into
**synthesizable Verilog RTL**, then pushes it through OpenROAD-flow-scripts to
get a real chip layout (GDS) plus area / timing / power numbers. **Runs in WSL.**

---

## What we synthesize (and what we don't)

Just the **accelerator compute core** — `tinymac_accel`: the int8 MAC array, the
saturating accumulator, the requantize datapath, and the sequencer FSM. Not the
PicoRV32, not the 256 KB RAM, no SRAM macro.

Why this scope (project plan "Option A"): it is the block whose area and timing
the Stage-5 knobs (`LANES`, `ACC_W`) actually move, it is small and synchronous,
and it avoids the fragile parts (SRAM macros, a big CPU) for a first ASAP7 run.
The MMIO peripheral wrapper (the `0x20000000` register file that masters main
RAM in Stage 4) is deliberately *out* of the synthesized block — it adds no
arithmetic, which is what physical design measures.

CONV1D is also out of scope for the first GDS: in hardware it lowers to matvec
(im2col) and reuses this exact datapath. The core implements MATVEC (cmd=1).

---

## The RTL ([`../rtl/accel/`](../rtl/accel))

Three small, lint-clean Verilog-2001 modules:

| File | Role |
|------|------|
| [`int8_mac_array.v`](../rtl/accel/int8_mac_array.v) | `LANES` parallel `(in−in_zp)·wt` int8 multiplies + adder tree → int32 partial sum. Per-lane enable masks the tail chunk when `K` isn't a multiple of `LANES`. |
| [`requantize.v`](../rtl/accel/requantize.v) | Q31 multiply + signed variable shift + `out_zp` + ReLU + int8 clamp. **Bit-exact mirror** of `tiny_vad_infer.c`'s `requantize()` and `sim_main.cpp`'s clamp/zp tail. The 64-bit signed multiply here is the expected critical path. |
| [`tinymac_accel.v`](../rtl/accel/tinymac_accel.v) | Top: output-stationary matvec sequencer. Per output channel, accumulates `K` products `LANES` at a time into an `ACC_W`-bit saturating accumulator, then requantizes to int8. |

**Default parameters = the Stage-5 grid optimum: `LANES=4`, `ACC_W=24`.**

### Operand interface (no memory mastering)

The core does not master the system bus. It exposes the indices it currently
needs (`o_m`, `o_k_base`) and consumes operands provided combinationally by the
environment — the same 0-wait-state model the Stage-4 testbench uses for RAM. A
DMA/MMIO front-end that fetches from real memory is a separate, later concern.

### Accumulator saturation — a subtlety vs the behavioral model

The behavioral model (`sim_main.cpp accel_saturate`) saturates **per MAC**. The
RTL saturates the accumulator **per chunk** (after each `LANES`-wide adder-tree
sum) — which is what parallel hardware actually does. At `ACC_W ≥ 24` no TinyVAD
layer ever overflows, so both policies give identical results; at `ACC_W = 32`
there is no saturation at all. The unit test verifies bit-exactness against a
golden that uses the RTL's per-chunk order.

---

## Correctness gate ([`../rtl/tb/`](../rtl/tb))

Before any physical design, a Verilator testbench
([`tinymac_tb.cpp`](../rtl/tb/tinymac_tb.cpp)) drives the RTL with the real
TinyVAD dense-layer shapes (FC0 64→32, FC1 32→2) plus 43 randomized/edge cases,
and asserts the int8 outputs are **bit-for-bit** identical to a software golden
that reuses the exact fixed-point math.

```bash
cd rtl/tb
make                 # LANES=4 ACC_W=24 (Stage-5 optimum)
make ACC_W=32        # also bit-exact
```

Result: **45/45 cases PASS, 0 mismatches** at both `ACC_W=24` and `ACC_W=32`.

### Real cycle count vs the idealized model

The testbench also reports the RTL's real cycle count. FC0 (M=32, K=64, 4 lanes):

- Behavioral idealized: `ceil(M·K / LANES) = ceil(2048/4) = 512` cycles.
- **RTL measured: 576 cycles** — `M × (ceil(K/LANES) + 2)` = `32 × 18`.

The ~12.5% gap is the real per-channel control overhead (bias load + requantize
cycles) that the Stage-4 `ceil(MACs/lanes)` model abstracts away. This is an
honest Stage-6 finding: the idealized speedup slightly overstates hardware.

---

## The physical flow ([`../physical/orfs/`](../physical/orfs))

Stage 6 splits into two halves:

1. **Synthesis** (Yosys) → gate netlist + cell area. Needs only `yosys` + a PDK
   liberty file, both already on disk. **Done — runs offline.**
2. **Place & route → GDS** (OpenROAD) → WNS/TNS, wirelength, power, layout.
   Needs an OpenROAD binary. **Blocked on this machine** (see below).

### Synthesis — `physical/orfs/synth_area.sh`

```bash
physical/orfs/synth_area.sh sky130hd     # → reports/sky130hd_{area.rpt,synth.log,netlist.v}
physical/orfs/synth_area.sh nangate45
```

Standalone Yosys: `read_verilog → synth -flatten → dfflibmap → abc -liberty →
stat`. Maps the RTL to real standard cells and reports area. Timing-unaware
(no clock constraint), so it gives area + gate count but not WNS.

### Place & route — blocked, and why

The intended driver is **bazel-orfs** (`physical/orfs/BUILD.bazel` + `sync.sh`,
run from `~/bazel-orfs/gallery`). On this machine it does **not** complete: the
gallery's bzlmod graph pulls a large Python tooling stack (numpy/scipy/pandas/…)
from PyPI, and the network times out (`Read timed out`), aborting analysis before
synthesis even starts. There is also no standalone OpenROAD binary installed
(`tools/install` is empty; the only `openroad` in the bazel cache is a wrapper
script, not an ELF). So floorplan/place/route/GDS — and therefore WNS/TNS,
wirelength, and power — are **not reproducible here without network access to
build/fetch OpenROAD**. The Yosys area numbers below are real; the P&R column is
pending an OpenROAD install.

---

## Results

### Synthesis (real, this machine)

`tinymac_accel`, `LANES=4 ACC_W=24`, flattened, mapped to each PDK's typical
corner:

| PDK | Node | Std cells | Flip-flops | Cell area |
|-----|------|-----------|-----------|-----------|
| **sky130hd** | 130 nm | 10,179 | 231 | **72,897 µm²** |
| **nangate45** | 45 nm | 12,032 | 231 | **14,518 µm²** |

The 231 flip-flops match the RTL state (accumulator, counters, config, pipeline
regs). Area is **arithmetic-dominated** — the most common cells are `nand2`,
`xnor2`, `xor2`, `maj3` (majority/carry), i.e. the MAC multipliers and the 64-bit
Q31 requantize multiply. That confirms the architectural expectation: the
multiplier datapath, not control, sets the area, so `LANES` and the requantize
width are the real area knobs.

> Cell area is pre-placement (no routing/whitespace). Post-P&R die area is larger
> by the placement utilization factor (~40% target → ~2.5× the cell area).

### Place & route (pending OpenROAD)

| Metric | sky130hd @ 10 ns | asap7 @ 5 ns |
|--------|------------------|--------------|
| WNS / TNS (ns) | pending OpenROAD | pending OpenROAD |
| Routed wirelength | pending | pending |
| Power (mW) | pending | pending |
| GDS | pending | pending |

To unblock: install OpenROAD (or restore network so bazel-orfs can build it),
then `physical/orfs/sync.sh` + `bazel build //tinymac:tinymac_accel_final`, or
point the classic ORFS `make` flow at the binary. The design, SDC, and BUILD are
all in place and ready to run.
