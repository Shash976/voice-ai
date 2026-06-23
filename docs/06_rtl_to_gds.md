# 06 — RTL to GDS (Stage 6)

Stage 6 turns the accelerator from a **C++ behavioral model** (Stage 4) into
**synthesizable Verilog RTL**, then pushes it through OpenROAD-flow-scripts to
get a real chip layout (GDS) plus area / timing / power numbers.

> **Status: working.** A full nangate45 GDS of `tinymac_accel` has been produced
> via the classic ORFS `make` flow, and a first **asap7 GDS** as well (1.0 ns
> constraint: 1433 µm², Fmax 509 MHz). RTL is bit-exact-verified; synthesis,
> place, route, and GDS all run, and the Stage-5 optimizer drives the flow
> directly. Remaining: realistic-clock re-sweep and requantize pipelining.
>
> Synthesis/lint run anywhere with Yosys; the full P&R→GDS flow runs where a real
> OpenROAD lives (the company VM at `/opt/OpenROAD-flow-scripts`).

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

### Accumulator saturation — behavioral model now matches the RTL

The RTL saturates the accumulator **per chunk** (after each `LANES`-wide
adder-tree sum) — which is what parallel hardware actually does. The behavioral
model (`sim_main.cpp`) originally saturated per MAC; it has been **changed to the
RTL's per-chunk order** so the two agree exactly (verified against the RTL FSM
and the testbench golden, including the partial last chunk when `K` isn't a
multiple of `LANES`). At `ACC_W ≥ 24` no TinyVAD layer ever overflows, so the
policies were already identical there; at `ACC_W = 16` the difference is real and
makes accuracy **lane-count-dependent** (measured 47/64 at 1 lane up to 58/64 at
32 lanes — wider chunks overflow less often mid-channel).

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

**Backported to the behavioral sim.** `sim_main.cpp` now models this overhead
(`ACCEL_CH_OVERHEAD = 2`): latency = `n_outputs × (ceil(K/LANES) + 2)` instead of
`ceil(M·K/LANES)`. The sim is rebuilt and the measured constants are pinned in
`optimizer/constants.py` (8 lanes: 61,400 cycles/inference; 16 lanes: 46,670 —
see the full per-lane table there).

---

## Getting RTL to synthesize: the strict-Yosys signedness gotcha

Modern Yosys (0.64, as shipped in the VM's ORFS) is far stricter about
signed/unsigned mixing than older builds, and asserts hard
(`genrtlil.cc:2214: arg->is_signed == sig.as_wire()->is_signed`). Three patterns
in the RTL had to be cleaned to pass it — all behavior-preserving:

1. **No `$signed()` on an unsigned whole wire.** `$signed(out_zp_reg)` where
   `out_zp_reg` is unsigned trips the assert. Fix: declare the reg `signed` and
   connect it directly (no cast). This was the final blocker.
2. **No signed `integer` parameters in unsigned expressions.** `parameter integer
   LANES` is *signed*; `{16'd0,k_base} + LANES` mixes it with an unsigned signal.
   Fix: a `localparam [31:0] LANES_U = LANES` unsigned copy.
3. **No mixed-signedness `?:` branches.** Every ternary arm must share signedness.

Lesson: keep every operand's signedness explicit and consistent. The Verilator
lint on the laptop (`yosys 0.9`) did **not** catch these — only the VM's 0.64 did.
The unit testbench confirmed each fix left the math bit-identical.

## The physical flow ([`../physical/orfs/make/`](../physical/orfs/make))

Two routes were tried:

- **bazel-orfs** — the modern Bazel driver. **Abandoned**: its gallery workspace
  fetches a Python stack from PyPI and the networks available timed out, aborting
  before synthesis. Its files (`BUILD.bazel`, `sync.sh`, a root `constraints.sdc`)
  have been removed from the repo; this note is the record.
- **Classic ORFS `make` flow** (`physical/orfs/make/`) — the working path. The
  company VM has a full OpenROAD at `/opt/OpenROAD-flow-scripts`. A design is just
  three files (`config.mk`, `constraint.sdc`, the RTL); `run.sh` stages them and
  invokes the ORFS Makefile through to GDS. **This is what produced the layout.**

> **asap7 unit gotcha (now handled in code):** asap7's liberty/SDC time unit is
> **picoseconds**, not nanoseconds — an asap7 SDC saying `5.0` constrains the
> design to 5 *ps*. `physical_runner.py` owns the conversion in both directions
> (`PLATFORM_TIME_UNIT`): optimizer clock values are always ns, SDC writes are
> scaled per platform, and parsed report times are normalized back to ns before
> storage. An earlier batch of asap7 results predating this fix was invalid and
> lives quarantined in `optimizer/results_physical_INVALID_psbug.jsonl`.

```bash
# one config, full RTL→GDS:
physical/orfs/make/run.sh                 # nangate45
physical/orfs/make/run.sh nangate45 gui_final   # + open the OpenROAD GUI

# many configs compared (grid search):
physical/orfs/make/sweep.sh               # sweeps LANES={1,2,4,8,16}
```

`sweep.sh` overrides the RTL parameters per run via ORFS `VERILOG_TOP_PARAMS`
(`chparam -set LANES <n>`), gives each config its own `FLOW_VARIANT` (separate
results + GDS), and collects area / WNS / power / Fmax into `sweep_results.csv`.

There is also a fast, offline **synthesis-only** area pass that needs just Yosys +
a liberty (no OpenROAD): `physical/orfs/synth_area.sh {sky130hd,nangate45}`.

---

## Results

### Synthesis only (Yosys, nangate45) — the area-vs-lanes trade-off

`chparam`-overriding `LANES`, flattened, mapped to NangateOpenCell typical:

| LANES | cells | cell area (µm²) | rel. area | throughput |
|-------|-------|-----------------|-----------|------------|
| 1  | 10,094 | 12,331 | 1.00× | 1× |
| 2  | 10,799 | 13,120 | 1.06× | ~2× |
| 4  | 12,098 | 14,589 | 1.18× | ~4× |
| 8  | 14,591 | 17,354 | 1.41× | ~8× |
| 16 | 19,571 | 22,949 | 1.86× | ~16× |

**Key insight:** 1→16 lanes is 16× the multipliers but only **1.86× the area**,
because the 64-bit requantize multiply + FSM + counters are fixed overhead every
config pays. More lanes is a *bargain* — you buy throughput cheaply in area.

(sky130hd, `LANES=4`: 10,179 cells / **72,897 µm²** — 130 nm cells are ~5× bigger
than 45 nm, as expected.)

### Full RTL→GDS (classic ORFS make, nangate45, LANES=4 ACC_W=24)

First complete layout — `6_final.gds`, viewable in KLayout / the OpenROAD GUI:

| Metric | Value | Note |
|--------|-------|------|
| Design area | **19,738 µm²** | post-place, 48% utilization |
| Flip-flops | 230 | matches RTL state (`finish` report, sequential cell count) |
| **Fmax** | **~269 MHz** (period_min 3.72 ns) | the real achievable speed |
| WNS @ 2.0 ns target | **−1.72 ns (VIOLATED)** | 2.0 ns was too aggressive |
| Setup violations @ 2 ns | 40 | all on the requantize path |
| Power | ~1.0 W | nangate45 worst-case switching — a *pessimistic upper bound* |

**The critical path is the requantize Q31 multiply**, confirmed by the timing
report: `i_qmult → full-adder chain → u_rq.q31 → u_rq.shifted → u_rq.biased →
o_out_data`. It is **independent of `LANES`** (the MAC adder tree is shallower
than the 32×32 multiply), so Fmax is roughly constant across all lane counts
while area grows with lanes — a clean latency-vs-area Pareto.

> **Architectural follow-up the report points to:** pipeline the requantize
> (split the 64-bit multiply across 2 cycles). Requantize runs once per output
> channel, not per MAC, so this ~doubles Fmax for almost no throughput cost.

### Closing the loop — the optimizer drives the flow (Stage 5 ↔ 6)

The ORFS make flow described here is also driven *programmatically* by an
optimizer that proposes a chip config, lets the tools build it, feeds the real
measured metrics (area / Fmax / power / timing — never the requested clock) back
as a reward, and picks the next config. Each non-cached trial is a full
place-and-route (minutes), with content-hashed variant names so RTL edits
invalidate stale builds, and `PARSE_FAIL` / `TIMEOUT` surfaced as penalties.

That optimizer — the multi-fidelity funnel that builds on exactly this plumbing —
now lives in the standalone **[eda-rl](https://github.com/Shash976/eda-rl)** repo.
See its README for the input → report → best-GDS pipeline; point it at this
checkout with `EDA_RL_DESIGN_ROOT=$(pwd)`.

> This code was independently reviewed for hallucinated APIs and wrong ORFS
> assumptions before landing; the report parsers were checked against real VM
> report strings, and the loop validated end-to-end in mock mode.

### First asap7 result (the project's 7 nm-class target)

With the unit conversion in place, the first asap7 full flow runs clean —
LANES=4, ACC_W=24, 1.0 ns constraint:

| Metric | asap7 @ 1.0 ns | nangate45 @ 2.0 ns (for scale) |
|--------|----------------|--------------------------------|
| Design area | **1433 µm²** | 19,738 µm² |
| Fmax | **509 MHz** (period_min 1.96 ns) | ~269 MHz |
| WNS | −0.96 ns (constraint aggressive, as intended) | −1.72 ns |
| GDS | `6_final.gds` produced, ~9 min | produced |

### Still to do

- ~~Backport per-channel cycle overhead to `sim_main.cpp`~~ — **done**, sim rebuilt,
  constants pinned in `optimizer/constants.py`.
- ~~asap7 retarget~~ — **first GDS produced** (above); a 12-config sweep at
  0.8–1.2 ns is the natural next batch and doubles as the surrogate's transfer
  test set.
- Re-sweep nangate45 at a **realistic clock** (≈4 ns) so configs meet timing,
  and/or sweep the clock to map Fmax per config.
- Pipeline the requantize multiply to lift Fmax (~2×) — the critical path the
  timing report identified; no synthesis recipe moves it (measured), only this
  RTL change does.
