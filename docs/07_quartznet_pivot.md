# Stage 7 — Pivot: from TinyVAD wake-gate to an on-chip QuartzNet 15x5 ASR accelerator

## Context

The project today builds a wake-gate: a 14.6 KB TinyVAD model on a PicoRV32 + TinyMAC
accelerator decides whether to wake a Raspberry Pi, which then runs Whisper. Stages 1–6
are done — the accelerator has a working nangate45 GDS (~19,738 µm², Fmax ≈269 MHz).

We are changing the product: **the chip should transcribe speech itself, replacing
Whisper on the Pi.** The target model is NVIDIA NeMo's QuartzNet 15x5 — a fully
convolutional CTC acoustic model, chosen because it is a pure 1D CNN (no attention, no
autoregression), streams naturally, and quantizes exceptionally well.

### Why this is a bigger change than it looks

Measured against the NeMo `quartznet_15x5.yaml` config (computed exactly: 18,846,016
params, matching the paper's stated 18.9M; the 5x5 variant likewise matches its 6.7M):

| | TinyVAD (today) | QuartzNet 15x5 |
|---|---|---|
| int8 weights | 14.6 KB | **18.85 MB** (1,290×) |
| MACs | ~0.2 M / inference | 18.85 M **per output frame** |
| Real-time compute | — | 942 MMAC/s @ 50 output frames/s |
| Accuracy | binary VAD | 4.19% WER test-clean, 10.98% test-other (greedy, no LM) |

Three findings drive the whole design:

1. **Arithmetic is cheap.** LANES=32 at 200–300 MHz is 7–10× real-time at full
   utilization — it only needs ~10–15% actual utilization to keep up. The repo's own
   sweep already shows 16× the lanes costs only 1.86× area.
2. **Memory is the entire problem.** Every weight is used *exactly once* per output
   frame — zero intra-frame reuse. Naive traffic is 942 MB/s. The current accelerator
   has literally no memory: no SRAM, no buffers, no bus mastering
   (`rtl/accel/tinymac_accel.v:14-20` scopes it out explicitly).
3. **90.6% of all MACs are plain GEMM** (76.7% pointwise + 13.9% residual 1×1); only
   9.3% is depthwise. The existing TinyMAC + im2col is already the right primitive — it
   needs to be widened and *fed*, not redesigned.

The unlock is **frame batching**: process `T_TILE` frames at once so each loaded weight
is reused `T_TILE` times, converting a bandwidth-bound problem into a compute-bound one.

### Scope decisions (confirmed)

- **One RTL, two operating points**: full 15x5 with weights streamed from external
  memory (primary, the product), plus a scaled-down variant resident entirely on-chip
  (secondary, proves the self-contained case).
- **Chip replaces Whisper** — WER matters, so 15x5 is the primary target.
- **fakeram45 macros on nangate45** for on-chip SRAM.
- **Hardware-first**: no retraining. Reuse the released NeMo checkpoint; effort goes
  into the memory hierarchy, RTL, and physical design.

---

## Target design point

Tiling sized against the available fakeram45 macros (largest is 2048×39 ≈ 9.75 KB; most
are 4 KB — so total SRAM must be built from a manageable *count* of small macros):

| T_TILE | act SRAM | wt buf | total | 4 KB macros | ext BW for real-time | latency/tile |
|---|---|---|---|---|---|---|
| 16 | 36 KB | 8 KB | 44 KB | 11 | 58.9 MB/s | 320 ms |
| **32** | **61 KB** | **8 KB** | **69 KB** | **18** | **29.4 MB/s** | **640 ms** |
| 64 | 111 KB | 8 KB | 119 KB | 30 | 14.7 MB/s | 1280 ms |
| 128 | 211 KB | 8 KB | 219 KB | 55 | 7.4 MB/s | 2560 ms |

**Build for `T_TILE=32`, `LANES=32`, `ACC_W=32`** as the default. `T_TILE` becomes a new
swept parameter trading SRAM area against external bandwidth — a genuinely new Pareto
axis, and a useful one, since the current LANES-only sweep is flat in Fmax.

---

## Architecture

### The one idea that saves most of the work

Depthwise convolution is *channel-independent*: output channel `c` is a K-tap dot product
of `in[c][t..t+K]` with `wt[c][0..K]`. Pointwise is a dot product of `in[:][t]` with
`wt[m][:]`. **Both are the same accumulate-over-K FSM the existing core already runs** —
they differ only in how `o_m`/`o_k_base` map to addresses.

So: **one MAC datapath, two address-generation modes.** `int8_mac_array.v` and the
`S_INIT_CH → S_MAC → S_REQ` FSM survive nearly intact; the new work is the address
generator, the memory system around it, and width.

Tiling depthwise over *channels* (rather than time) also avoids the halo blowup — C2 has
K=87 at dilation 2, a 173-frame receptive field, which would be ruinous if tiled by time.

### Block diagram

```
external weight memory (QSPI/DRAM — modeled, not on-chip)
        │  ~29 MB/s @ T_TILE=32
   ┌────▼──────────┐
   │ weight DMA    │──► wt_buf (2 × 4 KB, double-buffered 8-out-channel tile)
   └───────────────┘         │
                             ▼
 act_in  ─┐          ┌───────────────┐
 act_out  ├─ SRAM ──►│  MAC datapath │──► requantize (PIPELINED) ──► act_out
 act_res ─┘  (fakeram45)│ LANES × int8│
                      └───────▲───────┘
                              │
                     addr-gen: PW mode | DW mode
                              ▲
                    ┌─────────┴─────────┐
                    │ descriptor engine │◄── PicoRV32 (control only)
                    └───────────────────┘
```

### Component responsibilities

- **`rtl/accel/int8_mac_array.v`** — reused as-is, widened. Already cleanly parameterized
  and `generate`-unrolled.
- **`rtl/accel/requantize.v`** — reused, but the 64-bit Q31 multiply **must** be split
  across 2 cycles. It is already the critical path (3.72 ns, independent of LANES), and
  we now want 200–300 MHz. It runs once per output channel, not per MAC, so the
  throughput cost is <1%. `docs/06_rtl_to_gds.md:202-204` already flags this.
- **`rtl/accel/addr_gen.v`** (new) — PW/DW mode address generation into the SRAM tiles.
- **`rtl/accel/act_sram.v`, `wt_buf.v`** (new) — banked fakeram45 wrappers.
  **Blackboxed for synthesis** (LEF/LIB only, `GDS_ALLOW_EMPTY`), with a **behavioral
  model for Verilator**. This mirrors the split the repo already uses, where the 256 KB
  main memory lives in `sim/verilator/sim_main.cpp` rather than RTL.
- **`rtl/accel/wt_dma.v`** (new) — streams weights from the external interface. Needs a
  real `valid/ready` handshake; the current core has **no stall input at all**, which is
  the first thing that breaks once memory has variable latency.
- **`rtl/accel/quartznet_accel.v`** (new top) — descriptor engine + MMIO register file.
  Note the `0x20000000` register map currently exists *only* in `accel.h` and C++; there
  is no Verilog for it today.

### Firmware

`firmware/tinyengine_port/tiny_vad_infer.c:200-245` is a hand-written straight-line
sequence of 5 calls with every dimension a `#define`. That does not scale to 79 layers
and must become a **table-driven layer descriptor interpreter**. The
`tinyvad_conv1d_hook` / `tinyvad_dense_hook` dispatch pattern
(`tiny_vad_infer.c:28-29`) is good and carries over.

`conv_im2col.h:38`'s `IM2COL_MAX_K = 256` (a 256-byte stack array) must become a
tiled/streaming buffer — QuartzNet needs K up to 512.

**CTC greedy decode runs on the PicoRV32**, not the accelerator: argmax over 29 classes ×
T frames, then collapse repeats and blanks. Trivial compute, and a clean division of
labor — the CPU is the control processor, never the datapath.

---

## Work plan

### Stage A — Model export (short; no retraining)
Pull `stt_en_quartznet15x5` from NeMo → ONNX → **ONNX Runtime static per-channel int8
PTQ**. Q-ASR measured only **+0.29% WER** for W8A8 on QuartzNet, so this is low-risk.
Reuse the *output format* and the Q31 `(q_mult, rshift)` decomposition from
`sw/tinyml_reference/export_weights.py`, but the source is ONNX, not TFLite. Emit weights
as a binary blob (not a `.h` — 18.85 MB will not compile into firmware) plus a layer
descriptor table. Generate golden per-layer activations for bit-exactness checking.

### Stage B — C reference (host x86 first)
Table-driven interpreter over the descriptor table: separable conv (depthwise +
pointwise), residual 1×1 projections, BN folded into the requantize scales, CTC greedy
decode. Validate WER against the ONNX int8 model on a LibriSpeech subset. This is the
numerical golden model everything downstream is checked against.

### Stage C — Cycle-accurate model with a **real memory model** ⚠️ gate
Extend `sim/verilator/sim_main.cpp`. The current accelerator model
(`accel_execute()`, ~line 226) computes the whole matvec instantly and fakes latency with
`accel_done_at = cycle_count + latency`, against a magic 0-wait-state memory. **Replace
that with an explicit bandwidth/latency model** for external weight memory plus SRAM port
contention.

**Do not start RTL until this stage produces credible utilization numbers.** The memory
hierarchy is the entire design; if the dataflow doesn't hold up here, the RTL is wasted
work. Sweep `T_TILE` × `LANES` × external BW and confirm the ≥10–15% utilization needed
for real-time is actually reachable.

### Stage D — RTL + unit TB
Build the components above. Extend `rtl/tb/` — its golden-model methodology and the
`-GLANES/-GACC_W` ↔ `-DTB_*` parameter-consistency pattern (`rtl/tb/Makefile:20-23`)
carry over directly. Must be bit-exact against the Stage B reference per layer.

### Stage E — Physical, with SRAM macros ⚠️ highest new risk
First macro integration in this repo. Add `ADDITIONAL_LEFS` / `ADDITIONAL_LIBS` /
`GDS_ALLOW_EMPTY` and macro-placement settings to
`physical/orfs/make/nangate45/tinymac_accel/config.mk`. ~18 macros at `T_TILE=32`.

`run.sh` and `sweep.sh` largely survive — `sweep.sh:104`'s
`VERILOG_TOP_PARAMS="LANES $lanes ACC_W $acc"` mechanism is generic and just needs
`T_TILE` added to the grid, and the CSV scraping (`sweep.sh:136-151`) already collects
area/util/WNS/fmax/power.

**De-risk this early** — before Stage D is finished, push a trivial dummy design with 2–3
fakeram45 macros through the flow to prove macro placement works at all. Everything in
this repo to date is flip-flop-only.

### Stage F — Second operating point
Re-run the same RTL against a scaled-down QuartzNet-style net whose weights fit entirely
on-chip. From the variant sweep, **5 blocks × R=3 at C=128 ≈ 0.51 M params = 0.51 MB** is
the largest configuration that fits a credible on-chip SRAM budget. Same RTL, `T_TILE`
retuned, external DMA idle. Report both points.

---

## Verification

| Stage | Check |
|---|---|
| A | ONNX int8 WER on LibriSpeech test-clean within ~0.3% of FP32 (expect ~4.2–4.5%) |
| B | C reference bit-exact vs ONNX Runtime int8 per layer; end-to-end WER matches A |
| C | Cycle model reproduces Stage B outputs exactly; utilization ≥ real-time threshold |
| D | Verilator unit TB bit-exact vs Stage B golden, across LANES ∈ {8,16,32}, T_TILE ∈ {16,32,64} |
| D | Full-system Verilator run: transcript from PicoRV32 + accel matches Stage B on a held-out utterance set |
| E | `6_final.gds` produced; `sweep_results.csv` populated across LANES × T_TILE × clock |
| F | Scaled model runs on identical RTL with DMA disabled; WER reported honestly |

End-to-end smoke test, mirroring the existing `make run` flow:

```bash
cd sim/verilator && make run
```

---

## Risks and honest caveats

- **SRAM macros are the single biggest unknown.** Zero prior experience in this repo, and
  fakeram45 macros have no behavioral model and no GDS — they are placement/timing
  placeholders only. Mitigated by de-risking with a dummy design in Stage E.
- **This is a much larger project than Stages 1–6** — realistically several months of
  serious part-time work. Stage C is the designed off-ramp: if the dataflow numbers don't
  hold, you stop there having spent weeks, not months.
- **WER in the real product will be worse than 4.19%.** That figure is LibriSpeech
  test-clean; test-other is 10.98%. Pocket-recorder far-field audio is closer to the
  latter or worse. QuartzNet 15x5 is still *better* than whisper-tiny (~7.6% clean) on
  clean speech, which supports replacing Whisper — but set expectations accordingly.
- **`ACC_W` must go to 32.** `docs/06_rtl_to_gds.md` records that narrow accumulators
  saturate per-LANES-chunk, making accuracy lanes-dependent (47–58/64 on TinyVAD). With
  K up to 512, `ACC_W=24` will saturate badly. Drop it as a swept axis.
- **Yosys 0.64 signed/unsigned gotcha still applies** (`genrtlil.cc:2214`): no `$signed()`
  on unsigned whole wires, no signed `integer` params in unsigned expressions, no
  mixed-sign `?:` branches. Verilator lint and yosys 0.9 both miss these.
- **The external memory is modeled, never built.** The chip is not self-contained at the
  15x5 operating point. Stage F exists precisely to provide a self-contained data point
  alongside it.

## Not doing

- **Gemmini / NVDLA.** Gemmini is Chisel/Rocket/Chipyard — adopting it means abandoning
  PicoRV32 and your own RTL, which is the point of the project. NVDLA `nv_small` is an
  8×8 int8 array with 64 KB CBUF and *no dedicated SRAM* (all traffic hits system
  memory), its synthesis scripts target Design Compiler, and its SRAMs are behavioral
  models. Neither solves the 18.85 MB weight problem — you would inherit it either way.
- **Retraining or architecture search on the model.** Hardware-first, per scope.
- **Running QuartzNet on the PicoRV32 itself.** At ~20 MMAC/s it is ~47,000× short of
  real-time. The CPU is the control processor; that is the correct role for it.
