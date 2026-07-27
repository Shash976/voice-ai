# Stage 7 — QuartzNet 15x5 ASR accelerator (revised after agent recon)

## Context

The project today builds a wake-gate: a 14.6 KB TinyVAD model on PicoRV32 + the TinyMAC
accelerator decides whether to wake a Raspberry Pi, which runs Whisper. Stages 1–6 are
done (working nangate45 GDS, ~19,738 µm², Fmax ≈269 MHz).

**We are changing the product: the chip transcribes speech itself, replacing Whisper.**
Target model is NeMo QuartzNet 15x5 — a fully convolutional CTC acoustic model: pure 1D
CNN, no attention, no autoregression, quantizes at only **+0.29% WER** (Q-ASR, W8A8).

Verified sizing (`sw/tinyml_reference/quartznet_budget.py`; 18,846,016 params reproduces
the paper's 18.9M, and the 5x5 variant its 6.7M):

| | TinyVAD (today) | QuartzNet 15x5 |
|---|---|---|
| int8 weights | 14.6 KB | **18.85 MB** (1,290×) |
| MACs | ~0.2 M / inference | 18.85 M **per output frame** |
| Real-time | — | 942 MMAC/s @ 50 output frames/s |
| Accuracy | binary VAD | 4.19% WER test-clean, 10.98% test-other (greedy, no LM) |

Scope, as decided: **one RTL, two operating points** (full 15x5 with external weights;
scaled variant fully on-chip); **chip replaces Whisper** so WER matters; **fakeram45
macros on nangate45**; **hardware-first** — no retraining, reuse the released checkpoint.

---

## Revisions from agent reconnaissance

Three parallel agents did read-only recon before writing code. Four findings change the
design; they are folded into the architecture below.

### 1. Cross-layer frame tiling was numerically invalid — corrected to layer-sequential

The first draft proposed pushing 32-frame tiles through all 186 layers. The cumulative
depthwise receptive field is Σ(K−1)·dilation over the stack:

| | B1 | B2 | B3 | B4 | B5 | C2 | total |
|---|---|---|---|---|---|---|---|
| frames | 480 | 570 | 750 | 930 | 1110 | 172 | **4,012** |

**4,012 frames = 80.2 s of audio**, and NeMo uses symmetric padding so lookahead is needed
too. A 32-frame tile through the whole network silently discards every halo. The draft's
29.4 MB/s was just 18.85 MB / 0.64 s — i.e. it was pricing exactly that broken dataflow.

**Corrected dataflow: layer-sequential over the whole utterance, tiled *within* each
layer.** Per-layer halo is only (K−1)·dil — at most 172 frames (C2), typically 32–74.

### 2. Activations go to external memory, not on-chip SRAM

Layer-sequential needs full-length activation tensors. At 4 µm²/byte (below), the ~750 KB
working set would be **~3 mm² of SRAM** — larger than the rest of the chip by an order of
magnitude. So activations stream to external RAM, same as weights.

Revised traffic for a 10 s utterance:

| | size | direction | medium |
|---|---|---|---|
| weights + requant params | 19.68 MB | read-only | QSPI flash |
| activation traffic | 36.10 MB | read/write | external PSRAM |
| **total** | **55.78 MB** | | **5.6 MB/s** |

Resident activation working set: ~750 KB (3 tensors). **This is a more credible product
than the draft** — 5.6 MB/s against commodity parts (32 MB QSPI flash + 8 MB PSRAM),
versus the draft's 29.4 MB/s for a dataflow that didn't actually work.

### 3. fakeram45 density is counterintuitive — smaller macros win

Measured across all 22 geometries:

| macro | area µm² | capacity | µm²/byte |
|---|---|---|---|
| **fakeram45_1024x32** | 16,406 | 4 KB | **4.005** ← best |
| fakeram45_512x64 | 17,301 | 4 KB | 4.224 |
| fakeram45_2048x39 | 45,479 | 9.75 KB | 4.555 |
| fakeram45_128x256 | 33,990 | 4 KB | 8.298 |

The 9.75 KB macro is **12.1% worse per byte** than the 4 KB one — inverting normal
SRAM-compiler intuition (an artifact of fakeram45 being synthetic, but it is ground truth
for this flow). Building 69 KB from `2048x39` would cost ~68,500 µm² more than from
`1024x32` — ~3.5× the entire existing accelerator, purely from picking the big macro.

**Banking decision: pair two `1024x32` side-by-side as one logical 64-bit port** — best
density *and* a 64-bit port, at the cost of one extra instance to place.

### 4. SRAM dominates area — the cost model inverts

One 4 KB macro (16,406 µm²) is ~83% of the *entire* current placed accelerator
(19,738 µm²). On-chip SRAM will be ~90% of the die; at realistic 50–60% macro
utilization the die lands at **490,000–590,000 µm² (~25–30× today)**.

Stage 5's headline — "16× the MACs costs only 1.86× area" — remains true but becomes
nearly irrelevant to total area. **The dominant area knob is SRAM capacity, not LANES**,
which should re-weight the eda-rl search space.

### Smaller corrections
- **`ACC_W=32` is mandatory, not a swept axis.** Worst-case |acc| = 1024·255·127 =
  33,147,840 (~6 bits under 2³¹). ACC_W=24 saturates above 8,388,607 — i.e. any pointwise
  with c_in ≥ 260, which is every layer from B1 onward.
- **Requant-param blob is 837,228 B** (bias/qmult/rshift int32 × 69,769 output channels),
  +4.4% on top of weights, streaming alongside them. Unaccounted for in the first draft.
- **29 output logits**, not 28: NeMo has 28 labels (space, a–z, apostrophe) and
  `ConvASRDecoder` adds the CTC blank. Total becomes 18,847,040. Make `n_classes` a
  descriptor-table header field.
- `GDS_ALLOW_EMPTY ?= fakeram.*` is **already set** in the nangate45 platform config —
  do not set it in the design config. `6_final.gds` should be producible.
- **`ADDITIONAL_LIBS` must be set in the design config**, because `LIB_FILES` is composed
  inside the platform config which `variables.mk:40` includes *afterwards*. Set it later
  and it silently never arrives.
- fakeram45 signal pins are **west-edge only** with OBS blanketing metal1–3; only metal4+
  routes over a macro. Macro orientation is a first-order routability decision.

---

## Architecture

### The reuse insight

Depthwise output channel `c` is a K-tap dot product of `in[c][t..t+K]` with `wt[c][0..K]`;
pointwise is a dot product of `in[:][t]` with `wt[m][:]`. **Both are the same
accumulate-over-K FSM the existing core already runs** — they differ only in how
`o_m`/`o_k_base` map to addresses. So `int8_mac_array.v` and the
`S_INIT_CH → S_MAC → S_REQ` FSM survive nearly intact; the new work is address
generation, memory, and width.

Depthwise is tiled over *channels* (it is channel-independent), which avoids the C2 halo
blowup; pointwise is tiled over output channels and time.

```
QSPI flash (19.7 MB weights, read-only) ──┐
external PSRAM (~750 KB activations, r/w)─┤  ~5.6 MB/s aggregate
                                          ▼
                                    ┌──────────┐
                                    │   DMA    │──► wt_buf (double-buffered)
                                    └──────────┘
 act_in ─┐                                │
 act_mid ├── on-chip SRAM ◄───────────────┘
 act_out─┘  (fakeram45, ~64-96 KB)
              │            ▲
              ▼            │
      ┌───────────────┐    │
      │ MAC datapath  │────┘
      │ LANES × int8  │──► requantize (PIPELINED)
      └───────▲───────┘
              │  addr-gen: DW mode | PW mode
      ┌───────┴──────────┐
      │ descriptor engine│◄── PicoRV32 (control + CTC decode only)
      └──────────────────┘
```

### Components
- **`int8_mac_array.v`** — reused as-is, widened.
- **`requantize.v`** — pipelined across 2 cycles (in progress). It is the current critical
  path (3.72 ns) and is *independent of LANES*, so it caps Fmax at any width.
- **`addr_gen.v`** (new) — DW/PW address generation.
- **`act_sram.v` / `wt_buf.v`** (new) — banked fakeram45 wrappers; **blackboxed for
  synthesis, behavioral model for Verilator**, mirroring how the repo already keeps the
  256 KB main memory in `sim/verilator/sim_main.cpp` rather than RTL.
- **`ext_mem_if.v`** (new) — QSPI + PSRAM streaming with a real `valid/ready` handshake.
  The current core has **no stall input at all**; that breaks the moment memory has
  variable latency.
- **`quartznet_accel.v`** (new top) — descriptor engine + MMIO register file. Note the
  `0x20000000` map today exists only in `accel.h` and C++; there is no Verilog for it.

### Firmware
`tiny_vad_infer.c:200-245` is a straight-line sequence of 5 calls with `#define`
dimensions — it must become a **table-driven descriptor interpreter**. The
`tinyvad_conv1d_hook`/`tinyvad_dense_hook` pattern (`:28-29`) carries over.
`conv_im2col.h:38`'s `IM2COL_MAX_K = 256` must go (QuartzNet needs K up to 512).
**CTC greedy decode runs on the PicoRV32** — argmax over 29 classes, collapse repeats,
drop blanks. The CPU is control, never datapath.

---

## Work plan

**Stage A — Model export.** NeMo `stt_en_quartznet15x5` → ONNX → ONNX Runtime static
per-channel int8 PTQ. Reuse the Q31 `(q_mult, rshift)` decomposition from
`sw/tinyml_reference/export_weights.py`. Emit a weight blob (not a `.h` — 18.85 MB will
not compile into firmware) + descriptor table + golden activations.

**Stage B — C reference.** Table-driven interpreter, bit-exact with a NumPy golden model.
Validate WER on a LibriSpeech subset.

**Stage C — Cycle model with real memory ⚠️ gate.** Extend `sim/verilator/sim_main.cpp`.
Today `accel_execute()` computes the whole matvec instantly against a magic 0-wait-state
memory and fakes latency via `accel_done_at`. Replace with an explicit
bandwidth/latency model for QSPI + PSRAM plus SRAM port contention. **No RTL until this
produces credible utilization numbers** — the memory hierarchy is the entire design.

**Stage D — RTL + unit TB.** Extend `rtl/tb/`; its golden-model methodology and
`-GLANES/-GACC_W` ↔ `-DTB_*` consistency pattern carry over directly.

**Stage E — Physical with SRAM macros ⚠️ highest risk.** First macro integration in this
repo. Already de-risked by a 4→8 macro spike (`rtl/spike_sram/`). `sweep.sh:104`'s
`VERILOG_TOP_PARAMS` mechanism is generic and just needs the new params added.

**Stage F — Second operating point.** Same RTL against a scaled-down variant resident
entirely on-chip (~5 blocks × R=3 at C=128 ≈ 0.51 MB).

---

## Verification

| Stage | Check |
|---|---|
| A | ONNX int8 WER within ~0.3% of FP32 (expect ~4.2–4.5% test-clean) |
| B | C reference bit-exact vs NumPy per descriptor; WER matches A |
| C | Cycle model reproduces B exactly; utilization ≥ real-time threshold |
| D | Verilator TB bit-exact vs B golden across LANES ∈ {8,16,32} |
| D | Full-system run: PicoRV32 + accel transcript matches B on held-out utterances |
| E | `6_final.gds` produced; sweep CSV across LANES × SRAM capacity × clock |
| F | Scaled model on identical RTL, DMA idle; WER reported honestly |

```bash
cd sim/verilator && make run
```

---

## Risks and honest caveats

- **SRAM macros remain the biggest unknown**, though the spike de-risks it. fakeram45 has
  no behavioral model and no GDS — placement/timing placeholders only.
- **Requires a RISC-V toolchain** (`gcc-riscv64-linux-gnu`), currently missing. Blocks
  full-system sim at Stage D, nothing earlier.
- **Several months of work.** Stage C is the designed off-ramp.
- **Real-world WER will exceed 4.19%.** That is test-clean; test-other is 10.98%, and
  far-field pocket audio is worse. Still better than whisper-tiny (~7.6% clean), which
  supports replacing Whisper — but set expectations.
- **Yosys 0.64 signed/unsigned assert** (`genrtlil.cc:2214`): no `$signed()` on unsigned
  whole wires, no signed `integer` params in unsigned expressions, no mixed-sign `?:`.
  Verilator lint and yosys 0.9 both miss these — only real ORFS synthesis catches them.
- **External memory is modeled, never built.** Stage F provides the self-contained point.

## Not doing
- **Gemmini / NVDLA.** Gemmini is Chisel/Rocket/Chipyard — adopting it abandons PicoRV32
  and your own RTL, the point of the project. NVDLA `nv_small` is 8×8 int8 with 64 KB
  CBUF and *no dedicated SRAM*, its synth scripts target Design Compiler, its SRAMs are
  behavioral. Neither solves the 18.85 MB problem — you inherit it either way.
- **Retraining or NAS.** Hardware-first, per scope.
- **QuartzNet on the PicoRV32 itself** — ~20 MMAC/s is ~47,000× short. CPU is control.
