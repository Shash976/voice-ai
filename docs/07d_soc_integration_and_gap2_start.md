# 07d — RTL↔SoC integration completion (D1-D6) + Gap 2 start (A0, A2)

This session closed out Stage 7 Gap 1 (`docs/07c`'s "what's still missing"
item 1 — RTL↔PicoRV32 SoC integration) completely, across six increments
(D1-D6), and started Gap 2 (item 2 — real trained weights) with the
environment set up and the checkpoint↔topology mapping verified.

The plan this work follows is `/home/shashg/.claude/plans/generate-an-implementation-plan-pure-robin.md`
(written by an Opus-model Plan agent, reviewing three Explore agents' research
into the SoC/firmware/PTQ state of the repo). This doc records what actually
happened running that plan: every increment, every real bug found, and the
concrete state to pick up from next.

All work is on a stack of five PRs against `feat/quartznet-asr-accel`,
branched from a new branch `feat/quartznet-soc-and-weights` created first per
the plan's "don't disturb the existing branch" instruction — then split
further into one branch per PR so each is independently reviewable, each
based on the previous one's tip:

```
feat/quartznet-asr-accel                         (pre-existing)
  └─ feat/quartznet-soc-and-weights               (branch point only)
      └─ feat/quartznet-rtl-hostport              PR #1: D1-D3
          └─ feat/quartznet-soc-rtl-integration    PR #2: D4
              └─ feat/quartznet-full-config-shim   PR #3: D5
                  └─ feat/quartznet-nemo-export     PR #4: D6
                      └─ (this branch)              PR #5: Gap 2 A0+A2
```

Wait — see the actual PR numbers/links below; branch names and PR numbers
don't line up 1:1 in the list above because of how the stack grew. Use the
table in "PRs opened" for the authoritative mapping.

---

## Gap 1: RTL ↔ PicoRV32 SoC integration — D1 through D6 (all done)

### D1 — Interpreter refactor

`firmware/quartznet/quartznet_infer.{c,h}`: the activation arena is now a
settable pointer (`qn_set_arena()`, `QN_STATIC_ARENA` flag) instead of a
fixed static array, and `qn_ctc_greedy()` is factored into
`qn_ctc_greedy_buf()` over an explicit logits pointer. Needed because the
hardware-dispatch path's arena lives in accelerator PSRAM, not CPU RAM — the
320 KB arena at full utterance length exceeds the 256 KB RAM budget, and per
the plan, **moving the arena off-chip is the correct call, not a three-way
tradeoff** (shrinking caps utterances at ~1.7s; growing RAM fights Stage 7's
own finding that SRAM should go to the weight buffer, not activations).
Verified inert: `make host` unchanged (187/187 + 27/27 bit-exact, transcripts
byte-identical).

### D2 — Firmware driver + Makefile fix

New `firmware/quartznet/qn_accel.{h,c}` (the hardware-dispatch driver:
`qn_run_hw`, `qn_run_hw_step`, `qn_set_input_hw`, `qn_gather_output_hw`,
`qn_read_logits_hw`) and `qn_main.c` (bare-metal entry point — no filesystem;
table/input are linked-in const arrays via new `gen_fw_headers.py`,
weights/qparams live in QSPI and are never CPU-loaded). Fixed the `rv32`
Makefile target, which was broken (assumed an uninstalled toolchain, no link
step, missing `-fno-pic -fno-pie`) using the `riscv64-linux-gnu-gcc` pattern
already proven by `firmware/picorv32_baremetal`. New `rv32-fw` target links
`qn_firmware.bin` (13 KB, zero floating-point instructions).

### D3 — C++ shim

New `sim/verilator_qn/` directory: Verilates the **unmodified**
`rtl/soc/picorv32_soc.v`; `sim_main_qn.cpp` emulates the accelerator's
register file in C++ but delegates every real computation to the actual
`quartznet_infer.c` interpreter (not a hand-rolled reimplementation, so it
can't drift from the golden model). Result: **PicoRV32 firmware driving the
accelerator purely over MMIO produces a transcript byte-identical to the
x86 golden (`"xix"`, reduced config).**

### D4 — Real Verilog: MEM_* host-access port + SINGLE_STEP

`rtl/accel/quartznet_accel.v` gained `MEM_ADDR`/`MEM_CTRL`/`MEM_DATA`/
`MEM_FILL` (idx 33-36), replacing the simulation-only `bd_*` backdoor for real
SoC use, and `CTRL.SINGLE_STEP` (bit3) so firmware can halt the table walker
one descriptor at a time.

**Two real bugs found here, both via testing, not inspection:**
1. **A silently-dropped write.** The first working-looking design drove
   `ext_mem_if`'s `bd_*` inputs combinationally, in the same cycle as the
   register-file write that triggered a `MEM_DATA` write. It compiled and
   lint-checked clean, and single writes (no AUTOINC) worked — but every
   write made while `MEM_CTRL`'s AUTOINC bit was set was silently dropped,
   with zero error indication. Root-caused by comparing `$display` output
   inside `quartznet_accel.v` against `$display` output inside
   `ext_mem_if.v`'s own write path (only the latter is trustworthy —
   Verilator's procedural `$display` inside a clocked block can read a
   combinational value one delta-cycle "early" relative to what a downstream
   module actually uses at the same edge). Fixed by registering the request
   one cycle (`r_mem_we_q`/`r_mem_addr_q`/`r_mem_wdata_q`/`r_mem_sel_q`), the
   same pattern `q_req_valid`/`p_req_valid` already use elsewhere in this
   file — and then had to split the address mux so **reads** still see the
   live `r_mem_addr` (only writes need the registered snapshot), since the
   first fix broke reads by routing them through the same snapshot.
2. **A fill/write race.** `MEM_FILL`'s multi-cycle background operation
   (`S_MEMFILL`, one `bd_*` write per cycle, asserting `STATUS.BUSY` like a
   compute op) needs to be polled to completion — `qn_mem_fill()` in
   `qn_accel.c` didn't poll at all. A `while(BUSY){}` poll issued with no
   forced intervening clock edge can read stale `BUSY=0` from before the FSM
   even entered `S_MEMFILL`, so the caller thinks the fill is done when it's
   still running — and the mel-input preload that follows would then race
   the still-running fill and corrupt both.

`rtl/tb/quartznet_tb.cpp` gained `run_mem_port_test()`: a directed MMIO
register read/write-back unit test, then the same 27 descriptors preloaded
and read back entirely through `MEM_*`/`SINGLE_STEP` instead of `bd_*`.
**27/27 bit-exact at LANES∈{8,16,32}**; `bd_*`-based single-descriptor mode,
table-walk mode, 87/87 tiles, and the legacy TinyVAD 12/12 suite all
unaffected.

### D5 — Real SoC integration (the milestone)

New `rtl/soc/qn_soc.v`: wraps the **unmodified** `picorv32_soc.v` plus
`quartznet_accel.v`, with the ~25 lines of address decode this needed
(accelerator owns `0x2000_0000`'s 4KB window, `mmio_addr = cpu_mem_addr[9:2]`,
0-wait-state register file). New `sim/verilator_qn/sim_main_qn_rtl.cpp` +
`Makefile MODE=rtl`: Verilates `qn_soc.v` directly, **no C++ accelerator
emulation at all** — real Verilog answers every accelerator-range
transaction.

**Result: `make -C sim/verilator_qn run MODE=rtl` → `transcript("xix")`,
byte-identical to the x86 golden — real PicoRV32 firmware driving real
`quartznet_accel.v` RTL over a real bus.** This is the actual fidelity claim;
D3's shim result was a checkpoint on the way there.

**A third real bug, found only by running against real RTL:** D4 redesigned
`MEM_DATA` to be single-byte (a genuine hardware constraint — no
ready/wait-state signal on this bus, so a 32-bit transfer through the
byte-granular `bd_*` backdoor can't be atomic in one cycle), but
`qn_accel.c`'s `qn_mem_write()`/`qn_mem_read()` were never updated off their
original 32-bit-with-tail-handling design. Against `MODE=shim` this was
**invisible** — the shim's own C++ emulation of `MEM_DATA` was *also* still
32-bit, so both sides were self-consistently wrong and D3's "xix" result,
while real, was accidental. Against real RTL it produced immediate garbage
(`" cba cba cba..."`), caught by diffing against the known-good golden and
root-caused by logging the actual register writes reaching `qn_soc.v`.
Fixed in both places (both got *simpler* for it — the read-modify-write tail
handling from the 32-bit design is gone entirely):
`firmware/quartznet/qn_accel.c` and `sim/verilator_qn/sim_main_qn.cpp`
brought into agreement.

Confirmed `rtl/soc/picorv32_soc.v` / `rtl/picorv32/picorv32.v` have **zero
diff** — `make -C sim/verilator run` still 64/64.

### D6 — Full 187-descriptor config through the shim

Pure build-plumbing (no RTL touched): `firmware/quartznet/Makefile`'s
`gen-headers` and `sim/verilator_qn/Makefile`'s new `CONFIG=reduced|full`
variable let the same firmware/driver stack target the full 15x5 table
instead of the 27-descriptor reduced config.

**Result:** `make -C sim/verilator_qn run MODE=shim CONFIG=full` →
`transcript("fdiz'fh'mjcybzgdmk'kz'azmzsbciyqzkzsiy'yqsmfjbkqmiyapkcepcfjejycj'")`,
byte-identical to `build/quartznet/quartznet_golden.txt`.

**Also attempted (not required for D6, opportunistic):**
`MODE=rtl CONFIG=full` — timed out at 2 billion simulated cycles within a
10-minute wall-clock budget. **Correction (see the follow-up session that
closed this out):** this was never a throughput problem — `qn_soc.v`'s
`QSPI_BYTES`/`PSRAM_BYTES` parameters default to 1 MiB each, and
`sim/verilator_qn/Makefile`'s RTL build never forwarded `CONFIG=full`'s real
sizes at Verilate time, so the ~19.7 MB QSPI image was silently truncated,
the table header read back all zeros, and `quartznet_accel.v`'s
`S_BUFTBL_W` state compared a 4-bit counter against an unsigned-wrapped
`0xFFFFFFFF` that can never match — an infinite loop before a single
descriptor executed, not `ext_mem_if.v`'s lack of prefetch overlap. Once
`-GQSPI_BYTES=$(QSPI_BYTES) -GPSRAM_BYTES=$(PSRAM_BYTES)` were added to the
RTL Verilate flags, `MODE=rtl CONFIG=full` at the existing `T_OUT=70` (no
scope-down needed) completes in ~2m37s wall-clock at 554,542,940 cycles,
transcript byte-identical to both `MODE=shim CONFIG=full` and the committed
x86 golden. `ext_mem_if.v`'s blocking one-request-at-a-time design is real
and does cost real cycles (measured ~10.3× over the Stage C cost model's
overlapped-DMA estimate for the same op count), but it was never what made
this time out.

---

## Gap 2: Real trained weights — A0 and A2 (started)

### A0 — Environment + downloads

No conda on this machine (confirmed, matching `CLAUDE.md`'s prior note) — used
a plain venv (`venv/`, gitignored) instead. Installed: `torch==2.13.0+cpu`,
`onnx==1.22.0`, `onnxruntime==1.23.2`, `librosa==0.11.0`,
`torchaudio==2.11.0+cpu`, `soundfile==0.14.0`, `numpy==2.2.6`.

`environment.yml` updated to match: `onnxruntime>=1.28` was **unsatisfiable**
(checked against the live PyPI index — latest is 1.23.2), relaxed to
`>=1.23`. `onnx>=1.22` turned out to be satisfiable as-is (latest is exactly
1.22.0 — right at the edge). Added `torch`/`librosa`/`torchaudio`.

Downloaded (gitignored, not committed — see `.gitignore`):
- `stt_en_quartznet15x5.nemo` (67.7 MB) from NVIDIA NGC:
  `https://api.ngc.nvidia.com/v2/models/nvidia/nemo/stt_en_quartznet15x5/versions/1.0.0rc1/files/stt_en_quartznet15x5.nemo`
  (this 302-redirects to a signed, time-limited `xfiles.ngc.nvidia.com` URL —
  re-resolve from the API URL above if it needs re-downloading later, don't
  reuse a stale signed URL).
- `librispeech/dev-clean.tar.gz` (322 MB) and `librispeech/test-clean.tar.gz`
  from `https://www.openslr.org/resources/12/`. `dev-clean` is extracted to
  `librispeech/LibriSpeech/dev-clean/` (2703 `.flac` files); `test-clean` was
  downloaded but not yet extracted.

### A2 — Checkpoint → topology-verified export

New `sw/tinyml_reference/quartznet_nemo_export.py`. Confirmed (by directly
inspecting the checkpoint's `state_dict`, not just trusting the published
architecture) the exact tensor-naming scheme:

```
residual projection   encoder.encoder.{bid}.res.0.0.conv.weight   (conv)
                       encoder.encoder.{bid}.res.0.1.*             (BN)
depthwise (repeat r)  encoder.encoder.{bid}.mconv.{5r}.conv.weight
pointwise (repeat r)  encoder.encoder.{bid}.mconv.{5r+1}.conv.weight (conv)
                       encoder.encoder.{bid}.mconv.{5r+2}.*          (BN)
```

with two exceptions found only by inspection:
- **C3** (`bid=17`, the only non-separable *encoder* block) has no depthwise:
  a single conv+BN at `mconv.0`/`mconv.1` instead of the 5-slot pattern.
- **C4** (`quartznet_topology.py`'s synthetic 19th `BLOCKS` entry) is **not**
  in `encoder.encoder` at all — NeMo implements it as a separate
  `decoder.decoder_layers.0` `Conv1d` with a real bias (no BN to fold).

A third finding surfaced by the export code itself failing: **depthwise
convs have neither BN nor bias** in the float model (the topology.py header
comment already said this — "no BatchNorm and no activation between the
depthwise and the pointwise" — but the first cut of the export code only
branched on "has BN" vs. "has bias" and hit a `KeyError` on the first
depthwise layer; fixed with a third branch, bias = zero array).

Weight layout required **zero transposition**: PyTorch's `Conv1d`
`[out, in/groups, k]` squeezes to exactly `quartznet_descriptors.py`'s
`w[c*K+k]` (depthwise) / `w[oc*c_in+ic]` (pointwise) layout already.

```
$ python3 sw/tinyml_reference/quartznet_nemo_export.py stt_en_quartznet15x5.nemo
parameter count: 18,847,040 == EXPECTED_PARAMS (exact match)
171/171 weight-bearing descriptors bound to a named checkpoint tensor
zero unexplained unused tensors
G2.2: PASS
```

Output: `build/quartznet_nemo/checkpoint_map.json` (layer_id → tensor names)
and `build/quartznet_nemo/folded_weights.npz` (per-layer BN-folded fp32
weight+bias, gitignored build products — regenerate from the command above).

---

## PRs opened (stacked, each reviewable independently)

| # | Branch | Base | Contents |
|---|---|---|---|
| [#1](https://github.com/Shash976/voice-ai/pull/1) | `feat/quartznet-soc-and-weights` | `feat/quartznet-asr-accel` | D1-D3 |
| [#2](https://github.com/Shash976/voice-ai/pull/2) | `feat/quartznet-rtl-hostport` | #1 | D4 |
| [#3](https://github.com/Shash976/voice-ai/pull/3) | `feat/quartznet-soc-rtl-integration` | #2 | D5 |
| [#4](https://github.com/Shash976/voice-ai/pull/4) | `feat/quartznet-full-config-shim` | #3 | D6 |
| [#5](https://github.com/Shash976/voice-ai/pull/5) | `feat/quartznet-nemo-export` | #4 | Gap 2 A0+A2 |

**Update:** all six PRs above have since merged into `feat/quartznet-asr-accel`
(merge commit `381890d`). This doc is kept as a record of that session; the
follow-up work (Gap 1's remaining RTL demo item, then Gap 2's A1/A3-A7) is
tracked in `~/.claude/plans/gentle-baking-pelican.md` as its own stacked PR
sequence starting from `feat/quartznet-gap1-rtl-demo`.

---

## What's left

### Gap 1 — done
- ~~Combined full-config RTL demo~~ **done**, and at the original `T_OUT=70`
  (no scope-down needed — see the correction on D6 above): `MODE=rtl
  CONFIG=full` completes in ~2m37s wall-clock, 554,542,940 cycles, transcript
  byte-identical to `MODE=shim CONFIG=full` and the committed x86 golden. The
  earlier timeout was a Verilate-time parameter-forwarding bug
  (`QSPI_BYTES`/`PSRAM_BYTES` never reaching the RTL build), not a scale
  problem — fixed in `sim/verilator_qn/Makefile`.

### Gap 2 (the bulk of remaining work)
- ~~A1 — front-end numerical validation~~ **done** (`docs/07d`'s own A1
  detail is unchanged; see this section's item 1 above / `CLAUDE.md`).
- ~~A3 — FP32 baseline~~ **done, full detail `docs/07e_gap2_a3_fp32_baseline.md`.**
  Found and fixed a real bug along the way: A2's `BN_EPS=1e-5` should have
  been `1e-3` (NeMo hardcodes its own eps, does not use PyTorch's default),
  inflating every folded BN scale ~2-4x and overflowing logits to ~1e31
  before the fix. The plan's original gate ("within 0.3% of NeMo's published
  3.90%") also named the wrong checkpoint — 3.90% belongs to
  `quartznet_15x5_ls_sp`, not the `stt_en_quartznet15x5` checkpoint this repo
  actually uses. Revised and measured: **FP32 WER 4.4392% on dev-clean**,
  0.04% absolute from `stt_en_quartznet15x5`'s own published 4.4% —
  **G2.3: PASS**.
- ~~A4 — ORT static per-channel int8 PTQ calibration~~ **done, full detail
  `docs/07f_gap2_a4_int8_calibration.md`.** `dev-clean`, 200 utterances
  (5/speaker × 40 speakers, ≤10s by filtering not truncating), fixed seed,
  through this repo's own `quartznet_audio.py` front end (not NeMo's), per
  the plan — except `CalibrationMethod.Percentile` at **99.999%**, not the
  plan's 99.99% (measured: 99.99 leaves 10x less gate margin, since clipping
  error compounds across 15 residual blocks; 99.99 remains selectable via
  `--percentile` for A7's ablation). **G2.4: PASS** on both splits — dev-clean
  int8 4.5678% vs fp32 4.4392% (delta 0.1287%, gate 0.30%), test-clean int8
  4.5002% vs fp32 4.4716% (delta 0.0285%). Two mandatory (not tuning)
  `extra_options` found by running, not reading docs: `CalibStridedMinMax=1`
  (Percentile's default calibrator buffers every intermediate tensor for the
  whole calibration set and crashes on this model's variable-length
  utterances without it) and `MinimumRealRange=1e-3` (599 near-zero-gamma
  BN channels would otherwise produce folded int32 biases within a few
  percent of overflow — harmless to ORT, fatal to the firmware's real int32
  accumulator).
- ~~A5 — the format bridge~~ **done, full detail
  `docs/07g_gap2_a5_int8_export_bridge.md`.** New
  `sw/tinyml_reference/quartznet_export_int8.py`, per-channel `(q_mult, rshift)`
  via `export_weights.quantize_multiplier` (imported, not reimplemented —
  `export_weights.py` had to be refactored into an importable module first,
  it previously required `tflite_runtime`/`tensorflow` at module scope,
  neither installed here), bias domain confirmed correct with zero
  conversion (re-verified independently, not just trusted from A4's
  docstring), and the `OP_ADD` per-tensor path computed from the *real*
  per-branch scales via TFLite's `twice_max` normalization (replacing the
  placeholder's `0.5/0.5` hardcode) — verified against 200,000 random int8
  operand pairs per Add, 14/15 exact, 1/15 within 1 count (a rounding tie).
  **G2.5: PASS** — `make -C firmware/quartznet host-real`: 187/187
  descriptors + transcript bit-exact between the C interpreter and the
  NumPy reference, on real calibrated int8 weights. Transcript
  `"as for etching"` — real English, matching the real audio clip's
  ground truth prefix. One real bug fixed along the way:
  `DescriptorTable` hardcoded fake zero points
  (`zp_out = -128 if relu else 0`) valid only for seeded-random weights;
  real calibrated non-ReLU `out_zp` ranges −78..+93 — gained an optional
  `zp_out=` override.
- ~~A6 — end-to-end~~ **done, full detail `docs/07h_gap2_a6_end_to_end_firmware.md`.**
  Found the prior session's "`qn_transcribe.c` needs zero changes" claim was
  wrong: `mp3_to_text.py` was quantizing with a placeholder scale instead of
  the real calibrated model's own (from `quartznet_meta.json`), costing a
  measured +0.28% absolute WER — more than half of G2.6's 0.5% band, from a
  single wrong default; fixed with a new `--model-dir` flag. Also needed
  `QN_MAX_T_OUT` raised (the stock 128 capped the host build at ~3s of
  audio; test-clean's longest utterance is 34.96s) and new
  `transcribe-real`/`wer-firmware` Makefile targets. **G2.6: PASS** — real
  mp3 → real English transcript through the real firmware C interpreter
  (`"he hoped there would be stew for dinner..."`, 2 errors/29 words vs
  ground truth), and full test-clean firmware WER **4.5002%**, delta
  **0.0000%** from A4's int8-ORT number — verified genuine (460/2620
  hypotheses differ between the two independent paths; the aggregate error
  count coincides exactly).
- ~~A7 — calibration-size ablation~~ **done, full detail
  `docs/07i_gap2_a7_calibration_ablation.md`.** 50/200/500 utterances (all
  40 speakers at every size — only per-speaker depth varies, so calibration
  size is never confounded with speaker diversity), one shared fp32 ONNX
  export reused across all three so the calibration set is the only varying
  input. **G2.7: PASS** — spread 0.0901% (gate <0.1000%), but by only 5.4
  word errors out of 54,402 (0.0099% of the gate band) — a real pass, not
  a rounding artifact, and documented as thin rather than papered over. The
  result is non-monotone (50 utterances gives the *lowest* WER, 200 the
  *highest*), so the honest reading is "calibration saturates by 50
  utterances," not "200 was necessary." Also found and fixed a real bug
  along the way, present identically in all three WER gate drivers
  (A3/A4/A6): each wrote its canonical `wer_*.json` *before* checking
  `--limit`, so any smoke-test run after a real gate run silently clobbered
  that gate's own result file — confirmed on disk (`build/quartznet_int8/
  wer_int8_dev-clean.json` held a stray `--limit 50` run's 7.5061% instead
  of A4's real 4.5678%).

This closes out Gap 2's core work (A0–A7). All eight steps of this
session's stack (Gap 1's RTL demo fix, A1, A3–A7) are on stacked, open PRs
against `feat/quartznet-asr-accel` — see `~/.claude/plans/gentle-baking-pelican.md`
for the sequence and each step's own `docs/07{e,f,g,h,i}_*.md` writeup.

### Housekeeping
- `librispeech/test-clean.tar.gz` was extracted in a later session (needed
  by A3 onward's WER gates) — `librispeech/LibriSpeech/test-clean/` now has
  the full 2620-utterance corpus.
