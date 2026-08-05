# 07c — Stage D increment 2 (autonomous table walking) + mp3-to-text mechanical pipeline

This session had two scoped, independent goals, confirmed up front:

- **Goal A (RTL):** give `quartznet_accel.v` autonomous descriptor-table walking,
  verified via the existing `rtl/tb` C++ testbench harness only — **not** wired
  into `picorv32_soc.v` / `sim/verilator/sim_main.cpp`. That SoC integration is
  explicitly out of scope here (see "What's still missing" below).
- **Goal B (software):** wire the mp3/wav audio front end into the C interpreter
  end to end so a real audio file produces a real (if not yet accurate) CTC
  transcript, using the existing seeded-random weights. Real trained weights are
  explicitly out of scope here.

Both shipped and are verified. This doc records what changed, how it was
verified, two real RTL bugs found along the way, and the concrete remaining
work to get an actual "mp3 in → correct English text out, running on the chip"
system.

---

## Goal A — autonomous descriptor-table walking (Stage D increment 2)

### Before

`quartznet_accel.v` (Stage D increment 1) executed **one descriptor per CMD
trigger**: software staged all 26 config fields over MMIO, pulsed
`CTRL.START`, and polled `STATUS.DONE`. Five field groups were resolved by
software, not hardware — `IN_BASE`/`OUT_BASE`/`RES_BASE` (buffer arena
offsets), the three buffer pitches, and a pre-reduced `IN_CBASE` — computed by
`rtl/tb/quartznet_tb.cpp` standing in for future firmware.

### After

New MMIO registers (word index, byte addr = idx×4):

| idx | name | acc | notes |
|---|---|---|---|
| 29 | `TABLE_BASE` | RW | QSPI byte address of the descriptor-table image |
| 30 | `W_BLOB_BASE` | RW | added to each descriptor's blob-relative `W_OFF` |
| 31 | `QP_BLOB_BASE` | RW | added to each descriptor's blob-relative `BIAS_OFF`/`QMULT_OFF`/`RSHIFT_OFF` |
| 32 | `DESC_IDX` | R | current descriptor index (debug only) |

`CTRL` bit1 = `RUN_TABLE` (self-clearing, ignored while busy). `STATUS` bit2 =
`TABLE_DONE` (sticky, W1C) fires once, after the *last* descriptor, instead of
per-descriptor `DONE` (bit1, which still fires every descriptor, unchanged).
`STATUS` bit3 = `err_bad_in_off` (sticky, read-only, cleared automatically at
the next `RUN_TABLE`) — see the `in_off` note below.

New FSM states, inserted before the existing `S_ADDQ`:

```
S_HDR / S_HDR_W        fetch the 64B/16-word table header
S_BUFTBL / S_BUFTBL_W  fetch n_buffers x 8B buffer records, compute
                        cumulative arena offsets (BASE) as they arrive
S_DESC / S_DESC_W      fetch one descriptor (64B/16 words)
S_POPULATE             decode the fetched words into registers 3-28,
                        deriving BASE/PITCH from the buffer table and the
                        blob offsets from *_BLOB_BASE -- exactly what
                        software used to do
S_KICK                 one-cycle handoff to addr_gen (see bug #2 below)
```

After `S_KICK`, execution falls into the **same** `S_ADDQ`/`S_ELEM`/.../`S_FIN`
path single-descriptor mode already used and had verified — nothing in
`int8_mac_array.v`, `requantize.v`, `requantize_add.v`, or `addr_gen.v`
changed. `S_FIN` now loops back to `S_DESC` for the next descriptor (in table
mode) instead of returning to `S_IDLE`, until the last descriptor completes.

**`IN_CBASE` still has no divider.** Every table `quartznet_descriptors.py`
emits today has `in_off == 0` for every descriptor (hardcoded — see its
`pack_record()`), so the modulo `in_off % pitch` the C interpreter computes is
always a no-op. Autonomous mode formalizes this as a hardware precondition: a
fetched descriptor with `in_off != 0` latches `err_bad_in_off` instead of
silently computing a wrong address. A real divider (or a guarantee this stays
true) is exactly the same open question increment 1 already flagged — nothing
new, just now enforced with a diagnostic.

### Two real RTL bugs found via this work

**1. Multi-driver conflict.** `S_POPULATE` needs to write the same config
registers (`r_op`, `r_c_in`, `r_in_base`, ...) that the MMIO write path
(`case (mmio_addr)`) also writes for legacy single-descriptor mode. These
started out in two separate `always @(posedge clk)` blocks — illegal Verilog
(two procedural blocks cannot drive the same reg) and would have failed
synthesis (Verilator's lint didn't catch it directly, but two blocks each
containing `r_op <= ...;` is a real multi-driver hazard). Fixed by merging
quartznet_accel's two always-blocks into one, MMIO handling textually before
the state machine so a same-cycle conflict (not a real scenario — software
never pokes config registers mid-table-walk) would resolve deterministically
in the state machine's favor.

**2. A one-cycle start/busy race.** `addr_gen`'s `start` input needs to be
high on the *same* edge that `quartznet_accel`'s own state machine transitions
into the state that will check `ag_busy` — exactly how `S_IDLE` and
`start_pulse` already work in legacy mode (`start_pulse` is asserted *while
still in* `S_IDLE`, so `addr_gen` latches `cfg_*` and sets `running<=1` at the
very edge that also moves `state` to `S_ELEM`; the *next* cycle, `S_ELEM`
correctly sees `ag_busy=1`).

The first table-mode attempt set `tbl_start_pulse<=1` **inside** `S_POPULATE`
and transitioned directly to `S_ELEM` in the same cycle — one edge too early
relative to `addr_gen`. `addr_gen` and `quartznet_accel`'s own `S_ELEM` both
evaluate at the *same* edge using *pre-edge* register values, so `S_ELEM`
would read `ag_busy` as still 0 (the value from *before* `addr_gen`'s own
same-edge update) and immediately (wrongly) jump to `S_FIN` — or, once a
second fix landed, run with correct-looking config but at exactly the wrong
moment relative to when memory addresses were fetched, corrupting results.
Symptom: descriptor 0 always produced a *constant* wrong value across every
output element (e.g. `-128` or `-122`), which is the signature of "requantize
ran off *some* stable-but-wrong accumulator," not a random/timing-flaky bug.

Fix: added `S_KICK` between `S_POPULATE` and `S_ADDQ`/`S_ELEM`. `S_POPULATE`
sets `tbl_start_pulse<=1` and `state<=S_KICK` (not `S_ELEM` directly);
`S_KICK` is the state that actually decides `S_ADDQ` vs `S_ELEM`, one cycle
later — exactly mirroring the `S_IDLE`/`start_pulse` alignment. Confirmed via
targeted `$display` instrumentation (added, verified, removed) showing
`addr_gen`'s cfg/accumulator values were byte-identical between modes once
this landed.

**A third apparent bug was a test-methodology bug, not RTL.** The first
working-looking comparison read every descriptor's output slice back from
PSRAM *after the whole table finished* — but `BUF_A`/`BUF_B` are ping-pong
buffers that *later* descriptors legitimately overwrite (that reuse is the
entire point of the arena layout). By the time descriptor 26 finishes,
descriptor 0's output region has been overwritten many times over by
descriptors 3, 5, 7, 9, 13, 16, 20, 23 (confirmed by instrumenting every store
to that address). Fixed by polling `DESC_IDX` during the walk and snapshotting
each descriptor's output slice the instant it completes (before the DUT ticks
again) — the same thing the single-descriptor loop already did implicitly by
comparing right after each `CTRL.START`.

### Verification

`rtl/tb/quartznet_tb.cpp` gained `run_table_test()`: loads the descriptor
table + weight/qparam blobs into the QSPI backdoor once, stages
`TABLE_BASE`/`W_BLOB_BASE`/`QP_BLOB_BASE`/`T_OUT`, pulses `RUN_TABLE`, and
polls `DESC_IDX`/`TABLE_DONE` for the streaming per-descriptor compare above.
Runs on a fresh DUT instance so it starts from a clean reset, independent of
the single-descriptor test that runs first in the same binary. The existing
single-descriptor test path is untouched.

```
27/27 descriptors bit-exact in table-walk mode, LANES ∈ {8, 16, 32}
27/27 descriptors bit-exact, single-descriptor mode (unmodified regression)
87/87 compute tiles match quartznet_infer.c's walk (unmodified regression)
12/12 legacy TinyVAD suite, LANES ∈ {1,2,4,8,16,32} × ACC_W ∈ {24,32} (unmodified regression)
```

Files touched: `rtl/accel/quartznet_accel.v`, `rtl/tb/quartznet_tb.cpp`.
`addr_gen.v` and `ext_mem_if.v` are unchanged — the walker reuses the existing
QSPI read channel and the existing per-op execution states as-is.

---

## Goal B — mp3-to-text mechanical pipeline

### Before

`sw/tinyml_reference/quartznet_audio.py` (mp3/wav → int8 log-mel features) was
a fully self-tested, working module that **nothing else in the repo ever
called**. `firmware/quartznet/quartznet_infer.c`'s only input source was
`quartznet_ref.py`'s seeded-random blob generator.

### After

- **`sw/tinyml_reference/mp3_to_text.py`** (new) — calls
  `quartznet_audio.audio_file_to_int8()`, writes
  `build/quartznet/mp3_input.bin` (a name distinct from the golden's
  `quartznet_input.bin`, so `make goldens` output is never clobbered) plus a
  `mp3_input.t_out` sidecar (t_out varies with clip length). Accepts a real
  mp3/wav path, or generates a synthetic smoke clip via
  `quartznet_audio.synth_clip()` if none is given.
- **`firmware/quartznet/qn_transcribe.c`** (new) — loads a model's
  `quartznet_desc.bin`/`quartznet_weights.bin`/`quartznet_qparams.bin` plus the
  mp3-derived input blob, calls the real `qn_load`/`qn_set_input`/`qn_run`/
  `qn_ctc_greedy` API (`firmware/quartznet/quartznet_infer.h`) — no new
  interpreter code needed — and prints the resulting transcript.
- **`make transcribe [MP3=path/to/clip.mp3]`** in `firmware/quartznet/Makefile`.

Weights/qparams are reused unmodified from `make goldens`'s existing
seeded-random blobs — qparams were calibrated against the *original random*
input, not the mp3-derived one, so **the transcript is expected to be
gibberish**. This is the accepted outcome for this phase: it proves the
plumbing (mp3 → features → interpreter → CTC decode → string), not accuracy.

### Verification

```
$ make transcribe
...
=== model ../../build/quartznet_reduced   input .../mp3_input.bin ===
  descriptors 27   T_out 100
  transcript (1 symbols, gibberish expected -- seeded-random weights): "x"

=== model ../../build/quartznet   input .../mp3_input.bin ===
  descriptors 187   T_out 100
  transcript (89 symbols, gibberish expected -- seeded-random weights): "m'jmh'ymwnm..."

PASS -- mp3 -> accelerator interpreter -> CTC transcript, end to end
```

Confirmed shape-agnostic across clip lengths (tested 1.0s and 2.0s synthetic
clips, `t_out` 50 and 100). `make host` (the existing bit-exact golden test)
re-confirmed unaffected: `PASS — 2/2 configurations bit-exact`.

Note: the QuartzNet activation arena (`QN_ARENA_BYTES` in
`quartznet_infer.h`) caps `t_out` around 148 frames (`QN_MAX_T_OUT=128` ×
headroom) — clips longer than ~2.5s need `--seconds` truncation or the arena
needs resizing (see "SoC integration" below, which hits the same limit from
the hardware side).

Files added: `sw/tinyml_reference/mp3_to_text.py`,
`firmware/quartznet/qn_transcribe.c`. Files touched:
`firmware/quartznet/Makefile` (new `transcribe` target), `.gitignore` (the
`qn_transcribe` binary).

---

## What's still missing for a real mp3-to-text system

Two independent gaps remain, matching the two things this session explicitly
deferred:

### 1. RTL ↔ PicoRV32 SoC integration (Goal A's deferred half)

`quartznet_accel.v` — including the new table walker — has **only ever been
driven by the `rtl/tb` C++ testbench**, standing in for firmware. The
`0x20000000` MMIO region PicoRV32 firmware would actually poke exists **only**
as a C++ behavioral emulation in `sim/verilator/sim_main.cpp`
(`accel_execute()`); there is no Verilog instantiation of `quartznet_accel.v`
anywhere in `rtl/soc/picorv32_soc.v` or the Verilator sim. Concretely open:

- **Instantiate `quartznet_accel.v` into `picorv32_soc.v`** with real bus
  decode into its `mmio_we`/`mmio_addr`/`mmio_wdata`/`mmio_rdata` ports (or
  write a new C++ behavioral shim in `sim_main.cpp` in `accel_execute()`'s
  style, if a faster/lower-risk path is wanted before real RTL integration).
- **Reconcile the memory model.** `ext_mem_if.v` models two separate 1MB
  address spaces (QSPI read-only, PSRAM read/write) with a simulation-only
  backdoor; `sim_main.cpp`'s PicoRV32 model is a single flat 256KB `ram[]`
  with no QSPI/PSRAM split. Needs a decision: separate memory-mapped regions?
  DMA descriptors? Remove the backdoor and bus-master real reads?
- **PicoRV32 firmware driver.** The TinyVAD precedent
  (`firmware/picorv32_baremetal/accel.c`) stages fields over MMIO and polls —
  same pattern works here, but table-walk mode changes what firmware needs to
  do to almost nothing (stage `TABLE_BASE`/blob bases/`T_OUT` once, pulse
  `RUN_TABLE`, poll `TABLE_DONE`) versus what increment-1-only firmware would
  have needed (one `accel_run_desc()`-style call per descriptor). No
  `qn_desc`-level hook analogous to `tinyvad_conv1d_hook` exists yet in
  `quartznet_infer.h` — `qn_tile_hook` is observation-only.
- **Arena size vs RAM.** `quartznet_infer.c`'s static arena
  (`QN_ARENA_BYTES = QN_ARENA_PER_FRAME × QN_MAX_T_OUT` = 2560×128 = 320KB)
  already exceeds the 256KB `RAM_SIZE` both `sim_main.cpp` and
  `picorv32_baremetal/linker.ld` define. This needs resolving regardless of
  the mp3 pipeline specifically — either shrink `QN_MAX_T_OUT` for realistic
  utterance lengths, grow the RAM allocation, or move the arena into the same
  external PSRAM `quartznet_accel.v`'s own arena already lives in (the more
  architecturally consistent option, since the real chip's activations were
  always meant to live off-chip).
- Once wired, `mp3_to_text.py`'s blob format is *already* the right shape to
  feed this path directly — no format changes needed there.

### 2. Real trained weights (Goal B's deferred half)

Per `docs/07_quartznet_pivot.md` Stage A (still 0% implemented in code, 100%
specified in prose): NeMo `stt_en_quartznet15x5` → ONNX → ONNX Runtime static
per-channel int8 PTQ, reusing the Q31 `(q_mult, rshift)` decomposition from
`sw/tinyml_reference/export_weights.py` (a real, working example for a
*different* model, TinyVAD). Concretely open:

- **No code scaffolding exists** — no NeMo checkpoint loader, ONNX export
  script, or ORT calibration pass anywhere in the tree. Needs
  torch/onnx/onnxruntime/NeMo, none installed on this machine.
- **Calibration dataset is unspecified.** The audio front end's own finding
  (per-utterance z-score normalization pins the *input* tensor near
  standard-normal regardless of content, so the input needs no dataset
  calibration) does not extend to interior activations — those need a real
  calibration set, and neither the size nor the source (LibriSpeech subset?)
  is pinned down anywhere yet.
- **Numerical validation of the audio front end itself is still open** —
  `quartznet_audio.py`'s own header flags it as "correct shapes, correct
  scheme, plausible numbers" but never diffed against a real NeMo/torchaudio
  run end to end (neither torch nor librosa was installed when it was
  written).
- Once real weights + qparams land in the `quartznet_descriptors.py`-layout
  blob format, `qn_transcribe.c` needs **zero changes** — it already accepts
  arbitrary `quartznet_weights.bin`/`quartznet_qparams.bin` by path.

### The honest end state today

`mp3 file → quartznet_audio.py → mp3_to_text.py → int8 blob → qn_transcribe.c
→ real C interpreter → real CTC decoder → a string` all runs today, on x86,
with placeholder weights. Getting to "→ runs on the actual chip → correct
English text" needs both gaps above closed; they are independent and can be
worked in parallel (one is pure RTL/firmware, the other pure ML/Python).
