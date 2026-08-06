# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Pocket AI Voice Recorder with RISC-V TinyML Accelerator** — a 6-stage project culminating in a custom int8 MAC accelerator chip. The work flows from Pi software → TinyML C inference → PicoRV32 Verilator simulation → behavioral accelerator → RTL accelerator → RTL-to-GDS via OpenROAD-flow-scripts (ASAP7).

Full plan: `pocket_ai_voice_recorder_riscv_tinyml_plan.md`

> **Design-space optimization lives in a separate repo.** The RL/DSE engine that
> searches the ORFS flow for area/Fmax/power-optimal configs is now its own
> standalone tool: **[eda-rl](https://github.com/Shash976/eda-rl)**. To optimize
> this accelerator with it, install eda-rl and point it at a tinymac DesignSpec
> YAML with `EDA_RL_DESIGN_ROOT` set to this checkout:
> ```bash
> pip install -e <eda-rl checkout>
> EDA_RL_DESIGN_ROOT=$(pwd) \
>   eda-rl optimize --design <eda-rl>/eda_rl/designs/tinymac_accel.yaml \
>     --platform nangate45 --budget-hours 4
> ```

### Current status
- ✅ Stage 1–2: TinyVAD trained, int8 TFLite quantized, C inference matches TFLite within ±3 LSB (64/64 vectors pass)
- ✅ Stage 3: Verilator simulation working; 64/64 correct; SW baseline = ~11.2M cycles/inference (~112ms @ 100 MHz)
- ✅ Stage 4: Behavioral TinyMAC accelerator working; 64/64 correct; ~61.4K cycles/inference (~0.6ms @ 100MHz); ~182× speedup vs SW baseline (8 lanes). 16 lanes → ~46.7K cycles, ~240×.
- ✅ Stage 5: Design-space optimization — extracted to the standalone **[eda-rl](https://github.com/Shash976/eda-rl)** repo (multi-fidelity funnel optimizer over the ORFS flow; design-agnostic via a `DesignSpec` YAML). See that repo's README for the full pipeline; use the pointer above to run it against this accelerator.
- 🚧 Stage 6: Synthesizable accelerator RTL written (`rtl/accel/{int8_mac_array,requantize,tinymac_accel}.v`) + Verilator unit TB (`rtl/tb/`) bit-exact vs SW golden (45/45, LANES∈{2,4,8}, ACC_W∈{24,32}). **Full nangate45 GDS produced** via classic ORFS make flow on the company VM (`/opt/OpenROAD-flow-scripts`). LANES=4 ACC_W=24: ~19,738µm² (48% util), 230 FFs, **Fmax ≈269 MHz** (period_min 3.72ns); critical path = requantize Q31 multiply, **independent of LANES** → clean area↑/Fmax-flat Pareto. ⚠️ **Both of those facts changed in Stage 7** — requantize is now pipelined (Fmax 414.9 MHz) and the critical path moved to the LANES-*dependent* MAC accumulate path, so the Fmax-flat conclusion no longer holds and the LANES sweep must be re-run. **First asap7 GDS produced** (L4_A24 @ 1.0ns: 1433µm², Fmax 509 MHz, wns −0.96ns). Synth-only area sweep (`physical/orfs/synth_area.sh`): nangate45 L1=12.3K→L16=22.9K µm² (16× MACs, only 1.86× area). Flow files: `physical/orfs/make/{run.sh,sweep.sh,<plat>/tinymac_accel/{config.mk,constraint.sdc}}`. **Gotchas:** (a) bazel-orfs route abandoned (PyPI fetch times out); use classic make flow. (b) Yosys 0.64 asserts `genrtlil.cc:2214` on signed/unsigned mixing — NO `$signed()` on unsigned whole wires, NO signed `integer` params in unsigned exprs, NO mixed-sign `?:` branches (yosys 0.9 + Verilator lint miss these). (c) param sweep via ORFS `VERILOG_TOP_PARAMS="LANES n ACC_W w"` (chparam) + `FLOW_VARIANT` per config. **Behavioral sim matches RTL** on cycle model (`ACCEL_CH_OVERHEAD=2`: latency = `n_outputs×(ceil(K/LANES)+2)`) **and saturation order** (per-LANES-chunk, not per-MAC → acc16 accuracy is lanes-dependent, 47–58/64). Measured AVG_CYCLES: L8=61,400, L16=46,670. Remaining: realistic-clock re-sweep, asap7 sweep (first GDS done). ~~requantize pipelining~~ **done in Stage 7**. (Automated search over these configs is driven externally by the [eda-rl](https://github.com/Shash976/eda-rl) engine, which calls the same ORFS make flow.)
- 🚧 **Stage 7 — QuartzNet 15x5 ASR pivot** (branch `feat/quartznet-asr-accel`). Product change: the chip transcribes speech itself, replacing Whisper on the Pi. Design doc `docs/07_quartznet_pivot.md`; macro reference `docs/07a_sram_macro_notes.md`; setup `docs/07b_machine_setup.md`. **Done:** (a) **requantize pipelined over 2 cycles → 1.54× Fmax (268.8→414.9 MHz) for +3.3% area**, both variants measured on the same machine (reports committed in `physical/orfs/measured/`); critical path moved from the Q31 multiply to `i_in_chunk→acc`, which **is** LANES-dependent. (b) **TB widened past 8 lanes** (it had `static_assert(TB_LANES<=8)`; Verilator uses `VlWide` above 64-bit ports) — **12/12 pass, LANES∈{1,2,4,8,16,32} × ACC_W∈{24,32}**; LANES=32 is the design point. (c) **fakeram45 SRAM macro spike: full GDS, 0 DRC, WNS 0.00** — first macro integration in this repo. (d) **Software foundation: 187/187 descriptors reproduce** (`sw/tinyml_reference/quartznet_{topology,descriptors,ref}.py`). (e) ~~`firmware/quartznet/` C interpreter~~ **written and bit-exact** — table-driven over the QN_DESC format, four ops, static arena, zero tolerance: **187/187 descriptors + transcript byte-identical** on the full 15x5 and **27/27 + transcript** on the reduced config (the only one that exercises `OP_REQUANT`). Tiled T_TILE=32 (time) × DW_CH_TILE=64 (depthwise channels) with halo retention, which is 1,785 tiles at T_out=70 — so bit-exactness against the *untiled* NumPy golden is what proves the tiling. (f) ~~Stage C cycle model~~ **done — and the gate PASSES** (`sim/quartznet_cycles/quartznet_cycles.py`), driven off the interpreter's real schedule via `qn_tile_hook` → `firmware/quartznet/qn_schedule`, so cost model and interpreter cannot drift. Reproduces every committed figure (18,847,040 MACs/frame, 19.68 MB weights+qparams, 48.08 MB activations, 6.78 MB/s at 10 s, 90.65% PW). **Key numbers:** 18,847,040 params = 18.85 MB int8 = MACs *per output frame*; 942 MMAC/s for real-time; 90.6% of MACs are plain GEMM; layer-sequential dataflow with activations in external PSRAM ≈ 6.78 MB/s for a 10 s utterance. **Stage C results @ 414.9 MHz, 10 s utterance:** real-time at *every* LANES ∈ {8,16,32,64} — worst case 1.84× with a 64 KB weight buffer (QSPI-bound, flat across LANES), 3.3×→13.0× once weights stop re-streaming. LANES=32: 372.0 M cycles, 0.897 s, 10,511 MMAC/s, 79.2% array utilisation, 11.2× real-time. **Gotchas:** (a) cross-layer frame tiling is INVALID — cumulative depthwise receptive field is 4,012 frames (80.2 s); tile *within* each layer instead. (b) `ACC_W=32` is mandatory, not a swept axis (ACC_W=24 saturates for any pointwise with c_in≥260, i.e. every layer from B1 on). (c) `fakeram45_1024x32` is the densest macro at 4.005 µm²/byte — the larger `2048x39` is 12.1% *worse* per byte. (d) SRAM will be ~90% of die area, so the dominant area knob is SRAM capacity, not LANES. (e) **the on-chip WEIGHT buffer, not LANES, decides whether the part is QSPI-bound.** At 64 KB, 93/187 descriptors get re-streamed once per time tile → 282 MB/utterance, 14.3× the 19.68 MB floor, and QSPI sets the wall time at every LANES. The widest descriptor is 268,288 B, so **262 KB removes all re-streaming** and QSPI collapses to the floor (0.38 s); the design turns compute-bound. (f) SRAM ports: 8 paired `fakeram45_1024x32` bank pairs = 64 B/cycle, and a MAC eats 2 B (one activation + one weight), so 8 pairs sustain exactly LANES=32 — LANES=64 needs 16 pairs or utilisation falls to 46.3%. (g) the golden model reads every operand from channel 0 but writes at `out_off`, even for a channel-split ADD; that asymmetry *is* the format — matching it is required for bit-exactness. (h) ~~Stage D RTL increment 1~~ **done** — `rtl/accel/{addr_gen,ext_mem_if,quartznet_accel,requantize_add}.v`: a descriptor-driven sequencer dispatching OP_DW/OP_PW/OP_ADD/OP_REQUANT, reusing `int8_mac_array.v` unmodified and extending `requantize.v` (additive `out_raw` output) for ADD's per-tensor 3-multiplier mode via `requantize_add.v`. First real Verilog MMIO register file for the accelerator (previously only C++-emulated in `sim_main.cpp`); one descriptor per CMD trigger, not yet autonomous table-walking (register file is laid out for it — increment 2). **27/27 descriptors bit-exact** vs the real `quartznet_ref.py --reduced` golden at LANES∈{8,16,32}, cross-checked tile-for-tile against `firmware/quartznet/qn_schedule`'s real walk (needed because halo retention is invisible in the activations alone); **legacy TinyVAD `rtl/tb` suite still 12/12** (unmodified regression). One real RTL bug (ADD qparam fetch on the final word) found and fixed via this TB. Verilator here is 4.038, not 5.048 — `rtl/tb`/`quartznet_tb.cpp` needed a portability shim to build at all. (i) ~~mp3 audio front-end~~ **done** — `sw/tinyml_reference/quartznet_audio.py`: mp3/wav decode via `soundfile`≥0.12.1 (bundled libsndfile≥1.2 reads mp3 natively, no ffmpeg/sudo), hand-rolled polyphase resample to 16 kHz, `[time,64]` log-mel matching NeMo's actual released `quartznet_15x5.yaml` preprocessor (filterbank cross-checked vs real librosa, ~1e-9 max diff), placeholder int8 calibration. Finding: QuartzNet normalizes **per-utterance** (z-score over the whole clip), so the front end cannot stream frame-by-frame like TinyVAD's — a full utterance must land before the first frame quantizes. (j) ~~Stage D increment 2 — autonomous descriptor-table walking~~ **done, RTL-correctness scope** (full writeup `docs/07c_stage_d_increment2_and_mp3_pipeline.md`) — `quartznet_accel.v` gained `TABLE_BASE`/`W_BLOB_BASE`/`QP_BLOB_BASE`/`DESC_IDX` registers (idx 29–32), `CTRL.RUN_TABLE`, `STATUS.TABLE_DONE`/`err_bad_in_off`, and a fetch FSM (`S_HDR→S_BUFTBL→S_DESC→S_POPULATE→S_KICK`) that walks the in-ROM table autonomously, deriving BASE/PITCH/blob-offsets on-chip instead of via software — purely additive around the unmodified per-op execution core. **27/27 descriptors bit-exact in table-walk mode at LANES∈{8,16,32}** (`rtl/tb/quartznet_tb.cpp`'s new `run_table_test`); single-descriptor mode, the 87/87 tile-walk check, and the legacy TinyVAD 12/12 suite all unaffected. Two real bugs found+fixed via this TB: a multi-driver conflict (`S_POPULATE` and the MMIO path both wrote the same config registers from two different always-blocks — merged into one) and a one-cycle race between `addr_gen`'s `start` and `S_ELEM`'s `ag_busy` check (fixed with a new `S_KICK` handoff state mirroring `S_IDLE`/`start_pulse`'s alignment); a third apparent bug was a **test-methodology** bug, not RTL — comparing all descriptors' PSRAM output *after* the whole table finished reads corrupted data, since `BUF_A`/`BUF_B` are ping-pong buffers later descriptors legitimately overwrite (fixed by snapshotting each descriptor's slice via `DESC_IDX` polling as it completes). **Scope explicitly excludes** wiring `quartznet_accel.v` into `rtl/soc/picorv32_soc.v`/`sim/verilator/sim_main.cpp` — verified only via the standalone `rtl/tb` harness; the real SoC integration is unstarted (see item 1 below). (k) ~~mp3-to-text mechanical pipeline~~ **done** — `sw/tinyml_reference/mp3_to_text.py` wires the previously-disconnected `quartznet_audio.py` front end into a real blob (+`t_out` sidecar); `firmware/quartznet/qn_transcribe.c` (`make transcribe [MP3=...]`) runs it through the interpreter and real CTC decoder end to end, producing an actual (gibberish, since weights are still seeded-random) transcript string; verified against both build configs and multiple clip lengths, `make host` unaffected. **Remaining:** real int8 weights via NeMo→ONNX→ORT PTQ (deferred — RTL prioritized over software accuracy this round; zero code scaffolding exists yet, see item 2 below); LANES resweep and macro orientation (both need OpenROAD, not built here); Stage D increment 2's own SoC-integration half (item 1 below). (l) ~~Gap 1 (SoC integration, D1–D6)~~ **and Gap 2 A0/A2 done, A1 done** — full detail `docs/07d_soc_integration_and_gap2_start.md` and this section's item 1. **A1 (front-end numerical validation, gate G2.1):** `sw/tinyml_reference/quartznet_audio_validate.py` diffs `quartznet_audio.py`'s `extract_logmel()` against a direct torch.stft transcription reading the checkpoint's own `preprocessor.featurizer.{window,fb}` buffers (the exact tensors the trained model saw, not assumed config defaults) on 22 real LibriSpeech dev-clean clips: max fp32 log-mel diff **1.015e-04** (gate <1e-3), int8-quantized agreement **100.000%** within 1 LSB (gate ≥99.9%); a 12-mutation battery confirms the gate has teeth (every mutation caught at ≥1.4e-2, two-plus orders above threshold). **Four constants remain unverified by this gate** (preemph=0.97, mag_power=2.0, log_zero_guard=2⁻²⁴, norm eps=1e-5 — none are checkpoint tensors, so a shared error in the front end and its validator would pass silently); only Step 3's FP32 WER against NeMo's published number checks them independently. `quartznet_audio.py`'s header updated accordingly (no longer "PLUMBING ONLY, NOT ACCURACY-VALIDATED"). (m) ~~A3 (FP32 baseline + WER, gate G2.3)~~ **done, full writeup `docs/07e_gap2_a3_fp32_baseline.md`** — found and fixed a real bug in already-merged A2 code along the way: `quartznet_nemo_export.py`'s `BN_EPS=1e-5` was wrong (NeMo hardcodes `eps=1e-3` in `JasperBlock`, does not use PyTorch's BatchNorm1d default as the old comment assumed), inflating every folded BN scale ~2-4× and overflowing logits to ~1e31 before the fix (correct-eps logits absmax ~38, exact transcript). G2.2 gained a numeric guard (`|weight|`/`|bias|` < 100 per folded layer) so this class of bug can't recur silently — its other checks are all shape/name checks that passed identically either way. Also found: `expand()` gave C4 (the decoder) a spurious `relu=True` (NeMo's `ConvASRDecoder` has no activation) — latent in fp32 (byte-identical transcript either way, since ReLU never flips an already-positive argmax) but will matter for A5's int8 clamping; fixed now. New `sw/tinyml_reference/{quartznet_fp32,quartznet_wer,quartznet_run_fp32}.py`: an `nn.Module` forward-pass graph built directly off `quartznet_topology.expand()` (not the post-split packed table — `layer_id` numbering diverges from `folded_weights.npz`'s after C3), a hand-rolled Levenshtein WER scorer, and the gate driver. **The plan's original gate target was also wrong** — "3.90% published" belongs to a *different* NGC checkpoint (`quartznet_15x5_ls_sp`, LibriSpeech-only, shipped as loose `.pt` files) than the one this repo actually uses (`stt_en_quartznet15x5`, 7,057h multi-domain, `.nemo` tarball). Re-gated against `stt_en_quartznet15x5`'s own published 4.4% dev-clean WER: **measured 4.4392%** (2,415/54,402 words, full 2703-utterance corpus) — **0.04% absolute** from published, **G2.3: PASS**. Compute: CPU only (~10.5 min full dev-clean on this machine's 4 logical CPUs, 62× real-time) — GPU considered and rejected as not worth the dependency churn for the time saved. (n) ~~A4 (ORT static per-channel int8 PTQ, gate G2.4)~~ **done, full writeup `docs/07f_gap2_a4_int8_calibration.md`** — new `sw/tinyml_reference/{quartznet_calibrate,quartznet_run_int8_ort}.py`: exports `quartznet_fp32.QuartzNetFP32` to ONNX (legacy exporter, `dynamo=False` — `onnxscript` isn't installed), verifies the export round-trips exactly against direct PyTorch (transcripts identical), calibrates on 200 dev-clean utterances (5/speaker × 40 speakers, ≤10s by filtering not truncating — truncating would normalize `extract_logmel`'s per-feature z-score over a window never actually seen), and runs `quantize_static` (QDQ, per-channel, `QInt8`/`QInt8`, `CalibrationMethod.Percentile` at **99.999%**, not the plan's 99.99% — measured 10× more gate margin, since clipping compounds across 15 residual blocks; 99.99% stays selectable for A7). **G2.4: PASS** both splits — dev-clean int8 4.5678% vs fp32 4.4392% (delta 0.1287%, gate 0.30%), test-clean int8 4.5002% vs fp32 4.4716% (delta 0.0285%, informational/ungated per G2.4's definition being an FP32-dev-clean delta). Two `extra_options` are mandatory, not tuning, found by running not reading docs: `CalibStridedMinMax=1` (Percentile's default calibrator buffers every intermediate tensor for the whole set and crashes on this model's variable-length utterances without it) and `MinimumRealRange=1e-3` (599 near-zero-BN-gamma channels would otherwise fold to int32 biases within a few percent of overflow — harmless to ORT, fatal to the firmware's real int32 accumulator). **Scope note:** this is a QDQ-simulated int8 WER (31/171 convs + 14/15 Adds run fp32-accumulator DequantizeLinear→op→QuantizeLinear rather than a fused int8 kernel — inputs/weights/outputs are still genuinely int8, only the accumulator differs), validating calibration quality, not the firmware's bit-exact int32 path — that's A6. Handoff for A5: `build/quartznet_int8/qparams_ort.npz`, all 171 weight-bearing descriptors mapped by `layer_id` via `model.conv_ix` (never by ONNX bias-initializer name — `torch.onnx.export` dedups identical tensors, e.g. every all-zero depthwise bias collapses onto one shared initializer). `OP_ADD` deliberately left unextracted — the real per-branch scale-ratio normalization (measured `s_main/s_res` spans 0.75–3.82 across the 15 real Adds) is A5's own work.

---

## Environment Split

| Task | Machine |
|------|---------|
| Python ML (train, convert, export) | **Windows** (has GPU, PyTorch, TFLite) |
| Hardware (Verilator, RV32 cross-compile, ORFS) | **WSL** (Ubuntu on the same machine) |

The repo lives on Windows at `C:\Users\shash\Desktop\Code\voiceAI`. WSL has a **separate copy** at `~/voiceAI` (`/home/shashg/voiceAI`) — NOT a symlink to `/mnt/c/...`. Always edit files in the WSL copy when making hardware changes; sync back to Windows manually (or via git).

> **The split above describes the original dev box.** Stage 7 was done on a plain Linux
> machine with **no sudo**, where everything (Python *and* hardware tooling) runs in one
> place. See `docs/07b_machine_setup.md` for the portable setup. Two things to know before
> starting work anywhere new:
> - **Python goes in a conda env**: `conda env create -f environment.yml && conda activate voiceai`.
> - **A RISC-V toolchain may be missing.** It is needed only for firmware and the
>   full-system PicoRV32 sim — the unit TB, ORFS flows, and the whole Python chain do not
>   need it. The `sim/verilator/sim_main.cpp` latency change from Stage 7 (+1 drain cycle per
>   op) is verified against the RTL by `rtl/tb`, **and as of 2026-08 also end-to-end**: on a
>   machine with `riscv64-linux-gnu-gcc` present, `make -C sim/verilator run` completes
>   64/64 correct, `avg_cycles=61769` (LANES=8, ACC_W=32). That is 369 cycles above the
>   pre-Stage-7 documented `L8=61,400` baseline — far more than the `+1`/call change alone
>   predicts (4 accel calls/inference → ~+4 cycles) — but not investigated further; likely a
>   different toolchain built the original baseline binary, not a drain-cycle bug (`rtl/tb`
>   already proves that formula bit-exact against real RTL independently). Re-confirm on a
>   machine without this toolchain if the gap needs to be closed.

> **Toolchain state of the current machine (measured 2026-08, Stage 7 software work).**
> Probe before assuming — this box differs from both boxes described above.
> - **No conda.** System `python3` 3.10.12 + `numpy` 1.21.5 is enough for the entire pure
>   software chain: `quartznet_{topology,descriptors,ref}.py`, the goldens, and
>   `sim/quartznet_cycles/`. Only the NeMo/ONNX PTQ work needs the conda env.
> - **`riscv64-linux-gnu-gcc` 11 IS present** and targets `-march=rv32imc -mabi=ilp32`;
>   `firmware/picorv32_baremetal` builds clean (firmware.bin, 144,324 B). This contradicts
>   the "currently missing" note in `docs/07_quartznet_pivot.md` — and the full-system sim
>   **is** runnable here: `make -C sim/verilator run` completes 64/64 correct (see the
>   drain-cycle note above). `riscv32-unknown-elf-gcc` is absent; use the
>   `CROSS ?= riscv64-linux-gnu` default.
> - **No OpenROAD binary.** `~/OpenROAD-flow-scripts` is checked out but its build stopped
>   at 81% (`build_openroad.log`) and no `openroad` executable exists, so every ORFS item is
>   blocked here. System `yosys` is **0.9**, which is exactly the version that *misses* the
>   signed/unsigned assert 0.64 catches — do not treat a clean 0.9 run as synthesis passing.
> - **Verilator is 4.038**, not the **5.048** `docs/07b_machine_setup.md` documents; `rtl/tb`
>   and `sim/verilator` were not re-run here.

---

## Build Commands

All hardware/firmware commands run **in WSL**.

### Generated headers (run once after model changes)
```bash
# Windows (Python venv active)
python sw/tinyml_reference/export_weights.py     # → firmware/tinyengine_port/tiny_vad_weights.h
python sw/tinyml_reference/gen_test_vectors.py   # → firmware/tinyengine_port/tiny_vad_test_vectors.h
```

### Firmware (cross-compile for RV32)
```bash
cd firmware/picorv32_baremetal
make              # → firmware.bin
make size         # section sizes
make disasm       # disassembly (grep for FP instructions)
make clean
```

### Host-side C inference test (x86, fast sanity check)
```bash
cd firmware/tinyengine_port
make host         # gcc x86 binary
./test_infer_host # should print "64/64 passed"
```

### QuartzNet interpreter + Stage C cost model (Stage 7, x86 host — no RV32/Verilator/ORFS)
```bash
cd firmware/quartznet
make goldens      # regenerate both reference configs into build/ (python3 + numpy)
make host         # build + run the bit-exact test → "PASS — 2/2 configurations bit-exact"
make schedule     # build ./qn_schedule, the tile-schedule dumper

python3 sim/quartznet_cycles/quartznet_cycles.py            # 10 s utterance, LANES 8/16/32/64
python3 sim/quartznet_cycles/quartznet_cycles.py --help     # --seconds/--lanes/--banks/--wt-buf-kb
```
The goldens under `build/` are gitignored and reproducible from the seed — regenerate, never
commit. The cost model shells out to `qn_schedule`, which drives the *real* interpreter's tile
walker, so the schedule it prices cannot drift from the one the C executes.

### Verilator simulation (Stage 3)
```bash
cd sim/verilator
make check-deps   # verify prerequisites
make run          # build + compile firmware + run PicoRV32 simulation
make vcd          # same + VCD waveform dump → sim_out.vcd
make clean
```

Simulation prints CSV to stdout, stats to stderr. Expected output columns: `vec,label,result,correct,logit0,logit1,cycles`.

### ML training & conversion (Windows)
```bash
python train_tiny_vad.py         # → tiny_vad_best.pt, tiny_vad.onnx
python convert_to_tflite.py      # → tiny_vad_int8.tflite
```

---

## Architecture

### Data flow (end-to-end)
```
Audio (16 kHz mono)
  → extract_logmel() [speech_simulator.py]
  → int8[49×40] log-mel tensor
  → TinyVAD (speech/silence) → prob[1] > 0.5 → speech detected
  → if speech: whisper.cpp → transcript
```

### Quantization scheme
- **Input**: `float = INPUT_SCALE * (int8 − INPUT_ZP)`
- **Weights**: per-channel int8, scale extracted from TFLite `quantization_parameters.scales`
- **Requantization**: `real_mult = scale_in × weight_scale / scale_out` decomposed to Q31 `(q_mult, rshift)` pair where `shift` can be negative (left shift)
- `requantize(x, q_mult, shift)`: int64 accumulation, handles `shift < 0` via `val <<= (−shift)`

### Tensor layout throughout
All tensors use **[time, channel]** order (TFLite NHWC convention), not PyTorch's [channel, time]. This is critical — past layout bugs caused completely wrong outputs.

### TinyVAD model dimensions
| Layer | Input | Output |
|-------|-------|--------|
| Conv0 (k=5,s=2,p=2) | [49, 40] | [25, 32] |
| Conv1 (k=3,s=2,p=1) | [25, 32] | [13, 64] |
| GlobalAvgPool | [13, 64] | [64] |
| FC0 | [64] | [32] |
| FC1 | [32] | [2] (logits) |

Static scratch buffers: buf0[800], buf1[832], buf2[64], buf3[32] — ~2 KB total, no dynamic allocation.

### Memory map (Verilator sim)
| Address | Purpose |
|---------|---------|
| `0x00000000–0x0003FFFF` | 256 KB RAM (code + data + stack) |
| `0x10000000` | UART TX (write byte → stdout) |
| `0x10000004` | SIM_EXIT (write → halt sim) |
| `0x20000000–0x20000FFF` | TinyMAC accelerator registers (Stage 4) |

PicoRV32 resets to `0x00000000`. Stack grows down from `0x00040000`.

### PicoRV32 parameter names
The correct parameter name is `COMPRESSED_ISA` (not `ENABLE_COMPRESSED`). Other used params: `ENABLE_MUL`, `ENABLE_FAST_MUL`, `ENABLE_DIV`, `ENABLE_COUNTERS`, `REGS_INIT_ZERO`. `ENABLE_DIV` must be **1** — `global_avg_pool` uses a `div` instruction.

### Firmware build flags (critical)
The riscv64-linux-gnu toolchain defaults to PIE mode even with `-nostdlib`, causing GOT-indirect loads for linker symbols like `_stack_top`. Required flags to prevent this:
- `-fno-pic -fno-pie` in CFLAGS — forces direct `auipc+addi` addressing, no GOT
- `-no-pie -Wl,--build-id=none` in LDFLAGS — suppresses PT_PHDR and `.note.gnu.build-id` sections that would push `.text` away from address 0

### Verilator simulation loop
Combinatorial memory (0-wait-state): on negedge, present `mem_rdata` and assert `mem_ready`; CPU latches on posedge. This means 1 clock per memory transaction.

---

## Artifact Dependency Chain

```
train_tiny_vad.py
  → tiny_vad_best.pt
      → convert_to_tflite.py
          → tiny_vad_int8.tflite
              → export_weights.py → tiny_vad_weights.h
              → gen_test_vectors.py → tiny_vad_test_vectors.h
                  → firmware/picorv32_baremetal/ (Makefile)
                      → firmware.bin
                          → sim/verilator/sim_main.cpp → simulation
```

Both `tiny_vad_weights.h` and `tiny_vad_test_vectors.h` are **auto-generated** — do not edit by hand.

---

## Open work / next steps

### Stage 7 (current — branch `feat/quartznet-asr-accel`)
1. ~~Stage D increment 2's SoC-integration half~~ **done** — real Verilog bus decode (`rtl/soc/qn_soc.v`), the `MEM_*` host-access port, and firmware driver all landed (`docs/07d_soc_integration_and_gap2_start.md`, D1–D6, merged `381890d`); **the one item left, a full-config RTL demo, is also now done** — `MODE=rtl CONFIG=full` (real `qn_soc.v`, all 187 descriptors) completes in ~2m37s at `T_OUT=70` (no scope-down needed — the earlier "too slow" read was a `QSPI_BYTES`/`PSRAM_BYTES` Verilate-parameter bug, not a throughput wall; see `docs/07d`'s D6 correction), transcript byte-identical to the shim and the x86 golden. Gap 1 is closed. Remaining work is Gap 2 (real trained weights: A1 front-end validation, A3–A7 FP32/int8 WER + export), tracked as a stacked PR sequence starting from `feat/quartznet-gap1-rtl-demo` — see `~/.claude/plans/gentle-baking-pelican.md`.
2. **Real int8 weights** — NeMo `stt_en_quartznet15x5` → ONNX → ONNX Runtime static per-channel PTQ. *Deferred:* RTL was explicitly prioritized over software accuracy this round; needs torch/onnx/onnxruntime/NeMo, none installed here, and it changes no tensor shape or integer path already validated — everything so far runs on *seeded random* weights at the true shapes (including the new `make transcribe` mp3-to-text pipeline, Stage 7 (k) above — real weights are the only thing standing between it and an actually-legible transcript). Q-ASR measured only +0.29% WER for W8A8, so this is expected to be low-risk. The audio front-end (`quartznet_audio.py`) found the *input* tensor needs no dataset calibration (per-feature normalization pins it near-standard-normal regardless of clip) — PTQ effort belongs on interior activations, not the input. The calibration dataset for those interior activations is still unspecified anywhere (no size, no source named). Once real weight/qparam blobs land in the existing `quartznet_descriptors.py` layout, `firmware/quartznet/qn_transcribe.c` needs zero changes.
3. **Re-run the LANES sweep** — the pre-Stage-7 conclusion "area rises with LANES, Fmax stays flat" is now void; the critical path moved onto the LANES-dependent MAC accumulate path. *Deferred:* needs OpenROAD, which is not built on this machine (see Environment Split).
4. **Macro orientation experiment** — configs staged at `physical/orfs/make/nangate45/sram_spike/config_{mirror,outward,r0grid}.mk`. 4 macros routed with 0 DRC but that is too few to expose the west-edge-pin problem; test before committing to an ~18-macro floorplan. *Deferred:* same, needs OpenROAD.
5. **`ext_mem_if.v` is deliberately unoptimized** — one outstanding request per channel, sequencer blocks on every response, no prefetch/compute overlap. Its cycle counts (e.g. 3.4M for the *reduced* model at LANES=32) are not comparable to the Stage C cost model's 372M for the full 15x5 and should not be used for cycle estimates. Overlapping weight prefetch against activation reads is the obvious first optimization and needs no interface change.

### Carried over from Stage 6
6. **Realistic-clock re-sweep** at a clock near the (now 2.41 ns) critical path.
7. **ASAP7 sweep** — first GDS exists (L4_A24 @ 1.0 ns); next is ~12 configs at 0.8–1.2 ns.
8. **`physical/orfs/synth_area.sh` is not portable** — hardcodes `$HOME/OpenROAD-flow-scripts` and a bare `yosys`. Use `run.sh` (honours `ORFS_DIR`) meanwhile.

### Optimizer
The design-space optimizer lives in **[eda-rl](https://github.com/Shash976/eda-rl)**; its roadmap is tracked there. Note Stage 7 changes its search space: SRAM capacity, not LANES, is now the dominant area knob.
