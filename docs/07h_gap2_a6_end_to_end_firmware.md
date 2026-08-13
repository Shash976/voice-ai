# 07h — Gap 2 A6: end-to-end real-weight transcription through the firmware

Stage 7 Gap 2's A6: `make transcribe` on a real mp3 clip with real weights,
plus a full-corpus WER measurement through the **firmware's own C
interpreter** (`quartznet_infer.c`) — not ORT, not PyTorch. This is
specifically the bit-exact int32-accumulator path, the whole reason A6 is a
separate gate from A4's QDQ-simulated int8-ORT number.

## Result: G2.6 PASSES

```
$ make -C firmware/quartznet transcribe-real MP3=<real clip>
  transcript (159 symbols): "he hoped there would be stew for dinner turnips
  and carrots and bruised potatoes and fat mutton pieces to be ladled out in
  thick peppered flower fattened sauce"
```

Ground truth: *"HE HOPED THERE WOULD BE STEW FOR DINNER TURNIPS AND CARROTS
AND BRUISED POTATOES AND FAT MUTTON PIECES TO BE LADLED OUT IN THICK PEPPERED
FLOUR FATTENED SAUCE"* — 2 errors / 29 words, from a genuine lossy-encoded
mp3, through `quartznet_audio.py`'s mp3 decode + polyphase resample, the
real calibrated weights, and the firmware's own CTC decoder.

```
$ python3 sw/tinyml_reference/quartznet_run_firmware.py --split test-clean
...
firmware WER 4.5002% (2,366 errors / 52,576 words, 2620 utts) -- delta
0.0000% from int8-ORT 4.5002%, within 0.50%
2620/2620 utterances scored
G2.6: PASS
```

**Delta is exactly 0.0000%** — the firmware's aggregate WER lands on the
same number as A4's ORT measurement. Verified this is a genuine coincidence,
not a copied/cached file: **460 of 2620 hypotheses differ** between the two
paths (ORT's QDQ-format fp32 accumulator on 31/171 convs + 14/15 Adds vs the
firmware's real int32 accumulator everywhere), with the expected
int32-vs-fp32-rounding flavor of differences —
`reigned`→`reined`, `mutton`→`muttone`, `clasp`→`claspe`,
`zavir`↔`xavir`/`zevier`↔`zavier` — that happen to net out to the identical
error count over the full corpus.

## A real bug found: `CLAUDE.md`'s "zero changes" claim was wrong

The prior session's note (`CLAUDE.md`, Stage 7 item 2) claimed *"Once real
weight/qparam blobs land in the existing `quartznet_descriptors.py` layout,
`firmware/quartznet/qn_transcribe.c` needs zero changes."* Narrowly true for
the C argv plumbing — `qn_transcribe build/quartznet_real <blob> <t_out>`
does work unmodified against A5's real blobs. Four things were actually
wrong or missing:

**`mp3_to_text.py` quantized with a placeholder scale — real, measured
+0.28% absolute WER, more than half of G2.6's entire 0.5% gate band.**
`quartznet_audio.audio_file_to_int8()` was called with no `scale=`/
`zero_point=`, so it fell through to `calibrate(feat)` (a per-clip
percentile heuristic) and `IN_ZP_DEFAULT=-20` — the seeded-random
placeholder's own convention, appropriate only because seeded-random qparams
have nothing real to match anyway. The real calibrated model wants the
*fixed*, model-specific `in_scale=0.038837`/`in_zp=-44` baked into
`quartznet_desc.bin` by A5's export. Measured on a real mp3:
`calibrate()` returns 0.046260 vs the real 0.038837 — the model then
evaluates `0.8395·feat + 0.9321` instead of the real feature, a ~16%
amplitude compression plus a +0.93σ DC offset on every mel bin of a
z-scored feature. Measured cost, 200 test-clean utterances, same binary,
same weights, only the input quantization differing:

| input quantization | WER (200 utts) | errors |
|---|---|---|
| real `in_scale`/`in_zp` from `quartznet_meta.json` | 3.8412% | 178/4634 |
| placeholder `calibrate()` + `IN_ZP_DEFAULT` | 4.1217% | 191/4634 |

A single short clip still transcribed correctly either way — exactly why
this would have slipped through a smoke test rather than a full-corpus gate.
Fixed: `mp3_to_text.py` gains `--model-dir`, reading `in_scale`/`in_zp` from
`<dir>/quartznet_meta.json` (A5's sidecar) when pointed at a real calibrated
model; the seeded-random path (no `--model-dir`) is unchanged.

**`make transcribe` couldn't be pointed at real weights at all.** The
existing recipe ran `./qn_transcribe` with no arguments, which takes the
hardcoded seeded-random-dirs sweep. New `make transcribe-real` target.

**`QN_MAX_T_OUT=128` capped the host build at ~3 seconds of audio.** The
activation-arena bound is `2205·t_out ≤ 2560·QN_MAX_T_OUT`
(`QN_ARENA_PER_FRAME=2560` vs the real ~2205 B/frame used), so the stock
128 actually tops out around t_out=148 (~3.0s), not literally 128 frames as
the header comment implied. `test-clean`'s longest utterance is t_out=1748
(34.96s). Verified the bound fails *cleanly* (an explicit error, not memory
corruption) before raising it. Fixed: host build default raised to 2048
(5.24MB BSS, host-only — never touches the RV32 firmware build's own much
shorter `T_OUT`), overridable via `QN_MAX_T_OUT=`.

**The output unconditionally claimed "gibberish expected."** Made the
caveat conditional on the seeded-random sweep path (where it's true); real
calibrated weights get a plain transcript line, plus a new
`TRANSCRIPT\t<text>` machine-readable line every run prints, for
`quartznet_run_firmware.py` to parse.

## Architecture: why this isn't just A5's export loop re-run 2620 times

`quartznet_weights.bin`/`quartznet_qparams.bin`/`quartznet_desc.bin` are
**model-level** artifacts — identical across every utterance; only the input
blob (and `t_out`) vary per clip. Measured: a full `quartznet_export_int8.py`
run takes 1.29s: re-running it per utterance would waste ~56 minutes for
zero benefit. `quartznet_run_firmware.py` instead does only the
input-dependent work itself (`load_audio` → `extract_logmel` →
`quantize_features`, using the real `in_scale`/`in_zp` from
`quartznet_meta.json`) and invokes the already-built `qn_transcribe` binary
once per utterance via `subprocess`, 4 workers in parallel.

**Subprocess-per-utterance over a batched C harness or a `.so`/ctypes
binding**: process-start + loading the 18.85MB weight/table/qparam blobs
measured at ~40ms/call — under 2% of the ~39-minute full run at 4 workers,
not worth a new artifact and ABI coupling to buy back. The stronger property
this keeps: the transcripts this gate scores are produced by the **exact
same `qn_transcribe` binary** `make transcribe-real` runs, not a parallel
reimplementation of the same logic.

`t_out = t_in // 2` (floor, not ceil) — the descriptor format requires
`t_in == 2·t_out` exactly (`quartznet_ref.py` rejects a non-multiple), so
floor is the zero-invention choice; ORT/ONNX internally uses ceil on the
same input, meaning up to 1294/2620 utterances differ from the ORT path by
exactly one trailing 10ms mel frame. Unmeasurable in the WER (0.0000% delta
either way) and not the source of any traced transcript diff.

## New / changed files

- **`sw/tinyml_reference/quartznet_run_firmware.py`** (new) — the G2.6 gate
  driver, following the established JSONL + progress + gate-print pattern,
  parallelized over `qn_transcribe` subprocess calls.
- **`sw/tinyml_reference/mp3_to_text.py`** — `--model-dir` flag (the real
  bug fix above).
- **`firmware/quartznet/qn_transcribe.c`** — conditional gibberish caveat,
  machine-readable `TRANSCRIPT\t` line.
- **`firmware/quartznet/Makefile`** — `QN_MAX_T_OUT`/`QN_OPT` build knobs
  (`-O3 -march=native` measured **bit-identical** to `-O2` over 200 real
  utterances — pure speed, not a behavior change; `-O2` stays canonical),
  new `transcribe-real`/`wer-firmware` targets.
- **`CLAUDE.md`** — corrected the stale "zero changes" claim.

## Verification

```
$ make -C firmware/quartznet transcribe-real MP3=<real clip>
...
PASS (real transcript, see above)

$ python3 sw/tinyml_reference/quartznet_run_firmware.py --split test-clean
...
G2.6: PASS

$ make -C firmware/quartznet host
...
PASS — 2/2 configurations bit-exact   # unaffected regression

$ make -C firmware/quartznet host-real
...
PASS — 1/1 configurations bit-exact   # unaffected regression
```
