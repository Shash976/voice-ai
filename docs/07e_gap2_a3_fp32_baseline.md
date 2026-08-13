# 07e — Gap 2 A3: FP32 baseline forward pass + WER, and a real BN_EPS bug

This session built Stage 7 Gap 2's A3 (`docs/07d`'s "What's left" — the FP32
forward-pass graph + WER gate) and, in the process, found and fixed a real
numerical bug in A2's already-merged BatchNorm-folding code. This doc records
both: the bug (because it explains a still-open question about `docs/07d`'s
"None merged yet" state and the exact gate G2.3 now checks), and the new A3
implementation.

---

## The bug: `BN_EPS` was wrong by 100x

`sw/tinyml_reference/quartznet_nemo_export.py` (Gap 2 A2, merged in PR #5)
folded every BatchNorm layer using `BN_EPS = 1e-5`, commented as "PyTorch
BatchNorm1d default; model_config.yaml does not override it." That reasoning
is inverted: NeMo does not use PyTorch's default at all. `JasperBlock`'s
`_get_conv_bn_layer` (`nemo/collections/asr/parts/submodules/jasper.py`,
v1.23.0) hardcodes `nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.1)` —
which is exactly why `model_config.yaml` never mentions it: it isn't a config
knob, it's baked into the block builder.

This is not a rounding-level detail. The checkpoint's own `running_var`
values are *below* `1e-3` (e.g. `encoder.encoder.8.mconv.12.running_var.mean()
== 5.98e-05`), so with the wrong `eps=1e-5` the eps term is negligible and
every folded layer's effective scale is inflated ~2-4x. Compounded
geometrically over ~80 BatchNorm layers, this overflows: forward-pass logits
reach an absmax of **~2.8e31** with the wrong eps, versus **~38** with the
correct one. The wrong-eps transcript is pure noise
(`"  s o oa evn   ti igzeleoi e  "`); the correct-eps transcript on the same
utterance is exact: `"he hoped there would be stew for dinner turnips and
carrots and bruised potatoes and fat mutton piec..."`.

**Why A2's own gate (G2.2) didn't catch this:** G2.2 only checks parameter
count, tensor-name binding, and unused-tensor leftovers — all shape/name
checks that pass identically whether `BN_EPS` is right or 100x wrong. Only a
real forward pass (A3) exposes a scale bug. Fixed alongside the `BN_EPS`
correction: G2.2 now also asserts every folded layer's `|weight|`/`|bias|`
stays under a sane magnitude (100), specifically to catch this class of bug
without needing a downstream WER number.

**Action taken:** `BN_EPS` corrected to `1e-3` with the real citation;
`build/quartznet_nemo/folded_weights.npz` regenerated (gitignored build
product, not committed — regenerate via
`python3 sw/tinyml_reference/quartznet_nemo_export.py stt_en_quartznet15x5.nemo`).
G2.2 still passes with the new numeric guard.

---

## A second, latent bug found alongside it: C4's spurious ReLU

`quartznet_topology.expand()` computed `relu = not (spec.residual and
last_repeat)`. C4 (the decoder, `residual=False`) therefore got `relu=True`
— but NeMo's `ConvASRDecoder` is a bare `Conv1d(1024, 29, 1, bias=True)`
feeding CTC directly, no activation after it at all.

**Ablation, full dev-clean, 2703 utterances:** with vs. without the spurious
ReLU on C4's output, the greedy transcript is **byte-identical** (WER
4.4392% either way) — in fp32, ReLU never flips the argmax when the winning
logit is already positive, which it always was on real data. So this bug was
never visible in fp32 WER. It will matter for A5 (int8): `_store()` clamps at
`out_zp`, and any frame where the true max logit is negative would
incorrectly collapse to a tie resolved at index 0 (`' '`) instead of being
left alone. Fixed now (`expand()`'s last descriptor is asserted `relu=False`)
rather than deferred to A5, since it's a one-line fix already validated not
to disturb anything already passing.

A third, unrelated fix landed alongside these two: `quartznet_ref.ctc_greedy`
cast its input to `int32` before `argmax` — harmless for its original int8
caller (already representable) but silently truncated fp32 logits to
integers, corrupting the argmax. Removed the cast; `argmax` is dtype-agnostic
and both callers now share one decoder.

---

## A3: which published number G2.3 actually gates against

The plan's original gate — "FP32 WER within 0.3% absolute of NeMo's
published 3.90%" — named the wrong checkpoint. **3.90%** (test-clean, greedy,
no LM) is `quartznet_15x5_ls_sp`'s number: a LibriSpeech-only checkpoint
shipped as two loose `.pt` files (`JasperEncoder`/`JasperDecoderForCTC`), not
a `.nemo` archive, and the one the original QuartzNet paper's table cites.

This repo uses **`stt_en_quartznet15x5`** instead — deliberately, since A2's
`.nemo`-tarball loader is built around it, and it's the more realistic
checkpoint for a device that has to transcribe real-world speech (trained on
7,057h across LibriSpeech + WSJ + Fisher + Switchboard + Common Voice + NSC
Singapore, not just audiobooks). Its own NGC model card publishes **4.4% WER
on LibriSpeech dev-clean** (11.3% dev-other).

**Measured** (`sw/tinyml_reference/quartznet_run_fp32.py --split dev-clean`,
full 2703-utterance corpus): **4.4392%** (2,415 errors / 54,402 words) —
**0.04% absolute** from the published 4.4%. Far inside any reasonable gate,
and strong end-to-end validation of the whole chain: front end (A1) + this
forward-pass graph + BN-folded weights (now correct) + CTC decode + the new
WER scorer.

**Revised gate: G2.3 = FP32 WER on `dev-clean` within 0.3% absolute of
`stt_en_quartznet15x5`'s published 4.4%, i.e. `[4.10%, 4.70%]`.**
`test-clean` is reported alongside as an informational, *ungated* datapoint —
no published number exists for this specific checkpoint on that split to
compare against.

```
$ python3 sw/tinyml_reference/quartznet_run_fp32.py --split dev-clean
...
FP32 WER 4.4392% (2,415 errors / 54,402 words, 2703 utts) within 0.30% of
the published 4.40% (stt_en_quartznet15x5, dev-clean)
2703/2703 utterances scored
G2.3: PASS
```

(One caveat on the source: the NGC card's Performance section is prefixed
"measuring using Character Error Rate," which is boilerplate copy-pasted
across NeMo NGC cards and wrong here — 4.4% CER would imply ~1.5% WER,
implausibly good for this class of model, and the 4.4/11.3 pair matches the
WER-shaped pattern of `quartznet_15x5_ls_sp`'s own card (3.83/11.08 WER). Our
independent 4.4392% measurement settles it: it's WER.)

---

## New files

- **`sw/tinyml_reference/quartznet_fp32.py`** — `QuartzNetFP32(nn.Module)`,
  built directly from `quartznet_topology.expand()`'s raw output (*not* the
  post-`split_wide_layers()` packed table — the channel split of C3's
  512→1024 pointwise renumbers every `layer_id` from 184 onward, so the
  packed table's numbering and `folded_weights.npz`'s numbering diverge
  after C3; verified first divergence at `layer_id=184`). Real `nn.Conv1d`
  submodules (not raw `F.conv1d` calls) so each weight is a named
  initializer once this doubles as A4's ONNX export target — anonymous
  `Constant` nodes would force matching PTQ calibration output back to a
  `layer_id` by shape, which is ambiguous (many layers share e.g.
  `(256,256,1)`).
- **`sw/tinyml_reference/quartznet_wer.py`** — hand-rolled Levenshtein WER
  (no `jiwer` dependency, matching the repo's existing hand-rolled-DSP
  convention) + a LibriSpeech `.trans.txt`/`.flac` corpus loader, shared by
  every WER gate from here through A7.
- **`sw/tinyml_reference/quartznet_run_fp32.py`** — the G2.3 gate driver:
  audio → `quartznet_audio.extract_logmel()` → `quartznet_fp32` forward pass
  → `quartznet_ref.ctc_greedy()` → `quartznet_wer.corpus_wer()`. Writes a
  per-utterance JSONL (`build/quartznet_fp32/fp32_<split>.jsonl`) alongside
  the scalar gate result — the per-utterance ref/hyp pairs are reused by
  A4/A6's own WER gates rather than re-running the FP32 baseline three more
  times.

## Compute notes

Measured on this machine (WSL2, i9-13900H, 4 logical CPUs, no AVX-512, RTX
3050 4GB present but unused): **CPU, not GPU.** 4-thread forward pass ≈122
ms/utterance (62x real-time); full dev-clean (2703 utts, 2.98h audio) ≈10.5
minutes end-to-end including audio decode. Installing a CUDA `torch` wheel
was considered and rejected — it's a ~2.5GB dependency swap risking A1's
already-passed front-end gate and A4's ORT setup, to save under 10 minutes
on a run done only a handful of times across A3-A7. VRAM headroom was
checked anyway (not just assumed adequate): the longest LibriSpeech clip
(34.955s) peaks under 25MB of live activation against 4GB — a 40x margin —
so GPU would have worked, it just isn't worth the install cost here.

Operationally: single foreground run (backgrounded only because ~10.5
minutes exceeds this tool's blocking-call cap), no checkpoint/resume
machinery needed for crash-resilience at this runtime (though `--resume`
exists as a convenience, reading back the JSONL). `--limit N` gives a ~5s
smoke test that exercises the same code path without evaluating the gate.

## Verification

```
$ python3 sw/tinyml_reference/quartznet_nemo_export.py stt_en_quartznet15x5.nemo
...
folded weight/bias magnitudes all sane (< 100) -- no BN-fold scale blowup
G2.2: PASS

$ python3 sw/tinyml_reference/quartznet_run_fp32.py --split dev-clean
...
G2.3: PASS   (4.4392%, gate [4.10%, 4.70%])

$ python3 sw/tinyml_reference/quartznet_run_fp32.py --split test-clean
...
FP32 WER 4.4716% (2,351 errors / 52,576 words, 2620 utts) on test-clean --
INFORMATIONAL, no published number for this checkpoint on this split, not gated
2620/2620 utterances scored
G2.3: PASS
```

Regression: `make -C firmware/quartznet host` still `PASS — 2/2
configurations bit-exact` after the `expand()`/`ctc_greedy` changes (both
sides — the C interpreter and the NumPy golden — regenerate from the same
updated `expand()`, so this is a self-consistency check, not a claim that
the *specific* transcript strings match any previously-committed value —
`build/` is gitignored, nothing committed depended on the exact old
seeded-random transcript).
