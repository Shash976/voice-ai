# 07i — Gap 2 A7: calibration-size ablation

Stage 7 Gap 2's final core task: empirically justify A4's 200-utterance
calibration set size, rather than asserting it. Re-run calibration at 50,
200, and 500 utterances and require the resulting dev-clean int8 WER spread
to stay under 0.1% absolute.

## Result: G2.7 PASSES, by a real but thin margin

```
  size      WER %   errors    words  non-calib WER %
    50     4.4778     2436    54402           4.4950
   200     4.5678     2485    54402           4.5629
   500     4.5513     2476    54402           4.6305

full-corpus spread: 4.5678% (200) - 4.4778% (50) = 0.0901%
  (gate < 0.1000%, margin ~5.4 word errors)
G2.7: PASS
```

**Margin is 5.4 word errors out of 54,402** — 0.0099% of the entire gate
band. Six more word errors anywhere and this gate flips. This is a real
pass, not a rounding artifact, but the thinness is the actual finding and is
recorded here rather than left for a reviewer to notice independently.

## The result is non-monotone — and that changes what the gate actually shows

50 utterances gives the **lowest** WER; 200 the **highest**; 500 in between.
There is no "more calibration data is better" trend anywhere in this data.
The honest reading: **percentile activation-range estimation has already
saturated by 50 utterances** (4.5 minutes of audio) — the three points are
the same underlying number plus ±0.05%-scale sampling noise from which
utterances happened to land in each calibration draw.

**So this gate passing is evidence that 200 was already more than
sufficient — not evidence that 200 was necessary.** 50 would very likely
have worked too. Documented explicitly so this doesn't get cited later as
"200 was needed" — the data doesn't support that stronger claim.

## A real caveat: a different (excluded) metric fails the same gate

`quartznet_run_int8_ort.py` also reports "WER excluding calibration-overlap
utterances" as a diagnostic. On that metric:

```
calibration-overlap-EXCLUDED spread: 0.1355% -- exceeds the 0.1% gate band
```

This is **not** the gate metric (the three excluded subsets are different
sizes — 2653/2503/2203 utterances at 50/200/500 — so they aren't comparable
to each other, or to the full-corpus convention G2.3/G2.4 are themselves
defined on), and the full-corpus number is correct as the primary metric.
But the discrepancy is real and worth understanding: the exclusion-based
numbers rise monotonically with calibration size (4.4950% → 4.5629% →
4.6305%), consistent with the excluded subset getting progressively harder
as more of the easy short clips get pulled into calibration — a
subset-composition artifact of what's left over, not evidence the
calibration itself is worse at larger sizes. `quartznet_calibration_ablation.py`
prints and flags this explicitly rather than only gating on (and reporting)
the number that happens to pass.

## Design: same fp32 export shared across all three sizes, on purpose

The ONNX export (`quartznet_calibrate.export_onnx`) is run **once** and its
pre-processed graph is reused for all three `quantize_static` calls, via
`quartznet_calibrate.calibrate(..., pre_onnx=...)` (a new parameter — the
export is provably invariant to calibration-set size since the graph's time
axis is dynamic and the export doesn't depend on calibration data at all).
This makes the calibration clip *set* the only variable between the three
runs, eliminating a confound rather than arguing it away.

**Stratification**: all three sizes keep `n_speakers=40` — every dev-clean
speaker is represented at every size, and only per-speaker depth varies
(`select_calibration_clips`'s new `total=` parameter derives a per-speaker
count via `divmod`, assigning any remainder to the speakers with the
deepest ≤10s pools). Reducing `n_speakers` for the smaller sizes was
rejected: it would drop specific speakers entirely, confounding *calibration
size* with *speaker diversity* — exactly the variable this ablation exists
to isolate. A useful side effect: `total=200` reproduces A4's original
`per_speaker=5` clip list **exactly** (verified: identical clip list, and
the resulting `quartznet_int8.onnx` is a byte-for-byte match to A4's
committed model) — the remainder-first-then-RNG design consumes RNG in the
same order the original fixed-depth code did.

The 40-speaker ≤10s pool bounds the total this scheme can support without
raising per-speaker depth again — the thinnest speaker (13 eligible clips)
caps a perfectly even split at 520; 500 sits inside that ceiling with margin.

## A real bug found (and now fixed) while regenerating A4's own artifact

While re-running the 200-utterance point to compare against A4's committed
number, found `build/quartznet_int8/wer_int8_dev-clean.json` on disk did
**not** hold A4's real 4.5678% result — it held a stray `--limit 50`
smoke-test's 7.5061%, silently written over the real file. Root cause,
present identically in **all three** of this stack's WER gate drivers
(`quartznet_run_{fp32,int8_ort,firmware}.py` — A3, A4, A6): each wrote its
canonical `wer_*_<split>.json` unconditionally, *before* checking
`args.limit` and returning early. A `--limit` smoke test run at any point
after a real gate run would silently clobber that gate's own result file
with a tiny-sample number. Fixed in all three: the canonical JSON is now
written only when `not args.limit`. Verified: a `--limit 10` run after the
fix leaves the existing canonical file byte-for-byte unchanged.

## New / changed files

- **`sw/tinyml_reference/quartznet_calibration_ablation.py`** (new) — the
  G2.7 gate driver. Selects all three calibration sets, shares one fp32
  export across them, calls `quartznet_calibrate.calibrate()` (imported)
  per size, and `quartznet_run_int8_ort.py` (subprocess — no importable
  single-model-scoring entry point exists, and refactoring it would perturb
  the live G2.4 gate for no benefit) per size, always with an explicit
  `--out` so no ablation run can silently re-point `build/quartznet_int8/`
  — a live dependency of G2.5 (`host-real`) and G2.6 (`wer-firmware`) — at a
  differently-calibrated model.
- **`sw/tinyml_reference/quartznet_calibrate.py`** — `select_calibration_clips`
  gains `total:`; `main()`'s calibration body factored into a reusable
  `calibrate()` function (zero behavior change to the no-flags CLI path);
  new `--calib-size` CLI flag.
- **`sw/tinyml_reference/quartznet_run_{fp32,int8_ort,firmware}.py`** — the
  `--limit`-clobbers-the-canonical-file bug fix described above.

## Verification

```
$ python3 sw/tinyml_reference/quartznet_calibration_ablation.py
...
G2.7: PASS

$ python3 sw/tinyml_reference/quartznet_run_int8_ort.py --split dev-clean --limit 10
...
(--limit 10: smoke test only, gate not evaluated)
$ diff <(before) <(after)   # canonical wer_int8_dev-clean.json unchanged
```
