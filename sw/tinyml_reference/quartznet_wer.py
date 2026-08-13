"""quartznet_wer.py — hand-rolled word error rate + LibriSpeech corpus
loading, shared by every WER gate in Stage 7 Gap 2 (A3/G2.3 FP32, A4/G2.4
int8-ORT, A6/G2.6 firmware, A7/G2.7 calibration ablation).

Hand-rolled Levenshtein rather than a `jiwer`/`editdistance` dependency,
matching this repo's existing style (`quartznet_audio.resample_poly()`,
`quartznet_audio.build_mel_fb()` are both hand-rolled for the same reason).

── Normalization convention: lowercase, whitespace-split, nothing else ──────

Matches NeMo's own `nemo.collections.asr.metrics.wer.word_error_rate()`
(no punctuation stripping beyond what the LibriSpeech manifest builder
already does at ingestion: `text.lower().strip()`). LibriSpeech `.trans.txt`
is already all-caps with no punctuation except apostrophes, and apostrophe
is a real symbol in the model's own `LABELS` alphabet — stripping it would
merge "dont"/"don't" and artificially lower WER. Empirically validated: this
convention reproduces the `stt_en_quartznet15x5` model card's published
4.4% dev-clean WER to 0.04% absolute (see quartznet_run_fp32.py).

── Aggregation: corpus-level, not mean-of-per-utterance ─────────────────────

wer = sum(errors over all utterances) / sum(reference words over all
utterances) — matching NeMo's own metric. The mean of per-utterance rates
would overweight short utterances and give a different (and non-standard)
number.
"""
from __future__ import annotations

import pathlib

import numpy as np


def edit_distance(ref: list[str], hyp: list[str]) -> int:
    """Levenshtein distance, unit cost for substitute/insert/delete.

    Two-row DP: O(len(ref) * len(hyp)) time, O(len(hyp)) space. Reference
    utterances top out around 90 words, so the plain Python loop is not
    worth vectorizing further.
    """
    n, m = len(ref), len(hyp)
    prev = np.arange(m + 1, dtype=np.int32)
    cur = np.empty(m + 1, dtype=np.int32)
    for i in range(1, n + 1):
        cur[0] = i
        ri = ref[i - 1]
        for j in range(1, m + 1):
            cur[j] = min(prev[j] + 1,                        # deletion
                         cur[j - 1] + 1,                      # insertion
                         prev[j - 1] + (ri != hyp[j - 1]))    # substitution
        prev, cur = cur, prev
    return int(prev[m])


def normalize(text: str) -> list[str]:
    """LibriSpeech/NeMo convention: lowercase, split on whitespace. No
    punctuation stripping -- see module docstring."""
    return text.lower().split()


def corpus_wer(pairs) -> tuple[float, int, int]:
    """pairs: iterable of (reference_text, hypothesis_text) strings.

    Returns (wer, total_errors, total_ref_words). Corpus-level aggregation --
    sum(errors)/sum(words), not the mean of per-utterance rates.
    """
    err = words = 0
    for ref, hyp in pairs:
        rw = normalize(ref)
        err += edit_distance(rw, normalize(hyp))
        words += len(rw)
    return (err / words if words else float("nan")), err, words


def load_librispeech(root: pathlib.Path) -> list[tuple[pathlib.Path, str]]:
    """<root>/<speaker>/<chapter>/<speaker>-<chapter>-<utt>.flac plus one
    <speaker>-<chapter>.trans.txt per chapter dir, lines
    "<speaker>-<chapter>-<utt> TRANSCRIPT IN CAPS".

    Returns [(flac_path, reference_text), ...] sorted by path, one entry per
    utterance. Raises if any .flac has no matching transcript line -- a
    silent skip would quietly shrink the corpus a WER gate is measured over.
    """
    refs: dict[str, str] = {}
    for t in sorted(root.rglob("*.trans.txt")):
        for line in t.read_text().splitlines():
            if not line.strip():
                continue
            uid, _, text = line.partition(" ")
            refs[uid] = text

    flacs = sorted(root.rglob("*.flac"))
    missing = [p.stem for p in flacs if p.stem not in refs]
    if missing:
        raise SystemExit(
            f"{root}: {len(missing)} .flac file(s) with no transcript line "
            f"(first few: {missing[:5]})")
    return [(p, refs[p.stem]) for p in flacs]
