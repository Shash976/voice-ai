#!/usr/bin/env python3
"""quartznet_run_fp32.py — Stage 7 Gap 2 A3, gate G2.3: FP32 baseline WER.

mp3/flac -> quartznet_audio.extract_logmel() -> quartznet_fp32 forward pass
-> quartznet_ref.ctc_greedy() -> quartznet_wer.corpus_wer(), against real
LibriSpeech audio and transcripts.

── Which published number this gates against, and why ───────────────────────

The originally-planned gate ("within 0.3% absolute of NeMo's published
3.90%") turned out to name the WRONG checkpoint. 3.90% (test-clean, greedy,
no LM) belongs to `quartznet_15x5_ls_sp` -- a LibriSpeech-only checkpoint
shipped as two loose `.pt` files (JasperEncoder/JasperDecoderForCTC), not a
`.nemo` archive, and it is the checkpoint the original QuartzNet paper's
Table cites. This repo instead uses `stt_en_quartznet15x5` (the multi-domain
checkpoint: LibriSpeech + WSJ + Fisher + Switchboard + Common Voice + NSC
Singapore, 7,057h total) -- deliberately, per A2/quartznet_nemo_export.py's
own tarball-based `.nemo` loader, and it is the more realistic checkpoint
for a device that has to transcribe real-world (not just audiobook) speech.

`stt_en_quartznet15x5`'s own NGC model card publishes **4.4% WER on
LibriSpeech dev-clean** (11.3% dev-other). Measured here: 4.4392% on
dev-clean -- 0.04% absolute from the card, comfortably inside any reasonable
gate, and strong end-to-end validation of the whole pipeline (front end +
forward-pass graph + BN-folded weights + CTC decode + this WER scorer).

So: **G2.3 gates FP32 WER on dev-clean against 4.4% +/- 0.3% (i.e.
[4.10%, 4.70%])**. test-clean is reported alongside as an informational,
UNGATED datapoint -- no published number exists for this specific checkpoint
on test-clean to compare against.

(The NGC card's Performance section is prefixed "measuring using Character
Error Rate" -- boilerplate copy-pasted across NeMo NGC cards and wrong here:
4.4% CER would imply ~1.5% WER, implausibly good, and the 4.4/11.3 pair
matches the WER-shaped pattern of `quartznet_15x5_ls_sp`'s own card
(3.83/11.08 WER). Our independent 4.4392% measurement settles it.)

Run:
    python3 quartznet_run_fp32.py --split dev-clean            # the gate
    python3 quartznet_run_fp32.py --split test-clean           # informational
    python3 quartznet_run_fp32.py --split dev-clean --limit 100  # smoke test
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import quartznet_audio as qa               # noqa: E402
import quartznet_fp32 as qf                # noqa: E402
import quartznet_topology as qt            # noqa: E402
import quartznet_wer as qw                 # noqa: E402
from quartznet_ref import ctc_greedy       # noqa: E402

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]

EXPECTED_UTTS = {"dev-clean": 2703, "test-clean": 2620}
# WER% published on the stt_en_quartznet15x5 NGC model card. dev-clean is the
# only split with a real published number for THIS checkpoint -- see the
# module docstring for why test-clean has none and is reported ungated.
PUBLISHED_WER_PCT = {"dev-clean": 4.4, "test-clean": None}
GATE_BAND_PCT = 0.30


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", choices=["dev-clean", "test-clean"], default="dev-clean")
    ap.add_argument("--librispeech-root", type=pathlib.Path,
                     default=REPO_ROOT / "librispeech" / "LibriSpeech")
    ap.add_argument("--weights", type=pathlib.Path,
                     default=REPO_ROOT / "build" / "quartznet_nemo" / "folded_weights.npz")
    ap.add_argument("--out", type=pathlib.Path,
                     default=REPO_ROOT / "build" / "quartznet_fp32")
    ap.add_argument("--limit", type=int, default=None,
                     help="score only the first N utterances (smoke test)")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--resume", action="store_true",
                     help="skip utterances already in the output JSONL")
    args = ap.parse_args()

    torch.set_grad_enabled(False)
    torch.set_num_threads(args.threads)

    root = args.librispeech_root / args.split
    if not root.is_dir():
        raise SystemExit(f"{root} does not exist -- extract LibriSpeech {args.split} first")

    print(f"loading {args.weights} ...")
    model = qf.load(args.weights)
    n_weight_bearing = sum(1 for ld in qt.expand()
                            if ld.op in (qt.OP_DW, qt.OP_PW))
    print(f"  {len(model.convs)}/{n_weight_bearing} conv layers built")

    utts = qw.load_librispeech(root)
    if args.limit:
        utts = utts[:args.limit]
    print(f"{args.split}: {len(utts)} utterances")

    args.out.mkdir(parents=True, exist_ok=True)
    jsonl_path = args.out / f"fp32_{args.split}.jsonl"

    done: dict[str, dict] = {}
    if args.resume and jsonl_path.exists():
        for line in jsonl_path.read_text().splitlines():
            rec = json.loads(line)
            done[rec["id"]] = rec
        print(f"  --resume: {len(done)} utterances already scored, skipping")

    mode = "a" if args.resume and done else "w"
    t0 = time.time()
    running_err = sum(r["errors"] for r in done.values())
    running_words = sum(r["words"] for r in done.values())
    with open(jsonl_path, mode) as fh:
        for i, (flac, ref) in enumerate(utts):
            uid = flac.stem
            if uid in done:
                continue
            pcm = qa.load_audio(flac)
            feat = qa.extract_logmel(pcm)
            lg = qf.logits(model, feat)
            _, hyp = ctc_greedy(lg, qt.BLANK_IDX)
            err = qw.edit_distance(qw.normalize(ref), qw.normalize(hyp))
            words = len(qw.normalize(ref))
            running_err += err
            running_words += words
            rec = {"id": uid, "ref": ref, "hyp": hyp, "errors": err,
                   "words": words, "t_out": lg.shape[0],
                   "dur_s": pcm.size / qa.SAMPLE_RATE}
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            if (i + 1) % 200 == 0 or (i + 1) == len(utts):
                elapsed = time.time() - t0
                run_wer = 100.0 * running_err / max(running_words, 1)
                print(f"  {i + 1}/{len(utts)}  running_wer={run_wer:.4f}%  "
                      f"elapsed={elapsed:.0f}s", flush=True)

    # Score from the JSONL (not in-memory accumulation), so --resume and a
    # from-scratch run produce identically-scored output.
    records = [json.loads(l) for l in jsonl_path.read_text().splitlines()]
    wer, err, words = qw.corpus_wer((r["ref"], r["hyp"]) for r in records)
    wer_pct = 100.0 * wer
    empty_ids = [r["id"] for r in records if not r["hyp"].strip()]

    result = {"split": args.split, "n_utts": len(records), "errors": err,
              "words": words, "wer_pct": wer_pct, "elapsed_s": time.time() - t0}
    result_path = args.out / f"wer_fp32_{args.split}.json"
    if not args.limit:
        # Only ever write the canonical result file for a full, ungated-scope
        # run -- a --limit smoke test must never overwrite a real gate result
        # sitting at this same path with a tiny-sample number.
        result_path.write_text(json.dumps(result, indent=2))

    print()
    ok = True
    published = PUBLISHED_WER_PCT[args.split]
    if published is None:
        print(f"FP32 WER {wer_pct:.4f}% ({err:,} errors / {words:,} words, "
              f"{len(records)} utts) on {args.split} -- INFORMATIONAL, no "
              f"published number for this checkpoint on this split, not gated")
    else:
        lo, hi = published - GATE_BAND_PCT, published + GATE_BAND_PCT
        if lo <= wer_pct <= hi:
            print(f"FP32 WER {wer_pct:.4f}% ({err:,} errors / {words:,} words, "
                  f"{len(records)} utts) within {GATE_BAND_PCT:.2f}% of the "
                  f"published {published:.2f}% (stt_en_quartznet15x5, {args.split})")
        else:
            print(f"*** FP32 WER {wer_pct:.4f}% outside [{lo:.2f}%, {hi:.2f}%] "
                  f"(published {published:.2f}% for stt_en_quartznet15x5 on "
                  f"{args.split}) ***")
            ok = False

    if not args.limit:
        if len(records) != EXPECTED_UTTS[args.split]:
            print(f"*** scored {len(records)} utterances, expected "
                  f"{EXPECTED_UTTS[args.split]} ***")
            ok = False
        else:
            print(f"{len(records)}/{EXPECTED_UTTS[args.split]} utterances scored")

    if empty_ids:
        print(f"*** {len(empty_ids)} empty transcript(s): {empty_ids[:5]} ***")
        ok = False

    print(f"wrote {jsonl_path}")
    if not args.limit:
        print(f"wrote {result_path}")

    if args.limit:
        print(f"(--limit {args.limit}: smoke test only, gate not evaluated)")
        return
    if not ok:
        raise SystemExit("quartznet_run_fp32: G2.3 checks FAILED")
    print("G2.3: PASS")


if __name__ == "__main__":
    main()
