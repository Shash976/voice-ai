#!/usr/bin/env python3
"""quartznet_run_int8_ort.py — Stage 7 Gap 2 A4, gate G2.4: int8-ORT WER
within 0.3% absolute of the FP32 baseline (A3/G2.3).

Mirrors quartznet_run_fp32.py's structure exactly (same JSONL schema, same
progress/gate-print conventions) -- only the model call changes, from a
direct PyTorch forward pass to an onnxruntime.InferenceSession over the
statically-quantized graph `quartznet_calibrate.py` produces.

── Scope: this measures a QDQ-simulated int8 WER, not bit-exactness ─────────

31 of 171 convs and 14 of 15 Adds in the quantized graph run as
DequantizeLinear -> fp32 op -> QuantizeLinear rather than a fused int8
kernel (ORT's QDQ format only fuses at ExecutionProvider load time, and not
every node is eligible) -- inputs/weights/outputs are still genuinely int8
throughout, but the accumulator for those nodes is fp32 rather than the
firmware's int32. So this gate validates the CALIBRATION RANGES are good
(the thing A4 is actually responsible for), not that the RTL's exact
int32-accumulator path reproduces this WER -- that bit-exact claim is A6's
job, against real quartznet_infer.c on the descriptor blob A5 emits.

── Split: gated on dev-clean, test-clean reported alongside (ungated) ───────

G2.4 is defined as "within 0.3% of G2.3", and G2.3 is a dev-clean number --
gating on a different split would compare an int8 number against an fp32
number measured on different audio, not a quantization-induced delta.
200 of dev-clean's 2703 utterances were used for calibration (7.4%
overlap) -- empirically this does not inflate the result: dev-clean shows a
SMALLER fp32->int8 delta than the fully-disjoint test-clean split
(consistent with PTQ estimating activation ranges from ~18 minutes of audio,
not fitting decision boundaries the way training would).

Run:
    python3 quartznet_run_int8_ort.py --split dev-clean            # the gate
    python3 quartznet_run_int8_ort.py --split test-clean           # informational
    python3 quartznet_run_int8_ort.py --split dev-clean --limit 100  # smoke test
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import numpy as np
import onnxruntime as ort

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import quartznet_audio as qa               # noqa: E402
import quartznet_topology as qt            # noqa: E402
import quartznet_wer as qw                 # noqa: E402
from quartznet_ref import ctc_greedy       # noqa: E402

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]

EXPECTED_UTTS = {"dev-clean": 2703, "test-clean": 2620}
GATE_BAND_PCT = 0.30


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", choices=["dev-clean", "test-clean"], default="dev-clean")
    ap.add_argument("--librispeech-root", type=pathlib.Path,
                     default=REPO_ROOT / "librispeech" / "LibriSpeech")
    ap.add_argument("--model", type=pathlib.Path,
                     default=REPO_ROOT / "build" / "quartznet_int8" / "quartznet_int8.onnx")
    ap.add_argument("--fp32-wer-json", type=pathlib.Path, default=None,
                     help="defaults to build/quartznet_fp32/wer_fp32_<split>.json")
    ap.add_argument("--out", type=pathlib.Path,
                     default=REPO_ROOT / "build" / "quartznet_int8")
    ap.add_argument("--calibration-clips", type=pathlib.Path, default=None,
                     help="defaults to <model's dir>/calibration_clips.json")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    root = args.librispeech_root / args.split
    if not root.is_dir():
        raise SystemExit(f"{root} does not exist -- extract LibriSpeech {args.split} first")

    fp32_json = args.fp32_wer_json or (
        REPO_ROOT / "build" / "quartznet_fp32" / f"wer_fp32_{args.split}.json")
    if not fp32_json.exists():
        raise SystemExit(
            f"{fp32_json} not found -- run quartznet_run_fp32.py --split {args.split} "
            f"first (G2.4 gates the DELTA against G2.3's own measured number, "
            f"not a hardcoded literal)")
    fp32_wer_pct = json.loads(fp32_json.read_text())["wer_pct"]

    print(f"loading {args.model} ...")
    so = ort.SessionOptions()
    so.intra_op_num_threads = args.threads
    sess = ort.InferenceSession(str(args.model), so, providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name

    utts = qw.load_librispeech(root)
    if args.limit:
        utts = utts[:args.limit]
    print(f"{args.split}: {len(utts)} utterances")

    calib_path = args.calibration_clips or (args.model.parent / "calibration_clips.json")
    calib_ids: set[str] = set()
    if calib_path.exists():
        calib_ids = {pathlib.Path(p).stem for p in json.loads(calib_path.read_text())}

    args.out.mkdir(parents=True, exist_ok=True)
    jsonl_path = args.out / f"int8_{args.split}.jsonl"

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
            x = np.ascontiguousarray(feat.T, dtype=np.float32)[None]
            lg = sess.run(None, {in_name: x})[0][0].T
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

    records = [json.loads(l) for l in jsonl_path.read_text().splitlines()]
    wer, err, words = qw.corpus_wer((r["ref"], r["hyp"]) for r in records)
    wer_pct = 100.0 * wer
    empty_ids = [r["id"] for r in records if not r["hyp"].strip()]

    calib_overlap = sum(1 for r in records if r["id"] in calib_ids)
    non_calib_records = [r for r in records if r["id"] not in calib_ids]
    non_calib_wer = None
    if calib_overlap and non_calib_records:
        w2, e2, wd2 = qw.corpus_wer((r["ref"], r["hyp"]) for r in non_calib_records)
        non_calib_wer = 100.0 * w2

    result = {"split": args.split, "n_utts": len(records), "errors": err,
              "words": words, "wer_pct": wer_pct, "fp32_wer_pct": fp32_wer_pct,
              "delta_pct": wer_pct - fp32_wer_pct,
              "calibration_overlap_utts": calib_overlap,
              "non_calibration_wer_pct": non_calib_wer,
              "elapsed_s": time.time() - t0}
    (args.out / f"wer_int8_{args.split}.json").write_text(json.dumps(result, indent=2))

    print()
    ok = True
    delta = abs(wer_pct - fp32_wer_pct)
    if delta <= GATE_BAND_PCT:
        print(f"int8-ORT WER {wer_pct:.4f}% ({err:,} errors / {words:,} words, "
              f"{len(records)} utts) on {args.split} -- delta {delta:.4f}% from "
              f"FP32 baseline {fp32_wer_pct:.4f}%, within {GATE_BAND_PCT:.2f}%")
    else:
        print(f"*** int8-ORT WER {wer_pct:.4f}% -- delta {delta:.4f}% from FP32 "
              f"baseline {fp32_wer_pct:.4f}% exceeds {GATE_BAND_PCT:.2f}% ***")
        ok = False

    if calib_overlap:
        print(f"  ({calib_overlap}/{len(records)} utterances also used for "
              f"calibration; WER excluding them: "
              f"{non_calib_wer:.4f}% over {len(non_calib_records)} utts)")

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
    print(f"wrote {args.out / f'wer_int8_{args.split}.json'}")

    if args.limit:
        print(f"(--limit {args.limit}: smoke test only, gate not evaluated)")
        return
    if not ok:
        raise SystemExit("quartznet_run_int8_ort: G2.4 checks FAILED")
    print("G2.4: PASS")


if __name__ == "__main__":
    main()
