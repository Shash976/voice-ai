#!/usr/bin/env python3
"""quartznet_run_firmware.py — Stage 7 Gap 2 A6, gate G2.6: full-corpus WER
through the REAL FIRMWARE C interpreter (quartznet_infer.c via qn_transcribe),
not ORT and not PyTorch -- this is specifically the bit-exact int32-
accumulator path, the whole reason A6 is a separate gate from A4's QDQ-
simulated int8-ORT number.

Model-level artifacts (quartznet_desc.bin/quartznet_weights.bin/
quartznet_qparams.bin, from quartznet_export_int8.py) are identical across
every utterance -- only the input blob varies. This script does NOT re-run
quartznet_export_int8.py per utterance (measured: a full export is 1.29s;
2620 of those would waste ~56 minutes for zero benefit). Instead it does the
input-side work itself (load_audio -> extract_logmel -> quantize_features,
using the real calibrated in_scale/in_zp read from <model-dir>/
quartznet_meta.json, NOT quartznet_audio.calibrate()/IN_ZP_DEFAULT -- see
mp3_to_text.py's header for why that distinction is a real, measured +0.28%
WER bug, not pedantry) and invokes the already-built qn_transcribe binary
once per utterance as a subprocess.

── Why subprocess-per-utterance, not a batched C harness ────────────────────

Process-start + 18.85MB weight/table/qparam load measured at ~40ms/call --
7.8% of a ~21-minute full test-clean run at 4 workers, not worth a new
ctypes/.so artifact and ABI coupling for. The stronger property this buys:
the transcripts this gate scores are produced by the EXACT SAME qn_transcribe
binary `make transcribe-real` runs, not a parallel reimplementation.

Run:
    python3 quartznet_run_firmware.py --split test-clean            # the gate
    python3 quartznet_run_firmware.py --split test-clean --limit 50   # smoke test
"""
from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import quartznet_audio as qa               # noqa: E402
import quartznet_wer as qw                 # noqa: E402

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]

EXPECTED_UTTS = {"dev-clean": 2703, "test-clean": 2620}
GATE_BAND_PCT = 0.50


def run_one(binary: pathlib.Path, model_dir: pathlib.Path, tmp_root: pathlib.Path,
           flac: pathlib.Path, in_scale: float, in_zp: int) -> dict:
    pcm = qa.load_audio(flac)
    feat = qa.extract_logmel(pcm)
    t_out = feat.shape[0] // 2   # floor: the descriptor format requires t_in == 2*t_out exactly
    t_in = 2 * t_out
    inp = qa.quantize_features(feat[:t_in], in_scale, in_zp)

    work = pathlib.Path(tempfile.mkdtemp(dir=tmp_root, prefix=flac.stem + "_"))
    blob_path = work / "input.bin"
    try:
        qa.write_input_blob(blob_path, inp)
        proc = subprocess.run(
            [str(binary), str(model_dir), str(blob_path), str(t_out)],
            capture_output=True, text=True, timeout=120)
        if proc.returncode != 0:
            raise RuntimeError(
                f"{flac.name}: qn_transcribe exit {proc.returncode}\n"
                f"stdout={proc.stdout}\nstderr={proc.stderr}")
        hyp = None
        for line in proc.stdout.splitlines():
            if line.startswith("TRANSCRIPT\t"):
                hyp = line[len("TRANSCRIPT\t"):]
                break
        if hyp is None:
            raise RuntimeError(f"{flac.name}: no TRANSCRIPT line in output:\n{proc.stdout}")
    finally:
        blob_path.unlink(missing_ok=True)
        work.rmdir()

    return {"id": flac.stem, "hyp": hyp, "t_out": t_out, "dur_s": pcm.size / qa.SAMPLE_RATE}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", choices=["dev-clean", "test-clean"], default="test-clean")
    ap.add_argument("--librispeech-root", type=pathlib.Path,
                     default=REPO_ROOT / "librispeech" / "LibriSpeech")
    ap.add_argument("--model-dir", type=pathlib.Path,
                     default=REPO_ROOT / "build" / "quartznet_real")
    ap.add_argument("--binary", type=pathlib.Path,
                     default=REPO_ROOT / "firmware" / "quartznet" / "qn_transcribe")
    ap.add_argument("--int8-ort-wer-json", type=pathlib.Path, default=None,
                     help="defaults to build/quartznet_int8/wer_int8_<split>.json")
    ap.add_argument("--out", type=pathlib.Path,
                     default=REPO_ROOT / "build" / "quartznet_firmware")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    root = args.librispeech_root / args.split
    if not root.is_dir():
        raise SystemExit(f"{root} does not exist -- extract LibriSpeech {args.split} first")
    if not args.binary.exists():
        raise SystemExit(f"{args.binary} not found -- run `make -C firmware/quartznet "
                         f"qn_transcribe` first")

    meta_path = args.model_dir / "quartznet_meta.json"
    if not meta_path.exists():
        raise SystemExit(f"{meta_path} not found -- run quartznet_export_int8.py "
                         f"--out {args.model_dir} first")
    meta = json.loads(meta_path.read_text())
    in_scale, in_zp = float(meta["in_scale"]), int(meta["in_zp"])

    ort_json = args.int8_ort_wer_json or (
        REPO_ROOT / "build" / "quartznet_int8" / f"wer_int8_{args.split}.json")
    if not ort_json.exists():
        raise SystemExit(f"{ort_json} not found -- run quartznet_run_int8_ort.py "
                         f"--split {args.split} first (G2.6 gates the DELTA against "
                         f"A4's own measured number, not a hardcoded literal)")
    int8_ort_wer_pct = json.loads(ort_json.read_text())["wer_pct"]

    utts = qw.load_librispeech(root)
    if args.limit:
        utts = utts[:args.limit]
    print(f"{args.split}: {len(utts)} utterances  "
          f"(model={args.model_dir}, in_scale={in_scale:.6f}, in_zp={in_zp})")

    args.out.mkdir(parents=True, exist_ok=True)
    tmp_root = args.out / "tmp"
    tmp_root.mkdir(parents=True, exist_ok=True)
    jsonl_path = args.out / f"firmware_{args.split}.jsonl"

    done: dict[str, dict] = {}
    if args.resume and jsonl_path.exists():
        for line in jsonl_path.read_text().splitlines():
            rec = json.loads(line)
            done[rec["id"]] = rec
        print(f"  --resume: {len(done)} utterances already scored, skipping")

    todo = [(flac, ref) for flac, ref in utts if flac.stem not in done]

    mode = "a" if args.resume and done else "w"
    t0 = time.time()
    running_err = sum(r["errors"] for r in done.values())
    running_words = sum(r["words"] for r in done.values())
    n_done = len(done)
    with open(jsonl_path, mode) as fh, ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_one, args.binary, args.model_dir, tmp_root,
                               flac, in_scale, in_zp): (flac, ref)
                  for flac, ref in todo}
        for fut in futures:
            flac, ref = futures[fut]
            r = fut.result()
            err = qw.edit_distance(qw.normalize(ref), qw.normalize(r["hyp"]))
            words = len(qw.normalize(ref))
            running_err += err
            running_words += words
            n_done += 1
            rec = {"id": r["id"], "ref": ref, "hyp": r["hyp"], "errors": err,
                  "words": words, "t_out": r["t_out"], "dur_s": r["dur_s"]}
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            if n_done % 200 == 0 or n_done == len(utts):
                elapsed = time.time() - t0
                run_wer = 100.0 * running_err / max(running_words, 1)
                print(f"  {n_done}/{len(utts)}  running_wer={run_wer:.4f}%  "
                      f"elapsed={elapsed:.0f}s", flush=True)
    try:
        tmp_root.rmdir()
    except OSError:
        pass

    records = [json.loads(l) for l in jsonl_path.read_text().splitlines()]
    wer, err, words = qw.corpus_wer((r["ref"], r["hyp"]) for r in records)
    wer_pct = 100.0 * wer
    empty_ids = [r["id"] for r in records if not r["hyp"].strip()]

    result = {"split": args.split, "n_utts": len(records), "errors": err,
             "words": words, "wer_pct": wer_pct, "int8_ort_wer_pct": int8_ort_wer_pct,
             "delta_pct": wer_pct - int8_ort_wer_pct, "elapsed_s": time.time() - t0}
    (args.out / f"wer_firmware_{args.split}.json").write_text(json.dumps(result, indent=2))

    print()
    ok = True
    delta = abs(wer_pct - int8_ort_wer_pct)
    if delta <= GATE_BAND_PCT:
        print(f"firmware WER {wer_pct:.4f}% ({err:,} errors / {words:,} words, "
              f"{len(records)} utts) on {args.split} -- delta {delta:.4f}% from "
              f"int8-ORT {int8_ort_wer_pct:.4f}%, within {GATE_BAND_PCT:.2f}%")
    else:
        print(f"*** firmware WER {wer_pct:.4f}% -- delta {delta:.4f}% from int8-ORT "
              f"{int8_ort_wer_pct:.4f}% exceeds {GATE_BAND_PCT:.2f}% ***")
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
    print(f"wrote {args.out / f'wer_firmware_{args.split}.json'}")

    if args.limit:
        print(f"(--limit {args.limit}: smoke test only, gate not evaluated)")
        return
    if not ok:
        raise SystemExit("quartznet_run_firmware: G2.6 checks FAILED")
    print("G2.6: PASS")


if __name__ == "__main__":
    main()
