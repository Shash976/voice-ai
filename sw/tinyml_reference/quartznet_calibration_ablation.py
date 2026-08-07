#!/usr/bin/env python3
"""quartznet_calibration_ablation.py — Stage 7 Gap 2 A7, gate G2.7:
calibration-set-size sensitivity. Re-runs A4's calibration at 50/200/500
utterances and requires the resulting dev-clean int8 WER spread to stay
under 0.1% absolute — empirically justifying A4's 200-utterance choice
rather than asserting it.

── Same fp32 export reused for every size, on purpose ───────────────────────

The ONNX export step (quartznet_calibrate.export_onnx) is exported ONCE and
shared across all three `quantize_static` calls (via `calibrate(...,
pre_onnx=...)`), so the calibration clip SET is provably the only varying
input between the three runs -- eliminates a confound rather than arguing
it away (verified separately: the export is byte-identical regardless of
which clip traces it, since the graph's time axis is dynamic).

── Result is non-monotone: read it as "saturates early," not "more is better" ──

Measured: 50 utterances gives the LOWEST dev-clean WER, 200 the HIGHEST, 500
in between -- there is no "more calibration data helps" trend. The honest
reading: percentile activation-range estimation has already saturated by 50
utterances (4.5 minutes of audio); 200 buys margin against sampling noise,
500 buys nothing measurable. This script's gate PASSING is evidence that 200
was already more than sufficient, not evidence that 200 was necessary.

── A real caveat: the calibration-overlap-excluded metric can disagree ──────

quartznet_run_int8_ort.py also reports "WER excluding calibration overlap
utterances" as a diagnostic. That number is NOT used for this gate (the
three excluded subsets are different sizes -- 2653/2503/2203 utterances at
50/200/500 -- so they are not comparable to each other or to the full-corpus
number G2.3/G2.4 are themselves defined on). It CAN show a larger spread
than the full-corpus number; this script reports both but gates on the
full-corpus number only, and says so explicitly in its own output so the
discrepancy is never silently discovered by a reviewer instead.

Run:
    python3 quartznet_calibration_ablation.py                    # the gate
    python3 quartznet_calibration_ablation.py --sizes 50,200      # partial
"""
from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import quartznet_audio as qa               # noqa: E402
import quartznet_calibrate as qc           # noqa: E402
import quartznet_fp32 as qf                # noqa: E402

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
GATE_SPREAD_PCT = 0.10


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sizes", type=str, default="50,200,500")
    ap.add_argument("--weights", type=pathlib.Path,
                     default=REPO_ROOT / "build" / "quartznet_nemo" / "folded_weights.npz")
    ap.add_argument("--out", type=pathlib.Path,
                     default=REPO_ROOT / "build" / "quartznet_calib_ablation")
    ap.add_argument("--split", default="dev-clean")
    ap.add_argument("--percentile", type=float, default=99.999)
    ap.add_argument("--seed", type=int, default=qc.SEED)
    ap.add_argument("--fp32-wer-json", type=pathlib.Path, default=None)
    args = ap.parse_args()

    sizes = [int(s) for s in args.sizes.split(",")]
    args.out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    print(f"loading {args.weights} ...")
    model = qf.load(args.weights)

    print(f"selecting calibration clips for sizes {sizes} (40 speakers each)...")
    clip_sets = {n: qc.select_calibration_clips(seed=args.seed, total=n) for n in sizes}
    for n, clips in clip_sets.items():
        print(f"  {n}: {len(clips)} clips selected")

    # Export the fp32 graph ONCE (from the largest set's first clip -- shape
    # is irrelevant, the ONNX time axis is dynamic), then quant_pre_process
    # it ONCE, and pass that same pre-processed graph into every calibrate()
    # call below -- see module docstring for why this matters.
    largest = clip_sets[max(sizes)]
    fp32_onnx = args.out / "quartznet_fp32.onnx"
    qc.export_onnx(model, qa.extract_logmel(qa.load_audio(largest[0])), fp32_onnx)
    qc.verify_export_roundtrip(model, fp32_onnx, largest)
    pre_onnx = args.out / "quartznet_fp32_pre.onnx"
    from onnxruntime.quantization import quant_pre_process
    quant_pre_process(str(fp32_onnx), str(pre_onnx), skip_symbolic_shape=False)
    print(f"  shared fp32 export -> {fp32_onnx}, {pre_onnx}")

    per_size: dict[int, dict] = {}
    for n in sizes:
        out_dir = args.out / f"calib{n}"
        print(f"\ncalibrating at {n} utterances -> {out_dir} ...")
        result = qc.calibrate(model, clip_sets[n], out_dir, percentile=args.percentile,
                              pre_onnx=pre_onnx, want_qparams=False)
        print(f"  {result['n_clips']} clips, {result['elapsed_s']:.1f}s "
              f"-> {result['int8_onnx']}")

        wer_json = out_dir / f"wer_int8_{args.split}.json"
        cmd = [sys.executable,
               str(pathlib.Path(__file__).resolve().parent / "quartznet_run_int8_ort.py"),
               "--split", args.split, "--model", str(result["int8_onnx"]),
               "--out", str(out_dir)]  # --out MANDATORY: never let this default to
                                       # build/quartznet_int8/, which host-real/
                                       # wer-firmware (G2.5/G2.6) depend on.
        if args.fp32_wer_json:
            cmd += ["--fp32-wer-json", str(args.fp32_wer_json)]
        print(f"  running: {' '.join(cmd)}")
        proc = subprocess.run(cmd, capture_output=True, text=True)
        print(proc.stdout[-2000:])
        if proc.returncode not in (0, 1):  # 1 = gate-failed exit, still a real result
            raise SystemExit(f"*** quartznet_run_int8_ort.py crashed for size {n} ***\n"
                             f"{proc.stdout}\n{proc.stderr}")
        if not wer_json.exists():
            raise SystemExit(f"*** {wer_json} not written -- run failed ***\n{proc.stdout}")
        per_size[n] = json.loads(wer_json.read_text())

    wers = {n: d["wer_pct"] for n, d in per_size.items()}
    non_calib_wers = {n: d.get("non_calibration_wer_pct") for n, d in per_size.items()}
    spread = max(wers.values()) - min(wers.values())
    worst_lo = min(wers, key=wers.get)
    worst_hi = max(wers, key=wers.get)

    print("\n" + "=" * 70)
    print(f"{'size':>6} {'WER %':>10} {'errors':>8} {'words':>8} "
          f"{'non-calib WER %':>16}")
    for n in sizes:
        d = per_size[n]
        print(f"{n:>6} {d['wer_pct']:>10.4f} {d['errors']:>8} {d['words']:>8} "
              f"{(non_calib_wers[n] or float('nan')):>16.4f}")

    margin_words = min(d["words"] for d in per_size.values())
    margin_word_errors = abs(GATE_SPREAD_PCT - spread) / 100.0 * margin_words
    print(f"\nfull-corpus spread: {wers[worst_hi]:.4f}% ({worst_hi}) - "
          f"{wers[worst_lo]:.4f}% ({worst_lo}) = {spread:.4f}%  "
          f"(gate < {GATE_SPREAD_PCT:.4f}%, margin ~{margin_word_errors:.1f} word errors)")

    non_calib_vals = [v for v in non_calib_wers.values() if v is not None]
    if len(non_calib_vals) == len(sizes):
        nc_spread = max(non_calib_vals) - min(non_calib_vals)
        print(f"calibration-overlap-EXCLUDED spread: {nc_spread:.4f}% -- NOT gated "
              f"(the excluded subsets are different sizes per calibration set, so "
              f"not comparable to each other or to G2.3/G2.4's full-corpus "
              f"convention); informational only" +
              (" -- NOTE THIS EXCEEDS THE GATE BAND even though the gated metric "
               "does not" if nc_spread >= GATE_SPREAD_PCT else ""))

    result = {"split": args.split, "sizes": sizes, "wer_pct": wers,
             "non_calibration_wer_pct": non_calib_wers, "spread_pct": spread,
             "elapsed_s": time.time() - t0}
    (args.out / "ablation.json").write_text(json.dumps(result, indent=2))
    print(f"\nwrote {args.out / 'ablation.json'}")

    if spread < GATE_SPREAD_PCT:
        print(f"\nWER spread {spread:.4f}% across sizes {sizes} < {GATE_SPREAD_PCT:.4f}% "
              f"-- calibration is not size-sensitive in this range")
        print("G2.7: PASS")
    else:
        print(f"\n*** WER spread {spread:.4f}% across sizes {sizes} "
              f">= {GATE_SPREAD_PCT:.4f}% ***")
        raise SystemExit("quartznet_calibration_ablation: G2.7 checks FAILED")


if __name__ == "__main__":
    main()
