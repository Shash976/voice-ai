# mp3_to_text.py — glue: mp3/wav file -> int8 log-mel blob for firmware/quartznet.
#
# Bridges quartznet_audio.py's front end (until now a self-tested island, never
# called from anywhere else in the repo) to the C interpreter's input path.
# Writes build/quartznet/mp3_input.bin — deliberately a different filename from
# quartznet_ref.py's quartznet_input.bin, so `make goldens` output is never
# clobbered — plus a `mp3_input.t_out` sidecar, since t_out varies with clip
# length and firmware/quartznet/qn_transcribe.c needs it for qn_load().
#
# ── --model-dir: REQUIRED for the real calibrated model ──────────────────────
#
# Without --model-dir, weights/qparams default to `make goldens`' seeded-
# random blobs, and the input is quantized with quartznet_audio.calibrate()'s
# per-clip percentile scale + IN_ZP_DEFAULT=-20 -- a placeholder appropriate
# only because the seeded-random qparams have nothing real to match anyway.
# The resulting transcript is expected to be gibberish; this only proves the
# mp3 -> features -> blob -> interpreter plumbing.
#
# Stage 7 Gap 2 A6: pointed at a REAL calibrated model
# (quartznet_export_int8.py's --out, e.g. build/quartznet_real) via
# --model-dir, this reads that model's ACTUAL calibrated in_scale/in_zp from
# its quartznet_meta.json sidecar instead. This is not optional plumbing --
# using calibrate()/IN_ZP_DEFAULT against real calibrated weights measured
# +0.28% absolute WER on a 200-utterance sample (calibrate() returns a
# per-clip scale close to but not equal to the model's real fixed scale, plus
# IN_ZP_DEFAULT=-20 vs the real -44 shifts every mel bin by a real amount --
# together an ~16% amplitude compression plus a +0.93 sigma DC offset on a
# z-scored feature), more than half of G2.6's entire 0.5% WER gate band from
# a single wrong default.
#
# Usage:
#   python3 sw/tinyml_reference/mp3_to_text.py CLIP.mp3 --model-dir build/quartznet_real
#   python3 sw/tinyml_reference/mp3_to_text.py CLIP.mp3        # seeded-random, plumbing only
#   python3 sw/tinyml_reference/mp3_to_text.py                 # synthetic smoke clip
#   python3 sw/tinyml_reference/mp3_to_text.py CLIP.wav --seconds 2.0
#
# The QuartzNet activation arena is QN_MAX_T_OUT output frames
# (firmware/quartznet/quartznet_infer.h, host default 2048 -- see
# firmware/quartznet/Makefile); qn_transcribe.c reports an overrun clearly
# rather than corrupting memory, but --seconds is the fix for a build with a
# smaller arena (e.g. the RV32 firmware image).

import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import quartznet_audio as qa  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / "build" / "quartznet"


def main() -> int:
    ap = argparse.ArgumentParser(
        description="mp3/wav -> build/quartznet/mp3_input.bin (+ t_out sidecar)")
    ap.add_argument("audio", nargs="?", default=None,
                     help="mp3/wav/ogg/flac path; omit for a synthetic smoke clip")
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT),
                     help=f"output directory (default: {DEFAULT_OUT})")
    ap.add_argument("--seconds", type=float, default=None,
                     help="truncate the clip to this many seconds")
    ap.add_argument("--model-dir", type=pathlib.Path, default=None,
                     help="read in_scale/in_zp from <dir>/quartznet_meta.json "
                          "(quartznet_export_int8.py's --out). REQUIRED for "
                          "the real calibrated model -- omit only for the "
                          "seeded-random build/quartznet[_reduced] configs, "
                          "which have no meta.json and for which "
                          "calibrate()/IN_ZP_DEFAULT is the right convention.")
    args = ap.parse_args()

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.audio is None:
        import soundfile as sf
        secs = args.seconds if args.seconds is not None else 2.0
        pcm = qa.synth_clip("voiced", seconds=secs)
        audio_path = out_dir / "_synth_clip.wav"
        sf.write(str(audio_path), pcm, qa.SAMPLE_RATE)
        print(f"no audio file given — wrote a synthetic {secs:.1f}s smoke clip "
              f"to {audio_path}")
        seconds_arg = None  # already the right length; don't truncate twice
    else:
        audio_path = pathlib.Path(args.audio)
        seconds_arg = args.seconds

    scale, zero_point = None, qa.IN_ZP_DEFAULT
    quant_source = "PLACEHOLDER calibrate()/IN_ZP_DEFAULT"
    if args.model_dir is not None:
        meta_path = args.model_dir / "quartznet_meta.json"
        meta = json.loads(meta_path.read_text())
        scale, zero_point = float(meta["in_scale"]), int(meta["in_zp"])
        quant_source = f"real, from {meta_path}"

    res = qa.audio_file_to_int8(audio_path, scale=scale, zero_point=zero_point,
                                seconds=seconds_arg)

    blob_path = out_dir / "mp3_input.bin"
    n = qa.write_input_blob(blob_path, res["int8"])
    tout_path = out_dir / "mp3_input.t_out"
    tout_path.write_text(f"{res['t_out']}\n")

    print(f"{audio_path}")
    print(f"  duration   {res['duration_s']:.3f} s")
    print(f"  quantization scale={res['scale']:.6f} zp={res['zero_point']}  ({quant_source})")
    print(f"  t_in={res['t_in']}  t_out={res['t_out']}")
    print(f"  wrote      {blob_path}  ({n:,} B)")
    print(f"  wrote      {tout_path}  (t_out sidecar)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
