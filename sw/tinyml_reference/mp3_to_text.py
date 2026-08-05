# mp3_to_text.py — glue: mp3/wav file -> int8 log-mel blob for firmware/quartznet.
#
# Bridges quartznet_audio.py's front end (until now a self-tested island, never
# called from anywhere else in the repo) to the C interpreter's input path.
# Writes build/quartznet/mp3_input.bin — deliberately a different filename from
# quartznet_ref.py's quartznet_input.bin, so `make goldens` output is never
# clobbered — plus a `mp3_input.t_out` sidecar, since t_out varies with clip
# length and firmware/quartznet/qn_transcribe.c needs it for qn_load().
#
# Weights/qparams are NOT touched here: qn_transcribe reuses `make goldens`'
# seeded-random blobs unmodified. The resulting transcript is expected to be
# gibberish (the qparams were calibrated against random input, not this one) —
# this script only proves the mp3 -> features -> blob -> interpreter plumbing.
#
# Usage:
#   python3 sw/tinyml_reference/mp3_to_text.py CLIP.mp3
#   python3 sw/tinyml_reference/mp3_to_text.py                # synthetic smoke clip
#   python3 sw/tinyml_reference/mp3_to_text.py CLIP.wav --seconds 2.0
#
# The QuartzNet activation arena is capped at QN_MAX_T_OUT=128 output frames
# (firmware/quartznet/quartznet_infer.h) — clips longer than ~2.5s will exceed
# it; qn_transcribe.c reports this clearly rather than corrupting memory, but
# --seconds is the fix.

import argparse
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
                     help="truncate the clip to this many seconds "
                          "(keep under ~2.5s: QN_MAX_T_OUT=128 output frames)")
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

    res = qa.audio_file_to_int8(audio_path, seconds=seconds_arg)

    blob_path = out_dir / "mp3_input.bin"
    n = qa.write_input_blob(blob_path, res["int8"])
    tout_path = out_dir / "mp3_input.t_out"
    tout_path.write_text(f"{res['t_out']}\n")

    print(f"{audio_path}")
    print(f"  duration   {res['duration_s']:.3f} s")
    print(f"  t_in={res['t_in']}  t_out={res['t_out']}  "
          f"(QN_MAX_T_OUT=128{' -- EXCEEDED, truncate with --seconds' if res['t_out'] > 128 else ''})")
    print(f"  wrote      {blob_path}  ({n:,} B)")
    print(f"  wrote      {tout_path}  (t_out sidecar)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
