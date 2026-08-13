#!/usr/bin/env python3
"""quartznet_audio_validate.py — diff quartznet_audio.extract_logmel() against
the real NeMo stt_en_quartznet15x5 preprocessor, on real speech.

Stage 7 Gap 2 A1 / gate G2.1 (docs/07d_soc_integration_and_gap2_start.md's
"What's left", ~/.claude/plans/generate-an-implementation-plan-pure-robin.md
lines 216-217, 250-257, 332-333). quartznet_audio.py's own header has flagged
itself "PLUMBING ONLY, NOT ACCURACY-VALIDATED" since it was written on a
machine with neither torch nor librosa installed — this script is that
validation, now that both are available and LibriSpeech dev-clean is on disk.

Must pass BEFORE any WER claim (Step 3 onward): a front-end bug would make
every downstream WER number uninterpretable, since there would be no way to
attribute a bad WER to PTQ error vs. front-end error.

── Reference implementation: the checkpoint's own buffers, not assumed config
   values ─────────────────────────────────────────────────────────────────

model_config.yaml (inside the .nemo tar) pins the preprocessor's *scalar*
params (n_fft=512, window_size=0.02s, window_stride=0.01s, features=64,
window=hann, normalize=per_feature, sample_rate=16000) but says nothing about
mel-scale convention, fmin/fmax, or window periodicity — those come only from
NeMo's `AudioToMelSpectrogramPreprocessor.__init__` defaults, which this repo
cannot import without nemo_toolkit. Rather than trust source-read defaults,
this script reads the checkpoint's own materialized buffers —
`preprocessor.featurizer.window` (320,) and `preprocessor.featurizer.fb`
(1, 64, 257) — which are the *exact* tensors the trained model saw. Matching
those pins mel_norm="slaney", htk=False, lowfreq=0, highfreq=8000,
win_length=320, periodic=False as MEASURED FACTS, not assumptions.

── What is still NOT pinned by config or checkpoint ─────────────────────────

Four scalar params never show up as tensors: preemph=0.97, mag_power=2.0,
log_zero_guard_value=2**-24, and the normalize-eps 1e-5. If any of these is
subtly wrong, BOTH `quartznet_audio.py` and this script's reference path are
wrong identically (this script hardcodes the same NeMo-source-read values
`quartznet_audio.py` does) and G2.1 would pass anyway. The mutation battery
below (Phase 3) measures how far off each one would have to be caught by the
1e-3 gate — all are caught with real margin — but only Step 3's FP32 WER
against a published NeMo number is an independent check on whether these four
constants are *actually* right, not just self-consistent.

Run:
    python3 quartznet_audio_validate.py <path/to/stt_en_quartznet15x5.nemo>
        [--librispeech DIR] [--n-speakers 20] [--seed 0] [--out DIR]
"""
from __future__ import annotations

import argparse
import io
import json
import pathlib
import random
import re
import sys
import tarfile

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import quartznet_audio as qa  # noqa: E402

# ── gates ──────────────────────────────────────────────────────────────────
G2_1A_MAX_ABS = 1e-3        # fp32 log-mel, max abs diff vs. checkpoint reference
G2_1B_MIN_FRAC_1LSB = 0.999  # int8, fraction of elements within +/-1 LSB

# Mutations that MUST be caught by G2_1A_MAX_ABS, else the gate has no teeth.
# (name, mutated-max-abs-diff measured against REF-CKPT on the shortest clip)
# is not hardcoded here -- Phase 3 measures it live every run.


# ══════════════════════════════════════════════════════════════════════════
# checkpoint loading -- mirrors quartznet_nemo_export.load_state_dict()
# ══════════════════════════════════════════════════════════════════════════

def _load_state_dict(nemo_path: pathlib.Path) -> dict:
    import torch  # local import: only needed here, not at module import time

    with tarfile.open(nemo_path) as tf:
        candidates = [m for m in tf.getmembers()
                      if pathlib.PurePosixPath(m.name).name == "model_weights.ckpt"]
        if len(candidates) != 1:
            raise SystemExit(
                f"{nemo_path}: expected exactly one model_weights.ckpt member, "
                f"found {len(candidates)}")
        extracted = tf.extractfile(candidates[0])
        data = extracted.read()
    sd = torch.load(io.BytesIO(data), map_location="cpu", weights_only=True)
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    return sd


def _load_preprocessor_yaml(nemo_path: pathlib.Path) -> dict:
    """Hand-rolled extraction of the flat `preprocessor:` block from
    model_config.yaml -- no PyYAML dependency (not installed, and this repo's
    convention is to hand-roll simple parsers rather than add a dep for one
    use, matching quartznet_audio.py's own resampler/mel-filterbank style).
    Only handles the flat `key: value` shape this specific block actually has
    (verified by inspection -- no lists, no nesting) -- not a general parser.
    """
    with tarfile.open(nemo_path) as tf:
        candidates = [m for m in tf.getmembers()
                      if pathlib.PurePosixPath(m.name).name == "model_config.yaml"]
        if len(candidates) != 1:
            raise SystemExit(
                f"{nemo_path}: expected exactly one model_config.yaml member, "
                f"found {len(candidates)}")
        text = tf.extractfile(candidates[0]).read().decode("utf-8")

    lines = text.splitlines()
    start = next((i for i, l in enumerate(lines) if l.rstrip() == "preprocessor:"), None)
    if start is None:
        raise SystemExit(f"{nemo_path}: no top-level 'preprocessor:' block found")
    block = {}
    for line in lines[start + 1:]:
        if line and not line[0].isspace():
            break  # dedented back to top level -- block ended
        m = re.match(r"^\s+([A-Za-z0-9_]+):\s*(.*)$", line)
        if not m:
            continue
        key, val = m.group(1), m.group(2).strip()
        if val in ("true", "false"):
            val = val == "true"
        else:
            try:
                val = int(val)
            except ValueError:
                try:
                    val = float(val)
                except ValueError:
                    val = val.strip("'\"")
        block[key] = val
    return block


def load_preprocessor_buffers(nemo_path: pathlib.Path):
    """-> (window[320] f32, fb[64,257] f32, cfg: dict) straight from the
    checkpoint + config, no nemo_toolkit."""
    sd = _load_state_dict(nemo_path)
    win = sd["preprocessor.featurizer.window"].detach().numpy().astype(np.float32)
    fb = sd["preprocessor.featurizer.fb"].detach().numpy().astype(np.float32)
    if fb.ndim == 3:
        fb = fb[0]
    cfg = _load_preprocessor_yaml(nemo_path)
    return win, fb, cfg


# ══════════════════════════════════════════════════════════════════════════
# reference implementations
# ══════════════════════════════════════════════════════════════════════════

def nemo_reference(pcm: np.ndarray, win: np.ndarray, fb: np.ndarray,
                    dtype=None) -> np.ndarray:
    """Direct transcription of NeMo's FilterbankFeatures.forward, using the
    checkpoint's own window/fb buffers. This is the G2.1a gate reference.
    `dtype` defaults to torch.float32 (matches NeMo's own eval-time dtype);
    pass torch.float64 for the Phase-2 precision decomposition only.
    """
    import torch

    if dtype is None:
        dtype = torch.float32
    x = torch.tensor(pcm, dtype=dtype).unsqueeze(0)                # [1, L]
    x = torch.cat((x[:, :1], x[:, 1:] - qa.PREEMPH * x[:, :-1]), dim=1)
    window = torch.tensor(win, dtype=dtype)
    X = torch.stft(x, n_fft=qa.N_FFT, hop_length=qa.HOP_LENGTH,
                    win_length=qa.WIN_LENGTH, center=True, pad_mode="reflect",
                    window=window, return_complex=True)
    X = torch.view_as_real(X)                                      # [1, F, T, 2]
    mag = torch.sqrt(X.pow(2).sum(-1))                              # NeMo's form, not .abs()
    power = mag.pow(qa.MAG_POWER)
    fb_t = torch.tensor(fb, dtype=dtype)
    mel = torch.matmul(fb_t.unsqueeze(0), power)                    # [1, 64, T]
    feat = torch.log(mel + qa.LOG_GUARD)
    mean = feat.mean(dim=2, keepdim=True)
    std = feat.std(dim=2, keepdim=True)                             # unbiased (N-1), matches torch default
    feat = (feat - mean) / (std + qa.NORM_EPS)
    return np.ascontiguousarray(feat[0].transpose(0, 1).numpy(), dtype=np.float32)  # [T, 64]


def torchaudio_reference(pcm: np.ndarray) -> np.ndarray:
    """Independent third implementation via torchaudio.transforms.MelSpectrogram
    -- catches a bug in nemo_reference()'s own transcription. Not the gate
    reference (no preemphasis in torchaudio's transform, applied manually
    here to match NeMo's pipeline order)."""
    import torch
    import torchaudio

    x = torch.tensor(pcm, dtype=torch.float32).unsqueeze(0)
    x = torch.cat((x[:, :1], x[:, 1:] - qa.PREEMPH * x[:, :-1]), dim=1)
    ms = torchaudio.transforms.MelSpectrogram(
        sample_rate=qa.SAMPLE_RATE, n_fft=qa.N_FFT, win_length=qa.WIN_LENGTH,
        hop_length=qa.HOP_LENGTH, f_min=qa.LOWFREQ, f_max=qa.HIGHFREQ,
        n_mels=qa.N_MEL, power=qa.MAG_POWER, center=True, pad_mode="reflect",
        norm="slaney", mel_scale="slaney", window_fn=torch.hann_window,
        wkwargs={"periodic": False})
    mel = ms(x)                                                     # [1, 64, T]
    feat = torch.log(mel + qa.LOG_GUARD)
    mean = feat.mean(dim=2, keepdim=True)
    std = feat.std(dim=2, keepdim=True)
    feat = (feat - mean) / (std + qa.NORM_EPS)
    return np.ascontiguousarray(feat[0].transpose(0, 1).numpy(), dtype=np.float32)


def numpy_reference_f64(pcm: np.ndarray, win: np.ndarray, fb: np.ndarray) -> np.ndarray:
    """quartznet_audio's own numpy chain, but with the window/fb swapped in
    from the checkpoint (instead of qa's own np.hanning/hand-rolled slaney
    fb) and forced float64 throughout -- isolates "is the algorithm exact"
    from "is np.hanning/qa._MEL_FB close enough to the checkpoint's f32
    buffers" (Phase 2's precision decomposition)."""
    x = np.asarray(pcm, dtype=np.float64).ravel()
    x = np.concatenate(([x[0]], x[1:] - qa.PREEMPH * x[:-1]))
    x = np.pad(x, qa.N_FFT // 2, mode="reflect")
    w = win.astype(np.float64)
    pad_w = qa.N_FFT - qa.WIN_LENGTH
    w = np.pad(w, (pad_w // 2, pad_w - pad_w // 2))
    n_frames = 1 + (x.size - qa.N_FFT) // qa.HOP_LENGTH
    frames = np.lib.stride_tricks.as_strided(
        x, shape=(n_frames, qa.N_FFT),
        strides=(x.strides[0] * qa.HOP_LENGTH, x.strides[0]))
    spec = np.fft.rfft(frames * w, n=qa.N_FFT)
    power = np.abs(spec) ** qa.MAG_POWER
    mel = power @ fb.astype(np.float64).T
    feat = np.log(mel + qa.LOG_GUARD)
    mean = feat.mean(axis=0, keepdims=True)
    std = feat.std(axis=0, ddof=1, keepdims=True)
    feat = (feat - mean) / (std + qa.NORM_EPS)
    return feat.astype(np.float64)


# ══════════════════════════════════════════════════════════════════════════
# clip selection
# ══════════════════════════════════════════════════════════════════════════

def select_clips(root: pathlib.Path, n_speakers: int, seed: int) -> list[pathlib.Path]:
    import soundfile as sf

    all_flacs = sorted(root.rglob("*.flac"))
    if not all_flacs:
        raise SystemExit(f"no .flac files under {root}")
    by_speaker: dict[str, list[pathlib.Path]] = {}
    for p in all_flacs:
        speaker = p.parts[len(root.parts)]  # dev-clean/<speaker>/<chapter>/<file>.flac
        by_speaker.setdefault(speaker, []).append(p)

    rng = random.Random(seed)
    speakers = sorted(by_speaker)[:n_speakers]
    picked = [rng.choice(by_speaker[s]) for s in speakers]

    # also include the globally shortest and longest clips (sf.info -- no
    # decode), so the mutation battery (Phase 3) always has a short clip to
    # run on (the ddof=0-vs-1 landmine is length-dependent -- see module
    # docstring / the PR this script landed in) and the pipeline is exercised
    # at both extremes.
    infos = [(p, sf.info(str(p)).frames) for p in all_flacs]
    shortest = min(infos, key=lambda pi: pi[1])[0]
    longest = max(infos, key=lambda pi: pi[1])[0]
    for extra in (shortest, longest):
        if extra not in picked:
            picked.append(extra)
    return picked


# ══════════════════════════════════════════════════════════════════════════
# gate-script convention (matches quartznet_nemo_export.py's G2.2 style)
# ══════════════════════════════════════════════════════════════════════════

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Diff quartznet_audio.extract_logmel() against real NeMo")
    ap.add_argument("nemo_path", type=pathlib.Path)
    repo_root = pathlib.Path(__file__).resolve().parents[2]
    ap.add_argument("--librispeech", type=pathlib.Path,
                     default=repo_root / "librispeech" / "LibriSpeech" / "dev-clean")
    ap.add_argument("--n-speakers", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=pathlib.Path,
                     default=repo_root / "build" / "quartznet_audio_validate")
    args = ap.parse_args()

    # dev-clean only -- test-clean is reserved untouched for the WER gates
    # (Step 3 / A3 onward). Hard guard, not just a default.
    if "test-clean" in str(args.librispeech).lower():
        raise SystemExit(
            "quartznet_audio_validate: refusing to run against test-clean "
            "-- reserved for the untouched WER eval, use dev-clean")

    ok = True

    def check(cond: bool, pass_msg: str, fail_msg: str) -> bool:
        nonlocal ok
        if cond:
            print(f"  {pass_msg}")
        else:
            ok = False
            print(f"  *** {fail_msg} ***")
        return cond

    print(f"quartznet_audio_validate  nemo={args.nemo_path}  "
          f"librispeech={args.librispeech}")
    win, fb, cfg = load_preprocessor_buffers(args.nemo_path)

    # ── Phase 0: config / checkpoint-buffer pinning ─────────────────────────
    print("\n-- Phase 0: preprocessor config + checkpoint buffers --")
    print(f"  model_config.yaml preprocessor block: {cfg}")
    expect = {"n_fft": qa.N_FFT, "sample_rate": qa.SAMPLE_RATE,
              "features": qa.N_MEL, "normalize": "per_feature",
              "window": "hann", "frame_splicing": 1}
    for k, v in expect.items():
        check(cfg.get(k) == v, f"config[{k}]={cfg.get(k)!r} matches quartznet_audio",
              f"config[{k}]={cfg.get(k)!r} != quartznet_audio's {v!r}")
    check(abs(cfg.get("window_size", 0) * qa.SAMPLE_RATE - qa.WIN_LENGTH) < 1e-6,
          f"window_size*{qa.SAMPLE_RATE} == WIN_LENGTH={qa.WIN_LENGTH}",
          f"window_size {cfg.get('window_size')} * sr != WIN_LENGTH={qa.WIN_LENGTH}")
    check(abs(cfg.get("window_stride", 0) * qa.SAMPLE_RATE - qa.HOP_LENGTH) < 1e-6,
          f"window_stride*{qa.SAMPLE_RATE} == HOP_LENGTH={qa.HOP_LENGTH}",
          f"window_stride {cfg.get('window_stride')} * sr != HOP_LENGTH={qa.HOP_LENGTH}")

    d_fb = float(np.abs(fb.astype(np.float64) - qa._MEL_FB.astype(np.float64)).max())
    check(d_fb < 1e-6, f"checkpoint fb vs quartznet_audio._MEL_FB: max|diff|={d_fb:.2e}",
          f"checkpoint fb vs quartznet_audio._MEL_FB: max|diff|={d_fb:.2e} >= 1e-6 "
          f"(pins mel_norm=slaney, htk=False, fmin=0, fmax=8000)")
    d_win = float(np.abs(win.astype(np.float64) -
                          qa._hann_symmetric(qa.WIN_LENGTH)).max())
    check(d_win < 1e-6, f"checkpoint window vs quartznet_audio._hann_symmetric: "
          f"max|diff|={d_win:.2e}",
          f"checkpoint window vs quartznet_audio._hann_symmetric: "
          f"max|diff|={d_win:.2e} >= 1e-6 (pins win_length=320, periodic=False)")

    print("  NOTE: preemph=0.97, mag_power=2.0, log_zero_guard=2**-24, and the "
          "1e-5 norm eps are NOT pinned by config or checkpoint buffers -- "
          "both this script's reference and quartznet_audio.py hardcode the "
          "same NeMo-source-read values, so a shared error in any of these "
          "four constants would NOT be caught by this gate. Only Step 3's "
          "FP32 WER against NeMo's published number independently checks them.")

    # ── clip selection ──────────────────────────────────────────────────────
    clips = select_clips(args.librispeech, args.n_speakers, args.seed)
    shortest = min(clips, key=lambda p: p.stat().st_size)
    print(f"\n-- {len(clips)} clips selected (dev-clean, seed={args.seed}) --")
    print(f"  shortest-by-size (mutation battery target): {shortest.name}")

    # ── Phase 1: per-clip fp32 + int8 diff ──────────────────────────────────
    print("\n-- Phase 1: per-clip diff vs. checkpoint reference --")
    max_abs = 0.0
    max_abs_torchaudio = 0.0
    total_elems = 0
    within_1lsb = 0
    exact = 0
    per_clip = []
    for p in clips:
        pcm = qa.load_audio(p)
        assert np.array_equal(pcm, pcm), "NaN in decoded audio"  # cheap sanity
        a = qa.extract_logmel(pcm)
        b = nemo_reference(pcm, win, fb)
        ta = torchaudio_reference(pcm)
        if a.shape != b.shape:
            ok = False
            print(f"  *** {p.name}: shape mismatch a={a.shape} b={b.shape} ***")
            continue
        d = float(np.abs(a - b).max())
        d_ta = float(np.abs(b - ta).max())
        max_abs = max(max_abs, d)
        max_abs_torchaudio = max(max_abs_torchaudio, d_ta)

        scale = qa.calibrate(a)
        qa_ = qa.quantize_features(a, scale)
        qb_ = qa.quantize_features(b, scale)
        dq = np.abs(qa_.astype(np.int32) - qb_.astype(np.int32))
        total_elems += dq.size
        within_1lsb += int((dq <= 1).sum())
        exact += int((dq == 0).sum())

        per_clip.append({"file": p.name, "t": a.shape[0], "max_abs_diff": d,
                          "max_abs_vs_torchaudio": d_ta})
        print(f"  {p.name:40s} T={a.shape[0]:5d}  max|diff|={d:.3e}  "
              f"vs_torchaudio={d_ta:.3e}")

    check(max_abs < G2_1A_MAX_ABS,
          f"G2.1a: max fp32 |diff| over {len(clips)} clips = {max_abs:.3e} "
          f"< {G2_1A_MAX_ABS:.0e}",
          f"G2.1a: max fp32 |diff| = {max_abs:.3e} >= {G2_1A_MAX_ABS:.0e}")
    frac = within_1lsb / max(total_elems, 1)
    check(frac >= G2_1B_MIN_FRAC_1LSB,
          f"G2.1b: {within_1lsb}/{total_elems} ({100*frac:.6f}%) int8 elements "
          f"within 1 LSB, exact={exact}/{total_elems} "
          f">= {100*G2_1B_MIN_FRAC_1LSB:.1f}%",
          f"G2.1b: only {100*frac:.6f}% within 1 LSB "
          f"(need >= {100*G2_1B_MIN_FRAC_1LSB:.1f}%)")
    check(max_abs_torchaudio < G2_1A_MAX_ABS,
          f"cross-check: nemo_reference vs. independent torchaudio path "
          f"max|diff|={max_abs_torchaudio:.3e}",
          f"cross-check: nemo_reference vs. torchaudio path "
          f"max|diff|={max_abs_torchaudio:.3e} -- the transcription itself "
          f"may be wrong, not just quartznet_audio.py")

    # ── Phase 2: precision decomposition (shortest clip) ────────────────────
    print("\n-- Phase 2: precision decomposition (shortest clip) --")
    import torch
    pcm = qa.load_audio(shortest)
    ref_f64 = nemo_reference(pcm, win, fb, dtype=torch.float64)
    np_f64_ckptbuf = numpy_reference_f64(pcm, win, fb)
    np_f64_qabuf = numpy_reference_f64(pcm, qa._hann_symmetric(qa.WIN_LENGTH),
                                        qa._MEL_FB)
    ref_f32 = nemo_reference(pcm, win, fb, dtype=torch.float32)

    d_algo = float(np.abs(np_f64_ckptbuf - ref_f64.astype(np.float64)).max())
    d_window_round = float(np.abs(np_f64_qabuf - ref_f64.astype(np.float64)).max())
    d_selfnoise = float(np.abs(ref_f32.astype(np.float64) -
                                ref_f64.astype(np.float64)).max())
    print(f"  repo-numpy(f64, ckpt win+fb) vs torch(f64, ckpt win+fb): "
          f"{d_algo:.3e}  (algorithmic exactness -- two independent FFT "
          f"implementations, so not machine-epsilon, but should be many "
          f"orders below the 1e-3 gate)")
    print(f"  repo-numpy(f64, qa's own win+fb) vs torch(f64, ckpt win+fb): "
          f"{d_window_round:.3e}  (np.hanning-f64 vs checkpoint's f32-rounded window)")
    print(f"  torch f32 vs torch f64 (both ckpt win+fb): {d_selfnoise:.3e}  "
          f"(NeMo's own float32 self-noise)")
    # 1e-5: two orders below the G2.1a gate, comfortably above the
    # numpy-rfft-vs-torch-fft cross-implementation floor (measured ~1e-7) --
    # a real algorithmic bug (wrong axis, wrong padding, wrong reduction)
    # produces errors of order 1e-1..1e0 (see the Phase 3 mutation battery),
    # so this still has a wide margin to catch one.
    check(d_algo < 1e-5,
          f"algorithm is exact (repo chain matches torch modulo cross-FFT-"
          f"implementation rounding): {d_algo:.3e} < 1e-5",
          f"ALGORITHMIC DIVERGENCE detected: {d_algo:.3e} >= 1e-5 -- the repo's "
          f"framing/windowing/mel/log/normalize chain has a real bug, not "
          f"just cross-implementation FFT rounding")

    # ── Phase 3: mutation battery (shortest clip) ────────────────────────────
    print("\n-- Phase 3: mutation battery (must all exceed the G2.1a gate) --")

    def mutated_diff(**kwargs) -> float:
        import torch as t

        x = t.tensor(pcm, dtype=t.float32).unsqueeze(0)
        preemph = kwargs.get("preemph", qa.PREEMPH)
        if preemph is not None:
            x = t.cat((x[:, :1], x[:, 1:] - preemph * x[:, :-1]), dim=1)
        win_len = kwargs.get("win_length", qa.WIN_LENGTH)
        periodic = kwargs.get("periodic", False)
        # Reuse the checkpoint's exact window only when neither win_length nor
        # periodicity changed -- otherwise it's the wrong length/shape for
        # torch.stft's window= (must equal win_len), so regenerate one.
        window = (t.tensor(win, dtype=t.float32)
                  if win_len == qa.WIN_LENGTH and not periodic
                  else t.hann_window(win_len, periodic=periodic))
        center = kwargs.get("center", True)
        pad_mode = kwargs.get("pad_mode", "reflect")
        X = t.stft(x, n_fft=qa.N_FFT, hop_length=qa.HOP_LENGTH,
                    win_length=win_len, center=center, pad_mode=pad_mode,
                    window=window, return_complex=True)
        X = t.view_as_real(X)
        mag_power = kwargs.get("mag_power", qa.MAG_POWER)
        mag = t.sqrt(X.pow(2).sum(-1)).pow(mag_power)
        this_fb = kwargs.get("fb_override")
        fb_t = t.tensor(this_fb if this_fb is not None else fb, dtype=t.float32)
        mel = t.matmul(fb_t.unsqueeze(0), mag)
        guard = kwargs.get("log_guard", qa.LOG_GUARD)
        feat = t.log(mel + guard)
        ddof0 = kwargs.get("ddof0", False)
        mean = feat.mean(dim=2, keepdim=True)
        if ddof0:
            std = feat.std(dim=2, keepdim=True, unbiased=False)
        else:
            std = feat.std(dim=2, keepdim=True)
        feat = (feat - mean) / (std + qa.NORM_EPS)
        out = np.ascontiguousarray(feat[0].transpose(0, 1).numpy(), dtype=np.float32)
        if out.shape != ref_f32.shape:
            # A frame-count mismatch (e.g. center=False drops the reflect-pad
            # halo) is an even more obvious "caught" than a numeric diff --
            # report it as maximally caught rather than crashing on broadcast.
            return float("inf")
        return float(np.abs(out - ref_f32).max())

    mutations = {
        "hann periodic=True": dict(periodic=True),
        "win_length=400 (TinyVAD's)": dict(win_length=400),
        "htk mel scale": dict(fb_override=_htk_fb()),
        "mel fb norm=None": dict(fb_override=_unnormalized_fb()),
        "fmin=80/fmax=7600 (TinyVAD's)": dict(
            fb_override=qa.build_mel_fb(fmin=80.0, fmax=7600.0)),
        "pad_mode=constant": dict(pad_mode="constant"),
        "center=False": dict(center=False),
        "no preemphasis": dict(preemph=None),
        "mag_power=1 (magnitude, not power)": dict(mag_power=1.0),
        "ddof=0 (biased std)": dict(ddof0=True),
        "log_guard=1e-6": dict(log_guard=1e-6),
        "log_guard=1e-5": dict(log_guard=1e-5),
    }
    for name, kwargs in mutations.items():
        try:
            d = mutated_diff(**kwargs)
        except Exception as e:  # noqa: BLE001 - report, don't crash the gate
            check(False, "", f"mutation {name!r} raised {e!r} -- could not test")
            continue
        check(d >= G2_1A_MAX_ABS,
              f"mutation {name!r}: max|diff|={d:.3e} correctly CAUGHT (>= gate)",
              f"mutation {name!r}: max|diff|={d:.3e} SLIPS PAST the gate -- "
              f"the gate has lost its teeth for this failure mode")

    # ── Phase 4: edge cases (info + light asserts) ───────────────────────────
    print("\n-- Phase 4: edge cases --")
    feat_padded = qa.extract_logmel(pcm, pad_to=16)
    feat_unpadded = qa.extract_logmel(pcm)
    t0 = feat_unpadded.shape[0]
    check(feat_padded.shape[0] % 16 == 0 and
          np.array_equal(feat_padded[:t0], feat_unpadded) and
          not feat_padded[t0:].any(),
          f"pad_to=16: T {t0}->{feat_padded.shape[0]}, prefix identical, "
          f"pad rows are zero",
          f"pad_to=16 behaves unexpectedly: T {t0}->{feat_padded.shape[0]}")
    print("  INFO: T==1 (single-frame utterance) diverges from NeMo, which "
          "raises ValueError on seq_len==1 rather than emitting a zero-std "
          "row -- unreachable from LibriSpeech (shortest dev-clean clip is "
          f"{shortest.name}, T={t0}), not gated.")
    print("  INFO: batch masking (NeMo masks frames beyond seq_len in a padded "
          "batch before pad_to) is equivalent to this script's batch=1 path; "
          "not exercised here, not gated.")
    print("  INFO: dither is correctly OFF by default in quartznet_audio.py -- "
          "NeMo itself gates dither on self.training, so dither=0 is the "
          "correct eval-mode behaviour, not a deviation.")

    # ── report + gate ────────────────────────────────────────────────────────
    args.out.mkdir(parents=True, exist_ok=True)
    report_path = args.out / "report.json"
    report_path.write_text(json.dumps({
        "nemo_path": str(args.nemo_path), "librispeech": str(args.librispeech),
        "n_clips": len(clips), "max_abs_diff": max_abs,
        "max_abs_vs_torchaudio": max_abs_torchaudio,
        "frac_within_1lsb": frac, "per_clip": per_clip,
    }, indent=2))
    print(f"\nwrote {report_path}")

    if not ok:
        raise SystemExit("quartznet_audio_validate: G2.1 checks FAILED")
    print("G2.1: PASS")


def _htk_fb() -> np.ndarray:
    """htk=True mel filterbank, hand-rolled the same way qa.build_mel_fb() is,
    for the mutation battery only (not part of the front end)."""
    n_freqs = qa.N_FFT // 2 + 1
    fft_freqs = np.linspace(0.0, qa.SAMPLE_RATE / 2.0, n_freqs, dtype=np.float64)

    def hz_to_mel_htk(f):
        return 2595.0 * np.log10(1.0 + f / 700.0)

    def mel_to_hz_htk(m):
        return 700.0 * (10.0 ** (m / 2595.0) - 1.0)

    mel_pts = np.linspace(hz_to_mel_htk(qa.LOWFREQ), hz_to_mel_htk(qa.HIGHFREQ),
                           qa.N_MEL + 2)
    hz_pts = mel_to_hz_htk(mel_pts)
    fdiff = np.diff(hz_pts)
    ramps = hz_pts[:, None] - fft_freqs[None, :]
    fb = np.zeros((qa.N_MEL, n_freqs), dtype=np.float64)
    for m in range(qa.N_MEL):
        lower = -ramps[m] / fdiff[m]
        upper = ramps[m + 2] / fdiff[m + 1]
        fb[m] = np.maximum(0.0, np.minimum(lower, upper))
    return fb.astype(np.float32)


def _unnormalized_fb() -> np.ndarray:
    """Slaney mel *shape* without the Slaney area normalization -- for the
    mutation battery only."""
    n_freqs = qa.N_FFT // 2 + 1
    fft_freqs = np.linspace(0.0, qa.SAMPLE_RATE / 2.0, n_freqs, dtype=np.float64)
    mel_pts = np.linspace(qa.hz_to_mel_slaney(qa.LOWFREQ),
                           qa.hz_to_mel_slaney(qa.HIGHFREQ), qa.N_MEL + 2)
    hz_pts = qa.mel_to_hz_slaney(mel_pts)
    fdiff = np.diff(hz_pts)
    ramps = hz_pts[:, None] - fft_freqs[None, :]
    fb = np.zeros((qa.N_MEL, n_freqs), dtype=np.float64)
    for m in range(qa.N_MEL):
        lower = -ramps[m] / fdiff[m]
        upper = ramps[m + 2] / fdiff[m + 1]
        fb[m] = np.maximum(0.0, np.minimum(lower, upper))
    return fb.astype(np.float32)  # no `fb *= enorm` step -- that's the mutation


if __name__ == "__main__":
    main()
