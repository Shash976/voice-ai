# quartznet_audio.py
#
# Audio front end for the QuartzNet 15x5 ASR accelerator: mp3/wav/ogg/flac on
# disk  ->  16 kHz mono float32 PCM  ->  [time, 64] log-mel  ->  int8 blob in
# exactly the layout `quartznet_input.bin` uses.
#
# Run:
#   python3 sw/tinyml_reference/quartznet_audio.py --self-test
#   python3 sw/tinyml_reference/quartznet_audio.py CLIP.mp3 [CLIP2.wav ...] \
#           [--out DIR] [--seconds N] [--pad-to 16]
#
# ── Status: NUMERICALLY VALIDATED against the real checkpoint (Stage 7 Gap 2
#    A1 / gate G2.1, `quartznet_audio_validate.py`) ───────────────────────────
#
#   This module reproduces NeMo's `AudioToMelSpectrogramPreprocessor` in plain
#   numpy. `quartznet_audio_validate.py` diffs it against a direct torch.stft
#   transcription that reads the *checkpoint's own* window/mel-filterbank
#   buffers (`preprocessor.featurizer.{window,fb}` inside
#   `stt_en_quartznet15x5.nemo`) — i.e. the exact tensors the trained model
#   saw, not assumed config defaults. Measured on 22 real LibriSpeech
#   dev-clean clips (1.4s-32.6s, 40 speakers): max fp32 log-mel diff
#   1.015e-04 (gate: <1e-3), int8-quantized agreement 100.000% within 1 LSB
#   (gate: >=99.9%). A 12-mutation battery (wrong window periodicity/length,
#   htk vs. slaney mel, wrong fmin/fmax, wrong pad mode, no centering, no
#   preemphasis, wrong mag_power, biased-vs-unbiased std, wrong log guard)
#   confirms the gate actually has teeth — every mutation is caught with
#   diffs >=1.4e-2, two-plus orders above the gate threshold.
#
#   NOT independently checked by this gate — both this module and its
#   validator hardcode the same NeMo-source-read values, so a shared error
#   would pass silently: preemph=0.97, mag_power=2.0, log_zero_guard=2**-24,
#   the 1e-5 normalization epsilon. None of these appear as checkpoint
#   tensors. The only independent check on these four is Step 3's FP32 WER
#   against NeMo's published number.
#
#   Run `python3 quartznet_audio_validate.py <path/to/*.nemo>` to reproduce.
#
# ── Source of truth for every constant ───────────────────────────────────────
#
#   NeMo v1.23.0 `examples/asr/conf/quartznet/quartznet_15x5.yaml`, preprocessor
#   block (fetched and read, not assumed):
#
#       _target_      : nemo.collections.asr.modules.AudioToMelSpectrogramPreprocessor
#       sample_rate   : 16000
#       window_size   : 0.02          -> 320 samples
#       window_stride : 0.01          -> 160 samples   (== FPS_IN 100)
#       window        : "hann"
#       features      : 64            (== N_MEL)
#       n_fft         : 512
#       frame_splicing: 1
#       normalize     : "per_feature"
#       dither        : 0.00001
#
#   Everything else falls through to `AudioToMelSpectrogramPreprocessor.__init__`
#   defaults (nemo/collections/asr/modules/audio_preprocessing.py):
#
#       preemph=0.97, lowfreq=0, highfreq=None (-> sr/2 = 8000), log=True,
#       log_zero_guard_type="add", log_zero_guard_value=2**-24, mag_power=2.0,
#       mel_norm="slaney", pad_to=16, pad_value=0, exact_pad=False
#
#   and the implementation details from
#   `nemo/collections/asr/parts/preprocessing/features.py`:
#
#       window  = torch.hann_window(win_length, periodic=False)   <- NOT periodic
#       stft    = torch.stft(center=True, pad_mode="reflect")     <- reflect pad n_fft//2
#       fb      = librosa.filters.mel(sr, n_fft, n_mels, fmin, fmax, norm="slaney")
#                 -> librosa's default htk=False, i.e. the SLANEY mel scale
#       preemph = cat(x[:1], x[1:] - 0.97 * x[:-1])
#       forward : dither -> preemph -> stft -> |X| -> **2 -> mel -> log -> normalize
#       per_feature normalize: (x - mean) / (std + 1e-5), per mel bin, over time,
#                              with torch's UNBIASED std (N-1 denominator)
#
# ── Three ways this differs from TinyVAD's extract_logmel() ──────────────────
#
#   Do not copy assumptions across from `speech_simulator.py`.  It is a different
#   model with a different front end:
#
#     1. MEL SCALE.  TinyVAD uses HTK mel with no filterbank normalisation and
#        fmin=80/fmax=7600.  QuartzNet uses the SLANEY mel scale with Slaney area
#        normalisation and fmin=0/fmax=8000.  Different filter shapes AND
#        different filter gains.
#     2. WINDOW.  TinyVAD uses a 400-sample PERIODIC Hann.  QuartzNet uses a
#        320-sample SYMMETRIC (periodic=False) Hann.
#     3. NORMALISATION.  TinyVAD emits raw log(mel + 1e-6).  QuartzNet z-scores
#        EACH MEL BIN over the utterance's own time axis.  This is the big one —
#        see the streaming note below.
#
# ── !! Streaming consequence of "per_feature" normalisation !! ───────────────
#
#   `normalize="per_feature"` makes the front end NON-CAUSAL at utterance scale:
#   the mean and std of every mel bin are taken over the whole utterance, so no
#   frame can be normalised (and therefore quantised) until the last frame has
#   been seen.  A streaming implementation must either buffer the entire
#   utterance, or substitute a running/global normaliser and accept the accuracy
#   delta.  This is a front-end constraint, not an accelerator one — the
#   accelerator never sees unnormalised data — but it does mean the mel front end
#   cannot be pipelined frame-by-frame into the first conv the way TinyVAD's can.
#
# ── Tensor layout ────────────────────────────────────────────────────────────
#
#   [time, channel] everywhere, matching quartznet_topology.py and CLAUDE.md.
#   NOT [channel, time].  NeMo/torch produce [channel, time]; the transpose
#   happens exactly once, at the end of extract_logmel(), and is the only place
#   layout is allowed to be ambiguous.
#
# ── C portability ────────────────────────────────────────────────────────────
#
#   Written the way `speech_simulator.extract_logmel()` is written: explicit
#   numpy, every step with a direct C equivalent, so this doubles as the golden
#   reference for a future C feature extractor.  Two caveats for that port:
#     * `resample_poly()` is a real FIR polyphase resampler.  If the capture path
#       is already 16 kHz it is never called; the C port can skip it entirely.
#     * per-feature normalisation needs two passes over the utterance (see above).

from __future__ import annotations

import argparse
import math
import pathlib
import sys

import numpy as np

# ── constants (all verified against the NeMo config, see header) ─────────────

SAMPLE_RATE   = 16000
N_MEL         = 64          # preprocessor `features`      == topology N_MEL
N_FFT         = 512         # preprocessor `n_fft`
WIN_LENGTH    = 320         # window_size 0.02 s * 16000
HOP_LENGTH    = 160         # window_stride 0.01 s * 16000 == FPS_IN 100
LOWFREQ       = 0.0         # __init__ default
HIGHFREQ      = 8000.0      # __init__ default (None -> sample_rate / 2)
PREEMPH       = 0.97        # __init__ default
MAG_POWER     = 2.0         # __init__ default
LOG_GUARD     = 2.0 ** -24  # log_zero_guard_value, log_zero_guard_type="add"
NORM_EPS      = 1e-5        # features.py CONSTANT, added to std
DITHER        = 1e-5        # config value; DISABLED by default here, see below
PAD_TO        = 16          # __init__ default; disabled by default here, see below

# Mel input zero point.  Must equal the `in_zp` the descriptor table was built
# with — quartznet_descriptors.build_table(in_zp=-20).  The model computes
# (q - in_zp), so the affine convention is  real ~= scale * (q - in_zp).
# `_check_in_zp_matches_descriptors()` asserts this has not drifted.
IN_ZP_DEFAULT = -20

# Placeholder calibration target, in int8 counts.  Same knob and same default as
# quartznet_ref.make_blobs(target=48.0): the 99.5th percentile of |value| is
# mapped to this many counts.  48 of a possible 127 leaves ~2.6x headroom for the
# tail, which is what keeps the 0.5% of frames above the percentile from
# clipping hard.
CALIB_TARGET  = 48.0
CALIB_PCT     = 99.5


# ══════════════════════════════════════════════════════════════════════════════
# 1. decode + resample
# ══════════════════════════════════════════════════════════════════════════════

def resample_poly(x: np.ndarray, sr_in: int, sr_out: int,
                  half_zeros: int = 16, beta: float = 5.0) -> np.ndarray:
    """Rational-ratio FIR polyphase resampler.  Equivalent to scipy's
    `signal.resample_poly`, reimplemented because scipy is not a dependency.

    up/down are reduced by their gcd, so 44100 -> 16000 becomes 160/441.  The
    anti-alias prototype is a Kaiser-windowed sinc at cutoff
    `1 / max(up, down)` of the interpolated-rate Nyquist, normalised to unity DC
    gain and then scaled by `up` to undo the zero-stuffing loss.

    Output sample m is  y[m] = sum_n x[n] * h[m*down + half_taps - n*up],
    which is the standard upfirdn recurrence with the filter's group delay
    (`half_taps` taps at the interpolated rate) folded into the index so input
    and output timelines stay aligned: a unit impulse at x[0] lands on y[0].
    """
    if sr_in == sr_out:
        return x.astype(np.float32, copy=False)
    if x.size == 0:
        return x.astype(np.float32, copy=False)

    g    = math.gcd(int(sr_in), int(sr_out))
    up   = int(sr_out) // g
    down = int(sr_in) // g

    # prototype lowpass, odd length so the peak sits on an exact tap
    half_taps = half_zeros * max(up, down)
    n_taps    = 2 * half_taps + 1
    t         = np.arange(n_taps, dtype=np.float64) - half_taps
    cutoff    = 1.0 / max(up, down)                 # in cycles/sample*2 (Nyquist units)
    h         = cutoff * np.sinc(cutoff * t) * np.kaiser(n_taps, beta)
    h        *= up / h.sum()                        # unity DC gain, then zero-stuff gain

    # polyphase decomposition: hp[r, i] = h_padded[r + i*up]
    n_phase = int(math.ceil(n_taps / up))
    h_pad   = np.zeros(n_phase * up, dtype=np.float64)
    h_pad[:n_taps] = h
    hp = h_pad.reshape(n_phase, up).T.copy()        # [up, n_phase]

    n_out = int(math.ceil(x.size * up / down))

    # pad the input so every gather is in bounds: index n runs from n_max down to
    # n_max - n_phase + 1, and n_max can overshoot the end by the group delay.
    j_max  = (n_out - 1) * down + half_taps
    n_high = j_max // up
    xp = np.zeros(n_phase + max(n_high + 1, x.size), dtype=np.float64)
    xp[n_phase:n_phase + x.size] = x                # xp[n + n_phase] == x[n]

    y   = np.empty(n_out, dtype=np.float64)
    lag = np.arange(n_phase)

    # chunked so the [block, n_phase] gather matrix stays small
    block = max(1, (1 << 22) // max(n_phase, 1))
    for s in range(0, n_out, block):
        m     = np.arange(s, min(s + block, n_out))
        j     = m * down + half_taps
        n_max = j // up
        r     = j % up
        idx   = (n_max[:, None] - lag[None, :]) + n_phase      # into xp
        y[s:s + m.size] = np.einsum("bi,bi->b", xp[idx], hp[r])

    return y.astype(np.float32)


def load_audio(path, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Decode any libsndfile-readable file -> mono float32 PCM at `sample_rate`.

    Unlike `speech_simulator.load_audio()`, which raises on a sample-rate
    mismatch, this resamples: real mp3s are 44.1/48 kHz and rejecting them would
    make the front end useless.  mp3 decoding comes from libsndfile >= 1.1 (the
    `soundfile` wheel bundles 1.2.0), so no ffmpeg and no system package.
    """
    import soundfile as sf                    # imported late: optional dependency

    audio, sr = sf.read(str(path), dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)                # downmix, same rule as TinyVAD
    return resample_poly(audio, sr, sample_rate)


# ══════════════════════════════════════════════════════════════════════════════
# 2. mel filterbank — librosa's Slaney scale + Slaney norm, hand-rolled
# ══════════════════════════════════════════════════════════════════════════════
#
# Reimplements `librosa.filters.mel(..., htk=False, norm="slaney")`.  The Slaney
# mel scale is linear below 1 kHz (200/3 Hz per mel) and logarithmic above; the
# breakpoint constants below are librosa's, not an approximation of them.

_F_SP        = 200.0 / 3.0
_MIN_LOG_HZ  = 1000.0
_MIN_LOG_MEL = _MIN_LOG_HZ / _F_SP              # 15.0
_LOGSTEP     = math.log(6.4) / 27.0


def hz_to_mel_slaney(f):
    """Hz -> Slaney mel.  Matches librosa.hz_to_mel(f, htk=False)."""
    f = np.asarray(f, dtype=np.float64)
    scalar = f.ndim == 0
    f = np.atleast_1d(f)
    mel = f / _F_SP
    hi = f >= _MIN_LOG_HZ
    mel[hi] = _MIN_LOG_MEL + np.log(f[hi] / _MIN_LOG_HZ) / _LOGSTEP
    return float(mel[0]) if scalar else mel


def mel_to_hz_slaney(m):
    """Slaney mel -> Hz.  Matches librosa.mel_to_hz(m, htk=False)."""
    m = np.asarray(m, dtype=np.float64)
    scalar = m.ndim == 0
    m = np.atleast_1d(m)
    f = m * _F_SP
    hi = m >= _MIN_LOG_MEL
    f[hi] = _MIN_LOG_HZ * np.exp(_LOGSTEP * (m[hi] - _MIN_LOG_MEL))
    return float(f[0]) if scalar else f


def build_mel_fb(n_mel: int = N_MEL, n_fft: int = N_FFT,
                 sample_rate: int = SAMPLE_RATE,
                 fmin: float = LOWFREQ, fmax: float = HIGHFREQ) -> np.ndarray:
    """Slaney-normalised mel filterbank, [n_mel, n_fft//2 + 1], float32.

    Triangles are placed on a mel-uniform grid of n_mel+2 points, then each is
    area-normalised by 2/(f[m+2] - f[m]) — that division is what "slaney" norm
    means, and it is why these filters do NOT peak at 1.0 the way TinyVAD's do.
    """
    n_freqs   = n_fft // 2 + 1
    fft_freqs = np.linspace(0.0, sample_rate / 2.0, n_freqs, dtype=np.float64)

    mel_pts = np.linspace(hz_to_mel_slaney(fmin), hz_to_mel_slaney(fmax),
                          n_mel + 2, dtype=np.float64)
    hz_pts  = mel_to_hz_slaney(mel_pts)

    fdiff = np.diff(hz_pts)                              # [n_mel+1]
    ramps = hz_pts[:, None] - fft_freqs[None, :]         # [n_mel+2, n_freqs]

    fb = np.zeros((n_mel, n_freqs), dtype=np.float64)
    for m in range(n_mel):
        lower = -ramps[m]     / fdiff[m]
        upper =  ramps[m + 2] / fdiff[m + 1]
        fb[m] = np.maximum(0.0, np.minimum(lower, upper))

    # Slaney area normalisation
    enorm = 2.0 / (hz_pts[2:n_mel + 2] - hz_pts[:n_mel])
    fb *= enorm[:, None]
    return fb.astype(np.float32)


_MEL_FB = build_mel_fb()        # built once, constant for the process lifetime


# ══════════════════════════════════════════════════════════════════════════════
# 3. log-mel feature extraction
# ══════════════════════════════════════════════════════════════════════════════

def _hann_symmetric(n: int) -> np.ndarray:
    """torch.hann_window(n, periodic=False) == np.hanning(n)."""
    return np.hanning(n).astype(np.float64)


def extract_logmel(pcm: np.ndarray, *, normalize: str = "per_feature",
                   dither: float = 0.0, pad_to: int = 0,
                   seed: int | None = None) -> np.ndarray:
    """16 kHz mono float32 PCM -> [time, N_MEL] float32 log-mel features.

    Frame count is  floor(len(pcm) / HOP_LENGTH) + 1  (torch.stft with
    center=True), i.e. ~= duration_seconds * 100 + 1.

    `dither` defaults to 0.0, NOT the config's 1e-5.  NeMo adds Gaussian noise of
    that amplitude to the waveform for training robustness; leaving it on makes
    the front end non-deterministic, which would make bit-exactness testing
    against a C port impossible.  Pass `dither=DITHER, seed=...` to reproduce the
    training-time path.

    `pad_to` defaults to 0, NOT the config's 16.  Padding the time axis up to a
    multiple of 16 is a fixed-tensor batching artifact; the firmware tiles time
    itself and does not need it.  Pass `pad_to=16` for bit-parity with NeMo.
    """
    x = np.asarray(pcm, dtype=np.float64).ravel()
    if x.size == 0:
        return np.zeros((0, N_MEL), dtype=np.float32)

    # 1. dither (off by default — see docstring)
    if dither > 0.0:
        rng = np.random.default_rng(seed)
        x = x + dither * rng.standard_normal(x.size)

    # 2. preemphasis: y[0] = x[0]; y[t] = x[t] - 0.97 * x[t-1]
    if PREEMPH is not None:
        x = np.concatenate(([x[0]], x[1:] - PREEMPH * x[:-1]))

    # 3. STFT, center=True -> reflect-pad by n_fft//2 so frame 0 centres on
    #    sample 0.  Window is a 320-tap SYMMETRIC Hann, zero-padded to 512
    #    symmetrically (torch pads win_length up to n_fft centred).
    x = np.pad(x, N_FFT // 2, mode="reflect")
    win   = _hann_symmetric(WIN_LENGTH)
    pad_w = N_FFT - WIN_LENGTH
    win   = np.pad(win, (pad_w // 2, pad_w - pad_w // 2))

    n_frames = 1 + (x.size - N_FFT) // HOP_LENGTH
    frames = np.lib.stride_tricks.as_strided(
        x, shape=(n_frames, N_FFT),
        strides=(x.strides[0] * HOP_LENGTH, x.strides[0]),
    )
    spec = np.fft.rfft(frames * win, n=N_FFT)          # [n_frames, n_freqs]

    # 4. magnitude, then mag_power=2 -> power spectrum
    power = np.abs(spec) ** MAG_POWER

    # 5. mel projection.  power is [time, freq]; the filterbank is [mel, freq];
    #    (power @ fb.T) is [time, mel] — the layout the rest of the repo wants.
    mel = power @ _MEL_FB.T.astype(np.float64)         # [time, mel]

    # 6. log with the "add" zero guard
    feat = np.log(mel + LOG_GUARD)

    # 7. per-feature normalisation: z-score each mel bin over the time axis.
    #    torch's .std() is unbiased (N-1), hence ddof=1.  NON-CAUSAL — see header.
    if normalize == "per_feature":
        mean = feat.mean(axis=0, keepdims=True)
        std  = feat.std(axis=0, ddof=1, keepdims=True) if feat.shape[0] > 1 \
               else np.zeros((1, feat.shape[1]))
        feat = (feat - mean) / (std + NORM_EPS)
    elif normalize not in (None, "none"):
        raise ValueError(f"unsupported normalize={normalize!r}")

    # 8. optional pad_to (off by default — see docstring)
    if pad_to > 0 and feat.shape[0] % pad_to:
        feat = np.pad(feat, ((0, pad_to - feat.shape[0] % pad_to), (0, 0)))

    return np.ascontiguousarray(feat, dtype=np.float32)   # [time, N_MEL]


# ══════════════════════════════════════════════════════════════════════════════
# 4. int8 quantisation  (PLACEHOLDER CALIBRATION)
# ══════════════════════════════════════════════════════════════════════════════
#
# The real input scale belongs to the trained, PTQ'd model — it comes out of the
# ONNX Runtime calibration pass in Stage A and lives next to the weights.  That
# does not exist yet.  Until it does, the scale is derived from the clip itself
# using the same percentile rule quartznet_ref.make_blobs() uses for every other
# tensor, so the numbers are in a realistic int8 range and every downstream shape
# and code path is exercised.
#
# Swapping in the real value is a one-liner: pass `scale=<model input scale>` and
# `zero_point=<model input zp>` to quantize_features() and ignore calibrate().
#
# MEASURED, and relevant to Stage A: because "per_feature" normalisation forces
# every mel bin to zero mean and unit variance, the input distribution is nearly
# standard normal REGARDLESS of the clip.  On 3 real speech clips the whole
# feature tensor lived in [-1.50, +3.21] (skew +0.03), so:
#   * a SINGLE global (scale, zero_point) is valid for all utterances — the input
#     scale is a property of the normaliser, not of the audio.  Stage A does not
#     need to calibrate the input tensor over a dataset the way it must for the
#     activations further in.
#   * target=48 (inherited from make_blobs) is far too conservative here: it maps
#     the real clips into int8 [-56, +90], using ~100 of 256 codes and throwing
#     away ~1.3 bits.  Real PTQ should use roughly max|x| -> 127, i.e. target
#     nearer 110-120, or a plain min/max calibrator.
#   * IN_ZP_DEFAULT = -20 buys headroom on the POSITIVE side ((q - zp) spans
#     [-108, +147]), which happens to match the sign of the tail: normalised
#     log-mel is right-skewed (loud frames) with a hard floor on the left, since
#     a silent bin cannot go below its own mean by more than ~1.5 sigma.

def calibrate(feat: np.ndarray, target: float = CALIB_TARGET,
              pct: float = CALIB_PCT) -> float:
    """Placeholder input scale: map the `pct`-th percentile of |feat| to `target`
    int8 counts.  Mirrors quartznet_ref._calib_scale(), inverted — that one
    returns counts-per-unit for a requantiser, this returns units-per-count.
    """
    spread = float(np.percentile(np.abs(feat), pct))
    return max(spread, 1e-12) / target


def quantize_features(feat: np.ndarray, scale: float,
                      zero_point: int = IN_ZP_DEFAULT) -> np.ndarray:
    """[time, N_MEL] float32 -> [time, N_MEL] int8, affine, round-half-away.

    Convention matches the descriptor table and quartznet_ref: the model computes
    (q - zero_point), so  real ~= scale * (q - zero_point)  and therefore
    q = round(real / scale) + zero_point.
    """
    q = np.rint(feat.astype(np.float64) / scale) + zero_point
    return np.clip(q, -128, 127).astype(np.int8)


def dequantize_features(q: np.ndarray, scale: float,
                        zero_point: int = IN_ZP_DEFAULT) -> np.ndarray:
    return (q.astype(np.float32) - zero_point) * scale


def write_input_blob(path, q: np.ndarray) -> int:
    """Write `quartznet_input.bin`: raw int8, C-contiguous, [time, N_MEL].

    Byte-for-byte the same thing quartznet_ref.main() writes, so this file drops
    straight into firmware/quartznet once real weights exist.  Note the firmware
    consumes BUF_IN at the *input* frame rate (t_in = 2 * t_out, C1 is stride 2),
    so t_out = time // 2.
    """
    q = np.ascontiguousarray(q, dtype=np.int8)
    if q.ndim != 2 or q.shape[1] != N_MEL:
        raise ValueError(f"expected [time, {N_MEL}], got {q.shape}")
    path = pathlib.Path(path)
    path.write_bytes(q.tobytes())
    return q.nbytes


# ══════════════════════════════════════════════════════════════════════════════
# 5. one-call front end
# ══════════════════════════════════════════════════════════════════════════════

def audio_file_to_int8(path, *, scale: float | None = None,
                       zero_point: int = IN_ZP_DEFAULT,
                       seconds: float | None = None,
                       pad_to: int = 0) -> dict:
    """mp3/wav/ogg/flac path -> everything downstream needs, in one call."""
    pcm = load_audio(path)
    if seconds is not None:
        pcm = pcm[:int(seconds * SAMPLE_RATE)]
    feat = extract_logmel(pcm, pad_to=pad_to)
    sc   = calibrate(feat) if scale is None else scale
    q    = quantize_features(feat, sc, zero_point)
    return {
        "path": str(path), "pcm": pcm, "features": feat, "int8": q,
        "scale": sc, "zero_point": zero_point,
        "duration_s": pcm.size / SAMPLE_RATE,
        "t_in": q.shape[0], "t_out": q.shape[0] // 2,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 6. self-test / smoke test
# ══════════════════════════════════════════════════════════════════════════════

def _check_in_zp_matches_descriptors() -> str:
    """Guard against IN_ZP_DEFAULT drifting from build_table()'s in_zp default."""
    import inspect
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    import quartznet_descriptors as qd                      # noqa: E402
    want = inspect.signature(qd.build_table).parameters["in_zp"].default
    if want != IN_ZP_DEFAULT:
        raise AssertionError(
            f"IN_ZP_DEFAULT={IN_ZP_DEFAULT} but build_table(in_zp=)={want}")
    return f"in_zp {IN_ZP_DEFAULT} matches quartznet_descriptors.build_table"


def synth_clip(kind: str, seconds: float = 3.0,
               sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Deterministic synthetic clips, so the self-test needs no audio fixtures.

    `voiced` is a crude 5-formant source-filter vowel with an f0 contour and
    syllable-rate amplitude modulation — not speech, but it has the harmonic
    stack and formant structure that make a log-mel plot look speech-like.
    """
    n = int(seconds * sample_rate)
    t = np.arange(n) / sample_rate
    rng = np.random.default_rng(0xA0D10)

    if kind == "tone":
        return (0.5 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)

    if kind == "sweep":                       # 100 Hz -> 7 kHz log sweep
        f0, f1 = 100.0, 7000.0
        k = np.log(f1 / f0) / max(seconds, 1e-9)
        return (0.5 * np.sin(2 * np.pi * f0 * (np.exp(k * t) - 1) / k)).astype(np.float32)

    if kind == "voiced":
        f0 = 120.0 + 25.0 * np.sin(2 * np.pi * 0.7 * t)       # pitch contour
        glottal = np.zeros(n)
        for h in range(1, 41):                                 # harmonic stack
            glottal += (1.0 / h) * np.sin(2 * np.pi * h * np.cumsum(f0) / sample_rate)
        # 5 formants as Lorentzian resonators, applied in the frequency domain
        spec  = np.fft.rfft(glottal)
        freqs = np.fft.rfftfreq(n, 1.0 / sample_rate)
        env   = np.zeros_like(freqs)
        for fc, bw, g in ((700, 90, 1.0), (1220, 110, 0.6), (2600, 150, 0.35),
                          (3400, 200, 0.18), (4500, 250, 0.08)):
            env += g / (1.0 + ((freqs - fc) / bw) ** 2)
        out = np.fft.irfft(spec * env, n=n)
        syll = 0.5 + 0.5 * np.sin(2 * np.pi * 3.5 * t)         # ~3.5 syll/s
        gate = (np.sin(2 * np.pi * 0.4 * t) > -0.3).astype(float)   # pauses
        out = out * syll * gate + 0.002 * rng.standard_normal(n)
        peak = np.abs(out).max()
        return (0.7 * out / max(peak, 1e-9)).astype(np.float32)

    if kind == "silence":
        return (1e-4 * rng.standard_normal(n)).astype(np.float32)

    raise ValueError(kind)


def _fb_digest() -> tuple[int, float, float, float]:
    """Cheap fingerprint of the mel filterbank, checked in --self-test.

    The committed values were confirmed against the real
    `librosa.filters.mel(sr=16000, n_fft=512, n_mels=64, fmin=0, fmax=8000,
    norm='slaney')` (librosa 0.11.0) in a throwaway venv: max abs difference
    1.86e-09, which is exactly the float32 cast of `_MEL_FB`.  `hz_to_mel_slaney`
    and `mel_to_hz_slaney` matched librosa's htk=False versions to 0.0.
    """
    fb = _MEL_FB.astype(np.float64)
    return (int((fb > 0).sum()), float(fb.sum()), float(fb.max()), float(fb[0].sum()))


def _blob_roundtrip(q: np.ndarray) -> bool:
    """write_input_blob -> read back -> identical array, in the layout the
    firmware assumes (raw int8, C-contiguous, row pitch == N_MEL)."""
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as fh:
        tmp = pathlib.Path(fh.name)
    try:
        write_input_blob(tmp, q)
        back = np.frombuffer(tmp.read_bytes(), dtype=np.int8).reshape(-1, N_MEL)
        return back.shape == q.shape and bool((back == q).all())
    finally:
        tmp.unlink(missing_ok=True)


def self_test() -> int:
    ok = True

    def check(name, cond, detail=""):
        nonlocal ok
        ok = ok and bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}{'  ' + detail if detail else ''}")

    print("quartznet_audio self-test")
    print("\n-- constants --")
    print("  " + _check_in_zp_matches_descriptors())
    try:
        sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
        import quartznet_topology as qt
        check("N_MEL matches topology", qt.N_MEL == N_MEL, f"{qt.N_MEL}")
        check("FPS_IN matches hop", qt.FPS_IN == SAMPLE_RATE // HOP_LENGTH,
              f"{qt.FPS_IN} == {SAMPLE_RATE}/{HOP_LENGTH}")
    except ImportError as e:
        check("topology import", False, str(e))

    print("\n-- mel filterbank (Slaney scale + Slaney norm) --")
    nz, tot, mx, f0sum = _fb_digest()
    print(f"  shape={_MEL_FB.shape}  nonzero={nz}  sum={tot:.6f}  max={mx:.6f}")
    check("no empty filters", all(_MEL_FB[m].sum() > 0 for m in range(N_MEL)))
    check("slaney norm applied (peak != 1.0)", not np.isclose(mx, 1.0),
          f"max={mx:.4f}")
    # digest values verified against librosa 0.11.0 — see _fb_digest.__doc__
    check("filterbank digest matches librosa",
          nz == 498 and abs(tot - 2.0461816269) < 1e-6
          and abs(mx - 0.0211132672) < 1e-9 and abs(f0sum - 0.0285867928) < 1e-9,
          f"nz={nz} sum={tot:.10f} max={mx:.10f} fb0={f0sum:.10f}")

    print("\n-- resampler --")
    for sr_in in (8000, 22050, 44100, 48000):
        n = sr_in * 2
        tt = np.arange(n) / sr_in
        sig = np.sin(2 * np.pi * 1000.0 * tt).astype(np.float32)
        out = resample_poly(sig, sr_in, SAMPLE_RATE)
        exp_n = int(math.ceil(n * (SAMPLE_RATE / sr_in)))
        # compare against the analytic 1 kHz sine on the 16 kHz grid, ignoring
        # filter transients at both edges
        ref = np.sin(2 * np.pi * 1000.0 * np.arange(exp_n) / SAMPLE_RATE)
        err = np.abs(out[400:-400] - ref[400:-400]).max()
        check(f"{sr_in} -> 16000", out.size == exp_n and err < 2e-3,
              f"n={out.size}/{exp_n} maxerr={err:.2e}")
    dc = resample_poly(np.ones(4410, dtype=np.float32), 44100, 16000)
    check("DC gain == 1", abs(dc[200:-200].mean() - 1.0) < 1e-6,
          f"mean={dc[200:-200].mean():.8f}")

    print("\n-- feature extraction --")
    for secs in (1.0, 2.5):
        f = extract_logmel(synth_clip("voiced", secs))
        want_t = int(secs * SAMPLE_RATE) // HOP_LENGTH + 1
        check(f"{secs}s -> [{want_t}, {N_MEL}]", f.shape == (want_t, N_MEL),
              f"got {f.shape}")
    f = extract_logmel(synth_clip("voiced", 3.0))
    check("layout is [time, channel]", f.shape[1] == N_MEL and f.shape[0] > N_MEL,
          f"{f.shape}")
    check("per-feature mean ~ 0", np.abs(f.mean(axis=0)).max() < 1e-3,
          f"max|mean|={np.abs(f.mean(axis=0)).max():.2e}")
    check("per-feature std ~ 1", np.abs(f.std(axis=0, ddof=1) - 1.0).max() < 1e-2,
          f"max|std-1|={np.abs(f.std(axis=0, ddof=1) - 1.0).max():.2e}")
    check("finite", np.isfinite(f).all())
    check("pad_to=16 rounds up", extract_logmel(synth_clip("tone", 1.0),
                                                pad_to=16).shape[0] % 16 == 0)

    print("\n-- int8 quantisation --")
    sc = calibrate(f)
    q  = quantize_features(f, sc)
    check("int8 dtype + shape", q.dtype == np.int8 and q.shape == f.shape)
    check("in range", q.min() >= -128 and q.max() <= 127, f"[{q.min()}, {q.max()}]")
    pct_counts = np.percentile(np.abs(q.astype(np.int32) - IN_ZP_DEFAULT), CALIB_PCT)
    check("calibration hits target", abs(pct_counts - CALIB_TARGET) < 2.0,
          f"p{CALIB_PCT}=|q-zp|={pct_counts:.1f} target={CALIB_TARGET}")
    # Round-trip is only bounded by 1/2 LSB where the code did NOT saturate; the
    # percentile calibration deliberately lets the extreme tail clip.
    sat = (q == -128) | (q == 127)
    rt  = dequantize_features(q, sc)
    check("round-trip < 1/2 LSB where unsaturated",
          np.abs(rt - f)[~sat].max() <= sc * 0.5 + 1e-6,
          f"maxerr={np.abs(rt - f)[~sat].max():.6f} half_lsb={sc * 0.5:.6f}")
    check("saturation < 0.5% of samples", sat.mean() < 0.005,
          f"{100.0 * sat.mean():.3f}%")
    check("int8 blob round-trips through write/read", _blob_roundtrip(q))

    print("\n" + ("ALL PASS" if ok else "FAILURES PRESENT"))
    return 0 if ok else 1


# ══════════════════════════════════════════════════════════════════════════════

def _report(res: dict) -> None:
    f, q = res["features"], res["int8"]
    exp_t = int(res["duration_s"] * (SAMPLE_RATE / HOP_LENGTH)) + 1
    print(f"\n{res['path']}")
    print(f"  duration        {res['duration_s']:.3f} s  ({res['pcm'].size} samples @ 16 kHz)")
    print(f"  pcm             min={res['pcm'].min():+.4f} max={res['pcm'].max():+.4f} "
          f"rms={float(np.sqrt((res['pcm'].astype(np.float64) ** 2).mean())):.4f}")
    print(f"  features        shape={f.shape}  [time, mel]   expected t~={exp_t}")
    print(f"  log-mel         min={f.min():+.4f} max={f.max():+.4f} "
          f"mean={f.mean():+.6f} std={f.std():.4f}")
    print(f"  int8            min={q.min()} max={q.max()} "
          f"mean={q.mean():+.2f}  zp={res['zero_point']}  scale={res['scale']:.6f}")
    print(f"  saturated       {int((q == -128).sum() + (q == 127).sum())} / {q.size} "
          f"({100.0 * ((q == -128).sum() + (q == 127).sum()) / max(q.size, 1):.3f}%)")
    used = 2 * res["t_out"]
    print(f"  t_in={res['t_in']}  ->  t_out={res['t_out']} (C1 stride 2); model reads "
          f"{used} frames" + (f", drops {res['t_in'] - used} odd trailing frame"
                              if res["t_in"] != used else ""))
    print(f"  blob            {q.nbytes:,} B int8 [time, {N_MEL}]")


def main() -> int:
    ap = argparse.ArgumentParser(
        description="QuartzNet audio front end: mp3/wav -> int8 log-mel")
    ap.add_argument("files", nargs="*", help="audio files (mp3/wav/ogg/flac)")
    ap.add_argument("--self-test", action="store_true",
                    help="run the built-in checks and exit")
    ap.add_argument("--out", default=None,
                    help="directory to write <stem>_input.bin blobs into")
    ap.add_argument("--seconds", type=float, default=None,
                    help="truncate each clip to this many seconds")
    ap.add_argument("--pad-to", type=int, default=0,
                    help="pad time axis to a multiple of this (NeMo uses 16)")
    args = ap.parse_args()

    if args.self_test:
        return self_test()
    if not args.files:
        ap.print_help()
        return 2

    out_dir = pathlib.Path(args.out) if args.out else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

    for p in args.files:
        res = audio_file_to_int8(p, seconds=args.seconds, pad_to=args.pad_to)
        _report(res)
        if out_dir:
            dst = out_dir / (pathlib.Path(p).stem + "_input.bin")
            n = write_input_blob(dst, res["int8"])
            print(f"  wrote           {dst}  ({n:,} B)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
