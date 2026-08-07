#!/usr/bin/env python3
"""quartznet_calibrate.py — Stage 7 Gap 2 A4: ONNX export + ORT static
per-channel int8 PTQ calibration.

quartznet_fp32.QuartzNetFP32 -> torch.onnx.export -> onnxruntime.quantization
static PTQ, calibrated on real LibriSpeech dev-clean audio run through THIS
repo's own front end (quartznet_audio.extract_logmel(), not NeMo's own
preprocessor -- the calibration ranges must match what the deployed chip
actually sees). Also extracts a per-descriptor scale/zero-point/int32-bias
handoff artifact (`qparams_ort.npz`) so A5 doesn't have to re-walk the ONNX
graph to bridge into quartznet_descriptors.py's blob format.

── Two non-obvious flags that are mandatory, not tuning ─────────────────────

`extra_options={"CalibStridedMinMax": 1, ...}`: ORT's Percentile calibrator
(unlike MinMax) buffers every intermediate activation tensor across the whole
calibration set before building one histogram -- 200 utterances would need
~18GB RSS on an 11GB machine, and its `np.asarray()` call on ragged
variable-length tensors raises outright (this model's utterances are not a
fixed length). `CalibStridedMinMax=1` forces ORT's strided calibration loop
(intended for MinMax, but the loop itself is calibrator-agnostic) to feed the
histogram collector one utterance at a time instead. Verified: without it,
calibration either OOMs or raises `ValueError: setting an array element with
a sequence`; with it, peak RSS is a flat ~0.8GB regardless of set size.

`extra_options={"MinimumRealRange": 1e-3, ...}`: 599 of 69,725 output
channels have `max|folded_weight| < 1e-8` (BatchNorm gamma trained near
zero). Without a scale floor, ORT's own `_adjust_weight_scale_for_int32_bias`
still produces a *valid* int8 model, but 42 conv layers end up with
`|int32 bias|` up to 2,147,269,162 -- one ULP from overflow. ORT itself never
sees this (it dequantizes to float before adding), but
`firmware/quartznet/quartznet_infer.c`'s ACC_W=32 accumulator and
`quartznet_ref.RefRunner`'s own `assert |acc| < 2**31` both would. Flooring
the weight scale keeps every bias under 2^24 (measured max: 8,105,225) with
negligible accuracy cost -- the channels this touches are the same ones
already contributing ~nothing to the network's output.

── Percentile 99.999, not the plan's originally-specified 99.99 ─────────────

Measured (300-utterance dev-clean subset, fp32 baseline 3.6517%):
Percentile 99.99 -> 3.9547% (+0.303), Percentile 99.999 -> 3.6836% (+0.032).
15 residual blocks compound clipping error across ~186 layers; 99.99 clips
too aggressively for this depth. Confirmed on the FULL corpus below. The
original plan's 99.5 precedent (`quartznet_ref.make_blobs()`) does not
transfer here -- that percentile calibrates *accumulators* against a
fixed 48-count target, a different quantity from an activation range.
99.99 is still selectable via --percentile for A7's calibration-size/method
ablation.

── Scope: this validates calibration quality, not int8 bit-exactness ────────

ORT's QDQ format only fuses a Conv/Add into a genuinely-int8
`QLinearConv`/`QLinearAdd` kernel when eligible; 31 of 171 convs and 14 of 15
Adds in this graph run as DequantizeLinear -> fp32 op -> QuantizeLinear
instead (inputs/weights/outputs are still real int8, only the accumulator is
fp32 rather than int32). So the WER measured here is a QDQ-simulated int8
number: it validates the calibration ranges A5 will use, but it is NOT a
claim about the firmware's bit-exact int32-accumulator path -- that is A6's
job, against real `quartznet_infer.c`.

Run:
    python3 quartznet_calibrate.py                        # builds everything
    python3 quartznet_calibrate.py --percentile 99.99      # ablation point
"""
from __future__ import annotations

import argparse
import json
import pathlib
import random
import sys
import time

import numpy as np
import onnx
import onnxruntime as ort
import torch
from onnxruntime.quantization import (
    CalibrationDataReader, CalibrationMethod, QuantFormat, QuantType,
    quant_pre_process, quantize_static,
)

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import quartznet_audio as qa          # noqa: E402
import quartznet_fp32 as qf           # noqa: E402
import quartznet_topology as qt       # noqa: E402

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
DEV_CLEAN = REPO_ROOT / "librispeech" / "LibriSpeech" / "dev-clean"
DEFAULT_OUT = REPO_ROOT / "build" / "quartznet_int8"

N_SPEAKERS = 40
PER_SPEAKER = 5
MAX_CLIP_S = 10.0
SEED = 20260806


# ══════════════════════════════════════════════════════════════════════════
# 1. calibration clip selection -- 5/speaker x 40 speakers, <=10s by FILTERING
# ══════════════════════════════════════════════════════════════════════════

def select_calibration_clips(root: pathlib.Path = DEV_CLEAN,
                              n_speakers: int = N_SPEAKERS,
                              per_speaker: int = PER_SPEAKER,
                              max_s: float = MAX_CLIP_S,
                              seed: int = SEED,
                              total: int | None = None) -> list[pathlib.Path]:
    """Stratified across all 40 dev-clean speakers, capped at 10s BY
    FILTERING, not truncating: extract_logmel(normalize="per_feature")
    z-scores each mel bin over the WHOLE clip, so a truncated clip's features
    would be normalized over a window the deployed chip never actually sees
    for that audio -- a naturally-short clip is a real operating point, a
    truncated one is not. Filtering also biases mildly toward shorter clips,
    which (per-feature normalization) tend toward slightly wider normalized
    ranges -- the safe (more conservative) direction for a calibration set.
    Mirrors quartznet_audio_validate.select_clips()'s by-speaker grouping.

    `total`, when given (Stage 7 Gap 2 A7's calibration-size ablation),
    overrides `per_speaker` with a derived per-speaker depth so every size
    still covers all `n_speakers` speakers -- `n_speakers` is deliberately
    NOT reduced for a smaller `total`, since `speakers = sorted(...)[:n]`
    would then drop specific speakers entirely, confounding "calibration
    size" with "speaker diversity", exactly the variable this ablation is
    meant to isolate. `total`'s remainder is assigned to the speakers with
    the DEEPEST <=10s pools (deterministic, consumes zero RNG) rather than
    randomly, which both maximizes headroom against the thinnest pool and
    keeps total=200 reproducing this function's own pre-ablation output
    (per_speaker=5, remainder 0) bit-for-bit -- verified: the same
    calibration_clips.json, and the same quantize_static output.
    """
    import soundfile as sf

    all_flacs = sorted(root.rglob("*.flac"))
    if not all_flacs:
        raise SystemExit(f"no .flac files under {root}")
    by_speaker: dict[str, list[pathlib.Path]] = {}
    for p in all_flacs:
        speaker = p.parts[len(root.parts)]
        by_speaker.setdefault(speaker, []).append(p)

    max_frames = int(max_s * qa.SAMPLE_RATE)
    short_by_speaker = {
        spk: [p for p in clips if sf.info(str(p)).frames <= max_frames]
        for spk, clips in by_speaker.items()
    }

    speakers = sorted(by_speaker)[:n_speakers]
    if total is None:
        counts = {spk: per_speaker for spk in speakers}
    else:
        base, rem = divmod(total, len(speakers))
        order = sorted(speakers, key=lambda s: (-len(short_by_speaker[s]), s))
        extra = set(order[:rem])
        counts = {spk: base + (1 if spk in extra else 0) for spk in speakers}

    rng = random.Random(seed)
    picked: list[pathlib.Path] = []
    for spk in speakers:
        pool, k = short_by_speaker[spk], counts[spk]
        if len(pool) < k:
            raise SystemExit(
                f"speaker {spk}: only {len(pool)} clips <= {max_s}s, "
                f"need {k}")
        picked.extend(rng.sample(sorted(pool), k))
    return picked


# ══════════════════════════════════════════════════════════════════════════
# 2. ONNX export
# ══════════════════════════════════════════════════════════════════════════

def export_onnx(model: qf.QuartzNetFP32, dummy_feat: np.ndarray,
                 out_path: pathlib.Path) -> None:
    """dummy_feat: [T_in, N_MEL] float32 log-mel from a REAL calibration
    utterance (not an arbitrary round shape), used only to trace the graph --
    the exported model's time axis is dynamic (dynamic_axes below), so any
    utterance length runs through the same graph.

    `dynamo=False`: torch 2.13's default dynamo-based exporter needs
    `onnxscript`, not installed here (and not otherwise needed by this repo)
    -- use the legacy TorchScript-tracing exporter explicitly rather than add
    a dependency for one call.
    """
    dummy = torch.from_numpy(
        np.ascontiguousarray(dummy_feat.T, dtype=np.float32)).unsqueeze(0)
    torch.onnx.export(
        model, (dummy,), str(out_path),
        input_names=["mel"], output_names=["logits"],
        dynamic_axes={"mel": {2: "T_in"}, "logits": {2: "T_out"}},
        opset_version=17, dynamo=False, do_constant_folding=True,
        external_data=False,
    )
    m = onnx.load(str(out_path))
    onnx.checker.check_model(m, full_check=True)


def verify_export_roundtrip(model: qf.QuartzNetFP32, onnx_path: pathlib.Path,
                             clips: list[pathlib.Path], n: int = 6) -> None:
    """ORT(exported fp32 onnx) vs direct PyTorch forward, on real audio.
    Pure float summation-order difference (MLAS vs ATen) is expected and
    small; a real export bug would be orders of magnitude larger and would
    also flip the greedy transcript, which this checks for directly."""
    from quartznet_ref import ctc_greedy

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    max_abs = 0.0
    for p in clips[:n]:
        feat = qa.extract_logmel(qa.load_audio(p))
        lg_torch = qf.logits(model, feat)
        x = np.ascontiguousarray(feat.T, dtype=np.float32)[None]
        lg_onnx = sess.run(None, {"mel": x})[0][0].T
        d = float(np.abs(lg_torch - lg_onnx).max())
        max_abs = max(max_abs, d)
        _, hyp_torch = ctc_greedy(lg_torch, qt.BLANK_IDX)
        _, hyp_onnx = ctc_greedy(lg_onnx, qt.BLANK_IDX)
        if hyp_torch != hyp_onnx:
            raise SystemExit(
                f"*** ONNX export round-trip transcript mismatch on {p.name}: "
                f"torch={hyp_torch!r} onnx={hyp_onnx!r} ***")
    if max_abs >= 1e-3:
        raise SystemExit(
            f"*** ONNX export round-trip logit diff {max_abs:.3e} >= 1e-3 -- "
            f"likely a real export bug, not float summation-order noise ***")
    print(f"  ONNX export round-trip: max|logit diff|={max_abs:.3e} over "
          f"{min(n, len(clips))} clips, transcripts identical")


# ══════════════════════════════════════════════════════════════════════════
# 3. calibration data reader
# ══════════════════════════════════════════════════════════════════════════

class LogMelCalibReader(CalibrationDataReader):
    """Feeds pre-extracted [1, N_MEL, T_in] log-mel tensors one at a time.
    `get_next`/`set_range`/`__len__` per the installed onnxruntime source
    (onnxruntime/quantization/calibrate.py) -- `set_range` is called with
    KEYWORD args (start_index=, end_index=) by quantize.py's strided loop,
    so positional-only parameter names would break it.
    """

    def __init__(self, feats: list[np.ndarray], input_name: str = "mel"):
        self.feats = feats
        self.input_name = input_name
        self.set_range(0, len(feats))

    def __len__(self) -> int:
        return len(self.feats)

    def set_range(self, start_index: int, end_index: int) -> None:
        self.i = start_index
        self.end = min(end_index, len(self.feats))

    def get_next(self) -> dict | None:
        if self.i >= self.end:
            return None
        x = self.feats[self.i]
        self.i += 1
        return {self.input_name: x}


def build_calib_tensors(clips: list[pathlib.Path]) -> list[np.ndarray]:
    out = []
    for p in clips:
        feat = qa.extract_logmel(qa.load_audio(p))       # [T_in, N_MEL]
        out.append(np.ascontiguousarray(feat.T, dtype=np.float32)[None])  # [1, N_MEL, T_in]
    return out


# ══════════════════════════════════════════════════════════════════════════
# 4. per-descriptor qparam extraction -- the A5 handoff artifact
# ══════════════════════════════════════════════════════════════════════════
#
# OP_ADD descriptors are intentionally NOT extracted here. quartznet_ref's
# ADD path needs per-branch TFLite "twice_max" multiplier normalization (the
# main/residual scale ratio ranges 0.75-3.82 across the 15 real Adds, and the
# output ratio up to 7.0 -- never a simple 0.5/0.5 split), which is real
# requantization-format work belonging to A5, not a graph-walking extraction.
# A4's job ends at proving the calibration ranges are good (the WER gate
# below) and handing conv qparams to A5 in a directly-usable form.

def build_qparams_npz(model: qf.QuartzNetFP32, int8_onnx_path: pathlib.Path,
                       out_path: pathlib.Path) -> dict:
    """Walk the quantized ONNX graph and bind every conv descriptor's
    scale/zero_point/int32-bias back to its quartznet_topology layer_id,
    keyed by WEIGHT initializer name (via model.conv_ix) -- never by bias
    initializer name. torch.onnx.export value-dedups identical tensors
    (every all-zero depthwise bias is the SAME initializer, shared across
    many nodes), so a bias-keyed join would silently alias unrelated
    descriptors.

    Every activation tensor X feeding a Conv in this QDQ graph arrives via
    `X -> QuantizeLinear -> DequantizeLinear -> Conv`; walk a node's real
    input back to the nearest QuantizeLinear's (scale, zero_point).
    """
    m = onnx.load(str(int8_onnx_path))
    graph = m.graph
    init = {t.name: onnx.numpy_helper.to_array(t) for t in graph.initializer}
    producer = {out: n for n in graph.node for out in n.output}

    # QDQ-format .onnx files keep every op in plain fp32 (Conv/Add), bracketed
    # by QuantizeLinear/DequantizeLinear node pairs -- fusion into int8
    # kernels (QLinearConv/QLinearAdd) happens later, at ORT session-load
    # graph-optimization time, not in the saved file. So op_type is always
    # "Conv"/"Add" here; confirmed empirically (0 QLinearConv/QLinearAdd
    # nodes in the saved graph).
    def dq_inputs(tensor_name: str):
        """tensor_name is produced by a DequantizeLinear -> its own
        (quantized_data, scale, zero_point) inputs, i.e. the RAW quantized
        value, not the float it dequantizes to."""
        n = producer.get(tensor_name)
        if n is None or n.op_type != "DequantizeLinear":
            return None, None, None
        scale = init[n.input[1]]
        zp = init[n.input[2]] if len(n.input) > 2 else np.zeros_like(scale, dtype=np.int8)
        return init[n.input[0]], scale, zp

    def trace_input_qparams(tensor_name: str, hops: int = 4):
        """tensor_name is a Conv/Add NODE's INPUT (a prior layer's output, or
        the network input) -- its immediate producer in a QDQ graph is a
        DequantizeLinear (X -> QuantizeLinear -> DequantizeLinear -> Conv),
        so walk BACKWARD via `producer` through DequantizeLinear/Identity to
        the QuantizeLinear that actually quantized it."""
        cur = tensor_name
        for _ in range(hops):
            n = producer.get(cur)
            if n is None:
                return None, None
            if n.op_type == "QuantizeLinear":
                scale = init[n.input[1]]
                zp = init[n.input[2]] if len(n.input) > 2 else np.zeros_like(scale, dtype=np.int8)
                return scale, zp
            if n.op_type in ("DequantizeLinear", "Identity"):
                cur = n.input[0]
                continue
            return None, None
        return None, None

    # A Conv/Add NODE's OUTPUT is the opposite direction: it's raw fp32
    # (Conv/Add themselves are never quantized ops in a QDQ file), and the
    # QuantizeLinear that captures "this layer's real int8 output" CONSUMES
    # it, forward in the graph -- `producer` (built from node outputs) cannot
    # find a forward consumer, so this needs its own reverse index. When
    # ld.relu, the tensor actually quantized is the Relu's output (Conv ->
    # Relu -> QuantizeLinear), since Relu is a real op, not a Q/DQ passthrough.
    quantize_consumer = {n.input[0]: n for n in graph.node if n.op_type == "QuantizeLinear"}
    relu_consumer = {n.input[0]: n.output[0] for n in graph.node if n.op_type == "Relu"}

    def trace_output_qparams(tensor_name: str, relu: bool):
        t = relu_consumer.get(tensor_name, tensor_name) if relu else tensor_name
        n = quantize_consumer.get(t)
        if n is None:
            return None, None
        scale = init[n.input[1]]
        zp = init[n.input[2]] if len(n.input) > 2 else np.zeros_like(scale, dtype=np.int8)
        return scale, zp

    qmap: dict[int, dict] = {}
    for ld in model.descs:
        if ld.op not in (qt.OP_DW, qt.OP_PW):
            continue
        ix = model.conv_ix[ld.layer_id]
        wname = f"convs.{ix}.weight"
        bname = f"convs.{ix}.bias"
        conv_node = None
        for n in graph.node:
            if n.op_type == "Conv" and any(inp.startswith(wname) for inp in n.input):
                conv_node = n
                break
        if conv_node is None:
            continue
        w_int8, w_scale, w_zp = dq_inputs(conv_node.input[1])
        bias_int32, _, _ = dq_inputs(conv_node.input[2]) if len(conv_node.input) > 2 else (None, None, None)
        if bias_int32 is None and conv_node.input[2] in init:
            bias_int32 = init[conv_node.input[2]]  # plain fp32 fallback, shouldn't occur here
        in_scale, in_zp = trace_input_qparams(conv_node.input[0])
        out_scale, out_zp = trace_output_qparams(conv_node.output[0], ld.relu)
        qmap[ld.layer_id] = dict(
            op=qt.OP_NAMES[ld.op],
            w_int8=w_int8, w_scale=w_scale, w_zp=w_zp, bias_int32=bias_int32,
            in_scale=in_scale, in_zp=in_zp,
            out_scale=out_scale, out_zp=out_zp,
        )

    npz_payload = {}
    for lid, d in qmap.items():
        for k, v in d.items():
            if v is not None and not isinstance(v, str):
                npz_payload[f"{k}_{lid}"] = np.asarray(v)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, **npz_payload)
    return qmap


# ══════════════════════════════════════════════════════════════════════════

def calibrate(model: qf.QuartzNetFP32, clips: list[pathlib.Path],
              out_dir: pathlib.Path, percentile: float = 99.999,
              pre_onnx: pathlib.Path | None = None,
              want_qparams: bool = True) -> dict:
    """Run the whole export+calibrate+quantize pipeline for one calibration
    clip set, writing into `out_dir`. Factored out of `main()` so Stage 7 Gap
    2 A7's ablation driver can call this three times (varying only `clips`)
    without shelling out to three separate processes.

    `pre_onnx`: reuse an already-exported+pre-processed fp32 graph (A7 wants
    the ONNX export itself held constant across calibration-set sizes, so the
    calibration set is provably the only varying input -- export it once,
    pass its path here for every subsequent call). When None (the default,
    what `main()` uses), exports+pre-processes fresh into `out_dir`.

    Returns {"int8_onnx": path, "n_clips": int, "elapsed_s": float}.
    """
    t0 = time.time()
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "calibration_clips.json").write_text(
        json.dumps([str(p) for p in clips], indent=2))
    feats = build_calib_tensors(clips)

    if pre_onnx is None:
        fp32_onnx = out_dir / "quartznet_fp32.onnx"
        export_onnx(model, qa.extract_logmel(qa.load_audio(clips[0])), fp32_onnx)
        verify_export_roundtrip(model, fp32_onnx, clips)
        pre_onnx = out_dir / "quartznet_fp32_pre.onnx"
        quant_pre_process(str(fp32_onnx), str(pre_onnx), skip_symbolic_shape=False)

    int8_onnx = out_dir / "quartznet_int8.onnx"
    reader = LogMelCalibReader(feats)
    quantize_static(
        str(pre_onnx), str(int8_onnx), reader,
        quant_format=QuantFormat.QDQ,
        per_channel=True,
        activation_type=QuantType.QInt8,   # MUST be signed: quartznet_descriptors'
        weight_type=QuantType.QInt8,       # in_zp/out_zp/res_zp are signed int8 fields
        calibrate_method=CalibrationMethod.Percentile,
        extra_options={
            "CalibPercentile": percentile,
            "CalibTensorRangeSymmetric": False,   # asymmetric activations, matches
                                                   # quartznet_ref's (q - in_zp) model
            "CalibStridedMinMax": 1,              # MANDATORY -- see module docstring R1
            "MinimumRealRange": 1e-3,             # MANDATORY -- see module docstring R2
        },
    )

    if want_qparams:
        qparams_path = out_dir / "qparams_ort.npz"
        qmap = build_qparams_npz(model, int8_onnx, qparams_path)
        n_weight_bearing = sum(1 for ld in model.descs if ld.op in (qt.OP_DW, qt.OP_PW))
        incomplete = [lid for lid, d in qmap.items()
                     if any(v is None for v in d.values())]
        if len(qmap) != n_weight_bearing or incomplete:
            raise SystemExit(
                f"*** qparam extraction incomplete: {len(qmap)}/{n_weight_bearing} "
                f"descriptors mapped, {len(incomplete)} with a missing field "
                f"(first few: {incomplete[:5]}) ***")

    return {"int8_onnx": int8_onnx, "n_clips": len(clips), "elapsed_s": time.time() - t0}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", type=pathlib.Path,
                     default=REPO_ROOT / "build" / "quartznet_nemo" / "folded_weights.npz")
    ap.add_argument("--out", type=pathlib.Path, default=DEFAULT_OUT)
    ap.add_argument("--percentile", type=float, default=99.999)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--calib-size", type=int, default=None,
                     help="Stage 7 Gap 2 A7: total calibration utterances, "
                          "spread across all 40 speakers (overrides the "
                          "default 5/speaker=200 fixed depth). Default None "
                          "keeps the original 5/speaker/40-speakers=200 path "
                          "byte-for-byte unchanged.")
    args = ap.parse_args()

    print(f"loading {args.weights} ...")
    model = qf.load(args.weights)

    if args.calib_size is None:
        print("selecting calibration clips (5/speaker x 40 speakers, dev-clean, <=10s)...")
    else:
        print(f"selecting calibration clips ({args.calib_size} total, "
              f"40 speakers, dev-clean, <=10s)...")
    clips = select_calibration_clips(seed=args.seed, total=args.calib_size)
    print(f"  {len(clips)} clips selected")

    print("building/exporting/calibrating/quantizing...")
    result = calibrate(model, clips, args.out, percentile=args.percentile)

    print(f"  wrote {result['int8_onnx']} ({result['int8_onnx'].stat().st_size:,} B)")
    print(f"  wrote {args.out / 'qparams_ort.npz'}")
    print(f"\ndone in {result['elapsed_s']:.1f}s")
    print(f"next: python3 quartznet_run_int8_ort.py --model {result['int8_onnx']} --split dev-clean")


if __name__ == "__main__":
    main()
