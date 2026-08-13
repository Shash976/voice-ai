#!/usr/bin/env python3
"""quartznet_nemo_export.py — NeMo stt_en_quartznet15x5 checkpoint -> BN-folded
per-layer weights, mapped one-to-one onto quartznet_topology.py's LayerDesc list.

Stage 7 Gap 2 (docs/07_quartznet_pivot.md Stage A / docs/07c's "what's still
missing" item 2). This is the format bridge's FIRST half: it proves the
checkpoint<->topology mapping is complete and correct (exact parameter count,
every weight-bearing descriptor bound to a real tensor, zero unexplained
leftover tensors) and produces BN-folded per-layer (weight, bias) arrays in
fp32. It does NOT do int8 quantization or emit quartznet_descriptors.py's
binary blob format -- that is quartznet_calibrate.py (ORT static PTQ,
calibration-driven activation scales) and quartznet_export_int8.py (the
actual bridge into the blob format), a separate, later step, because those
need a real ONNX forward-pass graph and a calibration dataset this script
does not touch.

── Why no nemo_toolkit dependency ────────────────────────────────────────────

A .nemo file is a tar of model_config.yaml + a plain PyTorch state_dict
(model_weights.ckpt) -- nemo_toolkit's own multi-GB dependency chain (hydra,
omegaconf, pytorch-lightning, sentencepiece, webdataset...) buys nothing here.
torch.load() reads the checkpoint directly.

── The tensor-naming scheme (verified against the real checkpoint, not the
   NeMo source) ───────────────────────────────────────────────────────────

Confirmed by inspecting stt_en_quartznet15x5.nemo's actual state_dict keys.
For block index `bid` (0-indexed position in quartznet_topology.BLOCKS,
0..17 -- C4 is NOT block 18 in the checkpoint, see below) and repeat index
`r` within that block:

  residual projection   encoder.encoder.{bid}.res.0.0.conv.weight   (conv)
                         encoder.encoder.{bid}.res.0.1.*             (BN)
  depthwise (repeat r)  encoder.encoder.{bid}.mconv.{5r}.conv.weight
  pointwise (repeat r)  encoder.encoder.{bid}.mconv.{5r+1}.conv.weight (conv)
                         encoder.encoder.{bid}.mconv.{5r+2}.*          (BN)

C3 (bid=17, separable=False -- BLOCKS' only non-separable *encoder* block) has
no depthwise: a single conv+BN at mconv.0/mconv.1 instead of the 5-slot-per-
repeat pattern (there is only one repeat, but no depthwise to make room for).

C4 (quartznet_topology.py's synthetic last BLOCKS entry, index 18) is NOT
part of NeMo's `encoder.encoder` ModuleList at all -- NeMo implements it as a
separate `decoder.decoder_layers.0` Conv1d WITH A REAL BIAS (no BN to fold;
this is the only weight-bearing descriptor in the whole model that needs no
BN-fold math at all).

BatchNorm1d's default eps (1e-5, matching PyTorch's default -- model_config.yaml
does not override it) is what NeMo actually used to train, so it is what must
be used to fold.

── Weight layout: no transpose needed ────────────────────────────────────────

PyTorch Conv1d weight shape is [out_ch, in_ch/groups, kernel]. For a depthwise
conv (groups=out_ch=in_ch), squeezing the middle (=1) axis gives [C, K], which
flattens in C-major/K-minor order -- exactly quartznet_descriptors.py's
`w[c*K+k]`. For a pointwise conv (kernel=1), squeezing the last axis gives
[c_out, c_in], flattening as `w[oc*c_in+ic]` -- exactly matching
quartznet_descriptors.py's PW convention (the file header's own words: "SAME
[out, in, k] order as tiny_vad_infer.c"). So folding only touches VALUES, never
axis order.

Run:
    python3 quartznet_nemo_export.py <path/to/stt_en_quartznet15x5.nemo> [--out DIR]

Emits, into DIR (default: build/quartznet_nemo/):
    checkpoint_map.json   layer_id -> {conv, bn|bias, block_name, op} tensor names
    folded_weights.npz    per layer_id: `w{layer_id}` (fp32 weight, descriptor
                          layout) and `b{layer_id}` (fp32 bias, post-BN-fold or
                          the decoder's real bias)
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys
import tarfile

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import quartznet_topology as qt  # noqa: E402

# NOT PyTorch's BatchNorm1d default (1e-5). NeMo's JasperBlock hardcodes its
# own eps in _get_conv_bn_layer -- nemo/collections/asr/parts/submodules/
# jasper.py (v1.23.0): `nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.1)`.
# This is why model_config.yaml doesn't mention it: it isn't a config knob,
# it's hardcoded in the block builder. Verified this matters, not just a
# rounding nit -- the checkpoint's running_var values are BELOW 1e-3 (e.g.
# encoder.encoder.8.mconv.12.running_var.mean() = 5.98e-05), so with the
# wrong eps=1e-5 the eps term is negligible and every folded layer's scale is
# inflated ~2-4x; compounded over 80 BN layers this overflows to logits with
# absmax ~1e31 and a garbage greedy transcript. With the correct eps=1e-3 the
# same checkpoint produces logits absmax ~38 and a legible, correct
# transcript (see quartznet_run_fp32.py / docs/07d's Gap 2 A3 section).
BN_EPS = 1e-3

# quartznet_topology.BLOCKS index of C3 (last *separable=False* encoder block)
# and C4 (the synthetic decoder-as-a-block entry -- not in encoder.encoder at all).
C3_BLOCK_ID = 17
C4_BLOCK_ID = 18


def load_state_dict(nemo_path: pathlib.Path) -> dict:
    import io
    import torch  # local import: only needed here, not at module import time

    with tarfile.open(nemo_path) as tf:
        # Member name varies (some .nemo archives store "./model_weights.ckpt",
        # others "model_weights.ckpt") -- match by basename, not exact path.
        candidates = [m for m in tf.getmembers()
                      if pathlib.PurePosixPath(m.name).name == "model_weights.ckpt"]
        if len(candidates) != 1:
            raise SystemExit(
                f"{nemo_path}: expected exactly one model_weights.ckpt member, "
                f"found {len(candidates)}")
        member = candidates[0]
        extracted = tf.extractfile(member)
        if extracted is None:
            raise SystemExit(
                f"{nemo_path}: model_weights.ckpt member is not a regular file")
        data = extracted.read()
    # Read into memory (no extract-to-disk) and restrict unpickling to tensors/
    # primitives (weights_only=True) -- this is a checkpoint of unknown/external
    # provenance, not code we trust to run arbitrary pickle bytecode.
    sd = torch.load(io.BytesIO(data), map_location="cpu", weights_only=True)
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    return {k: v.detach().numpy() for k, v in sd.items() if hasattr(v, "detach")}


def fold_bn(weight: np.ndarray, bn_prefix: str, sd: dict) -> tuple[np.ndarray, np.ndarray]:
    """Conv1d(bias=False) -> BatchNorm1d, folded into (weight, bias).

    y = gamma * (W*x - mean) / sqrt(var + eps) + beta
      = (gamma/sqrt(var+eps)) * W * x + (beta - gamma*mean/sqrt(var+eps))
    """
    gamma = sd[f"{bn_prefix}.weight"]
    beta = sd[f"{bn_prefix}.bias"]
    mean = sd[f"{bn_prefix}.running_mean"]
    var = sd[f"{bn_prefix}.running_var"]
    scale = gamma / np.sqrt(var + BN_EPS)
    folded_weight = weight * scale[:, None, None]
    folded_bias = beta - mean * scale
    return folded_weight, folded_bias


def tensor_names(ld: qt.LayerDesc) -> dict | None:
    """Resolve a LayerDesc to its checkpoint tensor names, or None if it carries
    no weights (OP_ADD / OP_REQUANT)."""
    bid = ld.block_id
    if ld.op in (qt.OP_ADD, qt.OP_REQUANT):
        return None
    if ld.block_name.endswith("/res"):
        return {"conv": f"encoder.encoder.{bid}.res.0.0.conv.weight",
                "bn": f"encoder.encoder.{bid}.res.0.1"}
    if bid == C4_BLOCK_ID:
        return {"conv": "decoder.decoder_layers.0.weight",
                "bias": "decoder.decoder_layers.0.bias"}
    if bid == C3_BLOCK_ID:
        return {"conv": f"encoder.encoder.{bid}.mconv.0.conv.weight",
                "bn": f"encoder.encoder.{bid}.mconv.1"}
    m = re.search(r"\.r(\d+)$", ld.block_name)
    if not m:
        raise ValueError(f"unexpected block_name {ld.block_name!r} for {ld.op}")
    r = int(m.group(1))
    slot = 5 * r
    if ld.op == qt.OP_DW:
        return {"conv": f"encoder.encoder.{bid}.mconv.{slot}.conv.weight"}
    return {"conv": f"encoder.encoder.{bid}.mconv.{slot + 1}.conv.weight",
            "bn": f"encoder.encoder.{bid}.mconv.{slot + 2}"}


def export(nemo_path: pathlib.Path, out_dir: pathlib.Path) -> None:
    print(f"loading {nemo_path} ...")
    sd = load_state_dict(nemo_path)
    print(f"  {len(sd)} tensors in the checkpoint")

    layers = qt.expand()
    print(f"quartznet_topology.expand(): {len(layers)} descriptors")

    checkpoint_map: dict[str, dict] = {}
    folded: dict[str, np.ndarray] = {}
    used_tensors: set[str] = set()
    total_weights = 0

    for ld in layers:
        names = tensor_names(ld)
        if names is None:
            continue  # OP_ADD / OP_REQUANT: no weights, nothing to bind

        conv_w = sd[names["conv"]]
        used_tensors.add(names["conv"])

        if "bn" in names:
            bn_prefix = names["bn"]
            w, b = fold_bn(conv_w, bn_prefix, sd)
            for suffix in (".weight", ".bias", ".running_mean", ".running_var"):
                used_tensors.add(bn_prefix + suffix)
        elif "bias" in names:
            w = conv_w
            b = sd[names["bias"]]
            used_tensors.add(names["bias"])
        else:
            # Depthwise: no BN, no bias in the float model at all (topology.py's
            # header: "no BatchNorm and no activation between the depthwise and
            # the pointwise"). Genuinely zero, not a placeholder -- the int8
            # pipeline still carries a (bias, q_mult, rshift) triple per the
            # descriptor format, but bias is 0 until quantize_multiplier picks a
            # real output scale from calibration (a later step).
            w = conv_w
            b = np.zeros(ld.c_out, dtype=np.float32)

        # Descriptor layout: DW [C,K] (squeeze in_ch/groups==1), PW/C3/C4 [oc,ic]
        # (squeeze kernel==1) -- see the file header, no transpose needed either way.
        w_flat = w.squeeze(1) if ld.op == qt.OP_DW else w.squeeze(-1)
        if w_flat.size != ld.n_weights:
            raise SystemExit(
                f"descriptor {ld.layer_id} ({ld.block_name}): weight count "
                f"{w_flat.size} != expected {ld.n_weights}")

        folded[f"w{ld.layer_id}"] = w_flat.astype(np.float32).reshape(-1)
        folded[f"b{ld.layer_id}"] = b.astype(np.float32)
        checkpoint_map[str(ld.layer_id)] = {
            "block_name": ld.block_name, "op": qt.OP_NAMES[ld.op], **names}
        total_weights += w_flat.size

    # ── G2.2 checks ───────────────────────────────────────────────────────
    ok = True
    if total_weights != qt.EXPECTED_PARAMS:
        print(f"*** PARAM COUNT MISMATCH: {total_weights:,} != "
              f"{qt.EXPECTED_PARAMS:,} ***")
        ok = False
    else:
        print(f"parameter count: {total_weights:,} == EXPECTED_PARAMS (exact match)")

    n_weight_bearing = sum(1 for ld in layers if tensor_names(ld) is not None)
    if len(checkpoint_map) != n_weight_bearing:
        print(f"*** {n_weight_bearing - len(checkpoint_map)} weight-bearing "
              f"descriptor(s) unbound ***")
        ok = False
    print(f"{len(checkpoint_map)}/{n_weight_bearing} weight-bearing descriptors "
          f"bound to a named checkpoint tensor")

    # Every conv/bn/bias tensor actually used should be a REAL key (KeyError
    # above would already have caught a typo); check the other direction too:
    # what's left unused should only be preprocessor/spec_augment/BN-counter
    # bookkeeping, not a silently-skipped weight.
    all_tensor_keys = set(sd.keys())
    unused = sorted(k for k in all_tensor_keys - used_tensors
                     if not k.endswith("num_batches_tracked")
                     and not k.startswith("preprocessor.")
                     and not k.startswith("spec_augment."))
    if unused:
        print(f"*** {len(unused)} unexplained unused tensor(s): {unused[:10]} ***")
        ok = False
    else:
        print("zero unexplained unused tensors "
              "(everything outside the mapped layers is preprocessor/"
              "spec_augment/BN-counter bookkeeping)")

    # Numeric guard against a wrong BN_EPS (or any other BN-fold error) --
    # G2.2's other checks are all shape/name checks and pass identically
    # whether BN_EPS is right or 100x wrong, which is exactly how a real
    # ~1e31-logit-blowup bug shipped through this gate once already (see
    # BN_EPS's own comment above). A folded conv weight/bias in a sane fp32
    # range is necessary (not sufficient) for a correct fold -- catches gross
    # eps/scale errors here instead of only downstream in a WER number.
    bad_scale = [lid for lid in checkpoint_map
                 if float(np.abs(folded[f"w{lid}"]).max()) > 100.0
                 or float(np.abs(folded[f"b{lid}"]).max()) > 100.0]
    if bad_scale:
        print(f"*** {len(bad_scale)} layer(s) with folded |weight| or |bias| "
              f"> 100 (a BN_EPS/fold error inflates these silently -- shape/"
              f"name checks above cannot catch it): {bad_scale[:10]} ***")
        ok = False
    else:
        print("folded weight/bias magnitudes all sane (< 100) -- "
              "no BN-fold scale blowup")

    if not ok:
        raise SystemExit("quartznet_nemo_export: G2.2 checks FAILED")
    print("G2.2: PASS")

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "checkpoint_map.json").write_text(json.dumps(checkpoint_map, indent=2))
    np.savez(out_dir / "folded_weights.npz", **folded)
    print(f"wrote {out_dir / 'checkpoint_map.json'}")
    print(f"wrote {out_dir / 'folded_weights.npz'} ({len(folded) // 2} layers)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("nemo_path", type=pathlib.Path)
    ap.add_argument("--out", type=pathlib.Path,
                     default=pathlib.Path(__file__).resolve().parents[2]
                     / "build" / "quartznet_nemo")
    args = ap.parse_args()
    export(args.nemo_path, args.out)


if __name__ == "__main__":
    main()
