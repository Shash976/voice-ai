"""quartznet_fp32.py — FP32 forward-pass graph for QuartzNet 15x5, built
directly from quartznet_topology.expand() so layer_ids match
build/quartznet_nemo/folded_weights.npz (Stage 7 Gap 2 A2's export).

NOT built from split_wide_layers()/quartznet_descriptors.py: the channel
split of C3's 512->1024 pointwise renumbers every layer_id from 184 onward
(expand(): 186 descriptors: split_wide_layers(): 187), so the packed binary
table's numbering and folded_weights.npz's numbering diverge after C3.

Doubles as the ONNX export target for A4's ORT static per-channel PTQ
calibration: real nn.Conv1d submodules give each weight a named initializer
(e.g. "convs.7.weight") instead of an anonymous Constant node, so
onnxruntime.quantization's per-channel calibration output can be mapped back
to a layer_id by name, not by (ambiguous) shape.
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import quartznet_topology as qt  # noqa: E402


class QuartzNetFP32(nn.Module):
    """Descriptor-driven fp32 forward pass. `descs` defaults to
    quartznet_topology.expand()'s real 15x5 table; overridable for tests."""

    def __init__(self, npz: dict, descs: list[qt.LayerDesc] | None = None):
        super().__init__()
        self.descs = descs if descs is not None else qt.expand()
        self.convs = nn.ModuleList()
        self.conv_ix: dict[int, int] = {}
        for ld in self.descs:
            if ld.op == qt.OP_ADD:
                continue
            if ld.op == qt.OP_REQUANT:
                raise ValueError(
                    f"OP_REQUANT (descriptor {ld.layer_id}) is not emitted by "
                    f"the real 15x5 topology and has no fp32 handler")
            groups = ld.c_out if ld.op == qt.OP_DW else 1
            conv = nn.Conv1d(ld.c_in, ld.c_out, ld.k, stride=ld.stride,
                             padding=ld.pad, dilation=ld.dilation,
                             groups=groups, bias=True)
            w = torch.from_numpy(np.asarray(npz[f"w{ld.layer_id}"])).to(torch.float32)
            # descriptor layout -> torch Conv1d weight layout.
            #   DW: flat [C, K]   -> [C, 1, K]    (groups == C, in_channels/groups == 1)
            #   PW: flat [oc, ic] -> [oc, ic, 1]  (k == 1 for every PW in this table)
            conv.weight.data.copy_(
                w.view(ld.c_out, 1, ld.k) if ld.op == qt.OP_DW
                else w.view(ld.c_out, ld.c_in, 1))
            conv.bias.data.copy_(
                torch.from_numpy(np.asarray(npz[f"b{ld.layer_id}"])).to(torch.float32))
            self.conv_ix[ld.layer_id] = len(self.convs)
            self.convs.append(conv)
        self.eval()
        for p in self.parameters():
            p.requires_grad_(False)

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """mel: [1, N_MEL, T_in] -> logits: [1, N_CLASSES, T_out]."""
        buf: dict[int, torch.Tensor] = {qt.BUF_IN: mel}
        for ld in self.descs:
            if ld.op == qt.OP_ADD:
                y = F.relu(buf[ld.in_buf] + buf[ld.res_buf])
            else:
                y = self.convs[self.conv_ix[ld.layer_id]](buf[ld.in_buf])
                if ld.relu:
                    y = F.relu(y)
            buf[ld.out_buf] = y
        return buf[qt.BUF_LOGITS]


def load(npz_path=None) -> QuartzNetFP32:
    if npz_path is None:
        npz_path = (pathlib.Path(__file__).resolve().parents[2]
                    / "build" / "quartznet_nemo" / "folded_weights.npz")
    return QuartzNetFP32(dict(np.load(npz_path)))


@torch.no_grad()
def logits(model: QuartzNetFP32, feat: np.ndarray) -> np.ndarray:
    """feat: [T_in, N_MEL] float32 log-mel (quartznet_audio.extract_logmel's
    layout) -> [T_out, N_CLASSES] float32."""
    x = torch.from_numpy(np.ascontiguousarray(feat.T, dtype=np.float32)).unsqueeze(0)
    return model(x)[0].transpose(0, 1).numpy()
