"""Our probes: a per-modality projection followed by an LSTM or MLP trunk, with optional auxiliary heads.

    features (T, 6144) -> per-modality Linear(2048, P) + ReLU -> causal MHA (+ residual, LayerNorm)
        -> concat (3P) -> trunk (LSTM | per-step MLP) -> score head
                                                       -> mode head   (optional, failure-mode classes)
                                                       -> CLIP head   (optional, caption embedding)

Variants of the auxiliary heads are ``plain``, ``modes``, ``clip`` and ``modes_clip``.
The labelling variant (onset_pre, onset, ...) is a training choice, not part of the model.
"""

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class ProbeConfig:
    trunk: str  # "lstm" | "mlp"
    heads: str  # "plain" | "modes" | "clip" | "modes_clip"
    hidden: int = 256
    proj: int = 128  # per-modality projection width (hidden / 2)
    n_modes: int = 0
    clip_dim: int = 0
    dropout: float = 0.0


class Probe(nn.Module):
    """Returns a dict: ``score`` (B, T) logits, plus ``mode`` (B, T, n_modes) and ``clip`` (B, T, clip_dim) when enabled."""

    def __init__(self, cfg: ProbeConfig):
        super().__init__()
        self.cfg = cfg
        raise NotImplementedError

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> dict[str, torch.Tensor]:
        raise NotImplementedError


def build_probe(cfg: ProbeConfig) -> Probe:
    return Probe(cfg)
