"""SAFE failure predictors (Gu et al., NeurIPS 2025, arXiv 2506.09937): a per-step MLP with cumulative score, and an LSTM.

SAFE-MLP:  per-step score s_t = sigmoid(MLP(x_t)); the failure score is the running sum of s_t.
SAFE-LSTM: per-step score s_t = sigmoid(Linear(LSTM(x)_t)); trained with BCE against the episode label.
Both are single classes shared by every dataset (no per-policy copies).
"""

import torch
from torch import nn


class SafeMLP(nn.Module):
    def __init__(self, dim: int, hidden: int = 256):
        super().__init__()
        raise NotImplementedError

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """(B, T, dim) -> cumulative scores (B, T)."""
        raise NotImplementedError


class SafeLSTM(nn.Module):
    def __init__(self, dim: int, hidden: int = 256):
        super().__init__()
        raise NotImplementedError

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """(B, T, dim) -> per-step probabilities (B, T)."""
        raise NotImplementedError
