"""The single generic training loop.

Takes a model, a loss function and train/val batch iterators; handles the optimiser, early
stopping on validation loss (patience), seeding, and saving the best weights. Methods differ only
in the loss they pass in, never in a copy of this loop.
"""

from dataclasses import dataclass
from typing import Callable, Iterable

import torch


@dataclass(frozen=True)
class TrainConfig:
    lr: float = 1e-3
    weight_decay: float = 0.0
    max_epochs: int = 100
    patience: int = 15
    seed: int = 0


def pad_batch(arrays: list, lengths: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
    """Pad variable-length (n_i, d) episodes into (B, T, d) and return lengths (B,)."""
    raise NotImplementedError


def fit(
    model: torch.nn.Module,
    loss_fn: Callable[[torch.nn.Module, dict], torch.Tensor],
    train_batches: Callable[[], Iterable[dict]],
    val_batches: Callable[[], Iterable[dict]],
    cfg: TrainConfig,
    weights_path: str,
) -> dict:
    """Train, keep the best validation weights, write them to ``weights_path``, return a history dict."""
    raise NotImplementedError
