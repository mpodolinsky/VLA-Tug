"""Common interface for everything that turns an episode into a per-step failure score.

A ``Scorer`` is anything that, after fitting on training episodes, returns for every episode a
1-D array of per-step scores (higher = more likely to fail). Calibration and evaluation only
ever see these arrays, so learned probes, distance baselines and STAC are interchangeable.
"""

from abc import ABC, abstractmethod

import numpy as np


class Scorer(ABC):
    name: str

    @abstractmethod
    def fit(self, train_keys: list[str], val_keys: list[str]) -> None:
        """Fit on training episodes; ``val_keys`` is used only for early stopping."""

    @abstractmethod
    def score(self, keys: list[str]) -> dict[str, np.ndarray]:
        """Per-step scores for every key, over the full rollout (no cut)."""

    def save(self, path: str) -> None:
        """Write learned weights/statistics. Required for every method that has any."""
        raise NotImplementedError

    def load(self, path: str) -> None:
        raise NotImplementedError
