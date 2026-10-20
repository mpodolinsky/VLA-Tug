"""FailureSpot: weak-supervision pretraining on action signals (consistency, norm, EMA), then
fine-tuning on a small labelled budget (15% of training failures) with BCE. Our own implementation
(no official code). Action-chunk horizon H and replan K come from the DatasetSpec, not constants.
"""

import numpy as np

from ..models.base import Scorer


def action_signals(chunks: np.ndarray, replan_steps: int) -> np.ndarray:
    """(T, H, A) predicted chunks -> (T, 3) consistency, norm and EMA signals."""
    raise NotImplementedError


class FailureSpotScorer(Scorer):
    name = "failurespot"

    def fit(self, train_keys: list[str], val_keys: list[str]) -> None:
        raise NotImplementedError

    def score(self, keys: list[str]) -> dict[str, np.ndarray]:
        raise NotImplementedError
