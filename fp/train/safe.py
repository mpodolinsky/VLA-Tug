"""Training of SAFE-MLP and SAFE-LSTM with SAFE's losses.

MLP loss:  relu(cumulative score) on successes, -cumulative score on failures (per sequence mean).
LSTM loss: per-step BCE against the episode label.
Both use SAFE's labels (failure = positive at every step, i.e. onset_traj) and score at the shared
index k = (shortest rollout of the task) - 1 as in the original code.
"""

import numpy as np

from ..models.base import Scorer


class SafeScorer(Scorer):
    def __init__(self, kind: str):
        assert kind in ("mlp", "lstm")
        self.name = f"safe_{kind}"
        self.kind = kind

    def fit(self, train_keys: list[str], val_keys: list[str]) -> None:
        raise NotImplementedError

    def score(self, keys: list[str]) -> dict[str, np.ndarray]:
        raise NotImplementedError
