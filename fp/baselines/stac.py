"""STAC (Agia et al.): disagreement between K sampled action chunks across consecutive policy calls.

``stac`` is the original K-sample MMD statistic; ``stac_single`` is the single-sample variant.
Both are computed with SAFE's ``failure_metrics`` and cumulated over time. They read
``sampled_action_chunks`` (T, K=10, H, A) from data/<dataset>/stac/, so they exist only for
datasets with ``has_k10``. Calibration is max-conformal.
"""

import numpy as np

from ..models.base import Scorer


class StacScorer(Scorer):
    def __init__(self, single: bool):
        self.name = "stac_single" if single else "stac"
        self.single = single

    def fit(self, train_keys: list[str], val_keys: list[str]) -> None:
        """No parameters to fit."""

    def score(self, keys: list[str]) -> dict[str, np.ndarray]:
        raise NotImplementedError
