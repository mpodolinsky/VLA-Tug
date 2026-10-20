"""ActProbe: action-chunk magnitude and temporal-consistency error (ACM, TCE) with a small net
over task-conditioned action features. Inputs are the predicted action chunks only (no visual features).
The network is imported from the ActProbe reference code; this module only adapts the data.
"""

import numpy as np

from ..models.base import Scorer


class ActProbeScorer(Scorer):
    name = "actprobe"

    def fit(self, train_keys: list[str], val_keys: list[str]) -> None:
        raise NotImplementedError

    def score(self, keys: list[str]) -> dict[str, np.ndarray]:
        raise NotImplementedError
