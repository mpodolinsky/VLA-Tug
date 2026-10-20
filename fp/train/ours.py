"""Training of our probes: builds batches from a label variant and defines the loss.

Loss = masked BCE on the score (positives re-weighted by n_neg / n_pos)
     + lam_mode * class-balanced CE of the mode head on positive steps   (modes, modes_clip)
     + lam_clip * InfoNCE (tau 0.07) of the CLIP head on positive steps  (clip, modes_clip)

Also provides ``OursScorer``, the Scorer wrapper (fit = train with ``loop.fit`` and save weights).
"""

import numpy as np

from ..data.labels import LabelledEpisode
from ..models.base import Scorer
from ..models.ours import ProbeConfig


def probe_loss(outputs: dict, batch: dict, lam_mode: float, lam_clip: float):
    raise NotImplementedError


class OursScorer(Scorer):
    def __init__(self, cfg: ProbeConfig, label_variant: str):
        self.cfg = cfg
        self.label_variant = label_variant
        self.name = f"{cfg.trunk}_{cfg.heads}__{label_variant}"

    def fit(self, train_keys: list[str], val_keys: list[str]) -> None:
        raise NotImplementedError

    def score(self, keys: list[str]) -> dict[str, np.ndarray]:
        raise NotImplementedError
