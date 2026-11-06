"""The calibrated alarm of each method in the figure: one operating point, fitted on the split's calibration episodes.

    VLA-Tug      Learn-then-Test, a constant threshold: false-positive rate <= 15% and early true-positive rate >= 15%, jointly with confidence 90%
    SAFE-LSTM    SAFE's functional conformal band at level 15%: a threshold that varies with the step
    STAC         max-conformal at level 15%: a constant threshold

These are the calibrators of the paper's tables (``fp.eval.ltt_compare``), fitted on the same calibration episodes and the same saved scores.
"""

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from fp.data.manifest import Episode
from fp.data.splits import Split
from fp.eval import ltt_compare

FPR_TARGET = 0.15
TPR_EARLY_TARGET = 0.15
DELTA = 0.10  # LTT: with probability at least 90% over the calibration sample, both targets hold


@dataclass(frozen=True)
class Alarm:
    """What one method alarms on: its scores' saved configuration, how it was calibrated and its threshold at every step."""

    label: str
    method: str  # the saved configuration the scores come from, e.g. safe_lstm__lr1e-3_reg1e-2
    calibration: str  # shown on the page
    threshold: np.ndarray | None  # (longest rollout,); None when the calibrator could not certify its targets (the method never alarms)

    @property
    def constant(self) -> bool:
        return self.threshold is not None and bool(np.all(self.threshold == self.threshold[0]))

    def for_episode(self, n_steps: int) -> np.ndarray:
        """The threshold of the first ``n_steps`` steps; +inf when there is none, so that nothing alarms."""
        return np.full(n_steps, np.inf) if self.threshold is None else self.threshold[:n_steps]


def calibrate(kind: str, label: str, method: str, scores: Mapping[str, np.ndarray], split: Split, episodes: Mapping[str, Episode], window: int,
              fpr: float = FPR_TARGET, tpr_early: float = TPR_EARLY_TARGET) -> Alarm:
    """``kind`` is 'ltt', 'safe_band' or 'max_conformal'; the calibrator is fitted on the calibration episodes of ``split`` and nothing else, at the operating point
    (``fpr``, ``tpr_early``): the false-positive-rate level of all three, and the early true-positive-rate target of Learn-then-Test."""
    run = ltt_compare.prepare(scores, split, episodes, window)
    longest = max(len(s) for s in scores.values())
    if kind == "ltt":
        tau = ltt_compare.ltt_threshold(run, tpr_early, fpr, DELTA, ltt_compare.N_STARTS)
        threshold = None if tau is None else np.full(longest, tau)
        text = f"Learn-then-Test: FPR ≤ {fpr:.0%} and early TPR ≥ {tpr_early:.0%}, confidence {1 - DELTA:.0%}"
    elif kind == "safe_band":
        threshold = run.safe_band.threshold(fpr, longest)
        text = f"SAFE functional conformal band, FPR level {fpr:.0%}"
    elif kind == "max_conformal":
        threshold = np.full(longest, float(run.max_conformal.threshold(fpr, 1)[0]))
        text = f"max-conformal, FPR level {fpr:.0%}"
    else:
        raise ValueError(f"unknown calibration {kind!r}")
    return Alarm(label, method, text, threshold)
