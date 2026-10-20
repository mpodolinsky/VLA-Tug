"""Alpha sweep and the headline area.

``sweep`` evaluates one fitted calibrator on a set of episodes for increasing alpha (a looser
threshold each time) and returns one row per alpha: alpha, fpr, tpr, tpr_early, tpr_on_time, ldr.
By default the whole alpha grid is run (0.005 to 0.995), which is immune to a calibrator whose FPR is not monotone in alpha
(the SAFE band); ``stop_fpr`` is an optional early stop at the first alpha whose empirical FPR reaches it. Rows beyond
40% FPR are stored and ignored by the area.

``area_to_fpr`` is the one area function: area under TPR_early vs FPR from 0 to ``fpr_max``,
divided by ``fpr_max`` (so a curve at 100% everywhere scores 100). The curve starts at (0, 0), is
made monotone with a running maximum, and is held flat past its last point; the returned flag says
whether that extension was needed.
"""

from dataclasses import dataclass

import numpy as np

from ..calibration.base import Calibrator
from ..data.manifest import Episode
from .outcomes import classify
from .rates import rates_from_outcomes

FPR_MAX = 40.0
ALPHAS = np.round(np.arange(0.005, 1.0, 0.005), 3)  # 0.005 ... 0.995, strictly inside (0, 1)


@dataclass(frozen=True)
class Area:
    value: float  # in [0, 100] when y is a percentage
    extended: bool  # True if the curve never reached fpr_max


def area_to_fpr(fpr: np.ndarray, y: np.ndarray, fpr_max: float = FPR_MAX) -> Area:
    fpr, y = np.asarray(fpr, dtype=float), np.asarray(y, dtype=float)
    if fpr.shape != y.shape:
        raise ValueError("fpr and y must have the same shape")
    keep = ~(np.isnan(fpr) | np.isnan(y))
    fpr, y = np.concatenate([[0.0], fpr[keep]]), np.concatenate([[0.0], y[keep]])
    order = np.argsort(fpr, kind="stable")
    fpr, y = fpr[order], np.maximum.accumulate(y[order])
    # Several operating points can share one FPR; after the running max the last of them is the best.
    xs = np.unique(fpr)
    ys = np.array([y[fpr == x][-1] for x in xs])
    extended = bool(xs[-1] < fpr_max)
    y_end = float(np.interp(fpr_max, xs, ys))  # np.interp holds the last value flat past xs[-1]
    knots_x, knots_y = np.append(xs[xs < fpr_max], fpr_max), np.append(ys[xs < fpr_max], y_end)
    area = float(np.sum(np.diff(knots_x) * (knots_y[1:] + knots_y[:-1]) / 2.0))
    return Area(area / fpr_max, extended)


def area_of(
    calibrator: Calibrator,
    calibration_success_scores: list[np.ndarray],
    scores: dict[str, np.ndarray],
    episodes: list[Episode],
    window: int,
    alphas: np.ndarray = ALPHAS,
) -> Area:
    """Fit ``calibrator`` on the calibration successes, sweep alpha on ``episodes`` and return the area to FPR 40 of TPR early."""
    calibrator.fit(calibration_success_scores)
    rows = sweep(calibrator, scores, episodes, window, alphas)
    return area_to_fpr(np.array([r["fpr"] for r in rows]), np.array([r["tpr_early"] for r in rows]))


def sweep(
    calibrator: Calibrator,
    scores: dict[str, np.ndarray],
    episodes: list[Episode],
    window: int,
    alphas: np.ndarray = ALPHAS,
    stop_fpr: float | None = None,
) -> list[dict]:
    """One row per alpha, ascending. ``calibrator`` is already fitted on the calibration successes."""
    length = max(len(scores[e.key]) for e in episodes)
    rows = []
    for alpha in alphas:
        thr = calibrator.threshold(float(alpha), length)
        r = rates_from_outcomes([classify(scores[e.key], thr, e.onset, window) for e in episodes])
        rows.append(dict(alpha=float(alpha), fpr=r.fpr, tpr=r.tpr, tpr_early=r.tpr_early, tpr_on_time=r.tpr_on_time, ldr=r.ldr))
        if stop_fpr is not None and r.fpr >= stop_fpr:
            break
    return rows
