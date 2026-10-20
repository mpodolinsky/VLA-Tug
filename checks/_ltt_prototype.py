"""Prototype LTT procedure that is NOT in the package, shared by the LTT checks.

The risks are monotone in tau (a higher tau alarms on fewer episodes): the false-alarm risk falls with tau and the miss-early risk
rises. So the FPR hypotheses are walked from the highest tau downwards and the TPR-early hypotheses from the lowest tau upwards,
each as a fixed-sequence test at level delta / 2, and the accepted set is the intersection. A fixed-sequence test is valid for any
fixed order, and a union bound covers the two risks; monotonicity only makes it powerful.
"""

import numpy as np
from scipy.stats import binom

import _old_outputs  # noqa: F401  (puts the package on sys.path)
from fp.calibration.ltt import TAUS, RiskCounts


def _walk(p: np.ndarray, level: float, from_top: bool) -> np.ndarray:
    ok = np.zeros(len(p), dtype=bool)
    order = range(len(p) - 1, -1, -1) if from_top else range(len(p))
    for j in order:
        if p[j] > level:
            break
        ok[j] = True
    return ok


def two_walk_accept(counts: RiskCounts, alpha_tpr_early: float, alpha_fpr: float, delta: float) -> np.ndarray:
    p_fpr = binom.cdf(counts.n_false_alarm, counts.n_succ, alpha_fpr)
    p_early = binom.cdf(counts.n_miss_early, counts.n_fail, alpha_tpr_early)
    assert len(p_fpr) == len(TAUS)
    return _walk(p_fpr, delta / 2, from_top=True) & _walk(p_early, delta / 2, from_top=False)
