"""LTT against max-conformal at the same false-positive-rate targets (the table of ``cli.ltt_table``), from the saved scores of one probe.

For each FPR target f and each run (a split), two constant thresholds are chosen on the run's calibration episodes:

    max-conformal   level alpha = f, from the calibration successes alone. It promises an FPR level and nothing about TPR, so it has no TPR-early
                    target, and it always returns a threshold (+inf, which never alarms, when alpha is below 1 / (n + 1)).
    LTT             certifies FPR <= f together with a TPR-early target t, with ``n_starts`` fixed-sequence walks at level delta. t is the highest
                    target on a grid that at least ``MIN_SHARE`` of the runs certify, so that the row describes a typical run; a run that cannot
                    certify it has no threshold (LTT abstains).

Each threshold is measured on the run's held-out seen episodes and on its unseen episodes (``run_eval.rates_at``): the share of runs in which each
target holds, and the mean and standard deviation over the runs of the FPR and of the TPR early. LTT's shares are over the runs it certifies,
max-conformal's over all runs. The LTT guarantee is for episodes exchangeable with the calibration set, i.e. the seen episodes.
"""

import math
from dataclasses import dataclass
from typing import Mapping

import numpy as np

from ..calibration.ltt import RiskCounts, fixed_sequence_test, joint_p_values, risk_counts, select_threshold
from ..calibration.max_conformal import MaxConformal
from ..data.manifest import Episode
from ..data.splits import Split
from .run_eval import rates_at
from .tables import LttCells, LttRow

N_STARTS = 40
MIN_SHARE = 0.5
TPR_TARGETS = tuple(round(0.10 + 0.05 * i, 2) for i in range(18))  # the grid the TPR-early target is taken from: 10% ... 95%
FPR_TARGETS = (0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50)
LTT, MAX_CONFORMAL = "LTT", "Max-conformal"


@dataclass(frozen=True)
class Run:
    """One run prepared for both calibrators: what both fit on (the calibration episodes) and the episodes they are measured on."""
    scores: Mapping[str, np.ndarray]
    counts: RiskCounts
    max_conformal: MaxConformal
    seen: list[Episode]
    unseen: list[Episode]
    window: int


def prepare(scores: Mapping[str, np.ndarray], split: Split, episodes: Mapping[str, Episode], window: int) -> Run:
    cal = [episodes[k] for k in split.calibration]
    successes, failures = [e for e in cal if e.success], [e for e in cal if not e.success]
    success_scores = [scores[e.key] for e in successes]
    counts = risk_counts(success_scores, [scores[e.key] for e in failures], [e.onset for e in failures])
    return Run(scores, counts, MaxConformal().fit(success_scores), [episodes[k] for k in split.seen], [episodes[k] for k in split.unseen], window)


def ltt_threshold(run: Run, tpr_target: float, fpr_target: float, delta: float, n_starts: int) -> float | None:
    """The LTT threshold certifying TPR early >= ``tpr_target`` and FPR <= ``fpr_target`` (fractions), None if the run cannot certify it. The
    same steps as ``calibrate_ltt``, on the counts of the run, which do not depend on the targets."""
    p = joint_p_values(run.counts, 1 - tpr_target, fpr_target)
    return select_threshold(fixed_sequence_test(p, delta, n_starts), run.counts).tau


def strongest_tpr_target(runs: list[Run], fpr_target: float, delta: float, n_starts: int, tpr_targets=TPR_TARGETS, min_share: float = MIN_SHARE) -> float | None:
    """The highest TPR-early target on ``tpr_targets`` that at least ``min_share`` of the runs certify together with ``fpr_target``; None if there is none."""
    needed = math.ceil(min_share * len(runs))
    certified = [t for t in tpr_targets if sum(ltt_threshold(r, t, fpr_target, delta, n_starts) is not None for r in runs) >= needed]
    return max(certified, default=None)


def set_cells(thresholds: list[tuple[Run, float]], fpr_target: float, tpr_target: float | None, eval_set: str) -> LttCells | None:
    """The cells of one evaluation set, over the runs that have a threshold and episodes in the set (None when there are none)."""
    rates = [rates_at(tau, run.scores, getattr(run, eval_set), run.window) for run, tau in thresholds if getattr(run, eval_set)]
    if not rates:
        return None
    fpr, tpr = np.array([r["fpr"] for r in rates]), np.array([r["tpr_early"] for r in rates])
    return LttCells(100 * float(np.mean(fpr <= 100 * fpr_target)), None if tpr_target is None else 100 * float(np.mean(tpr >= 100 * tpr_target)),
                    (float(fpr.mean()), float(fpr.std())), (float(tpr.mean()), float(tpr.std())))


def compare(runs: list[Run], fpr_targets=FPR_TARGETS, *, delta: float, n_starts: int = N_STARTS, tpr_targets=TPR_TARGETS,
            min_share: float = MIN_SHARE) -> list[LttRow]:
    """Two rows per FPR target, LTT then max-conformal. When no TPR-early target is certified by enough runs the LTT row has no target and no cells."""
    rows = []
    for f in fpr_targets:
        t = strongest_tpr_target(runs, f, delta, n_starts, tpr_targets, min_share)
        ltt = [] if t is None else [(r, tau) for r in runs if (tau := ltt_threshold(r, t, f, delta, n_starts)) is not None]
        rows.append(LttRow(f, t, LTT, True, len(ltt), len(runs), *(set_cells(ltt, f, t, s) if ltt else None for s in ("seen", "unseen"))))
        conformal = [(r, float(r.max_conformal.threshold(f, 1)[0])) for r in runs]
        rows.append(LttRow(f, None, MAX_CONFORMAL, False, len(runs), len(runs), *(set_cells(conformal, f, None, s) for s in ("seen", "unseen"))))
    return rows
