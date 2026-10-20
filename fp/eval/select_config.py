"""Choose the best configuration of a baseline on the VALIDATION episodes, never on the unseen tasks.

For one baseline (e.g. Euclidean with its 6 configurations) and one dataset with several runs (splits):
  1. per run, fit each configuration on the run's training episodes;
  2. calibrate it with its protocol calibrator on the run's calibration successes;
  3. sweep alpha on the run's validation episodes and take the area under TPR early vs FPR up to 40%;
  4. the best configuration is the one with the highest MEAN validation area over the dataset's runs
     (ties: the first in grid order).

The only episodes this module ever asks a scorer to score are the ``calibration`` and ``val`` keys of each split; ``seen`` and
``unseen`` are never read here (a test checks this).
"""

from dataclasses import dataclass
from typing import Callable, Mapping

import numpy as np

from ..calibration.protocol import make_calibrator
from ..data.manifest import Episode
from ..data.splits import Split
from ..metrics.curves import area_of
from ..models.base import Scorer


@dataclass(frozen=True)
class Selection:
    best: str  # name of the winning configuration
    mean_val_area: dict[str, float]  # configuration name -> mean validation area over the runs (grid order)
    per_run: dict[str, list[float]]  # configuration name -> validation area of each run (same run order for every configuration)


def validation_area(scorer: Scorer, split: Split, episodes: Mapping[str, Episode], window: int) -> float:
    """Fit ``scorer`` on the split's training episodes and return its validation area (calibrator from the protocol)."""
    n_fail = sum(not episodes[k].success for k in split.val)
    if n_fail == 0 or n_fail == len(split.val):
        raise ValueError(f"the validation episodes of {split.name} need both failures and successes to have an area")
    scorer.fit(split.train, split.val)
    scores = scorer.score(split.calibration + split.val)
    cal_success = [scores[k] for k in split.calibration if episodes[k].success]
    val_episodes = [episodes[k] for k in split.val]
    calibrator = make_calibrator(scorer.name, seed=split.seed)
    return area_of(calibrator, cal_success, scores, val_episodes, window).value


LIBEROX_KINDS = ("task", "signature", "skill")


def split_group(split: Split) -> str:
    """The held-out set a split belongs to: 'task', 'signature' or 'skill' for LIBERO-X splits (named task_seed0,
    signature_seed1, skill_<name>), 'all' for every other split (the LIBERO-10 runs are pooled)."""
    first = split.name.split("_")[0]
    return first if first in LIBEROX_KINDS else "all"


def select_best_per_group(
    make_scorers: Callable[[], list[Scorer]],
    splits: list[Split],
    episodes: Mapping[str, Episode],
    window: int,
    group: Callable[[Split], str] = split_group,
) -> dict[str, "Selection"]:
    """One selection per group of splits (LIBERO-X: the 3 task seeds, the 3 signature seeds, the 8 skill folds; LIBERO-10: all runs)."""
    groups: dict[str, list[Split]] = {}
    for s in splits:
        groups.setdefault(group(s), []).append(s)
    return {g: select_best(make_scorers, members, episodes, window) for g, members in groups.items()}


def select_best(
    make_scorers: Callable[[], list[Scorer]],
    splits: list[Split],
    episodes: Mapping[str, Episode],
    window: int,
) -> Selection:
    """``make_scorers`` returns fresh, unfitted scorers (one per configuration, always in the same order) for each split."""
    per_run: dict[str, list[float]] = {}
    for split in splits:
        for scorer in make_scorers():
            per_run.setdefault(scorer.name, []).append(validation_area(scorer, split, episodes, window))
    mean = {name: float(np.mean(v)) for name, v in per_run.items()}
    best = max(mean, key=lambda n: (mean[n], -list(mean).index(n)))  # highest mean; ties go to the earlier configuration
    return Selection(best, mean, per_run)
