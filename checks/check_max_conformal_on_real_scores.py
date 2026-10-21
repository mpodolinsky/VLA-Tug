"""Max-conformal on the saved scores of earlier models: is the false-alarm guarantee met, do ties matter, is the sweep monotone?

    python -I checks/check_max_conformal_on_real_scores.py

Per run (calibration successes of the run's split):
  * ties: number of distinct calibration maxima, and how many equal the threshold itself;
  * realised false-alarm rate on the held-out seen half (exchangeable with the calibration half, so the guarantee applies)
    for the alarm rule score >= threshold (the package) and score > threshold (the rule that is valid with ties);
  * FPR on the unseen tasks along the full alpha grid: it must be non-decreasing in alpha (threshold is non-increasing in alpha).
Runs: LIBERO-X key-plot models (14 each), LIBERO-10 SAFE-MLP and SAFE-LSTM (pi0.5 and GR00T, seeds 0-2).
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))  # python -I does not add the script folder
from _old_outputs import LIBERO10, LIBEROX_MODELS, liberox_episodes, liberox_runs, load_episodes, unseen_tasks  # noqa: E402
from fp.calibration.max_conformal import MaxConformal  # noqa: E402
from fp.data.splits import build_split, pick_unseen_tasks  # noqa: E402
from fp.metrics.curves import ALPHAS_FULL as ALPHAS  # noqa: E402  (this check looks at how FPR moves over every alpha, not only the headline ones)

LEVELS = (0.05, 0.1, 0.2, 0.3)


def run_stats(ep, scores, split):
    cal = [scores[k] for k in split.calibration if ep[k].success]
    mc = MaxConformal().fit(cal)
    seen = np.array([scores[k].max() for k in split.seen if ep[k].success])
    unseen = np.array([scores[k].max() for k in split.unseen if ep[k].success])
    out = dict(n=len(cal), distinct=len(np.unique(mc.maxima)), n_seen_succ=len(seen))
    for a in LEVELS:
        thr = mc.threshold(a, 1)[0]
        out[a] = dict(ge=float(np.mean(seen >= thr)), gt=float(np.mean(seen > thr)), tied=int(np.sum(mc.maxima == thr)) if np.isfinite(thr) else 0,
                      inf=bool(np.isinf(thr)))
    fpr = np.array([np.mean(unseen >= mc.threshold(a, 1)[0]) for a in ALPHAS])
    out["fpr_drops"] = int((np.diff(fpr) < -1e-12).sum())
    return out


def report(label, runs):
    n = [r["n"] for r in runs]
    seen = [r["n_seen_succ"] for r in runs]
    print(f"{label:22s} runs {len(runs):2d}  calibration successes {min(n)}-{max(n)}  held-out seen successes {min(seen)}-{max(seen)}"
          f"  runs with any tied calibration maxima {sum(r['distinct'] < r['n'] for r in runs)}"
          f"  FPR non-monotone in alpha: {sum(r['fpr_drops'] > 0 for r in runs)} runs")
    for a in LEVELS:
        ge = np.array([r[a]["ge"] for r in runs])
        gt = np.array([r[a]["gt"] for r in runs])
        tied = sum(r[a]["tied"] > 1 for r in runs)
        print(f"    alpha {a:.2f}: held-out FPR  >= rule {100 * ge.mean():5.1f}%   > rule {100 * gt.mean():5.1f}%   (SE {100 * ge.std() / np.sqrt(len(ge)):.1f}; "
              f"runs with several maxima equal to the threshold: {tied})")


print("== LIBERO-X")
eps = liberox_episodes()
ep = {e.key: e for e in eps}
for name, pattern in LIBEROX_MODELS.items():
    runs = []
    for rs in liberox_runs().values():
        for lv, seed in rs:
            sc = np.load(Path(str(pattern).format(lv=lv)) / f"scores_seed{seed}.npz")
            runs.append(run_stats(ep, sc, build_split(eps, lv, seed, lambda e: e.task, unseen_tasks(lv))))
    report(name, runs)

print("\n== LIBERO-10")
for policy in ("pi05", "groot"):
    eps10, _ = load_episodes(LIBERO10[policy] / "collect-stac-k10/manifest.csv")
    ep10 = {e.key: e for e in eps10}
    for method in ("mlp", "lstm"):
        runs = []
        for seed in range(3):
            split = build_split(eps10, f"paper_seed{seed}", seed, lambda e: e.task, pick_unseen_tasks([e.task for e in eps10], seed))
            sc = np.load(LIBERO10[policy] / f"safe-baseline/split65-15-20-perseed/{method}/scores_seed{seed}.npz")
            runs.append(run_stats(ep10, sc, split))
        report(f"{policy} SAFE-{method.upper()}", runs)
