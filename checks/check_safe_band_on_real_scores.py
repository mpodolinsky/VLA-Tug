"""The SAFE band on the saved scores of earlier models: how well-behaved is it, and what does the quantile rule change?

    python -I checks/check_safe_band_on_real_scores.py

For each run (calibration successes of the run's split, band seed = split seed) and the 40-point alpha grid of the old key plot:
  * smallest modulation sigma_alpha(t) over t and alpha (division by ~1e-8 would blow up the conformity scores);
  * is the band b_alpha(t) pointwise non-increasing in alpha, and is the measured FPR(alpha) on the unseen tasks non-decreasing?
    (an early stop at the first FPR >= 40 would assume so; the package runs the full grid);
  * area under TPR early vs FPR to 40% with SAFE's interpolated quantile and with the finite-sample rank.
Runs: LIBERO-X key-plot models (14 runs each), LIBERO-10 SAFE-MLP and SAFE-LSTM (pi0.5 and GR00T, seeds 0-2).
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))  # python -I does not add the script folder
from _old_outputs import ALPHAS40, LIBERO10, LIBEROX_MODELS, WINDOW, liberox_episodes, liberox_runs, load_episodes, unseen_tasks  # noqa: E402
from fp.calibration.safe_band import SafeBand, tfunc_modulation  # noqa: E402
from fp.data.splits import build_split, pick_unseen_tasks  # noqa: E402
from fp.metrics.curves import area_to_fpr, sweep  # noqa: E402


def diagnose(eps_by_key, scores, split, seed, window):
    """Returns dict with min sigma, monotonicity violations and the two areas for one run."""
    cal = [scores[k] for k in split.calibration if eps_by_key[k].success]
    band = SafeBand(seed=seed).fit(cal)
    mu = band.fit_part.mean(axis=0, keepdims=True)
    sigmas = np.stack([tfunc_modulation(band.fit_part, mu, a)[0] for a in ALPHAS40])
    length = band.fit_part.shape[1]
    bands = np.stack([band.threshold(a, length) for a in ALPHAS40])
    unseen = [eps_by_key[k] for k in split.unseen]
    out = dict(
        min_sigma=float(sigmas.min()),
        sigma_ratio=float(sigmas.min() / sigmas.max()),
        band_increases=float((np.diff(bands, axis=0) > 1e-12).mean()),  # share of (alpha step, t) pairs where the band rises with alpha
        band_max_rise=float(max(np.diff(bands, axis=0).max(), 0.0)),
    )
    for rule, fs in (("interp", False), ("finite", True)):
        rows = sweep(SafeBand(seed=seed, finite_sample=fs).fit(cal), scores, unseen, window, alphas=ALPHAS40, stop_fpr=None)
        fpr = np.array([r["fpr"] for r in rows])
        a = area_to_fpr(fpr, np.array([r["tpr_early"] for r in rows]))
        out[f"area_{rule}"], out[f"extended_{rule}"] = a.value, a.extended
        # what the optional early stop (stop_fpr = 40) would give; its rows are a prefix of the full grid, which is the package default
        first = next((i for i, f in enumerate(fpr) if f >= 40.0), len(fpr) - 1)
        stopped = rows[: first + 1]
        out[f"area_stop_{rule}"] = area_to_fpr(np.array([r["fpr"] for r in stopped]), np.array([r["tpr_early"] for r in stopped])).value
        out[f"fpr_drops_{rule}"] = int((np.diff(fpr) < -1e-9).sum())
        out[f"fpr_max_drop_{rule}"] = float(max(-np.diff(fpr).min(), 0.0))
    return out


def summarise(label, results):
    r = results
    print(f"{label:34s} runs {len(r):2d} | min sigma {min(x['min_sigma'] for x in r):.2e} (min ratio to max sigma {min(x['sigma_ratio'] for x in r):.1e})"
          f" | band rises with alpha in {sum(x['band_increases'] > 0 for x in r)} runs (max rise {max(x['band_max_rise'] for x in r):.3f})"
          f" | FPR drops with alpha: interp {sum(x['fpr_drops_interp'] > 0 for x in r)} runs (max {max(x['fpr_max_drop_interp'] for x in r):.1f} pp),"
          f" finite {sum(x['fpr_drops_finite'] > 0 for x in r)} runs (max {max(x['fpr_max_drop_finite'] for x in r):.1f} pp)")


print("== LIBERO-X (unseen tasks)")
eps = liberox_episodes()
ep = {e.key: e for e in eps}
runs = liberox_runs()
areas = {}
for name, pattern in LIBEROX_MODELS.items():
    results = {}
    for sp, rs in runs.items():
        for lv, seed in rs:
            f = Path(str(pattern).format(lv=lv)) / f"scores_seed{seed}.npz"
            split = build_split(eps, lv, seed, lambda e: e.task, unseen_tasks(lv))
            results.setdefault(sp, []).append(diagnose(ep, np.load(f), split, seed, WINDOW["liberox"]))
    summarise(name, [x for v in results.values() for x in v])
    areas[name] = results

print("\narea under TPR early vs FPR to 40% (unseen tasks), SAFE band; per cell: interpolated rule | finite-sample rule, full grid; [same with the sweep stopped at the first FPR >= 40]")
for name, results in areas.items():
    cells = []
    for sp, v in results.items():
        mean = lambda key: np.mean([x[key] for x in v])
        cells.append(f"{sp} {mean('area_interp'):5.1f} | {mean('area_finite'):5.1f} [{mean('area_stop_interp'):5.1f} | {mean('area_stop_finite'):5.1f}]")
    print(f"  {name:16s} " + "   ".join(cells))

print("\n== LIBERO-10 SAFE-MLP and SAFE-LSTM (unseen tasks)")
for policy in ("pi05", "groot"):
    eps10, _ = load_episodes(LIBERO10[policy] / "collect-stac-k10/manifest.csv")
    ep10 = {e.key: e for e in eps10}
    for method in ("mlp", "lstm"):
        results = []
        for seed in range(3):
            split = build_split(eps10, f"paper_seed{seed}", seed, lambda e: e.task, pick_unseen_tasks([e.task for e in eps10], seed))
            sc = np.load(LIBERO10[policy] / f"safe-baseline/split65-15-20-perseed/{method}/scores_seed{seed}.npz")
            results.append(diagnose(ep10, sc, split, seed, WINDOW[policy]))
        summarise(f"{policy} SAFE-{method.upper()}", results)
        print(f"{'':34s} area interp {np.mean([x['area_interp'] for x in results]):5.1f} | finite {np.mean([x['area_finite'] for x in results]):5.1f}"
              f" (stopped short: {sum(x['extended_interp'] for x in results)}/3 | {sum(x['extended_finite'] for x in results)}/3)")
