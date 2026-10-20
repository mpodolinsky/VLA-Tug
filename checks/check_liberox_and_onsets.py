"""Checks on LIBERO-X (the key-plot data) and on the onset definition.

    python -I checks/check_liberox_and_onsets.py

1. Splits: build_split with stratum (outcome, level) against every old LIBERO-X split file; smallest calibration-success count.
2. Curves: the new MaxConformal + sweep on the old 40-point alpha grid against the rows stored for the key plot
   (outputs/early_vs_fpr/levels_by_split.json), for four models; then area_to_fpr of the new rows as a sanity number.
3. Onsets: failures whose onset (frame // replan_steps) is outside the rollout, per dataset.

Not part of the package: it reads outputs/ of the main checkout, which the package never does.
"""

import csv
import json
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))  # python -I does not add the script folder
from _old_outputs import ALPHAS40, LIBEROX_MODELS, MAIN, OLD_SPLITS, X, liberox_runs, load_episodes, unseen_tasks  # noqa: E402
from _previous import Float32Threshold, PreviousMaxConformal  # noqa: E402
from fp.calibration.max_conformal import MaxConformal  # noqa: E402
from fp.data.splits import build_split  # noqa: E402
from fp.metrics.curves import area_to_fpr, sweep  # noqa: E402

WINDOW = 6


def old_split_test(test, level_of, success_of, seed, cal_frac=0.5):
    """levels_modality/eval_levels.py split_test, copied verbatim apart from the lookups."""
    rng = random.Random(seed)
    strata = {}
    for d in sorted(test):
        strata.setdefault((success_of[d], level_of[d]), []).append(d)
    cal, ev = [], []
    for key in sorted(strata):
        dirs = strata[key]
        rng.shuffle(dirs)
        k = int(round(cal_frac * len(dirs)))
        cal += dirs[:k]
        ev += dirs[k:]
    return sorted(cal), sorted(ev)


eps, rows = load_episodes(X / "collect-stac-k10-level123/manifest.csv")
ep = {e.key: e for e in eps}
success_of = {r["dir"]: r["success"] for r in rows}
level_of = {r["dir"]: r["level"] for r in rows}

print(f"== 1. LIBERO-X splits ({len(eps)} episodes)")
n_files = n_ok = 0
min_cal = (10**9, "")
mismatch = []
for f in sorted(OLD_SPLITS.glob("*_seed*.json")):
    name, seed = f.stem.rsplit("_seed", 1)
    seed = int(seed)
    old = json.load(open(f))["episodes"]
    new = build_split(eps, name, seed, lambda e: e.task, unseen_tasks(name))
    oc, oe = old_split_test(old["test"], level_of, success_of, seed)
    same = (new.train == sorted(old["train"]) and new.val == sorted(old["val"]) and new.unseen == sorted(old["unseen"])
            and new.calibration == oc and new.seen == oe)
    n_files += 1
    n_ok += same
    if not same:
        mismatch.append(f.stem)
    n_cal = sum(ep[k].success for k in new.calibration)
    if n_cal < min_cal[0]:
        min_cal = (n_cal, f.stem)
print(f"{n_ok} of {n_files} split files reproduced exactly (train, val, unseen, calibration half, seen half); mismatches: {mismatch or 'none'}")
print(f"smallest number of calibration successes in any split: {min_cal[0]} ({min_cal[1]})")

print("== 2. key-plot curves (max-conformal alpha sweep, 40-point grid, unseen tasks)")
by_split = json.load(open(MAIN / "outputs/early_vs_fpr/levels_by_split.json"))
RUNS, MODELS = liberox_runs(), LIBEROX_MODELS
areas = {}  # (model, split) -> area of the package's max-conformal rule
areas_prev = {}  # same with the previous interpolated rule
for name, pattern in MODELS.items():
    worst = worst32 = 0.0
    n_cmp = n_diff = 0
    for sp, runs in RUNS.items():
        stored = by_split[sp].get(name, [])
        i = 0
        for lv, seed in runs:
            f = Path(str(pattern).format(lv=lv)) / f"scores_seed{seed}.npz"
            if not f.exists():
                continue
            sc = np.load(f)
            old = json.load(open(OLD_SPLITS / f"{lv}_seed{seed}.json"))["episodes"]
            new = build_split(eps, lv, seed, lambda e: e.task, unseen_tasks(lv))
            cal_scores = [sc[k] for k in new.calibration if ep[k].success]

            def stored_diff(calibrator):
                r_ = sweep(calibrator.fit(cal_scores), sc, [ep[k] for k in new.unseen], WINDOW, alphas=ALPHAS40, stop_fpr=None)
                pts = sorted(((x["tpr"], x["tpr_early"], x["tpr_on_time"], x["fpr"]) for x in r_), key=lambda p: p[0])
                mine = np.maximum.accumulate(np.vstack([np.zeros(4), np.array(pts)]), axis=0)[:, [3, 1, 2, 0]]
                return float(np.abs(mine - np.array(stored[i])).max()), r_

            d64, rows_prev = stored_diff(PreviousMaxConformal())
            d32, _ = stored_diff(Float32Threshold())
            worst, worst32, n_diff = max(worst, d64), max(worst32, d32), n_diff + (d64 > 1e-9)
            rows_new = sweep(MaxConformal().fit(cal_scores), sc, [ep[k] for k in new.unseen], WINDOW, alphas=ALPHAS40, stop_fpr=None)
            n_cmp += 1
            ap = area_to_fpr(np.array([r["fpr"] for r in rows_prev]), np.array([r["tpr_early"] for r in rows_prev]))
            areas_prev.setdefault((name, sp), []).append(ap)
            i += 1
            a = area_to_fpr(np.array([r["fpr"] for r in rows_new]), np.array([r["tpr_early"] for r in rows_new]))
            areas.setdefault((name, sp), []).append(a)
    print(f"{name:16s} {n_cmp:2d} runs; previous rule in float64: {n_diff} runs differ from the stored rows, max {worst:.4f} pp; previous rule in float32: max {worst32:.6f} pp")

print("== sanity: area under TPR early vs FPR to 40% on the old 40-point grid; mean over runs (runs whose curve stopped short of 40% FPR)")
for label, table in (("package rule: exact order statistic", areas), ("previous rule: interpolated quantile", areas_prev)):
    print(label)
    for name in MODELS:
        cells = []
        for sp in RUNS:
            a = table.get((name, sp), [])
            if a:
                cells.append(f"{sp} {np.mean([x.value for x in a]):5.1f} ({sum(x.extended for x in a)}/{len(a)})")
        print(f"  {name:16s} " + "   ".join(cells))

print("== 3. onsets outside the rollout (failures with onset = frame // replan_steps >= n_policy, or no onset label)")
for label, path in (("libero10_groot", MAIN / "outputs/groot-libero-10/collect-stac-k10/manifest.csv"),
                    ("libero10_pi05", MAIN / "outputs/pi0.5-libero-10/collect-stac-k10/manifest.csv"),
                    ("liberox_groot", X / "collect-stac-k10-level123/manifest.csv")):
    rs = list(csv.DictReader(open(path)))
    fails = [r for r in rs if r["success"] != "True"]
    blank = sum(r["vlm_failure_onset_frame"].strip() == "" for r in fails)
    past = [r for r in fails if r["vlm_failure_onset_frame"].strip() and int(float(r["vlm_failure_onset_frame"])) // int(r["replan_steps"]) >= int(r["n_policy"])]
    print(f"{label}: {len(fails)} failures, {blank} without onset, {len(past)} with onset at or past the last step"
          + (f" (largest overshoot {max(int(float(r['vlm_failure_onset_frame'])) // int(r['replan_steps']) - int(r['n_policy']) + 1 for r in past)} steps)" if past else ""))
