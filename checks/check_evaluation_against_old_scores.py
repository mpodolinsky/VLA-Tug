"""fp.eval.run_eval.evaluate_scores against the previous pipeline, end to end on its saved LIBERO-X scores.

    REPO_MAIN=/path/to/project python -I checks/check_evaluation_against_old_scores.py

For the four LIBERO-X models of the previous key plot (SAFE-LSTM, LSTM onset / onset_pre / onset_traj) and each of the 14 runs, the old
score files go through ``evaluate_scores`` (split from ``make_splits``, calibrator fitted on the split's calibration successes, the
previous interpolated max-conformal rule compared in float32 as the previous code did, the previous 40-point alpha grid) and the
unseen curve is compared with the rows the previous pipeline stored (outputs/early_vs_fpr/levels_by_split.json). This tests the glue --
which episodes calibrate, which are evaluated -- independently of any retraining. It then prints the mean unseen area of the package's
own rule on the old 40-point grid (the numbers first documented in docs/CALIBRATION.md), on the protocol's grid (alpha <= 0.4) and on the
full alpha grid.

Not part of the package: it reads outputs/ of the main checkout, which the package never does.
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))  # python -I does not add the script folder
from _old_outputs import ALPHAS40, LIBEROX_MODELS, MAIN, liberox_runs  # noqa: E402
from _previous import Float32Threshold  # noqa: E402
from fp.calibration.max_conformal import MaxConformal  # noqa: E402
from fp.data.inputs import load_inputs  # noqa: E402
from fp.eval.run_eval import evaluate_scores  # noqa: E402
from fp.metrics.curves import ALPHAS, ALPHAS_FULL  # noqa: E402

inputs = load_inputs("liberox_groot")
stored_rows = json.load(open(MAIN / "outputs/early_vs_fpr/levels_by_split.json"))
areas = {}
print("== unseen curves of the old score files through evaluate_scores, against the stored rows (previous rule, 40-point grid)")
for name, pattern in LIBEROX_MODELS.items():
    worst, n_runs = 0.0, 0
    for group, runs in liberox_runs().items():
        stored = stored_rows[group].get(name, [])
        i = 0
        for level, seed in runs:
            f = Path(str(pattern).format(lv=level)) / f"scores_seed{seed}.npz"
            if not f.exists():
                continue
            with np.load(f) as z:
                scores = {k: z[k] for k in z.files}
            split = inputs.splits[level if level.startswith("skill_") else f"{level}_seed{seed}"]
            old_rule = evaluate_scores(scores, split, inputs.episodes, inputs.window, Float32Threshold(), ALPHAS40)["unseen"]
            points = sorted(((r["tpr"], r["tpr_early"], r["tpr_on_time"], r["fpr"]) for r in old_rule.rows), key=lambda p: p[0])
            mine = np.maximum.accumulate(np.vstack([np.zeros(4), np.array(points)]), axis=0)[:, [3, 1, 2, 0]]
            worst = max(worst, float(np.abs(mine - np.array(stored[i])).max()))
            n_runs, i = n_runs + 1, i + 1
            for label, grid in (("40-point grid", ALPHAS40), ("alpha <= 0.4", ALPHAS), ("full grid", ALPHAS_FULL)):
                exact = evaluate_scores(scores, split, inputs.episodes, inputs.window, MaxConformal(), grid)["unseen"]
                areas.setdefault((label, name, group), []).append(exact.area)
    print(f"{name:16s} {n_runs:2d} runs: largest difference to the stored rows {worst:.2e} percentage points")

for label in ("40-point grid", "alpha <= 0.4", "full grid"):
    print(f"== mean unseen area to FPR 40 (package rule: exact order statistic, {label}), by held-out set")
    for name in LIBEROX_MODELS:
        print(f"  {name:16s}" + "".join(f"  {group} {np.mean(areas[(label, name, group)]):5.1f}" for group in liberox_runs() if (label, name, group) in areas))
print("The areas first documented (docs/CALIBRATION.md) are the 40-point-grid rows; the protocol's grid is alpha <= 0.4 (the full grid is shown for reference).")
