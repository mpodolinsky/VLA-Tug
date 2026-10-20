"""Compare the new pure-logic modules with the previous pipeline on real LIBERO-10 files (SAFE-LSTM, seeds 0-2).

    python -I checks/check_against_previous_pipeline.py pi05|groot

Checks: episode splits, SAFE-band first-alarm steps of every failure at 7 alphas (seen and unseen groups), unseen-task rates
and the max-conformal formula. Not part of the package: it reads outputs/ of the main checkout, which the package never does.
"""

import csv
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np

PKG = Path(__file__).resolve().parents[1]  # failure_probes/
WT = PKG.parent  # repo root of this checkout (holds scripts/)
# Checkout that holds the old outputs/ (the main repo when run from a worktree); override with REPO_MAIN.
MAIN = Path(os.environ["REPO_MAIN"]) if "REPO_MAIN" in os.environ else (WT.parents[2] if WT.parent.name == "worktrees" else WT)
sys.path.insert(0, str(PKG))

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _previous import PreviousMaxConformal  # noqa: E402
from fp.calibration.safe_band import SafeBand  # noqa: E402
from fp.data.manifest import Episode  # noqa: E402
from fp.data.splits import build_split, pick_unseen_tasks  # noqa: E402
from fp.metrics.outcomes import alarm_step, classify  # noqa: E402
from fp.metrics.rates import rates_from_outcomes  # noqa: E402

POLICY = sys.argv[1] if len(sys.argv) > 1 else "pi05"
DIRS = {"pi05": "pi0.5-libero-10", "groot": "groot-libero-10"}
WINDOW = {"pi05": 10, "groot": 6}[POLICY]

spec = importlib.util.spec_from_file_location("old_dt", WT / f"scripts/baselines/safe/detection_time_{POLICY}.py")
old = importlib.util.module_from_spec(spec)
sys.modules["old_dt"] = old
spec.loader.exec_module(old)

OUT = MAIN / "outputs" / DIRS[POLICY]
rows = list(csv.DictReader(open(OUT / "collect-stac-k10/manifest.csv")))
by_dir = {r["dir"]: r for r in rows}
eps = []
for r in rows:
    ok = r["success"] == "True"
    onset = None if ok else int(float(r["vlm_failure_onset_frame"])) // int(r["replan_steps"])
    eps.append(Episode(r["dir"], ok, r["task"], int(r["n_policy"]), onset, r["scene_variant"]))
ep_by_key = {e.key: e for e in eps}
res = json.load(open(OUT / "safe-baseline/split65-15-20-perseed/detection_time.json"))["result"]["lstm"]
ALPHAS = ["0.01", "0.05", "0.1", "0.15", "0.2", "0.25", "0.3"]

print(f"== {DIRS[POLICY]}: window {WINDOW} steps, {len(eps)} episodes")
for seed in (0, 1, 2):
    old_split = json.load(open(OUT / f"baseline_split_65_15_20_seed{seed}.json"))["episodes"]
    new = build_split(eps, f"paper_seed{seed}", seed, lambda e: e.task, pick_unseen_tasks([e.task for e in eps], seed))
    old_cal, old_eval = old.split_test(old_split, by_dir, seed, 0.5)
    same = dict(train=new.train == sorted(old_split["train"]), val=new.val == sorted(old_split["val"]), unseen=new.unseen == sorted(old_split["unseen"]),
                calibration=new.calibration == old_cal, seen=new.seen == old_eval)
    print(f"seed {seed} split identical to previous: {same}")

    sc = np.load(OUT / f"safe-baseline/split65-15-20-perseed/lstm/scores_seed{seed}.npz")
    cal_scores = [sc[k] for k in new.calibration if ep_by_key[k].success]
    band = SafeBand(seed=seed).fit(cal_scores)
    maxc = PreviousMaxConformal().fit(cal_scores)  # the interpolated rule of the previous pipeline (the package now uses the exact order statistic)
    n_diff = n_tot = 0
    rate_diff = []
    for a in ALPHAS:
        old_run = res[str(seed)][a]
        for keys, old_eps in ((new.unseen, old_run["episodes"]), (new.seen, old_run["seen_eval"]["episodes"])):
            thr = band.threshold(float(a), max(len(sc[k]) for k in keys))
            old_alarm = {x["dir"]: x["alarm"] for x in old_eps}
            for k in keys:
                if not ep_by_key[k].success:
                    n_tot += 1
                    n_diff += alarm_step(sc[k], thr) != old_alarm[k]
        thr = band.threshold(float(a), max(len(sc[k]) for k in new.unseen))
        r = rates_from_outcomes([classify(sc[k], thr, ep_by_key[k].onset, WINDOW) for k in new.unseen])
        rate_diff.append(max(abs(r.fpr - 100 * old_run["fpr"]), abs(r.tpr - 100 * old_run["tpr"]), abs(r.tpr_on_time - 100 * old_run["tpr_on_time"]),
                             abs(r.tpr_early - 100 * (old_run["tpr_early"] + old_run["tpr_on_time"]))))
    print(f"seed {seed} SAFE band: failure alarm steps differing from previous: {n_diff} of {n_tot}; max unseen rate difference {max(rate_diff):.4f} pp")
    m = np.array([s.max() for s in cal_scores])
    for a in (0.05, 0.2):
        lvl = min(1.0, np.ceil((len(m) + 1) * (1 - a)) / len(m))
        assert abs(maxc.threshold(a, 1)[0] - float(np.quantile(m, lvl))) < 1e-12
    print(f"seed {seed} max-conformal equals the previous formula; calibration successes: {len(cal_scores)}")
