"""LTT on the saved scores of earlier models: what can it certify at these sample sizes, and does the guarantee hold held-out?

    python -I checks/check_ltt_on_real_scores.py

1. Calibration failure and success counts per dataset, and the smallest alpha a single binomial test can reach with zero
   violations at level L:  alpha_min = 1 - L ** (1 / n).
2. For several target pairs, on the LIBERO-X key-plot models (14 runs each), for the package's five-start procedure and for the
   two-walk prototype (_ltt_prototype.py): certified runs, which tau is selected (grid index), and for the certified runs the
   TPR_early and FPR of the selected tau on the held-out seen half (and how often both targets hold there) and on the unseen tasks.
"""

import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))  # python -I does not add the script folder
from _ltt_prototype import two_walk_accept  # noqa: E402
from _old_outputs import LIBERO10, LIBEROX_MODELS, liberox_episodes, liberox_runs, load_episodes, unseen_tasks  # noqa: E402
from fp.calibration.ltt import TAUS, fixed_sequence_test, joint_p_values, risk_counts, select_threshold  # noqa: E402
from fp.data.splits import build_split, pick_unseen_tasks  # noqa: E402

DELTA = 0.10
PAIRS = [(0.7, 0.2), (0.7, 0.3), (0.6, 0.3), (0.5, 0.3)]  # (alpha_tpr_early, alpha_fpr)
PROCEDURES = {
    "five starts": lambda counts, a_t, a_f: fixed_sequence_test(joint_p_values(counts, a_t, a_f), DELTA, 5),
    "two walks": lambda counts, a_t, a_f: two_walk_accept(counts, a_t, a_f, DELTA),
}


def rates_at(ep, scores, keys, idx):
    """(TPR_early, FPR) in % of the constant threshold TAUS[idx] on the given episodes."""
    fail = [k for k in keys if not ep[k].success]
    succ = [k for k in keys if ep[k].success]
    c = risk_counts([scores[k] for k in succ], [scores[k] for k in fail], [ep[k].onset for k in fail])
    return 100 * float(c.tpr_early[idx]), 100 * float(c.fpr[idx])


def counts_of(ep, scores, keys):
    fail = [k for k in keys if not ep[k].success]
    return risk_counts([scores[k] for k in keys if ep[k].success], [scores[k] for k in fail], [ep[k].onset for k in fail])


print("== 1. sample sizes and the smallest alpha reachable with zero violations: alpha_min = 1 - L^(1/n)")
eps = liberox_episodes()
ep = {e.key: e for e in eps}
runs = liberox_runs()
sizes = {"LIBERO-X (14 runs)": []}
for rs in runs.values():
    for lv, seed in rs:
        split = build_split(eps, lv, seed, lambda e: e.task, unseen_tasks(lv))
        sizes["LIBERO-X (14 runs)"].append((sum(not ep[k].success for k in split.calibration), sum(ep[k].success for k in split.calibration)))
for policy in ("pi05", "groot"):
    eps10, _ = load_episodes(LIBERO10[policy] / "collect-stac-k10/manifest.csv")
    e10 = {e.key: e for e in eps10}
    sizes[f"LIBERO-10 {policy} (3 seeds)"] = []
    for seed in range(3):
        split = build_split(eps10, f"paper_seed{seed}", seed, lambda e: e.task, pick_unseen_tasks([e.task for e in eps10], seed))
        sizes[f"LIBERO-10 {policy} (3 seeds)"].append((sum(not e10[k].success for k in split.calibration), sum(e10[k].success for k in split.calibration)))
for name, ns in sizes.items():
    nf, nsu = np.array(ns).T
    print(f"{name:26s} calibration failures {nf.min()}-{nf.max()}, successes {nsu.min()}-{nsu.max()}")
    for lvl, lab in ((DELTA / 5, "five starts, level 0.02"), (DELTA / 2, "two walks, level 0.05")):
        print(f"{'':26s} alpha_min ({lab}): alpha_tpr_early >= {1 - lvl ** (1 / nf.max()):.2f}-{1 - lvl ** (1 / nf.min()):.2f}"
              f", alpha_fpr >= {1 - lvl ** (1 / nsu.max()):.2f}-{1 - lvl ** (1 / nsu.min()):.2f}")

print("\n== 2. LIBERO-X, 14 runs per model. Per procedure: certified runs; selected tau (grid index 0-180: count); "
      "held-out seen half TPR_early / FPR (both targets hold in k of n); unseen TPR_early / FPR")
for name, pattern in LIBEROX_MODELS.items():
    loaded = []
    for rs in runs.values():
        for lv, seed in rs:
            sc = np.load(Path(str(pattern).format(lv=lv)) / f"scores_seed{seed}.npz")
            split = build_split(eps, lv, seed, lambda e: e.task, unseen_tasks(lv))
            loaded.append((sc, split, counts_of(ep, sc, split.calibration)))
    print(f"\n{name}")
    for a_t, a_f in PAIRS:
        print(f"  alpha_tpr_early {a_t}, alpha_fpr {a_f}")
        for pname, accept in PROCEDURES.items():
            chosen, seen, unseen, holds = [], [], [], 0
            for sc, split, counts in loaded:
                res = select_threshold(accept(counts, a_t, a_f), counts)
                if not res.certified:
                    continue
                idx = int(np.argmin(np.abs(TAUS - res.tau)))
                chosen.append(idx)
                s = rates_at(ep, sc, split.seen, idx)
                seen.append(s)
                unseen.append(rates_at(ep, sc, split.unseen, idx))
                holds += (s[0] >= 100 * (1 - a_t)) and (s[1] <= 100 * a_f)
            if not chosen:
                print(f"    {pname:11s} certified  0/14")
                continue
            tally = ", ".join(f"{i}x{c}" for i, c in sorted(Counter(chosen).items()))
            sm, um = np.mean(seen, axis=0), np.mean(unseen, axis=0)
            print(f"    {pname:11s} certified {len(chosen):2d}/14 | tau index {tally} | seen {sm[0]:5.1f} / {sm[1]:5.1f} (hold {holds}/{len(chosen)})"
                  f" | unseen {um[0]:5.1f} / {um[1]:5.1f}")
