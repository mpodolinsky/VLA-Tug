"""Is the LTT guarantee actually met? Simulation with known true risks.

    python -I checks/check_ltt_validity_simulation.py

Synthetic episodes (failures with an onset whose score ramps up around it, successes with noise and occasional spikes) come from one
generator. The true risks at every tau of the grid are computed on a large reference sample. Over many calibration draws we count how
often a procedure ACCEPTS a tau whose true TPR_early or FPR violates its target (family-wise error, must be <= delta), and how often the
SELECTED tau violates, and how often anything is certified. Procedures: the package's five-start fixed-sequence test and the prototype
two-walk test of check_ltt_on_real_scores.py.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # python -I does not add the script folder
from _ltt_prototype import two_walk_accept  # noqa: E402
from fp.calibration.ltt import TAUS, fixed_sequence_test, joint_p_values, risk_counts, select_threshold  # noqa: E402

L = 35
rng = np.random.default_rng(0)


def failures(n):
    onsets = rng.integers(8, 26, n)
    lead = rng.integers(-3, 12, n)  # steps before the onset at which the score starts to rise (negative: after)
    slope = rng.uniform(0.4, 2.0, n)
    t = np.arange(L)[None, :]
    z = -4 + slope[:, None] * np.clip(t - (onsets - lead)[:, None], 0, None) + rng.normal(0, 0.5, (n, L))
    return list(1 / (1 + np.exp(-z))), list(onsets)


def successes(n):
    z = -4 + rng.normal(0, 0.9, (n, L)) + 2.5 * (rng.random((n, L)) < 0.03)
    return list(1 / (1 + np.exp(-z)))


def true_risks(n_ref=20000):
    f, o = failures(n_ref)
    s = successes(n_ref)
    c = risk_counts(s, f, o)
    return 1 - c.tpr_early, c.fpr  # (miss_early risk, false-alarm risk) per tau


R_MISS, R_FA = true_risks()
DELTA, REPS = 0.10, 600
for n_cal, (a_tpr, a_fpr) in ((60, (0.6, 0.2)), (60, (0.5, 0.3)), (120, (0.6, 0.2)), (120, (0.5, 0.15)), (30, (0.7, 0.3))):
    bad = (R_MISS > a_tpr) | (R_FA > a_fpr)  # taus violating a target in truth
    print(f"\nn_cal = {n_cal} failures and {n_cal} successes, targets alpha_tpr_early {a_tpr}, alpha_fpr {a_fpr}: "
          f"{int((~bad).sum())} of {len(TAUS)} grid taus are truly valid")
    out = {"five starts": [], "two walks": []}
    for _ in range(REPS):
        f, o = failures(n_cal)
        counts = risk_counts(successes(n_cal), f, o)
        p = joint_p_values(counts, a_tpr, a_fpr)
        for name, acc in (("five starts", fixed_sequence_test(p, DELTA, 5)), ("two walks", two_walk_accept(counts, a_tpr, a_fpr, DELTA))):
            sel = select_threshold(acc, counts)
            sel_idx = int(np.argmin(np.abs(TAUS - sel.tau))) if sel.certified else None
            out[name].append((bool((acc & bad).any()), sel.certified, bool(sel.certified and bad[sel_idx]), int(acc.sum())))
    for name, rows in out.items():
        r = np.array(rows, dtype=float)
        cert = r[:, 1].sum()
        print(f"  {name:12s} accepted a truly invalid tau in {100 * r[:, 0].mean():4.1f}% of draws (limit {100 * DELTA:.0f}%)"
              f" | certified {100 * r[:, 1].mean():5.1f}% | selected tau invalid in {100 * r[:, 2].sum() / max(cert, 1):4.1f}% of certified draws"
              f" | accepted taus (mean) {r[:, 3].mean():5.1f}")
