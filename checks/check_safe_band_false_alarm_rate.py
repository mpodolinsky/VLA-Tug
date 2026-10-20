"""Realised false-alarm rate of the SAFE band on exchangeable success episodes, through SafeBand.fit / threshold / alarm_step.

Compared with the formula alpha + (1 - 2 alpha) / (m + 1) (m = size of the conformal part) for SAFE's interpolated quantile, and
with alpha for the finite-sample rank. Fixed-length episodes and variable-length episodes (lengths uniform on 20..40, so edge
padding matters). Numbers in docs/CALIBRATION.md.

    python -I checks/check_safe_band_false_alarm_rate.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fp.calibration.safe_band import SafeBand  # noqa: E402
from fp.metrics.outcomes import alarm_step  # noqa: E402

rng = np.random.default_rng(0)
REPS, N_TEST = 300, 300


def episodes(n, variable):
    lengths = rng.integers(20, 41, n) if variable else np.full(n, 40)
    return [np.cumsum(rng.random(T) * 0.05) + rng.normal(0, 0.03, T) for T in lengths]


def false_alarm_rate(n_cal, alpha, variable, finite_sample, seed):
    band = SafeBand(seed=seed, finite_sample=finite_sample).fit(episodes(n_cal, variable))
    tests = episodes(N_TEST, variable)
    thr = band.threshold(alpha, max(len(t) for t in tests))
    return np.mean([alarm_step(t, thr) is not None for t in tests])


print("realised false-alarm rate (mean over %d calibration draws); columns: formula for SAFE's rule | SAFE rule fixed length | SAFE rule variable length | finite-sample rule variable length" % REPS)
for n_cal in (20, 26, 66, 200):
    m = n_cal - int(n_cal * 0.3)
    print(f"n = {n_cal} calibration successes, conformal part m = {m}")
    for alpha in (0.05, 0.1, 0.2, 0.3):
        cells = [alpha + (1 - 2 * alpha) / (m + 1)]
        for variable, fs in ((False, False), (True, False), (True, True)):
            cells.append(np.mean([false_alarm_rate(n_cal, alpha, variable, fs, s) for s in range(REPS)]))
        print(f"  alpha {alpha:.2f}: " + " | ".join(f"{c:.3f}" for c in cells))
