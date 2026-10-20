"""Pool a sample of real episodes from the raw rollouts and compare with the per-modality pooled caches of the earlier experiments.

    REPO_MAIN=/path/to/project python -I checks/check_pooling_against_cached_features.py [--per-dataset 8] [--full-scan]

For every dataset, episodes spread over the manifest (successes and failures, short and long) are pooled with fp.data.pooling.pool_rollout and
must equal the cached array bit for bit. With --full-scan every array of every cache is also checked for non-finite values.
Not part of the package: it reads the raw rollouts and the old caches under REPO_MAIN.
"""

import argparse
import csv
import os
import sys
import zipfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fp.data.pooling import pool_rollout  # noqa: E402

MAIN = Path(os.environ["REPO_MAIN"]) / "outputs"
DATASETS = {
    "libero10_groot": (MAIN / "groot-libero-10/collect-stac-k10", MAIN / "groot-libero-10/ours-libero10/cache/modality/features_pooled.npz"),
    "libero10_pi05": (MAIN / "pi0.5-libero-10/collect-stac-k10", MAIN / "pi0.5-libero-10/ours-libero10/cache/modality/features_pooled.npz"),
    "liberox_groot": (MAIN / "groot-libero-x/collect-stac-k10-level123", MAIN / "groot-libero-x/levels-baseline-modality/cache/features_pooled.npz"),
}

p = argparse.ArgumentParser()
p.add_argument("--per-dataset", type=int, default=8)
p.add_argument("--full-scan", action="store_true")
args = p.parse_args()

for name, (raw, cache) in DATASETS.items():
    rows = list(csv.DictReader(open(raw / "manifest.csv")))
    rows.sort(key=lambda r: int(r["n_policy"]))
    # evenly spaced by length: includes the shortest and the longest, successes and failures
    picks = [rows[round(i * (len(rows) - 1) / (args.per_dataset - 1))] for i in range(args.per_dataset)]
    exact = 0
    with np.load(cache) as z:
        for r in picks:
            fresh = pool_rollout(raw / r["dir"] / "rollout.npz")
            exact += np.array_equal(fresh, z[r["dir"]])
        print(f"{name}: {exact} of {len(picks)} sampled episodes pooled from the raw rollouts equal the cache bit for bit "
              f"({sum(r['success'] == 'True' for r in picks)} successes, {sum(r['success'] != 'True' for r in picks)} failures, "
              f"{picks[0]['n_policy']} to {picks[-1]['n_policy']} steps)")
        if args.full_scan:
            bad = [k for k in z.files if not np.isfinite(z[k]).all()]
            print(f"   full scan of the cache ({len(z.files)} arrays): {len(bad)} with non-finite values")
