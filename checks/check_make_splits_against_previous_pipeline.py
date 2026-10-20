"""make_splits on the real local data against the previous pipeline's split files.

    REPO_MAIN=/path/to/project python -I checks/check_make_splits_against_previous_pipeline.py

Needs data/<dataset>/manifest.csv (and splits/ for LIBERO-X) in this repository and the old outputs/ under REPO_MAIN. For every run the
train / val / unseen episode lists must equal the old file's, and calibration + seen must equal the old test part. Not part of the package.
"""

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fp.data.splits import make_splits  # noqa: E402

MAIN = Path(os.environ["REPO_MAIN"])
OLD_ROOT = {
    "libero10_groot": MAIN / "outputs/groot-libero-10",
    "libero10_pi05": MAIN / "outputs/pi0.5-libero-10",
    "liberox_groot": MAIN / "outputs/groot-libero-x/levels-baseline/splits",
}


def old_path(dataset: str, name: str) -> Path:
    """The previous pipeline's file for a run of this package."""
    root = OLD_ROOT[dataset]
    if name.startswith("paper_seed"):
        return root / f"baseline_split_65_15_20_seed{name.removeprefix('paper_seed')}.json"
    if name.startswith("skill_"):
        return root / f"{name}_seed0.json"
    return root / f"{name}.json"  # task_seed<k>, signature_seed<k>


for dataset in OLD_ROOT:
    splits = make_splits(dataset)
    different = []
    for s in splits:
        old = json.load(open(old_path(dataset, s.name)))["episodes"]
        same = (s.train == sorted(old["train"]) and s.val == sorted(old["val"]) and s.unseen == sorted(old["unseen"])
                and sorted(s.calibration + s.seen) == sorted(old["test"]))
        if not same:
            different.append(s.name)
    print(f"{dataset}: {len(splits) - len(different)} of {len(splits)} runs equal the previous split files" + (f"; different: {different}" if different else ""))
