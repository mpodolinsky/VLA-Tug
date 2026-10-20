"""Paths and loaders for the previous pipeline's files (outputs/ of the main checkout), shared by the checks.

Importing this module also puts the package on sys.path. Nothing in fp/ imports it.
"""

import csv
import os
import sys
from pathlib import Path

import numpy as np

PKG = Path(__file__).resolve().parents[1]  # failure_probes/
WT = PKG.parent  # repo root of this checkout
# Checkout that holds the old outputs/ (the main repo when run from a worktree); override with REPO_MAIN.
MAIN = Path(os.environ["REPO_MAIN"]) if "REPO_MAIN" in os.environ else (WT.parents[2] if WT.parent.name == "worktrees" else WT)
for p in (PKG, Path(__file__).resolve().parent):
    sys.path.insert(0, str(p))

from fp.data.manifest import Episode  # noqa: E402

X = MAIN / "outputs/groot-libero-x"
SPLITS_DIR = X / "finetune_set_v2/splits"
OLD_SPLITS = X / "levels-baseline/splits"
LIBERO10 = {"pi05": MAIN / "outputs/pi0.5-libero-10", "groot": MAIN / "outputs/groot-libero-10"}
WINDOW = {"pi05": 10, "groot": 6, "liberox": 6}
ALPHAS40 = np.linspace(0.005, 0.5, 40)  # the fine grid of the previous key plot

# The four LIBERO-X models of the key plot: scores_seed<s>.npz live in <pattern with {lv}>.
LIBEROX_MODELS = {
    "SAFE-LSTM": X / "levels-baseline-modality" / "{lv}" / "safe_lstm",
    "LSTM onset": X / "ours-levels-pm" / "{lv}" / "lstm_modes_clip__onset",
    "LSTM onset_pre": X / "ours-levels-modality" / "{lv}" / "lstm_modes_clip__onset_pre",
    "LSTM onset_traj": X / "ours-levels-modality" / "{lv}" / "lstm_modes_clip__onset_traj",
}


def load_episodes(path: Path) -> tuple[list[Episode], list[dict]]:
    """Episodes from an old manifest; onset = frame // replan_steps, condition = level (LIBERO-X) or scene variant (LIBERO-10)."""
    rows = list(csv.DictReader(open(path)))
    eps = []
    for r in rows:
        ok = r["success"] == "True"
        onset = None if ok else int(float(r["vlm_failure_onset_frame"])) // int(r["replan_steps"])
        eps.append(Episode(r["dir"], ok, r["task"], int(r["n_policy"]), onset, r.get("level", r.get("scene_variant", ""))))
    return eps, rows


def liberox_episodes() -> list[Episode]:
    return load_episodes(X / "collect-stac-k10-level123/manifest.csv")[0]


def unseen_tasks(name: str) -> set[str]:
    rel = {"task": "leave_task_out", "signature": "leave_signature_out"}.get(name) or f"leave_skill_out/{name[len('skill_'):]}"
    return {line.strip()[: -len(".bddl")] for line in open(SPLITS_DIR / rel / "test.txt") if line.strip()}


def liberox_runs() -> dict[str, list[tuple[str, int]]]:
    """The 14 runs of the key plot: task and signature with seeds 0-2, each of the 8 skill folds with seed 0."""
    skills = sorted(p.name for p in (SPLITS_DIR / "leave_skill_out").iterdir() if p.is_dir())
    return {"task": [("task", s) for s in range(3)], "signature": [("signature", s) for s in range(3)],
            "skill": [(f"skill_{k}", 0) for k in skills]}
