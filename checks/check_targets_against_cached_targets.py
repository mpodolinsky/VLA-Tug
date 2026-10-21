"""Compare data/<dataset>/targets.npz with the phrase and mode caches of the earlier experiments.

    REPO_MAIN=/path/to/project python -I checks/check_targets_against_cached_targets.py

For every dataset: the phrase list, the CLIP embeddings (allclose; computed again on this machine), the phrase ids of every episode and the
failure mode of every failed episode must equal the old phrases.npz / modes.json. Not part of the package.
"""

import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fp.data.targets import MODE_NAMES, load_targets  # noqa: E402

OUT = Path(os.environ["REPO_MAIN"]) / "outputs"
OLD = {
    "libero10_groot": OUT / "groot-libero-10/ours-libero10/cache",
    "libero10_pi05": OUT / "pi0.5-libero-10/ours-libero10/cache",
    "liberox_groot": OUT / "groot-libero-x/ours-levels/cache",
}

for dataset, cache in OLD.items():
    new = load_targets(dataset)
    old = np.load(cache / "phrases.npz")
    old_modes = json.load(open(cache / "modes.json"))
    same_phrases = new.phrases == [str(p) for p in old["phrases"]]
    max_dev = float(np.abs(new.embeddings - old["embeddings"]).max()) if new.embeddings.shape == old["embeddings"].shape else float("nan")
    ids_equal = sum(np.array_equal(new.phrase_ids[k], old[f"ids/{k}"]) for k in new.phrase_ids)
    modes_equal = sum(MODE_NAMES[new.modes[k]] == old_modes[k] for k in new.modes if k in old_modes)
    print(f"{dataset}: phrases equal {same_phrases} ({len(new.phrases)}); embeddings max deviation {max_dev:.2e}; "
          f"phrase ids equal in {ids_equal} of {len(new.phrase_ids)} episodes; modes equal in {modes_equal} of {len(new.modes)} failures "
          f"(old file has {len(old_modes)} modes)")
