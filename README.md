# vla-probe

Early failure detection for vision-language-action (VLA) policies from frozen backbone embeddings. This repository holds the calibration methods (SAFE functional band, max-conformal, Learn-then-Test), the evaluation metric (area under TPR early vs FPR up to FPR = 40%, where TPR early counts an alarm at or before the failure onset), the dataset splits, and the embedding-distance baselines of SAFE.

## Data

The annotated per-token embeddings used in this work are published on the Hugging Face Hub:

| setting | policy | benchmark | dataset |
|---|---|---|---|
| `libero10_groot` | GR00T-N1.7 | LIBERO-10 with occlusions | [`podolinsky/groot-libero-10-features-v3`](https://huggingface.co/datasets/podolinsky/groot-libero-10-features-v3) |
| `libero10_pi05` | pi0.5 | LIBERO-10 with occlusions | [`podolinsky/pi0.5-libero-10-features-v3`](https://huggingface.co/datasets/podolinsky/pi0.5-libero-10-features-v3) |
| `liberox_groot` | GR00T-N1.7 | LIBERO-X, levels 1-3 | [`podolinsky/gr00t-libero-x-features-v3`](https://huggingface.co/datasets/podolinsky/gr00t-libero-x-features-v3) |

Per episode, a dataset holds the raw per-token backbone features and action chunks (`rollout.npz`), metadata (`rollout.json`), failure labels (`labels.json`, `labels.npz`) and a top-level `manifest.csv`.

## Installation

Python 3.11.

```
pip install -e ".[dev]"
```

## Getting the data

Each dataset also holds the pooled features in `pooled/features.npz` (one float32 array of shape (steps, 6144) per episode, the mean over the base-image, wrist-image and language tokens) and the probe targets in `pooled/targets.npz` (failure modes, timeline phrases and their CLIP embeddings; see `pooled/README.md` in each dataset). This fetches them, the manifest and (for LIBERO-X) the split files into `data/<dataset>/`:

```
python -m cli.download libero10_groot      # or libero10_pi05, liberox_groot
```

Both files can also be recomputed from the raw rollouts and labels with `python -m cli.prepare` and `python -m cli.targets`; see [`data/README.md`](data/README.md).

The real-robot dataset `real8_groot` (GR00T on eight tasks) is not published. `python -m cli.import_real real8_groot <collection folder>` builds its `data/` folder from the collected rollouts and labels (see `data/README.md`).

## Tests

```
pytest
```

The tests use synthetic data and need no download. Where the code is a port of SAFE (Gu et al., 2025), the tests also compare it with SAFE's own code if a local SAFE checkout is available:

```
SAFE_ROOT=/path/to/SAFE SAFE_CONFORMAL_DIR=/path/to/SAFE/failure_prob/utils/conformal pytest
```

On a machine with unrelated pytest plugins installed system-wide, run `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest`.

## Documentation

- [`docs/CALIBRATION.md`](docs/CALIBRATION.md): the three calibrators (SAFE band, max-conformal, Learn-then-Test), their guarantees, and which method uses which.
- [`docs/SPLITS.md`](docs/SPLITS.md): how episodes are split into train, validation, calibration, held-out and unseen sets.
- [`docs/DISTANCE_BASELINES.md`](docs/DISTANCE_BASELINES.md): the embedding-distance baselines taken from SAFE and how their configuration is selected.
- [`docs/ACTION_BASELINES.md`](docs/ACTION_BASELINES.md): STAC (eight configurations), ActProbe and FailureSpot, which read the sampled action chunks; decisions, deviations and what the checkpoint selection does.
- [`docs/FIGURES.md`](docs/FIGURES.md): the main table and the curve figures (`python -m cli.figures`): what each shows, how the methods in them are chosen, how the costs are measured; also the ablation table (heads and training strategy of our probe) and the table of Learn-then-Test against max-conformal (`python -m cli.ltt_table`).

## Licences

Code: MIT (`LICENSE`). Documentation, published features and results: CC BY 4.0 (`LICENSE-DATA-CC-BY-4.0.txt`). Code ported from other projects keeps its own licence, listed in `NOTICE`.
