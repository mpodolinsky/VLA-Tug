# data

Local data, ignored by git. One folder per dataset (`libero10_groot`, `libero10_pi05`, `liberox_groot`):

```
<dataset>/
  manifest.csv   one row per rollout, as published with the Hub dataset named in fp/config.py
  splits/        LIBERO-X only: the held-out task lists (leave_task_out, leave_signature_out, leave_skill_out/<skill>/test.txt)
  features.npz   pooled features: one float32 array (steps, 6144) per episode, keyed by the manifest `dir`,
                 columns [base image | wrist image | language] of 2048 each
  targets.npz    targets of the auxiliary heads: the failure mode of each failed episode, the timeline phrase of every
                 policy step, and CLIP ViT-B/32 text embeddings of the phrases (format in fp/data/targets.py)
  actions.npz    the action chunks the policy sampled at every call: one float32 array (steps, 10, H, 7) per episode, keyed by
                 the manifest `dir` (H = 16 for GR00T, 10 for pi0.5; 7 = position, rotation, gripper); sample 0 is the executed chunk.
                 The input of the action-based baselines (STAC, ActProbe, FailureSpot); format in fp/data/actions.py
```

`python -m cli.download <dataset>` fetches `manifest.csv`, `splits/` (unchanged), the pooled features and the targets, which the Hub datasets hold as `pooled/features.npz` and `pooled/targets.npz`. `features.npz` can also be produced from the raw rollouts of the Hub datasets:

```
python -m cli.prepare <dataset> <folder holding the raw rollouts, one <dir>/rollout.npz per episode>
```

Each modality is averaged over its tokens (the language tokens over the unpadded ones only); see `fp/data/pooling.py`.

`actions.npz` is extracted from the same raw rollouts (only the one array is read from each file, which takes seconds); every episode is checked on the way (sample 0 equals the executed chunk, the replan step and the number of policy calls match, all values finite):

```
python -m cli.prepare_actions <dataset> <folder holding the raw rollouts, one <dir>/rollout.npz per episode>
```

`targets.npz` is built from the `labels.json` / `labels.npz` of each episode (same folder layout as above; needs `pip install -e ".[targets]"` for the CLIP text encoder):

```
python -m cli.targets <dataset> <folder holding the labels, one <dir>/labels.json per episode>
```
