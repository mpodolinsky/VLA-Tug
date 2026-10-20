# data

Local data, ignored by git. One folder per dataset (`libero10_groot`, `libero10_pi05`, `liberox_groot`):

```
<dataset>/
  manifest.csv   one row per rollout, as published with the Hub dataset named in fp/config.py
  splits/        LIBERO-X only: the held-out task lists (leave_task_out, leave_signature_out, leave_skill_out/<skill>/test.txt)
  features.npz   pooled features: one float32 array (steps, 6144) per episode, keyed by the manifest `dir`,
                 columns [base image | wrist image | language] of 2048 each
```

`python -m cli.download <dataset>` fetches all of it from the Hub (`manifest.csv` and `splits/` unchanged, and the pooled features, which the Hub datasets hold as `pooled/features.npz`). `features.npz` can also be produced from the raw rollouts of the Hub datasets:

```
python -m cli.prepare <dataset> <folder holding the raw rollouts, one <dir>/rollout.npz per episode>
```

Each modality is averaged over its tokens (the language tokens over the unpadded ones only); see `fp/data/pooling.py`.
