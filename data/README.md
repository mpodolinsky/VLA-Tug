# data

Per dataset (`libero10_groot`, `libero10_pi05`, `liberox_groot`):

```
<dataset>/
  manifest.csv     one row per rollout (outcome, task, scene, onset, length)
  labels.json      failure onset labels
  splits/          split files
  features.npz     per-modality pooled features, (n_steps, 6144) per episode  [large, not committed]
  stac/            sampled action chunks for STAC (K=10)                       [large, not committed]
```

Everything here is produced by the `prepare` step from the raw rollouts. Large arrays are git-ignored.
