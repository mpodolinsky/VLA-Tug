# Embedding-distance baselines (`fp/baselines/distance.py`)

Mahalanobis, Euclidean (k-NN) and PCA-KMeans, copied from SAFE's `EmbedModel` (`baselines/SAFE/failure_prob/model/embed.py`; SAFE: Gu et al., NeurIPS 2025, arXiv 2506.09937). The three distance functions are copied verbatim, `DistanceScorer.fit` is SAFE's `train_epoch`, and `_step_scores` is the distance part of SAFE's `forward`. These replace the earlier `embed-dist` versions (z-score + PCA(128) Mahalanobis, Euclidean to the class mean), which were not SAFE's.

## What each one computes

Every step of a training episode labelled success is a success step, every step of a failed training episode a failure step. The score of a step with feature $x$ is

$$s(x)=D_{\rm succ}(x)-D_{\rm fail}(x),\qquad S(t)=\sum_{u\le t}s(x_u)\ \ (\text{if }\texttt{cumsum}).$$

| | $D_c(x)$ | fitted from |
|---|---|---|
| Mahalanobis | $\sqrt{(x-\mu_c)^\top(\Sigma_c+10^{-4}I)^{+}(x-\mu_c)}$ | mean and covariance of the raw features of class $c$ |
| Euclidean | $\frac1k\sum_{y\in\mathrm{kNN}_k(x;\mathcal X_c)}\lVert x-y\rVert_2$ | all stored steps of class $c$ |
| PCA-KMeans | $\min_j\lVert Px-m_{c,j}\rVert_2$ | PCA to `pca_dim`, then `n_clusters` k-means centroids per class |

## What differs from SAFE's code
- No `BaseModel`, `Config` or `DataLoader`: the scorer takes per-episode arrays and the success flags; episodes are scored one at a time instead of in a zero-padded batch (same scores at the valid steps; checked).
- `seed` is passed to PCA and KMeans, which SAFE leaves unseeded.
- Scoring before fitting is an error (SAFE warns and returns zeros).
- SAFE's cosine variant is left out.

## SAFE behaviours kept as they are
- Mahalanobis on the raw features, in float32, with a pseudo-inverse; for our 6,144-d per-modality features that is a 6,144 x 6,144 covariance per class.
- Euclidean is a k-NN distance to every stored step, not a distance to a class mean.
- PCA-KMeans projects $x$ with the **success** PCA for both distances, although the failure centroids are in the failure PCA's coordinates (`tests/test_distance_baselines.py` reproduces and pins this).
- Only success and failure training episodes enter; the validation keys are ignored.

## Configurations: run all, report the best (decided and implemented; nothing has been run yet)
Defaults of the scorer are SAFE's: `topk` 10, `pca_dim` 64, `n_clusters` 16, `cumsum` True, `use_success_only` False. The plan is to run SAFE's whole sweep (`baselines/distance_grid.py`, 26 configurations with `use_success_only=False`):

| baseline | axes | configurations |
|---|---|--:|
| Mahalanobis | `cumsum` in {False, True} | 2 |
| Euclidean (k-NN) | `topk` in {1, 5, 10} x `cumsum` | 6 |
| PCA-KMeans | `pca_dim` in {32, 64, 128} x `n_clusters` in {16, 32, 64} x `cumsum` | 18 |

(SAFE also sweeps the feature and token index; our feature is fixed.) Each configuration is named `<baseline>__<tag>` (e.g. `euclidean__k5_cumsum`) and saves its own weights and numbers, so all 26 are on disk and the full table can always be shown.

**Selection rule (decided): the best configuration of each baseline is the one with the highest mean validation area, never looking at the unseen tasks** (`fp/eval/select_config.py`). Per run (split): fit the configuration on the training episodes, calibrate it with the SAFE band on the calibration successes, sweep alpha on the **validation** episodes and take the area under TPR early vs FPR to 40%. The best configuration is the one with the highest mean validation area over the runs of its group; ties go to the earlier configuration in grid order. **Groups (decided): LIBERO-10 pools all its runs into one choice per baseline; LIBERO-X chooses separately for each held-out set**, i.e. over the 3 seeds of the task split, over the 3 seeds of the signature split and over the 8 folds of the skill split, so each baseline has three configurations on LIBERO-X. "Best over the seeds" is read as the highest mean over the group's runs, not the single best-scoring run (which would be picking the split that happens to suit a configuration). The module only ever asks a scorer to score the calibration and validation episodes of a split (a test enforces it, including that a scorer which refuses unseen-task episodes gives an identical selection). Caveats: on LIBERO-10 the validation set holds only a handful of failures, so the choice is noisy; selection on seen-task validation need not pick what is best under task shift; and the choice is per group of runs, not per run (a per-run choice would be noisier and harder to report). The validation sets of the task and signature seeds are different random draws from the same seen-task pool (see `SPLITS.md`), so the three seeds are not three independent validation sets.

Cost to keep in mind: Mahalanobis needs a 6,144 x 6,144 covariance and pseudo-inverse per class, PCA-KMeans fits two PCAs and two k-means per configuration, and the configurations of one baseline differ in only a few parameters (the Mahalanobis `cumsum` pair shares one fit, the three Euclidean `topk` share one neighbour search with k = 10), which a later optimisation could exploit.

## Calibrator
The SAFE band (decided), as SAFE itself evaluates them. The score can be negative, so the maximum over steps is not the final value, unlike STAC.

## Verification
`tests/test_distance_baselines.py`: unit tests of the copied functions, behaviour of the scorer, and an exact comparison with SAFE's own `EmbedModel` (SAFE's `embed.py` is executed with stand-ins for its config and base class) for the three distances and the four `use_success_only` x `cumsum` settings.
