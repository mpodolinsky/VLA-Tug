# Splits and what a seed changes

`fp/data/splits.py` builds, for every run: **train / val / test** (65 / 15 / 20 %) from the episodes of the *seen* tasks, stratified by outcome and condition (scene variant on LIBERO-10, level on LIBERO-X); the test part is halved into **calibration** and **seen** (held-out seen half); and **unseen** holds every episode of the held-out tasks. `make_splits(dataset)` returns every run of a dataset from `data/<dataset>/manifest.csv` (and, for LIBERO-X, `data/<dataset>/splits/`). It reproduces the split files of the earlier experiments exactly: 20 of 20 runs (3 for each LIBERO-10 policy, 14 for LIBERO-X).

## The runs of each dataset

| dataset | runs | what is held out |
|---|---|---|
| LIBERO-10 (GR00T, pi0.5) | paper seeds 0, 1, 2 | 3 tasks, drawn from the seed (`Random(seed + 1000)`): different tasks per seed |
| LIBERO-X task split | seeds 0, 1, 2 | the same 43 tasks (430 episodes) in every seed |
| LIBERO-X signature split | seeds 0, 1, 2 | the same 56 tasks, 16 signatures (560 episodes) in every seed |
| LIBERO-X skill split | 8 folds (seed 0 each) | all tasks of one skill; a different skill per fold (e.g. stack_objects: 17 tasks, 170 episodes) |

## What the seed changes on LIBERO-X task and signature splits
Measured on the split files of the earlier experiments, for seeds 0-4: across seeds the **unseen tasks and the unseen episodes are identical**. Only the shuffle of the *seen* episodes into train / val / test changes (about 65% of the seed-0 training episodes are also training episodes in another seed, which is what chance gives), and with it the calibration and seen halves. For learned models the seed also changes the initialisation; for the SAFE band it changes the shuffle of the calibration successes; for PCA / KMeans, the seed.

Consequences:
- the 3 task runs (and the 3 signature runs) are evaluated on **the same unseen episodes**: they measure the variance from training data, calibration data and initialisation, not variation over different unseen tasks, so their spread must not be read as spread over unseen tasks;
- the validation sets of the seeds are different random draws from the same seen-task pool, so they are not independent;
- only the 8 skill folds and the 3 LIBERO-10 seeds hold out different tasks.
