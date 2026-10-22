# What the methods cost

Time and memory of our probes and the baselines, measured on one machine, to compare methods on cost as well as accuracy. The numbers are read from the `history.json` of the runs by `python -m cli.report` ("What the runs cost" section), not typed in; this file explains them and records the comparison made on 2026-10-21.

**Machine:** NVIDIA GeForce RTX 4090 (24 GB), 32 CPU threads, 31 GB RAM, torch 2.7.1 (CUDA 12.6), Python 3.11. Each history also records its hardware.

## What is measured, and what it includes

| field | meaning |
|---|---|
| `fit_seconds` | training (probes, SAFE models) or fitting the stored statistics (distance baselines). A k-NN has no fit: it stores the training steps. |
| `score_seconds`, `score_ms_per_step` | one forward pass over **every episode of the dataset** (2,180 episodes, 235,130 steps on LIBERO-X), per step in milliseconds |
| `state_mb` | what a fitted scorer must keep to score a new episode: the weights (probes, SAFE), the stored training steps (k-NN), the statistics (Mahalanobis, PCA-KMeans) |
| `peak_gpu_gb`, `peak_rss_gb` | peaks of the process while the run was made. **They include the features the process keeps loaded for the dataset's other runs** (5.8 GB for LIBERO-X, on the GPU for the probes, in RAM for the baselines), so they overstate one run; subtract that to read the method's own footprint. |

Not included: loading the data from disk (5 to 7 s for LIBERO-X), the calibration and evaluation (about 45 s for all 1,800 runs). Several workers shared the GPU and CPU during the sweeps, which lengthens times; the benchmark below was made with the machine otherwise idle.

## Comparison on the largest LIBERO-X split

`skill_toggle_appliance_off`: 1,358 training episodes, 2,180 scored. One run each, no other job on the machine (the SAFE rows were timed while the PCA-KMeans sweep ran on the CPU, which can only lengthen them).

| method (configuration) | fit | score all episodes | ms / step | state | peak GPU | peak RAM |
|---|--:|--:|--:|--:|--:|--:|
| probe `lstm_plain` (onset_pre) | 5.7 s (17 epochs) | 0.2 s | 0.0008 | 26.4 MB | 6.9 GB | 6.5 GB |
| probe `lstm_mha_modes_clip` | 7.3 s (19 epochs) | 0.2 s | 0.0008 | 6.9 MB | 6.3 GB | 6.6 GB |
| probe `mlp_plain` | 2.4 s (18 epochs) | 0.1 s | 0.0004 | 13.3 MB | 6.2 GB | 6.6 GB |
| SAFE-LSTM, lr 3e-4 | 31 s (69 epochs) | 1 s | 0.004 | 26.2 MB | 8.4 GB | 6.8 GB |
| SAFE-LSTM, lr 3e-5 (slowest) | 50 s (127 epochs) | 1 s | 0.004 | 26.2 MB | 8.4 GB | 6.8 GB |
| SAFE-MLP, lr 1e-3 | 18 s (52 epochs) | 1 s | 0.004 | 6.3 MB | 6.6 GB | 6.9 GB |
| SAFE-MLP, lr 1e-5 (slowest) | 135 s (1,000 epochs, the cap) | 1 s | 0.004 | 6.3 MB | 6.6 GB | 6.9 GB |
| Mahalanobis | 19 s | 4 s | 0.015 | 302 MB | 10.6 GB | 6.9 GB |
| PCA-KMeans (d128, c64) | 59 s | 24 s | 0.10 | 3 MB | 7.2 GB | 12.2 GB |
| Euclidean k-NN (k = 10) | 5 s (stores the data) | 60 s | 0.25 | **3.6 GB** | 7.2 GB | 6.7 GB |

The SAFE rows are single timing runs (the SAFE sweep records its own in its histories); the probe rows are one benchmark run each with the fields the sweeps now record.

**Reading it**
- A trained probe scores a step about **300 times faster** than the k-NN baseline (0.0008 against 0.25 ms) and keeps **100 to 500 times less** (7 to 26 MB against 3.6 GB), because a k-NN has to hold every training step and compare each new step with all of them. The cost of the k-NN grows with the training set; the probe's does not.
- Mahalanobis keeps 302 MB: two means and two 6,144 x 6,144 pseudo-inverse covariance matrices, independent of the amount of data. PCA-KMeans keeps almost nothing (centroids and one basis, 3 MB) but needs 12 GB of RAM *while fitting* (copies of the training steps for the PCA).
- Training a probe takes seconds (early stopping ends it at 17 to 19 epochs); SAFE needs tens of seconds to minutes, and its slowest-learning configurations run for the full 1,000 epochs.
- How to read the memory columns: for the probes the 5.8 GB of standardised features sit on the GPU, so their 6.2 to 6.9 GB peaks are that plus 0.4 to 1.1 GB for the probe itself. The SAFE models move only the training and validation episodes (about 4.6 GB) to the GPU, so SAFE-LSTM's 8.4 GB is that plus about 3.8 GB for the LSTM's activations (batches of 512 rollouts run in chunks of 128). The distance baselines keep the features in RAM (about 6.7 GB peak RSS); Mahalanobis has the highest GPU peak (10.6 GB), set by the pseudo-inverses of the covariances.

## Totals over the full sweeps

Sum of the run times of every run of a family (the runs of a sweep overlapped: 2 to 3 workers shared the machine), LIBERO-X, 14 splits:

| | runs | total hours |
|---|--:|--:|
| our probes (first sweep, fit + score recorded together) | 896 | 6.1 |
| Euclidean (6 configurations per split) | 84 | 1.3 |
| Mahalanobis (2 per split) | 28 | 0.15 |
| PCA-KMeans (18 per split) | 252 | 3.9 |

The PCA-KMeans total contains one outlier of 576 s for a single scoring (typically 5 to 25 s): the CPU was oversubscribed by an unrelated test run at that moment. The probes' first sweep recorded only a total time per run, so its fit/score split and its memory are n/a in the report; the benchmark above supplies them for three probes.

## Reproducing

```
python -m cli.report            # the cost table is its last section; also results/report.txt
python -m cli.train --datasets liberox_groot --variants onset_pre --splits skill_toggle_appliance_off --probes lstm_plain lstm_mha_modes_clip mlp_plain --out results/_bench
python -m cli.run_baselines --families euclidean --datasets liberox_groot --splits skill_toggle_appliance_off --methods euclidean__k10_cumsum --out results/_bench
```
Run one at a time on an idle machine for clean times. The sweeps' own histories carry the same fields for every run.
