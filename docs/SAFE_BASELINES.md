# SAFE-MLP and SAFE-LSTM (`fp/models/safe.py`, `fp/train/safe.py`, `fp/baselines/safe_grid.py`)

The two learned failure predictors of SAFE (Gu et al., NeurIPS 2025, arXiv 2506.09937), written from SAFE's code in its default configuration: `IndepModel` (`failure_prob/model/indep.py`) is SAFE-MLP and `LstmModel` (`failure_prob/model/lstm.py`) is SAFE-LSTM. The losses are those of `forward_compute_loss`, `aggregate_monitor_loss`, `compute_regularization_loss` and `RolloutDataset` in the same repository. Every SAFE-derived file is listed in `NOTICE`; SAFE states no licence, which has to be settled before anything is published.

## The models and the loss

| | SAFE-MLP | SAFE-LSTM |
|---|---|---|
| network | per-step MLP 6144 → 256 → 1 with ReLU and a sigmoid | one-layer LSTM (256), linear layer, sigmoid |
| score at step $t$ | running sum of the step probabilities (rises over the episode) | the step probability |
| loss per episode, mean over its steps | success: $\mathrm{relu}(S_t)$; failure: $-S_t$ | binary cross-entropy of $p_t$ against the label (failure = 1) |

Episode losses are weighted by class, $w_c = 1/((n_c+1)/n)$ with the counts of the training episodes, summed over the batch and divided by the batch size; an L2 penalty $\lambda_{reg}\sum w^2$ over the non-bias parameters is added. Labels are SAFE's: every step of a failed rollout is positive (our `onset_traj`), nothing about the onset. Optimiser: Adam, batch 512 whole rollouts, no learning-rate decay, no gradient clipping. A batch of 512 rollouts through the LSTM on 6,144-d inputs needs well over 14 GB of GPU memory, so each batch is run in chunks of 128 whose gradients are accumulated, every chunk weighted by its share of the batch: SAFE's monitor loss is a mean over the episodes of the batch plus the constant penalty, so the update is the one of the whole batch of 512 (a test compares the weights after training with and without chunks, including a partial last chunk). It saves memory and changes nothing else; the history records `micro_batch`. Only these default branches are implemented; SAFE's other options (more layers, a history window, a cumulative sum for the LSTM, the hard-negative loss, time weights, the threshold loss) are off in its default configuration and are not part of this baseline.

Features: the pooled per-modality vector (6,144-d), **raw** (SAFE's `normalize_hidden_states` is off by default); the same input as every other model here. SAFE also sweeps which embedding and token it uses; ours is fixed.

## Deviation from SAFE: early stopping (decided)

SAFE trains for a fixed **1,000 epochs** and keeps the last epoch; it uses no validation set to stop or select. **Here training stops early**: after every epoch the model is scored on the validation episodes with SAFE's own class-weighted data loss, training stops after **50 epochs without a new best** (patience 50, 1,000 epochs remain the cap), and the weights of the best epoch are kept. The criterion leaves out the L2 penalty: that term is a function of the weights alone (about 6.6 million of them in the LSTM, so at $\lambda_{reg}=0.1$ it starts in the hundreds against a cross-entropy near 0.7) and would pick the epoch by weight shrinkage; it is logged as a separate learning curve. Everything else (loss, weights, optimiser, batch, learning rates, penalty) is SAFE's. Each run's `history.json` records the rule (`early_stopping`), the epochs run, the best epoch, whether it stopped early and the learning curves.

## Configurations: run all, report the best (decided)

SAFE's launch script sweeps the learning rate and $\lambda_{reg}$; ours runs the same 12 configurations per model on all 20 splits:

| model | learning rate | $\lambda_{reg}$ | configurations |
|---|---|---|--:|
| SAFE-MLP | 1e-5, 1e-4, 3e-4, 1e-3 | 1e-3, 1e-2, 1e-1 | 12 |
| SAFE-LSTM | 3e-5, 1e-4, 3e-4, 1e-3 | 1e-3, 1e-2, 1e-1 | 12 |

A method is named `<family>__lr<lr>_reg<lambda_reg>` (e.g. `safe_lstm__lr3e-4_reg1e-2`); the order of `method_names` (learning rate first, then $\lambda_{reg}$, both ascending) is the order ties are broken in. The best configuration of a family is chosen per group of splits on the mean **validation** area, exactly as for the distance baselines and for our probes (`select_config.choose_best`), never on the unseen tasks. Calibrator: the SAFE band (the protocol's, as SAFE evaluates its own models).

## The extra families: SAFE's loss on rollouts cut at the onset (decided, not SAFE as published)

`safe_mlp_pre` and `safe_lstm_pre` keep SAFE's loss, models, weights and grid and change only the data: failed training and validation episodes are cut at the failure onset (our `onset_pre`), so every step kept is positive. This is **not** SAFE as published; it separates what the labels contribute from what the architecture contributes, and it is a separate family in every table so that it never enters the selection of the real baseline. Scores are still computed over the full rollout. Calibrator: the SAFE band, like the other SAFE models.

## What a run saves and costs

`results/<dataset>/<split>/<method>/` holds `scores.npz`, `weights.pt` (6.3 MB for the MLP, 26 MB for the LSTM) and `history.json` (configuration, seed, split hash, code version, `features: raw`, the early-stopping record and curves, seconds to fit and score, peak GPU and RAM, size of the fitted model, hardware). The seed of a run is its split's seed. Cost: see "What the runs cost" in the report and `COMPUTE.md`.

## Running them

```
python -m cli.run_baselines --families safe_mlp safe_lstm --dry-run     # units and how many runs exist
python -m cli.run_baselines --families safe_mlp safe_lstm               # then safe_mlp_pre safe_lstm_pre
python -m cli.evaluate && python -m cli.report
```
Resumable; a configuration that fails is logged and skipped.

## Verification

`tests/test_safe_models.py`: the networks are causal and ignore padding; the MLP score is the running sum of positive step probabilities; the loss terms, class weights, monitor loss and L2 penalty by hand; the grid, names and protocol; the scorer separates held-out failures from successes, stops early on the data loss and keeps the best epoch, records the deviation, reproduces its scores after saving and loading, and is deterministic for a seed; the label variants cut or keep failed episodes as described; a run through the driver, evaluated with the band. With a SAFE checkout (`SAFE_ROOT`), SAFE's own `base.py`, `indep.py` and `lstm.py` are executed and their outputs, monitor losses and regularisation losses equal ours for identical weights, for padded batches with both outcomes; and the text of SAFE's class-weight formula is checked.
