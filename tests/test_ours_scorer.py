"""OursScorer end to end on a small synthetic dataset (tiny widths, CPU)."""

import numpy as np
import pytest
import torch
from sklearn.metrics import roc_auc_score

from fp.data.features import fit_standardiser
from fp.data.manifest import Episode
from fp.data.targets import MODE_NAMES, Targets
from fp.models.ours import ProbeConfig
from fp.train.loop import TrainConfig
from fp.train.ours import OursScorer

D, WINDOW, N_PHRASES = 8, 3, 20
TRAIN_CFG = TrainConfig(lr=5e-3, batch=8, max_epochs=40, patience=40, seed=0)


def cfg(**kw):
    return ProbeConfig(**{"d_modality": D, "hidden": 16, "proj": 8, "clip_dim": 6, **kw})


@pytest.fixture(scope="module")
def data():
    """40 episodes: failures have a shifted mean in the first columns, a mode, and an onset; every step has a phrase."""
    rng = np.random.default_rng(0)
    episodes, features, ids, modes = {}, {}, {}, {}
    for i in range(40):
        fail = i % 3 == 0
        n = int(rng.integers(14, 22))
        x = rng.normal(size=(n, 3 * D)).astype(np.float32)
        key = f"{'f' if fail else 's'}{i:02d}"
        onset = int(rng.integers(8, n - 2)) if fail else None
        if fail:
            x[:, :4] += 2.0
            modes[key] = int(rng.integers(0, len(MODE_NAMES)))
        episodes[key], features[key] = Episode(key, not fail, f"t{i % 5}", n, onset, "a"), x
        ids[key] = rng.integers(0, N_PHRASES, size=n).astype(np.int32)
    emb = rng.normal(size=(N_PHRASES, 16)).astype(np.float32)
    targets = Targets([f"phrase {j}" for j in range(N_PHRASES)], emb / np.linalg.norm(emb, axis=1, keepdims=True), ids, modes)
    keys = list(episodes)
    return episodes, features, targets, keys[:24], keys[24:32], keys[32:]  # train, val, test


def scorer(data, c=None, variant="onset_pre", **kw):
    episodes, features, targets, *_ = data
    return OursScorer(c or cfg(), variant, episodes, features, targets, WINDOW, TRAIN_CFG, device="cpu", **kw)


@pytest.fixture(scope="module")
def fitted(data):
    s = scorer(data)
    s.fit(data[3], data[4])
    return s


def test_the_name_is_the_architecture_and_the_label_variant(data):
    assert scorer(data, cfg(mha=True, modes=True, clip=True), "onset").name == "lstm_mha_modes_clip__onset"


def test_scores_are_probabilities_for_every_step_of_every_requested_episode(data, fitted):
    out = fitted.score(data[5])
    assert list(out) == data[5]
    for key, p in out.items():
        assert p.shape == (data[0][key].n_steps,) and p.dtype == np.float32 and ((p > 0) & (p < 1)).all()


@pytest.mark.parametrize("c", [cfg(), cfg(trunk="mlp", clip=True), cfg(mha=True, modes=True, clip=True)], ids=lambda c: c.name)
def test_the_probe_learns_to_separate_failures_from_successes_on_held_out_episodes(data, c):
    s = scorer(data, c)
    s.fit(data[3], data[4])
    episodes = data[0]
    out = s.score(data[5])
    y = [0 if episodes[k].success else 1 for k in data[5]]
    assert roc_auc_score(y, [out[k].mean() for k in data[5]]) > 0.9
    assert s.result.val["score"][s.result.best_epoch] < s.result.val["score"][0]


def test_steps_before_the_variants_cut_get_the_same_score_on_the_cut_rollout(data, fitted):
    """The probe is causal: scoring the rollout cut at step 10 gives the first 10 values of scoring it whole."""
    key = data[5][0]
    full = fitted.score([key])[key]
    cut = {**fitted.features, key: fitted.features[key][:10]}
    fitted.features, original = cut, fitted.features
    try:
        assert np.allclose(fitted.score([key])[key], full[:10], atol=1e-6)
    finally:
        fitted.features = original


def test_the_same_seed_gives_identical_scores(data, fitted):
    again = scorer(data)
    again.fit(data[3], data[4])
    a, b = fitted.score(data[5]), again.score(data[5])
    assert all(np.array_equal(a[k], b[k]) for k in a)


def test_early_stopping_follows_the_validation_score_bce_even_with_the_other_heads(data):
    """The selection value is the weighted BCE of the validation items alone: recomputed here from the fitted model's scores."""
    s = scorer(data, cfg(mha=True, modes=True, clip=True))
    s.fit(data[3], data[4])
    train = [s._item(k, s.mean, s.std) for k in data[3]]
    y = torch.cat([it.y for it in train])
    pos_weight = float((1 - y).sum() / y.sum())
    probs, steps, total = s.score(data[4]), 0, 0.0
    for key in data[4]:
        item = s._item(key, s.mean, s.std)
        p = torch.from_numpy(probs[key][: len(item.y)]).double()
        total += float(-(pos_weight * item.y * torch.log(p) + (1 - item.y) * torch.log(1 - p)).sum())
        steps += len(item.y)
    assert s.result.best_epoch == int(np.argmin(s.result.val["score"]))  # chosen by the score term, whatever the other terms did
    assert s.result.val["score"][s.result.best_epoch] == pytest.approx(total / steps, rel=1e-3)


def test_every_head_term_is_reported_for_every_epoch_and_finite(data):
    s = scorer(data, cfg(modes=True, clip=True))
    s.fit(data[3], data[4])
    for history in (s.result.train, s.result.val):
        assert set(history) == {"score", "mode", "clip"}
        assert all(len(v) == len(s.result.val["score"]) and np.isfinite(v).all() for v in history.values())


def test_the_feature_statistics_come_from_the_training_episodes_only(data, fitted):
    mean, std = fit_standardiser([data[1][k] for k in data[3]])
    assert np.array_equal(fitted.mean, mean) and np.array_equal(fitted.std, std)


def test_save_and_load_reproduce_the_scores(data, fitted, tmp_path):
    fitted.save(tmp_path / "w.pt")
    loaded = scorer(data)
    loaded.load(tmp_path / "w.pt")
    a, b = fitted.score(data[5]), loaded.score(data[5])
    assert all(np.array_equal(a[k], b[k]) for k in a)


def test_loading_weights_into_another_architecture_or_label_variant_is_an_error(data, fitted, tmp_path):
    fitted.save(tmp_path / "w.pt")
    with pytest.raises(ValueError, match="this scorer is"):
        scorer(data, cfg(trunk="mlp")).load(tmp_path / "w.pt")
    with pytest.raises(ValueError, match="this scorer is"):
        scorer(data, variant="onset").load(tmp_path / "w.pt")


def test_scoring_before_fitting_is_an_error(data):
    with pytest.raises(RuntimeError):
        scorer(data).score(data[5])


def test_a_failed_training_episode_without_a_mode_is_an_error(data):
    episodes, features, targets, train, val, _ = data
    broken = Targets(targets.phrases, targets.embeddings, targets.phrase_ids, {k: v for k, v in targets.modes.items() if k != train[0]})
    s = OursScorer(cfg(modes=True), "onset", episodes, features, broken, WINDOW, TRAIN_CFG, device="cpu")
    with pytest.raises(KeyError):
        s.fit(train, val)
