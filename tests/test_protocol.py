import itertools

import numpy as np
import pytest

from fp.calibration.max_conformal import MaxConformal
from fp.baselines.distance_grid import all_grid_scorers
from fp.calibration.protocol import MAX_CONFORMAL, SAFE_BAND, calibrator_name, make_calibrator
from fp.calibration.safe_band import SafeBand
from fp.data.labels import VARIANTS


@pytest.mark.parametrize("method", ["safe_mlp", "safe_lstm", "failurespot"])
def test_safe_models_and_failurespot_use_the_safe_band(method):
    assert calibrator_name(method) == SAFE_BAND


@pytest.mark.parametrize("method", ["stac", "stac_single", "actprobe"])
def test_stac_and_actprobe_use_max_conformal(method):
    assert calibrator_name(method) == MAX_CONFORMAL


def test_every_probe_name_gets_max_conformal():
    for trunk, heads, variant in itertools.product(("lstm", "mlp"), ("plain", "modes", "clip", "modes_clip"), VARIANTS):
        assert calibrator_name(f"{trunk}_{heads}__{variant}") == MAX_CONFORMAL


@pytest.mark.parametrize("method", ["mahalanobis", "euclidean", "pca_kmeans"])
def test_the_embedding_distance_baselines_use_the_safe_band(method):
    assert calibrator_name(method) == SAFE_BAND


def test_every_configuration_of_the_distance_baselines_gets_the_safe_band():
    episodes = {}  # configurations are named without needing any data
    for scorer in all_grid_scorers(episodes, {}):
        assert calibrator_name(scorer.name) == SAFE_BAND


def test_a_distance_configuration_of_an_unknown_baseline_is_an_error():
    with pytest.raises(KeyError):
        calibrator_name("cosine__k10_cumsum")


@pytest.mark.parametrize("name", ["safe_gru", "lstm_modes__not_a_variant", "gru_plain__onset", ""])
def test_unknown_names_are_an_error_not_a_default(name):
    with pytest.raises(KeyError):
        calibrator_name(name)


def test_make_calibrator_returns_the_right_class_and_passes_the_seed():
    band = make_calibrator("safe_lstm", seed=7)
    assert isinstance(band, SafeBand) and band.seed == 7 and not band.finite_sample  # SAFE-faithful quantile
    assert isinstance(make_calibrator("lstm_modes_clip__onset_pre"), MaxConformal)


def test_stac_max_over_steps_is_the_final_value_because_its_scores_are_non_negative():
    """The reason max-conformal on STAC equals thresholding the final accumulated score."""
    rng = np.random.default_rng(0)
    cumulated = [np.cumsum(rng.random(rng.integers(20, 60))) for _ in range(30)]  # non-negative per-step scores
    m = MaxConformal().fit(cumulated)
    assert np.array_equal(m.maxima, np.sort([c[-1] for c in cumulated]))
