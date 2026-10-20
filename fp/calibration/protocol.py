"""Which calibrator each method gets in the main results. The single place that says so.

    SAFE-MLP, SAFE-LSTM        SAFE functional band (their own)
    FailureSpot                SAFE-style band
    our probes                 max-conformal
    STAC (original, single)    max-conformal. STAC accumulates non-negative per-step scores, so the maximum over steps is
                               the final value of the accumulated score.
    ActProbe                   max-conformal (as in the ActProbe paper)
    Mahalanobis, Euclidean,
    PCA-KMeans                 SAFE band, as SAFE itself evaluates them (their score can be negative, so the maximum over
                               steps is not the final value, unlike STAC). All configurations of a baseline, named
                               <baseline>__<tag> (see baselines/distance.py), get the same calibrator.

Our probes may additionally be evaluated with the SAFE band as a secondary table, and LTT is a separate evaluation; neither
changes this table.
"""

import re

from ..data.labels import VARIANTS
from .base import Calibrator
from .max_conformal import MaxConformal
from .safe_band import SafeBand

SAFE_BAND = "safe_band"
MAX_CONFORMAL = "max_conformal"


PROTOCOL: dict[str, str] = {
    "safe_mlp": SAFE_BAND,
    "safe_lstm": SAFE_BAND,
    "failurespot": SAFE_BAND,
    "mahalanobis": SAFE_BAND,
    "euclidean": SAFE_BAND,
    "pca_kmeans": SAFE_BAND,
    "stac": MAX_CONFORMAL,
    "stac_single": MAX_CONFORMAL,
    "actprobe": MAX_CONFORMAL,
}

# Our probes are named <trunk>_<heads>__<label variant>, e.g. lstm_modes_clip__onset_pre (see train/ours.py).
_OURS = re.compile(rf"^(lstm|mlp)_(plain|modes|clip|modes_clip)__({'|'.join(VARIANTS)})$")
# One configuration of an embedding-distance baseline: <baseline>__<tag>, e.g. euclidean__k10_cumsum (see baselines/distance.py).
_DISTANCE_CONFIG = re.compile(r"^(mahalanobis|euclidean|pca_kmeans)__[a-z0-9_]+$")


def calibrator_name(method: str) -> str:
    """'safe_band' or 'max_conformal' for the method; KeyError for an unknown method."""
    if method in PROTOCOL:
        return PROTOCOL[method]
    if _OURS.match(method):
        return MAX_CONFORMAL
    if _DISTANCE_CONFIG.match(method):
        return PROTOCOL[method.split("__")[0]]
    raise KeyError(f"unknown method {method!r}; known: {sorted(PROTOCOL)}, our probes <lstm|mlp>_<heads>__<variant> "
                   f"and distance configurations <baseline>__<tag>")


def make_calibrator(method: str, seed: int = 0) -> Calibrator:
    """An unfitted calibrator for the method. ``seed`` is the shuffle seed of the SAFE band (the split seed)."""
    name = calibrator_name(method)
    return SafeBand(seed=seed) if name == SAFE_BAND else MaxConformal()
