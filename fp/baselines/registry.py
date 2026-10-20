"""Name -> Scorer factory for every method (baselines and ours).

Which calibrator a method gets is decided in one place, ``calibration/protocol.py``; the registry only builds scorers.

A baseline that cannot be built for a dataset raises ``NotImplementedError`` with the reason;
``run_all`` catches exactly that and records the method as skipped in the summary, so nothing
is dropped silently.
"""

from typing import Callable

from ..config import DatasetSpec
from ..models.base import Scorer

METHODS: dict[str, Callable[..., Scorer]] = {}  # name -> factory(spec, split) -> Scorer


def register(name: str):
    def wrap(factory):
        METHODS[name] = factory
        return factory

    return wrap


def build(name: str, spec: DatasetSpec, split) -> Scorer:
    if name not in METHODS:
        raise KeyError(f"unknown method {name!r}; registered: {sorted(METHODS)}")
    return METHODS[name](spec, split)
