"""Behaviour of the previous pipeline that the package deliberately no longer has, kept only so the checks can reproduce old numbers."""

import numpy as np

from fp.calibration.max_conformal import MaxConformal


class PreviousMaxConformal(MaxConformal):
    """Linear-interpolated quantile at level min(1, ceil((n + 1)(1 - alpha)) / n), as before."""

    def threshold(self, alpha, length):
        n = len(self.maxima)
        level = min(1.0, float(np.ceil((n + 1) * (1.0 - alpha))) / n)
        return np.full(length, np.quantile(self.maxima, level))


def headnet_to_probe_state(state: dict) -> dict:
    """Parameter names of the earlier experiments' HeadNet -> the names in fp.models.ours.Probe."""
    renames = {"proj": "attention.proj", "mha": "attention.attn", "mha_norm": "attention.norm", "lstm": "trunk"}
    out = {}
    for key, value in state.items():
        head, _, rest = key.partition(".")
        out[f"{renames[head]}.{rest}" if head in renames else key] = value  # trunk (step MLP), score, mode, sem keep their names
    return out


class Float32Threshold(PreviousMaxConformal):
    """The previous code compared float32 scores with a Python float, which numpy 1.x rounds to float32."""

    def threshold(self, alpha, length):
        return super().threshold(alpha, length).astype(np.float32)
