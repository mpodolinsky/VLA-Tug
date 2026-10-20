"""Behaviour of the previous pipeline that the package deliberately no longer has, kept only so the checks can reproduce old numbers."""

import numpy as np

from fp.calibration.max_conformal import MaxConformal


class PreviousMaxConformal(MaxConformal):
    """Linear-interpolated quantile at level min(1, ceil((n + 1)(1 - alpha)) / n), as before."""

    def threshold(self, alpha, length):
        n = len(self.maxima)
        level = min(1.0, float(np.ceil((n + 1) * (1.0 - alpha))) / n)
        return np.full(length, np.quantile(self.maxima, level))


class Float32Threshold(PreviousMaxConformal):
    """The previous code compared float32 scores with a Python float, which numpy 1.x rounds to float32."""

    def threshold(self, alpha, length):
        return super().threshold(alpha, length).astype(np.float32)
