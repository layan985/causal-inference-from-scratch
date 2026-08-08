from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .linear import ols


@dataclass(frozen=True)
class RDDResult:
    tau: float
    stderr: float
    n_left: int
    n_right: int
    bandwidth: float


def _triangular(u):
    return np.maximum(1 - np.abs(u), 0)


def local_linear_rdd(y, running, cutoff=0.0, bandwidth=1.0) -> RDDResult:
    y = np.asarray(y, float)
    x = np.asarray(running, float) - cutoff
    mask = np.abs(x) <= bandwidth
    y, x = y[mask], x[mask]
    d = (x >= 0).astype(float)
    w = _triangular(x / bandwidth)
    keep = w > 0
    y, x, d, w = y[keep], x[keep], d[keep], w[keep]
    X = np.column_stack([d, x, d*x])
    sw = np.sqrt(w)
    res = ols(y*sw, X*sw[:, None], add_intercept=True, hc1=True)
    # beta order: intercept, treatment jump, slope-left, slope-change
    return RDDResult(float(res.beta[1]), float(res.stderr[1]), int((x<0).sum()), int((x>=0).sum()), float(bandwidth))
