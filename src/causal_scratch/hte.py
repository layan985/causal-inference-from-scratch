from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .linear import ols


@dataclass(frozen=True)
class LinearCATEResult:
    baseline: float
    modifiers: np.ndarray
    cate: np.ndarray


def linear_cate(y, treatment, X):
    """Linear heterogeneous-treatment model: Y ~ X + T + T*X.

    Educational benchmark only: identification still requires unconfoundedness or randomization.
    """
    y = np.asarray(y, float); t = np.asarray(treatment, float)
    X = np.asarray(X, float)
    if X.ndim == 1:
        X = X[:, None]
    design = np.column_stack([X, t, X * t[:, None]])
    res = ols(y, design, add_intercept=True, hc1=True)
    p = X.shape[1]
    baseline = float(res.beta[1+p])
    modifiers = res.beta[2+p:2+2*p]
    cate = baseline + X @ modifiers
    return LinearCATEResult(baseline, modifiers, cate)
