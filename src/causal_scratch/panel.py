from __future__ import annotations
import numpy as np
import pandas as pd
from .linear import ols, LinearResult


def _demean_by_group(values: np.ndarray, groups) -> np.ndarray:
    df = pd.DataFrame(values)
    group = pd.Series(groups).reset_index(drop=True)
    return (df - df.groupby(group).transform("mean")).to_numpy()


def within_estimator(y, X, entity) -> LinearResult:
    y = np.asarray(y, dtype=float).reshape(-1, 1)
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    yd = _demean_by_group(y, entity).reshape(-1)
    Xd = _demean_by_group(X, entity)
    return ols(yd, Xd, add_intercept=False, hc1=True)


def two_way_demean(values, entity, time, tol: float = 1e-12, max_iter: int = 1000):
    arr = np.asarray(values, dtype=float)
    was_1d = arr.ndim == 1
    if was_1d:
        arr = arr[:, None]
    out = arr.copy()
    for _ in range(max_iter):
        old = out.copy()
        out = _demean_by_group(out, entity)
        out = _demean_by_group(out, time)
        if np.max(np.abs(out - old)) < tol:
            break
    return out.reshape(-1) if was_1d else out
