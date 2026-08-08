from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class SyntheticControlResult:
    weights: np.ndarray
    synthetic_pre: np.ndarray
    pre_rmse: float


def _project_simplex(v: np.ndarray) -> np.ndarray:
    """Euclidean projection onto {w>=0, sum(w)=1}."""
    v = np.asarray(v, float)
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u) - 1
    rho = np.nonzero(u - cssv / (np.arange(len(u)) + 1) > 0)[0][-1]
    theta = cssv[rho] / (rho + 1)
    return np.maximum(v - theta, 0)


def synthetic_control(treated_pre, donor_pre, learning_rate=0.05, max_iter=20000, tol=1e-11) -> SyntheticControlResult:
    y = np.asarray(treated_pre, float).reshape(-1)
    X = np.asarray(donor_pre, float)
    if X.ndim != 2 or X.shape[0] != len(y):
        raise ValueError("donor_pre must be T x J and align with treated_pre")
    J = X.shape[1]
    w = np.ones(J) / J
    # Lipschitz-scaled step makes projected gradient stable across scales.
    spectral = np.linalg.norm(X, 2) ** 2
    step = learning_rate / max(spectral, 1e-12)
    for _ in range(max_iter):
        grad = 2 * X.T @ (X @ w - y)
        new = _project_simplex(w - step * grad)
        if np.max(np.abs(new - w)) < tol:
            w = new
            break
        w = new
    synth = X @ w
    rmse = float(np.sqrt(np.mean((y - synth) ** 2)))
    return SyntheticControlResult(w, synth, rmse)
