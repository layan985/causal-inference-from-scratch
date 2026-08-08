from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class LinearResult:
    beta: np.ndarray
    residuals: np.ndarray
    fitted: np.ndarray
    vcov: np.ndarray
    stderr: np.ndarray
    nobs: int
    k: int


def _as_2d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return x[:, None] if x.ndim == 1 else x


def ols(y, X, add_intercept: bool = True, hc1: bool = False) -> LinearResult:
    y = np.asarray(y, dtype=float).reshape(-1)
    X = _as_2d(np.asarray(X, dtype=float))
    if add_intercept:
        X = np.column_stack([np.ones(len(X)), X])
    if len(y) != len(X):
        raise ValueError("y and X must have the same number of rows")
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    fitted = X @ beta
    residuals = y - fitted
    n, k = X.shape
    bread = np.linalg.pinv(X.T @ X)
    if hc1:
        meat = X.T @ (X * residuals[:, None] ** 2)
        vcov = (n / max(n - k, 1)) * bread @ meat @ bread
    else:
        sigma2 = float(residuals @ residuals) / max(n - k, 1)
        vcov = sigma2 * bread
    stderr = np.sqrt(np.clip(np.diag(vcov), 0, None))
    return LinearResult(beta, residuals, fitted, vcov, stderr, n, k)


def cluster_robust_covariance(X, residuals, clusters, small_sample: bool = True) -> np.ndarray:
    X = _as_2d(np.asarray(X, dtype=float))
    residuals = np.asarray(residuals, dtype=float).reshape(-1)
    clusters = np.asarray(clusters)
    if not (len(X) == len(residuals) == len(clusters)):
        raise ValueError("X, residuals, and clusters must align")
    n, k = X.shape
    bread = np.linalg.pinv(X.T @ X)
    meat = np.zeros((k, k), dtype=float)
    unique = np.unique(clusters)
    for g in unique:
        mask = clusters == g
        score = X[mask].T @ residuals[mask]
        meat += np.outer(score, score)
    vcov = bread @ meat @ bread
    G = len(unique)
    if small_sample and G > 1 and n > k:
        vcov *= (G / (G - 1)) * ((n - 1) / (n - k))
    return vcov
