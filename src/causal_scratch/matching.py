from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class MatchResult:
    att: float
    matched_control_index: np.ndarray
    distances: np.ndarray


def _sigmoid(z):
    z = np.clip(z, -35, 35)
    return 1 / (1 + np.exp(-z))


def logistic_propensity(treatment, X, max_iter=100, tol=1e-10, ridge=1e-8):
    t = np.asarray(treatment, float).reshape(-1)
    X = np.asarray(X, float)
    if X.ndim == 1:
        X = X[:, None]
    X = np.column_stack([np.ones(len(X)), X])
    beta = np.zeros(X.shape[1])
    for _ in range(max_iter):
        p = _sigmoid(X @ beta)
        W = p * (1-p)
        grad = X.T @ (t-p) - ridge*beta
        H = -(X.T @ (X * W[:, None])) - ridge*np.eye(X.shape[1])
        step = np.linalg.pinv(H) @ grad
        new = beta - step
        if np.max(np.abs(new-beta)) < tol:
            beta = new
            break
        beta = new
    return _sigmoid(X @ beta), beta


def nearest_neighbor_match(y, treatment, propensity, caliper=None, replace=True) -> MatchResult:
    y = np.asarray(y, float)
    t = np.asarray(treatment, int)
    p = np.asarray(propensity, float)
    ti = np.where(t == 1)[0]
    ci = list(np.where(t == 0)[0])
    matched, dists, effects = [], [], []
    for i in ti:
        if not ci:
            break
        distances = np.abs(p[ci] - p[i])
        jpos = int(np.argmin(distances))
        j = ci[jpos]
        dist = float(distances[jpos])
        if caliper is not None and dist > caliper:
            continue
        matched.append(j); dists.append(dist); effects.append(y[i]-y[j])
        if not replace:
            ci.pop(jpos)
    if not effects:
        raise ValueError("No matches survived")
    return MatchResult(float(np.mean(effects)), np.asarray(matched), np.asarray(dists))
