from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .linear import ols, LinearResult


@dataclass(frozen=True)
class IVResult:
    second_stage: LinearResult
    first_stage_f: float
    x_hat: np.ndarray


def two_stage_least_squares(y, x_endog, z, controls=None, add_intercept: bool = True) -> IVResult:
    y = np.asarray(y, float).reshape(-1)
    x = np.asarray(x_endog, float).reshape(-1)
    z = np.asarray(z, float)
    if z.ndim == 1:
        z = z[:, None]
    if controls is None:
        C = np.empty((len(y), 0))
    else:
        C = np.asarray(controls, float)
        if C.ndim == 1:
            C = C[:, None]
    Z = np.column_stack([z, C])
    fs = ols(x, Z, add_intercept=add_intercept, hc1=False)
    x_hat = fs.fitted
    X2 = np.column_stack([x_hat, C])
    ss = ols(y, X2, add_intercept=add_intercept, hc1=True)

    # Partial first-stage F for excluded instruments via restricted-vs-full SSR.
    if C.shape[1] == 0:
        restricted = ols(x, np.empty((len(x), 0)), add_intercept=True)
    else:
        restricted = ols(x, C, add_intercept=True)
    q = z.shape[1]
    ssr_r = float(restricted.residuals @ restricted.residuals)
    ssr_u = float(fs.residuals @ fs.residuals)
    df_u = max(len(y) - fs.k, 1)
    f = ((ssr_r - ssr_u) / max(q, 1)) / (ssr_u / df_u)
    return IVResult(ss, float(f), x_hat)
