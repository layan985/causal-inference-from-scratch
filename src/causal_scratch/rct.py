from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class RCTResult:
    ate: float
    stderr: float


def difference_in_means(y, treatment) -> RCTResult:
    y = np.asarray(y, float); t = np.asarray(treatment, int)
    yt, yc = y[t==1], y[t==0]
    if len(yt) < 2 or len(yc) < 2:
        raise ValueError("Need at least two observations per arm")
    ate = yt.mean() - yc.mean()
    se = np.sqrt(yt.var(ddof=1)/len(yt) + yc.var(ddof=1)/len(yc))
    return RCTResult(float(ate), float(se))


def randomization_inference(y, treatment, reps=5000, seed=0) -> dict:
    y = np.asarray(y, float); t = np.asarray(treatment, int)
    obs = difference_in_means(y, t).ate
    rng = np.random.default_rng(seed)
    stats = np.empty(reps)
    for r in range(reps):
        stats[r] = difference_in_means(y, rng.permutation(t)).ate
    p = (1 + np.sum(np.abs(stats) >= abs(obs))) / (reps + 1)
    return {"observed": float(obs), "p_value_two_sided": float(p), "null_stats": stats}
