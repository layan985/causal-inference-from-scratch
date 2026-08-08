from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DID2x2Result:
    att: float
    treated_change: float
    control_change: float


def did_2x2(y, treated, post) -> DID2x2Result:
    df = pd.DataFrame({"y": y, "treated": treated, "post": post})
    means = df.groupby(["treated", "post"])["y"].mean()
    needed = [(0, 0), (0, 1), (1, 0), (1, 1)]
    if any(k not in means.index for k in needed):
        raise ValueError("All treated/control x pre/post cells are required")
    tchg = means.loc[(1, 1)] - means.loc[(1, 0)]
    cchg = means.loc[(0, 1)] - means.loc[(0, 0)]
    return DID2x2Result(float(tchg - cchg), float(tchg), float(cchg))


def group_time_att(df: pd.DataFrame, outcome: str, unit: str, time: str, cohort: str) -> pd.DataFrame:
    """Simple unconditional group-time ATT using never/not-yet-treated controls.

    For cohort g and time t>=g, compare each group's change from g-1 to t with
    units untreated through t. This mirrors the core comparison logic of modern
    staggered DiD but omits covariate adjustment and inference.
    """
    d = df[[outcome, unit, time, cohort]].copy()
    cohorts = sorted(c for c in d[cohort].dropna().unique())
    rows = []
    for g in cohorts:
        base = g - 1
        treated_units = set(d.loc[d[cohort] == g, unit])
        for t in sorted(x for x in d[time].unique() if x >= g):
            control_units = set(d.loc[d[cohort].isna() | (d[cohort] > t), unit])
            if not treated_units or not control_units:
                continue
            piv = d[d[unit].isin(treated_units | control_units) & d[time].isin([base, t])].pivot_table(
                index=unit, columns=time, values=outcome, aggfunc="mean"
            ).dropna()
            if base not in piv.columns or t not in piv.columns:
                continue
            change = piv[t] - piv[base]
            tr = change.loc[change.index.intersection(treated_units)]
            co = change.loc[change.index.intersection(control_units)]
            if len(tr) == 0 or len(co) == 0:
                continue
            rows.append({"cohort": g, "time": t, "event_time": t-g, "att": float(tr.mean()-co.mean()),
                         "n_treated": len(tr), "n_control": len(co)})
    return pd.DataFrame(rows)


def aggregate_event_time(att_gt: pd.DataFrame) -> pd.DataFrame:
    if att_gt.empty:
        return pd.DataFrame(columns=["event_time", "att", "n_cohorts"])
    return (att_gt.groupby("event_time")
            .apply(lambda g: pd.Series({"att": np.average(g["att"], weights=g["n_treated"]),
                                       "n_cohorts": g["cohort"].nunique()}), include_groups=False)
            .reset_index())
