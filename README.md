# Causal inference from scratch

A package call can return a coefficient without making the comparison, weighting, or variance calculation obvious. I wrote small versions of common estimators so I could trace those steps and deliberately break their assumptions.

The code is educational. It is not a substitute for `fixest`, `did`, `rdrobust`, `ivreg`, or other maintained packages in empirical work.

## Implemented

- OLS with classical and HC1 covariance
- cluster-robust covariance
- one-way fixed effects and iterative two-way demeaning
- IV / 2SLS with a first-stage diagnostic
- 2×2 difference in differences
- cohort-time staggered-DiD comparisons and event-time aggregation
- local-linear regression discontinuity
- synthetic control with simplex weights
- propensity-score estimation and nearest-neighbor matching
- randomized-experiment difference in means and permutation inference
- a linear treatment-interaction model

Each estimator has a synthetic data-generating process with known parameters. The tests ask whether the implementation recovers those parameters or respects an algebraic constraint. That is useful for finding coding mistakes; it does not prove the estimator's assumptions hold in real data.

## Run

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e '.[dev]'
pytest
python examples/demo.py
```

## Limits I am keeping visible

- The staggered-DiD code demonstrates comparison logic; it does not reproduce every estimator or inference option in maintained packages.
- The RDD implementation is local-linear and does not include the full bias-corrected inference used by `rdrobust`.
- The synthetic-control optimizer is intentionally small and is not a general constrained-optimization package.
- Passing a known-truth simulation says nothing about selection, anticipation, interference, or measurement error in an application.

More detailed assumptions and failure modes are in [docs/ASSUMPTIONS.md](docs/ASSUMPTIONS.md).

## Next experiment

The next useful addition is not another estimator list. It is a simulation that compares nominal and actual rejection rates for cluster-robust inference with few treated clusters, including a wild-cluster bootstrap benchmark.
