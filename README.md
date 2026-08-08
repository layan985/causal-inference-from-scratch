# causal-inference-from-scratch

A technical-test repository for learning causal inference by implementing the estimators rather than hiding behind package syntax.

**Goal:** be able to derive, code, test, break, and explain the estimators that show up in empirical-economics RA/predoc work.

## Implemented from scratch

- OLS with classical and HC1 covariance
- cluster-robust sandwich covariance
- one-way fixed-effects / within estimator
- iterative two-way demeaning
- IV / 2SLS with excluded-instrument first-stage F diagnostic
- 2x2 difference-in-differences
- cohort-time staggered DiD comparison logic and event-time aggregation
- local-linear regression discontinuity with triangular kernel
- synthetic control via projected-gradient simplex weights
- propensity-score logistic regression via Newton iterations
- nearest-neighbor matching with optional caliper
- randomized experiment difference-in-means
- permutation/randomization inference
- linear CATE / treatment-interaction model

The code is intentionally small enough to read. It is **educational**, not a replacement for battle-tested production packages such as `fixest`, `did`, `rdrobust`, `ivreg`, or specialized synthetic-control libraries.

## Why this is harder than a notebook portfolio

Every estimator has:

1. an implementation;
2. a synthetic data-generating process with known truth;
3. a test that must recover that truth;
4. an assumptions/failure-modes document;
5. an oral-exam question bank;
6. timed RA benchmark tasks.

A green test suite therefore means more than “the code ran”: it means the estimator survived a known-truth experiment.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e '.[dev]'
pytest
python examples/demo.py
```

## What to add next

The next release should add wild-cluster bootstrap, covariate-adjusted doubly robust DiD, honest causal trees/forests, RDD bias correction, weak-IV robust inference, randomization inference under clustered assignment, sensitivity analysis for omitted variables, and simulation notebooks comparing estimators under deliberate assumption failures.

## Standard for claiming mastery

Do not say “I know DiD” because you can call a package. You should be able to explain the estimand, identify the comparison group, code the simplest version from cell means, explain staggered-treatment contamination, diagnose anticipation and pre-trends, select clustering based on assignment, and reproduce the result in a production package.
