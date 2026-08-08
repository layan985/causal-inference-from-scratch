# RA technical bench

Run these under time pressure. The objective is not only the final coefficient: leave an auditable analysis trail.

## Task 01 — panel construction (90 min)
Five CSVs contain inconsistent firm identifiers, duplicate fiscal years, two currencies, and silent unit changes. Build a unique panel, document exclusions, and emit a QA report.

## Task 02 — 5 million rows (90 min)
Use DuckDB/SQL plus Python to aggregate and merge without loading the raw table fully into memory. Record wall time and peak memory.

## Task 03 — event study (90 min)
Given treatment cohorts and raw outcomes, construct event time, select valid controls, estimate dynamic ATT, and diagnose pre-trends. Explain why vanilla TWFE can fail.

## Task 04 — RDD (60 min)
Recover a discontinuity from a raw running variable, produce bandwidth sensitivity, a manipulation diagnostic plan, and a placebo-cutoff table.

## Task 05 — IV failure (60 min)
One instrument is weak and one violates exclusion. Diagnose both before estimating the preferred causal model.

## Task 06 — replication bug hunt (90 min)
A supplied script contains duplicated units, post-treatment controls, wrong clustering, and an off-by-one event-time error. Find and document all four.
