# Glossary

## Baseline

The first disciplined run for a dataset config. It establishes the initial incumbent and creates the benchmark every later mutation must beat.

## Incumbent

The best promoted run currently registered for a dataset config.

## Benchmark Status

The answer to: did this run improve the benchmark? Typical values are `baseline_established`, `better_than_incumbent`, and `not_better_than_incumbent`.

## Implementation Readiness

The answer to: is this run credible enough to carry toward implementation? This is stricter than benchmark improvement.

## Diagnosis

The loop's current reading of what is going wrong or right. Examples include `overfit`, `underfit`, `class_imbalance`, `plateau`, and `healthy`.

## Quality Floor

The minimum acceptable primary metric for a benchmark profile. A run can improve the incumbent and still miss the quality floor.

## Promotion Threshold

The minimum improvement needed to replace the incumbent. This prevents trivial gains from being treated as meaningful progress.

## Mutation

A bounded change to the current model setup. In v1 these are conservative XGBoost parameter templates rather than free-form rewrites.

## Proposal

The next experiment the loop intends to run, including hypothesis, rationale, risk, and exact parameter overrides.

## Reason Codes

Structured labels attached to a run that explain why it was promoted or rejected and what diagnosis/readiness conditions applied.

## Smoke Benchmark

A clean benchmark used to prove the loop works end to end.

## Stress Benchmark

A messier benchmark used to test whether the loop stays disciplined under missingness, weak signal, or rare categories.

## Implementation-Like Benchmark

A benchmark intended to resemble a plausible first deployment target, so the loop can be judged on implementation plausibility as well as metric improvement.

## Artifact Bundle

The files written for each run, such as `summary.md`, `metrics.json`, `assessment.json`, `diagnosis.json`, and `feature_importances.csv`.
