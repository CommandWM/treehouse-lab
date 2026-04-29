# Evaluation Policy

Treehouse Lab should separate two decisions:

- `benchmark better`
- `implementation ready`

The first asks whether a run beat the incumbent under the configured promotion threshold.
The second asks whether the run is clean enough to be a credible implementation candidate.

## Benchmark Better

A run is benchmark-better only when:

- validation performance beats the incumbent
- the gain clears `promote_if_delta_at_least`
- the split policy was preserved
- the run completed successfully

This is the promotion decision.

## Implementation Ready

A run is implementation-ready only when all configured readiness checks pass.

The current readiness checks are:

- promotion bar: the run either established the first baseline or beat the incumbent
- minimum primary metric: the validation metric clears the configured floor
- train-validation gap: the overfit gap stays within bounds
- validation-test gap: holdout behavior stays close enough to validation behavior
- runtime budget: the run remains inside the allowed runtime
- feature budget: the final feature count remains within bounds when configured

These checks are dataset-specific because a smoke benchmark and an implementation-like benchmark should not demand identical bars.

## Why This Split Matters

Without this policy split, the project drifts into one of two bad states:

- leaderboard thinking, where any tiny metric win is treated as progress
- deployment thinking too early, where no experiment counts unless it is already production-perfect

Treehouse Lab needs both stages:

1. prove improvement discipline
2. prove implementation plausibility

## Common Outcomes

The easiest way to read a run is to treat the two decisions as separate gates:

### 1. Benchmark Better + Implementation Ready

This is the clean win.

Interpretation:

- the run beat the incumbent enough to justify promotion
- it also stayed inside the configured readiness limits

This is the closest thing Treehouse Lab has to "ship the new incumbent."

### 2. Benchmark Better + Not Implementation Ready

This is progress, but not a clean handoff candidate.

Interpretation:

- the run improved validation enough to matter
- but the readiness checks say it still needs work

Typical reasons:

- overfit gap widened too far
- validation and test behavior diverged
- runtime or feature budget was exceeded

### 3. Not Benchmark Better + Implementation Ready

This is a stable run that still did not earn promotion.

Interpretation:

- the run may be sensible in isolation
- but it did not improve the incumbent enough to justify replacement

This is a useful reminder that "clean" and "worth promoting" are not the same decision.

### 4. Not Benchmark Better + Not Implementation Ready

This is a reject.

Interpretation:

- the run did not help the benchmark story
- and it also failed the readiness bar

This is exactly the kind of dead-end that should be visible in the journal so the loop avoids pretending it made progress.

## Config Shape

Each dataset config can declare:

```yaml
benchmark:
  pack: v1
  profile: implementation_like
  objective: Check whether the loop can produce a model worth carrying toward implementation.

evaluation_policy:
  minimum_primary_metric: 0.84
  max_train_validation_gap: 0.04
  max_validation_test_gap: 0.03
  max_runtime_seconds: 30
  max_feature_count: 20
  require_promotion_for_readiness: true
```

## Expected Artifact Output

Each run summary should make the following obvious without opening raw JSON:

- benchmark profile
- benchmark status
- implementation readiness
- which checks passed or failed
- why the decision was made
