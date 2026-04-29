# Benchmark Pack

Treehouse Lab v1 should prove disciplined improvement, not leaderboard chasing.

The benchmark pack is therefore organized around three questions:

- `smoke`: does the loop work end to end on a clean task?
- `stress`: does it stay sane on messy tabular data?
- `implementation_like`: does it produce a model you could plausibly carry toward production?

The benchmark pack is the repo's internal proving ground. It is not meant to become a giant leaderboard suite.

## Included benchmarks

### Smoke

Config: `configs/datasets/smoke_breast_cancer.yaml`

Purpose:

- verify the runner, promotion logic, incumbent registry, and narrative flow
- establish a fast, stable regression target for the repo itself
- make failures obvious when core loop changes break something basic

### Stress

Config: `configs/datasets/stress_churn.yaml`

Purpose:

- exercise train-only preprocessing
- test robustness to missing values and rare categories
- make sure small bounded mutations still behave sensibly when the data is less clean

This benchmark uses the synthetic churn generator in a deliberately messier profile rather than relying on Kaggle competition mechanics.

### Implementation-Like

Config: `configs/datasets/implementation_churn.yaml`

Purpose:

- evaluate whether a winning model is still compact and understandable
- check whether the run clears a stricter readiness bar, not just the benchmark bar
- mimic the kind of mixed-type business tabular data that is easier to imagine implementing

## How to use the pack

Recommended order:

```bash
treehouse-lab baseline configs/datasets/smoke_breast_cancer.yaml
treehouse-lab baseline configs/datasets/stress_churn.yaml
treehouse-lab baseline configs/datasets/implementation_churn.yaml
```

Then run bounded loops:

```bash
treehouse-lab loop configs/datasets/smoke_breast_cancer.yaml --steps 3
treehouse-lab loop configs/datasets/stress_churn.yaml --steps 3
treehouse-lab loop configs/datasets/implementation_churn.yaml --steps 3
```

## Interpreting results

Every run should now answer two separate questions:

- `Is this benchmark-better?`
- `Is this implementation-ready?`

Those are related, but not identical.

A run can be benchmark-better while still failing implementation readiness if it:

- overfits too much
- has unstable validation versus holdout behavior
- exceeds the feature budget
- exceeds the runtime budget
- fails the configured minimum metric bar

## Where Kaggle fits

Kaggle is still useful, but only as an external probe.

It is appropriate for questions like:

- do we look directionally competitive on a recognizable public task?
- does our loop generalize beyond bundled examples?

It is not the center of the project. The center is a disciplined benchmark-and-implementation loop.

## Public probes

The current public-dataset additions are meant to widen domain coverage without changing the core loop:

- `configs/datasets/bank_marketing_uci.yaml`: mixed-type imbalanced binary business classification
- `configs/datasets/adult_uci.yaml`: mixed-type binary census-income classification with missing values
- `configs/datasets/covertype_uci.yaml`: large multiclass land-cover classification outside the business domain

Recommended fetch/setup:

```bash
python3 scripts/fetch_bank_marketing.py
python3 scripts/fetch_adult.py
python3 scripts/fetch_covertype.py
```

## Fixed public suite

The first v1.3 benchmark contract lives at:

```bash
configs/benchmark_suites/public_v1_3.yaml
```

Run it without optional AutoGluon when you want a fast Treehouse/plain-XGBoost pass:

```bash
treehouse-lab benchmark-suite configs/benchmark_suites/public_v1_3.yaml --skip-autogluon
```

Run it with the practical AutoGluon reference after preparing the benchmark environment:

```bash
./scripts/setup_benchmark_env.sh
.venv-benchmarks/bin/python -m treehouse_lab.cli benchmark-suite configs/benchmark_suites/public_v1_3.yaml
```

The suite keeps the public benchmark story small and fixed:

- Bank Marketing: mixed-type imbalanced business classification
- Adult: mixed-type census-income classification with missing categorical values
- Covertype: larger multiclass land-cover classification

The suite currently compares plain XGBoost, Treehouse baseline, Treehouse bounded loop, and practical AutoGluon. FLAML is intentionally tracked as the next external runner rather than bundled into this first suite contract.

## 2.0 benchmark direction

For `2.0`, the benchmark story should widen in one specific way: external comparison without turning the repo into a model zoo.

The recommended shape is:

- keep the current smoke / stress / implementation-like pack as the internal regression contract
- add a small public comparison suite with fixed seeds, split policy, and runtime budgets
- compare:
  - plain XGBoost baseline
  - Treehouse Lab baseline
  - Treehouse Lab bounded loop
  - FLAML
  - AutoGluon
- report:
  - primary metric
  - runtime
  - artifact quality
  - auditability / reviewability

The goal is not to win every benchmark. The goal is to explain when Treehouse Lab is the right tool:

- when a team wants disciplined bounded search instead of opaque automation
- when promotion logic and run narratives matter
- when exported artifacts and reviewable decisions are more important than squeezing out every last benchmark point

Explicit non-goal for `2.0`:

- broadening into CatBoost, LightGBM, and a larger learner matrix before the XGBoost-first loop has earned its position
