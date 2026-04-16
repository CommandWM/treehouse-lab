# Booster Seat

Booster Seat is a Karpathy-style autoresearch loop for tabular machine learning.

The idea is simple: give an agent a constrained playground around XGBoost style models, let it propose experiments, run them safely, keep only the winners, and leave behind a readable research log instead of a pile of notebook debris.

## Why this exists

Karpathy's `autoresearch` pattern is brilliant, but it was built around small language model training loops. Tabular ML is a better first proving ground for most real teams:

- experiments are cheaper
- metrics are clearer
- rollback is trivial
- strong baselines already exist
- business integration is much easier

Booster Seat is aimed at that gap.

## Design principles

- **Constrained search beats chaos.** The agent should mutate within a safe, explicit search space.
- **Evaluation is sacred.** No leakage, no test-set gaming, no notebook nonsense.
- **Promotion must be earned.** A run only wins if it beats the incumbent under the agreed metric and guardrails.
- **Every run tells a story.** Outputs should be understandable by a human reviewing them later.

## Initial scope

Version 1 is intentionally narrow:

- binary classification
- single-table tabular datasets
- XGBoost as the primary learner
- Optuna for hyperparameter search
- OpenFE for optional feature generation
- MLflow for experiment tracking
- local, offline evaluation only

Later versions can add LightGBM, CatBoost, time series, richer feature stores, and scheduled retraining.

## Core loop

1. Train a strong baseline.
2. Let the agent propose one mutation.
3. Run guarded evaluation.
4. Compare against the incumbent.
5. Promote only meaningful improvements.
6. Log the result, rationale, diff, and artifacts.
7. Repeat.

## Proposed architecture

- `datasets/` dataset specs and split policies
- `configs/` model spaces, budgets, and guardrails
- `src/booster_seat/` runner, evaluator, registry, journal, mutators
- `program.md` agent instructions for Codex or Claude Code
- `docs/` architecture notes, roadmap, and experiment policy

## Open source building blocks

Booster Seat is not trying to reimplement the whole ecosystem. It should stand on the shoulders of tools that already work:

- [XGBoost](https://github.com/dmlc/xgboost)
- [Optuna](https://optuna.org/)
- [OpenFE](https://github.com/IIIS-Li-Group/OpenFE)
- [MLflow](https://mlflow.org/)
- [Featuretools](https://docs.featuretools.com/en/stable/)
- [AutoGluon](https://auto.gluon.ai/stable/tutorials/tabular/index.html) as a benchmark, not necessarily the core engine

## What this repo is really building

The missing layer is the orchestration:

- agent program and research policy
- mutation boundaries
- evaluation guardrails
- incumbent promotion logic
- experiment journal and reporting
- reproducible dataset-aware runs

That is the product.

## MVP milestones

See [docs/mvp.md](docs/mvp.md) for the build plan.

## Name

Why "Booster Seat"?

Because the project gives boosting models a disciplined leg-up, and because a little cheekiness makes the repo more memorable than yet another grimly serious `tabular-autoresearch` clone.

## License

MIT
