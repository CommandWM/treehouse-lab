# Treehouse Lab

<p align="center">
  <img src="docs/assets/treehouse-lab.png" alt="Treehouse Lab logo" width="420" />
</p>

Treehouse Lab is a Karpathy-style autoresearch loop for tabular machine learning.

The idea is simple: give an agent a constrained playground around XGBoost style models, let it propose experiments, run them safely, keep only the winners, and leave behind a readable research log instead of a pile of notebook debris.

## Why this exists

Karpathy's `autoresearch` pattern is brilliant, but it was built around small language model training loops. Tabular ML is a better first proving ground for most real teams:

- experiments are cheaper
- metrics are clearer
- rollback is trivial
- strong baselines already exist
- business integration is much easier

Treehouse Lab is aimed at that gap.

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

## Current repo status

This repository now includes a runnable MVP slice:

- dataset specs in `configs/datasets/`
- a baseline and bounded-candidate runner in `src/treehouse_lab/`
- local artifact bundles and an incumbent registry under `runs/`
- a Streamlit demo UI in `app.py`

The full autoresearch loop is still in progress, but the baseline path is no longer just a placeholder.

## Quickstart

Install the package and the Streamlit UI extra:

```bash
pip install -e '.[ui]'
```

Run one of the example baselines from the CLI:

```bash
treehouse-lab baseline configs/datasets/breast_cancer.yaml
treehouse-lab baseline configs/datasets/churn_demo.yaml
```

Run the demo interface:

```bash
streamlit run app.py
```

If your local XGBoost install cannot load, for example because `libomp` is missing on macOS, the runner falls back to sklearn gradient boosting so the examples remain runnable.

## Included examples

Two examples are bundled so the repo is usable offline:

- `breast_cancer.yaml`: a real binary classification dataset via sklearn
- `churn_demo.yaml`: a synthetic business-style churn example with mixed feature types

This keeps the onboarding path self-contained while the public benchmark examples evolve.

## Proposed architecture

- `datasets/` dataset specs and split policies
- `configs/` model spaces, budgets, and guardrails
- `src/treehouse_lab/` runner, evaluator, registry, journal, mutators
- `program.md` agent instructions for Codex or Claude Code
- `docs/` architecture notes, roadmap, and experiment policy

## Open source building blocks

Treehouse Lab is not trying to reimplement the whole ecosystem. It should stand on the shoulders of tools that already work:

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
See [docs/autonomous-loop.md](docs/autonomous-loop.md) for the next implementation phase.

## Name

Why "Treehouse Lab"?

Because it feels like a place to tinker, test, and build sharp ideas above the noise, while still sounding like a serious public project instead of a tax form.

## License

MIT
