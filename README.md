# Treehouse Lab

<p align="center">
  <img src="docs/assets/treehouse-lab.png" alt="Treehouse Lab logo" width="420" />
</p>

Treehouse Lab is a Karpathy-style autoresearch loop for tabular machine learning.

The idea is simple: give an agent a constrained playground around XGBoost-style models, let it propose experiments, run them safely, keep only the winners, and leave behind a readable research log instead of a pile of notebook debris.

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

## What exists now

This repository now includes a real runnable slice:

- dataset specs in `configs/datasets/`
- a baseline runner and bounded candidate execution in `src/treehouse_lab/`
- a deterministic autonomous loop with proposal selection and narratives
- local artifact bundles and an incumbent registry under `runs/`
- loop summaries under `runs/loops/`
- a Streamlit demo UI in `app.py`
- benchmark-pack configs for smoke, stress, and implementation-like evaluation

This is still an MVP, but it is no longer just scaffolding. You can now establish incumbents, inspect the next bounded proposal, run a short autonomous loop, and review the resulting narratives and artifacts.

## Quickstart

Install the package and the Streamlit UI extra:

```bash
pip install -e '.[ui]'
```

Run a baseline:

```bash
treehouse-lab baseline configs/datasets/breast_cancer.yaml
treehouse-lab baseline configs/datasets/churn_demo.yaml
```

Run the v1 benchmark pack:

```bash
treehouse-lab baseline configs/datasets/smoke_breast_cancer.yaml
treehouse-lab baseline configs/datasets/stress_churn.yaml
treehouse-lab baseline configs/datasets/implementation_churn.yaml
```

Preview the next bounded experiment without executing it:

```bash
treehouse-lab propose configs/datasets/churn_demo.yaml
```

Run the bounded autonomous loop:

```bash
treehouse-lab loop configs/datasets/churn_demo.yaml --steps 3
```

Benchmark-pack loop check:

```bash
treehouse-lab loop configs/datasets/implementation_churn.yaml --steps 3
```

Run the demo interface:

```bash
streamlit run app.py
```

If your local XGBoost install cannot load, for example because `libomp` is missing on macOS, the runner falls back to sklearn gradient boosting so the examples remain runnable.

## CLI commands

Treehouse Lab currently exposes four core commands:

- `treehouse-lab baseline <config>`: train and log the initial incumbent for a dataset spec
- `treehouse-lab candidate <config> --name ... --set ...`: run one explicit bounded mutation
- `treehouse-lab propose <config>`: inspect the next deterministic proposal without executing it
- `treehouse-lab loop <config> --steps N`: run a short autonomous research cycle with promote/reject decisions and narratives

The important distinction is that `propose` and `loop` do not freestyle. They operate inside explicit mutation templates and the declared search space in `configs/search_space.yaml`.

## Included examples

Examples are bundled so the repo is usable offline:

- `breast_cancer.yaml`: a real binary classification dataset via sklearn
- `churn_demo.yaml`: a synthetic business-style churn example with mixed feature types
- `smoke_breast_cancer.yaml`: a clean benchmark-pack smoke test
- `stress_churn.yaml`: a messier benchmark-pack stress test
- `implementation_churn.yaml`: a benchmark-pack implementation-like test

This keeps the onboarding path self-contained while the benchmark pack evolves.

The benchmark pack is documented in [docs/benchmarks.md](docs/benchmarks.md), and the readiness criteria are documented in [docs/evaluation-policy.md](docs/evaluation-policy.md).

## How the loop works

The current autonomous loop is intentionally conservative:

1. Read the dataset config, current incumbent, and recent journal history.
2. Score a small set of bounded mutation templates.
3. Select exactly one next proposal.
4. Run the proposal against the current incumbent parameters.
5. Promote only if the result clears the configured threshold.
6. Write artifacts, a narrative, and a loop summary.

The proposal engine currently prefers simple parameter-only moves first:

- `regularization_tighten`
- `learning_rate_tradeoff`
- `capacity_increase`
- `imbalance_adjustment`

Feature generation is stage-gated but not yet fully executed. That is deliberate. The first goal is to prove disciplined iteration before adding a more complex branch.

## Proposed architecture

- `datasets/` dataset specs and split policies
- `configs/` model spaces, budgets, and guardrails
- `src/treehouse_lab/` runner, loop controller, registry, journal, mutators, narratives
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

## Working through a cycle

The cleanest way to think about Treehouse Lab right now is:

- start with a dataset spec
- establish a baseline incumbent
- inspect or run the next bounded proposal
- review whether the run was promoted or rejected
- read the resulting narrative instead of reverse-engineering raw metrics

For now, that is the right level of interaction. The repo is aimed at tightening the research loop first, then layering on richer UI and eventual serving or system integration later.

## Name

Why "Treehouse Lab"?

Because it feels like a place to tinker, test, and build sharp ideas above the noise, while still sounding like a serious public project instead of a tax form.

## License

MIT
