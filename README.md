# Treehouse Lab

<p align="center">
  <img src="docs/assets/treehouse-lab.png" alt="Treehouse Lab logo" width="420" />
</p>

Treehouse Lab is a Karpathy-style autoresearch loop for tabular machine learning.

Current checkpoint: `v1.0.0`. The repo now has a real end-to-end local workflow: dataset-first intake, bounded autoresearch loops, grounded LLM assistance, coach-triggered bounded execution, local settings for provider credentials, and exportable model artifacts with optional container packaging.

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

## V1 status

Version 1 is intentionally practical rather than broad. It now covers:

- dataset-first config intake for local CSV-backed tabular problems
- bounded baseline, candidate, diagnose, propose, and loop workflows
- React workbench for current state, journal inspection, settings, and coaching
- optional LLM guidance through Ollama, Claude Code or Codex, and OpenAI-compatible APIs
- structured coach recommendations that can trigger valid bounded runs
- exportable model artifacts with optional FastAPI and Docker wrappers

What it still does not try to be:

- a hosted training platform
- an unconstrained agent that edits the search space on its own
- a hardened production serving stack

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

This is now the `v1` slice. You can establish incumbents, inspect the next bounded proposal, run a short autonomous loop, use the research coach, execute bounded coach recommendations, and export a trained model as a reusable handoff package.

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

Run the legacy Streamlit demo interface:

```bash
streamlit run app.py
```

The Streamlit surface is now meant to teach the loop as well as operate it: a guided blueprint view, current diagnosis/proposal summary, and an in-app glossary mirror the underlying run artifacts.

Run the React UI instead:

```bash
pip install -e '.[web]'
treehouse-lab-api
```

In a second terminal:

```bash
cd frontend
npm install
npm run dev
```

The React UI is the preferred path for the richer guided interface. Streamlit can remain as a lightweight fallback while the React surface evolves.

To enable the optional grounded research coach in the Current State view:

```bash
pip install -e '.[web,llm]'
```

You can now manage LLM provider settings from the React UI under `Settings`. Treehouse Lab stores those values in a local untracked file at `.treehouse_lab/llm_settings.json`, and the next advisor or coach request will pick them up immediately. This is the preferred path if you want to rotate keys without bouncing back to shell exports.

Treehouse Lab now supports three LLM interaction paths for the advisor:

1. `ollama`
2. `agent_cli` for Claude Code or Codex
3. `openai_compatible`

`openai` remains available as a direct shortcut for the native OpenAI API.

## Path 1: Ollama

Default path: local Ollama.

```bash
export TREEHOUSE_LAB_LLM_PROVIDER=ollama
export TREEHOUSE_LAB_OLLAMA_BASE_URL=http://localhost:11434
export TREEHOUSE_LAB_LLM_MODEL=gemma3:4b
```

To let the bounded autonomous loop use the LLM to choose among eligible candidates instead of only showing post-hoc advice:

```bash
export TREEHOUSE_LAB_LOOP_LLM_SELECTION=true
```

That selection pass is still bounded: Treehouse Lab generates the candidate mutations deterministically first, then the LLM chooses from that explicit candidate list and records its rationale.

If you want to use signed-in Ollama cloud models without giving Treehouse Lab an API key, sign in through the local Ollama daemon and point the coach at a cloud model name:

```bash
ollama signin
export TREEHOUSE_LAB_LLM_MODEL=gpt-oss:120b-cloud
```

If you want to call Ollama cloud directly instead, point the base URL at `https://ollama.com` and set `OLLAMA_API_KEY`:

```bash
export TREEHOUSE_LAB_OLLAMA_BASE_URL=https://ollama.com
export OLLAMA_API_KEY=your_key_here
export TREEHOUSE_LAB_LLM_MODEL=glm-4.6
```

## Path 2: Claude Code Or Codex

Use a local coding-agent CLI as the advisor backend:

```bash
export TREEHOUSE_LAB_LLM_PROVIDER=agent_cli
export TREEHOUSE_LAB_AGENT_CLI=codex
export TREEHOUSE_LAB_LLM_MODEL=gpt-5.4-mini
```

Or:

```bash
export TREEHOUSE_LAB_LLM_PROVIDER=agent_cli
export TREEHOUSE_LAB_AGENT_CLI=claude
export TREEHOUSE_LAB_LLM_MODEL=sonnet
```

The Codex path runs `codex exec` in read-only mode. The Claude path runs `claude -p` with tools disabled, so both stay in explanation mode instead of editing the repo.

## Path 3: OpenAI-Compatible API

For any service that exposes an OpenAI-compatible chat endpoint:

```bash
export TREEHOUSE_LAB_LLM_PROVIDER=openai_compatible
export TREEHOUSE_LAB_OPENAI_COMPATIBLE_BASE_URL=https://your-provider.example/v1
export TREEHOUSE_LAB_OPENAI_COMPATIBLE_API_KEY=your_key_here
export TREEHOUSE_LAB_LLM_MODEL=provider/model
```

## Direct OpenAI Shortcut

If you want the native OpenAI API directly:

```bash
export TREEHOUSE_LAB_LLM_PROVIDER=openai
export OPENAI_API_KEY=your_key_here
export TREEHOUSE_LAB_LLM_MODEL=gpt-5.4-mini
```

If your local XGBoost install cannot load, for example because `libomp` is missing on macOS, the runner falls back to sklearn gradient boosting so the examples remain runnable.

## CLI commands

Treehouse Lab currently exposes four core commands:

- `treehouse-lab baseline <config>`: train and log the initial incumbent for a dataset spec
- `treehouse-lab candidate <config> --name ... --set ...`: run one explicit bounded mutation
- `treehouse-lab propose <config>`: inspect the next deterministic proposal without executing it
- `treehouse-lab diagnose <config>`: inspect the current diagnosis plus the next bounded proposal
- `treehouse-lab loop <config> --steps N`: run a short autonomous research cycle with promote/reject decisions and narratives
- `treehouse-lab export <config>`: package the incumbent as a reusable model artifact with optional scoring and container wrappers

The important distinction is that `diagnose`, `propose`, and `loop` do not freestyle. They operate inside explicit mutation templates and the declared search space in `configs/search_space.yaml`.

## Exporting a model

Each run now writes a reusable `model_bundle.pkl` artifact containing:

- the trained model
- the fitted preprocessing contract from the training split
- the primary metric, params, and run metadata

To export the current incumbent for a dataset into a small handoff package:

```bash
treehouse-lab export configs/datasets/bank-valid-test.yaml
```

By default this writes to `exports/<config-key>/<run-id>/` and includes:

- `model_bundle.pkl`
- `app.py` with a minimal FastAPI `/predict` endpoint
- `Dockerfile` for containerizing that optional scorer
- `requirements.txt`
- `manifest.json`

The important point is that `model_bundle.pkl` is the primary handoff artifact. Consumers can either load that bundle directly in Python or use the generated API/container wrapper if that is more convenient.

Load the artifact directly:

```python
from treehouse_lab.exporting import load_exported_model_bundle

bundle = load_exported_model_bundle("exports/bank-valid-test/<run-id>/model_bundle.pkl")
print(bundle.feature_preprocessor.input_columns)
# Pass records that include all expected input columns.
predictions = bundle.predict_records([...])
```

Run the optional scorer locally:

```bash
cd exports/bank-valid-test/<run-id>
uvicorn app:app --host 0.0.0.0 --port 8000
```

Build the optional container image:

```bash
cd exports/bank-valid-test/<run-id>
docker build -t treehouse-lab-export .
docker run --rm -p 8000:8000 treehouse-lab-export
```

This is intentionally simple and Python-specific. It is meant as an exportable handoff package, not a hardened model-serving platform.

## Included examples

Examples are bundled so the repo is usable offline:

- `breast_cancer.yaml`: a real binary classification dataset via sklearn
- `churn_demo.yaml`: a synthetic business-style churn example with mixed feature types
- `smoke_breast_cancer.yaml`: a clean benchmark-pack smoke test
- `stress_churn.yaml`: a messier benchmark-pack stress test
- `implementation_churn.yaml`: a benchmark-pack implementation-like test

This keeps the onboarding path self-contained while the benchmark pack evolves.

The benchmark pack is documented in [docs/benchmarks.md](docs/benchmarks.md), the readiness criteria are documented in [docs/evaluation-policy.md](docs/evaluation-policy.md), and the core terms are collected in [docs/glossary.md](docs/glossary.md).

## UI Architecture

The Python research engine remains the source of truth:

- dataset configs in `configs/datasets/`
- incumbents and journal entries in `runs/`
- runner, diagnosis, and loop logic in `src/treehouse_lab/`

The React UI is a separate presentation layer:

- a thin FastAPI layer in `src/treehouse_lab/api.py`
- a Vite React client in `frontend/`

That split is deliberate. It keeps the Python loop stable while making the teaching surface easier to control and extend.

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
