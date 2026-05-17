# Treehouse Lab

<p align="center">
  <img src="docs/assets/treehouse-lab.png" alt="Treehouse Lab logo" width="420" />
</p>

<p align="center">
  <a href="https://github.com/CommandWM/treehouse-lab/actions/workflows/ci.yml"><img src="https://github.com/CommandWM/treehouse-lab/actions/workflows/ci.yml/badge.svg" alt="CI status" /></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-green.svg" alt="MIT License" /></a>
</p>

Treehouse Lab is an audit-first workbench for tabular machine learning.

Current checkpoint: `v1.1.0`. The repo now has a real end-to-end local workflow: dataset-first intake, bounded XGBoost-first experiments, grounded LLM assistance, coach-triggered bounded execution, local settings for provider credentials, explicit binary or multiclass classification support, benchmark comparisons, experiment journals, promotion policy, and exportable model bundles with optional container packaging.

The practical goal is simple: bring a local tabular dataset into a reviewable experiment loop, run only bounded candidate changes, promote only meaningful improvements, and leave behind enough artifacts for another person to audit the result without digging through a notebook pile.

## Why this exists

Treehouse Lab takes inspiration from Karpathy's `autoresearch` pattern: tight experiment loops, explicit search, and readable evidence after each run. The core product is narrower and more practical: an XGBoost-first lab for dataset intake, bounded mutation, incumbent promotion, journaled decisions, benchmark comparison, and model export.

Tabular ML is a better first proving ground for most real teams:

- experiments are cheaper
- metrics are clearer
- rollback is trivial
- strong baselines already exist
- business integration is much easier

Treehouse Lab is aimed at that gap.

## Design principles

- **Constrained search beats chaos.** Candidate experiments should mutate within a safe, explicit search space.
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

- binary and multiclass classification
- single-table tabular datasets
- XGBoost as the primary learner
- Optuna for hyperparameter search
- bounded train-only feature generation, with optional broader feature tooling later
- MLflow for experiment tracking
- local, offline evaluation only

Later work should deepen the XGBoost-first loop with stronger benchmarks, richer public datasets, and better auditability before the repo expands into additional learners.

## Core loop

1. Train a strong baseline.
2. Propose one bounded mutation.
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
- a FastAPI + React workbench for intake, current state, journal inspection, settings, and architecture
- benchmark-pack configs for smoke, stress, and implementation-like evaluation

This is now the `v1` slice. You can establish incumbents, inspect the next bounded proposal, run a short autonomous loop, use the research coach, execute bounded coach recommendations, and export a trained model as a reusable handoff package.

## Quickstart

Install the package and the local web UI dependencies:

```bash
pip install -e '.[web]'
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

Run the external comparison harness:

```bash
python3 scripts/fetch_bank_marketing.py
python3 scripts/fetch_adult.py
python3 scripts/fetch_covertype.py
./scripts/setup_benchmark_env.sh
TREEHOUSE_LAB_LLM_PROVIDER=agent_cli \
TREEHOUSE_LAB_AGENT_CLI=codex \
TREEHOUSE_LAB_LLM_MODEL=gpt-5.4-mini \
TREEHOUSE_LAB_LOOP_LLM_SELECTION=true \
.venv-benchmarks/bin/python -m treehouse_lab.cli compare \
  configs/datasets/bank_marketing_uci.yaml \
  --loop-steps 3 \
  --autogluon-profile practical \
  --llm-summary
```

Run the fixed public benchmark suite contract:

```bash
treehouse-lab benchmark-suite configs/benchmark_suites/public_v1_3.yaml --skip-autogluon --skip-flaml
```

Remove `--skip-autogluon --skip-flaml` when you are using the benchmark environment from `./scripts/setup_benchmark_env.sh`.

Run the new public dataset probes directly:

```bash
treehouse-lab baseline configs/datasets/adult_uci.yaml
treehouse-lab baseline configs/datasets/covertype_uci.yaml
```

Benchmark-pack loop check:

```bash
treehouse-lab loop configs/datasets/implementation_churn.yaml --steps 3
```

Run the API for the local workbench:

```bash
treehouse-lab-api
```

In a second terminal:

```bash
cd frontend
npm install
npm run dev
```

The React UI is the primary path for the guided local interface.

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
- `treehouse-lab compare <config>`: run a side-by-side comparison harness across plain XGBoost, Treehouse Lab, and optional external AutoML runners
  Use `--llm-summary` to ask the configured provider to synthesize where Treehouse Lab adds product value beyond raw metric comparison.
  The default AutoGluon mode is `--autogluon-profile practical`, which keeps the benchmark rerunnable instead of turning it into a long opaque sweep.
  The default FLAML mode uses a small `xgboost,rf,extra_tree` estimator list and the same validation holdout, so it stays an external benchmark rather than a new core learner surface.
- `treehouse-lab benchmark-suite <suite>`: run a fixed suite of dataset comparison configs and write one summary across all dataset reports
- `treehouse-lab export <config>`: package the incumbent as a reusable model artifact with optional scoring and container wrappers

The important distinction is that `diagnose`, `propose`, and `loop` do not freestyle. They operate inside explicit mutation templates and the declared search space in `configs/search_space.yaml`.

`compare` is intentionally separate from the core loop. It uses the same dataset config and split policy, but it exists to benchmark Treehouse Lab against external baselines such as plain XGBoost and optional AutoML runners rather than to widen the product's core learner surface.

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
- `bank_marketing_uci.yaml`: an optional public UCI business-style probe for external comparison
- `adult_uci.yaml`: an optional public mixed-type census-income probe
- `covertype_uci.yaml`: an optional public multiclass scale probe outside the business domain

This keeps the onboarding path self-contained while the benchmark pack evolves.

The benchmark pack is documented in [docs/benchmarks.md](docs/benchmarks.md), the decision-facing benchmark report guide is in [docs/benchmark-report.md](docs/benchmark-report.md), the readiness criteria are documented in [docs/evaluation-policy.md](docs/evaluation-policy.md), the export handoff contract is documented in [docs/export-contract.md](docs/export-contract.md), and the core terms are collected in [docs/glossary.md](docs/glossary.md).

If you want the shortest user-facing path through the current product, start with [docs/walkthrough.md](docs/walkthrough.md).
If you want to see what the product actually emits before running it yourself, read [docs/sample-outputs.md](docs/sample-outputs.md).
If you want the v1.2 polish closeout and review checklist, read [docs/v1-2-polish.md](docs/v1-2-polish.md).

## UI Architecture

The Python research engine remains the source of truth:

- dataset configs in `configs/datasets/`
- incumbents and journal entries in `runs/`
- runner, diagnosis, and loop logic in `src/treehouse_lab/`

The React UI is a separate presentation layer:

- a thin FastAPI layer in `src/treehouse_lab/api.py`
- a Vite React client in `frontend/`

That split is deliberate. It keeps the Python loop stable while making the teaching surface easier to control and extend.

## Near-term roadmap

The v1.2 polish layer is closed around the current `v1.1` loop. The next useful work is not a broad rewrite; it is backing the benchmark story with fresh suite outputs and then deepening bounded XGBoost search:

- run the fixed public suite and replace placeholder report cells with real evidence
- keep feature-generation decisions easy to audit and compare across runs
- keep the guided React workbench intake-first as new capabilities are added
- evaluate a broader feature stage later without relaxing leakage guardrails or bounded mutation policy

## 2.0 direction

Version `2.0` should not be a model-zoo release. The stronger direction is:

- stay XGBoost-first and bounded rather than rushing into CatBoost, LightGBM, or a broader learner matrix
- benchmark Treehouse Lab honestly against plain XGBoost, FLAML, and AutoGluon on a small public dataset suite
- use those comparisons to explain where Treehouse Lab is better: auditability, promotion policy, artifact quality, and human-readable research flow
- keep the public demo and benchmark story tight enough that a new user can understand the product quickly

See [docs/benchmark-report.md](docs/benchmark-report.md) for the product-facing interpretation layer: when to use Treehouse Lab, when to prefer plain XGBoost or AutoML, and what evidence must be present before a result is publishable.

See [docs/roadmap.md](docs/roadmap.md) for the integrated roadmap and how the current GitHub roadmap issues map onto the shipped `v1.1` state.

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

Once those bounded parameter moves plateau, the loop can now escalate to one explicit feature-generation branch:

- `feature_generation_enable`

That branch is still intentionally conservative. It fits a train-only numeric interaction plan with a hard feature cap, then reuses the same generated-feature contract for validation, test, and exported inference.

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
- [FLAML](https://github.com/microsoft/FLAML)
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

See [docs/mvp.md](docs/mvp.md) for the build plan and remaining shareability work.

## Working through a cycle

The cleanest way to think about Treehouse Lab right now is:

- start with a dataset spec
- establish a baseline incumbent
- inspect or run the next bounded proposal
- review whether the run was promoted or rejected
- read the resulting narrative instead of reverse-engineering raw metrics

Two decisions should always stay separate in that review:

- `benchmark better`: did this run actually beat the incumbent enough to matter?
- `implementation ready`: did it also stay inside the configured runtime, overfit, holdout, and feature-budget limits?

For now, that is the right level of interaction. The repo is aimed at tightening the research loop first, then layering on richer UI and eventual serving or system integration later.

## Name

Why "Treehouse Lab"?

Because it feels like a place to tinker, test, and build sharp ideas above the noise, while still sounding like a serious public project instead of a tax form.

## License

MIT
