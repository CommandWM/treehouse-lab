# Treehouse Lab

<p align="center">
  <img src="docs/assets/treehouse-lab.png" alt="Treehouse Lab logo" width="420" />
</p>

<p align="center">
  <a href="https://github.com/CommandWM/treehouse-lab/actions/workflows/ci.yml"><img src="https://github.com/CommandWM/treehouse-lab/actions/workflows/ci.yml/badge.svg" alt="CI status" /></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-green.svg" alt="MIT License" /></a>
</p>

Treehouse Lab is an audit-first, contract-first workbench for tabular machine learning.

Current package version: `v1.2.0`. The product direction is Treehouse Lab 2.0: a reproducible XGBoost-first workflow for tabular classification and regression that moves from dataset intake to bounded experiments, journals, honest benchmark comparisons, and scoring-ready export bundles through stable CLI, API, and artifact contracts.

Treehouse is agent-accessible, not agent-dependent. A coding agent should be able to operate it through documented commands and JSON/API contracts, but the core loop must remain useful to a human at a terminal.

## What Treehouse Is

- a local workbench for single-table tabular classification and regression
- a bounded experiment loop around XGBoost-first baselines and candidates
- an incumbent registry with promotion and rejection decisions
- a journal/reporting layer that explains what changed and why
- a benchmark comparison harness against plain XGBoost and selected AutoML references
- an export handoff path for a trained scorer bundle

## What Treehouse Is Not

- broad AutoML
- a general-purpose agent framework
- a notebook-only workflow
- a hosted training platform
- hardened production serving infrastructure
- a model zoo chasing every learner family before the XGBoost-first loop is proven

## Core Principles

- **Contract-first.** CLI commands, JSON outputs, API responses, manifests, journals, reports, and exported scorers should have documented shapes.
- **Audit-first.** Every run should show the dataset, target, split policy, metric movement, run diff, promotion decision, readiness result, and artifacts.
- **Bounded improvement.** Candidate experiments come from explicit search spaces and mutation templates, not free-form code changes.
- **No leakage.** Split policy, target handling, preprocessing, and generated features must protect validation/test integrity.
- **Export is handoff.** The generated scorer is a minimal local package for downstream integration, not a production platform.

## Current Runnable Surface

The repo already includes:

- dataset specs in `configs/datasets/`
- benchmark suite configs in `configs/benchmark_suites/`
- bounded runner, proposal, loop, export, and comparison code in `src/treehouse_lab/`
- local artifacts and incumbents under `runs/`
- exported model bundles under `exports/`
- a FastAPI workbench API in `src/treehouse_lab/api.py`
- a Vite React workbench in `frontend/`
- export contract docs and scorer smoke tests

The current implementation supports binary and multiclass classification. Regression is part of the 2.0 direction, but it is not yet implemented in the runner, intake, benchmark, or export contracts. Time series, deep learning, and broad learner expansion remain outside the 2.0 center of gravity.

## Quickstart

Install the package and local web dependencies:

```bash
pip install -e '.[web]'
```

Run a baseline:

```bash
treehouse-lab baseline configs/datasets/breast_cancer.yaml
treehouse-lab baseline configs/datasets/churn_demo.yaml
```

Preview and run bounded next steps:

```bash
treehouse-lab propose configs/datasets/churn_demo.yaml
treehouse-lab diagnose configs/datasets/churn_demo.yaml
treehouse-lab loop configs/datasets/churn_demo.yaml --steps 3
```

Run the fixed public comparison suite without optional external AutoML runners:

```bash
treehouse-lab benchmark-suite configs/benchmark_suites/public_v1_3.yaml --skip-autogluon --skip-flaml
```

Fetch public probes and run the fuller benchmark environment when needed:

```bash
python3 scripts/fetch_bank_marketing.py
python3 scripts/fetch_adult.py
python3 scripts/fetch_covertype.py
./scripts/setup_benchmark_env.sh
.venv-benchmarks/bin/python -m treehouse_lab.cli benchmark-suite configs/benchmark_suites/public_v1_3.yaml
```

Start the local workbench API:

```bash
treehouse-lab-api
```

Start the React workbench in a second terminal:

```bash
cd frontend
npm install
npm run dev
```

The frontend runs at `http://127.0.0.1:5173/`. The API health endpoint is `http://127.0.0.1:8000/api/health`.

## CLI Commands

Current commands:

- `treehouse-lab baseline <config>`: train and log the initial incumbent for a dataset spec.
- `treehouse-lab candidate <config> --name ... --set ...`: run one explicit bounded mutation.
- `treehouse-lab propose <config>`: inspect the next bounded proposal without executing it.
- `treehouse-lab diagnose <config>`: inspect current diagnosis plus the next bounded proposal.
- `treehouse-lab loop <config> --steps N`: run a bounded loop with promote/reject decisions and narratives.
- `treehouse-lab compare <config>`: compare plain XGBoost, Treehouse baseline, Treehouse loop, and optional AutoML references for one dataset.
- `treehouse-lab benchmark-suite <suite>`: run a fixed suite of dataset comparison configs.
- `treehouse-lab export <config>`: package the incumbent as a reusable model artifact with an optional local scorer wrapper.

The 2.0 canonical workflow is documented in [CLI Contract](docs/cli.md). Missing canonical commands should be marked as planned until implemented; do not imply a command exists before the CLI supports it.

## Optional LLM Guidance

Treehouse can call an LLM-backed advisor or selection path, but that layer is optional and bounded.

When enabled, Treehouse generates candidate mutations deterministically first. The LLM can choose from that explicit list or summarize the comparison, but it must not invent split changes, target changes, undeclared features, or new evaluation policy.

Common provider modes are:

- `ollama`
- `agent_cli` for local Codex or Claude Code explanation-only calls
- `openai_compatible`
- `openai`

The React workbench settings view writes local, untracked LLM settings to `.treehouse_lab/llm_settings.json`. Shell environment variables remain available for scripted runs.

## Exporting A Model

Each successful run writes a reusable `model_bundle.pkl` artifact with:

- the trained model
- the fitted preprocessing contract from the training split
- input column expectations
- primary metric and run metadata

Export the current incumbent:

```bash
treehouse-lab export configs/datasets/bank_marketing_uci.yaml
```

Exports are written under `exports/<config-key>/<run-id>/` and include:

- `model_bundle.pkl`
- `app.py`
- `Dockerfile`
- `requirements.txt`
- `manifest.json`

The bundle is the primary artifact. `app.py` and `Dockerfile` are convenience wrappers for local scorer smoke tests and downstream handoff.

Run the optional local scorer:

```bash
cd exports/<config-key>/<run-id>
uvicorn app:app --host 0.0.0.0 --port 8000
```

Then call:

```bash
curl -s -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"records":[{"example_feature":1}]}'
```

Use [Export Contract](docs/export-contract.md) before integrating an exported package.

## Benchmarks

Treehouse benchmarks should answer when the bounded, audit-first workflow is worth using. They should not fake superiority over every AutoML runner.

The benchmark story compares:

- plain XGBoost baseline
- Treehouse baseline
- Treehouse bounded loop
- optional FLAML reference
- optional AutoGluon reference

Report raw metrics, runtime/budget, promotion decisions, artifact quality, and implementation-readiness status separately. A run can be benchmark-better and still not implementation-ready.

Classification and regression benchmarks should use separate suites and metric policies. Regression should not reuse classification promotion thresholds or probability/scorer response shapes.

See [Benchmark Pack](docs/benchmarks.md), [Benchmark Report](docs/benchmark-report.md), and [Evaluation Policy](docs/evaluation-policy.md).

## Documentation Map

- [2.0 Vision](docs/vision-2.0.md): product thesis, users, non-goals, and success criteria.
- [Architecture](docs/architecture.md): contract-first component hierarchy.
- [Contracts](docs/contracts.md): index of CLI, API, export, journal/report, benchmark, and agent contracts.
- [Walkthrough](docs/walkthrough.md): shortest local path through the current workbench.
- [Sample Outputs](docs/sample-outputs.md): examples of baseline, proposal, journal, and compare artifacts.
- [Export Contract](docs/export-contract.md): exported scorer package and manifest behavior.
- [Roadmap](docs/roadmap.md): 2.0 workstreams and ordering.

## Loop Contract

The bounded loop:

1. reads the dataset config, incumbent, and recent journal history
2. builds eligible candidates from declared templates
3. selects one next proposal
4. runs it against the fixed split and evaluation policy
5. promotes only if metric and readiness checks pass
6. writes artifacts, narrative, and journal entries
7. stops when budget or guardrails say to stop

The loop is documented in [Bounded Loop Contract](docs/autonomous-loop.md). The filename is historical; the contract is the important part.

## Development Checks

Run focused tests while editing:

```bash
python -m pytest tests/test_exporting.py
python -m pytest tests/test_version_metadata.py
```

Run the broader suite before claiming shared behavior is safe:

```bash
ruff check .
python -m pytest
```

## License

MIT
