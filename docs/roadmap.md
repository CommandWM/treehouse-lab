# Roadmap

## Current Position

Implementation baseline: `v1.2.0`.

Treehouse Lab is moving toward a 2.0 release framed as a contract-first, audit-first, agent-accessible tabular ML workbench.

The repo already has useful building blocks:

- dataset-first intake and generated dataset specs
- bounded XGBoost-first classification baseline, candidate, diagnose, propose, loop, compare, and export flows
- incumbent promotion and human-readable run summaries
- a React workbench for intake, current state, journal inspection, settings, and architecture
- capped train-only feature generation for plateaued loops
- export contracts and scorer smoke tests
- fixed public benchmark-suite wiring

The next work should sharpen contracts and evidence, not broaden the learner surface.

## 2.0 North Star

A user should be able to move from dataset to defensible model artifact through a bounded, reproducible workflow:

```bash
treehouse init --dataset data/churn.csv --target churn --task classification
treehouse profile
treehouse baseline
treehouse loop --budget 5
treehouse compare
treehouse export --run best
treehouse serve exported/churn_model
```

Some of those command names are still planned rather than implemented. The contract docs should always distinguish current commands from target workflow names.

The same 2.0 workflow should cover regression with XGBoost once the underlying task support lands:

```bash
treehouse init --dataset data/housing.csv --target sale_price --task regression
treehouse profile
treehouse baseline
treehouse loop --budget 5
treehouse compare
treehouse export --run best
treehouse serve exported/housing_model
```

2.0 succeeds when Treehouse can produce:

- dataset profile
- baseline metrics
- candidate run history
- incumbent decision log
- promotion and rejection reasons
- journal and report artifacts
- export manifest
- model bundle
- minimal scorer API
- documented CLI/API contracts

## Workstreams

### 1. Identity And Contracts

Make the project legible before adding more behavior.

Deliverables:

- 2.0 vision doc
- README opening aligned to audit-first tabular ML
- architecture refresh
- contracts index
- explicit non-goals
- release checklist

### 2. CLI As The Primary Serious Interface

The CLI should be stable enough for humans, scripts, CI, and coding agents.

Deliverables:

- canonical CLI workflow doc
- implemented or explicitly deferred command aliases
- stable JSON success and failure contracts
- help coverage for public commands
- examples that identify expected artifacts

### 3. API And Export Handoff

The local API and exported scorer package should be useful to downstream systems without source-code inspection.

Deliverables:

- API contract docs
- response-shape tests
- manifest schema
- prediction request and response schemas
- export smoke tests tied to the documented contract

### 4. Evaluation, Benchmarking, And Anti-Leakage

Treehouse should be honest about model quality and readiness.

Deliverables:

- refreshed evaluation policy
- benchmark harness contract
- reusable benchmark report template
- leakage and split guardrail tests
- deterministic promotion and rejection policy tests
- regression metrics, readiness checks, and benchmark suites that are separate from classification policy

### 5. Agent Accessibility

Agents should be consumers of public contracts, not privileged authors of hidden behavior.

Deliverables:

- agent usage doc
- machine-readable command guide
- task cards for profile, baseline, loop, compare, export, and scorer smoke tests
- agent-accessibility evaluation
- guardrails against free-form mutation of splits, targets, or evaluation policy

## Non-Goals For 2.0

- broad AutoML
- production serving infrastructure
- general-purpose agent framework behavior
- notebook-only workflows
- CatBoost, LightGBM, deep learning, or time-series expansion before the XGBoost-first classification and regression contracts are solid
- benchmark claims without fresh run artifacts and report evidence

## First PR-Sized Slice

Start with the documentation and instruction layer:

1. update `AGENTS.md`
2. add `docs/vision-2.0.md`
3. rewrite the README opening around 2.0
4. add `docs/contracts.md`
5. add `docs/cli.md`

That makes later implementation work safer because every public surface has a named contract to satisfy.
