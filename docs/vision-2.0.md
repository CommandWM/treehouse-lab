# Treehouse Lab 2.0 Vision

Treehouse Lab 2.0 is a contract-first, audit-first, agent-accessible workbench for reproducible tabular ML experiments, bounded model improvement loops, and scoring-ready model handoff.

## Product Thesis

Treehouse sits between notebook ML, ad hoc XGBoost scripts, broad AutoML, and brittle agent demos.

It should provide:

- CLI/API-driven workflows with durable artifacts instead of notebook-only handoff
- bounded experiments, journals, promotion policy, and export bundles instead of ad hoc scripts
- narrow XGBoost-first classification and regression support instead of broad opaque AutoML
- agent-accessible commands and contracts instead of agent dependency

Treehouse is not trying to beat every AutoML tool on raw metric. It is trying to make tabular ML experimentation reviewable, bounded, reproducible, and handoff-ready.

## Target Users

- humans running local tabular experiments
- CLI users and scripts that need stable outputs
- API clients that need documented request/response shapes
- downstream systems that need a simple model handoff package
- coding agents that can follow documented commands without guessing internals

## North-Star Workflow

The target 2.0 workflow is:

```bash
treehouse init --dataset data/churn.csv --target churn --task classification
treehouse profile
treehouse baseline
treehouse loop --budget 5
treehouse compare
treehouse export --run best
treehouse serve exported/churn_model
```

The same workflow should apply to regression datasets once regression support is implemented:

```bash
treehouse init --dataset data/housing.csv --target sale_price --task regression
treehouse profile
treehouse baseline
treehouse loop --budget 5
treehouse compare
treehouse export --run best
treehouse serve exported/housing_model
```

The current CLI does not implement every command name in that sequence yet. Until it does, docs must mark missing commands as planned and point to the implemented equivalents.

## Success Criteria

2.0 succeeds when:

- a user can run a complete dataset-to-export workflow from the CLI
- the same workflow has documented API surfaces where implemented
- every major artifact has a documented contract
- experiments produce readable journals and promotion/rejection decisions
- exported scorer packages can be smoke-tested independently
- benchmarks compare Treehouse honestly against plain XGBoost and selected AutoML references
- a coding agent can operate Treehouse through documented commands without source-code guessing

## Principles

### Contract-First

Important surfaces need stable contracts:

- CLI command inputs and outputs
- JSON payloads
- API request and response shapes
- export manifest fields
- journal and report sections
- benchmark report format

### Audit-First

Every experiment should explain:

- what dataset was used
- what target was predicted
- how the split was created
- what changed
- why it changed
- which metric moved
- whether the run promoted
- whether readiness checks passed
- whether leakage was avoided

### Bounded Improvement

Agents and loop controllers can choose from explicit options, such as:

- feature selection
- capped generated features
- class weighting
- monotonic or parameter constraints when declared
- parameter search inside bounds

They should not silently rewrite splits, mutate targets, change evaluation policy, or broaden the learner surface.

### XGBoost-First

2.0 should focus on supervised tabular XGBoost workflows:

- binary classification
- multiclass classification
- regression

Regression should use its own metrics, readiness checks, scorer response shape, benchmark suites, and export manifest fields. It should not be bolted onto classification by pretending continuous targets are labels.

Time series, deep learning, and broad model zoos can wait.

### Export As Handoff

The export bundle should be inspectable and smoke-testable. It should not pretend to be a full production serving platform.

## Non-Goals

- broad AutoML
- autonomous data scientist behavior
- production model serving infrastructure
- notebook-only workflows
- general-purpose agent framework behavior
- benchmark claims without evidence
