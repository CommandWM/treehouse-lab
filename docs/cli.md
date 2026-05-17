# CLI Contract

The CLI is the primary serious interface for Treehouse Lab. It should be useful to humans, scripts, CI, and coding agents.

## Implemented Commands

### `treehouse-lab baseline <config>`

Purpose: train and log the initial incumbent for a dataset spec.

Inputs:

- dataset config path

Outputs:

- JSON result payload on stdout
- run artifacts under `runs/`
- incumbent registry update when promotion criteria are met

### `treehouse-lab candidate <config> --name <name> --set KEY=VALUE ...`

Purpose: run one explicit bounded candidate mutation.

Inputs:

- dataset config path
- mutation name
- parameter overrides
- optional hypothesis

Outputs:

- JSON result payload
- run artifacts
- journal entry
- incumbent decision

### `treehouse-lab propose <config>`

Purpose: inspect the next bounded proposal without executing it.

Outputs:

- proposal JSON with mutation type, hypothesis, rationale, expected upside, risk, overrides, and grounding

### `treehouse-lab diagnose <config>`

Purpose: inspect current diagnosis and the next bounded proposal.

Outputs:

- diagnosis JSON
- proposal JSON
- current incumbent context where available

### `treehouse-lab loop <config> --steps N`

Purpose: run up to `N` bounded loop steps.

Outputs:

- loop summary JSON
- run artifacts for each executed step
- journal entries
- promotion/rejection decisions

### `treehouse-lab compare <config>`

Purpose: compare Treehouse against plain XGBoost and optional AutoML references on the same dataset contract.

Common options:

- `--loop-steps N`
- `--skip-autogluon`
- `--skip-flaml`
- `--llm-summary`
- `--autogluon-profile practical|full`

Outputs:

- comparison report artifacts
- JSON/structured summary payload
- benchmark and readiness interpretation

### `treehouse-lab benchmark-suite <suite>`

Purpose: run a fixed suite of dataset comparison configs.

Outputs:

- suite output directory
- per-dataset reports
- suite summary

### `treehouse-lab export <config>`

Purpose: package the incumbent or requested run as a model handoff artifact.

Common options:

- `--run-id`
- `--output-dir`

Outputs:

- export manifest
- `model_bundle.pkl`
- optional local scorer files

## Target 2.0 Workflow

The target 2.0 flow is:

```bash
treehouse init --dataset data/churn.csv --target churn --task classification
treehouse profile
treehouse baseline
treehouse loop --budget 5
treehouse compare
treehouse export --run best
treehouse serve exported/churn_model
```

Regression should use the same target command shape once implemented:

```bash
treehouse init --dataset data/housing.csv --target sale_price --task regression
treehouse profile
treehouse baseline
treehouse loop --budget 5
treehouse compare
treehouse export --run best
treehouse serve exported/housing_model
```

Current gaps:

- `init` is planned unless mapped to the current intake API/UI flow.
- `profile` is planned unless mapped to existing inspect/profile behavior.
- `--task regression` is planned; the current runner and intake path are classification-only.
- `serve` is planned unless kept as export-local `uvicorn app:app`.
- `--budget` is planned unless mapped to current `--steps`.
- `treehouse` is a target alias; current command is `treehouse-lab`.

Docs and agent task cards must not imply planned commands are implemented.

## JSON Output

Current successful commands print JSON to stdout. 2.0 should make the JSON contract explicit for both success and failure cases.

Expected stable fields should include, where applicable:

- command status
- config key
- run id
- metric summary
- artifact paths
- warnings
- errors
- promotion decision
- readiness decision

Failure output should be machine-readable and should avoid raw tracebacks by default once the failure contract is implemented.

## Agent Usage Rules

Agents should:

- call documented commands
- parse JSON output
- report artifact paths
- summarize promotion/rejection decisions
- avoid editing core code during operation tasks

Agents should not:

- change split policy
- mutate the target
- invent search-space changes
- bypass readiness checks
- claim benchmark wins without report artifacts
