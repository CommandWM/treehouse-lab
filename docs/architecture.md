# Architecture

Treehouse Lab should be deterministic and contract-driven first. Agent access, UI affordances, and API wrappers sit on top of the same core surfaces.

## Layering

The intended hierarchy is:

1. Core engine
2. CLI
3. Local workbench API
4. Exported scorer package
5. Agent-accessible command/API wrapper
6. UI

The lower layers own behavior. The upper layers should expose that behavior without changing split policy, target handling, mutation boundaries, or promotion rules.

## Core Engine

The Python engine owns:

- dataset config loading
- split policy
- preprocessing
- baseline and candidate execution
- evaluation metrics
- readiness checks
- incumbent promotion
- journal and artifact writing
- export bundle generation

The engine should not depend on the React UI, a hosted service, or an LLM.

## Dataset Contract

Dataset specs define:

- dataset key and source path
- target column
- task type
- task family: classification or regression
- primary metric
- split policy
- categorical and numeric handling
- leakage constraints
- runtime and feature budgets

Split policy changes are review-sensitive. They should not happen silently inside a candidate, loop, UI action, or agent task.

## Mutation Contract

Mutations are templates, not free-form code surgery.

Allowed mutation families should map to declared search-space bounds and explicit runtime behavior, such as:

- regularization tightening
- learning-rate tradeoff
- capacity changes
- imbalance handling
- capped train-only feature generation

Every mutation should record:

- what changed
- why it was tried
- which bounds allowed it
- what evidence supported it
- whether it was promoted or rejected

## Evaluation Contract

Evaluation owns:

- train/validation/test separation
- target and proxy leakage checks
- primary metric calculation
- task-specific metric policy for classification versus regression
- train-validation gap checks
- readiness checks
- promotion threshold handling

Benchmark-better and implementation-ready are separate decisions. A run can improve the incumbent metric and still fail readiness checks.

## Artifact Contract

Runs should leave durable artifacts:

- metrics
- params
- proposal/rationale
- run narrative
- journal entry
- incumbent decision
- model bundle when available
- export manifest when exported

Artifacts are not just logs. They are the audit trail that makes the experiment reviewable.

## Interface Contracts

The public interfaces are:

- CLI commands and JSON output
- local workbench API request/response shapes
- exported scorer request/response shapes
- export manifest fields
- benchmark report sections
- agent task-card inputs and outputs

These contracts should be documented before downstream tools depend on them.

## Agent Boundary

Agents can operate Treehouse through documented commands and APIs. They should not:

- rewrite splits
- mutate targets
- change evaluation policy
- invent undeclared features
- broaden the learner matrix
- claim benchmark wins without artifacts

The project becomes agent-accessible by exposing boring, typed, documented surfaces.
