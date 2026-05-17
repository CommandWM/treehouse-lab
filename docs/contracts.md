# Contracts

Treehouse Lab 2.0 should expose stable, boring contracts. This contract-first and audit-first posture is what makes the project usable by humans, scripts, downstream systems, and coding agents.

## CLI Contract

Owner: [CLI Contract](cli.md)

The CLI contract should document:

- command purpose
- required and optional arguments
- expected artifacts
- success output
- failure output
- exit-code expectations
- whether JSON output is stable

Current commands include `baseline`, `candidate`, `propose`, `diagnose`, `loop`, `compare`, `benchmark-suite`, and `export`.

## API Contract

Owner: planned `docs/api.md`

The API contract should distinguish:

- local workbench API endpoints
- exported scorer endpoints
- request body shapes
- response body shapes
- error payloads
- OpenAPI/Swagger/ReDoc availability where implemented

Do not claim an endpoint exists until it is implemented and tested.

## Export Contract

Owner: [Export Contract](export-contract.md)

The export contract documents:

- expected export files
- `manifest.json` fields
- scorer `/health`, `/schema`, and `/predict` behavior
- Python bundle loading
- local scorer limitations
- production-serving non-goals

2.0 should add schemas for manifest, prediction request, and prediction response payloads.

## Evaluation Contract

Owner: [Evaluation Policy](evaluation-policy.md)

The evaluation contract covers:

- split policy
- target handling
- leakage guardrails
- primary metric handling
- classification versus regression metric handling
- readiness checks
- promotion threshold behavior
- benchmark-better versus implementation-ready decisions

Split, target, and metric changes require explicit review.

## Loop Contract

Owner: [Bounded Loop Contract](autonomous-loop.md)

The loop contract covers:

- candidate generation
- mutation bounds
- deterministic ranking
- optional LLM selection
- weak-cycle guardrails
- promotion/rejection decisions
- journal output
- stop conditions

## Benchmark Contract

Owners: [Benchmark Pack](benchmarks.md), [Benchmark Report](benchmark-report.md)

The benchmark contract should report:

- dataset summary
- task family
- split policy
- comparison targets
- metric table
- runtime/budget table
- promotion/rejection summary
- implementation-readiness summary
- honest conclusion

Never publish fake benchmark numbers or blanket superiority claims.

## Agent Contract

Owner: planned `docs/agent-usage.md`

Agent-accessible usage should define:

- allowed commands
- forbidden behaviors
- JSON consumption rules
- task-card setup
- expected artifacts
- pass/fail criteria

Agents should consume public contracts. They should not get special hidden behavior.
