# MVP Plan

## Goal

Build a disciplined autoresearch loop for single-table tabular classification with XGBoost, starting with binary and multiclass problems.

## Phase 1: Baseline runner

Deliverables:
- dataset config format
- reproducible train/validation/test split handling
- baseline XGBoost trainer
- MLflow run logging
- simple artifact bundle with config, metrics, and feature importances

Exit criteria:
- one command can train and log a baseline run end to end

## Phase 2: Search and promotion

Deliverables:
- Optuna-driven hyperparameter search
- incumbent registry
- promotion threshold policy
- experiment comparison report
- failure-safe experiment journal

Exit criteria:
- system can propose and evaluate multiple candidates and select a winner

## Phase 3: Agent loop

Deliverables:
- agent-facing `program.md`
- constrained mutation templates
- experiment proposal schema
- automated promote/reject logging

Exit criteria:
- Codex or Claude Code can run bounded experiments without trampling evaluation policy

## Phase 4: Feature generation

Deliverables:
- optional OpenFE integration
- feature budget caps
- leakage-safe transform policy
- side-by-side comparison of baseline vs feature-augmented runs

Exit criteria:
- generated features can help when useful without turning the project into a science fair

## Phase 5: Shareability

Deliverables:
- benchmark pack with smoke, stress, and implementation-like examples
- clean README walkthrough
- reproducible quickstart
- screenshots or sample experiment logs

Exit criteria:
- someone can clone the repo, run the benchmark pack, and understand the promote-vs-ready distinction in under ten minutes

## Non-goals for v1

- general LLM literature review automation
- distributed training
- time series search
- multimodal tabular problems
- production deployment hooks
- automatic paper writing or blog generation
