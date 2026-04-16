# AGENTS.md

This repo is for disciplined tabular autoresearch, not vibes.

## Repository intent

Build a Karpathy-style experiment loop for tabular ML, starting with XGBoost.

## Rules for coding agents

- Keep the search space explicit.
- Do not introduce test-set leakage.
- Prefer small, reviewable diffs.
- Explain why a change should improve the incumbent before implementing it.
- If a mutation is speculative, keep it isolated.
- Do not add heavyweight dependencies casually.

## Early priorities

1. Baseline runner
2. Evaluation policy
3. Incumbent registry
4. Journal and reporting
5. Bounded agent mutation templates

## Avoid

- notebook-only workflows
- silent split changes
- magic defaults with no documentation
- broad rewrites without measurable benefit
