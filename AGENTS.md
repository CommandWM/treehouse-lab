# AGENTS.md

This repo is for disciplined tabular autoresearch, not vibes.

## Repository Intent

Treehouse Lab is an audit-first, contract-first workbench for reproducible tabular ML experiments, starting with XGBoost-first classification and expanding to XGBoost regression under the same contracts.

The 2.0 direction is not "an agent" and not broad AutoML. Treehouse should help a human, CLI user, API client, platform, or coding agent run bounded, auditable experiments and hand off a scoring-ready model package with a clear contract.

## Rules For Coding Agents

- Keep the search space explicit.
- Do not introduce test-set leakage.
- Prefer small, reviewable diffs.
- Explain why a change should improve the incumbent before implementing it.
- If a mutation is speculative, keep it isolated behind an explicit template or contract.
- Do not add heavyweight dependencies casually.
- Do not claim benchmark superiority without fresh benchmark artifacts, environment notes, commit evidence, and report outputs.
- Use documented CLI, API, export, and artifact contracts before reaching into internals.
- Treat LLM or agent assistance as optional, bounded, and contract-driven.

## Validation Expectations

- Run `ruff check .` for Python style changes when relevant.
- Run `python -m pytest` for behavior, docs-contract, export, or API changes when feasible.
- For narrow changes, run the smallest focused pytest target first, then broaden if shared behavior is touched.
- If tests are skipped, state why and what remains unverified.

## 2.0 Priorities

1. Stable CLI workflow and machine-readable outputs for classification first, then regression.
2. Explicit API, export, manifest, journal, and benchmark contracts.
3. Evaluation policy, split invariance, leakage guardrails, and promotion/rejection tests.
4. Honest benchmark reporting against plain XGBoost and selected AutoML references.
5. Agent-accessible usage through documented commands, schemas, reports, and task cards.

## Avoid

- notebook-only workflows
- silent split changes
- silent target or feature-set changes
- magic defaults with no documentation
- broad rewrites without measurable benefit
- free-form agent mutation outside declared templates
- production-serving claims for the exported local scorer
