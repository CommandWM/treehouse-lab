# Bounded Loop Contract

This file defines the bounded experiment loop. The filename is historical; the important contract is that Treehouse does not run open-ended AutoML or free-form agent code.

## Purpose

The loop turns one established incumbent into a short sequence of auditable candidate experiments.

It should:

1. read the dataset config, current incumbent, and recent journal history
2. generate eligible candidates from explicit mutation templates
3. select exactly one next proposal
4. run that proposal under the fixed split and evaluation policy
5. promote or reject the result deterministically
6. write metrics, artifacts, rationale, and decision reasons
7. stop when the budget or guardrails say to stop

## Inputs

- dataset config path
- search-space config
- current incumbent state
- recent journal entries
- loop step budget
- optional LLM selection setting

## Mutation Boundaries

Candidate proposals must come from declared mutation families, such as:

- `regularization_tighten`
- `learning_rate_tradeoff`
- `capacity_increase`
- `imbalance_adjustment`
- `feature_generation_enable`

Every candidate must map to explicit parameter or feature-generation edits. If the edit cannot be represented as a bounded proposal, it does not belong in the loop.

## Ranking And Selection

Treehouse can rank candidates deterministically. Optional LLM guidance can choose among eligible candidates only after deterministic candidate generation.

The selector should record:

- deterministic top candidate
- selected candidate
- whether LLM guidance was used
- whether selection changed mutation family
- rationale and grounding references

LLM guidance is exploration evidence, not a metric claim.

## Weak-Cycle Guard

If a mutation family repeatedly fails to promote or repeatedly produces weak outcomes, the loop should block that family for the current step when another eligible bounded candidate exists.

The journal and comparison report should show:

- blocked mutation family
- fallback family
- why the guard fired
- whether the fallback promoted

## Feature-Generation Gate

Feature generation stays conservative.

It can run only when:

- simpler parameter moves have already been tried or are low-value
- the plan is fit on train only
- validation, test, and exported inference reuse the fitted feature contract
- new feature count remains capped
- added complexity is compared against metric lift and readiness gates

## Promotion Contract

A candidate promotes only when:

- it improves the configured primary metric enough to beat the incumbent
- readiness checks pass
- the run stays within declared runtime and feature budgets
- no leakage or split-policy violation is detected

Rejected candidates still matter. They should leave enough evidence to explain why they did not replace the incumbent.

## Journal Contract

Each loop step should record:

- `proposal_id`
- `mutation_type`
- `hypothesis`
- `rationale`
- `params_override`
- `feature_generation`
- `incumbent_before`
- metrics and readiness checks
- promote/reject decision
- decision reason
- recommended next step when available

## Stop Conditions

The loop should stop when:

- max step count is reached
- no eligible proposal exists
- runtime budget is exhausted
- guardrails reject the remaining candidate space
- repeated weak cycles leave no useful bounded fallback

Stopping with a clear rationale is valid output.
