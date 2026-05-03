# Autonomous Loop Plan

## Purpose

Treehouse Lab now has a runnable MVP slice:

- explicit dataset specs
- baseline and bounded candidate execution
- artifact bundles
- dataset-aware incumbent tracking
- run journaling
- a teaching UI

The next step is to turn that single-step flow into a disciplined multi-step research loop.

The goal is not open-ended AutoML. The goal is a bounded loop that:

1. reads the current incumbent and journal
2. proposes one justified next experiment
3. runs the experiment inside guardrails
4. promotes or rejects the result
5. writes a narrative a human can audit
6. decides what to try next

## Product Shape

The full cycle should feel like a lab notebook with execution attached, not a notebook pile and not an unconstrained agent.

The user-facing story is:

- start with a strong incumbent
- run one bounded experiment at a time
- explain every change before it runs
- keep only meaningful wins
- leave behind a readable research trail

## Proposed Modules

The next implementation pass should add the following modules under `src/treehouse_lab/`.

### `proposals.py`

Owns the structured proposal object for one experiment.

Core types:

- `ExperimentProposal`
- `ProposalDecisionContext`
- `ProposalRisk`

Suggested fields for `ExperimentProposal`:

- `proposal_id`
- `dataset_key`
- `mutation_type`
- `mutation_name`
- `hypothesis`
- `rationale`
- `expected_upside`
- `risk_level`
- `params_override`
- `feature_generation`
- `depends_on_run_id`
- `stage`
- `grounding`

Key functions:

- `build_baseline_proposal(config) -> ExperimentProposal`
- `build_mutation_proposal(...) -> ExperimentProposal`
- `proposal_to_dict(proposal) -> dict`

### `mutations.py`

Owns the mutation template library and the logic for generating bounded overrides.

Core types:

- `MutationTemplate`
- `MutationCandidate`

Recommended template families:

- `regularization_tighten`
- `capacity_increase`
- `learning_rate_tradeoff`
- `imbalance_adjustment`
- `feature_filtering`
- `feature_generation_enable`
- `objective_metric_alignment`

Key functions:

- `list_templates(stage: str) -> list[MutationTemplate]`
- `generate_candidates(config, incumbent, journal) -> list[MutationCandidate]`
- `apply_template(template, incumbent_params, search_space, seed) -> dict`

Rules:

- every template must map to explicit parameter edits
- templates must respect `configs/search_space.yaml`
- a template can be skipped if its preconditions are not met
- every generated proposal should include bounded local grounding with the measured evidence and repo references used to justify the mutation

### `loop.py`

Owns the orchestration layer for multi-step runs.

Core types:

- `LoopConfig`
- `LoopStepResult`
- `LoopSummary`

Key class:

- `AutonomousLoopController`

Suggested methods:

- `run_loop(config_path, max_steps=3) -> LoopSummary`
- `ensure_incumbent() -> ExperimentResult`
- `choose_next_proposal() -> ExperimentProposal | None`
- `run_proposal(proposal) -> ExperimentResult`
- `should_stop(history) -> bool`

Stopping conditions:

- reached max step count
- no eligible next proposal
- last `N` proposals all rejected with low upside
- runtime budget exhausted

### `narratives.py`

Owns the readable experiment story for each run and for the loop as a whole.

Core types:

- `RunNarrative`
- `LoopNarrative`

Key functions:

- `build_run_narrative(proposal, result, incumbent_before) -> RunNarrative`
- `build_loop_summary(history) -> LoopNarrative`
- `render_markdown(narrative) -> str`

Each run narrative should answer:

- what changed
- why it was tried
- what happened
- whether it was promoted
- what should be tried next

### `features.py`

Owns the feature-generation stage when we enable it.

Core types:

- `FeatureGenerationPlan`
- `FeatureGenerationResult`

Key functions:

- `should_enable_feature_generation(history, incumbent) -> bool`
- `build_feature_plan(config, bundle) -> FeatureGenerationPlan`
- `run_feature_plan(plan, split) -> FeatureGenerationResult`

Guardrails:

- feature generation is off until simpler parameter moves plateau
- fit transforms on train only
- apply to validation/test without leakage
- cap new feature count
- compare complexity increase against metric lift

## Data Flow

The control path should look like this:

1. Load dataset config and search space.
2. Load incumbent for that dataset.
3. Load recent journal entries for that dataset.
4. Build proposal context.
5. Generate eligible mutation candidates.
6. Rank the candidates.
7. Select exactly one proposal.
8. Run it through the existing execution layer.
9. Compare to incumbent.
10. Write narrative and journal entry.
11. Repeat or stop.

## Ranking Logic

The loop does not need an LLM to rank the first generation of proposals. A deterministic scorer is enough for the first useful version.

Suggested heuristics:

- prefer simple parameter-only changes first
- prefer proposals that address observed overfitting
- prefer proposals with high attribution and low complexity
- delay feature generation until parameter templates stop producing wins

Signals the scorer can use:

- train vs validation gap
- incumbent plateau
- recent rejection streak
- previous template family outcomes

If a mutation family has repeated recent non-promoting attempts below the promotion threshold, the loop applies a weak-cycle guard before execution. The guard blocks that repeated family for the current choice and selects the first different bounded candidate when one is available. The selected proposal records a `cycle_guard` payload so the journal and comparison report show which family was blocked, which fallback ran, and why.

## First 3-Step Autonomous Loop

The first real loop should run exactly three steps after baseline establishment.

### Step 0: Establish baseline

- if no incumbent exists, run baseline
- write baseline narrative

### Step 1: Regularization pass

Intent:
- test whether the incumbent is too expressive

Likely edits:
- lower `max_depth`
- increase `min_child_weight`
- reduce `subsample`
- reduce `colsample_bytree`

### Step 2: Learning-rate tradeoff

Intent:
- test whether a slower learner with more trees improves stability

Likely edits:
- lower `learning_rate`
- increase `n_estimators`

### Step 3: Capacity or class-balance adjustment

Intent:
- choose based on the evidence from the first two experiments

If the model is still underfitting:
- increase capacity moderately

If class imbalance is visible:
- test `scale_pos_weight`

The loop should end with:

- current incumbent
- ordered run history
- promoted vs rejected experiments
- recommended next experiment

## Journal Changes

The journal should keep the existing raw result fields and add proposal-aware fields.

Add to each entry:

- `proposal_id`
- `mutation_type`
- `stage`
- `rationale`
- `expected_upside`
- `risk_level`
- `incumbent_before`
- `recommended_next_step`

This keeps the journal useful both for the UI and for future autonomous selection.

## UI Changes

The Streamlit app should move from "run one candidate" to "watch the research chain."

Recommended additions:

- `Loop` tab
- proposal preview card
- current incumbent panel
- timeline of experiments
- promote/reject badges
- markdown narrative for each run
- loop summary panel

The UI should answer these questions quickly:

- why this experiment was chosen
- what changed
- did it win
- what should happen next

## Suggested File Ownership

This is the minimum next code slice:

- `src/treehouse_lab/proposals.py`
- `src/treehouse_lab/mutations.py`
- `src/treehouse_lab/loop.py`
- `src/treehouse_lab/narratives.py`
- `src/treehouse_lab/features.py`

Likely updates:

- `src/treehouse_lab/runner.py`
- `src/treehouse_lab/journal.py`
- `src/treehouse_lab/cli.py`
- `app.py`

## CLI Expansion

The CLI should grow from single-run commands into loop-aware commands.

Add:

```bash
treehouse-lab loop configs/datasets/breast_cancer.yaml --steps 3
treehouse-lab propose configs/datasets/churn_demo.yaml
```

Expected behavior:

- `propose` shows the next bounded experiment without executing it
- `loop` runs the next `N` experiments with narratives and stop rules

## Feature Generation Stage Gate

Feature generation should not be available during the first loop pass by default.

Enable it only if:

- at least two parameter-only proposals have been evaluated
- recent improvements are below the promotion threshold
- the incumbent still appears improvable

This keeps the first autonomous loop interpretable.

## Recommended Build Order

Implement in this order:

1. proposal schema
2. mutation template library
3. deterministic proposal ranking
4. loop controller with `max_steps`
5. run narratives
6. `loop` CLI command
7. Streamlit loop timeline
8. feature-generation stage gate

## Exit Criteria For This Phase

This phase is done when:

- one command runs a 3-step bounded research loop
- each step has a readable proposal and narrative
- the incumbent updates correctly by dataset
- the UI can show the research chain
- the loop never edits code or data outside the declared policy

That gets Treehouse Lab from "single experiment demo" to "disciplined autoresearch system."
