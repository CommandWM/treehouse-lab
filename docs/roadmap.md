# Roadmap

## Current position

Treehouse Lab `v1.1.0` is no longer a pre-MVP sketch, and v1.2 Product Polish is closed.

The repo already has:

- dataset-first intake and generated dataset specs
- a bounded XGBoost-first baseline / candidate / diagnose / propose / loop flow
- incumbent promotion and human-readable run summaries
- a React workbench for intake, current state, journal inspection, settings, and architecture
- a capped train-only feature-generation branch for plateaued loops

That means the roadmap should now optimize for real benchmark evidence and bounded search quality, not breadth.

## v1.2 Product Polish Closeout

v1.2 Product Polish is closed as a shareability and auditability layer around the shipped v1.1 product. The durable closeout is [v1.2 Product Polish](v1-2-polish.md).

Closed Linear scope:

- `COM-5`: walkthrough and screenshot review path
- `COM-6`: public dataset paths for `bank_marketing_uci`, `adult_uci`, and `covertype_uci`
- `COM-7`: sample outputs for baseline, proposal, journal, and compare report
- `COM-8`: feature-generation decision visibility
- `COM-9`: benchmark-better versus implementation-ready clarity across docs and workbench
- `COM-18`: final README positioning, version consistency, export contract, scorer tests, and benchmark report example

What remains after v1.2 is evidence generation, not polish: run the fixed public suite, fill real report outputs, then continue bounded XGBoost search depth.

## What `2.0` should be

Version `2.0` should be the release where Treehouse Lab proves why it exists.

The core move is not "add more learners." The core move is:

- stay XGBoost-first
- benchmark honestly against strong external baselines
- show that Treehouse Lab adds value through bounded search, promotion policy, artifacts, and auditability

## `2.0` priorities

### 1. XGBoost-first benchmark harness

Build a small public dataset suite with fixed seeds, split policy, and runtime budgets.

Compare:

- plain XGBoost baseline
- Treehouse Lab baseline
- Treehouse Lab bounded loop
- FLAML
- AutoGluon

Report:

- primary metric
- runtime
- artifact quality
- interpretability / auditability

The output should answer a simple product question:

- when should someone use Treehouse Lab instead of FLAML or AutoGluon?

### 2. Better bounded search, not a broader learner matrix

Deepen the current loop without relaxing the rules:

- bounded Optuna search inside `configs/search_space.yaml`
- richer mutation templates with explicit validation
- stronger feature-generation audit trails
- tighter candidate-vs-incumbent comparison

The point is to improve the quality of the existing bounded loop, not to chase breadth.

### 3. Strong public demo quality

Make the repo easy to understand and share:

- keep the v1.2 walkthrough, screenshots, sample outputs, export contract, and benchmark report example current
- replace benchmark report placeholders with real fixed-suite outputs before making external claims
- package the evidence into a clear 2.0 positioning story

## Explicit non-goals for `2.0`

These are intentionally not the focus:

- CatBoost or LightGBM expansion
- open-ended model-zoo work
- unconstrained agent code rewriting
- giant public benchmark suites across dozens of datasets
- production-serving platform work

## How the current GitHub roadmap issues map now

The open GitHub roadmap issues are useful, but several describe work that is already shipped in whole or in part.

### Effectively shipped or mostly absorbed by `v1.1`

- issue `#3`: dataset spec and first benchmark dataset
- issue `#4`: reproducible split handling and leakage checks
- issue `#5`: runnable baseline XGBoost command
- issue `#6`: run logging and artifact persistence
- issue `#7`: incumbent registry and promotion policy
- issue `#8`: human-readable journal
- issue `#10`: Codex-safe mutation templates, at least for the current bounded loop

### Still active, but should be reframed

- issue `#9`: bounded Optuna search should deepen the current XGBoost-first loop, not broaden the model surface
- issue `#11`: benchmark against FLAML and AutoGluon should become a centerpiece of `2.0`
- issue `#12`: public-demo polish should support the benchmark-and-positioning story, not just repo cosmetics

## Product stance

Treehouse Lab should feel like a disciplined lab notebook with execution attached.

If the project keeps that identity, `2.0` can become a credible answer to a real question:

- not "can we add more models?"
- but "can we make tabular autoresearch reviewable, bounded, and worth using?"
