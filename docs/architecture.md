# Architecture Sketch

## Components

### 1. Dataset spec
Defines:
- target column
- primary metric
- split policy
- categorical / numeric handling
- leakage constraints
- runtime budget

### 2. Baseline trainer
Responsible for:
- loading data
- applying the split policy
- training the incumbent baseline
- logging metrics and artifacts

### 3. Evaluator
Responsible for:
- running candidate experiments
- enforcing reproducibility
- checking runtime budget
- comparing to incumbent
- labeling implementation readiness separately from benchmark wins

### 4. Mutation engine
For MVP, mutations should be templates rather than free-form code surgery:
- hyperparameter mutation
- feature budget mutation
- OpenFE on or off
- imbalance strategy mutation

### 5. Incumbent registry
Stores:
- best known config
- score
- run id
- promotion history
- latest readiness assessment

### 6. Journal
Each run should capture:
- hypothesis
- config diff
- metrics
- artifacts
- promote or reject decision
- implementation-ready or not-ready decision
- human-readable explanation

## Guiding constraint

This system should feel more like a research lab notebook with guardrails than an unconstrained code-writing agent.

That constraint is the whole point.
