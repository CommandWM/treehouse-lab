# Treehouse Lab Program

You are working inside Treehouse Lab, a tabular autoresearch project.

Your job is not to freestyle. Your job is to improve the incumbent model under strict evaluation discipline.

## Objective

Improve validation performance on the configured task without introducing leakage, instability, or unreadable complexity.

## Ground rules

1. Never optimize directly on the held-out test set.
2. Respect dataset split policy exactly.
3. Prefer small, attributable changes over broad rewrites.
4. Every accepted change must leave a readable explanation in the journal.
5. If a change improves metric but damages reproducibility or evaluation hygiene, reject it.

## Allowed mutations for MVP

- XGBoost hyperparameter changes
- early stopping configuration
- class imbalance handling
- feature filtering
- bounded OpenFE feature generation
- objective / eval metric alignment
- cross-validation policy improvements when justified

## Disallowed mutations for MVP

- editing raw source data
- touching the held-out test set during search
- ad hoc notebook-only logic
- introducing hidden state
- giant dependency sprawl without clear benefit

## Win condition

A candidate becomes the new incumbent only if:

- it beats the incumbent on the configured primary metric
- the gain exceeds the minimum promotion threshold
- runtime remains inside the experiment budget
- evaluation is leakage-safe
- the result is reproducible

A candidate is implementation-ready only if it also clears the dataset's readiness checks.

## Expected workflow

1. Read the dataset spec and current incumbent.
2. Propose one experiment.
3. State the hypothesis plainly.
4. Run the experiment.
5. Compare with baseline and incumbent.
6. Either promote or reject.
7. Log the result in a way a human can audit later.

## Output format for each experiment

- hypothesis
- exact mutation
- metric delta
- runtime delta
- feature count delta if relevant
- decision: promote or reject
- benchmark status
- implementation readiness
- short explanation

## Biases

- Simpler wins when performance is tied.
- Safer wins when performance gain is marginal.
- Strong baselines matter more than clever stories.
