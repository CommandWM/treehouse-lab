# Benchmark Report Example

This page is a concrete report structure for comparing Treehouse Lab against plain XGBoost and optional AutoML references.

It is intentionally a template. The current checkout does not include a generated `outputs/benchmark_suites/` result for the fixed public suite, so all metric cells below are placeholders. Replace every `TBD` cell with values from a real run before using this as a benchmark claim.

Use this page with:

```bash
treehouse-lab benchmark-suite configs/benchmark_suites/public_v1_3.yaml --skip-autogluon --skip-flaml
```

Or, after preparing optional references:

```bash
./scripts/setup_benchmark_env.sh
.venv-benchmarks/bin/python -m treehouse_lab.cli benchmark-suite configs/benchmark_suites/public_v1_3.yaml
```

## Example Summary

Report date: `TBD`

Suite: `configs/benchmark_suites/public_v1_3.yaml`

Run directory: `outputs/benchmark_suites/TBD`

Commit: `TBD`

Environment notes:

- Python: `TBD`
- XGBoost: `TBD`
- AutoGluon: `TBD` or `unavailable`
- FLAML: `TBD` or `unavailable`
- Fixed seed: `42`
- Loop steps: `3` by default, except suite-level overrides

## Executive Read

Treehouse Lab should be judged on two axes:

- Metric position: did the Treehouse baseline or bounded loop beat plain XGBoost under the same split and metric policy?
- Operating quality: did the run leave enough evidence for a reviewer to understand the promotion decision, rejected work, readiness gates, and next bounded step?

The honest claim should be narrow:

- Treehouse wins when it produces a competitive or better score while also leaving an auditable journal, bounded proposal history, explicit promotion decision, and implementation-readiness assessment.
- Treehouse may lose when a plain XGBoost baseline is already strong enough, when AutoGluon or FLAML finds a materially better score under the same budget, or when the team only cares about raw leaderboard position.

Do not publish a blanket claim that Treehouse Lab beats AutoML. This report should explain when Treehouse is the better workflow, not pretend it is always the highest-scoring runner.

## Dataset Results

Fill this table from `outputs/benchmark_suites/TBD/summary.json` and each dataset-level `report.md`.

| dataset | primary metric | plain XGBoost test | Treehouse baseline test | Treehouse loop test | AutoGluon test | FLAML test | metric read |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `bank_marketing_uci` | `roc_auc` | `TBD` | `TBD` | `TBD` | `TBD` or `unavailable` | `TBD` or `unavailable` | `TBD: explain winner and margin without overclaiming` |
| `adult_uci` | `roc_auc` | `TBD` | `TBD` | `TBD` | `TBD` or `unavailable` | `TBD` or `unavailable` | `TBD: explain winner and margin without overclaiming` |
| `covertype_uci` | `accuracy` | `TBD` | `TBD` | `TBD` | `TBD` or `unavailable` | `TBD` or `unavailable` | `TBD: explain winner and margin without overclaiming` |

## Runtime And Practicality

Runtime matters because a small metric lift is not automatically useful if it requires a much heavier workflow.

| dataset | plain XGBoost runtime | Treehouse loop runtime | AutoGluon runtime | FLAML runtime | practical read |
| --- | --- | --- | --- | --- | --- |
| `bank_marketing_uci` | `TBD` | `TBD` | `TBD` or `unavailable` | `TBD` or `unavailable` | `TBD: was the extra runtime justified by metric or audit value?` |
| `adult_uci` | `TBD` | `TBD` | `TBD` or `unavailable` | `TBD` or `unavailable` | `TBD: was the extra runtime justified by metric or audit value?` |
| `covertype_uci` | `TBD` | `TBD` | `TBD` or `unavailable` | `TBD` or `unavailable` | `TBD: did capped loop steps and external budgets stay practical?` |

## Workflow Comparison

This is where Treehouse Lab can win even when the raw score is close.

| runner | search style | auditability | promotion policy | artifact quality | implementation readiness |
| --- | --- | --- | --- | --- | --- |
| Plain XGBoost | single explicit baseline | low to medium; depends on saved logs | none beyond manual review | minimal model and metric outputs | manual judgment required |
| Treehouse Lab baseline | explicit XGBoost incumbent | high; run summary, assessment, and incumbent registry | baseline establishes the comparison point | structured local artifacts | readiness gates recorded |
| Treehouse Lab bounded loop | bounded candidate mutations | high; proposals, journal entries, rejected runs, and loop summaries stay inspectable | promotes only meaningful improvements under configured guardrails | highest-detail Treehouse artifact trail | readiness is separate from benchmark improvement |
| AutoGluon practical | broad automated model search | medium to low; useful model artifacts but less explicit search trace | internal AutoML selection | external AutoML model directory | implementation-readiness policy must be reviewed outside Treehouse |
| FLAML budgeted | budgeted automated search | medium; explicit budget and estimator list, less detailed decision trail | internal AutoML selection | external AutoML model output | implementation-readiness policy must be reviewed outside Treehouse |

## Where Treehouse Wins

Treehouse Lab is a strong fit when the report can show:

- the bounded loop matches or improves on plain XGBoost under the same split policy
- the promoted run has a clear reason code and does not rely on test-set tuning
- rejected mutations remain visible in the journal instead of disappearing
- the next step is constrained by `configs/search_space.yaml`
- feature generation, if used, is train-only, capped, and explained
- implementation readiness is reported separately from benchmark status
- exported artifacts are complete enough for another person to inspect or rerun

The product win is not just "higher score." The product win is a credible score with a readable path to the decision.

## Where Treehouse May Lose

Treehouse Lab should concede the comparison when:

- plain XGBoost is equal or better and the project does not need the extra audit trail
- AutoGluon or FLAML produces a materially better metric within the same practical budget
- the dataset requires model families or feature semantics outside the current bounded XGBoost-first surface
- the target is regression, hosted training, streaming inference, or production orchestration
- the bounded loop stops without finding a useful improvement and the artifacts do not clarify a better next move

Those losses are useful. They keep Treehouse Lab's product scope legible and make the next product slice easier to prioritize.

## Promotion And Readiness

For each dataset, include the Treehouse promotion decision from the generated dataset report.

| dataset | Treehouse benchmark status | implementation readiness | reason codes | decision |
| --- | --- | --- | --- | --- |
| `bank_marketing_uci` | `TBD` | `TBD` | `TBD` | `TBD: promote, reject, or keep investigating` |
| `adult_uci` | `TBD` | `TBD` | `TBD` | `TBD: promote, reject, or keep investigating` |
| `covertype_uci` | `TBD` | `TBD` | `TBD` | `TBD: promote, reject, or keep investigating` |

Use plain language:

- `better_than_incumbent` is a benchmark result, not an automatic implementation decision.
- `implementation_ready` means the run cleared the current policy gates, not that it is production-deployed.
- `not_better_than_incumbent` can still be useful if the journal explains why the mutation failed and what bounded move remains.

## Evidence To Attach

A real report should link or quote:

- `outputs/benchmark_suites/TBD/summary.json`
- each dataset-level `report.md`
- the Treehouse run summary for the promoted incumbent
- the relevant journal entries for rejected candidates
- any loop summary under `runs/loops/`
- exported model bundle paths if implementation readiness is claimed

## Final Interpretation Template

Use this structure after real numbers are inserted:

```md
Treehouse Lab was strongest on `TBD` because `TBD`.

Plain XGBoost remained the better choice on `TBD` because `TBD`.

AutoGluon or FLAML was the better raw-metric reference on `TBD` because `TBD`.

The benchmark result supports using Treehouse Lab when `TBD`, but it does not support claiming `TBD`.

The next benchmark work should be `TBD`.
```

## Where Real Results Go

When the fixed public suite has been run, replace:

- every `TBD` metric and runtime cell in "Dataset Results" and "Runtime And Practicality"
- the run directory, commit, and package versions in "Example Summary"
- the promotion status, readiness status, and reason codes in "Promotion And Readiness"
- the narrative placeholders in "Final Interpretation Template"

Keep the workflow comparison even after real numbers are added. It is the part of the report that explains why Treehouse Lab may be worth using when scores are close, and why it may not be worth using when raw metric performance is the only goal.
