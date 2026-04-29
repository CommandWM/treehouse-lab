# Sample Outputs

Treehouse Lab is easier to evaluate when the output shape is visible before you run the full loop yourself.

This page captures representative real outputs from the current product surface:

1. baseline
2. bounded proposal
3. journal entry
4. compare report

These are intentionally excerpts rather than full raw artifacts. The goal is to show what the product leaves behind and how to read it.

## Baseline

Command:

```bash
treehouse-lab baseline configs/datasets/adult_uci.yaml
```

Representative summary excerpt:

```md
# baseline

- run_id: `20260422T034923061630Z-baseline`
- benchmark_pack: `optional`
- benchmark_profile: `external_probe`
- backend: `xgboost`
- primary_metric: `roc_auc`
- validation_roc_auc: `0.9302`
- runtime_seconds: `0.64`
- decision: `promote`

## Assessment

- benchmark_status: `baseline_established`
- benchmark_summary: Baseline established at validation roc_auc 0.9302.
- implementation_readiness: `implementation_ready`
- promotion_bar: `True`
- minimum_primary_metric: `True`
- train_validation_gap: `True`
- validation_test_gap: `True`
- runtime_budget: `True`
- feature_budget: `True`

## Diagnosis

- primary_tag: `class_imbalance`
- summary: Validation roc_auc is 0.9302. Positive rate is 0.2392, so imbalance handling is worth considering.
- recommended_direction: Test bounded class weighting before broader changes.
```

What this shows:

- the baseline is not just a score dump
- the run gets a benchmark decision and an implementation decision
- the next bounded direction is already explicit

## Proposal

Command:

```bash
treehouse-lab propose configs/datasets/adult_uci.yaml
```

Representative excerpt:

```json
{
  "dataset_key": "adult_uci",
  "depends_on_run_id": "20260422T034923061630Z-baseline",
  "mutation_name": "imbalance-adjustment",
  "mutation_type": "imbalance_adjustment",
  "stage": "parameter_tuning",
  "risk_level": "medium",
  "params_override": {
    "scale_pos_weight": 3.18
  },
  "hypothesis": "Explicit positive-class weighting may improve ranking quality on skewed targets.",
  "expected_upside": "Improved ranking on the positive class without changing the dataset split policy.",
  "rationale": "Diagnosis: Validation roc_auc is 0.9302. Positive rate is 0.2392, so imbalance handling is worth considering. The positive-class rate is 0.239, which is far enough from parity to justify a bounded class-balance adjustment.",
  "llm_review": {
    "candidate_count": 4,
    "message": "LLM loop selection is disabled, so Treehouse Lab used deterministic candidate ranking.",
    "status": "disabled"
  }
}
```

What this shows:

- the proposal is tied to a specific incumbent
- the exact mutation stays explicit
- the rationale stays inside the declared search space
- LLM review, when present, is visible rather than hidden

## Journal

Representative journal-style entry excerpt from a rejected bounded step on `smoke_breast_cancer`:

```json
{
  "name": "capacity-increase",
  "metric": 0.9911562397641664,
  "promoted": false,
  "assessment": {
    "benchmark_status": "not_better_than_incumbent",
    "implementation_readiness": "needs_more_work"
  },
  "diagnosis": {
    "primary_tag": "plateau",
    "summary": "Validation roc_auc is 0.9912. Recent deltas are small, so the loop may be plateauing."
  },
  "decision_reason": "Validation roc_auc changed by -0.0007, which did not clear the promotion threshold.",
  "proposal": {
    "mutation_name": "capacity-increase",
    "params_override": {
      "max_depth": 5,
      "min_child_weight": 1,
      "n_estimators": 288
    }
  },
  "reason_codes": [
    "diagnosis_plateau",
    "failed_promotion_bar",
    "passed_feature_budget",
    "passed_minimum_primary_metric",
    "passed_runtime_budget",
    "passed_train_validation_gap",
    "passed_validation_test_gap",
    "rejected_below_threshold"
  ]
}
```

What this shows:

- the journal keeps rejected work, not just winners
- a run can still clear most checks and remain `needs_more_work`
- the rejection reason and the exact proposal stay attached to the entry

## Compare Report

Command:

```bash
.venv-benchmarks/bin/python -m treehouse_lab.cli compare \
  configs/datasets/bank_marketing_uci.yaml \
  --loop-steps 1 \
  --autogluon-profile practical
```

Representative report excerpt:

```md
## Results

| runner | status | validation | test | runtime_s | readiness | benchmark_status |
| --- | --- | --- | --- | --- | --- | --- |
| Plain XGBoost Baseline | completed | 0.9434 | 0.9343 | 0.36 | implementation_ready | baseline_established |
| Treehouse Lab Baseline | completed | 0.9434 | 0.9343 | 1.25 | implementation_ready | baseline_established |
| Treehouse Lab 1-Step Loop | completed | 0.9434 | 0.9343 | 1.15 | implementation_ready | baseline_established |
| AutoGluon Tabular (Practical) | completed | 0.9461 | 0.9371 | 18.73 | implementation_ready | baseline_established |

## Outcome gates

| runner | benchmark decision | implementation decision | quick read |
| --- | --- | --- | --- |
| Plain XGBoost Baseline | baseline_established | implementation_ready | Strong result: credible benchmark position and ready under current policy. |
| Treehouse Lab Baseline | baseline_established | implementation_ready | Strong result: credible benchmark position and ready under current policy. |
| Treehouse Lab 1-Step Loop | baseline_established | implementation_ready | Strong result: credible benchmark position and ready under current policy. |
| AutoGluon Tabular (Practical) | baseline_established | implementation_ready | Strong result: credible benchmark position and ready under current policy. |

## Workflow traits

| runner | search_style | artifact_trail | journal | bounded_next_step | llm_guidance |
| --- | --- | --- | --- | --- | --- |
| Plain XGBoost Baseline | manual_baseline_only | minimal | no | no | no |
| Treehouse Lab Baseline | bounded_baseline | full | yes | not_yet | no |
| Treehouse Lab 1-Step Loop | bounded_loop | full_plus_loop_summary | yes | yes | no |
| AutoGluon Tabular (Practical) | opaque_automl | model_directory_only | no | no | no |

## Feature-generation decisions

| runner | considered | selected | applied | generated features | outcome gates | complexity read |
| --- | --- | --- | --- | --- | --- | --- |
| Plain XGBoost Baseline | no | no | no | 0 | baseline_established / implementation_ready | No bounded feature branch was selected. |
| Treehouse Lab Baseline | no | no | no | 0 | baseline_established / implementation_ready | No bounded feature branch was selected. |
| Treehouse Lab 1-Step Loop | no | no | no | 0 | baseline_established / implementation_ready | No bounded feature branch was selected. |
| AutoGluon Tabular (Practical) | no | no | no | 0 | baseline_established / implementation_ready | No bounded feature branch was selected. |
```

What this shows:

- compare is about product positioning, not just leaderboard numbers
- Treehouse exposes the operating layer that plain baselines and AutoML do not
- the report now separates benchmark position from implementation readiness
- feature-generation complexity is visible as a bounded decision, not hidden in raw artifacts

## How To Use This Page

If you are reviewing Treehouse Lab quickly, read the outputs in this order:

1. baseline
2. proposal
3. journal
4. compare report

That sequence shows the product shape more honestly than a single metric table.
