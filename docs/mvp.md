# 2.0 Foundation Baseline

This file records the implementation baseline that proposal grounding can cite. It is not a release checklist.

## Goal

Keep the current XGBoost-first classification loop bounded, auditable, and handoff-ready while the 2.0 contract surface is formalized for both classification and regression.

## Foundation Already Present

- dataset config format
- reproducible train/validation/test split handling
- baseline runner
- bounded candidate runner
- incumbent registry
- promotion threshold policy
- experiment journal
- deterministic proposal ranking
- bounded loop controller
- capped train-only feature generation
- comparison harness
- export bundle and minimal scorer wrapper

## Planned Regression Foundation

Regression belongs in the 2.0 direction, but it needs its own contracts instead of reusing classification behavior.

Required foundation:

- regression target profiling instead of classification label encoding
- `XGBRegressor` runner path
- regression metrics such as RMSE, MAE, and R2
- regression-specific readiness and overfit checks
- regression scorer response shape
- regression benchmark fixtures and report sections

## Guardrails That Must Stay True

- generated features are fit on train only
- validation and test transforms reuse the train-fitted contract
- feature count caps remain explicit
- candidate params stay inside `configs/search_space.yaml`
- the target column is never part of the feature matrix
- promotion decisions record both metric movement and readiness status
- optional LLM guidance chooses only from explicit candidates
- regression targets are never coerced into classification labels

## 2.0 Contract Gaps

- canonical CLI workflow docs for classification and regression
- stable CLI JSON success and failure contracts
- API request/response docs
- manifest and prediction schemas
- leakage and split guardrail tests
- deterministic promotion/rejection tests
- agent task cards and evaluation harness

## Non-Goals

- general LLM literature review automation
- distributed training
- time-series search
- multimodal tabular problems
- production deployment platform work
- automatic paper writing or blog generation
