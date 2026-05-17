# Benchmark Report

Treehouse Lab's benchmark story is a decision guide, not a leaderboard.

The useful question is not "does Treehouse Lab always beat every AutoML runner?" It is:

- when does a bounded, audit-first XGBoost loop make more sense than a plain baseline?
- when should a team use FLAML or AutoGluon instead?
- what evidence should be reviewed before claiming a model is implementation-ready?

Use this page with the fixed public suite in `configs/benchmark_suites/public_v1_3.yaml`.

For a concrete fill-in structure, use [Benchmark Report Example](benchmark-report-example.md). It shows where real suite results, runtime notes, promotion decisions, and implementation-readiness evidence should be inserted without fabricating benchmark numbers.

## Short Answer

| Situation | Best fit | Why |
| --- | --- | --- |
| You need a fast, transparent metric anchor | Plain XGBoost | It is the cleanest baseline and has the least workflow overhead. |
| You need reviewable iteration with artifacts and promotion policy | Treehouse Lab | It keeps the search bounded, records proposals, journals rejected runs, and separates benchmark progress from readiness. |
| You need broad automated model search and can tolerate less audit detail | AutoGluon or FLAML | They are stronger external automation references when raw automation breadth matters more than a readable lab trail. |
| You need regression, hosted training, or a production serving platform | Not Treehouse Lab v1 | The current product is local, classification-first, and XGBoost-first. |

The positioning claim should be modest: Treehouse Lab is strongest when a team cares about constrained search, reviewability, and handoff artifacts as much as the score.

## Fixed Suite Contract

The v1.3 suite is intentionally small:

- `bank_marketing_uci`: mixed-type imbalanced business classification
- `adult_uci`: mixed-type census-income classification with missing categorical values
- `covertype_uci`: larger multiclass land-cover classification

It compares:

- plain XGBoost baseline
- Treehouse Lab baseline
- Treehouse Lab bounded loop
- AutoGluon Tabular with the practical profile
- FLAML with a small time budget and explicit estimator list

Fetch the public datasets first:

```bash
python3 scripts/fetch_bank_marketing.py
python3 scripts/fetch_adult.py
python3 scripts/fetch_covertype.py
```

Run the fast local check without optional AutoML runners:

```bash
treehouse-lab benchmark-suite configs/benchmark_suites/public_v1_3.yaml --skip-autogluon --skip-flaml
```

Run the full practical comparison after preparing the benchmark environment:

```bash
./scripts/setup_benchmark_env.sh
.venv-benchmarks/bin/python -m treehouse_lab.cli benchmark-suite configs/benchmark_suites/public_v1_3.yaml
```

The suite writes one directory under `outputs/benchmark_suites/` by default. Each dataset gets a comparison `report.md`, and the suite writes a top-level `summary.json`.

## What To Review

A useful benchmark report should compare more than the primary metric.

| Evidence | What it answers |
| --- | --- |
| Validation and test metrics | Did the runner improve under the same fixed split contract? |
| Runtime | Was the improvement practical under the configured budget? |
| Readiness gates | Did the run clear overfit, validation-test gap, runtime, and feature-budget checks? |
| Workflow traits | Did the runner leave a journal, proposal, bounded next step, and artifact trail? |
| Feature-generation decision | Was extra complexity considered, applied, and justified? |
| Grounding and LLM selection | Did any LLM-guided choice stay inside the explicit candidate set and cite local evidence? |
| Weak-cycle fallback | Did the loop avoid repeating a low-value mutation family when progress stalled? |

This matters because a run can be benchmark-better and still not implementation-ready. Treehouse Lab should make that distinction visible instead of flattening everything into one score.

## How To Read Runner Results

### Plain XGBoost

Plain XGBoost is the metric anchor. If Treehouse Lab cannot clearly explain what it adds beyond this baseline, the report should say so.

Use it to answer:

- is the dataset already solved by a strong simple model?
- what score and runtime should the bounded loop beat?
- is extra workflow overhead justified?

### Treehouse Lab Baseline

The Treehouse baseline should be close to the plain XGBoost anchor because it uses the same learner family. Its value is the extra contract around artifacts, assessment, and incumbent state.

Use it when:

- you need a clean incumbent registry
- you need run summaries and exported artifacts
- you want a baseline that can be extended by bounded proposals

### Treehouse Lab Bounded Loop

The bounded loop is the core product bet. It should not be judged only on whether one short loop beats AutoML. Judge it on whether it makes disciplined next-step decisions and records why each step ran.

Use it when:

- the team wants small attributable moves instead of opaque automation
- rejected experiments should remain inspectable
- feature generation must stay capped and train-only
- LLM guidance is useful only if it chooses from explicit candidates and records the tradeoff

### FLAML

FLAML is the lightweight AutoML reference. It is useful when a team wants a quick automated search under a budget.

Treehouse Lab should not pretend to replace FLAML for raw automated sweep breadth. The honest comparison is whether Treehouse gives enough audit trail and handoff quality to justify its narrower search surface.

### AutoGluon

AutoGluon is the broader practical AutoML reference. It is useful for checking whether Treehouse is competitive against a stronger automation stack.

If AutoGluon wins raw metric, that is not automatically a Treehouse failure. The report should ask whether Treehouse's artifacts, bounded proposal history, and readiness gates are more valuable for the intended workflow.

## Strengths To Claim Carefully

Treehouse Lab can make a credible case when the report shows:

- a plain baseline, Treehouse baseline, and bounded loop all use the same split policy
- the loop improves or stops with a clear rationale
- rejected runs remain visible in the journal
- readiness gates explain why a promising metric may still need work
- external AutoML runners are treated as references, not hidden competitors
- LLM-guided choices are bounded and reviewable rather than free-form

The strongest product claim is auditability under useful constraints.

## Limits To State Plainly

Treehouse Lab v1 is not the right tool when:

- the target is continuous regression
- the team wants a broad model zoo
- the only goal is the highest possible leaderboard score
- the work needs hosted training, monitoring, or production orchestration
- the dataset needs custom feature semantics outside the current bounded templates

Those limits are part of the product stance. They keep the loop legible.

## Publishable Report Checklist

Before treating a benchmark result as publishable, confirm:

- the dataset fetch commands were run from a clean checkout
- the suite config was `configs/benchmark_suites/public_v1_3.yaml`
- the seed, split policy, loop steps, AutoGluon profile, and FLAML budget were not changed silently
- unavailable optional runners are labeled unavailable rather than omitted
- each dataset report links to or quotes its generated `report.md`
- the top-level summary references `summary.json`
- the writeup separates metric value, workflow value, and implementation readiness

If any of those are missing, call the result a local probe rather than a benchmark report.
