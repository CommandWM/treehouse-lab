# v1.2 Product Polish

Treehouse Lab v1.2 Product Polish is the shareability and clarity closeout for the current v1.1 product surface.

This page is the durable handoff for the Linear v1.2 polish scope. It records what is now present in the repo, where to review it, and what still belongs to later benchmark or v1.3 implementation work.

## Scope Closed

| Linear item | Closeout evidence |
| --- | --- |
| `COM-5` polished walkthroughs and screenshots | `docs/walkthrough.md` now provides the short end-to-end workbench path and references the current screenshot set. |
| `COM-6` public dataset paths | `bank_marketing_uci`, `adult_uci`, and `covertype_uci` are documented as the fixed public probe set, with fetch commands and suite config wiring. |
| `COM-7` cleaner sample outputs | `docs/sample-outputs.md` shows baseline, proposal, journal, and compare-report excerpts in the expected reading order. |
| `COM-8` feature-generation decision visibility | The compare report and workbench expose whether the feature branch was considered, selected, applied, capped, and justified. |
| `COM-9` benchmark-vs-readiness clarity | README, benchmark docs, sample outputs, report examples, and the workbench keep benchmark progress separate from implementation readiness. |
| `COM-18` v1.1 positioning/export/report polish | `docs/export-contract.md`, `docs/benchmark-report-example.md`, and version metadata tests complete the final v1.2 closeout slice. |

## Review Path

Use this sequence for a quick public-facing review:

1. Start with `docs/walkthrough.md` to understand the intake-first flow.
2. Open the current screenshot set:
   - `docs/assets/screenshots/intake.png`
   - `docs/assets/screenshots/current-state.png`
   - `docs/assets/screenshots/journal.png`
   - `docs/assets/screenshots/architecture.png`
3. Read `docs/sample-outputs.md` to see baseline, bounded proposal, journal, and compare artifacts.
4. Read `docs/benchmark-report.md` for interpretation rules.
5. Use `docs/benchmark-report-example.md` only as a fill-in structure for real suite results.
6. Use `docs/export-contract.md` when reviewing model handoff and generated scorer behavior.

## Public Dataset Path

The v1.2 public dataset story is no longer a single Bank Marketing example.

Current public probe keys:

- `bank_marketing_uci`: implementation-minded, imbalanced business classification
- `adult_uci`: mixed-type census-income classification with missing categorical values
- `covertype_uci`: larger multiclass classification outside the business domain

Fetch commands:

```bash
python3 scripts/fetch_bank_marketing.py
python3 scripts/fetch_adult.py
python3 scripts/fetch_covertype.py
```

Fixed suite command:

```bash
treehouse-lab benchmark-suite configs/benchmark_suites/public_v1_3.yaml --skip-autogluon --skip-flaml
```

This documents the path through the repo. It is not a benchmark claim. A benchmark claim needs a fresh `outputs/benchmark_suites/` run directory, a commit hash, environment notes, and dataset-level `report.md` evidence.

## Feature-Generation Review

Feature generation remains bounded and train-only. The closeout criteria are:

- the proposed mutation shows whether the feature branch is on or off
- the strategy and feature cap stay visible before execution
- generated feature details remain attached to run artifacts when the branch is applied
- compare reports include feature-generation decision rows
- added complexity is read against both benchmark status and implementation readiness

## Benchmark And Readiness Review

The v1.2 polish layer keeps two questions separate:

- Did the run improve the benchmark position?
- Did the run clear implementation-readiness gates?

That distinction appears in:

- `docs/evaluation-policy.md`
- `docs/walkthrough.md`
- `docs/sample-outputs.md`
- `docs/benchmark-report.md`
- `docs/benchmark-report-example.md`
- the Current State and Journal views in `frontend/src/App.jsx`

## What Moves To Later Work

The v1.2 polish layer is closed, but the following are still later work:

- run the fixed public suite and replace `TBD` placeholders with real outputs
- deepen bounded XGBoost search quality under `COM-14`
- package the real benchmark evidence into the later `COM-15` positioning release

