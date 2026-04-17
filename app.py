from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from treehouse_lab.config import load_experiment_config, load_yaml_file
from treehouse_lab.journal import load_incumbent, load_journal_entries
from treehouse_lab.loop import AutonomousLoopController
from treehouse_lab.runner import TreehouseLabRunner

PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_CONFIG_DIR = PROJECT_ROOT / "configs" / "datasets"
SEARCH_SPACE_PATH = PROJECT_ROOT / "configs" / "search_space.yaml"
GLOSSARY_PATH = PROJECT_ROOT / "docs" / "glossary.md"


@st.cache_data(show_spinner=False)
def list_example_specs() -> list[Path]:
    return sorted(DATASET_CONFIG_DIR.glob("*.yaml"))


@st.cache_data(show_spinner=False)
def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


@st.cache_data(show_spinner=False)
def load_glossary_sections(path: Path) -> list[tuple[str, str]]:
    sections: list[tuple[str, str]] = []
    heading: str | None = None
    body: list[str] = []

    for raw_line in read_text(path).splitlines():
        line = raw_line.rstrip()
        if line.startswith("## "):
            if heading is not None:
                sections.append((heading, "\n".join(part for part in body if part.strip()).strip()))
            heading = line.removeprefix("## ").strip()
            body = []
            continue
        if heading is not None and not line.startswith("# "):
            body.append(line)

    if heading is not None:
        sections.append((heading, "\n".join(part for part in body if part.strip()).strip()))
    return sections


def inject_app_css() -> None:
    st.markdown(
        """
        <style>
        :root {
          --paper: #f5f1e8;
          --ink: #1e2a24;
          --muted: #5d685f;
          --accent: #335c4a;
          --accent-soft: #dce9df;
          --gold: #b58d3b;
          --line: rgba(30, 42, 36, 0.15);
          --warn: #8a4b2d;
          --ok: #2e684e;
        }
        .stApp {
          background:
            radial-gradient(circle at top right, rgba(181, 141, 59, 0.08), transparent 26%),
            linear-gradient(180deg, #fbf8f2 0%, #f4efe4 100%);
        }
        .block-container {
          padding-top: 2rem;
          padding-bottom: 2rem;
        }
        h1, h2, h3 {
          color: var(--ink);
          letter-spacing: -0.02em;
        }
        .treehouse-hero {
          background: linear-gradient(140deg, rgba(51, 92, 74, 0.96), rgba(30, 42, 36, 0.96));
          border: 1px solid rgba(255,255,255,0.08);
          border-radius: 22px;
          color: #f8f6f0;
          padding: 1.4rem 1.5rem 1.2rem 1.5rem;
          box-shadow: 0 18px 50px rgba(30, 42, 36, 0.12);
          margin-bottom: 1rem;
        }
        .treehouse-kicker {
          color: rgba(248, 246, 240, 0.78);
          font-size: 0.8rem;
          text-transform: uppercase;
          letter-spacing: 0.16em;
          margin-bottom: 0.6rem;
        }
        .treehouse-title {
          font-size: 2.1rem;
          line-height: 1.05;
          margin-bottom: 0.6rem;
          font-weight: 700;
        }
        .treehouse-copy {
          color: rgba(248, 246, 240, 0.86);
          max-width: 60rem;
          font-size: 1rem;
        }
        .meta-row {
          display: flex;
          flex-wrap: wrap;
          gap: 0.45rem;
          margin-top: 0.9rem;
        }
        .meta-pill {
          border: 1px solid rgba(255,255,255,0.14);
          background: rgba(255,255,255,0.08);
          border-radius: 999px;
          padding: 0.32rem 0.7rem;
          font-size: 0.84rem;
        }
        .guide-card {
          background: rgba(255, 255, 255, 0.72);
          border: 1px solid var(--line);
          border-radius: 18px;
          padding: 1rem 1rem 0.9rem 1rem;
          min-height: 100%;
          box-shadow: 0 10px 24px rgba(30, 42, 36, 0.05);
        }
        .guide-card h4 {
          color: var(--ink);
          margin: 0 0 0.45rem 0;
          font-size: 1rem;
        }
        .guide-card p {
          color: var(--muted);
          margin: 0;
          line-height: 1.5;
          font-size: 0.94rem;
        }
        .blueprint-shell {
          background: rgba(255, 255, 255, 0.82);
          border: 1px solid var(--line);
          border-radius: 22px;
          padding: 1.1rem;
          box-shadow: 0 16px 36px rgba(30, 42, 36, 0.06);
        }
        .blueprint-header {
          display: flex;
          justify-content: space-between;
          align-items: flex-start;
          gap: 1rem;
          margin-bottom: 1rem;
        }
        .blueprint-title {
          font-size: 1.2rem;
          font-weight: 700;
          color: var(--ink);
          margin-bottom: 0.35rem;
        }
        .blueprint-subtitle {
          color: var(--muted);
          line-height: 1.5;
          font-size: 0.94rem;
        }
        .blueprint-stage-row {
          display: grid;
          grid-template-columns: repeat(6, minmax(0, 1fr));
          gap: 0.7rem;
          align-items: center;
        }
        .stage-card {
          background: linear-gradient(180deg, rgba(220, 233, 223, 0.72), rgba(255, 255, 255, 0.95));
          border: 1px solid var(--line);
          border-radius: 18px;
          padding: 0.85rem;
          min-height: 8.8rem;
        }
        .stage-label {
          color: var(--gold);
          font-size: 0.74rem;
          text-transform: uppercase;
          letter-spacing: 0.12em;
          margin-bottom: 0.4rem;
        }
        .stage-name {
          color: var(--ink);
          font-weight: 700;
          font-size: 0.98rem;
          margin-bottom: 0.45rem;
        }
        .stage-copy {
          color: var(--muted);
          font-size: 0.88rem;
          line-height: 1.45;
        }
        .stage-arrow {
          color: var(--gold);
          text-align: center;
          font-size: 1.4rem;
          font-weight: 700;
        }
        .signal-strip {
          display: grid;
          grid-template-columns: repeat(4, minmax(0, 1fr));
          gap: 0.7rem;
          margin-top: 1rem;
        }
        .signal-card {
          border-radius: 18px;
          border: 1px solid var(--line);
          padding: 0.85rem;
          background: rgba(255, 255, 255, 0.9);
        }
        .signal-label {
          color: var(--muted);
          font-size: 0.76rem;
          text-transform: uppercase;
          letter-spacing: 0.12em;
          margin-bottom: 0.35rem;
        }
        .signal-value {
          color: var(--ink);
          font-size: 1rem;
          font-weight: 700;
          margin-bottom: 0.25rem;
        }
        .signal-copy {
          color: var(--muted);
          font-size: 0.86rem;
          line-height: 1.4;
        }
        .status-note {
          border-left: 4px solid var(--accent);
          padding: 0.7rem 0.9rem;
          background: rgba(220, 233, 223, 0.7);
          border-radius: 0 12px 12px 0;
          color: var(--ink);
        }
        .artifact-card {
          border: 1px solid var(--line);
          border-radius: 16px;
          padding: 0.85rem;
          background: rgba(255,255,255,0.82);
        }
        @media (max-width: 1080px) {
          .blueprint-stage-row {
            grid-template-columns: 1fr;
          }
          .stage-arrow {
            transform: rotate(90deg);
          }
          .signal-strip {
            grid-template-columns: 1fr 1fr;
          }
        }
        @media (max-width: 720px) {
          .signal-strip {
            grid-template-columns: 1fr;
          }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def format_metric(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.4f}"


def format_list(values: list[str]) -> str:
    return ", ".join(values) if values else "none"


def render_metric_grid(result: dict[str, object]) -> None:
    metrics = result["metrics"]
    comparison = result["comparison_to_incumbent"]
    assessment = result.get("assessment", {})
    diagnosis = result.get("diagnosis", {})

    columns = st.columns(4)
    columns[0].metric("Validation ROC AUC", f"{metrics['validation_roc_auc']:.4f}")
    columns[1].metric("Test ROC AUC", f"{metrics['test_roc_auc']:.4f}")
    columns[2].metric("Validation Accuracy", f"{metrics['validation_accuracy']:.4f}")
    delta = comparison.get("delta")
    columns[3].metric("Delta vs Incumbent", "n/a" if delta is None else f"{delta:.4f}")

    if assessment:
        st.caption(
            f"Benchmark status: `{assessment['benchmark_status']}` | "
            f"Implementation readiness: `{assessment['implementation_readiness']}`"
        )
    if diagnosis:
        st.caption(f"Diagnosis: `{diagnosis['primary_tag']}` | {diagnosis['recommended_direction']}")


def render_hero(config_path: Path, selected_config: object, diagnosis_preview: dict[str, object]) -> None:
    diagnosis = diagnosis_preview.get("diagnosis", {})
    next_proposal = diagnosis_preview.get("next_proposal")
    st.markdown(
        f"""
        <div class="treehouse-hero">
          <div class="treehouse-kicker">Treehouse Lab Research Surface</div>
          <div class="treehouse-title">{selected_config.name}</div>
          <div class="treehouse-copy">
            {selected_config.description}
          </div>
          <div class="meta-row">
            <div class="meta-pill">pack: {selected_config.benchmark.pack}</div>
            <div class="meta-pill">profile: {selected_config.benchmark.profile}</div>
            <div class="meta-pill">primary metric: {selected_config.primary_metric}</div>
            <div class="meta-pill">diagnosis: {diagnosis.get('primary_tag', 'no_incumbent')}</div>
            <div class="meta-pill">next mutation: {next_proposal['mutation_name'] if next_proposal else 'baseline'}</div>
            <div class="meta-pill">config: {config_path.name}</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_guide_cards() -> None:
    columns = st.columns(3)
    cards = [
        (
            "Read the system",
            "Start with the blueprint. It shows what the loop knows, what stage it is in, and what it wants to try next.",
        ),
        (
            "Run one step at a time",
            "Use `baseline`, then `diagnose`, then a bounded candidate or short loop. The point is to see why the system moves, not just that it moves.",
        ),
        (
            "Judge two things",
            "Every run answers two different questions: did it beat the benchmark, and is it implementation-ready?",
        ),
    ]
    for column, (title, copy) in zip(columns, cards, strict=True):
        column.markdown(
            f"""
            <div class="guide-card">
              <h4>{title}</h4>
              <p>{copy}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_blueprint(config_path: Path, selected_config: object, incumbent: dict[str, object] | None, diagnosis_preview: dict[str, object]) -> None:
    diagnosis = diagnosis_preview.get("diagnosis", {})
    next_proposal = diagnosis_preview.get("next_proposal")
    incumbent_metric = None if incumbent is None else float(incumbent.get("metric", 0.0))
    incumbent_assessment = {} if incumbent is None else incumbent.get("assessment", {})
    incumbent_diagnosis = {} if incumbent is None else incumbent.get("diagnosis", {})

    stage_cards = [
        ("01", "Dataset spec", "Lock the benchmark profile, split policy, quality floor, and runtime budget before any search begins."),
        ("02", "Baseline / incumbent", "Establish or read the current best promoted run. This is the anchor every later mutation must beat."),
        ("03", "Diagnosis", f"Current read: {diagnosis.get('summary', 'No incumbent yet. Run the baseline first.')}"),
        ("04", "Proposal", "Generate one bounded mutation with a plain hypothesis, expected upside, and explicit risk."),
        ("05", "Evaluation gate", "Measure validation, holdout, runtime, and feature budget. Do not touch the held-out test during search."),
        ("06", "Promote or reject", "A run can be benchmark-better and still fail implementation readiness. Keep both labels visible."),
    ]

    blueprint_html_parts = [
        '<div class="blueprint-shell">',
        '<div class="blueprint-header">',
        '<div>',
        '<div class="blueprint-title">Autoresearch Blueprint</div>',
        '<div class="blueprint-subtitle">A readable map of the loop, inspired by system-blueprint style docs: explicit inputs, a visible control flow, and plain-language state.</div>',
        "</div>",
        f'<div class="meta-pill">selected config: {config_path.name}</div>',
        "</div>",
        '<div class="blueprint-stage-row">',
    ]
    for label, title, copy in stage_cards:
        blueprint_html_parts.append(
            f"""
            <div class="stage-card">
              <div class="stage-label">{label}</div>
              <div class="stage-name">{title}</div>
              <div class="stage-copy">{copy}</div>
            </div>
            """
        )
    blueprint_html_parts.append("</div>")

    blueprint_html_parts.append(
        f"""
        <div class="signal-strip">
          <div class="signal-card">
            <div class="signal-label">Incumbent</div>
            <div class="signal-value">{format_metric(incumbent_metric)}</div>
            <div class="signal-copy">Current validation metric for the promoted run.</div>
          </div>
          <div class="signal-card">
            <div class="signal-label">Benchmark Status</div>
            <div class="signal-value">{incumbent_assessment.get('benchmark_status', 'none')}</div>
            <div class="signal-copy">Whether the current best run established or improved the benchmark.</div>
          </div>
          <div class="signal-card">
            <div class="signal-label">Implementation Readiness</div>
            <div class="signal-value">{incumbent_assessment.get('implementation_readiness', 'not_started')}</div>
            <div class="signal-copy">Whether the current best run clears the stricter readiness policy.</div>
          </div>
          <div class="signal-card">
            <div class="signal-label">Next Mutation</div>
            <div class="signal-value">{next_proposal['mutation_name'] if next_proposal else 'baseline'}</div>
            <div class="signal-copy">The next bounded move suggested by diagnosis-aware proposal selection.</div>
          </div>
        </div>
        """
    )
    blueprint_html_parts.append("</div>")
    st.markdown("".join(blueprint_html_parts), unsafe_allow_html=True)

    st.markdown("#### Current state notes")
    notes = []
    if incumbent is None:
        notes.append("No incumbent exists yet. Start with the baseline for this dataset config.")
    else:
        notes.append(f"Current incumbent metric: `{incumbent_metric:.4f}`.")
        notes.append(f"Current incumbent diagnosis: `{incumbent_diagnosis.get('primary_tag', 'n/a')}`.")
        notes.append(f"Current readiness label: `{incumbent_assessment.get('implementation_readiness', 'n/a')}`.")
    if diagnosis:
        notes.append(f"Preferred mutations: `{format_list(diagnosis.get('preferred_mutations', []))}`.")
        notes.append(f"Avoided mutations: `{format_list(diagnosis.get('avoided_mutations', []))}`.")
    st.markdown("\n".join(f"- {note}" for note in notes))


def render_state_snapshot(incumbent: dict[str, object] | None, diagnosis_preview: dict[str, object]) -> None:
    diagnosis = diagnosis_preview.get("diagnosis", {})
    next_proposal = diagnosis_preview.get("next_proposal")
    incumbent_assessment = {} if incumbent is None else incumbent.get("assessment", {})
    incumbent_diagnosis = {} if incumbent is None else incumbent.get("diagnosis", {})

    left, right = st.columns([1.05, 0.95])
    with left:
        st.markdown("#### Loop readout")
        st.markdown(
            f"""
            <div class="status-note">
              <strong>Diagnosis</strong><br/>
              {diagnosis.get('summary', 'No diagnosis yet.')}<br/><br/>
              <strong>Recommended direction</strong><br/>
              {diagnosis.get('recommended_direction', 'Run the baseline first.')}
            </div>
            """,
            unsafe_allow_html=True,
        )
        if next_proposal:
            st.markdown("#### Suggested next proposal")
            st.json(
                {
                    "mutation_name": next_proposal["mutation_name"],
                    "hypothesis": next_proposal["hypothesis"],
                    "risk_level": next_proposal["risk_level"],
                    "params_override": next_proposal["params_override"],
                },
                expanded=False,
            )
    with right:
        st.markdown("#### Current incumbent")
        if incumbent is None:
            st.info("No incumbent has been promoted yet.")
        else:
            st.json(
                {
                    "metric": incumbent.get("metric"),
                    "benchmark_status": incumbent_assessment.get("benchmark_status"),
                    "implementation_readiness": incumbent_assessment.get("implementation_readiness"),
                    "diagnosis": incumbent_diagnosis.get("primary_tag"),
                    "reason_codes": incumbent.get("reason_codes", []),
                },
                expanded=False,
            )


def run_and_render_baseline(config_path: Path) -> None:
    runner = TreehouseLabRunner(config_path)
    result = runner.run_baseline().to_dict()
    st.session_state["last_result"] = result
    st.success(f"Baseline completed. Artifacts written to `{result['artifact_dir']}`.")
    render_metric_grid(result)
    st.json(result, expanded=False)


def run_and_render_candidate(config_path: Path, mutation_name: str, hypothesis: str, overrides: dict[str, object]) -> None:
    runner = TreehouseLabRunner(config_path)
    result = runner.run_candidate(mutation_name=mutation_name, overrides=overrides, hypothesis=hypothesis).to_dict()
    st.session_state["last_result"] = result
    st.success(f"Candidate completed. Decision: {'promote' if result['promoted'] else 'reject'}.")
    render_metric_grid(result)
    st.json(result, expanded=False)


st.set_page_config(page_title="Treehouse Lab", layout="wide")
inject_app_css()

example_specs = list_example_specs()
if not example_specs:
    st.error("No example specs were found under configs/datasets.")
    st.stop()

selected_spec = st.sidebar.selectbox("Dataset config", example_specs, format_func=lambda path: path.stem.replace("_", " ").title())
selected_config = load_experiment_config(selected_spec)
search_space = load_yaml_file(SEARCH_SPACE_PATH)
incumbent = load_incumbent(PROJECT_ROOT, selected_spec.stem)
journal_entries = load_journal_entries(PROJECT_ROOT, selected_spec.stem)
controller = AutonomousLoopController(selected_spec)
diagnosis_preview = controller.diagnose().to_dict()

render_hero(selected_spec, selected_config, diagnosis_preview)

st.sidebar.markdown("### Teaching Surface")
st.sidebar.write("Use this app as a guided walkthrough of the loop: baseline, diagnose, propose, evaluate, and review artifacts.")
st.sidebar.caption("The UI is intentionally opinionated: it should help a person understand why the loop moved, not just dump JSON.")

st.sidebar.markdown("### Quick legend")
st.sidebar.markdown(
    "\n".join(
        [
            "- `benchmark status`: did the run beat the benchmark?",
            "- `implementation readiness`: would you trust it enough to carry forward?",
            "- `diagnosis`: what failure mode is the loop currently reacting to?",
        ]
    )
)

guide_tab, blueprint_tab, baseline_tab, candidate_tab, journal_tab, glossary_tab = st.tabs(
    ["Guide", "Blueprint", "Baseline", "Candidate", "Journal", "Glossary"]
)

with guide_tab:
    st.subheader("How To Read Treehouse Lab")
    render_guide_cards()
    st.markdown("")
    render_blueprint(selected_spec, selected_config, incumbent, diagnosis_preview)
    st.markdown("")
    render_state_snapshot(incumbent, diagnosis_preview)
    st.markdown("#### What to do first")
    st.markdown(
        "\n".join(
            [
                "1. Read the diagnosis and suggested next proposal.",
                "2. Run the baseline if no incumbent exists yet.",
                "3. Compare benchmark status against implementation readiness.",
                "4. Open the run artifacts and journal once a run completes.",
            ]
        )
    )

with blueprint_tab:
    st.subheader("System Blueprint")
    render_blueprint(selected_spec, selected_config, incumbent, diagnosis_preview)
    left, right = st.columns([1, 1])
    with left:
        st.markdown("#### Dataset spec")
        st.code(read_text(selected_spec), language="yaml")
    with right:
        st.markdown("#### Agent policy")
        st.code(read_text(PROJECT_ROOT / "program.md"), language="markdown")
    st.markdown("#### Current diagnosis payload")
    st.json(diagnosis_preview, expanded=False)

with baseline_tab:
    st.subheader("Run the baseline")
    st.write("Use this when a config has no incumbent yet or when you want to re-establish the initial reference point for the loop.")
    st.markdown(
        """
        <div class="artifact-card">
          <strong>What you get</strong><br/>
          A baseline run writes `summary.md`, `assessment.json`, `diagnosis.json`, model params, metrics, and feature importances into `runs/`.
        </div>
        """,
        unsafe_allow_html=True,
    )
    if st.button("Run Baseline", type="primary", width="stretch"):
        run_and_render_baseline(selected_spec)
    if "last_result" in st.session_state:
        st.markdown("#### Latest result")
        render_metric_grid(st.session_state["last_result"])

with candidate_tab:
    st.subheader("Run one bounded mutation")
    next_proposal = diagnosis_preview.get("next_proposal")
    param_space = search_space["xgboost"]
    incumbent_params = {} if incumbent is None else dict(incumbent.get("params", {}))
    default_params = dict(selected_config.model.params)
    default_params.update(incumbent_params)
    if next_proposal:
        default_params.update(next_proposal.get("base_params", {}))
        default_params.update(next_proposal.get("params_override", {}))
        st.markdown(
            f"""
            <div class="status-note">
              <strong>Suggested next mutation</strong><br/>
              `{next_proposal['mutation_name']}`<br/><br/>
              {next_proposal['diagnosis_summary']}
            </div>
            """,
            unsafe_allow_html=True,
        )

    col1, col2, col3 = st.columns(3)
    overrides = {
        "n_estimators": col1.slider(
            "n_estimators",
            min_value=int(param_space["n_estimators"][0]),
            max_value=int(param_space["n_estimators"][1]),
            value=int(default_params.get("n_estimators", 300)),
            step=20,
        ),
        "max_depth": col2.slider(
            "max_depth",
            min_value=int(param_space["max_depth"][0]),
            max_value=int(param_space["max_depth"][1]),
            value=int(default_params.get("max_depth", 6)),
            step=1,
        ),
        "learning_rate": col3.slider(
            "learning_rate",
            min_value=float(param_space["learning_rate"][0]),
            max_value=float(param_space["learning_rate"][1]),
            value=float(default_params.get("learning_rate", 0.05)),
            step=0.01,
        ),
        "min_child_weight": col1.slider(
            "min_child_weight",
            min_value=int(param_space["min_child_weight"][0]),
            max_value=int(param_space["min_child_weight"][1]),
            value=int(default_params.get("min_child_weight", 1)),
            step=1,
        ),
        "subsample": col2.slider(
            "subsample",
            min_value=float(param_space["subsample"][0]),
            max_value=float(param_space["subsample"][1]),
            value=float(default_params.get("subsample", 0.9)),
            step=0.05,
        ),
        "colsample_bytree": col3.slider(
            "colsample_bytree",
            min_value=float(param_space["colsample_bytree"][0]),
            max_value=float(param_space["colsample_bytree"][1]),
            value=float(default_params.get("colsample_bytree", 0.8)),
            step=0.05,
        ),
    }
    mutation_name = st.text_input("Mutation name", value="candidate-tune" if not next_proposal else str(next_proposal["mutation_name"]))
    hypothesis = st.text_area(
        "Hypothesis",
        value=(
            "A smaller, better-regularized tree ensemble should improve validation ROC AUC without making the run harder to explain."
            if not next_proposal
            else str(next_proposal["hypothesis"])
        ),
    )
    if st.button("Run Candidate", width="stretch"):
        run_and_render_candidate(selected_spec, mutation_name, hypothesis, overrides)

with journal_tab:
    st.subheader("Run journal")
    if journal_entries:
        journal_frame = pd.DataFrame(journal_entries).sort_values("run_id", ascending=False)
        preview_columns = [
            column
            for column in ["run_id", "name", "metric", "promoted", "reason_codes", "diagnosis", "assessment"]
            if column in journal_frame.columns
        ]
        st.dataframe(journal_frame[preview_columns], width="stretch")
        selected_run = st.selectbox("Inspect run", journal_frame["run_id"])
        selected_entry = next(entry for entry in journal_entries if entry["run_id"] == selected_run)
        left, right = st.columns([1, 1])
        with left:
            st.markdown("#### Run summary")
            st.json(
                {
                    "name": selected_entry.get("name"),
                    "metric": selected_entry.get("metric"),
                    "promoted": selected_entry.get("promoted"),
                    "benchmark_status": selected_entry.get("assessment", {}).get("benchmark_status"),
                    "implementation_readiness": selected_entry.get("assessment", {}).get("implementation_readiness"),
                    "diagnosis": selected_entry.get("diagnosis", {}).get("primary_tag"),
                },
                expanded=False,
            )
        with right:
            st.markdown("#### Reason codes")
            st.write(selected_entry.get("reason_codes", []))
        st.markdown("#### Full journal entry")
        st.json(selected_entry, expanded=False)
    else:
        st.info("No runs recorded yet for this dataset config.")

with glossary_tab:
    st.subheader("Glossary")
    st.caption("This mirrors the project glossary so the interface can explain the loop in plain language while you use it.")
    for term, definition in load_glossary_sections(GLOSSARY_PATH):
        with st.expander(term):
            st.write(definition)
    st.markdown("#### Source")
    st.code(read_text(GLOSSARY_PATH), language="markdown")

st.markdown("---")
st.caption("Artifacts and the incumbent registry live under `runs/`. This interface is meant to teach the loop as well as operate it.")
