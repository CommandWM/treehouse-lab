from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from treehouse_lab.config import load_experiment_config, load_yaml_file
from treehouse_lab.journal import load_incumbent, load_journal_entries
from treehouse_lab.runner import TreehouseLabRunner

PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_CONFIG_DIR = PROJECT_ROOT / "configs" / "datasets"
SEARCH_SPACE_PATH = PROJECT_ROOT / "configs" / "search_space.yaml"


@st.cache_data(show_spinner=False)
def list_example_specs() -> list[Path]:
    return sorted(DATASET_CONFIG_DIR.glob("*.yaml"))


@st.cache_data(show_spinner=False)
def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def render_metric_grid(result: dict[str, object]) -> None:
    metrics = result["metrics"]
    comparison = result["comparison_to_incumbent"]
    columns = st.columns(4)
    columns[0].metric("Validation ROC AUC", f"{metrics['validation_roc_auc']:.4f}")
    columns[1].metric("Test ROC AUC", f"{metrics['test_roc_auc']:.4f}")
    columns[2].metric("Validation Accuracy", f"{metrics['validation_accuracy']:.4f}")
    delta = comparison.get("delta")
    delta_text = "n/a" if delta is None else f"{delta:.4f}"
    columns[3].metric("Delta vs Incumbent", delta_text)


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


st.set_page_config(page_title="Treehouse Lab", page_icon="🌲", layout="wide")

st.title("Treehouse Lab")
st.caption("Disciplined tabular autoresearch with explicit guardrails, readable artifacts, and bounded experiment mutations.")

example_specs = list_example_specs()
if not example_specs:
    st.error("No example specs were found under configs/datasets.")
    st.stop()

selected_spec = st.sidebar.selectbox("Example", example_specs, format_func=lambda path: path.stem.replace("_", " ").title())
selected_config = load_experiment_config(selected_spec)
search_space = load_yaml_file(SEARCH_SPACE_PATH)
incumbent = load_incumbent(PROJECT_ROOT, selected_spec.stem)
journal_entries = load_journal_entries(PROJECT_ROOT)

st.sidebar.markdown("### Why Streamlit")
st.sidebar.write(
    "Treehouse Lab benefits from a lightweight dashboard: config inspection, one-click runs, comparison against the incumbent, and a readable run journal."
)
st.sidebar.caption("The runner prefers XGBoost and falls back to sklearn gradient boosting if the local XGBoost runtime is unavailable.")

overview_tab, baseline_tab, candidate_tab, journal_tab = st.tabs(["Overview", "Baseline", "Candidate", "Journal"])

with overview_tab:
    left, right = st.columns([1.1, 0.9])
    with left:
        st.subheader(selected_config.name)
        st.write(selected_config.description)
        st.markdown(
            "\n".join(
                [
                    "- One dataset spec per example keeps the search space explicit.",
                    "- The held-out test set is preserved while search decisions are made on validation metrics.",
                    "- Every run emits a readable artifact bundle plus a journal entry.",
                    "- Promotion is threshold-based, not vibes-based.",
                ]
            )
        )
        st.markdown("#### Dataset spec")
        st.code(read_text(selected_spec), language="yaml")
    with right:
        st.markdown("#### Current incumbent")
        if incumbent:
            st.json(incumbent, expanded=False)
        else:
            st.info("No incumbent has been promoted yet.")
        st.markdown("#### Agent policy")
        st.code(read_text(PROJECT_ROOT / "program.md"), language="markdown")

with baseline_tab:
    st.subheader("Run the baseline")
    st.write("This uses the configured dataset spec and model defaults, then writes artifacts into `runs/`.")
    if st.button("Run Baseline", type="primary", width="stretch"):
        run_and_render_baseline(selected_spec)
    if "last_result" in st.session_state:
        st.markdown("#### Latest result")
        render_metric_grid(st.session_state["last_result"])

with candidate_tab:
    st.subheader("Run one bounded mutation")
    st.write("This is intentionally not free-form. You tweak a few parameters inside the declared search space and compare against the incumbent.")
    param_space = search_space["xgboost"]
    default_params = selected_config.model.params

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
    mutation_name = st.text_input("Mutation name", value="candidate-tune")
    hypothesis = st.text_area(
        "Hypothesis",
        value="A smaller, better-regularized tree ensemble should improve validation ROC AUC without making the run harder to explain.",
    )
    if st.button("Run Candidate", width="stretch"):
        run_and_render_candidate(selected_spec, mutation_name, hypothesis, overrides)

with journal_tab:
    st.subheader("Run journal")
    if journal_entries:
        journal_frame = pd.DataFrame(journal_entries).sort_values("run_id", ascending=False)
        st.dataframe(journal_frame, width="stretch")
        selected_run = st.selectbox("Inspect run", journal_frame["run_id"])
        selected_entry = next(entry for entry in journal_entries if entry["run_id"] == selected_run)
        st.json(selected_entry, expanded=False)
    else:
        st.info("No runs recorded yet. Run a baseline or candidate experiment first.")

st.markdown("---")
st.caption("Artifacts and the incumbent registry live under `runs/`. The current UI is a teaching surface for the MVP, not the final research loop.")
