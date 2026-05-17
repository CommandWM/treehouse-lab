from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def read_project_file(relative_path: str) -> str:
    return (PROJECT_ROOT / relative_path).read_text(encoding="utf-8")


def test_v12_polish_closeout_doc_covers_linear_scope() -> None:
    closeout_path = PROJECT_ROOT / "docs" / "v1-2-polish.md"

    assert closeout_path.exists(), "v1.2 closeout should have a durable docs landing page"
    closeout = closeout_path.read_text(encoding="utf-8")

    for issue_id in ("COM-5", "COM-6", "COM-7", "COM-8", "COM-9", "COM-18"):
        assert issue_id in closeout

    for screenshot in (
        "docs/assets/screenshots/intake.png",
        "docs/assets/screenshots/current-state.png",
        "docs/assets/screenshots/journal.png",
        "docs/assets/screenshots/architecture.png",
    ):
        assert screenshot in closeout

    for dataset_key in ("bank_marketing_uci", "adult_uci", "covertype_uci"):
        assert dataset_key in closeout

    for artifact in (
        "docs/walkthrough.md",
        "docs/sample-outputs.md",
        "docs/benchmark-report-example.md",
        "docs/export-contract.md",
    ):
        assert artifact in closeout

    assert "not a benchmark claim" in closeout


def test_readme_and_roadmap_point_to_v12_closeout() -> None:
    readme = read_project_file("README.md")
    roadmap = read_project_file("docs/roadmap.md")
    sample_outputs = read_project_file("docs/sample-outputs.md")

    assert "## v1.2 polish status" in readme
    assert "v1.2 Product Polish is closed" in readme
    assert "not a benchmark claim" in readme
    assert "[docs/v1-2-polish.md](docs/v1-2-polish.md)" in readme
    assert "v1.2 Product Polish is closed" in roadmap
    assert "## Feature-Generation Decision" in sample_outputs
    assert "train-only" in sample_outputs

    for issue_id in ("COM-5", "COM-6", "COM-7", "COM-8", "COM-9", "COM-18"):
        assert issue_id in roadmap


def test_workbench_copy_surfaces_v12_polish_decisions() -> None:
    app = read_project_file("frontend/src/App.jsx")

    assert "Benchmark progress and implementation readiness" in app
    assert "Feature Decision" in app
    assert "train-only" in app
    assert "cap stays explicit" in app
