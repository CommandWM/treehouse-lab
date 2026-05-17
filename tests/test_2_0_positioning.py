from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def read_project_file(relative_path: str) -> str:
    return (PROJECT_ROOT / relative_path).read_text(encoding="utf-8")


def test_core_docs_center_2_0_contract_first_positioning() -> None:
    readme = read_project_file("README.md")
    agents = read_project_file("AGENTS.md")
    roadmap = read_project_file("docs/roadmap.md")
    vision = read_project_file("docs/vision-2.0.md")
    contracts = read_project_file("docs/contracts.md")

    for text in (readme, agents, roadmap, vision, contracts):
        assert "contract-first" in text
        assert "audit-first" in text

    assert "agent-accessible, not agent-dependent" in readme
    assert "classification and regression" in readme
    assert "classification and regression support" in vision
    assert "--task regression" in roadmap
    assert "not broad AutoML" in agents
    assert "broad AutoML" in vision
    assert "production serving infrastructure" in roadmap
    assert "Agents should consume public contracts" in contracts
    assert "The current implementation supports binary and multiclass classification" in readme


def test_stale_v12_closeout_is_not_publicly_linked() -> None:
    assert not (PROJECT_ROOT / "docs" / "v1-2-polish.md").exists()

    for relative_path in ("README.md", "docs/roadmap.md"):
        text = read_project_file(relative_path)
        assert "v1.2 Product Polish" not in text
        assert "docs/v1-2-polish.md" not in text
        assert "COM-5" not in text
        assert "COM-18" not in text


def test_cli_contract_distinguishes_implemented_from_planned_commands() -> None:
    cli_doc = read_project_file("docs/cli.md")

    for command in (
        "baseline",
        "candidate",
        "propose",
        "diagnose",
        "loop",
        "compare",
        "benchmark-suite",
        "export",
    ):
        assert f"`treehouse-lab {command}" in cli_doc

    assert "`init` is planned" in cli_doc
    assert "`profile` is planned" in cli_doc
    assert "`--task regression` is planned" in cli_doc
    assert "`serve` is planned" in cli_doc
