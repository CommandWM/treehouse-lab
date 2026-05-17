from __future__ import annotations

import json
import tomllib
from pathlib import Path

import treehouse_lab
from treehouse_lab import api
from treehouse_lab.exporting import _fastapi_app_template


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_version_metadata_matches_current_checkpoint_docs() -> None:
    pyproject = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    frontend_package = json.loads((PROJECT_ROOT / "frontend" / "package.json").read_text(encoding="utf-8"))
    frontend_lock = json.loads((PROJECT_ROOT / "frontend" / "package-lock.json").read_text(encoding="utf-8"))
    package_version = pyproject["project"]["version"]
    checkpoint = f"v{package_version}"

    assert package_version == "1.2.0"
    assert treehouse_lab.__version__ == package_version
    assert api.app.version == package_version
    assert frontend_package["version"] == package_version
    assert frontend_lock["version"] == package_version
    assert frontend_lock["packages"][""]["version"] == package_version
    assert f'version="{package_version}"' in _fastapi_app_template()
    assert f"Current package version: `{checkpoint}`" in (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
    assert f"Implementation baseline: `{checkpoint}`" in (PROJECT_ROOT / "docs" / "roadmap.md").read_text(
        encoding="utf-8"
    )
