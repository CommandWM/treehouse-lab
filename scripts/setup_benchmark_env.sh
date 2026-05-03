#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${1:-python3.12}"
ENV_DIR="${TREEHOUSE_LAB_BENCHMARK_ENV:-.venv-benchmarks}"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Python interpreter not found: $PYTHON_BIN" >&2
  exit 1
fi

"$PYTHON_BIN" -m venv "$ENV_DIR"
"$ENV_DIR/bin/python" -m pip install --upgrade pip setuptools wheel
"$ENV_DIR/bin/python" -m pip install -e ".[dev,llm]"

# Keep external benchmark dependencies out of the core package. This environment
# is intentionally for optional comparison runners only.
"$ENV_DIR/bin/python" -m pip install autogluon.tabular
"$ENV_DIR/bin/python" -m pip install flaml

echo "Benchmark environment ready at $ENV_DIR"
echo "Use: $ENV_DIR/bin/python -m treehouse_lab.cli compare ..."
