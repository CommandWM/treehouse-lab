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

# The comparison harness only imports autogluon.tabular, so keep the benchmark
# environment intentionally narrower than a full AutoGluon meta-package install.
"$ENV_DIR/bin/python" -m pip install autogluon.tabular

echo "Benchmark environment ready at $ENV_DIR"
echo "Use: $ENV_DIR/bin/python -m treehouse_lab.cli compare ..."
