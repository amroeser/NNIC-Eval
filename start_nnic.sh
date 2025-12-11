#!/usr/bin/env bash
set -e

# Simple start script for NNIC Eval on macOS/Linux
# - creates/uses .venv
# - installs requirements
# - runs all experiments (all datasets x all splits) with multiple runs per config

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$PROJECT_ROOT/.venv"

if [ ! -d "$VENV_DIR" ]; then
  python3 -m venv "$VENV_DIR"
fi

# shellcheck disable=SC1090
source "$VENV_DIR/bin/activate"

pip install --upgrade pip
pip install -r "$PROJECT_ROOT/requirements.txt"

python "$PROJECT_ROOT/run_all_experiments.py"

# Run significance analysis (paired tests over multi-run accuracies) and
# write results/significance_report.md
(
  cd "$PROJECT_ROOT" && python -m analysis.significance
)
