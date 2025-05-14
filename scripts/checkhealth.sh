#!/bin/sh
project_root=$(dirname $(dirname "$0"))
. "$project_root/.venv/bin/activate"
# Actual checkhealth
echo "Running tests..."
pytest
echo "Running ruff..."
ruff check
# echo "Running mypy..."
# mypy
