#!/bin/bash
set -e

export PATH="$HOME/.local/bin:$PATH"

uv sync --dev --extra lint
uv run codespell app
find app -maxdepth 1 -name '*.py' -exec uv run yapf --in-place {} +
uv run mypy app
