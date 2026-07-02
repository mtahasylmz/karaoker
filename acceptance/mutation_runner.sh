#!/usr/bin/env bash
# gherkin-mutator adapter: regenerate acceptance tests from the (possibly
# mutated) feature files, then run them. Exit code = suite result, which is
# what the mutator scores. Leans on .acceptance-cache/ so surviving unmutated
# steps stay warm.
set -euo pipefail
cd "$(dirname "$0")/.."
uv run python acceptance/generate.py
exec uv run pytest acceptance/generated -q -x
