#!/usr/bin/env bash
# Reusable start instructions: validate only; experiment commands are separate.
set -euo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"
.venv/bin/python -m pip check
.venv/bin/python scripts/source_manifest.py
.venv/bin/python scripts/check_workspace.py --device auto
