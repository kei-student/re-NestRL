#!/usr/bin/env bash
# Idempotent dependency setup and checks only. Never starts or resumes a run.
set -euo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"
"${NESTRL_PYTHON:-python3.12}" -c 'import sys; assert sys.version_info[:2] == (3, 12), "Python 3.12 required"'
if [[ ! -e .venv ]]; then
  "${NESTRL_PYTHON:-python3.12}" -m venv .venv
fi
.venv/bin/python - <<'PY'
import importlib.metadata as metadata
import pathlib
import sys
assert sys.version_info[:2] == (3, 12)
assert pathlib.Path(sys.prefix).resolve() == pathlib.Path('.venv').resolve()
names = {d.metadata['Name'].lower().replace('_', '-') for d in metadata.distributions()}
assert 'nestrl' not in names, 'Use a dedicated re-NestRL venv; do not mix with NestRL'
PY
constraint=requirements/constraints-cpu-py312.txt
.venv/bin/python -m pip install -c "$constraint" 'pip==25.0.1' 'setuptools==78.1.0'
.venv/bin/python -m pip install -c "$constraint" --index-url https://download.pytorch.org/whl/cpu 'torch==2.15.0+cpu'
.venv/bin/python -m pip install -c "$constraint" --no-build-isolation -e .
.venv/bin/python -m pip check
.venv/bin/python scripts/bootstrap_upstream.py
.venv/bin/python scripts/source_manifest.py
.venv/bin/python scripts/check_workspace.py --device auto --allow-dirty
