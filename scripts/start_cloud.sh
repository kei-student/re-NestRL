#!/bin/sh
# Validate the prepared runtime. Do not start/resume training here.
set -eu
project_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$project_root"
. .venv/bin/activate
python scripts/check_workspace.py --device cpu
