#!/bin/sh
# Restore the common runtime; experiments are specified separately.
set -eu
project_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$project_root"
python312=${NESTRL_PYTHON:-python3.12}
"$python312" -c 'import sys; assert sys.version_info[:2] == (3, 12), "Python 3.12 required"'
if [ ! -x .venv/bin/python ]; then
  "$python312" -m venv .venv
fi
. .venv/bin/activate
python -c 'from pathlib import Path; import sys; assert Path(sys.prefix).resolve() == (Path.cwd()/".venv").resolve()'
export PIP_CONSTRAINT="$project_root/constraints/cloud-cpu-py312.txt"
python -m pip install 'pip==25.0.1' 'setuptools==84.2.1'
python - <<'PY'
from pathlib import Path
import subprocess
import sys
pins = [line for line in Path("constraints/cloud-cpu-py312.txt").read_text().splitlines()
        if line and not line.startswith(("#", "torch=="))]
subprocess.run([sys.executable, "-m", "pip", "install", *pins], check=True)
PY
python -m pip install --index-url https://download.pytorch.org/whl/cpu 'torch==2.15.0+cpu'
python -m pip install -e .
python scripts/bootstrap_upstream.py
python scripts/check_workspace.py --device cpu --allow-dirty
