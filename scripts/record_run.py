#!/usr/bin/env python3
"""Record an explicitly supplied command and its inputs without choosing an experiment."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def source_manifest(root: Path) -> dict[str, str]:
    names = subprocess.check_output(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        cwd=root,
    ).decode().split("\0")
    selected = {}
    for name in sorted(set(filter(None, names))):
        path = root / name
        if path.is_file() and not path.is_symlink() and (
            name.startswith(("src/", "scripts/", "configs/", "constraints/"))
            or name in {"pyproject.toml", ".gitattributes"}
        ):
            selected[name] = digest(path.read_bytes())
    return selected


def record_run(root: Path, output: Path, run_id: str, config: Path, command: list[str]) -> int:
    config_bytes = config.read_bytes()
    config_value = json.loads(config_bytes)
    source = source_manifest(root)
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    packages = sorted({(d.metadata["Name"], d.version) for d in importlib.metadata.distributions()})
    output.mkdir(parents=True, exist_ok=False)
    (output / "config.json").write_bytes(config_bytes)
    for name, expected in source.items():
        content = (root / name).read_bytes()
        if digest(content) != expected:
            raise RuntimeError("Source changed while recording its snapshot")
        target = output / "source" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
    payload = {
        "schema_version": 1,
        "run_id": run_id,
        "git_commit": head,
        "git_working_tree_clean": not bool(subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=root, text=True
        ).strip()),
        "source_sha256": source,
        "source_snapshot_directory": "source",
        "config_file_sha256": digest(config_bytes),
        "config_canonical_sha256": digest(json.dumps(
            config_value, sort_keys=True, ensure_ascii=False, separators=(",", ":")
        ).encode()),
        "command": command,
        "command_cwd": "repository_root",
        "python": platform.python_version(),
        "operating_system": platform.system(),
        "architecture": platform.machine(),
        "distributions": dict(packages),
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "state": "running",
    }

    def save() -> None:
        temporary = output / "run.json.tmp"
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temporary.replace(output / "run.json")

    save()
    try:
        with (output / "stdout.log").open("wb") as stdout, (output / "stderr.log").open("wb") as stderr:
            result = subprocess.run(command, cwd=root, stdout=stdout, stderr=stderr, check=False)
        code = result.returncode
    except OSError as error:
        code = 127
        payload["error"] = str(error)
    except BaseException:
        payload["state"] = "interrupted"
        payload["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        save()
        raise
    payload["return_code"] = code
    payload["source_unchanged"] = source == source_manifest(root)
    payload["config_unchanged"] = config.is_file() and config_bytes == config.read_bytes()
    payload["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    payload["state"] = "completed" if code == 0 else "failed"
    if not payload["source_unchanged"] or not payload["config_unchanged"]:
        payload["state"] = "inputs_changed"
        code = code or 1
    save()
    return code


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New run directory; never overwritten")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("supply the experiment command after --")
    if sys.version_info[:2] != (3, 12) or Path(sys.prefix).resolve() != (ROOT / ".venv").resolve():
        parser.error("activate this repository's Python 3.12 .venv")
    return record_run(ROOT, args.output.resolve(), args.run_id, args.config.resolve(), command)


if __name__ == "__main__":
    raise SystemExit(main())
