"""Single-writer runs, atomic JSON, and hash-verified checkpoint bundles."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import socket
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch

from .config import SCHEMA, canonical_hash


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def safe_name(value: str) -> str:
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.-]{0,95}", value) or value in (".", ".."):
        raise ValueError("Run IDs must be simple names without path separators")
    return value


@contextmanager
def new_run(root: Path, run_id: str):
    directory = root / safe_name(run_id)
    directory.mkdir(parents=True, exist_ok=False)
    lock = directory / ".writer.lock"
    with lock.open("x", encoding="utf-8") as stream:
        json.dump({"host": socket.gethostname(), "pid": os.getpid()}, stream)
    try:
        yield directory
    except BaseException as error:
        atomic_json(directory / "failure.json", {"type": type(error).__name__, "message": str(error)})
        raise
    finally:
        lock.unlink(missing_ok=True)


def runtime_metadata(device: str) -> dict:
    return {"python": platform.python_version(), "platform": platform.platform(), "torch": torch.__version__,
            "numpy": np.__version__, "device": device, "host": socket.gethostname(), "threads": torch.get_num_threads(),
            "cuda_runtime": torch.version.cuda, "deterministic_algorithms": torch.are_deterministic_algorithms_enabled()}


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is not available")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS is not available")
    return torch.device(requested)


def save_checkpoint(directory: Path, model, optimizer, metadata: dict) -> Path:
    checkpoints = directory / "checkpoints"
    checkpoints.mkdir(exist_ok=True)
    path = checkpoints / "final.pt"
    temporary = checkpoints / "final.pt.tmp"
    # This format supports inference transfer. Exact trajectory resume is deliberately unsupported.
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict()}, temporary)
    os.replace(temporary, path)
    manifest = {**metadata, "schema": SCHEMA, "checkpoint": path.name, "sha256": file_hash(path)}
    atomic_json(path.with_suffix(".json"), manifest)
    atomic_json(checkpoints / "latest.json", {"manifest": "final.json", "manifest_sha256": file_hash(path.with_suffix(".json"))})
    return path


def load_checkpoint(path: Path, device: torch.device, expected_config: str | None = None) -> tuple[dict, dict]:
    path = Path(path)
    manifest_path = path.with_suffix(".json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    latest = json.loads((path.parent / "latest.json").read_text(encoding="utf-8"))
    if latest != {"manifest": manifest_path.name, "manifest_sha256": file_hash(manifest_path)}:
        raise ValueError("Checkpoint latest.json does not authenticate the manifest")
    if manifest.get("schema") != SCHEMA or manifest.get("checkpoint") != path.name or file_hash(path) != manifest.get("sha256"):
        raise ValueError("Checkpoint schema/name/SHA-256 mismatch")
    if expected_config is not None and manifest.get("config_sha256") != expected_config:
        raise ValueError("Checkpoint belongs to a different experiment configuration")
    return torch.load(path, map_location=device, weights_only=True), manifest


def freeze_manifest(path: Path, value: dict) -> str:
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != value:
            raise ValueError(f"Frozen manifest differs: {path}")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
    return canonical_hash(value)
