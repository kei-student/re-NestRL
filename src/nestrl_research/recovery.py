"""Atomic, hash-checked CPU training snapshots and single-writer recovery."""

from contextlib import contextmanager
import json
import os
from pathlib import Path
import random
import socket

import numpy as np
import torch

if os.name == "nt":
    import msvcrt
else:
    import fcntl

from .artifacts import atomic_json, file_hash, safe_name


def sync_directory(path):
    if os.name == "nt":
        return  # Windows does not expose directory fsync through os.open.
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


@contextmanager
def training_run(root, run_id, resume=False):
    """flock releases on process death; a stale PID file never authorizes writes."""
    directory = Path(root) / safe_name(run_id)
    if resume:
        if not (directory / "run.json").is_file():
            raise ValueError("No existing training run to resume")
    else:
        directory.mkdir(parents=True, exist_ok=False)
    with (directory / ".training.lock").open("a+") as lock:
        try:
            if os.name == "nt":
                if lock.tell() == 0:
                    lock.write("0")
                    lock.flush()
                lock.seek(0)
                msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (BlockingIOError, PermissionError):
            raise RuntimeError("Training run already has an active writer") from None
        try:
            if (directory / "complete.json").exists():
                raise ValueError("Completed training runs cannot be resumed")
            atomic_json(directory / ".writer.lock", {"pid": os.getpid(), "host": socket.gethostname()})
            try:
                yield directory
            except BaseException as error:
                atomic_json(directory / "failure.json", {"type": type(error).__name__, "message": str(error)})
                raise
            finally:
                (directory / ".writer.lock").unlink(missing_ok=True)
        finally:
            if os.name == "nt":
                lock.seek(0)
                msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(lock, fcntl.LOCK_UN)


def rng_state():
    numpy = np.random.get_state()
    return {"python": random.getstate(), "torch": torch.get_rng_state(),
            "numpy": (numpy[0], torch.from_numpy(numpy[1].astype(np.int64)), *numpy[2:])}


def restore_rng(state):
    random.setstate(state["python"])
    torch.set_rng_state(state["torch"])
    numpy = state["numpy"]
    np.random.set_state((numpy[0], numpy[1].numpy().astype(np.uint32), *numpy[2:]))


def save_resume(directory, model, optimizer, collector, metadata, update, elapsed):
    folder = directory / "resume"
    folder.mkdir(exist_ok=True)
    path = folder / f"update-{update:06d}.pt"
    temporary = path.with_suffix(".pt.tmp")
    payload = {"schema": "re-nestrl-resume-v1", "metadata": metadata, "update": update,
               "elapsed_seconds": elapsed, "model": model.state_dict(),
               "optimizer": optimizer.state_dict(), "collector": collector.state_dict(), "rng": rng_state()}
    with temporary.open("wb") as stream:
        torch.save(payload, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    # Publish only after the entire checkpoint is durable. Keep the preceding
    # generation too; an interrupted save never invalidates the published one.
    pointer = folder / "latest.json"
    atomic_json(pointer, {"file": path.name, "sha256": file_hash(path), "update": update})
    with pointer.open("rb") as stream:
        os.fsync(stream.fileno())
    sync_directory(folder)
    for old in sorted(folder.glob("update-*.pt"))[:-2]:
        old.unlink()
    return pointer


def load_resume(directory, expected):
    pointer = json.loads((directory / "resume" / "latest.json").read_text())
    name = pointer["file"]
    if name != f"update-{pointer['update']:06d}.pt":
        raise ValueError("Invalid resume checkpoint name")
    path = directory / "resume" / name
    if file_hash(path) != pointer["sha256"]:
        raise ValueError("Resume checkpoint SHA-256 mismatch")
    state = torch.load(path, map_location="cpu", weights_only=True)
    if state["schema"] != "re-nestrl-resume-v1" or state["update"] != pointer["update"]:
        raise ValueError("Invalid resume checkpoint schema/update")
    saved = state["metadata"]
    for key in ("config_sha256", "registry_sha256", "implementation_source_sha256",
                "environment_source_sha256", "stage", "split", "seed", "ego_side", "run_id", "training_partners"):
        if saved[key] != expected[key]:
            raise ValueError(f"Resume checkpoint differs: {key}")
    for key in ("python", "torch", "numpy", "device", "threads", "deterministic_algorithms"):
        if saved["runtime"][key] != expected["runtime"][key]:
            raise ValueError(f"Resume runtime differs: {key}")
    return state


def recover_metrics(directory, update):
    """Discard a partial/uncommitted tail, never duplicate updates after resume."""
    path = directory / "metrics.jsonl"
    committed = []
    for line in path.read_text().splitlines():
        try:
            metric = json.loads(line)
        except json.JSONDecodeError:
            break
        if metric["update"] > update:
            break
        committed.append(metric)
    if [m["update"] for m in committed] != list(range(1, update + 1)):
        raise ValueError("Metrics are missing committed updates")
    temporary = path.with_suffix(".jsonl.tmp")
    temporary.write_text("".join(json.dumps(m, allow_nan=False) + "\n" for m in committed))
    os.replace(temporary, path)
