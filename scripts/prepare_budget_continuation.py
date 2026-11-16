#!/usr/bin/env python3
"""Fork a completed CPU run with a larger budget, preserving its full state.

This explicit operation creates a new experiment identity. It never resumes,
rewrites, or removes the completed parent and never starts training itself.
"""

from contextlib import contextmanager
from dataclasses import replace
import argparse
import json
import os
from pathlib import Path
import shutil

import torch

from nestrl_research.artifacts import atomic_json, file_hash, load_checkpoint, runtime_metadata, safe_name
from nestrl_research.config import canonical_hash, load_config
from nestrl_research.pipeline import load_l1_pool, registry_spec
from nestrl_research.provenance import source_metadata
from nestrl_research.recovery import load_resume


def assert_same(left, right):
    """Exact comparison, including optimizer/collector/GRU and every RNG byte."""
    if isinstance(left, torch.Tensor):
        if not isinstance(right, torch.Tensor) or left.dtype != right.dtype or not torch.equal(left, right):
            raise ValueError("Continuation tensor state changed")
    elif isinstance(left, dict):
        if not isinstance(right, dict) or left.keys() != right.keys():
            raise ValueError("Continuation state keys changed")
        for key in left:
            assert_same(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        if type(left) is not type(right) or len(left) != len(right):
            raise ValueError("Continuation state sequence changed")
        for a, b in zip(left, right):
            assert_same(a, b)
    elif left != right:
        raise ValueError("Continuation scalar state changed")


@contextmanager
def parent_lock(directory):
    # The original trainer always creates this lock. Open existing bytes only.
    with (directory / ".training.lock").open("r+") as lock:
        if os.name == "nt":
            import msvcrt
            lock.seek(0)
            msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise RuntimeError("Parent has an active training writer") from None
        try:
            yield
        finally:
            if os.name == "nt":
                lock.seek(0)
                msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(lock, fcntl.LOCK_UN)


def prepare(parent_root, parent_run_id, child_root, child_run_id, target_updates,
            expected_config_sha256, expected_snapshot_sha256):
    parent_root, child_root = Path(parent_root).resolve(), Path(child_root).resolve()
    parent = parent_root / safe_name(parent_run_id)
    child_run_id = safe_name(child_run_id)
    if child_root == parent_root or child_root.is_relative_to(parent) or child_run_id == parent_run_id:
        raise ValueError("Continuation requires a distinct output root and run ID")
    if child_root.exists():
        raise FileExistsError("Continuation destination already exists; inspect it instead of overwriting")
    with parent_lock(parent):
        config = load_config(parent / "config.json")
        if config.sha256 != expected_config_sha256:
            raise ValueError("Parent config SHA-256 differs from the approved plan")
        old_updates = config.total_steps // config.batch_size
        if type(target_updates) is not int or target_updates <= old_updates:
            raise ValueError("Target update budget must strictly increase")
        metadata = json.loads((parent / "run.json").read_text())
        if metadata["stage"] not in ("l2", "generalist") or metadata["split"] != "construction":
            raise ValueError("Only construction L2/generalist continuations are supported")
        if metadata["run_id"] != parent_run_id or metadata["config_sha256"] != config.sha256:
            raise ValueError("Parent run identity/config mismatch")
        registry = json.loads((parent_root / "registry.json").read_text())
        if registry != registry_spec(config) or canonical_hash(registry) != metadata["registry_sha256"]:
            raise ValueError("Parent registry mismatch")
        for key in ("implementation_source_sha256", "environment_source_sha256"):
            if metadata[key] != source_metadata()[key]:
                raise ValueError("Parent source mismatch: " + key)
        runtime = runtime_metadata("cpu")
        for key in ("python", "torch", "numpy", "device", "threads", "deterministic_algorithms"):
            if metadata["runtime"][key] != runtime[key]:
                raise ValueError("Parent CPU runtime mismatch: " + key)
        state = load_resume(parent, metadata)
        pointer = json.loads((parent / "resume/latest.json").read_text())
        if pointer["sha256"] != expected_snapshot_sha256 or state["update"] != old_updates:
            raise ValueError("Parent final snapshot SHA-256/update mismatch")
        final, manifest = load_checkpoint(parent / "checkpoints/final.pt", "cpu", config.sha256)
        completion = json.loads((parent / "complete.json").read_text())
        if (completion["sha256"] != manifest["sha256"] or completion["steps"] != config.total_steps
                or manifest["steps"] != config.total_steps):
            raise ValueError("Parent completion budget/weight mismatch")
        for key in ("run_id", "stage", "split", "seed", "ego_side", "registry_sha256", "training_partners"):
            if manifest[key] != metadata[key]:
                raise ValueError("Parent final manifest mismatch: " + key)
        assert_same(state["model"], final["model"])
        assert_same(state["optimizer"], final["optimizer"])
        metrics = [json.loads(line) for line in (parent / "metrics.jsonl").read_text().splitlines()]
        if ([m["update"] for m in metrics] != list(range(1, old_updates + 1))
                or any(m["steps"] != m["update"] * config.batch_size for m in metrics)):
            raise ValueError("Parent metrics budget/order mismatch")
        if metadata["stage"] == "l2":
            _, partners = load_l1_pool(config, parent_root, "construction", "cpu")
            if partners != metadata["training_partners"]:
                raise ValueError("Parent frozen partner provenance mismatch")
        watched = [parent / p for p in ("config.json", "run.json", "metrics.jsonl", "complete.json",
                    "resume/latest.json", "checkpoints/final.pt", "checkpoints/final.json", "checkpoints/latest.json")]
        watched += [parent / "resume" / pointer["file"], parent_root / "registry.json"]
        before = {p: file_hash(p) for p in watched}
        new_config = replace(config, total_steps=target_updates * config.batch_size)
        new_config.validate()
        if {k:v for k,v in config.to_dict().items() if k != "total_steps"} != {
                k:v for k,v in new_config.to_dict().items() if k != "total_steps"}:
            raise ValueError("Only the total budget may change")
        child_root.mkdir(parents=True)
        directory = child_root / child_run_id
        directory.mkdir()
        atomic_json(child_root / "config.json", new_config.to_dict())
        atomic_json(directory / "config.json", new_config.to_dict())
        new_registry = registry_spec(new_config)
        atomic_json(child_root / "registry.json", new_registry)
        if metadata["stage"] == "l2":
            catalog = Path(config.l1_pool.catalog)
            shutil.copytree(parent_root / "partners", child_root / "partners")
            if (child_root / catalog).exists():
                if file_hash(child_root / catalog) != file_hash(parent_root / catalog):
                    raise ValueError("Copied catalog differs")
            else:
                (child_root / catalog).parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(parent_root / catalog, child_root / catalog)
            _, partners = load_l1_pool(new_config, child_root, "construction", "cpu")
            if partners != metadata["training_partners"]:
                raise ValueError("Child frozen partner provenance mismatch")
        new_metadata = {**metadata, "config_sha256": new_config.sha256,
                        "registry_sha256": canonical_hash(new_registry), "run_id": child_run_id}
        atomic_json(directory / "run.json", new_metadata)
        shutil.copyfile(parent / "metrics.jsonl", directory / "metrics.jsonl")
        folder = directory / "resume"
        folder.mkdir()
        snapshot = folder / pointer["file"]
        torch.save({**state, "metadata": new_metadata}, snapshot)
        atomic_json(folder / "latest.json", {**pointer, "sha256": file_hash(snapshot)})
        # Authenticate/read back using the normal resume loader before publishing readiness.
        copied = load_resume(directory, new_metadata)
        for key in state:
            if key != "metadata":
                assert_same(state[key], copied[key])
        if before != {p:file_hash(p) for p in watched}:
            raise ValueError("Parent changed while preparing continuation")
        lineage = {"schema":"re-nestrl-budget-continuation-v1", "parent_run_id":parent_run_id,
            "parent_config_sha256":config.sha256, "parent_snapshot_sha256":pointer["sha256"],
            "parent_final_sha256":manifest["sha256"], "parent_update":old_updates,
            "child_run_id":child_run_id, "child_config_sha256":new_config.sha256,
            "child_snapshot_sha256":file_hash(snapshot), "target_updates":target_updates,
            "total_steps":new_config.total_steps, "additional_updates":target_updates-old_updates,
            "state_equal_except_explicit_config_registry_run_metadata":True,
            "parent_files_unchanged":True, "backend":"cpu", "training_started":False,
            "command_source_sha256":file_hash(Path(__file__))}
        atomic_json(directory / "continuation.json", lineage)
        atomic_json(child_root / "prepared.json", lineage)
        return lineage


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--parent-root", required=True, type=Path)
    p.add_argument("--parent-run-id", required=True)
    p.add_argument("--child-root", required=True, type=Path)
    p.add_argument("--child-run-id", required=True)
    p.add_argument("--target-updates", required=True, type=int)
    p.add_argument("--expected-config-sha256", required=True)
    p.add_argument("--expected-snapshot-sha256", required=True)
    args = p.parse_args()
    torch.set_num_threads(1)
    print(json.dumps(prepare(**vars(args)), sort_keys=True))


if __name__ == "__main__":
    main()
