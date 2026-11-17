#!/usr/bin/env python3
"""Fork a paused CPU L2 into a smaller total budget, without changing its state."""
import argparse
from contextlib import contextmanager
from dataclasses import replace
import json
import os
from pathlib import Path
import shutil

import torch

from nestrl_research.artifacts import atomic_json, file_hash, runtime_metadata, safe_name
from nestrl_research.config import canonical_hash, load_config
from nestrl_research.pipeline import load_l1_pool, registry_spec
from nestrl_research.provenance import source_metadata
from nestrl_research.recovery import load_resume


def assert_same(a, b):
    if isinstance(a, torch.Tensor):
        if not isinstance(b, torch.Tensor) or a.dtype != b.dtype or not torch.equal(a, b):
            raise ValueError("Budget handoff changed a tensor")
    elif isinstance(a, dict):
        if not isinstance(b, dict) or a.keys() != b.keys():
            raise ValueError("Budget handoff changed state keys")
        for key in a:
            assert_same(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        if type(a) is not type(b) or len(a) != len(b):
            raise ValueError("Budget handoff changed a sequence")
        for left, right in zip(a, b):
            assert_same(left, right)
    elif a != b:
        raise ValueError("Budget handoff changed a scalar")


@contextmanager
def parent_lock(directory):
    with (directory/".training.lock").open("r+") as lock:
        if os.name == "nt":
            import msvcrt
            lock.seek(0)
            msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            try:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise RuntimeError("Parent still has an active writer") from None
        try:
            yield
        finally:
            if os.name == "nt":
                lock.seek(0)
                msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def partner_provenance(config, root):
    _, partners = load_l1_pool(config, root, "construction", "cpu")
    if config.l2_mixture is not None:
        partners = [{**p, "level":"l1"} for p in partners] + [
            {"id":f"l0-{1-config.ego_side}-{r}", "level":"l0", "recipe":r,
             "side":1-config.ego_side, "frozen":True} for r in config.l2_mixture.l0_recipes]
    return partners


def prepare(parent_root, parent_run_id, child_root, child_run_id, target_updates,
            expected_config_sha256, expected_snapshot_sha256):
    parent_root, child_root = Path(parent_root).resolve(), Path(child_root).resolve()
    parent = parent_root/safe_name(parent_run_id)
    safe_name(child_run_id)
    if child_root == parent_root or child_root.is_relative_to(parent_root) or child_run_id == parent_run_id:
        raise ValueError("Use a distinct child root and run ID")
    if child_root.exists():
        raise FileExistsError("Child exists; inspect without overwriting or initializing again")
    with parent_lock(parent):
        config = load_config(parent/"config.json")
        if config.sha256 != expected_config_sha256 or config.l1_pool is None:
            raise ValueError("Parent config differs or lacks authenticated frozen L1s")
        old_budget = config.total_steps//config.batch_size
        if type(target_updates) is not int or not 0 < target_updates < old_budget:
            raise ValueError("Target must be a positive, smaller total budget")
        metadata = json.loads((parent/"run.json").read_text())
        registry = json.loads((parent_root/"registry.json").read_text())
        if (metadata["stage"] != "l2" or metadata["split"] != "construction"
                or metadata["run_id"] != parent_run_id or metadata["config_sha256"] != config.sha256
                or registry != registry_spec(config) or canonical_hash(registry) != metadata["registry_sha256"]):
            raise ValueError("Parent identity/config/registry differs")
        current_source = source_metadata()
        for key in ("implementation_source_sha256","environment_source_sha256"):
            if metadata[key] != current_source[key]:
                raise ValueError("Parent source differs: "+key)
        current_runtime = runtime_metadata("cpu")
        for key in ("python","torch","numpy","device","threads","deterministic_algorithms"):
            if metadata["runtime"][key] != current_runtime[key]:
                raise ValueError("Parent CPU runtime differs: "+key)
        state = load_resume(parent, metadata)
        assert_same(state["metadata"],metadata)
        pointer = json.loads((parent/"resume/latest.json").read_text())
        start_update = state["update"]
        if pointer["sha256"] != expected_snapshot_sha256:
            raise ValueError("Parent snapshot SHA differs")
        if not 0 <= start_update <= target_updates or (parent/"complete.json").exists():
            raise ValueError("Saved update already exceeds target or parent is complete")
        partners = partner_provenance(config,parent_root)
        if partners != metadata["training_partners"]:
            raise ValueError("Parent frozen partner provenance differs")
        lines = (parent/"metrics.jsonl").read_bytes().splitlines(keepends=True)
        metrics = [json.loads(line) for line in lines]
        if (len(metrics) < start_update or len(metrics) > old_budget
                or [m["update"] for m in metrics] != list(range(1,len(metrics)+1))
                or any(m["steps"] != m["update"]*config.batch_size for m in metrics)):
            raise ValueError("Parent metrics have an invalid prefix")
        watched = [p for p in parent.rglob("*") if p.is_file()]
        watched += [parent_root/"registry.json"]
        watched += [p for p in (parent_root/"partners").rglob("*") if p.is_file()]
        watched += [parent_root/n for n in ("config.json","selection.json","origin-selection.json")
                    if (parent_root/n).is_file()]
        before = {p:file_hash(p) for p in watched}
        new_config = replace(config,total_steps=target_updates*config.batch_size)
        new_config.validate()
        left,right=config.to_dict(),new_config.to_dict()
        left.pop("total_steps");right.pop("total_steps")
        if left != right:
            raise ValueError("Only total_steps may change")
        child_root.mkdir(parents=True)
        child = child_root/child_run_id
        child.mkdir()
        atomic_json(child_root/"config.json",new_config.to_dict())
        atomic_json(child/"config.json",new_config.to_dict())
        new_registry=registry_spec(new_config)
        atomic_json(child_root/"registry.json",new_registry)
        shutil.copytree(parent_root/"partners",child_root/"partners")
        if partner_provenance(new_config,child_root) != partners:
            raise ValueError("Child frozen partners differ")
        new_metadata={**metadata,"run_id":child_run_id,"config_sha256":new_config.sha256,
                      "registry_sha256":canonical_hash(new_registry)}
        atomic_json(child/"run.json",new_metadata)
        (child/"metrics.jsonl").write_bytes(b"".join(lines[:start_update]))
        folder=child/"resume";folder.mkdir()
        snapshot=folder/pointer["file"]
        torch.save({**state,"metadata":new_metadata},snapshot)
        atomic_json(folder/"latest.json",{**pointer,"sha256":file_hash(snapshot)})
        copied=load_resume(child,new_metadata)
        for key in state:
            if key != "metadata":
                assert_same(state[key],copied[key])
        for name in ("selection.json","origin-selection.json"):
            path=parent_root/name
            if path.exists():
                shutil.copyfile(path,child_root/("parent-selection.json" if name=="selection.json" else name))
        if (parent_root/"selection.json").exists():
            selection=json.loads((parent_root/"selection.json").read_text())
            atomic_json(child_root/"selection.json",{
                **selection,"config_sha256":new_config.sha256,
                "config_file_sha256":file_hash(child_root/"config.json"),
                "updates":target_updates,"steps":new_config.total_steps,
                "budget_override":{"parent_config_sha256":config.sha256,"old_updates":old_budget,
                                   "target_updates":target_updates,"reason":"user requested smaller total budget"},
                "training_started":False})
        if before != {p:file_hash(p) for p in watched}:
            raise ValueError("Parent files changed during handoff")
        lineage={
            "schema":"re-nestrl-budget-reduction-v1","parent_run_id":parent_run_id,
            "parent_config_sha256":config.sha256,"parent_snapshot_sha256":pointer["sha256"],
            "parent_planned_updates":old_budget,"parent_saved_update":start_update,
            "child_run_id":child_run_id,"child_config_sha256":new_config.sha256,
            "child_registry_sha256":canonical_hash(new_registry),"child_snapshot_sha256":file_hash(snapshot),
            "target_updates":target_updates,"target_steps":new_config.total_steps,
            "remaining_updates":target_updates-start_update,
            "parent_metric_tail_excluded":len(metrics)-start_update,
            "state_equal_except_config_registry_run_metadata":True,"parent_files_unchanged":True,
            "backend":"cpu","training_started":False,"driver_sha256":file_hash(Path(__file__))}
        atomic_json(child/"budget-change.json",lineage)
        atomic_json(child_root/"prepared.json",lineage)
        return lineage


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("parent-root","child-root"):
        p.add_argument("--"+name,type=Path,required=True)
    for name in ("parent-run-id","child-run-id","expected-config-sha256","expected-snapshot-sha256"):
        p.add_argument("--"+name,required=True)
    p.add_argument("--target-updates",type=int,required=True)
    args=p.parse_args()
    torch.set_num_threads(1)
    print(json.dumps(prepare(**vars(args)),indent=2))


if __name__=="__main__":
    main()
