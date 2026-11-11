#!/usr/bin/env python3
"""Stage hash-verified construction L1s for L2, preserving original artifacts."""

import argparse
import json
from pathlib import Path, PurePosixPath
import zipfile

import numpy as np
import torch

from nestrl_research.artifacts import atomic_json, file_hash
from nestrl_research.config import ROOT, load_config
from nestrl_research.pipeline import ensure_registry, load_l1_pool
from nestrl_research.rollout import FrozenPartners


def prepare(config_path, index_path, archives_dir, output_root):
    config = load_config(config_path)
    index = json.loads(Path(index_path).read_text())
    rows = {row["seed"]: row for row in index["models"]}
    if len(rows) != len(index["models"]) or not set(config.construction_seeds).issubset(rows):
        raise ValueError("Missing or duplicate construction L1s in model index")
    planned, partners, registry_bytes = {}, [], None
    archives_dir, output_root = Path(archives_dir), Path(output_root)
    for seed in config.construction_seeds:
        row = rows[seed]
        if row["stage"] != "l1" or row["split"] != "construction" or row["config_sha256"] != config.sha256:
            raise ValueError(f"Incompatible construction policy/config: {seed}")
        archive_name = row["archive"]
        if Path(archive_name).name != archive_name:
            raise ValueError("Unsafe archive filename")
        archive = archives_dir / archive_name
        if file_hash(archive) != row["archive_sha256"]:
            raise ValueError(f"Archive SHA-256 mismatch: {archive.name}")
        with zipfile.ZipFile(archive) as bundle:
            checkpoint = PurePosixPath(row["archive_checkpoint_path"])
            if checkpoint.is_absolute() or ".." in checkpoint.parts:
                raise ValueError("Unsafe checkpoint path")
            run = checkpoint.parent.parent
            registry_path = str(run.parent / "registry.json")
            candidate = bundle.read(registry_path)
            if registry_bytes is not None and json.loads(candidate) != json.loads(registry_bytes):
                raise ValueError("L1 registries differ")
            registry_bytes = candidate
            for name in ("checkpoints/final.pt", "checkpoints/final.json", "checkpoints/latest.json",
                         "config.json", "run.json", "complete.json"):
                planned[Path(f"l1-construction-{seed}") / name] = bundle.read(str(run / name))
        partners.append({"seed": seed, "archive": archive_name, "archive_sha256": row["archive_sha256"],
                         "checkpoint_sha256": row["checkpoint_sha256"], "source_commit": row["source_commit"]})
    from nestrl_research.pipeline import registry_spec
    if json.loads(registry_bytes) != registry_spec(config):
        raise ValueError("L1 registry differs from target configuration")
    # Validate every destination before writing; repeat preparation is additive.
    for relative, contents in planned.items():
        path = output_root / relative
        if path.exists() and path.read_bytes() != contents:
            raise ValueError(f"Existing artifact differs; will not overwrite: {path}")
    ensure_registry(config, output_root)
    for relative, contents in planned.items():
        path = output_root / relative
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("xb") as stream:
                stream.write(contents)
    models, provenance = load_l1_pool(config, output_root, "construction", "cpu")
    for seed, item in zip(config.construction_seeds, provenance):
        directory = output_root / f"l1-construction-{seed}"
        complete = json.loads((directory / "complete.json").read_text())
        final = json.loads((directory / "checkpoints/final.json").read_text())
        if (complete["steps"] != config.total_steps or final["steps"] != config.total_steps
                or complete["sha256"] != item["checkpoint_sha256"]
                or rows[seed]["checkpoint_sha256"] != item["checkpoint_sha256"]):
            raise ValueError(f"Incomplete or mismatched L1: {seed}")
    before = [{k: v.clone() for k, v in model.state_dict().items()} for model in models]
    pool = FrozenPartners(config, config.l1_ego_side, "cpu", np.random.default_rng(7), models)
    actions = pool.act(np.zeros((config.num_envs, 105), dtype=np.float32), [])
    assert actions.shape == (config.num_envs,) and np.all((actions >= 0) & (actions < 6))
    for model, original in zip(models, before):
        assert not model.training and all(not p.requires_grad for p in model.parameters())
        assert all(torch.equal(value, original[k]) for k, value in model.state_dict().items())
    report = {"config_sha256": config.sha256, "learner_side": config.ego_side,
              "partner_side": config.l1_ego_side, "partner_seeds": list(config.construction_seeds),
              "partner_count": len(models), "all_frozen": True, "batched_inference": "passed",
              "source_release_id": index.get("release_id"), "source_commit": index.get("source_commit"),
              "partners": partners, "provenance": provenance,
              "held_out_evaluation": "Separate validation/test L1s have not been staged; do not evaluate generalization on construction L1s."}
    atomic_json(output_root / "l2-preparation.json", report)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=ROOT / "configs/paper-eight-l1.json")
    p.add_argument("--model-index", required=True, type=Path)
    p.add_argument("--archives-dir", required=True, type=Path)
    p.add_argument("--output-root", required=True, type=Path)
    args = p.parse_args()
    torch.set_num_threads(1)
    result = prepare(args.config, args.model_index, args.archives_dir, args.output_root)
    print(json.dumps({k: v for k, v in result.items() if k not in ("partners", "provenance")}, indent=2))


if __name__ == "__main__":
    main()
