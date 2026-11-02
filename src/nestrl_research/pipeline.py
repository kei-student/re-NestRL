"""Train nested levels independently; frozen construction partners are the only training pool."""

from __future__ import annotations

import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from .artifacts import atomic_json, file_hash, freeze_manifest, load_checkpoint, new_run, runtime_metadata
from .config import ExperimentConfig, RECIPES, SCHEMA
from .model import RecurrentActorCritic
from .ppo import compute_gae, ppo_update
from .provenance import source_metadata
from .rollout import RolloutCollector


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_model(config: ExperimentConfig, device):
    return RecurrentActorCritic(obs_dim=105, action_dim=6, hidden_dim=config.hidden_dim,
                               hidden_dims=tuple(config.hidden_dims)).to(device)


def l1_id(split: str, seed: int) -> str:
    return f"l1-{split}-{seed}"


def partition_seeds(config, split):
    if split not in ("construction", "validation", "test"):
        raise ValueError("Unknown registry partition")
    return getattr(config, f"{split}_seeds")


def registry_spec(config: ExperimentConfig) -> dict:
    return {"schema": SCHEMA, "config_sha256": config.sha256,
            "l0": [{"id": f"l0-{side}-{recipe}", "recipe": recipe, "side": side, "frozen": True} for side in (0, 1) for recipe in RECIPES],
            "l1": [{"id": l1_id(split, seed), "split": split, "seed": seed, "side": config.l1_ego_side}
                   for split in ("construction", "validation", "test") for seed in partition_seeds(config, split)]}


def ensure_registry(config, output_root):
    return freeze_manifest(output_root / "registry.json", registry_spec(config))


def load_policy(config, checkpoint, device):
    state, manifest = load_checkpoint(Path(checkpoint), device, config.sha256)
    current_source = source_metadata()
    if manifest.get("environment_source_sha256") != current_source["environment_source_sha256"]:
        raise ValueError("Checkpoint environment source differs from the current reconstruction")
    model = make_model(config, device)
    model.load_state_dict(state["model"], strict=True)
    model.eval()
    return model, manifest


def load_l1_pool(config, output_root, split, device):
    models, provenance = [], []
    for seed in partition_seeds(config, split):
        identity = l1_id(split, seed)
        checkpoint = output_root / identity / "checkpoints" / "final.pt"
        model, manifest = load_policy(config, checkpoint, device)
        expected = {"stage": "l1", "split": split, "seed": seed, "ego_side": config.l1_ego_side, "run_id": identity}
        if any(manifest.get(key) != value for key, value in expected.items()):
            raise ValueError(f"L1 artifact does not match frozen registry: {identity}")
        models.append(model)
        provenance.append({"id": identity, "split": split, "checkpoint": checkpoint.relative_to(output_root).as_posix(),
                           "checkpoint_sha256": file_hash(checkpoint), "manifest_sha256": file_hash(checkpoint.with_suffix(".json"))})
    return models, provenance


def train(config: ExperimentConfig, output_root: Path, stage: str, device, *, split="construction", seed=None, run_id=None):
    if stage not in ("l1", "l2", "generalist"):
        raise ValueError("Supported training stages: l1, l2, generalist")
    config.validate()
    source = source_metadata()
    output_root = Path(output_root)
    registry_hash = ensure_registry(config, output_root)
    if stage == "l1":
        if seed not in partition_seeds(config, split):
            raise ValueError("L1 seed is not in the requested frozen partition")
        expected_id = l1_id(split, seed)
        if run_id not in (None, expected_id):
            raise ValueError("L1 run ID is fixed by the immutable registry")
        run_id, ego_side = expected_id, config.l1_ego_side
    else:
        if split != "construction":
            raise ValueError("L2/Generalist training may use construction policies only")
        seed = config.learner_seed if seed is None else seed
        run_id, ego_side = run_id or f"{stage}-{seed}", config.ego_side
    seed_all(seed)
    model = make_model(config, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.ppo.lr, eps=1e-5)
    partner_models, provenance = (load_l1_pool(config, output_root, "construction", device) if stage == "l2" else (None, []))
    from .artifacts import save_checkpoint
    start_time = time.monotonic()
    with new_run(output_root, run_id) as directory:
        metadata = {**source, "run_id": run_id, "stage": stage, "split": split, "seed": seed, "ego_side": ego_side,
                    "config_sha256": config.sha256, "registry_sha256": registry_hash,
                    "training_partners": provenance or [{"id": f"l0-{1-ego_side}-{recipe}", "frozen": True} for recipe in RECIPES],
                    "runtime": runtime_metadata(str(device)), "resumable": False}
        atomic_json(directory / "config.json", config.to_dict())
        atomic_json(directory / "run.json", metadata)
        collector = RolloutCollector(config, model, ego_side, seed + 100_000, device, partner_models)
        with (directory / "metrics.jsonl").open("x", encoding="utf-8") as stream:
            for update in range(config.total_steps // config.batch_size):
                model.eval()
                batch, bootstrap, episodes = collector.collect()
                batch["advantages"], batch["returns"] = compute_gae(batch["rewards"], batch["old_values"], batch["dones"],
                                                                    bootstrap, config.ppo.gamma, config.ppo.gae_lambda)
                model.train()
                metrics = ppo_update(model, optimizer, batch, config.ppo)
                metrics.update({"update": update + 1, "steps": (update + 1) * config.batch_size, "episodes": len(episodes),
                                "success_rate": sum(e["success"] for e in episodes) / len(episodes) if episodes else None,
                                "elapsed_seconds": time.monotonic() - start_time})
                stream.write(json.dumps(metrics, allow_nan=False) + "\n")
                stream.flush()
                print(json.dumps({"run_id": run_id, **metrics}, allow_nan=False), flush=True)
        atomic_json(directory / "episodes.json", collector.episodes)
        checkpoint = save_checkpoint(directory, model, optimizer, {**metadata, "steps": config.total_steps})
        atomic_json(directory / "complete.json", {"checkpoint": "checkpoints/final.pt", "sha256": file_hash(checkpoint),
                                                 "steps": config.total_steps, "elapsed_seconds": time.monotonic() - start_time})
    return checkpoint
