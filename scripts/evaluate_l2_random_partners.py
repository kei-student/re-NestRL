#!/usr/bin/env python3
"""Frozen L2 evaluation with a uniformly sampled L1 at every five-episode round."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from nestrl_research.artifacts import atomic_json, file_hash, freeze_manifest, new_run, runtime_metadata
from nestrl_research.config import L1PoolConfig, ROOT, canonical_hash, load_config
from nestrl_research.environment import CookingEnv
from nestrl_research.evaluation import PolicyContext, summarize
from nestrl_research.partner_pool import load_catalog_pool
from nestrl_research.pipeline import ensure_registry, load_policy, seed_all
from nestrl_research.provenance import source_metadata


def sample_partner_schedule(partner_ids, rounds, selection_seed):
    ids = tuple(partner_ids)
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("Partner identities must be nonempty and unique")
    if type(rounds) is not int or rounds < 1:
        raise ValueError("rounds must be positive")
    if type(selection_seed) is not int or not 0 <= selection_seed < 2**32:
        raise ValueError("selection_seed must be an unsigned 32-bit integer")
    rng = np.random.Generator(np.random.PCG64(selection_seed))
    return [ids[int(i)] for i in rng.integers(len(ids), size=rounds)]


def tensor_digest(model):
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


@torch.no_grad()
def evaluate_random_partners(config, output_root, checkpoint, *, partner_pool, rounds,
                             evaluation_seed, selection_seed, run_id,
                             deterministic=False, driver_source_commit=None):
    config.validate()
    if (config.l1_pool is None or not isinstance(partner_pool, L1PoolConfig)
            or partner_pool.catalog != config.l1_pool.catalog
            or partner_pool.catalog_sha256 != config.l1_pool.catalog_sha256):
        raise ValueError("Evaluation must use the original authenticated L1 construction catalog")
    if type(evaluation_seed) is not int or not 0 <= evaluation_seed < 2**32:
        raise ValueError("evaluation_seed must be an unsigned 32-bit integer")
    schedule = sample_partner_schedule(partner_pool.partner_ids, rounds, selection_seed)
    if evaluation_seed + 1000 * (rounds - 1) + config.episodes_per_round - 1 >= 2**32:
        raise ValueError("Evaluation episode seeds exceed the unsigned 32-bit range")
    output_root, checkpoint = Path(output_root), Path(checkpoint)
    ensure_registry(config, output_root)
    checkpoint_sha = file_hash(checkpoint)
    ego, manifest = load_policy(config, checkpoint, "cpu")
    if manifest["ego_side"] == config.l1_ego_side:
        raise ValueError("Learner and partner must occupy opposite roles")
    ego.requires_grad_(False)
    models, provenance = load_catalog_pool(config, output_root, "cpu", pool=partner_pool)
    trained = {p["id"] for p in manifest["training_partners"]}
    provenance = [{**p, "seen_in_l2_training": p["id"] in trained} for p in provenance]
    peers = {p["id"]: m for p, m in zip(provenance, models)}
    model_hashes = [tensor_digest(m) for m in [ego, *models]]
    input_files = {checkpoint: checkpoint_sha}
    for p in provenance:
        input_files[output_root / p["checkpoint"]] = p["checkpoint_sha256"]
    protocol = {
        **source_metadata(), "config_sha256": config.sha256,
        "learner_checkpoint_sha256": checkpoint_sha, "partner_level": "l1",
        "partition": "construction_diagnostic", "partners": provenance,
        "partner_pool_sha256": canonical_hash(asdict(partner_pool)),
        "partner_sampling": "uniform with replacement at each round",
        "selection_rng": "numpy.PCG64", "selection_seed": selection_seed,
        "total_rounds": rounds, "episodes_per_round": config.episodes_per_round,
        "seed": evaluation_seed, "episode_seed_rule": "seed + 1000 * global_round + episode_position",
        "deterministic_actions": deterministic, "weight_updates": False,
        "driver_sha256": file_hash(Path(__file__)),
        "driver_source_commit": driver_source_commit,
    }
    with new_run(output_root, run_id) as directory:
        schedule_value = {"partner_ids": list(partner_pool.partner_ids), "selection_seed": selection_seed,
                          "round_partner_ids": schedule}
        freeze_manifest(directory / "round_schedule.json", schedule_value)
        protocol["round_schedule_file_sha256"] = file_hash(directory / "round_schedule.json")
        freeze_manifest(directory / "protocol.json", protocol)
        atomic_json(directory / "runtime.json", runtime_metadata("cpu"))
        episodes = []
        with (directory / "traces.jsonl").open("x", encoding="utf-8") as trace:
            for round_index, identity in enumerate(schedule):
                seed_all(evaluation_seed + 1000 * round_index)
                ego_context = PolicyContext(ego, "cpu", deterministic)
                partner = PolicyContext(peers[identity], "cpu", deterministic)
                for position in range(config.episodes_per_round):
                    env = CookingEnv(asdict(config.environment))
                    episode_seed = evaluation_seed + 1000 * round_index + position
                    observation = env.reset(seed=episode_seed)
                    total_return = 0.0
                    for step in range(config.environment.horizon):
                        joint = [0, 0]
                        joint[manifest["ego_side"]] = ego_context.act(observation[manifest["ego_side"]])
                        joint[1 - manifest["ego_side"]] = partner.act(observation[1 - manifest["ego_side"]])
                        observation, reward, terminated, truncated, info = env.step(joint)
                        total_return += reward
                        trace.write(json.dumps({
                            "partner_id": identity, "round": round_index, "episode_in_round": position,
                            "step": step, "actions": joint, "reward": reward,
                            "events": info["events"], "success": info["success"],
                        }) + "\n")
                        if terminated or truncated:
                            break
                    episodes.append({
                        "partner_id": identity, "round": round_index, "episode_in_round": position,
                        "seed": episode_seed, "steps": step + 1, "success": bool(info["success"]),
                        "recipe": info["recipe"], "return": total_return,
                    })
                    ego_context.episode_boundary()
                    partner.episode_boundary()
                if (round_index + 1) % 100 == 0 or round_index + 1 == rounds:
                    atomic_json(directory / "progress.json", {
                        "completed_rounds": round_index + 1, "target_rounds": rounds,
                        "episodes": len(episodes), "weight_updates": False,
                    })
        if checkpoint_sha != file_hash(checkpoint):
            raise ValueError("Learner checkpoint changed during evaluation")
        if model_hashes != [tensor_digest(m) for m in [ego, *models]]:
            raise ValueError("An in-memory policy changed during evaluation")
        for path, expected in input_files.items():
            if file_hash(path) != expected:
                raise ValueError(f"Frozen input changed: {path}")
        summary = {
            "profile": config.profile, "protocol": protocol,
            "primary_measure": "pooled_episode_success_rate under uniform random round partners",
            "interpretation": "Known construction partners, not held-out generalization.",
            "scheduled_rounds_per_partner": {p: schedule.count(p) for p in partner_pool.partner_ids},
            **summarize(episodes, config.episodes_per_round),
        }
        atomic_json(directory / "episodes.json", episodes)
        atomic_json(directory / "summary.json", summary)
        atomic_json(directory / "artifacts.json", {
            "schema": config.schema, "weight_updates": False,
            "files": {n: {"sha256": file_hash(directory / n)} for n in
                      ("protocol.json", "runtime.json", "round_schedule.json",
                       "traces.jsonl", "episodes.json", "summary.json", "progress.json")},
        })
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--partner-pool", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--rounds", type=int, default=3000)
    parser.add_argument("--evaluation-seed", type=int, default=6001)
    parser.add_argument("--selection-seed", type=int, required=True)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--driver-source-commit")
    args = parser.parse_args()
    if args.threads != 1:
        parser.error("This evaluation protocol requires one torch/BLAS thread")
    torch.set_num_threads(1)
    pool = L1PoolConfig(**json.loads(args.partner_pool.read_text()))
    result = evaluate_random_partners(
        load_config(args.config), args.output_root, args.checkpoint, partner_pool=pool,
        rounds=args.rounds, evaluation_seed=args.evaluation_seed,
        selection_seed=args.selection_seed, run_id=args.run_id,
        driver_source_commit=args.driver_source_commit,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

