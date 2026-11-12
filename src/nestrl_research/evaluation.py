"""Held-out evaluation and auditable episode traces; never update weights."""

from __future__ import annotations

from dataclasses import asdict
import math
from pathlib import Path

import numpy as np
import torch

from .artifacts import atomic_json, file_hash, freeze_manifest, new_run, runtime_metadata
from .config import RECIPES
from .environment import CookingEnv
from .peers import ScriptedPeer
from .provenance import source_metadata
from .pipeline import ensure_registry, load_l1_pool, load_policy, seed_all


class PolicyContext:
    def __init__(self, model, device, deterministic):
        self.model, self.device, self.deterministic = model, device, deterministic
        self.reset()

    def reset(self):
        self.hidden = self.model.initial_state(1, self.device)
        self.previous = torch.tensor([-1], device=self.device)
        self.starts = torch.tensor([True], device=self.device)
        self.resets = self.starts.clone()

    @torch.no_grad()
    def act(self, observation):
        action, _, _, self.hidden = self.model.act(torch.as_tensor(observation, device=self.device).unsqueeze(0),
                                                  self.previous, self.starts, self.hidden, self.resets,
                                                  deterministic=self.deterministic)
        self.previous = action
        self.starts.fill_(False)
        self.resets.fill_(False)
        return int(action[0])

    def episode_boundary(self):
        self.previous.fill_(-1)
        self.starts.fill_(True)


def _wilson_interval(successes: int, episodes: int) -> list[float]:
    """Two-sided 95% score interval, without a normal approximation at 0/1."""
    z = 1.959963984540054
    rate = successes / episodes
    denominator = 1 + z * z / episodes
    center = (rate + z * z / (2 * episodes)) / denominator
    radius = z * math.sqrt(rate * (1 - rate) / episodes + z * z / (4 * episodes * episodes)) / denominator
    return [max(0.0, center - radius), min(1.0, center + radius)]


def summarize(episodes: list[dict], episodes_per_round: int) -> dict:
    """Summarize complete rounds, preserving macro and pooled denominators.

    Missing/duplicate episode positions are invalid evaluation data: an
    incomplete round cannot be reported as an all-episodes-successful round.
    Different partners may have different numbers of complete rounds.
    """
    if not isinstance(episodes_per_round, int) or isinstance(episodes_per_round, bool) or episodes_per_round <= 0:
        raise ValueError("episodes_per_round must be a positive integer")
    if not episodes:
        raise ValueError("Evaluation episodes must be nonempty")
    positions_by_round = {}
    for episode in episodes:
        identity = episode["partner_id"]
        round_index = episode["round"]
        position = episode["episode_in_round"]
        if not isinstance(identity, str) or not identity:
            raise ValueError("Episode partner IDs must be nonempty strings")
        if not isinstance(round_index, int) or isinstance(round_index, bool) or round_index < 0:
            raise ValueError("Round indices must be nonnegative integers")
        if (not isinstance(position, int) or isinstance(position, bool)
                or not 0 <= position < episodes_per_round):
            raise ValueError("Episode positions are outside the round")
        if not isinstance(episode["success"], (bool, np.bool_)):
            raise ValueError("Episode success must be boolean")
        if episode["success"] and episode.get("recipe") not in RECIPES:
            raise ValueError("Successful episodes must identify a permitted completed recipe")
        if not math.isfinite(episode["return"]):
            raise ValueError("Episode returns must be finite")
        round_positions = positions_by_round.setdefault((identity, round_index), set())
        if position in round_positions:
            raise ValueError("Duplicate episode position within a partner round")
        round_positions.add(position)
    expected_positions = set(range(episodes_per_round))
    if any(positions != expected_positions for positions in positions_by_round.values()):
        raise ValueError("Evaluation contains an incomplete round")
    partner_ids = sorted({episode["partner_id"] for episode in episodes})
    counts = {}
    by_partner = {}
    for identity in partner_ids:
        partner_episodes = [episode for episode in episodes if episode["partner_id"] == identity]
        successes = sum(bool(episode["success"]) for episode in partner_episodes)
        counts[identity] = {"successes": successes, "episodes": len(partner_episodes),
                            "wilson95": _wilson_interval(successes, len(partner_episodes))}
        by_partner[identity] = successes / len(partner_episodes)
    rates = list(by_partner.values())
    by_position = [float(np.mean([e["success"] for e in episodes if e["episode_in_round"] == position]))
                   for position in range(episodes_per_round)]
    round_ids = {(e["partner_id"], e["round"]) for e in episodes}
    rounds = [all(e["success"] for e in episodes if (e["partner_id"], e["round"]) == identity) for identity in round_ids]
    mean = float(np.mean(rates))
    successes = sum(bool(episode["success"]) for episode in episodes)
    recipe_counts = {recipe: sum(bool(episode["success"]) and episode.get("recipe") == recipe
                                  for episode in episodes) for recipe in RECIPES}
    recipe_entropy = (-sum((count / successes) * math.log(count / successes)
                           for count in recipe_counts.values() if count) if successes else None)
    return {"episodes": len(episodes), "successes": successes, "rounds": len(round_ids),
            "success_rate": mean, "pooled_episode_success_rate": successes / len(episodes),
            "success_rate_aggregation": "unweighted mean of per-partner episode success rates",
            "per_partner": by_partner, "per_partner_counts": counts,
            "per_episode_position": by_position, "all_episodes_round_success_rate": float(np.mean(rounds)),
            "partner_rate_std_population": float(np.std(rates)),
            "partner_rate_cv": float(np.std(rates) / mean) if mean else None,
            "mean_return": float(np.mean([e["return"] for e in episodes])),
            "recipe_counts": recipe_counts,
            "successful_recipe_entropy_nats": recipe_entropy,
            "successful_recipe_entropy_normalized": recipe_entropy / math.log(len(RECIPES)) if successes else None,
            "recipe_entropy_conditioning": "completed recipes in successful episodes only"}


def passes_each_partner(summary, partner_ids, episodes_per_partner, minimum_successes):
    """Apply an inclusive count threshold to every predeclared partner."""
    if not 0 <= minimum_successes <= episodes_per_partner or episodes_per_partner < 1:
        raise ValueError("Invalid success threshold or denominator")
    counts = summary["per_partner_counts"]
    if set(counts) != set(partner_ids) or summary["episodes"] != len(partner_ids) * episodes_per_partner:
        raise ValueError("Evaluation partner set or total denominator differs")
    for count in counts.values():
        if count["episodes"] != episodes_per_partner or not 0 <= count["successes"] <= episodes_per_partner:
            raise ValueError("Evaluation per-partner denominator/count differs")
    return all(counts[identity]["successes"] >= minimum_successes for identity in partner_ids)


@torch.no_grad()
def evaluate(config, output_root: Path, checkpoint: Path, device, *, partner_level="l1", split="test", run_id=None,
             deterministic=False, rounds=None, evaluation_seed=None):
    if split not in ("validation", "test") or partner_level not in ("l0", "l1"):
        raise ValueError("Evaluation uses validation/test L1 or the three fixed L0 conventions")
    config.validate()
    rounds = config.evaluation_rounds if rounds is None else rounds
    evaluation_seed = config.evaluation_seed if evaluation_seed is None else evaluation_seed
    if not isinstance(rounds, int) or isinstance(rounds, bool) or rounds < 1:
        raise ValueError("Evaluation rounds must be a positive integer")
    if (not isinstance(evaluation_seed, int) or isinstance(evaluation_seed, bool)
            or not 0 <= evaluation_seed < 2**32):
        raise ValueError("Evaluation seed must be an integer in [0, 2**32)")
    output_root = Path(output_root)
    ensure_registry(config, output_root)
    ego, manifest = load_policy(config, checkpoint, device)
    ego_side = manifest["ego_side"]
    if partner_level == "l1":
        if ego_side == config.l1_ego_side:
            raise ValueError("L1 evaluation partners occupy the same role as this learner")
        models, provenance = load_l1_pool(config, output_root, split, device)
        peers = [(item["id"], model) for item, model in zip(provenance, models)]
    else:
        peers = [(f"l0-{1-ego_side}-{recipe}", recipe) for recipe in RECIPES]
        provenance = [{"id": identity, "frozen": True} for identity, _ in peers]
    protocol = {**source_metadata(), "config_sha256": config.sha256, "learner_checkpoint_sha256": file_hash(checkpoint),
                "partner_level": partner_level, "partition": split if partner_level == "l1" else "fixed_conventions",
                "partners": provenance, "rounds": rounds, "episodes_per_round": config.episodes_per_round,
                "seed": evaluation_seed, "deterministic_actions": deterministic, "weight_updates": False}
    run_id = run_id or f"eval-{manifest['run_id']}-{partner_level}-{split}"
    with new_run(output_root, run_id) as directory:
        freeze_manifest(directory / "protocol.json", protocol)
        atomic_json(directory / "runtime.json", runtime_metadata(str(device)))
        episodes = []
        trace_file = directory / "traces.jsonl"
        with trace_file.open("x", encoding="utf-8") as trace:
            for partner_index, (identity, peer) in enumerate(peers):
                for round_index in range(rounds):
                    seed_all(evaluation_seed + 100_000 * partner_index + 1000 * round_index)
                    ego_context = PolicyContext(ego, device, deterministic)
                    partner = (PolicyContext(peer, device, deterministic) if partner_level == "l1"
                               else ScriptedPeer(peer, agent_id=1-ego_side, seed=evaluation_seed + round_index))
                    for episode_index in range(config.episodes_per_round):
                        env = CookingEnv(asdict(config.environment))
                        episode_seed = evaluation_seed + 100_000 * partner_index + 1000 * round_index + episode_index
                        observation = env.reset(seed=episode_seed)
                        total_return = 0.0
                        for step in range(config.environment.horizon):
                            joint = [0, 0]
                            joint[ego_side] = ego_context.act(observation[ego_side])
                            joint[1-ego_side] = (partner.act(observation[1-ego_side]) if partner_level == "l1" else partner.act(env))
                            observation, reward, terminated, truncated, info = env.step(joint)
                            total_return += reward
                            # Full joint actions and environment events permit later behavioral analysis.
                            trace.write(__import__("json").dumps({"partner_id": identity, "round": round_index,
                                        "episode_in_round": episode_index, "step": step, "actions": joint,
                                        "reward": reward, "events": info["events"], "success": info["success"]}) + "\n")
                            if terminated or truncated:
                                break
                        episodes.append({"partner_id": identity, "round": round_index, "episode_in_round": episode_index,
                                         "seed": episode_seed, "steps": step + 1, "success": bool(info["success"]),
                                         "recipe": info["recipe"], "return": total_return})
                        ego_context.episode_boundary()
                        if partner_level == "l1":
                            partner.episode_boundary()
                        else:
                            partner.reset()
        interpretation = (
            "Smoke validates execution only; performance reproduction requires the full preregistered training."
            if config.profile == "smoke" else
            "Results use the registered reconstruction settings; author-observation equivalence and full paper replication remain unverified."
        )
        summary = {"profile": config.profile, "protocol": protocol, "interpretation": interpretation,
                   **summarize(episodes, config.episodes_per_round)}
        atomic_json(directory / "episodes.json", episodes)
        atomic_json(directory / "summary.json", summary)
        atomic_json(directory / "artifacts.json", {
            "schema": config.schema,
            "files": {name: {"sha256": file_hash(directory / name)} for name in
                      ("protocol.json", "runtime.json", "traces.jsonl", "episodes.json", "summary.json")},
            "weight_updates": False,
        })
    return summary
