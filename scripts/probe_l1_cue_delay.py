#!/usr/bin/env python3
"""Paired, evaluation-only cue-delay probe of an explicitly selected policy."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import torch

from nestrl_research.artifacts import file_hash, runtime_metadata
from nestrl_research.behavior import analyze_episode
from nestrl_research.config import load_config
from nestrl_research.environment import CookingEnv
from nestrl_research.evaluation import PolicyContext
from nestrl_research.peers import ScriptedPeer
from nestrl_research.pipeline import load_policy, seed_all

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    request = json.loads(args.config.read_text())
    config = load_config(ROOT / request["training_config"])
    if config.sha256 != request["training_config_sha256"]:
        raise ValueError("training configuration mismatch")
    checkpoint = ROOT / request["checkpoint"]
    if file_hash(checkpoint) != request["checkpoint_sha256"]:
        raise ValueError("checkpoint mismatch")
    torch.set_num_threads(1)
    model, manifest = load_policy(config, checkpoint, torch.device("cpu"))
    if manifest["seed"] != request["model_seed"]:
        raise ValueError("wrong policy")
    reference = ROOT / request["evaluation_run"]
    keys = ("partner_id", "round", "episode_in_round")
    selected = {tuple(case[k] for k in keys): case for case in request["cases"]}
    originals = {key: [] for key in selected}
    reference_trace_hash = file_hash(reference / "traces.jsonl")
    with (reference / "traces.jsonl").open() as stream:
        for line in stream:
            row = json.loads(line)
            key = tuple(row[k] for k in keys)
            if key in originals:
                originals[key].append(row)
    args.output.mkdir(parents=True, exist_ok=False)
    results = []
    for key, case in selected.items():
        for delay in request["initial_l0_noop_steps"]:
            seed_all(case["seed"])
            context = PolicyContext(model, torch.device("cpu"), False)
            peer = ScriptedPeer(case["partner_id"].removeprefix("l0-0-"), agent_id=0)
            env = CookingEnv(asdict(config.environment))
            obs = env.reset(seed=case["seed"])
            trace, total = [], 0.0
            for t in range(env.horizon):
                actions = [0 if t < delay else peer.act(env), context.act(obs[1])]
                obs, reward, done, truncated, info = env.step(actions)
                row = {"step": t, "actions": actions, "reward": reward,
                       "events": info["events"], "success": info["success"]}
                if delay == 0:
                    old = originals[key][t]
                    if any(row[k] != old[k] for k in row):
                        raise ValueError("baseline inference differs from original evaluation")
                trace.append(row)
                total += reward
                if done or truncated:
                    break
            meta = {**case, "steps": len(trace), "return": total, "success": env.success,
                    "recipe": env.completed_recipe}
            behavior = analyze_episode(CookingEnv(asdict(config.environment)), meta, trace)
            results.append({"initial_l0_noop_steps": delay, "behavior": behavior,
                            "trace": trace, "baseline_matches_original": True if delay == 0 else None})
    if file_hash(checkpoint) != request["checkpoint_sha256"]:
        raise ValueError("checkpoint changed")
    result = {"analysis_id": request["analysis_id"], "model_seed": request["model_seed"],
              "checkpoint_sha256": request["checkpoint_sha256"], "checkpoint_unchanged": True,
              "reference_trace_sha256": reference_trace_hash, "runtime": runtime_metadata("cpu"),
              "weight_updates": False, "results": results}
    (args.output / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    for row in results:
        b = row["behavior"]
        print(json.dumps({"partner": b["partner_id"], "delay": row["initial_l0_noop_steps"],
                          "cue_created": b["visible_cue_created_step"],
                          "pickup_step": b["first_pickup_step"], "success": b["success"]}), flush=True)


if __name__ == "__main__":
    main()
