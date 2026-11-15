#!/usr/bin/env python3
"""Replay saved left-L1 joint actions, audit transitions and describe failures."""
from collections import Counter, deque
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from itertools import groupby
import argparse
import json
from pathlib import Path

from nestrl_research.artifacts import file_hash
from nestrl_research.config import load_config
from nestrl_research.environment import CookingEnv, INGREDIENT_NAMES, RECIPE_INGREDIENTS
from nestrl_research.peers import ScriptedPeer

ROOT = Path(__file__).resolve().parents[1]
LEFT = INGREDIENT_NAMES[:3]
RIGHT = INGREDIENT_NAMES[3:]


def pickup_distances(env):
    agent = env.world.agents[0]
    targets = {env.world.world_objects[n][0].location: n for n in LEFT}
    offsets = {1: (-1, 0), 2: (1, 0), 3: (0, 1), 4: (0, -1)}
    queue = deque([(agent.location, agent.orientation, 0)])
    seen, distances = {(agent.location, agent.orientation)}, {}
    while queue:
        location, direction, count = queue.popleft()
        dx, dy = offsets[direction]
        front = (location[0] + dx, location[1] + dy)
        if front in targets:
            distances.setdefault(targets[front], count + 1)
        for action, (dx, dy) in offsets.items():
            candidate = (location[0] + dx, location[1] + dy)
            dest = candidate if env.world.square_walkable(candidate) else location
            if (dest, action) not in seen:
                seen.add((dest, action))
                queue.append((dest, action, count + 1))
    return distances


def snapshot(env):
    """Compact exact scene state: holding codes 0 none,1..6 food,7 plate."""
    agents = env.world.agents
    plate = env.world.world_objects["Plate"][0]
    foods = [env.world.world_objects[name][0] for name in INGREDIENT_NAMES]
    def code(held):
        if held is None:
            return 0
        return 7 if held is plate else INGREDIENT_NAMES.index(type(held).__name__) + 1
    return {"a": [[*agent.location, agent.orientation, code(agent.holding)] for agent in agents],
            "f": [[*food.location, int(food.done()),
                   3 if food in plate.content else 1 if agents[0].holding is food
                   else 2 if agents[1].holding is food else 0] for food in foods],
            "p": list(plate.location), "pc": [INGREDIENT_NAMES.index(type(f).__name__) for f in plate.content],
            "lock": [INGREDIENT_NAMES.index(n) if n else -1 for n in env.chosen_ingredients],
            "success": bool(env.success)}


def own_state(frame):
    return (frame["a"][0], frame["f"][:3], sorted(i for i in frame["pc"] if i < 3))


def replay_episode(env, episode, trace, keep_frames=False):
    env.reset(seed=episode["seed"])
    recipe = episode["partner_id"].removeprefix("l0-1-")
    expected = RECIPE_INGREDIENTS[recipe][0]
    partner = ScriptedPeer(recipe, agent_id=1)
    distances = pickup_distances(env)
    nearest = [n for n, d in distances.items() if d == min(distances.values())]
    initial = snapshot(env)
    previous = initial
    frames = [initial] if keep_frames else []
    pick_order, chop_order, additions = [], [], []
    first_pick, first_chop, cue, wrong_plate = None, None, None, None
    l0_add = None
    unchanged = []
    noops, waiting_l0, new_holds, blocked = 0, 0, 0, 0
    total = 0.0
    for index, row in enumerate(trace):
        t = index + 1
        if row["step"] != index or row["actions"][1] != partner.act(env):
            raise ValueError("Non-contiguous trace or right L0 policy mismatch")
        before_plate = set(previous["pc"])
        expected_indices = {INGREDIENT_NAMES.index(n) for n in RECIPE_INGREDIENTS[recipe]}
        if INGREDIENT_NAMES.index(RECIPE_INGREDIENTS[recipe][1]) in before_plate and before_plate != expected_indices:
            waiting_l0 += row["actions"][1] == 0
        _, reward, done, truncated, info = env.step(row["actions"])
        if (reward != row["reward"] or info["events"] != row["events"]
                or info["success"] != row["success"]):
            raise ValueError(f"Saved transition mismatch at t={t}")
        current = snapshot(env)
        if own_state(previous) == own_state(current):
            unchanged.append(t)
        noops += row["actions"][0] == 0
        held = env.world.agents[0].holding
        if held is not None and type(held).__name__ in LEFT and not held.done() and current["a"][0][3] != previous["a"][0][3]:
            name = type(held).__name__
            first_pick = t if first_pick is None else first_pick
            if name not in pick_order:
                pick_order.append(name)
        for event in info["events"]:
            if event["type"] == "chop" and event["object"] in LEFT:
                first_chop = t if first_chop is None else first_chop
                chop_order.append(event["object"])
            blocked += event["type"] == "ingredient_limit_blocked" and event["agent"] == 0
            new_holds += event["type"] == "new_hold"
        added = set(current["pc"]) - before_plate
        additions.extend({"t": t, "food": INGREDIENT_NAMES[i]} for i in sorted(added) if i < 3)
        if any(i >= 3 for i in added) and l0_add is None:
            l0_add = t
        if set(current["pc"]) - expected_indices and wrong_plate is None:
            wrong_plate = t
        if cue is None:
            for i in range(3, 6):
                if current["f"][i][:3] != initial["f"][i][:3] and env.visible(0, current["f"][i][:2]):
                    cue = t
                    break
        total += reward
        if (done or truncated) != (index == len(trace) - 1):
            raise ValueError("Episode boundary mismatch")
        if keep_frames:
            frames.append({**current, "actions": row["actions"], "events": info["events"], "reward": reward})
        previous = current
    if (len(trace) != episode["steps"] or total != episode["return"]
            or env.success != episode["success"] or env.completed_recipe != episode["recipe"]):
        raise ValueError("Episode summary mismatch")
    cutoff = min(first_pick or (len(trace) + 1), (cue + 1) if cue else len(trace) + 1)
    longest = current_run = 0
    for t in range(1, cutoff):
        current_run = current_run + 1 if t in unchanged else 0
        longest = max(longest, current_run)
    features = {**episode, "expected_left_food": expected, "raw_pickup_order": pick_order,
                "initial_left_pose": initial["a"][0][:3], "pickup_distances": distances,
                "nearest_foods": nearest, "first_pickup_nearest": bool(pick_order and pick_order[0] in nearest),
                "chop_order": chop_order, "plate_additions": additions,
                "final_plate": [INGREDIENT_NAMES[i] for i in previous["pc"]],
                "first_pickup_t": first_pick, "first_chop_t": first_chop, "observable_cue_t": cue,
                "first_wrong_plate_t": wrong_plate, "l0_plate_add_t": l0_add,
                "l0_waiting_after_own_plate_steps": waiting_l0,
                "l1_noop_steps": noops, "l1_unchanged_steps": len(unchanged),
                "prepickup_precue_longest_unchanged": longest,
                "first_pickup_after_cue": first_pick is not None and cue is not None and first_pick > cue,
                "first_pickup_matches_partner": bool(pick_order and pick_order[0] == expected),
                "chopped_expected": expected in chop_order,
                "wrong_food_chopped": any(n != expected for n in chop_order),
                "ingredient_limit_blocked_steps": blocked,
                "interaction_reward": new_holds * env.config.interact_reward,
                "delivery_reward": env.config.complete_reward if env.success else 0.0}
    features["other_progress_reward"] = total - features["interaction_reward"] - features["delivery_reward"]
    return features, frames


def summarize(rows):
    n = len(rows)
    return {"episodes": n, "successes": sum(r["success"] for r in rows),
            "raw_pickup_order": dict(Counter("/".join(r["raw_pickup_order"]) for r in rows)),
            "chop_order": dict(Counter("/".join(r["chop_order"]) for r in rows)),
            "left_plate_addition_order": dict(Counter("/".join(a["food"] for a in r["plate_additions"]) for r in rows)),
            "final_plate_contents": dict(Counter("/".join(sorted(r["final_plate"])) for r in rows)),
            "counts": {k: sum(bool(r[k]) for r in rows) for k in
                       ("first_pickup_matches_partner", "first_pickup_nearest", "first_pickup_after_cue", "chopped_expected", "wrong_food_chopped")},
            "means": {k: sum(r[k] for r in rows) / n for k in
                      ("return", "steps", "l1_noop_steps", "l1_unchanged_steps", "prepickup_precue_longest_unchanged",
                       "l0_waiting_after_own_plate_steps", "ingredient_limit_blocked_steps", "interaction_reward",
                       "delivery_reward", "other_progress_reward")},
            "timings": {k: {"count": len(v), "mean": sum(v)/len(v) if v else None,
                             "min": min(v) if v else None, "max": max(v) if v else None}
                        for k in ("first_pickup_t", "first_chop_t", "observable_cue_t", "first_wrong_plate_t")
                        for v in [[r[k] for r in rows if r[k] is not None]]}}


def analyze_model(args):
    request, model, output = args
    condition, seed = model["condition"], request["model_seed"]
    base = ROOT / request["evaluation_root"]
    run = base / "evaluation" / f"e{condition}" / "runs" / model["run_id"]
    inputs = {n: file_hash(run / n) for n in ("episodes.json", "traces.jsonl", "protocol.json", "artifacts.json")}
    artifacts = json.loads((run / "artifacts.json").read_text())
    for n in inputs:
        if n != "artifacts.json" and artifacts["files"][n]["sha256"] != inputs[n]:
            raise ValueError("Input differs from evaluation artifact hashes")
    evaluation_plan = json.loads((base / "evaluation" / f"e{condition}" / "plan.json").read_text())
    config = load_config(ROOT / evaluation_plan["training_config"])
    checkpoint = base / "models" / f"e{condition}" / f"l1-construction-{seed}" / "checkpoints/final.pt"
    checkpoint_hash = file_hash(checkpoint)
    if checkpoint_hash != model["checkpoint_sha256"] or config.l1_ego_side != 0:
        raise ValueError("Model identity mismatch")
    meta = json.loads((checkpoint.with_suffix(".json")).read_text())
    if meta["config_sha256"] != config.sha256 or meta["ego_side"] != 0:
        raise ValueError("Checkpoint role/config mismatch")
    for name, digest in meta["implementation_sources"].items():
        if file_hash(ROOT / name) != digest:
            raise ValueError("Replay implementation differs from training")
    episodes = json.loads((run / "episodes.json").read_text())
    key = lambda r: (r["partner_id"], r["round"], r["episode_in_round"])
    by_key = {key(r): r for r in episodes}
    env = CookingEnv(asdict(config.environment))
    rows, clips = [], []
    with (run / "traces.jsonl").open() as stream:
        for identity, grouped in groupby((json.loads(line) for line in stream), key=key):
            ep = by_key.pop(identity)
            visual = ep["round"] == request["visual_round"] and ep["episode_in_round"] in request["visual_episodes"]
            feature, frames = replay_episode(env, ep, list(grouped), visual)
            rows.append(feature)
            if visual:
                clips.append({"condition": condition, "model_seed": seed, **feature, "frames": frames})
    if by_key or len(rows) != request["episodes_per_model"] or len(clips) != 15:
        raise ValueError("Incomplete replay or visualization selection")
    if checkpoint_hash != file_hash(checkpoint) or inputs != {n: file_hash(run/n) for n in inputs}:
        raise ValueError("Frozen input changed during replay")
    result = {"condition": condition, "seed": seed, "checkpoint_sha256": checkpoint_hash,
              "config_sha256": config.sha256, "input_sha256": inputs, "replay_verified": True,
              "all": summarize(rows), "by_partner": {p: summarize([r for r in rows if r["partner_id"] == p])
                                                     for p in sorted({r["partner_id"] for r in rows})}}
    out = Path(output)
    (out / f"episodes-e{condition}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (out / f"clips-e{condition}.json").write_text(json.dumps(clips, separators=(",", ":")) + "\n")
    (out / f"summary-e{condition}.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    request = json.loads(args.config.read_text())
    summary_path = ROOT / request["evaluation_root"] / "summary.json"
    if file_hash(summary_path) != request["evaluation_summary_sha256"]:
        raise ValueError("Evaluation summary hash mismatch")
    if request["weight_updates"] or request["policy_inference"] or (request["l1_side"], request["l0_side"]) != (0, 1):
        raise ValueError("This analysis only replays saved left-L1/right-L0 actions")
    models = [r for r in json.loads(summary_path.read_text())["models"] if r["seed"] == request["model_seed"]]
    if sorted(r["condition"] for r in models) != request["conditions"]:
        raise ValueError("Selected model set differs")
    args.output.mkdir(parents=True, exist_ok=False)
    with ProcessPoolExecutor(max_workers=request["workers"]) as pool:
        results = list(pool.map(analyze_model, [(request, m, str(args.output)) for m in models]))
    clips = [clip for condition in request["conditions"]
             for clip in json.loads((args.output / f"clips-e{condition}.json").read_text())]
    (args.output / "clips.json").write_text(json.dumps(clips, separators=(",", ":")) + "\n")
    result = {"analysis_id": request["analysis_id"], "models": results,
              "episodes": sum(r["all"]["episodes"] for r in results), "visualized_episodes": len(clips),
              "weight_updates": False, "policy_inference": False, "replay_verified": True}
    (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    (args.output / "manifest.json").write_text(json.dumps({p.name: file_hash(p) for p in sorted(args.output.iterdir())
                                                          if p.is_file()}, indent=2) + "\n")
    print(json.dumps({"episodes": result["episodes"], "clips": len(clips), "replay_verified": True}))


if __name__ == "__main__":
    main()
