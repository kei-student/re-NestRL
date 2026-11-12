"""Descriptive analysis of saved cooking actions, without policy inference.

Times are zero-based joint-action indices. A cue produced by action r can
first affect the next decision, r + 1. Pickup is a measurable commitment proxy;
it does not identify the earlier, latent choice of a navigation goal.
"""

from __future__ import annotations

from collections import Counter, deque
from typing import Any

from .environment import CookingEnv, RECIPE_INGREDIENTS

RIGHT = ("Onion", "Carrot", "Broccoli")
LEFT = ("Lettuce", "Tomato", "Potato")


def pickup_distances(env: CookingEnv, agent_id: int = 1) -> dict[str, int]:
    """Shortest number of actions including the final pickup (ties retained)."""
    world = env.world
    agent = world.agents[agent_id]
    queue = deque([(agent.location, agent.orientation, 0)])
    seen = {(agent.location, agent.orientation)}
    targets = {world.world_objects[name][0].location: name for name in RIGHT}
    distances: dict[str, int] = {}
    occupied = {a.location for i, a in enumerate(world.agents) if i != agent_id}
    offsets = {1: (-1, 0), 2: (1, 0), 3: (0, 1), 4: (0, -1)}
    while queue:
        location, orientation, distance = queue.popleft()
        dx, dy = offsets[orientation]
        front = (location[0] + dx, location[1] + dy)
        if front in targets:
            distances.setdefault(targets[front], distance + 1)
        for action, (dx, dy) in offsets.items():
            candidate = (location[0] + dx, location[1] + dy)
            nxt = candidate if world.square_walkable(candidate) and candidate not in occupied else location
            state = (nxt, action)
            if state not in seen:
                seen.add(state)
                queue.append((*state, distance + 1))
    return distances


def ego_state(env: CookingEnv) -> tuple:
    """State changed by useful interactions, even when the chef does not move."""
    agent = env.world.agents[1]
    plate = env.world.world_objects["Plate"][0]
    return (agent.location, agent.orientation, id(agent.holding),
            tuple((env.world.world_objects[name][0].location,
                   bool(env.world.world_objects[name][0].done()),
                   env.world.world_objects[name][0] in plate.content) for name in RIGHT),
            plate.location, tuple(type(food).__name__ for food in plate.content))


def analyze_episode(env: CookingEnv, episode: dict, trace: list[dict]) -> dict:
    env.reset(seed=episode["seed"])
    initial_position = list(env.world.agents[1].location)
    distances = pickup_distances(env)
    nearest = sorted(name for name, d in distances.items() if d == min(distances.values()))
    first_pick = None
    first_left_pick = None
    visible_cue = None
    first_chop = None
    pickup_order, chop_order, chop_times = [], [], {}
    first_add_right = None
    right_additions = []
    noop_steps, unchanged_steps = [], []
    total = 0.0
    for t, row in enumerate(trace):
        if row["step"] != t:
            raise ValueError("non-contiguous action trace")
        agents = env.world.agents
        old_holds = [a.holding for a in agents]
        before = ego_state(env)
        plate = env.world.world_objects["Plate"][0]
        old_contents = {type(f).__name__ for f in plate.content}
        _, reward, terminated, truncated, info = env.step(row["actions"])
        if (reward != row["reward"] or info["events"] != row["events"]
                or info["success"] != row["success"]):
            raise ValueError(f"saved trace replay mismatch at action {t}")
        if row["actions"][1] == 0:
            noop_steps.append(t)
        if before == ego_state(env):
            unchanged_steps.append(t)
        for i, names in ((0, LEFT), (1, RIGHT)):
            held = agents[i].holding
            name = type(held).__name__
            if held is not old_holds[i] and name in names:
                if i == 0 and first_left_pick is None:
                    first_left_pick = t
                if i == 1 and not held.done():
                    if first_pick is None:
                        first_pick = t
                    if name not in pickup_order:
                        pickup_order.append(name)
        for event in info["events"]:
            if event["type"] == "chop" and event["object"] in RIGHT:
                name = event["object"]
                chop_order.append(name)
                chop_times[name] = t
                if first_chop is None:
                    first_chop = t
        additions = ({type(f).__name__ for f in plate.content} - old_contents) & set(RIGHT)
        if additions:
            first_add_right = t if first_add_right is None else first_add_right
            right_additions.extend(sorted(additions))
        if visible_cue is None and any(env.visible(1, env.world.world_objects[name][0].location)
                                       for name in LEFT):
            visible_cue = t
        total += reward
        if (terminated or truncated) != (t == len(trace) - 1):
            raise ValueError("episode boundary mismatch")
    if (len(trace) != episode["steps"] or total != episode["return"]
            or env.success != episode["success"] or env.completed_recipe != episode["recipe"]):
        raise ValueError("episode summary mismatch")
    expected = RECIPE_INGREDIENTS[episode["partner_id"].removeprefix("l0-0-")][1]
    # The decision that creates the cue cannot condition on that same cue.
    before_cue = visible_cue + 1 if visible_cue is not None else len(trace)
    uncommitted_end = min(first_pick if first_pick is not None else len(trace), before_cue)
    pre_unchanged = [t for t in unchanged_steps if t < uncommitted_end]
    longest, current, previous = 0, 0, -2
    for t in pre_unchanged:
        current = current + 1 if t == previous + 1 else 1
        longest, previous = max(longest, current), t
    after_visible = first_pick is not None and visible_cue is not None and first_pick > visible_cue
    excess = first_pick + 1 - min(distances.values()) if first_pick is not None else None
    return {
        **episode, "initial_position": initial_position, "nearest_ingredients": nearest,
        "pickup_distances": distances, "pickup_order": pickup_order, "chop_order": chop_order,
        "first_pickup_step": first_pick, "first_chop_step": first_chop,
        "l0_first_pickup_step": first_left_pick, "visible_cue_created_step": visible_cue,
        "first_right_plate_add_step": first_add_right, "right_plate_additions": right_additions,
        "first_pickup_nearest": bool(pickup_order and pickup_order[0] in nearest),
        "first_pickup_target": bool(pickup_order and pickup_order[0] == expected),
        "chopped_count": len(set(chop_order)), "all_three_chopped": len(set(chop_order)) == 3,
        "multiple_chopped": len(set(chop_order)) >= 2,
        "wrong_ingredients_chopped": len(set(chop_order) - {expected}),
        "chopped_before_visible_cue": sum(t <= (visible_cue if visible_cue is not None else len(trace))
                                         for t in chop_times.values()),
        "all_three_before_plate_add": first_add_right is not None and len(chop_times) == 3
            and max(chop_times.values()) < first_add_right,
        "pickup_after_l0_pickup": first_pick is not None and first_left_pick is not None
            and first_pick > first_left_pick,
        "pickup_after_visible_cue": after_visible,
        "pickup_excess_over_nearest_actions": excess,
        "precommit_precue_noop_steps": sum(t < uncommitted_end for t in noop_steps),
        "precommit_precue_unchanged_steps": len(pre_unchanged),
        "precommit_precue_longest_unchanged": longest,
        "waiting_candidate": bool(after_visible and longest >= 3 and excess >= 3),
        "noop_steps": len(noop_steps), "unchanged_steps": len(unchanged_steps),
        "right_plate_matches_partner": right_additions == [expected],
    }


def summarize(rows: list[dict]) -> dict[str, Any]:
    n = len(rows)
    if not n:
        raise ValueError("empty behavioral sample")
    flags = ("success", "first_pickup_nearest", "first_pickup_target", "all_three_chopped",
             "multiple_chopped", "all_three_before_plate_add", "pickup_after_l0_pickup",
             "pickup_after_visible_cue", "waiting_candidate", "right_plate_matches_partner")
    numbers = ("steps", "return", "chopped_count", "wrong_ingredients_chopped",
               "chopped_before_visible_cue", "noop_steps", "unchanged_steps",
               "precommit_precue_noop_steps", "precommit_precue_unchanged_steps")
    result: dict[str, Any] = {"episodes": n,
        "counts": {key: sum(bool(row[key]) for row in rows) for key in flags},
        "rates": {key: sum(bool(row[key]) for row in rows) / n for key in flags},
        "means": {key: sum(row[key] for row in rows) / n for key in numbers},
        "first_pickup_counts": dict(Counter(row["pickup_order"][0] if row["pickup_order"] else "none"
                                            for row in rows)),
        "chop_order_counts": dict(Counter("/".join(row["chop_order"]) for row in rows)),
    }
    for key in ("first_pickup_step", "l0_first_pickup_step", "visible_cue_created_step",
                "first_chop_step", "pickup_excess_over_nearest_actions"):
        values = [row[key] for row in rows if row[key] is not None]
        result["means"][key] = sum(values) / len(values) if values else None
        result.setdefault("timing_denominators", {})[key] = len(values)
    return result
