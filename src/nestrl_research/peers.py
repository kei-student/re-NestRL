"""Frozen recipe peers with shortest-path navigation for the Figure 1 layout."""

from __future__ import annotations

from collections import deque
from types import SimpleNamespace
from typing import Any

import numpy as np

from .environment import RECIPE_INGREDIENTS, upstream_modules


class ScriptedPeer:
    """Prepare one side's ingredient, combine on the divider, then serve.

    This is a reconstruction of the paper's level-0 recipe experts, using a
    shortest-path controller instead of PACE's random event-priority policy.
    Random action probability is an explicit optional peer diversity parameter.
    """

    def __init__(self, recipe: str, agent_id: int = 1, seed: int = 0,
                 random_action_probability: float = 0.0):
        if recipe not in RECIPE_INGREDIENTS:
            raise ValueError(f"unsupported recipe: {recipe}")
        if not 0 <= random_action_probability <= 1:
            raise ValueError("random_action_probability must be in [0, 1]")
        self.recipe = recipe
        self.set_id(agent_id)
        self.seed = int(seed)
        self.random_action_probability = float(random_action_probability)
        self.objects = upstream_modules()["gym_cooking.cooking_world.world_objects"]
        self.reset()

    def set_id(self, agent_id: int) -> None:
        if agent_id not in (0, 1):
            raise ValueError("agent_id must be 0 or 1")
        self.agent_id = int(agent_id)

    def reset(self, seed: int | None = None) -> None:
        if seed is not None:
            self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)

    @staticmethod
    def _navigation_action(world, agent_id: int, target: tuple[int, int]) -> int:
        """BFS over (floor position, orientation), including blocked turn moves."""
        agent = world.agents[agent_id]
        start = (agent.location, agent.orientation)
        queue = deque([(start, None)])
        visited = {start}
        occupied = {other.location for index, other in enumerate(world.agents) if index != agent_id}
        while queue:
            (location, orientation), first_action = queue.popleft()
            pose = SimpleNamespace(location=location)
            if world.get_target_location(pose, orientation) == target:
                return 5 if first_action is None else first_action
            for action in (1, 2, 3, 4):
                candidate = world.get_target_location(pose, action)
                if not (0 <= candidate[0] < world.width and 0 <= candidate[1] < world.height):
                    continue
                next_location = candidate if world.square_walkable(candidate) and candidate not in occupied else location
                state = (next_location, action)
                if state not in visited:
                    visited.add(state)
                    queue.append((state, action if first_action is None else first_action))
        return 0

    def act(self, env_or_world: Any) -> int:
        world = getattr(env_or_world, "world", env_or_world)
        if self.random_action_probability and self.rng.random() < self.random_action_probability:
            return int(self.rng.integers(0, 6))
        o = self.objects
        agent = world.agents[self.agent_id]
        ingredient_name = RECIPE_INGREDIENTS[self.recipe][self.agent_id]
        food = world.world_objects[ingredient_name][0]
        plate = world.world_objects["Plate"][0]
        cutting_board = world.world_objects["CutBoard"][self.agent_id]
        recipe_names = set(RECIPE_INGREDIENTS[self.recipe])
        plate_names = {type(content).__name__ for content in plate.content}
        if isinstance(agent.holding, o.Plate):
            target = world.world_objects["DeliverSquare"][0].location if plate_names == recipe_names else (3, 3)
        elif agent.holding is not None:
            target = plate.location if agent.holding.done() else cutting_board.location
        elif food not in plate.content:
            # Interacting chops once; the following interaction picks it up.
            if food.location == cutting_board.location and not food.done():
                target = cutting_board.location
            else:
                target = food.location
        elif self.agent_id == 1 and plate_names == recipe_names:
            target = plate.location
        else:
            return 0
        return self._navigation_action(world, self.agent_id, target)

    def __call__(self, env_or_world: Any) -> int:
        return self.act(env_or_world)


RecipePeer = ScriptedPeer
