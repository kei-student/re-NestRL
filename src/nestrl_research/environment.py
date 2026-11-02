"""Figure 1 cooking task backed by the unmodified, pinned PACE world.

The paper does not publish its observation indices.  We retain the PACE dense
partial construction (93 values on this layout), followed by 12 reserved zeros.
This explicit reconstruction is not a claim of observation-level equivalence.
"""

from __future__ import annotations

import builtins
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np


RECIPES = ("LettuceOnionSalad", "TomatoCarrotSalad", "PotatoBroccoliSalad")
RECIPE_INGREDIENTS = {
    "LettuceOnionSalad": ("Lettuce", "Onion"),
    "TomatoCarrotSalad": ("Tomato", "Carrot"),
    "PotatoBroccoliSalad": ("Potato", "Broccoli"),
}
OBSERVATION_SCHEMA = "pace_dense_partial_93_plus_reserved12_v1"
UPSTREAM_COMMIT = "f008c19aa743ec209550375c9ee9c25379ac8b24"


@dataclass(frozen=True)
class EnvironmentConfig:
    recipes: tuple[str, ...] = RECIPES
    horizon: int = 40
    interact_reward: float = 0.5
    progress_reward: float = 1.0
    complete_reward: float = 10.0
    step_cost: float = 0.0
    observation_dim: int = 105
    layout: str = "paper_figure1"
    observation_schema: str = OBSERVATION_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(self, "recipes", tuple(self.recipes))
        if not self.recipes or any(recipe not in RECIPES for recipe in self.recipes):
            raise ValueError(f"recipes must be a nonempty subset of {RECIPES}")
        if len(set(self.recipes)) != len(self.recipes):
            raise ValueError("recipes must not contain duplicates")
        if self.horizon < 1:
            raise ValueError("horizon must be positive")
        if self.observation_dim != 105 or self.layout != "paper_figure1":
            raise ValueError("only the 105-slot paper_figure1 reconstruction is supported")
        if self.observation_schema != OBSERVATION_SCHEMA:
            raise ValueError(f"unsupported observation schema: {self.observation_schema}")
        rewards = (self.interact_reward, self.progress_reward, self.complete_reward, self.step_cost)
        if not all(np.isfinite(value) and value >= 0 for value in rewards):
            raise ValueError("reward magnitudes and step_cost must be finite and nonnegative")


@lru_cache(maxsize=1)
def upstream_modules() -> dict[str, ModuleType]:
    """Execute upstream modules with a local import map, never sys.modules aliases.

    PACE's package initializer imports obsolete gym/rebar packages.  The world
    and recipe modules themselves need only Python and NumPy.  A module-local
    import function resolves their historical absolute imports without loading
    that initializer, editing upstream files, or shadowing installed packages.
    """
    root = Path(__file__).resolve().parents[2] / "external" / "PACE" / "environment" / "overcooked"
    relative_names = (
        "cooking_world.constants",
        "cooking_world.abstract_classes",
        "cooking_world.world_objects",
        "cooking_world.cooking_world",
        "cooking_book.recipe",
        "cooking_book.recipe_drawer",
    )
    modules: dict[str, ModuleType] = {}

    def local_import(name: str, globals=None, locals=None, fromlist=(), level=0):
        if name.startswith("gym_cooking."):
            if level or not fromlist or name not in modules:
                raise ImportError(f"unsupported upstream import: {name}")
            return modules[name]
        return builtins.__import__(name, globals, locals, fromlist, level)

    for relative_name in relative_names:
        public_name = f"gym_cooking.{relative_name}"
        source = root / "gym_cooking" / Path(*relative_name.split(".")).with_suffix(".py")
        if not source.is_file():
            raise FileNotFoundError("Pinned PACE source is missing; run scripts/bootstrap_upstream.py")
        module = ModuleType(f"_nestrl_pace.{relative_name}")
        module.__file__ = str(source)
        module.__dict__["__builtins__"] = dict(vars(builtins), __import__=local_import)
        modules[public_name] = module
        exec(compile(source.read_bytes(), str(source), "exec"), module.__dict__)
    return modules


class CookingEnv:
    """Two-player, simultaneous-action cooking environment.

    ``reset(seed)`` returns a float32 array of shape (2, 105). ``step`` accepts
    two actions and returns observation, shared reward, terminated, truncated,
    and info. Seed is retained for reproducibility; the Figure 1 layout is fixed.
    """

    observation_dim = 105
    n_actions = 6
    n_agents = 2

    def __init__(self, config: EnvironmentConfig | Mapping[str, Any] | None = None):
        if config is None:
            config = EnvironmentConfig()
        elif isinstance(config, Mapping):
            config = EnvironmentConfig(**dict(config))
        elif not isinstance(config, EnvironmentConfig):
            config = EnvironmentConfig(**asdict(config))
        self.config = config
        self.horizon = config.horizon
        modules = upstream_modules()
        self.objects = modules["gym_cooking.cooking_world.world_objects"]
        self._world_class = modules["gym_cooking.cooking_world.cooking_world"].CookingWorld
        self._recipe_module = modules["gym_cooking.cooking_book.recipe_drawer"]
        self.seed = None
        self.reset()

    def _make_world(self):
        objects = self.objects
        world = self._world_class()
        world.width, world.height = 7, 8
        # Maintain the upstream parser's object insertion order.
        for y in range(world.height):
            for x in range(world.width):
                is_counter = x in (0, 3, 6) or y in (0, 7)
                world.add_object((objects.Counter if is_counter else objects.Floor)((x, y)))
        for name, location in (
            ("CutBoard", (0, 4)), ("CutBoard", (5, 7)), ("DeliverSquare", (6, 3)),
        ):
            original = world.get_objects_at(location, objects.StaticObject)[0]
            world.delete_object(original)
            world.add_object(getattr(objects, name)(location))
        for name, location in (
            ("Plate", (3, 3)), ("Lettuce", (0, 3)), ("Tomato", (0, 1)),
            ("Potato", (0, 2)), ("Onion", (6, 4)), ("Carrot", (6, 5)),
            ("Broccoli", (6, 6)),
        ):
            world.add_object(getattr(objects, name)(location))
        for index, location in enumerate(((2, 5), (5, 5))):
            world.agents.append(objects.Agent(location, world.COLORS[index], f"agent-{index + 1}"))
        world.index_objects()
        return world

    def reset(self, seed: int | None = None) -> np.ndarray:
        if seed is not None:
            self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)
        self.world = self._make_world()
        self.recipe_graphs = [self._recipe_module.RECIPES[name]() for name in self.config.recipes]
        for graph in self.recipe_graphs:
            graph.update_recipe_state(self.world)
        self._objects_in_recipes = {node.root_type.__name__ for graph in self.recipe_graphs for node in graph.node_list}
        self._rewarded_holds: set[tuple[Any, ...]] = set()
        self.t = 0
        self.success = False
        self.completed_recipe: str | None = None
        self._done = False
        return self.observe()

    def visible(self, agent_id: int, location: Sequence[int]) -> bool:
        """Own compartment plus the shared divider is visible."""
        own_x = self.world.agents[agent_id].location[0]
        return location[0] <= 3 if own_x < 3 else location[0] >= 3

    def observe(self) -> np.ndarray:
        return np.stack([self._observe_agent(index) for index in range(2)])

    def _observe_agent(self, agent_id: int) -> np.ndarray:
        o = self.objects
        agent = self.world.agents[agent_id]
        own_pos = np.asarray(agent.location)
        shape = np.asarray([self.world.width, self.world.height], dtype=np.float32)
        positions, directions, states = [], [], []
        for index in (agent_id, 1 - agent_id):
            other = self.world.agents[index]
            visible = index == agent_id or self.visible(agent_id, other.location)
            position = np.asarray(other.location) if index == agent_id else np.asarray(other.location) - own_pos
            positions.extend([*(position / shape * visible), float(visible)])
            directions.extend([*(np.eye(4)[other.orientation - 1] * visible), float(visible)])
        next_to_counter = np.zeros(4)
        for objects in self.world.world_objects.values():
            for obj in objects:
                if isinstance(obj, o.Floor):
                    continue
                if not isinstance(obj, o.Counter):
                    visible = self.visible(agent_id, obj.location)
                    positions.extend([*((np.asarray(obj.location) - own_pos) / shape * visible), float(visible)])
                if isinstance(obj, o.Food):
                    visible = self.visible(agent_id, obj.location)
                    states.extend([float(obj.done() and visible), float(visible)])
                elif isinstance(obj, o.StaticObject):
                    delta = tuple(np.asarray(obj.location) - own_pos)
                    if delta in ((1, 0), (0, 1), (-1, 0), (0, -1)):
                        next_to_counter[((1, 0), (0, 1), (-1, 0), (0, -1)).index(delta)] = 1
        holding = np.zeros(len(o.GAME_CLASSES_HOLDABLE_IDX))
        holding_state = np.zeros(len(o.FOOD_CLASSES))
        if agent.holding is not None:
            held = agent.holding
            holding[o.GAME_CLASSES_HOLDABLE_IDX[type(held).__name__]] = 1
            contents = [held] if isinstance(held, o.Food) else held.content
            for obj in contents:
                name = type(obj).__name__
                holding[o.GAME_CLASSES_HOLDABLE_IDX[name]] = 1
                holding_state[o.FOOD_CLASSES_IDX[name]] = obj.done()
        front = np.zeros(len(o.GAME_CLASSES) - 1)
        front_state = np.zeros(len(o.FOOD_CLASSES))
        target = self.world.get_target_location(agent, agent.orientation)
        for obj in self.world.get_objects_at(target):
            if isinstance(obj, o.Floor):
                continue
            name = type(obj).__name__
            front[o.OBJ_IDX[name]] = 1
            if isinstance(obj, o.Food):
                front_state[o.FOOD_CLASSES_IDX[name]] = obj.done()
        observation = np.concatenate((positions, directions, states, next_to_counter, holding,
                                      holding_state, front, front_state, np.zeros(12))).astype(np.float32)
        if observation.shape != (105,):
            raise RuntimeError(f"observation schema drift: {observation.shape}")
        return observation

    def _snapshot(self) -> dict[int, tuple[Any, ...]]:
        o = self.objects
        return {
            id(obj): (type(obj).__name__, obj.location, obj.done() if isinstance(obj, o.Food) else None)
            for obj in self.world.get_object_list() if isinstance(obj, o.DynamicObject)
        }

    def step(self, actions: Sequence[int]) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if self._done:
            raise RuntimeError("episode is over; call reset before step")
        if len(actions) != 2 or any(not isinstance(action, (int, np.integer)) or not 0 <= action < 6 for action in actions):
            raise ValueError("actions must be two integers in [0, 5]")
        before = self._snapshot()
        self.world.perform_agent_actions(self.world.agents, [int(action) for action in actions])
        self.t += 1
        events: list[dict[str, Any]] = []
        for object_id, (name, location, done) in self._snapshot().items():
            previous = before[object_id]
            if done and not previous[2]:
                events.append({"type": "chop", "object": name, "location": list(location)})
            if location != previous[1]:
                events.append({"type": "move_object", "object": name, "from": list(previous[1]), "to": list(location)})
        rewards = np.zeros(len(self.recipe_graphs), dtype=np.float64)
        o = self.objects
        for index, agent in enumerate(self.world.agents):
            held = agent.holding
            if held is None or type(held).__name__ not in self._objects_in_recipes:
                continue
            if isinstance(held, o.Food):
                state = (bool(held.done()),)
            else:
                state = tuple(sorted(type(food).__name__ for food in held.content))
            signature = (index, id(held), state)
            if signature not in self._rewarded_holds:
                self._rewarded_holds.add(signature)
                rewards += self.config.interact_reward
                events.append({"type": "new_hold", "agent": index, "object": type(held).__name__, "state": list(state)})
        for index, graph in enumerate(self.recipe_graphs):
            if not graph.is_completable(self.world):
                continue
            before_goals = sum(graph.goals_completed(self._recipe_module.NUM_GOALS))
            graph.update_recipe_state(self.world)
            after_goals = sum(graph.goals_completed(self._recipe_module.NUM_GOALS))
            if graph.completed():
                rewards[index] += self.config.complete_reward
                self.success = True
                self.completed_recipe = graph.name
            else:
                rewards[index] += (before_goals - after_goals) * self.config.progress_reward
        reward = float(max(rewards) - self.config.step_cost)
        terminated = self.success
        truncated = self.t >= self.config.horizon and not terminated
        self._done = terminated or truncated
        if terminated:
            events.append({"type": "complete", "recipe": self.completed_recipe})
        info = {"success": self.success, "recipe": self.completed_recipe, "events": events, "t": self.t,
                "observation_schema": OBSERVATION_SCHEMA}
        return self.observe(), reward, terminated, truncated, info

    def close(self) -> None:
        pass
