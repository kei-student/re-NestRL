"""Test the visibility/cooking-limit matrix without training any policies."""

from dataclasses import asdict, replace
import json
from pathlib import Path
import unittest

import numpy as np

from nestrl_research.config import ExperimentConfig, load_config
from nestrl_research.environment import CookingEnv, EnvironmentConfig, INGREDIENT_NAMES, RECIPES, observation_schema_for
from nestrl_research.peers import ScriptedPeer

ROOT = Path(__file__).resolve().parents[1]
VARIANTS = ("full", "single-ingredient", "full-single-ingredient")


def variant(visibility="own_side", limit=None, **kwargs):
    return EnvironmentConfig(visibility=visibility, max_ingredients_per_agent=limit,
                             observation_schema=observation_schema_for(visibility, limit), **kwargs)


def prepare_first_then_second(env):
    """Actually cook/store Onion, then attempt to chop Carrot, using actions."""
    trace = []

    def step(action):
        trace.append([0, action])
        return env.step(trace[-1])

    first = ScriptedPeer(RECIPES[0], agent_id=1)
    for _ in range(20):
        result = step(first.act(env))
        if any(e["type"] == "chop" for e in result[-1]["events"]):
            break
    else:
        raise AssertionError("first ingredient never chopped")
    step(first.act(env))  # retrieve chopped Onion
    for _ in range(15):
        step(first._navigation_action(env.world, 1, (6, 4)))
        if env.world.agents[1].holding is None:
            break
    else:
        raise AssertionError("could not store Onion")
    second = ScriptedPeer(RECIPES[1], agent_id=1)
    for _ in range(15):
        result = step(second.act(env))
        if any(e["type"] in ("chop", "ingredient_limit_blocked") for e in result[-1]["events"]):
            return result, trace
    raise AssertionError("second ingredient never reached board")


class VisibilityTests(unittest.TestCase):
    def test_full_view_reveals_both_rooms_positions_directions_and_food_states(self):
        env = CookingEnv(variant("full"))
        obs = env.reset(seed=12)
        self.assertEqual(obs.shape, (2, 105))
        np.testing.assert_array_equal(obs[:, :36].reshape(2, 12, 3)[:, :, 2], 1)
        np.testing.assert_array_equal(obs[:, [40, 45]], 1)  # both orientation visibility bits
        np.testing.assert_array_equal(obs[:, 47:58:2], 1)  # all six food visibility bits
        np.testing.assert_array_equal(obs[:, 93:], 0)
        before = obs[1].copy()
        env.world.agents[0].move_to((1, 1))
        env.world.agents[0].change_orientation(3)
        env.world.world_objects["Lettuce"][0].chop()
        after = env.observe()[1]
        self.assertFalse(np.array_equal(before, after))
        np.testing.assert_allclose(after[3:6], [*(np.array((1, 1)) - env.world.agents[1].location) / (7, 8), 1])
        self.assertEqual(after[46], 1)  # Lettuce chopped

    def test_visibility_changes_observations_without_changing_physics_or_rewards(self):
        partial, full = CookingEnv(), CookingEnv(variant("full"))
        for recipe in RECIPES:
            partial.reset(seed=15)
            full.reset(seed=15)
            peers = [ScriptedPeer(recipe, agent_id=i) for i in (0, 1)]
            for _ in range(40):
                actions = [p.act(partial) for p in peers]
                _, r1, d1, t1, i1 = partial.step(actions)
                _, r2, d2, t2, i2 = full.step(actions)
                self.assertEqual((r1, d1, t1, i1["events"]), (r2, d2, t2, i2["events"]))
                self.assertEqual([a.location for a in partial.world.agents], [a.location for a in full.world.agents])
                if d1 or t1:
                    break
            self.assertTrue(d1)


class IngredientLimitTests(unittest.TestCase):
    def test_second_type_is_blocked_without_chop_reward_and_history_replay_restores_lock(self):
        for visibility in ("own_side", "full"):
            with self.subTest(visibility=visibility):
                config = variant(visibility, 1, random_start=False)
                env = CookingEnv(config)
                env.reset(seed=8)
                (obs, reward, _, _, info), actions = prepare_first_then_second(env)
                self.assertEqual(env.chosen_ingredients, [None, "Onion"])
                self.assertFalse(env.world.world_objects["Carrot"][0].done())
                self.assertTrue(env.world.world_objects["Onion"][0].done())
                self.assertEqual(reward, 0)
                self.assertEqual(info["executed_actions"], [0, 0])
                self.assertIn({"type": "ingredient_limit_blocked", "agent": 1, "object": "Carrot", "chosen": "Onion"}, info["events"])
                self.assertEqual(obs[1, 93 + INGREDIENT_NAMES.index("Onion")], 1)
                replay = CookingEnv(config)
                replay.reset(seed=8)
                for pair in actions:
                    replay_obs, *_ = replay.step(pair)
                np.testing.assert_array_equal(obs, replay_obs)
                self.assertEqual(env.chosen_ingredients, replay.chosen_ingredients)
                _, again_reward, _, _, again = env.step([0, 5])
                self.assertEqual(again_reward, 0)
                self.assertTrue(any(e["type"] == "ingredient_limit_blocked" for e in again["events"]))

    def test_unrestricted_condition_still_cooks_multiple_ingredients(self):
        env = CookingEnv(variant(random_start=False))
        env.reset(seed=8)
        (_, _, _, _, info), _ = prepare_first_then_second(env)
        self.assertTrue(env.world.world_objects["Carrot"][0].done())
        self.assertFalse(any(e["type"] == "ingredient_limit_blocked" for e in info["events"]))

    def test_raw_pickups_do_not_commit_and_reset_allows_a_new_type(self):
        env = CookingEnv(variant(limit=1))
        env.reset(seed=1)
        agent = env.world.agents[1]
        agent.move_to((5, 4))
        agent.change_orientation(2)
        env.step([0, 5])  # raw Onion pickup
        self.assertEqual(type(agent.holding).__name__, "Onion")
        self.assertEqual(env.chosen_ingredients, [None, None])
        env.step([0, 5])  # replace raw Onion
        env.world.world_objects["Carrot"][0].move_to((5, 7))
        agent.move_to((5, 6))
        agent.change_orientation(3)
        env.step([0, 5])
        self.assertEqual(env.chosen_ingredients, [None, "Carrot"])
        obs = env.reset(seed=2)
        self.assertEqual(env.chosen_ingredients, [None, None])
        np.testing.assert_array_equal(obs[:, 93:], 0)
        env.world.world_objects["Broccoli"][0].move_to((5, 7))
        env.world.agents[1].move_to((5, 6))
        env.world.agents[1].change_orientation(3)
        env.step([0, 5])
        self.assertEqual(env.chosen_ingredients, [None, "Broccoli"])

    def test_two_agents_commit_independently_and_partial_view_hides_peer_lock(self):
        for visibility in ("own_side", "full"):
            env = CookingEnv(variant(visibility, 1))
            for side, (food, board, position, direction) in enumerate((
                ("Tomato", (0, 4), (1, 4), 1), ("Broccoli", (5, 7), (5, 6), 3)
            )):
                env.world.world_objects[food][0].move_to(board)
                env.world.agents[side].move_to(position)
                env.world.agents[side].change_orientation(direction)
            obs, _, _, _, info = env.step([5, 5])
            self.assertEqual(env.chosen_ingredients, ["Tomato", "Broccoli"])
            self.assertEqual(len([e for e in info["events"] if e["type"] == "ingredient_selected"]), 2)
            self.assertEqual(obs[0, 93 + INGREDIENT_NAMES.index("Tomato")], 1)
            self.assertEqual(obs[1, 93 + INGREDIENT_NAMES.index("Broccoli")], 1)
            if visibility == "own_side":
                np.testing.assert_array_equal(obs[:, 99:], 0)
            else:
                self.assertEqual(obs[0, 99 + INGREDIENT_NAMES.index("Broccoli")], 1)
                self.assertEqual(obs[1, 99 + INGREDIENT_NAMES.index("Tomato")], 1)

    def test_all_three_variants_keep_all_recipes_solvable_from_random_starts(self):
        for visibility, limit in (("full", None), ("own_side", 1), ("full", 1)):
            env = CookingEnv(variant(visibility, limit))
            for recipe in RECIPES:
                for seed in range(12):
                    with self.subTest(visibility=visibility, limit=limit, recipe=recipe, seed=seed):
                        env.reset(seed=seed)
                        peers = [ScriptedPeer(recipe, agent_id=i) for i in (0, 1)]
                        total = 0
                        for _ in range(40):
                            _, reward, done, truncated, info = env.step([p.act(env) for p in peers])
                            total += reward
                            if done or truncated:
                                break
                        self.assertTrue(done)
                        self.assertEqual(info["recipe"], recipe)
                        self.assertEqual(total, 16.5)


class VariantConfigTests(unittest.TestCase):
    def test_cloud_launch_plan_assigns_exactly_four_models_per_condition(self):
        plan = json.loads((ROOT / "docs/experiments/l1-environment-variants-4agents-20261113-plan.json").read_text())
        self.assertEqual(len(plan["tasks"]), 5)
        all_seeds = []
        for condition in (1, 2, 3):
            tasks = [t for t in plan["tasks"] if t["condition"] == condition]
            self.assertEqual([len(t["seeds"]) for t in tasks], [4] if condition == 1 else [2, 2])
            cfg = load_config(ROOT / tasks[0]["config"])
            assigned = [s for t in tasks for s in t["seeds"]]
            self.assertEqual(assigned, list(cfg.construction_seeds))
            self.assertEqual(cfg.total_steps // cfg.batch_size, 400)
            for task in tasks:
                self.assertEqual(task["config_sha256"], cfg.sha256)
                self.assertNotIn("--max-updates", task["launch_command"])
                self.assertLessEqual(task["requested_memory_gib_per_model"] * len(task["seeds"]), 12)
            all_seeds.extend(assigned)
        self.assertEqual(len(all_seeds), 12)
        self.assertEqual(len(set(all_seeds)), 12)
        self.assertTrue(set(all_seeds).isdisjoint(range(1001, 1041)))

    def test_new_configs_are_disjoint_and_keep_random_starts_and_400_updates(self):
        used = set()
        baseline = load_config(ROOT / "configs/l1-random-start-400u-pilot-20261112.json")
        expected = {"full": ("full", None), "single-ingredient": ("own_side", 1),
                    "full-single-ingredient": ("full", 1)}
        for name in VARIANTS:
            config = load_config(ROOT / "configs" / f"l1-random-start-400u-{name}-20261113.json")
            self.assertTrue(config.environment.random_start)
            self.assertEqual(config.total_steps // config.batch_size, 400)
            self.assertEqual(config.total_steps, 28_800_000)
            self.assertEqual(config.episodes_per_round, 5)
            self.assertEqual((config.environment.visibility, config.environment.max_ingredients_per_agent), expected[name])
            self.assertEqual(config.ppo, baseline.ppo)
            for field in ("horizon", "interact_reward", "progress_reward", "complete_reward", "step_cost"):
                self.assertEqual(getattr(config.environment, field), getattr(baseline.environment, field))
            self.assertNotEqual(config.sha256, baseline.sha256)
            seeds = set((*config.construction_seeds, *config.validation_seeds, *config.test_seeds,
                         config.learner_seed, config.evaluation_seed))
            self.assertTrue(used.isdisjoint(seeds))
            used.update(seeds)
            self.assertEqual(CookingEnv(asdict(config.environment)).observe().shape, (2, 105))
        self.assertTrue(used.isdisjoint(range(1001, 1041)))

    def test_legacy_canonical_hashes_and_default_behavior_are_unchanged(self):
        expected = {"paper": "adb7e5ce6b32308185a2f1595e9c84a0557a4aba0331d3cdd99fe50087ca91a2",
                    "l1-random-start-400u-pilot-20261112": "fd5c44797dc80917d390eedb5007997e1a8f2d5495affd07616cb08275818454"}
        for name, digest in expected.items():
            config = load_config(ROOT / "configs" / f"{name}.json")
            self.assertEqual(config.sha256, digest)
            self.assertEqual(config.environment.visibility, "own_side")
            self.assertIsNone(config.environment.max_ingredients_per_agent)

    def test_invalid_limits_visibility_and_schema_pairs_fail_closed(self):
        for limit in (0, 2, -1, True, False, 1.0, "1"):
            with self.subTest(limit=limit), self.assertRaises(ValueError):
                variant(limit=limit)
        with self.assertRaises(ValueError):
            variant("unknown")
        with self.assertRaises(ValueError):
            EnvironmentConfig(visibility="full")  # schema still names partial view
        base = ExperimentConfig()
        with self.assertRaises(ValueError):
            replace(base, environment=replace(base.environment, visibility="full")).validate()


if __name__ == "__main__":
    unittest.main()
