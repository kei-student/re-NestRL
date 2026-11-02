import sys
import unittest

import numpy as np

from nestrl_research.environment import CookingEnv, EnvironmentConfig, RECIPES
from nestrl_research.peers import ScriptedPeer


class EnvironmentTests(unittest.TestCase):
    def test_fixed_layout_and_observation_contract(self):
        env = CookingEnv()
        obs = env.reset(seed=7)
        self.assertEqual(obs.shape, (2, 105))
        self.assertEqual(obs.dtype, np.float32)
        np.testing.assert_array_equal(obs[:, 93:], 0)
        self.assertEqual((env.world.width, env.world.height), (7, 8))
        self.assertEqual([agent.location for agent in env.world.agents], [(2, 5), (5, 5)])
        self.assertEqual([board.location for board in env.world.world_objects["CutBoard"]], [(0, 4), (5, 7)])
        self.assertEqual(env.world.world_objects["Plate"][0].location, (3, 3))
        self.assertFalse(env.world.square_walkable((3, 2)))
        self.assertEqual(len(env.world.world_objects["Floor"]), 24)

    def test_all_recipe_experts_coordinate_on_both_sides(self):
        expected_foods = (("Lettuce", "Onion"), ("Tomato", "Carrot"), ("Potato", "Broccoli"))
        for recipe, expected_steps, ingredients in zip(RECIPES, (16, 19, 17), expected_foods):
            with self.subTest(recipe=recipe):
                env = CookingEnv()
                env.reset(seed=13)
                peers = [ScriptedPeer(recipe, agent_id=side) for side in (0, 1)]
                total_reward = 0.0
                for step in range(1, 41):
                    _, reward, terminated, truncated, info = env.step([peer(env) for peer in peers])
                    total_reward += reward
                    if terminated or truncated:
                        break
                self.assertTrue(terminated)
                self.assertFalse(truncated)
                self.assertTrue(info["success"])
                self.assertEqual(info["recipe"], recipe)
                self.assertEqual(step, expected_steps)
                self.assertAlmostEqual(total_reward, 16.5)
                self.assertEqual({type(food).__name__ for food in env.world.world_objects["Plate"][0].content},
                                 set(ingredients))

    def test_partial_visibility_and_reserved_slots(self):
        env = CookingEnv()
        obs = env.reset()
        # Position triples: own, peer, board0, board1, delivery, plate,
        # Lettuce, Tomato, Potato, Onion, Carrot, Broccoli.
        np.testing.assert_array_equal(obs[0, 3:6], [0, 0, 0])
        np.testing.assert_array_equal(obs[1, 3:6], [0, 0, 0])
        self.assertEqual(obs[0, 3 * 5 + 2], 1)  # shared plate
        self.assertEqual(obs[1, 3 * 5 + 2], 1)
        np.testing.assert_array_equal(obs[0, 3 * 9:3 * 10], 0)  # hidden Onion
        np.testing.assert_array_equal(obs[1, 3 * 6:3 * 7], 0)  # hidden Lettuce
        # Moving a hidden opponent does not reveal its coordinates/direction.
        env.world.agents[1].move_to((4, 1))
        env.world.agents[1].change_orientation(4)
        after = env.observe()
        np.testing.assert_array_equal(after[0], obs[0])
        np.testing.assert_array_equal(after[:, 93:], 0)

    def test_noop_horizon_is_truncation_and_requires_reset(self):
        env = CookingEnv()
        initial = env.reset(seed=2)
        for step in range(40):
            obs, reward, terminated, truncated, info = env.step((0, 0))
            self.assertEqual(reward, 0)
            self.assertFalse(terminated)
            self.assertEqual(truncated, step == 39)
            self.assertFalse(info["success"])
            np.testing.assert_array_equal(obs, initial)
        with self.assertRaises(RuntimeError):
            env.step((0, 0))
        np.testing.assert_array_equal(env.reset(seed=2), initial)

    def test_repeated_pickup_does_not_repeat_interaction_bonus(self):
        env = CookingEnv()
        env.world.agents[0].move_to((1, 3))
        env.world.agents[0].change_orientation(1)
        _, first, *_ = env.step((5, 0))
        _, put_down, *_ = env.step((5, 0))
        _, repeated, *_ = env.step((5, 0))
        self.assertEqual(first, 0.5)
        self.assertEqual(put_down, 0)
        self.assertEqual(repeated, 0)

    def test_chopping_reward_is_shared_and_given_once(self):
        env = CookingEnv()
        food = env.world.world_objects["Lettuce"][0]
        food.move_to((0, 4))
        env.world.agents[0].move_to((1, 4))
        env.world.agents[0].change_orientation(1)
        _, reward, _, _, info = env.step((5, 0))
        self.assertEqual(reward, 1.0)
        self.assertTrue(food.done())
        self.assertTrue(any(event["type"] == "chop" for event in info["events"]))
        _, reward, *_ = env.step((0, 0))
        self.assertEqual(reward, 0)

    def test_other_ingredient_combinations_never_succeed(self):
        for ingredient_names in (("Lettuce", "Carrot"), ("Lettuce",), ("Lettuce", "Onion", "Carrot")):
            with self.subTest(ingredients=ingredient_names):
                env = CookingEnv()
                plate = env.world.world_objects["Plate"][0]
                for name in ingredient_names:
                    food = env.world.world_objects[name][0]
                    food.chop()
                    plate.add_content(food)
                plate.move_to((6, 3))
                _, _, terminated, _, info = env.step((0, 0))
                self.assertFalse(terminated)
                self.assertFalse(info["success"])
                self.assertIsNone(info["recipe"])

    def test_raw_food_cannot_be_plated(self):
        env = CookingEnv()
        o = env.objects
        plate = env.world.world_objects["Plate"][0]
        with self.assertRaises(Exception):
            plate.add_content(env.world.world_objects["Lettuce"][0])
        self.assertEqual(plate.content, [])
        self.assertIsInstance(env.world.world_objects["Carrot"][0], o.ChopFood)

    def test_config_and_action_validation(self):
        for actions in ((-1, 0), (6, 0), (0,), (0.1, 0)):
            with self.subTest(actions=actions), self.assertRaises(ValueError):
                CookingEnv().step(actions)
        with self.assertRaises(ValueError):
            EnvironmentConfig(recipes=("LettuceCarrotSalad",))
        with self.assertRaises(ValueError):
            EnvironmentConfig(horizon=0)

    def test_local_loader_does_not_register_legacy_packages(self):
        names = ("gym_cooking", "gym", "pettingzoo", "rebar")
        before = {name: sys.modules.get(name) for name in names}
        CookingEnv()
        self.assertEqual(before, {name: sys.modules.get(name) for name in names})

    def test_noisy_peer_is_reproducible(self):
        env = CookingEnv()
        first = ScriptedPeer(RECIPES[0], seed=9, random_action_probability=1)
        second = ScriptedPeer(RECIPES[0], seed=9, random_action_probability=1)
        self.assertEqual([first(env) for _ in range(20)], [second(env) for _ in range(20)])


if __name__ == "__main__":
    unittest.main()
