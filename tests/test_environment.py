import sys
import unittest

import numpy as np

from nestrl_research.environment import CookingEnv, EnvironmentConfig, RECIPES
from nestrl_research.peers import ScriptedPeer


class EnvironmentTests(unittest.TestCase):
    def test_fixed_layout_and_observation_contract(self):
        env = CookingEnv(EnvironmentConfig(random_start=False))
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
                env = CookingEnv(EnvironmentConfig(random_start=False))
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

    def test_random_starts_cover_only_own_room_floors_and_match_observations(self):
        env = CookingEnv()
        seen = [set(), set()]
        for seed in range(128):
            obs = env.reset(seed=seed)
            locations = [agent.location for agent in env.world.agents]
            self.assertNotEqual(*locations)
            for side, agent in enumerate(env.world.agents):
                x, y = agent.location
                self.assertIn(x, (1, 2) if side == 0 else (4, 5))
                self.assertIn(y, range(1, 7))
                self.assertTrue(env.world.get_objects_at(agent.location, env.objects.Floor))
                self.assertEqual(agent.orientation, 1)
                np.testing.assert_allclose(obs[side, :3], [x / 7, y / 8, 1])
                np.testing.assert_array_equal(obs[side, 93:], 0)
                seen[side].add(agent.location)
        self.assertEqual(seen[0], {(x, y) for x in (1, 2) for y in range(1, 7)})
        self.assertEqual(seen[1], {(x, y) for x in (4, 5) for y in range(1, 7)})

    def test_random_reset_stream_advances_and_reseeding_reproduces_it(self):
        first, second = CookingEnv(), CookingEnv()
        initial = first.reset(seed=123)
        np.testing.assert_array_equal(initial, second.reset(seed=123))
        sequence = [initial]
        for _ in range(12):
            first.step((0, 0))
            sequence.append(first.reset())
            np.testing.assert_array_equal(sequence[-1], second.reset())
            self.assertEqual(first.t, 0)
            self.assertFalse(first.success)
        for side in (0, 1):
            self.assertGreater(len({tuple(obs[side, :2]) for obs in sequence}), 1)
        np.testing.assert_array_equal(first.reset(seed=123), sequence[0])
        for expected in sequence[1:]:
            np.testing.assert_array_equal(first.reset(), expected)

    def test_recipe_experts_complete_from_random_starts(self):
        env = CookingEnv()
        for recipe in RECIPES:
            for seed in range(8):
                with self.subTest(recipe=recipe, seed=seed):
                    env.reset(seed=seed)
                    peers = [ScriptedPeer(recipe, agent_id=side) for side in (0, 1)]
                    total_reward = 0.0
                    for _ in range(env.horizon):
                        _, reward, terminated, truncated, info = env.step([peer(env) for peer in peers])
                        total_reward += reward
                        if terminated or truncated:
                            break
                    self.assertTrue(terminated)
                    self.assertFalse(truncated)
                    self.assertEqual(info["recipe"], recipe)
                    self.assertAlmostEqual(total_reward, 16.5)

    def test_legacy_fixed_starts_remain_fixed_across_resets(self):
        env = CookingEnv(EnvironmentConfig(random_start=False))
        for seed in (1, None, 2, None):
            env.reset(seed=seed)
            self.assertEqual([agent.location for agent in env.world.agents], [(2, 5), (5, 5)])

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
        for value in (None, "false", 0, 1):
            with self.subTest(random_start=value), self.assertRaises(ValueError):
                EnvironmentConfig(random_start=value)

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
