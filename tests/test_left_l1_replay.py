"""Left-side replay must distinguish cooperation from mismatched plate contents."""
import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from analyze_left_l1_replay import pickup_distances, replay_episode, summarize
from nestrl_research.environment import CookingEnv, EnvironmentConfig
from nestrl_research.peers import ScriptedPeer


def saved_episode(left_recipe, right_recipe, limited=False):
    config = EnvironmentConfig(visibility="full", max_ingredients_per_agent=1 if limited else None,
        observation_schema="pace_dense_full_93_plus_ingredient_locks12_v1" if limited else "pace_dense_full_93_plus_reserved12_v1")
    env = CookingEnv(config)
    env.reset(seed=6001)
    left, right = ScriptedPeer(left_recipe, 0), ScriptedPeer(right_recipe, 1)
    trace, total = [], 0.0
    for t in range(40):
        actions = [left.act(env), right.act(env)]
        _, reward, done, truncated, info = env.step(actions)
        total += reward
        trace.append({"step": t, "actions": actions, "reward": reward, "success": info["success"], "events": info["events"]})
        if done or truncated:
            break
    episode = {"partner_id": f"l0-1-{right_recipe}", "round": 0, "episode_in_round": 0,
               "seed": 6001, "steps": len(trace), "success": env.success,
               "recipe": env.completed_recipe, "return": total}
    return config, episode, trace


class LeftReplayTests(unittest.TestCase):
    def test_left_pickup_distance_matches_expert_from_every_initial_pose(self):
        for x in (1, 2):
            for y in range(1, 7):
                for direction in range(1, 5):
                    for recipe, food in (("LettuceOnionSalad", "Lettuce"),
                                         ("TomatoCarrotSalad", "Tomato"),
                                         ("PotatoBroccoliSalad", "Potato")):
                        env = CookingEnv()
                        env.world.agents[0].move_to((x, y))
                        env.world.agents[0].orientation = direction
                        expected = pickup_distances(env)[food]
                        peer = ScriptedPeer(recipe, 0)
                        for count in range(1, 20):
                            env.step([peer.act(env), 0])
                            if env.world.agents[0].holding is not None:
                                break
                        self.assertEqual(count, expected, (x, y, direction, recipe))

    def test_correct_recipe_replay_has_exact_scene_and_1_based_events(self):
        config, ep, trace = saved_episode("LettuceOnionSalad", "LettuceOnionSalad", True)
        row, frames = replay_episode(CookingEnv(config), ep, trace, True)
        self.assertTrue(row["success"])
        self.assertEqual(row["chop_order"], ["Lettuce"])
        self.assertEqual(len(frames), ep["steps"] + 1)
        self.assertEqual(frames[row["first_chop_t"]]["f"][0][2], 1)
        self.assertEqual(frames[row["first_chop_t"] - 1]["f"][0][2], 0)
        self.assertEqual(set(row["final_plate"]), {"Lettuce", "Onion"})
        self.assertIsNone(row["first_wrong_plate_t"])
        self.assertTrue(frames[-1]["success"])
        self.assertEqual(row["delivery_reward"], 10)
        self.assertEqual(frames[-1]["lock"], [0, 3])

    def test_wrong_left_food_is_not_misreported_as_waiting_for_delivery(self):
        config, ep, trace = saved_episode("LettuceOnionSalad", "TomatoCarrotSalad", True)
        row, _ = replay_episode(CookingEnv(config), ep, trace)
        self.assertFalse(row["success"])
        self.assertEqual(row["expected_left_food"], "Tomato")
        self.assertFalse(row["chopped_expected"])
        self.assertTrue(row["wrong_food_chopped"])
        self.assertIsNotNone(row["first_wrong_plate_t"])
        self.assertEqual(set(row["final_plate"]), {"Lettuce", "Carrot"})
        self.assertGreater(row["l0_waiting_after_own_plate_steps"], 0)
        self.assertEqual(row["delivery_reward"], 0)
        self.assertGreater(row["return"], 0)
        self.assertEqual(summarize([row])["chop_order"], {"Lettuce": 1})

    def test_replay_rejects_changed_reward_peer_action_and_episode_boundary(self):
        config, ep, trace = saved_episode("LettuceOnionSalad", "LettuceOnionSalad")
        for kind in ("reward", "peer", "boundary"):
            bad = copy.deepcopy(trace)
            if kind == "reward":
                bad[0]["reward"] += 1
            elif kind == "peer":
                bad[0]["actions"][1] = (bad[0]["actions"][1] + 1) % 6
            else:
                bad = bad[:-1]
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                replay_episode(CookingEnv(config), ep, bad)
