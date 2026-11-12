"""Check that behavioral timing respects observation and cooking transitions."""

import copy
import unittest

from nestrl_research.behavior import analyze_episode, ego_state, pickup_distances, summarize
from nestrl_research.environment import CookingEnv
from nestrl_research.peers import ScriptedPeer


def scripted_episode(wait_for_cue=False):
    env = CookingEnv()
    env.reset(seed=6001)
    peers = [ScriptedPeer("LettuceOnionSalad", agent_id=i) for i in range(2)]
    trace = []
    total = 0
    for step in range(40):
        cue = env.visible(1, env.world.world_objects["Lettuce"][0].location)
        actions = [peers[0].act(env), 0 if wait_for_cue and not cue else peers[1].act(env)]
        _, reward, done, truncated, info = env.step(actions)
        trace.append({"step": step, "actions": actions, "reward": reward,
                      "events": info["events"], "success": info["success"]})
        total += reward
        if done or truncated:
            break
    meta = {"seed": 6001, "steps": len(trace), "return": total, "success": env.success,
            "recipe": env.completed_recipe, "partner_id": "l0-0-LettuceOnionSalad",
            "round": 0, "episode_in_round": 0}
    return meta, trace


class BehaviorTests(unittest.TestCase):
    def test_deliberately_waiting_peer_is_distinguished_from_proactive_peer(self):
        active = analyze_episode(CookingEnv(), *scripted_episode())
        waiting = analyze_episode(CookingEnv(), *scripted_episode(True))
        self.assertTrue(active["success"] and waiting["success"])
        self.assertFalse(active["pickup_after_visible_cue"])
        self.assertFalse(active["waiting_candidate"])
        self.assertTrue(waiting["pickup_after_visible_cue"])
        self.assertTrue(waiting["waiting_candidate"])
        self.assertGreater(waiting["first_pickup_step"], waiting["visible_cue_created_step"])
        self.assertEqual(active["chopped_count"], 1)
        self.assertEqual(waiting["right_plate_additions"], ["Onion"])

    def test_chopping_is_not_unchanged_stationary_state(self):
        env = CookingEnv()
        env.reset(seed=6001)
        peer = ScriptedPeer("LettuceOnionSalad", agent_id=1)
        for _ in range(30):
            before = ego_state(env)
            _, _, done, truncated, info = env.step([0, peer.act(env)])
            if any(e["type"] == "chop" for e in info["events"]):
                self.assertEqual(before[0], ego_state(env)[0])
                self.assertNotEqual(before, ego_state(env))
                return
            if done or truncated:
                break
        self.fail("scripted chef never chopped")

    def test_pickup_distance_matches_shortest_path_peer_in_all_initial_poses(self):
        for x in (4, 5):
            for y in range(1, 7):
                for direction in range(1, 5):
                    for recipe, name in (("LettuceOnionSalad", "Onion"),
                                         ("TomatoCarrotSalad", "Carrot"),
                                         ("PotatoBroccoliSalad", "Broccoli")):
                        env = CookingEnv()
                        env.world.agents[1].location = (x, y)
                        env.world.agents[1].orientation = direction
                        expected = pickup_distances(env)[name]
                        peer = ScriptedPeer(recipe, agent_id=1)
                        for count in range(1, 20):
                            env.step([0, peer.act(env)])
                            if env.world.agents[1].holding is not None:
                                break
                        self.assertEqual(count, expected, (x, y, direction, name))

    def test_replay_rejects_corrupt_events_and_summary(self):
        meta, trace = scripted_episode()
        bad = copy.deepcopy(trace)
        bad[0]["reward"] += 1
        with self.assertRaisesRegex(ValueError, "replay mismatch"):
            analyze_episode(CookingEnv(), meta, bad)
        with self.assertRaisesRegex(ValueError, "summary mismatch"):
            analyze_episode(CookingEnv(), {**meta, "return": meta["return"] + 1}, trace)

    def test_summary_retains_counts_and_timing_denominators(self):
        rows = [analyze_episode(CookingEnv(), *scripted_episode(wait)) for wait in (False, True)]
        result = summarize(rows)
        self.assertEqual(result["episodes"], 2)
        self.assertEqual(result["counts"]["waiting_candidate"], 1)
        self.assertEqual(result["rates"]["waiting_candidate"], .5)
        self.assertEqual(result["timing_denominators"]["visible_cue_created_step"], 2)


if __name__ == "__main__":
    unittest.main()
