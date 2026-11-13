"""Right-side frozen L0 serves; left-side L1 learns preparation and handoff."""
from dataclasses import asdict, replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from nestrl_research.artifacts import file_hash, load_checkpoint
from nestrl_research.config import ExperimentConfig, EnvironmentConfig, PPOConfig, load_config
from nestrl_research.environment import CookingEnv, RECIPES, RECIPE_INGREDIENTS
from nestrl_research.evaluation import evaluate
from nestrl_research.peers import ScriptedPeer
from nestrl_research.pipeline import train

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "docs/experiments/l1-left-l0-serving-3env-400u-s101-20261113-plan.json"


class RightServingL0Tests(unittest.TestCase):
    def test_partial_l1_can_wait_for_visible_food_then_follow_within_horizon(self):
        p = json.loads(PLAN.read_text())
        c = load_config(ROOT / p["conditions"][1]["config"])
        env = CookingEnv(asdict(c.environment))
        mapping = {foods[1]: recipe for recipe, foods in RECIPE_INGREDIENTS.items()}
        for recipe in RECIPES:
            for lx in (1, 2):
                for ly in range(1, 7):
                    for rx in (4, 5):
                        for ry in range(1, 7):
                            env.reset(seed=101)
                            env.world.agents[0].move_to((lx, ly))
                            env.world.agents[1].move_to((rx, ry))
                            l0 = ScriptedPeer(recipe, agent_id=1)
                            follower = None
                            for _ in range(40):
                                if follower is None:
                                    for food_name, inferred_recipe in mapping.items():
                                        food = env.world.world_objects[food_name][0]
                                        if env.visible(0, food.location):
                                            follower = ScriptedPeer(inferred_recipe, agent_id=0)
                                            break
                                left_action = 0 if follower is None else follower.act(env)
                                _, _, done, truncated, info = env.step([left_action, l0.act(env)])
                                if done or truncated:
                                    break
                            self.assertTrue(done, (recipe, (lx, ly), (rx, ry), env.t))
                            self.assertEqual(info["recipe"], recipe)

    def test_all_joint_starts_and_recipes_are_solvable_by_right_serving_l0(self):
        plan = json.loads(PLAN.read_text())
        left = [(x, y) for x in (1, 2) for y in range(1, 7)]
        right = [(x, y) for x in (4, 5) for y in range(1, 7)]
        count = 0
        for condition in plan["conditions"]:
            config = load_config(ROOT / condition["config"])
            env = CookingEnv(asdict(config.environment))
            for recipe in RECIPES:
                for left_start in left:
                    for right_start in right:
                        env.reset(seed=101)
                        env.world.agents[0].move_to(left_start)
                        env.world.agents[1].move_to(right_start)
                        left_expert = ScriptedPeer(recipe, agent_id=0)
                        l0 = ScriptedPeer(recipe, agent_id=1)
                        total = 0
                        for _ in range(40):
                            self.assertNotIsInstance(env.world.agents[0].holding, env.objects.Plate)
                            was_carrying_plate = isinstance(env.world.agents[1].holding, env.objects.Plate)
                            joint = [left_expert.act(env), l0.act(env)]
                            _, reward, done, truncated, info = env.step(joint)
                            total += reward
                            if done or truncated:
                                break
                        self.assertTrue(done, (condition["label"], recipe, left_start, right_start))
                        self.assertEqual(info["recipe"], recipe)
                        self.assertTrue(was_carrying_plate)
                        self.assertEqual(joint[1], 5)
                        self.assertEqual(env.world.world_objects["Plate"][0].location, (6, 3))
                        self.assertEqual(total, 16.5)
                        if config.environment.max_ingredients_per_agent == 1:
                            self.assertEqual(env.chosen_ingredients, list(RECIPE_INGREDIENTS[recipe]))
                        count += 1
        self.assertEqual(count, 1296)

    def test_l0_prepares_own_food_and_waits_without_taking_incomplete_plate(self):
        env = CookingEnv(EnvironmentConfig(random_start=False))
        l0 = ScriptedPeer(RECIPES[0], agent_id=1)
        plate = env.world.world_objects["Plate"][0]
        for _ in range(20):
            env.step([0, l0.act(env)])
            if env.world.world_objects["Onion"][0] in plate.content:
                break
        self.assertEqual([type(f).__name__ for f in plate.content], ["Onion"])
        for _ in range(5):
            self.assertEqual(l0.act(env), 0)
            env.step([0, l0.act(env)])
            self.assertIsNone(env.world.agents[1].holding)
            self.assertEqual(plate.location, (3, 3))
            self.assertFalse(env.success)

    def test_l0_does_not_serve_wrong_or_extra_ingredients_or_change_recipe(self):
        for names in (("Tomato", "Onion"), ("Lettuce", "Onion", "Carrot")):
            env = CookingEnv()
            plate = env.world.world_objects["Plate"][0]
            for name in names:
                food = env.world.world_objects[name][0]
                food.chop()
                plate.add_content(food)
            plate.move_to((3, 3))
            l0 = ScriptedPeer(RECIPES[0], agent_id=1)
            for _ in range(5):
                self.assertEqual(l0.act(env), 0)
                env.step([0, l0.act(env)])
                self.assertEqual(l0.recipe, RECIPES[0])
                self.assertFalse(env.success)
                self.assertEqual(plate.location, (3, 3))

    def test_roles_configs_and_task_namespaces_are_fixed_and_separate(self):
        p = json.loads(PLAN.read_text())
        self.assertEqual(len(p["tasks"]), 5)
        self.assertEqual(p["role_assignment"]["l0_side"], 1)
        self.assertEqual(p["role_assignment"]["l1_side"], 0)
        roots = []
        for condition in (1, 2, 3):
            tasks = [t for t in p["tasks"] if t["condition"] == condition]
            self.assertEqual([len(t["seeds"]) for t in tasks], [4] if condition == 1 else [2, 2])
            self.assertEqual([s for t in tasks for s in t["seeds"]], [101, 102, 103, 104])
            for t in tasks:
                c = load_config(ROOT / t["config"])
                self.assertEqual((c.l1_ego_side, c.ego_side), (0, 1))
                self.assertEqual(c.sha256, t["config_sha256"])
                self.assertEqual(c.total_steps // c.batch_size, 400)
                self.assertTrue(c.environment.random_start)
                self.assertNotIn("--max-updates", t["launch_command"])
                self.assertIn("l1-left-l0-serving", t["output_root"])
                roots.append(t["output_root"])
        self.assertEqual(len(set(roots)), 5)


class LeftL1PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_training_and_evaluation_use_left_l1_and_only_right_l0(self):
        config = ExperimentConfig(profile="test-left-l1", environment=EnvironmentConfig(horizon=3),
            num_envs=2, rollout_steps=4, total_steps=8,
            ppo=PPOConfig(epochs=1, minibatches=2, bptt_steps=2),
            construction_seeds=(101,), validation_seeds=(111,), test_seeds=(121,),
            l1_ego_side=0, ego_side=1, evaluation_rounds=1)
        with tempfile.TemporaryDirectory() as d, patch("builtins.print"):
            root = Path(d)
            checkpoint = train(config, root, "l1", torch.device("cpu"), seed=101)
            _, manifest = load_checkpoint(checkpoint, "cpu", config.sha256)
            self.assertEqual(manifest["ego_side"], 0)
            self.assertEqual({p["id"] for p in manifest["training_partners"]}, {f"l0-1-{r}" for r in RECIPES})
            before = file_hash(checkpoint)
            summary = evaluate(config, root, checkpoint, torch.device("cpu"), partner_level="l0", run_id="eval-left")
            self.assertEqual(set(summary["per_partner"]), {f"l0-1-{r}" for r in RECIPES})
            self.assertEqual(before, file_hash(checkpoint))

    def test_left_l1_cpu_resume_retains_model_optimizer_and_history_exactly(self):
        config = ExperimentConfig(profile="test-left-resume", environment=EnvironmentConfig(horizon=7),
            num_envs=2, rollout_steps=4, total_steps=32,
            ppo=PPOConfig(epochs=1, minibatches=2, bptt_steps=2),
            construction_seeds=(101,), validation_seeds=(111,), test_seeds=(121,), l1_ego_side=0, ego_side=1)
        with tempfile.TemporaryDirectory() as d, patch("builtins.print"):
            root = Path(d)
            full = train(config, root / "full", "l1", torch.device("cpu"), seed=101)
            train(config, root / "resumed", "l1", torch.device("cpu"), seed=101, max_updates=2)
            resumed = train(config, root / "resumed", "l1", torch.device("cpu"), seed=101, resume=True)
            a, _ = load_checkpoint(full, "cpu", config.sha256)
            b, _ = load_checkpoint(resumed, "cpu", config.sha256)
            torch.testing.assert_close(a, b, rtol=0, atol=0)
            self.assertEqual(json.loads((full.parents[1] / "episodes.json").read_text()),
                             json.loads((resumed.parents[1] / "episodes.json").read_text()))


if __name__ == "__main__":
    unittest.main()
