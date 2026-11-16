"""The frozen evaluation plan must preserve the trained mirrored task."""
import json
from pathlib import Path
import unittest

from nestrl_research.config import load_config
from nestrl_research.pipeline import registry_spec

ROOT = Path(__file__).resolve().parents[1]

class MirroredEvaluationPlanTests(unittest.TestCase):
    def test_full_budget_all_models_all_frozen_partners_same_schedule(self):
        plan = json.loads((ROOT / "configs/evaluations/l1-mirrored-full-single-20261116.json").read_text())
        cfg = load_config(ROOT / plan["training_config"])
        self.assertEqual(plan["training_config_sha256"], cfg.sha256)
        self.assertEqual(plan["seeds"], list(cfg.construction_seeds))
        self.assertEqual(plan["checkpoint_steps"], cfg.total_steps)
        self.assertEqual(plan["checkpoint_updates"], cfg.total_steps // cfg.batch_size)
        self.assertEqual((cfg.l1_ego_side, cfg.environment.observation_dim), (0, 108))
        self.assertEqual(cfg.environment.layout, "mirrored_dual_delivery_v1")
        self.assertEqual((cfg.environment.visibility, cfg.environment.max_ingredients_per_agent), ("full", 1))
        self.assertTrue(cfg.environment.random_start)
        self.assertEqual(plan["partner_ids"], [p["id"] for p in registry_spec(cfg)["l0"] if p["side"] == 1])
        self.assertEqual((plan["rounds_per_partner"], plan["episodes_per_round"]), (200, cfg.episodes_per_round))
        self.assertEqual(plan["episodes_per_partner"], plan["rounds_per_partner"] * plan["episodes_per_round"])
        self.assertEqual((plan["episodes_per_partner"], plan["minimum_successes_per_partner"]), (1000, 950))
        self.assertEqual(plan["evaluation_seed"], 6001)
        self.assertFalse(plan["deterministic_actions"])
        self.assertFalse(plan["weight_updates"])
        self.assertEqual((plan["device"], plan["threads_per_worker"], plan["workers"]), ("cpu", 1, 4))
