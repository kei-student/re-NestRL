"""Reject invalid experimental contracts and accidental run overwrites."""

from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

from nestrl_research.artifacts import new_run
from nestrl_research.config import ROOT, EnvironmentConfig, ExperimentConfig, PPOConfig, canonical_hash, load_config


class ContractTests(unittest.TestCase):
    def test_random_start_400_update_campaign_has_twenty_unique_assigned_seeds(self):
        campaign = "l1-random-start-400u-20261112"
        config = load_config(ROOT / "configs" / f"{campaign}.json")
        plan = json.loads((ROOT / "docs/experiments" / f"{campaign}-plan.json").read_text())
        self.assertTrue(config.environment.random_start)
        self.assertEqual((config.num_envs, config.rollout_steps), (360, 200))
        self.assertEqual(config.total_steps, 400 * config.batch_size)
        self.assertEqual(plan["config_sha256"], config.sha256)
        self.assertEqual(len(plan["tasks"]), 10)
        assigned = [seed for task in plan["tasks"] for seed in task["seeds"]]
        self.assertEqual(len(assigned), 20)
        self.assertEqual(len(set(assigned)), 20)
        self.assertEqual(set(assigned), set(config.construction_seeds))
        self.assertFalse(set(assigned) & set(plan["seed_audit"]["recorded_seeds_before_campaign"]))
        self.assertEqual(plan["total_steps"], 20 * config.total_steps)
        self.assertEqual(len({task["output_root"] for task in plan["tasks"]}), 10)
        for task in plan["tasks"]:
            self.assertEqual(len(task["seeds"]), 2)
            self.assertEqual(task["internal_run_ids"], [f"l1-construction-{seed}" for seed in task["seeds"]])
            self.assertNotIn("--max-updates", task["launch_command"])
            self.assertIn(plan["config_path"], task["launch_command"])
            self.assertEqual(task["record_command"][-len(task["launch_command"]):], task["launch_command"])

    def test_legacy_configs_keep_fixed_starts_and_original_canonical_hashes(self):
        paths = list((ROOT / "configs").glob("paper*.json"))
        paths += [ROOT / "configs/smoke.json", ROOT / "configs/history/paper-l1-112-127.json"]
        for path in paths:
            data = json.loads(path.read_text())
            if "random_start" in data["environment"]:
                continue
            with self.subTest(config=path.name):
                config = load_config(path)
                self.assertFalse(config.environment.random_start)
                self.assertEqual(config.sha256, canonical_hash(data))
                self.assertNotIn("random_start", config.to_dict()["environment"])

    def test_random_start_profiles_are_explicit_distinct_and_round_trip(self):
        self.assertTrue(ExperimentConfig().environment.random_start)
        for name in ("paper", "smoke"):
            with self.subTest(profile=name):
                legacy = load_config(ROOT / "configs" / f"{name}.json")
                config = load_config(ROOT / "configs" / f"{name}-random-start.json")
                self.assertTrue(config.environment.random_start)
                self.assertNotEqual(config.profile, legacy.profile)
                self.assertNotEqual(config.sha256, legacy.sha256)
                self.assertNotEqual(replace(legacy, environment=replace(legacy.environment, random_start=True)).sha256,
                                    legacy.sha256)
                with tempfile.TemporaryDirectory() as temporary:
                    path = Path(temporary) / "config.json"
                    path.write_text(json.dumps(config.to_dict()))
                    self.assertEqual(load_config(path).sha256, config.sha256)

    def test_invalid_random_start_is_rejected_before_training(self):
        for value in (None, "false", 0, 1):
            with self.subTest(random_start=value), self.assertRaisesRegex(ValueError, "random_start"):
                replace(ExperimentConfig(), environment=EnvironmentConfig(random_start=value)).validate()

    def test_seed_overlap_and_incomplete_update_budget_are_rejected(self):
        for config in (replace(ExperimentConfig(), test_seeds=(101,)),
                       replace(ExperimentConfig(), total_steps=72_001),
                       replace(ExperimentConfig(), rollout_steps=201)):
            with self.subTest(config=config), self.assertRaises(ValueError):
                config.validate()

    def test_invalid_ppo_numbers_fail_before_training(self):
        for changes in ({"bptt_steps": 0}, {"lr": float("nan")}, {"gamma": 1.1}, {"entropy": -0.1}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                replace(ExperimentConfig(), ppo=replace(PPOConfig(), **changes)).validate()

    def test_existing_run_or_unsafe_name_cannot_be_reused(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with new_run(root, "first") as directory:
                self.assertTrue((directory / ".writer.lock").exists())
            self.assertFalse((directory / ".writer.lock").exists())
            with self.assertRaises(FileExistsError), new_run(root, "first"):
                self.fail("An existing run was overwritten")
            with self.assertRaises(ValueError), new_run(root, "../outside"):
                self.fail("Run ID escaped its output root")

    def test_failed_run_retains_failure_evidence_and_releases_lock(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaisesRegex(RuntimeError, "fixture"), new_run(root, "failed"):
                raise RuntimeError("fixture")
            self.assertTrue((root / "failed" / "failure.json").is_file())
            self.assertFalse((root / "failed" / ".writer.lock").exists())
