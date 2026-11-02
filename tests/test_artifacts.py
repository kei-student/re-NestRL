"""Reject invalid experimental contracts and accidental run overwrites."""

from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

from nestrl_research.artifacts import new_run
from nestrl_research.config import ExperimentConfig, PPOConfig


class ContractTests(unittest.TestCase):
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
