"""Interrupted CPU training must reproduce uninterrupted model/optimizer state."""

from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from nestrl_research import pipeline, recovery
from nestrl_research.artifacts import load_checkpoint
from nestrl_research.config import ExperimentConfig, EnvironmentConfig, PPOConfig


class RecoveryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.config = ExperimentConfig(environment=EnvironmentConfig(horizon=7), num_envs=2,
                                       rollout_steps=4, total_steps=32,
                                       ppo=PPOConfig(epochs=1, minibatches=2, bptt_steps=2),
                                       construction_seeds=(101,), validation_seeds=(201,), test_seeds=(301,))

    def train(self, root, **kwargs):
        with patch("builtins.print"):
            return pipeline.train(self.config, root, "l1", torch.device("cpu"), seed=101, **kwargs)

    def assert_tree_equal(self, left, right):
        if isinstance(left, torch.Tensor):
            torch.testing.assert_close(left, right, rtol=0, atol=0)
        elif isinstance(left, dict):
            self.assertEqual(left.keys(), right.keys())
            for key in left:
                self.assert_tree_equal(left[key], right[key])
        elif isinstance(left, (list, tuple)):
            self.assertEqual(len(left), len(right))
            for a, b in zip(left, right):
                self.assert_tree_equal(a, b)
        else:
            self.assertEqual(left, right)

    def test_resume_is_bit_exact_including_partial_episodes_gru_optimizer_and_rng(self):
        full = self.train(self.root / "full")
        partial_root = self.root / "partial"
        pointer = self.train(partial_root, max_updates=2)
        directory = pointer.parents[1]
        self.assertFalse((directory / "complete.json").exists())
        self.assertEqual(json.loads(pointer.read_text())["update"], 2)
        # Simulate death after logging an update but before publishing its checkpoint.
        with (directory / "metrics.jsonl").open("a") as stream:
            stream.write('{"update": 3}\n{"update":')
        resumed = self.train(partial_root, resume=True)
        a, _ = load_checkpoint(full, "cpu")
        b, _ = load_checkpoint(resumed, "cpu")
        self.assert_tree_equal(a, b)
        self.assertEqual(json.loads((full.parents[1] / "episodes.json").read_text()),
                         json.loads((directory / "episodes.json").read_text()))
        metrics = [json.loads(line) for line in (directory / "metrics.jsonl").read_text().splitlines()]
        self.assertEqual([m["update"] for m in metrics], [1, 2, 3, 4])
        self.assertEqual(len(list((directory / "resume").glob("update-*.pt"))), 2)
        for name in ("full", "partial"):
            folder = self.root / name / "l1-construction-101"
            p = json.loads((folder / "resume/latest.json").read_text())
            state = torch.load(folder / "resume" / p["file"], weights_only=True)
            if name == "full":
                expected = state
            else:
                self.assert_tree_equal(expected["collector"], state["collector"])
                self.assert_tree_equal(expected["rng"], state["rng"])

    def test_interrupted_checkpoint_publish_keeps_previous_generation_recoverable(self):
        root = self.root / "run"
        pointer = self.train(root, max_updates=1)
        original = recovery.atomic_json
        def fail_pointer(path, value):
            if path == pointer:
                raise OSError("simulated disk interruption")
            return original(path, value)
        with patch.object(recovery, "atomic_json", side_effect=fail_pointer):
            with self.assertRaisesRegex(OSError, "interruption"):
                self.train(root, resume=True, max_updates=1)
        self.assertEqual(json.loads(pointer.read_text())["update"], 1)
        self.assertTrue(self.train(root, resume=True).is_file())

    def test_failure_before_first_update_can_resume_initial_state(self):
        root = self.root / "initial"
        with patch.object(pipeline, "ppo_update", side_effect=RuntimeError("before first update")):
            with self.assertRaisesRegex(RuntimeError, "first update"):
                self.train(root)
        directory = root / "l1-construction-101"
        self.assertEqual(json.loads((directory / "resume/latest.json").read_text())["update"], 0)
        # A process killed without running finally can leave this diagnostic file.
        (directory / ".writer.lock").write_text('{"pid": 999999}')
        self.assertTrue(self.train(root, resume=True).is_file())

    def test_hash_mismatch_and_changed_runtime_or_source_are_rejected(self):
        pointer = self.train(self.root / "run", max_updates=1)
        directory = pointer.parents[1]
        p = json.loads(pointer.read_text())
        file = pointer.parent / p["file"]
        state = torch.load(file, weights_only=True)
        metadata = state["metadata"]
        for key in ("config_sha256", "implementation_source_sha256"):
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, key):
                recovery.load_resume(directory, {**metadata, key: "different"})
        with self.assertRaisesRegex(ValueError, "threads"):
            recovery.load_resume(directory, {**metadata, "runtime": {**metadata["runtime"], "threads": 99}})
        with file.open("ab") as stream:
            stream.write(b"corrupt")
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            self.train(self.root / "run", resume=True)

    def test_live_writer_and_completed_run_cannot_be_resumed(self):
        root = self.root / "run"
        self.train(root, max_updates=1)
        with recovery.training_run(root, "l1-construction-101", resume=True):
            with self.assertRaisesRegex(RuntimeError, "active writer"):
                self.train(root, resume=True)
        self.train(root, resume=True)
        with self.assertRaisesRegex(ValueError, "Completed"):
            self.train(root, resume=True)

    def test_l2_resume_preserves_frozen_partner_history(self):
        root = self.root / "l2"
        self.train(root)
        with patch("builtins.print"):
            full = pipeline.train(self.config, root, "l2", "cpu", run_id="full")
            pipeline.train(self.config, root, "l2", "cpu", run_id="partial", max_updates=1)
            resumed = pipeline.train(self.config, root, "l2", "cpu", run_id="partial", resume=True)
        self.assert_tree_equal(load_checkpoint(full, "cpu")[0], load_checkpoint(resumed, "cpu")[0])


if __name__ == "__main__":
    unittest.main()
