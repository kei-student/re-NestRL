"""Integration checks for nesting, independent history, and frozen artifacts."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from nestrl_research import pipeline
from nestrl_research.artifacts import atomic_json, file_hash, load_checkpoint, save_checkpoint
from nestrl_research.config import EnvironmentConfig, ExperimentConfig, PPOConfig
from nestrl_research.evaluation import evaluate
from nestrl_research.ppo import compute_gae, ppo_update
from nestrl_research.provenance import source_metadata
from nestrl_research.rollout import FrozenPartners, RolloutCollector


def tiny_config() -> ExperimentConfig:
    return ExperimentConfig(
        profile="test",
        environment=EnvironmentConfig(horizon=2),
        ppo=PPOConfig(epochs=1, minibatches=2, bptt_steps=2),
        num_envs=2, rollout_steps=4, total_steps=8,
        evaluation_rounds=1,
        construction_seeds=(101,), validation_seeds=(201,), test_seeds=(301,),
    )


class WaitPeer:
    def __init__(self, recipe, agent_id, seed):
        self.agent_id = agent_id

    def act(self, environment):
        return 5


class StaggeredEpisodeEnv:
    """Alternate one- and two-step tasks to expose accidental global resets."""

    instances = []

    def __init__(self, config):
        self.index = len(self.instances)
        self.instances.append(self)
        self.horizon = self.index % 2 + 1
        self.joint_actions = []
        self.episode = -1

    def reset(self, seed=None):
        self.episode += 1
        self.t = 0
        return self._observe()

    def _observe(self):
        observation = np.zeros((2, 105), dtype=np.float32)
        observation[:, 0] = self.index + 1
        observation[:, 1] = np.arange(2)
        observation[:, 2] = self.t
        return observation

    def step(self, actions):
        self.joint_actions.append(tuple(actions))
        self.t += 1
        return self._observe(), 1.0, False, self.t >= self.horizon, {"success": False}


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        pipeline.seed_all(98)
        self.config = tiny_config()
        self.device = torch.device("cpu")

    def test_round_resets_are_per_environment_and_episode_memory_is_carried(self):
        config = replace(self.config, rollout_steps=12, total_steps=24)
        model = pipeline.make_model(config, self.device)
        StaggeredEpisodeEnv.instances = []
        with patch("nestrl_research.rollout.CookingEnv", StaggeredEpisodeEnv), \
             patch("nestrl_research.rollout.ScriptedPeer", WaitPeer):
            collector = RolloutCollector(config, model, ego_side=1, seed=42, device=self.device)
            batch, bootstrap, episodes = collector.collect()
        expected_resets = torch.zeros(12, 2, dtype=torch.bool)
        expected_resets[0] = True
        expected_resets[[5, 10], 0] = True
        expected_resets[10, 1] = True
        torch.testing.assert_close(batch["resets"], expected_resets)
        self.assertTrue(batch["episode_starts"][:, 0].all())
        torch.testing.assert_close(batch["episode_starts"][:, 1], torch.arange(12) % 2 == 0)
        self.assertGreater(batch["hidden_states"][1, 0].abs().sum().item(), 0)
        self.assertFalse(batch["resets"][1, 0])
        replay = model(batch["observations"], batch["previous_actions"], batch["episode_starts"],
                       model.initial_state(2), batch["resets"])
        torch.testing.assert_close(replay["values"], batch["old_values"])
        torch.testing.assert_close(replay["hidden"], collector.hidden)
        self.assertEqual(bootstrap.shape, (2,))
        self.assertEqual(len(episodes), 18)
        self.assertTrue(all(joint[0] == 5 for env in collector.envs for joint in env.joint_actions))
        self.assertTrue(all(peer.agent_id == 0 for peer in collector.partners.scripts))

    def test_learned_pool_larger_than_recipes_is_frozen_and_keeps_separate_states(self):
        models = [pipeline.make_model(self.config, self.device) for _ in range(4)]
        with patch("nestrl_research.rollout.ScriptedPeer", side_effect=AssertionError("Unexpected scripted peer")):
            partners = FrozenPartners(self.config, side=1, device=self.device,
                                      rng=np.random.default_rng(0), models=models)
            partners.indices[:] = 0
            partners.resets[:] = False
            partners.hidden[:, 1] = 2
            actions = partners.act(np.zeros((2, 105), dtype=np.float32), [], deterministic=True)
            self.assertEqual(actions.shape, (2,))
            self.assertFalse(torch.allclose(partners.hidden[:, 0], partners.hidden[:, 1]))
            before = partners.hidden.clone()
            partners.advance(actions, [True, False], [False, False])
            torch.testing.assert_close(partners.hidden, before)
            self.assertEqual(partners.previous[0].item(), -1)
            self.assertFalse(partners.resets.any())
            partners.advance(actions, [True, True], [True, False])
        for model in models:
            self.assertFalse(model.training)
            self.assertTrue(all(not parameter.requires_grad for parameter in model.parameters()))
            self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))

    def test_real_environment_rollout_and_update_preserve_frozen_partner_weights(self):
        model = pipeline.make_model(self.config, self.device)
        partner = pipeline.make_model(self.config, self.device)
        before = {name: parameter.detach().clone() for name, parameter in partner.named_parameters()}
        collector = RolloutCollector(self.config, model, ego_side=0, seed=3,
                                     device=self.device, partner_models=[partner])
        batch, bootstrap, _ = collector.collect()
        batch["advantages"], batch["returns"] = compute_gae(
            batch["rewards"], batch["old_values"], batch["dones"], bootstrap,
            self.config.ppo.gamma, self.config.ppo.gae_lambda,
        )
        metrics = ppo_update(model, torch.optim.Adam(model.parameters(), lr=0.001), batch, self.config.ppo)
        self.assertEqual(metrics["valid_steps"], 8)
        self.assertEqual(collector.partners.side, 1)
        for name, parameter in partner.named_parameters():
            torch.testing.assert_close(parameter, before[name])
            self.assertIsNone(parameter.grad)
            self.assertFalse(parameter.requires_grad)

    def test_l2_training_only_uses_frozen_construction_and_opposite_roles(self):
        with tempfile.TemporaryDirectory() as temporary, patch("builtins.print"):
            root = Path(temporary)
            for split, seed in (("construction", 101), ("validation", 201), ("test", 301)):
                pipeline.train(self.config, root, "l1", self.device, split=split, seed=seed)
            checkpoint = pipeline.train(self.config, root, "l2", self.device)
            _, manifest = load_checkpoint(checkpoint, self.device, self.config.sha256)
            self.assertEqual(manifest["ego_side"], 0)
            self.assertEqual(manifest["stage"], "l2")
            self.assertEqual([item["split"] for item in manifest["training_partners"]], ["construction"])
            self.assertEqual([item["id"] for item in manifest["training_partners"]], ["l1-construction-101"])
            for item in manifest["training_partners"]:
                partner = root / item["checkpoint"]
                self.assertEqual(item["checkpoint_sha256"], file_hash(partner))
                self.assertEqual(item["manifest_sha256"], file_hash(partner.with_suffix(".json")))
                _, partner_manifest = load_checkpoint(partner, self.device, self.config.sha256)
                self.assertEqual(partner_manifest["ego_side"], 1)
                self.assertTrue(all(peer["id"].startswith("l0-0-") for peer in partner_manifest["training_partners"]))
            generalist = pipeline.train(self.config, root, "generalist", self.device)
            _, generalist_manifest = load_checkpoint(generalist, self.device, self.config.sha256)
            self.assertEqual(generalist_manifest["ego_side"], 0)
            self.assertTrue(all(peer["id"].startswith("l0-1-") for peer in generalist_manifest["training_partners"]))
            checkpoint_before = file_hash(checkpoint)
            summary = evaluate(self.config, root, checkpoint, self.device, split="test", run_id="held-out")
            self.assertEqual(summary["episodes"], self.config.episodes_per_round)
            self.assertEqual(summary["protocol"]["partition"], "test")
            self.assertFalse(summary["protocol"]["weight_updates"])
            self.assertEqual([item["id"] for item in summary["protocol"]["partners"]], ["l1-test-301"])
            self.assertEqual(checkpoint_before, file_hash(checkpoint))
            with self.assertRaisesRegex(ValueError, "construction"):
                pipeline.train(self.config, root, "l2", self.device, split="test", run_id="leak")
            self.assertFalse((root / "leak").exists())
            with self.assertRaisesRegex(ValueError, "frozen partition"):
                pipeline.train(self.config, root, "l1", self.device, split="construction", seed=301)

    def test_registry_is_immutable_and_checkpoint_hashes_and_labels_are_verified(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            original = pipeline.ensure_registry(self.config, root)
            self.assertEqual(original, pipeline.ensure_registry(self.config, root))
            with self.assertRaisesRegex(ValueError, "Frozen manifest differs"):
                pipeline.ensure_registry(replace(self.config, construction_seeds=(102,)), root)
            model = pipeline.make_model(self.config, self.device)
            directory = root / "l1-construction-101"
            directory.mkdir()
            checkpoint = save_checkpoint(directory, model, torch.optim.Adam(model.parameters()), {
                **source_metadata(),
                "config_sha256": self.config.sha256, "stage": "l1", "split": "test", "seed": 101,
                "ego_side": 1, "run_id": directory.name,
            })
            # Valid file hashes alone do not authorize a policy from the wrong split.
            with self.assertRaisesRegex(ValueError, "frozen registry"):
                pipeline.load_l1_pool(self.config, root, "construction", self.device)
            manifest_path = checkpoint.with_suffix(".json")
            manifest = json.loads(manifest_path.read_text())
            manifest["split"] = "construction"
            atomic_json(manifest_path, manifest)
            with self.assertRaisesRegex(ValueError, "latest.json"):
                pipeline.load_l1_pool(self.config, root, "construction", self.device)
            atomic_json(checkpoint.parent / "latest.json", {
                "manifest": "final.json", "manifest_sha256": file_hash(manifest_path),
            })
            models, provenance = pipeline.load_l1_pool(self.config, root, "construction", self.device)
            self.assertEqual(len(models), 1)
            self.assertEqual(provenance[0]["split"], "construction")
            with checkpoint.open("ab") as stream:
                stream.write(b"corruption")
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                pipeline.load_l1_pool(self.config, root, "construction", self.device)


if __name__ == "__main__":
    unittest.main()
