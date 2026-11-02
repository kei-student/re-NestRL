"""Check round memory, evaluation denominators, and immutable inference artifacts."""

from __future__ import annotations

from dataclasses import replace
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from nestrl_research.artifacts import file_hash
from nestrl_research.config import EnvironmentConfig, ExperimentConfig, RECIPES
from nestrl_research.evaluation import PolicyContext, evaluate, summarize


def episode_records(partner: str, round_index: int, outcomes: list[bool]) -> list[dict]:
    return [{"partner_id": partner, "round": round_index, "episode_in_round": position,
             "success": success, "recipe": RECIPES[0] if success else None,
             "return": 10.0 if success else 0.0}
            for position, success in enumerate(outcomes)]


class SummaryTests(unittest.TestCase):
    def test_macro_and_pooled_rates_preserve_different_partner_denominators(self):
        episodes = (episode_records("a", 0, [True] * 5)
                    + episode_records("b", 0, [False] * 5)
                    + episode_records("b", 1, [False] * 5))
        summary = summarize(episodes, 5)
        self.assertEqual(summary["episodes"], 15)
        self.assertEqual(summary["successes"], 5)
        self.assertEqual(summary["rounds"], 3)
        self.assertEqual(summary["success_rate"], 0.5)
        self.assertAlmostEqual(summary["pooled_episode_success_rate"], 1 / 3)
        self.assertAlmostEqual(summary["all_episodes_round_success_rate"], 1 / 3)
        self.assertEqual(summary["per_partner"], {"a": 1.0, "b": 0.0})
        self.assertEqual(summary["per_episode_position"], [1 / 3] * 5)
        self.assertEqual(summary["partner_rate_std_population"], 0.5)
        self.assertEqual(summary["partner_rate_cv"], 1.0)
        self.assertEqual(summary["per_partner_counts"]["b"]["episodes"], 10)
        self.assertEqual(summary["per_partner_counts"]["a"]["successes"], 5)
        self.assertAlmostEqual(summary["per_partner_counts"]["a"]["wilson95"][0], 0.5655175352)
        self.assertAlmostEqual(summary["per_partner_counts"]["b"]["wilson95"][1], 0.2775327999)

    def test_round_success_requires_all_episodes_to_succeed(self):
        records = episode_records("a", 0, [True, True, True, True, False])
        self.assertEqual(summarize(records, 5)["all_episodes_round_success_rate"], 0.0)

    def test_zero_success_cv_is_none_and_summary_is_json_serializable(self):
        records = episode_records("a", 0, [False] * 5) + episode_records("b", 0, [False] * 5)
        summary = summarize(records, 5)
        self.assertIsNone(summary["partner_rate_cv"])
        self.assertEqual(summary["success_rate"], 0.0)
        self.assertEqual(summary["pooled_episode_success_rate"], 0.0)
        self.assertEqual(summary["recipe_counts"], dict.fromkeys(RECIPES, 0))
        self.assertIsNone(summary["successful_recipe_entropy_nats"])
        self.assertIsNone(summary["successful_recipe_entropy_normalized"])
        json.dumps(summary, allow_nan=False)

    def test_recipe_entropy_counts_successful_deliveries_with_fixed_three_recipe_support(self):
        outcomes = [True, True, True, False, False]
        records = episode_records("a", 0, outcomes)
        for index, recipe in enumerate(RECIPES):
            records[index]["recipe"] = recipe
        records[-1]["recipe"] = RECIPES[0]  # A failed attempt contributes no completed recipe.
        summary = summarize(records, 5)
        self.assertEqual(summary["recipe_counts"], dict.fromkeys(RECIPES, 1))
        self.assertAlmostEqual(summary["successful_recipe_entropy_nats"], math.log(3))
        self.assertAlmostEqual(summary["successful_recipe_entropy_normalized"], 1.0)
        self.assertNotIn("interpretation", summary)
        collapsed = summarize(episode_records("a", 0, [True] * 5), 5)
        self.assertEqual(collapsed["recipe_counts"], {RECIPES[0]: 5, RECIPES[1]: 0, RECIPES[2]: 0})
        self.assertEqual(collapsed["successful_recipe_entropy_nats"], 0.0)
        self.assertEqual(collapsed["successful_recipe_entropy_normalized"], 0.0)

    def test_success_without_a_permitted_completed_recipe_is_rejected(self):
        complete = episode_records("a", 0, [True] * 5)
        for recipe in (None, "TomatoOnionSalad"):
            with self.subTest(recipe=recipe), self.assertRaises(ValueError):
                summarize([{**complete[0], "recipe": recipe}, *complete[1:]], 5)

    def test_empty_missing_duplicate_or_out_of_range_episodes_are_rejected(self):
        complete = episode_records("a", 0, [True] * 5)
        malformed = [[], complete[:-1], complete + [complete[0]],
                     [*complete[:-1], {**complete[-1], "episode_in_round": 5}]]
        for records in malformed:
            with self.subTest(records=records), self.assertRaises(ValueError):
                summarize(records, 5)
        for size in (0, -1, True, 2.5):
            with self.subTest(size=size), self.assertRaises(ValueError):
                summarize(complete, size)

    def test_nonfinite_returns_and_non_boolean_success_are_rejected(self):
        complete = episode_records("a", 0, [True] * 5)
        for field, value in (("return", float("nan")), ("return", float("inf")), ("success", 10)):
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                summarize([{**complete[0], field: value}, *complete[1:]], 5)


class RecordingPolicy(torch.nn.Module):
    """A history counter makes accidental episode resets observable."""

    def __init__(self):
        super().__init__()
        self.increment = torch.nn.Parameter(torch.tensor(1.0))
        self.calls = []

    def initial_state(self, batch, device=None):
        return torch.zeros(1, batch, 1, device=device)

    def act(self, observations, previous_actions, starts, hidden, resets, deterministic=False):
        self.calls.append({"grad_enabled": torch.is_grad_enabled(), "previous": previous_actions.clone(),
                           "starts": starts.clone(), "resets": resets.clone(), "deterministic": deterministic})
        next_hidden = hidden * (~resets).view(1, -1, 1) + self.increment
        actions = torch.zeros(observations.shape[0], dtype=torch.long, device=observations.device)
        scalar = next_hidden[0, :, 0]
        return actions, scalar * 0, scalar, next_hidden


class PolicyContextTests(unittest.TestCase):
    def test_memory_spans_five_episodes_then_round_reset_clears_it(self):
        model = RecordingPolicy()
        context = PolicyContext(model, torch.device("cpu"), deterministic=True)
        observation = np.zeros(105, dtype=np.float32)
        for episode in range(5):
            for _ in range(2):
                context.act(observation)
            self.assertEqual(context.hidden.item(), 2 * (episode + 1))
            context.episode_boundary()
            self.assertTrue(context.starts.item())
            self.assertFalse(context.resets.item())
            self.assertEqual(context.previous.item(), -1)
        self.assertFalse(context.hidden.requires_grad)
        self.assertFalse(any(call["grad_enabled"] for call in model.calls))
        self.assertEqual([call["starts"].item() for call in model.calls], [True, False] * 5)
        self.assertEqual([call["resets"].item() for call in model.calls], [True] + [False] * 9)
        self.assertTrue(all(call["deterministic"] for call in model.calls))
        context.reset()
        self.assertEqual(context.hidden.item(), 0)
        context.act(observation)
        self.assertEqual(context.hidden.item(), 1)
        self.assertIsNone(model.increment.grad)


class FakeEnvironment:
    def __init__(self, config):
        self.horizon = config["horizon"]

    def reset(self, seed):
        self.seed, self.step_index = seed, 0
        return np.zeros((2, 105), dtype=np.float32)

    def step(self, actions):
        self.step_index += 1
        done = self.step_index == self.horizon
        success = done and self.seed % 2 == 0
        return (np.zeros((2, 105), dtype=np.float32), 1.0, success, done and not success,
                {"success": success, "recipe": "LettuceOnionSalad" if success else None,
                 "events": [{"type": "fixture_step", "actions": list(actions)}]})


class EvaluationArtifactTests(unittest.TestCase):
    def test_test_split_inference_writes_complete_hash_verified_traces_without_updates(self):
        config = replace(ExperimentConfig(), profile="smoke", environment=EnvironmentConfig(horizon=2),
                         evaluation_rounds=2, num_envs=2, rollout_steps=40, total_steps=160,
                         test_seeds=(31, 32))
        ego = RecordingPolicy().eval()
        partners = [RecordingPolicy().eval(), RecordingPolicy().eval()]
        provenance = [{"id": "l1-test-31", "split": "test"}, {"id": "l1-test-32", "split": "test"}]
        before = [model.increment.detach().clone() for model in [ego, *partners]]
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            checkpoint = output / "learner.pt"
            checkpoint.write_bytes(b"immutable checkpoint fixture")
            with (patch("nestrl_research.evaluation.load_policy", return_value=(ego, {"ego_side": 0, "run_id": "learner"})),
                  patch("nestrl_research.evaluation.load_l1_pool", return_value=(partners, provenance)) as pool,
                  patch("nestrl_research.evaluation.CookingEnv", FakeEnvironment)):
                summary = evaluate(config, output, checkpoint, torch.device("cpu"), run_id="evaluation", deterministic=True)
            pool.assert_called_once_with(config, output, "test", torch.device("cpu"))
            self.assertEqual(summary["episodes"], 20)
            self.assertEqual(summary["rounds"], 4)
            self.assertEqual(summary["successes"], 8)
            self.assertEqual(summary["recipe_counts"][RECIPES[0]], 8)
            self.assertIn("Smoke validates", summary["interpretation"])
            directory = output / "evaluation"
            protocol = json.loads((directory / "protocol.json").read_text())
            self.assertEqual(protocol["partition"], "test")
            self.assertEqual(protocol["config_sha256"], config.sha256)
            self.assertEqual(protocol["learner_checkpoint_sha256"], file_hash(checkpoint))
            self.assertFalse(protocol["weight_updates"])
            self.assertTrue(protocol["deterministic_actions"])
            records = json.loads((directory / "episodes.json").read_text())
            traces = [json.loads(line) for line in (directory / "traces.jsonl").read_text().splitlines()]
            self.assertEqual(len(records), 20)
            self.assertEqual(len(traces), 40)
            for trace in traces:
                self.assertEqual(trace["actions"], [0, 0])
                self.assertEqual(trace["events"][0]["type"], "fixture_step")
                self.assertNotIn("first_commit", trace)
            manifest = json.loads((directory / "artifacts.json").read_text())
            for filename, expected in manifest["files"].items():
                self.assertEqual(expected["sha256"], file_hash(directory / filename))
            self.assertFalse((directory / ".writer.lock").exists())
        for model, value in zip([ego, *partners], before):
            torch.testing.assert_close(model.increment, value)
            self.assertIsNone(model.increment.grad)
            self.assertTrue(model.calls)
            self.assertFalse(any(call["grad_enabled"] for call in model.calls))
        self.assertEqual(sum(call["resets"].item() for call in ego.calls), 4)

    def test_paper_profile_interpretation_does_not_describe_a_smoke_run(self):
        config = replace(ExperimentConfig(), environment=EnvironmentConfig(horizon=1),
                         evaluation_rounds=1, test_seeds=(31,))
        ego, partner = RecordingPolicy().eval(), RecordingPolicy().eval()
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            checkpoint = output / "learner.pt"
            checkpoint.write_bytes(b"inference fixture")
            with (patch("nestrl_research.evaluation.load_policy", return_value=(ego, {"ego_side": 0, "run_id": "learner"})),
                  patch("nestrl_research.evaluation.load_l1_pool", return_value=([partner], [{"id": "test-partner"}])),
                  patch("nestrl_research.evaluation.CookingEnv", FakeEnvironment)):
                summary = evaluate(config, output, checkpoint, torch.device("cpu"), run_id="evaluation")
        self.assertNotIn("Smoke", summary["interpretation"])
        self.assertIn("reconstruction", summary["interpretation"])

    def test_construction_split_cannot_be_evaluated_as_held_out(self):
        with self.assertRaises(ValueError):
            evaluate(ExperimentConfig(), Path("unused"), Path("unused.pt"), torch.device("cpu"), split="construction")


if __name__ == "__main__":
    unittest.main()
