"""Behavioral checks for recurrent memory and actual PPO learning."""

from __future__ import annotations

from types import SimpleNamespace
import unittest

import torch
from torch.distributions import Categorical

from nestrl_research.model import RecurrentActorCritic
from nestrl_research.ppo import compute_gae, ppo_update


class AdvantageTests(unittest.TestCase):
    def test_episode_terminal_blocks_bootstrap_and_next_episode(self) -> None:
        advantages, returns = compute_gae(
            rewards=torch.tensor([[1.0], [2.0], [10.0]]),
            values=torch.tensor([[0.5], [0.25], [1.0]]),
            dones=torch.tensor([[False], [True], [False]]),
            bootstrap=torch.tensor([2.0]), gamma=0.9, gae_lambda=1.0,
        )
        torch.testing.assert_close(returns[:, 0], torch.tensor([2.8, 2.0, 11.8]))
        torch.testing.assert_close(advantages[:, 0], torch.tensor([2.3, 1.75, 10.8]))

    def test_finite_horizon_terminal_ignores_bootstrap(self) -> None:
        arguments = dict(
            rewards=torch.tensor([[3.0]]), values=torch.tensor([[2.0]]),
            bootstrap=torch.tensor([100.0]), gamma=0.99, gae_lambda=0.95,
        )
        terminal, _ = compute_gae(dones=torch.tensor([[True]]), **arguments)
        truncated, _ = compute_gae(dones=torch.tensor([[False]]), **arguments)
        self.assertAlmostEqual(terminal.item(), 1.0)
        self.assertAlmostEqual(truncated.item(), 100.0)

    def test_nonfinite_rollout_is_rejected(self) -> None:
        with self.assertRaises(FloatingPointError):
            compute_gae(torch.tensor([[float("nan")]]), torch.zeros(1, 1),
                        torch.zeros(1, 1, dtype=torch.bool), torch.zeros(1), 0.99, 0.95)


class LearningTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        torch.set_num_threads(1)

    def setUp(self) -> None:
        torch.manual_seed(77)
        self.model = RecurrentActorCritic(obs_dim=5, action_dim=3, hidden_dim=8, hidden_dims=(12, 12))

    def test_only_explicit_context_resets_clear_memory(self) -> None:
        observations = torch.randn(3, 2, 5)
        previous = torch.full((3, 2), -1, dtype=torch.long)
        starts = torch.ones(3, 2, dtype=torch.bool)
        resets = torch.tensor([[False, False], [True, True], [False, False]])
        zero = self.model.initial_state(2)
        carried = torch.ones_like(zero) * 2
        result_zero = self.model(observations, previous, starts, zero, resets)
        result_carried = self.model(observations, previous, starts, carried, resets)
        self.assertFalse(torch.allclose(result_zero["values"][0], result_carried["values"][0]))
        torch.testing.assert_close(result_zero["logits"][1:], result_carried["logits"][1:])
        torch.testing.assert_close(result_zero["hidden"], result_carried["hidden"])

    def test_context_reset_cuts_gradient_to_earlier_observations(self) -> None:
        observations = torch.randn(3, 1, 5, requires_grad=True)
        result = self.model(
            observations, torch.zeros(3, 1, dtype=torch.long),
            torch.zeros(3, 1, dtype=torch.bool), resets=torch.tensor([[False], [True], [False]]),
        )
        result["values"][-1].sum().backward()
        torch.testing.assert_close(observations.grad[0], torch.zeros_like(observations.grad[0]))
        self.assertGreater(observations.grad[1:].abs().sum().item(), 0)

    def test_single_step_act_matches_forward(self) -> None:
        observations = torch.randn(2, 5)
        previous = torch.tensor([-1, 2])
        starts = torch.tensor([True, False])
        hidden = self.model.initial_state(2)
        resets = torch.tensor([True, False])
        actions, log_probs, values, next_hidden = self.model.act(
            observations, previous, starts, hidden, resets, deterministic=True
        )
        expected = self.model(observations[None], previous[None], starts[None], hidden, resets[None])
        self.assertEqual(actions.shape, (2,))
        torch.testing.assert_close(actions, expected["logits"][0].argmax(-1))
        torch.testing.assert_close(log_probs, Categorical(logits=expected["logits"][0]).log_prob(actions))
        torch.testing.assert_close(values, expected["values"][0])
        torch.testing.assert_close(next_hidden, expected["hidden"])
        self.assertFalse(next_hidden.requires_grad)

    def _batch(self, time: int = 5, count: int = 2) -> dict[str, torch.Tensor]:
        observations = torch.randn(time, count, 5)
        previous = torch.randint(-1, 3, (time, count))
        starts = torch.zeros(time, count, dtype=torch.bool)
        starts[0] = True
        resets = starts.clone()
        hidden = self.model.initial_state(count)
        hidden_states, log_probs, values, actions = [], [], [], []
        for step in range(time):
            hidden_states.append(hidden[0].clone())
            action, log_prob, value, hidden = self.model.act(
                observations[step], previous[step], starts[step], hidden, resets[step]
            )
            actions.append(action)
            log_probs.append(log_prob)
            values.append(value)
        values_tensor = torch.stack(values)
        advantages = torch.linspace(-1, 2, time * count).view(time, count)
        return {
            "observations": observations, "previous_actions": previous,
            "episode_starts": starts, "resets": resets,
            "hidden_states": torch.stack(hidden_states), "actions": torch.stack(actions),
            "old_log_probs": torch.stack(log_probs), "old_values": values_tensor,
            "returns": values_tensor + advantages, "advantages": advantages,
        }

    def _config(self, **overrides: object) -> SimpleNamespace:
        values = dict(epochs=2, minibatches=3, bptt_steps=3, clip=0.2,
                      entropy=0.03, value_coef=0.5, max_grad_norm=15.0)
        values.update(overrides)
        return SimpleNamespace(**values)

    def test_ppo_updates_encoder_and_recurrent_parameters_with_uneven_sequences(self) -> None:
        batch = self._batch()
        encoder_before = self.model.encoder[0].weight.detach().clone()
        gru_before = self.model.gru.weight_hh_l0.detach().clone()
        actor_before = self.model.actor.weight.detach().clone()
        metrics = ppo_update(
            self.model, torch.optim.Adam(self.model.parameters(), lr=0.001), batch,
            self._config(), generator=torch.Generator().manual_seed(123),
        )
        self.assertFalse(torch.equal(encoder_before, self.model.encoder[0].weight))
        self.assertFalse(torch.equal(gru_before, self.model.gru.weight_hh_l0))
        self.assertFalse(torch.equal(actor_before, self.model.actor.weight))
        self.assertGreater(self.model.encoder[0].weight.grad.abs().sum().item(), 0)
        self.assertEqual(metrics["updates"], 6)
        self.assertEqual(metrics["segments"], 4)
        self.assertEqual(metrics["valid_steps"], 10)
        self.assertTrue(all(torch.isfinite(torch.tensor(value)) for value in metrics.values()))

    def test_segment_start_states_reproduce_collected_policy(self) -> None:
        batch = self._batch(time=7, count=3)
        metrics = ppo_update(
            self.model, torch.optim.Adam(self.model.parameters(), lr=0), batch,
            self._config(epochs=1, minibatches=4, bptt_steps=3),
            generator=torch.Generator().manual_seed(4),
        )
        self.assertAlmostEqual(metrics["approx_kl"], 0.0, places=7)
        self.assertEqual(metrics["clip_fraction"], 0.0)
        self.assertEqual(metrics["segments"], 9)
        self.assertEqual(metrics["valid_steps"], 21)

    def test_padding_and_invalid_transitions_do_not_affect_loss(self) -> None:
        batch = self._batch(time=3, count=1)
        batch["valid_mask"] = torch.tensor([[True], [True], [False]])
        batch["advantages"][-1] = 1e10
        batch["returns"][-1] = 1e10
        batch["hidden_states"] = batch["hidden_states"].unsqueeze(1)
        metrics = ppo_update(
            self.model, torch.optim.Adam(self.model.parameters(), lr=0.001),
            batch, self._config(epochs=1, minibatches=18, bptt_steps=2),
        )
        self.assertEqual(metrics["updates"], 1)
        self.assertEqual(metrics["valid_steps"], 2)
        self.assertLess(metrics["loss"], 100)

    def test_nonfinite_ppo_batch_fails_before_parameter_updates(self) -> None:
        batch = self._batch()
        batch["old_log_probs"][0, 0] = float("nan")
        before = {name: parameter.detach().clone() for name, parameter in self.model.named_parameters()}
        with self.assertRaises(FloatingPointError):
            ppo_update(self.model, torch.optim.Adam(self.model.parameters()), batch, self._config())
        for name, parameter in self.model.named_parameters():
            torch.testing.assert_close(parameter, before[name])


if __name__ == "__main__":
    unittest.main()
