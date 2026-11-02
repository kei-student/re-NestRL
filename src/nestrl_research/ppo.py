"""Generalized advantage estimation and truncated recurrent PPO.

Each sampled sequence starts from its collected, detached hidden state.
Sequence padding contributes neither losses nor advantage normalization.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor
from torch.distributions import Categorical

from .model import RecurrentActorCritic


def _require_finite(name: str, tensor: Tensor) -> None:
    if not torch.isfinite(tensor).all():
        raise FloatingPointError(f"Non-finite {name}")


@torch.no_grad()
def compute_gae(
    rewards: Tensor,
    values: Tensor,
    dones: Tensor,
    bootstrap: Tensor,
    gamma: float,
    gae_lambda: float,
) -> tuple[Tensor, Tensor]:
    """Return advantages and value targets for a time-major rollout.

    A true ``dones[t,b]`` denotes a finite-horizon episode terminal: neither
    its bootstrap value nor the next episode's advantage enters the target.
    A rollout boundary alone is not terminal and uses ``bootstrap[b]``.
    """
    if rewards.ndim != 2 or not rewards.numel():
        raise ValueError("Expected nonempty rewards [T,B]")
    if values.shape != rewards.shape or dones.shape != rewards.shape:
        raise ValueError("Rewards, values and dones must have identical [T,B] shapes")
    if bootstrap.shape != rewards.shape[1:]:
        raise ValueError("Expected bootstrap [B]")
    if not rewards.is_floating_point() or not values.is_floating_point() or not bootstrap.is_floating_point():
        raise ValueError("Rewards, values and bootstrap must be floating-point tensors")
    if values.device != rewards.device or dones.device != rewards.device or bootstrap.device != rewards.device:
        raise ValueError("All GAE tensors must share a device")
    if not 0 <= gamma <= 1 or not 0 <= gae_lambda <= 1:
        raise ValueError("gamma and gae_lambda must be in [0,1]")
    for name, value in (("rewards", rewards), ("values", values), ("bootstrap", bootstrap)):
        _require_finite(name, value)
    advantages = torch.empty_like(rewards)
    next_advantage = torch.zeros_like(bootstrap)
    next_value = bootstrap
    for step in reversed(range(rewards.shape[0])):
        continues = (~dones[step].bool()).to(rewards)
        delta = rewards[step] + gamma * continues * next_value - values[step]
        next_advantage = delta + gamma * gae_lambda * continues * next_advantage
        advantages[step] = next_advantage
        next_value = values[step]
    return advantages, advantages + values


def _prepare_batch(
    model: RecurrentActorCritic, batch: dict[str, Tensor]
) -> tuple[dict[str, Tensor], Tensor, Tensor]:
    required = (
        "observations", "previous_actions", "episode_starts", "resets", "hidden_states",
        "actions", "old_log_probs", "old_values", "returns", "advantages",
    )
    missing = [name for name in required if name not in batch]
    if missing:
        raise ValueError(f"Missing PPO batch fields: {', '.join(missing)}")
    observations = batch["observations"]
    if observations.ndim != 3 or observations.shape[-1] != model.obs_dim:
        raise ValueError("Expected observations [T,B,D]")
    time, environments, _ = observations.shape
    if time <= 0 or environments <= 0:
        raise ValueError("PPO batch must be nonempty")
    for name in required:
        value = batch[name]
        if value.device != observations.device:
            raise ValueError(f"PPO batch device mismatch: {name}")
        if name not in ("observations", "hidden_states") and value.shape != (time, environments):
            raise ValueError(f"Expected {name} [T,B]")
        if value.is_floating_point():
            _require_finite(name, value)
    hidden_states = batch["hidden_states"]
    if hidden_states.ndim == 4 and hidden_states.shape[1] == 1:
        hidden_states = hidden_states[:, 0]
    if hidden_states.shape != (time, environments, model.hidden_dim):
        raise ValueError("Expected hidden_states [T,B,H] or [T,1,B,H]")
    valid = batch.get("valid_mask", torch.ones_like(batch["actions"], dtype=torch.bool)).bool()
    if valid.shape != (time, environments) or valid.device != observations.device:
        raise ValueError("Expected valid_mask [T,B] on the rollout device")
    if not valid.any():
        raise ValueError("PPO batch has no valid transitions")
    for name in ("actions", "previous_actions"):
        if batch[name].dtype not in (torch.int32, torch.int64):
            raise ValueError(f"{name} must contain integer indices")
    if torch.any((batch["actions"] < 0) | (batch["actions"] >= model.action_dim)):
        raise ValueError("Actions are outside the policy action space")
    clean_batch = {name: value.detach() for name, value in batch.items()}
    advantages = clean_batch["advantages"].clone()
    selected = advantages[valid]
    advantages[valid] = (selected - selected.mean()) / (selected.std(unbiased=False) + 1e-8)
    clean_batch["advantages"] = advantages
    return clean_batch, hidden_states.detach(), valid


def _sequence_minibatch(
    batch: dict[str, Tensor],
    hidden_states: Tensor,
    valid: Tensor,
    segments: list[tuple[int, int, int]],
) -> tuple[dict[str, Tensor], Tensor, Tensor]:
    """Pad selected (environment, start, stop) sequences to equal length."""
    length = max(stop - start for _, start, stop in segments)
    count = len(segments)
    names = (
        "observations", "previous_actions", "episode_starts", "resets", "actions",
        "old_log_probs", "old_values", "returns", "advantages",
    )
    padded = {}
    for name in names:
        original = batch[name]
        shape = (length, count) + original.shape[2:]
        padded[name] = torch.full(
            shape, -1 if name == "previous_actions" else 0,
            dtype=original.dtype, device=original.device,
        )
    mask = torch.zeros(length, count, dtype=torch.bool, device=valid.device)
    initial_states = []
    for column, (environment, start, stop) in enumerate(segments):
        size = stop - start
        for name in names:
            padded[name][:size, column] = batch[name][start:stop, environment]
        mask[:size, column] = valid[start:stop, environment]
        initial_states.append(hidden_states[start, environment])
    return padded, torch.stack(initial_states, dim=0).unsqueeze(0).detach(), mask


def ppo_update(
    model: RecurrentActorCritic,
    optimizer: torch.optim.Optimizer,
    batch: dict[str, Tensor],
    config: Any,
    generator: torch.Generator | None = None,
) -> dict[str, float]:
    """Optimize whole recurrent sequences using shuffled truncated BPTT.

    ``config`` supplies epochs, minibatches, bptt_steps, clip, entropy,
    value_coef and max_grad_norm. Optional clip_value_loss defaults to True.
    A CPU generator makes sequence order reproducible across device choices.
    """
    if min(config.epochs, config.minibatches, config.bptt_steps) <= 0:
        raise ValueError("PPO epochs, minibatches and BPTT length must be positive")
    if config.clip <= 0 or config.entropy < 0 or config.value_coef < 0 or config.max_grad_norm <= 0:
        raise ValueError("Invalid PPO loss or gradient clipping configuration")
    batch, hidden_states, valid = _prepare_batch(model, batch)
    time, environments = valid.shape
    segments = [
        (environment, start, min(start + config.bptt_steps, time))
        for environment in range(environments)
        for start in range(0, time, config.bptt_steps)
        if valid[start : start + config.bptt_steps, environment].any()
    ]
    minibatches = min(config.minibatches, len(segments))
    totals = {
        "loss": 0.0, "policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0,
        "approx_kl": 0.0, "clip_fraction": 0.0, "grad_norm": 0.0,
    }
    weighted_steps = 0
    updates = 0
    model.train()
    for _ in range(config.epochs):
        permutation = torch.randperm(len(segments), generator=generator, device="cpu")
        for indices in torch.tensor_split(permutation, minibatches):
            selected = [segments[index] for index in indices.tolist()]
            samples, initial_hidden, mask = _sequence_minibatch(batch, hidden_states, valid, selected)
            result = model(
                samples["observations"], samples["previous_actions"],
                samples["episode_starts"], initial_hidden, samples["resets"],
            )
            distribution = Categorical(logits=result["logits"])
            new_log_probs = distribution.log_prob(samples["actions"])[mask]
            log_ratio = new_log_probs - samples["old_log_probs"][mask]
            ratio = log_ratio.exp()
            advantages = samples["advantages"][mask]
            surrogate = torch.minimum(
                ratio * advantages,
                ratio.clamp(1 - config.clip, 1 + config.clip) * advantages,
            )
            policy_loss = -surrogate.mean()
            values = result["values"][mask]
            targets = samples["returns"][mask]
            value_errors = (values - targets).square()
            if getattr(config, "clip_value_loss", True):
                old_values = samples["old_values"][mask]
                clipped_values = old_values + (values - old_values).clamp(-config.clip, config.clip)
                value_errors = torch.maximum(value_errors, (clipped_values - targets).square())
            value_loss = 0.5 * value_errors.mean()
            entropy = distribution.entropy()[mask].mean()
            loss = policy_loss + config.value_coef * value_loss - config.entropy * entropy
            approx_kl = ((ratio - 1) - log_ratio).mean()
            clip_fraction = ((ratio - 1).abs() > config.clip).float().mean()
            for name, value in (
                ("PPO loss", loss), ("PPO KL", approx_kl), ("PPO entropy", entropy),
            ):
                _require_finite(name, value)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), config.max_grad_norm, error_if_nonfinite=True
            )
            optimizer.step()
            count = int(mask.sum().item())
            metrics = {
                "loss": loss, "policy_loss": policy_loss, "value_loss": value_loss,
                "entropy": entropy, "approx_kl": approx_kl,
                "clip_fraction": clip_fraction, "grad_norm": grad_norm,
            }
            for name, value in metrics.items():
                totals[name] += float(value.detach().item()) * count
            weighted_steps += count
            updates += 1
    return {
        **{name: value / weighted_steps for name, value in totals.items()},
        "updates": float(updates),
        "segments": float(len(segments)),
        "valid_steps": float(valid.sum().item()),
    }
