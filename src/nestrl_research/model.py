"""Recurrent actor--critic used for the fresh NestRL reproduction.

Episode boundaries are visible to the policy, but do not clear its memory.
Only an explicit context reset (a new peer interaction round) clears the GRU.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn
from torch.distributions import Categorical
from torch.nn import functional as F


class RecurrentActorCritic(nn.Module):
    """Shared MLP and GRU with independent policy and value heads.

    Previous-action one-hot vectors and episode-start flags are explicit
    reproduction choices; the paper does not fully specify the GRU inputs.
    A previous action of -1 encodes the absence of a previous action.
    """

    def __init__(
        self,
        obs_dim: int = 105,
        action_dim: int = 6,
        hidden_dim: int = 128,
        hidden_dims: Sequence[int] = (128, 128),
    ) -> None:
        super().__init__()
        if obs_dim <= 0 or action_dim <= 0 or hidden_dim <= 0:
            raise ValueError("Model dimensions must be positive")
        if not hidden_dims or any(size <= 0 for size in hidden_dims):
            raise ValueError("The encoder requires positive hidden dimensions")
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.hidden_dims = tuple(hidden_dims)
        layers: list[nn.Module] = []
        input_dim = obs_dim + action_dim + 1
        for output_dim in self.hidden_dims:
            layers.extend((nn.Linear(input_dim, output_dim), nn.ReLU()))
            input_dim = output_dim
        self.encoder = nn.Sequential(*layers)
        self.gru = nn.GRU(input_dim, hidden_dim)
        self.actor = nn.Linear(hidden_dim, action_dim)
        self.critic = nn.Linear(hidden_dim, 1)
        self._initialize_parameters()

    def _initialize_parameters(self) -> None:
        for layer in self.encoder:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=2**0.5)
                nn.init.zeros_(layer.bias)
        for name, parameter in self.gru.named_parameters():
            if "weight" in name:
                for gate in parameter.chunk(3, dim=0):
                    nn.init.orthogonal_(gate)
            else:
                nn.init.zeros_(parameter)
        nn.init.orthogonal_(self.actor.weight, gain=0.01)
        nn.init.zeros_(self.actor.bias)
        nn.init.orthogonal_(self.critic.weight, gain=1.0)
        nn.init.zeros_(self.critic.bias)

    def initial_state(
        self, batch: int, device: torch.device | str | None = None
    ) -> Tensor:
        if batch <= 0:
            raise ValueError("State batch size must be positive")
        parameter = next(self.parameters())
        return torch.zeros(
            1, batch, self.hidden_dim,
            device=parameter.device if device is None else device,
            dtype=parameter.dtype,
        )

    def forward(
        self,
        observations: Tensor,
        previous_actions: Tensor,
        episode_starts: Tensor,
        hidden: Tensor | None = None,
        resets: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """Process time-major sequences and return the final recurrent state."""
        if observations.ndim != 3 or observations.shape[-1] != self.obs_dim:
            raise ValueError(f"Expected observations [T,B,{self.obs_dim}]")
        time, batch, _ = observations.shape
        if time <= 0 or batch <= 0:
            raise ValueError("Sequences and batches must be nonempty")
        if previous_actions.shape != (time, batch):
            raise ValueError("Expected previous_actions [T,B]")
        if previous_actions.dtype not in (torch.int32, torch.int64):
            raise ValueError("Previous actions must be integer indices")
        if torch.any((previous_actions < -1) | (previous_actions >= self.action_dim)):
            raise ValueError("Previous actions must be -1 or valid action indices")
        if episode_starts.shape != (time, batch):
            raise ValueError("Expected episode_starts [T,B]")
        if resets is None:
            resets = torch.zeros_like(episode_starts, dtype=torch.bool)
        if resets.shape != (time, batch):
            raise ValueError("Expected resets [T,B]")
        if hidden is None:
            hidden = self.initial_state(batch, observations.device)
        if hidden.shape != (1, batch, self.hidden_dim):
            raise ValueError(f"Expected hidden state [1,B,{self.hidden_dim}]")
        action_one_hot = F.one_hot(
            previous_actions.clamp(min=0).long(), num_classes=self.action_dim
        ).to(dtype=observations.dtype)
        action_one_hot = action_one_hot * (previous_actions >= 0).unsqueeze(-1)
        inputs = torch.cat(
            (observations, action_one_hot, episode_starts.unsqueeze(-1).to(observations)),
            dim=-1,
        )
        encoded = self.encoder(inputs)
        outputs = []
        for step in range(time):
            hidden = hidden * (~resets[step].bool()).to(hidden).view(1, batch, 1)
            output, hidden = self.gru(encoded[step : step + 1], hidden)
            outputs.append(output)
        features = torch.cat(outputs, dim=0)
        return {
            "logits": self.actor(features),
            "values": self.critic(features).squeeze(-1),
            "hidden": hidden,
        }

    @torch.no_grad()
    def act(
        self,
        observations: Tensor,
        previous_actions: Tensor,
        episode_starts: Tensor,
        hidden: Tensor | None = None,
        resets: Tensor | None = None,
        deterministic: bool = False,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Select one action per batch item without recording gradients."""
        if observations.ndim != 2:
            raise ValueError("Expected single-step observations [B,D]")
        result = self(
            observations.unsqueeze(0),
            previous_actions.unsqueeze(0),
            episode_starts.unsqueeze(0),
            hidden,
            None if resets is None else resets.unsqueeze(0),
        )
        distribution = Categorical(logits=result["logits"][0])
        actions = distribution.logits.argmax(dim=-1) if deterministic else distribution.sample()
        return actions, distribution.log_prob(actions), result["values"][0], result["hidden"]
