"""Strict, hashable experiment contracts. All non-paper choices are explicit."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RECIPES = ("LettuceOnionSalad", "TomatoCarrotSalad", "PotatoBroccoliSalad")
UPSTREAM_COMMIT = "f008c19aa743ec209550375c9ee9c25379ac8b24"
SCHEMA = "re-nestrl-v1"


def canonical_hash(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


@dataclass(frozen=True)
class EnvironmentConfig:
    recipes: tuple[str, ...] = RECIPES
    horizon: int = 40
    interact_reward: float = 0.5
    progress_reward: float = 1.0
    complete_reward: float = 10.0
    step_cost: float = 0.0
    observation_schema: str = "pace_dense_partial_93_plus_reserved12_v1"


@dataclass(frozen=True)
class PPOConfig:
    lr: float = 1e-3
    clip: float = 0.2
    entropy: float = 0.03
    gamma: float = 0.99
    gae_lambda: float = 0.95
    epochs: int = 4
    minibatches: int = 18
    max_grad_norm: float = 15.0
    value_coef: float = 0.5
    clip_value_loss: bool = True
    bptt_steps: int = 20


@dataclass(frozen=True)
class ExperimentConfig:
    schema: str = SCHEMA
    profile: str = "paper"
    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    hidden_dim: int = 128
    hidden_dims: tuple[int, ...] = (128, 128)
    num_envs: int = 360
    rollout_steps: int = 200
    total_steps: int = 10_080_000
    episodes_per_round: int = 5
    evaluation_rounds: int = 10
    construction_seeds: tuple[int, ...] = (101, 102, 103)
    validation_seeds: tuple[int, ...] = (201,)
    test_seeds: tuple[int, ...] = (301, 302, 303, 304, 305)
    learner_seed: int = 401
    evaluation_seed: int = 501
    l1_ego_side: int = 1
    ego_side: int = 0
    context_reset: str = "round"
    previous_action_input: bool = True
    episode_boundary_input: bool = True
    finite_horizon_returns: bool = True

    @property
    def batch_size(self) -> int:
        return self.num_envs * self.rollout_steps

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def sha256(self) -> str:
        return canonical_hash(self.to_dict())

    def validate(self) -> None:
        if self.schema != SCHEMA or self.environment.observation_schema != "pace_dense_partial_93_plus_reserved12_v1":
            raise ValueError("Unsupported schema; use a new implementation version for new semantics")
        if tuple(self.environment.recipes) != RECIPES:
            raise ValueError("This reconstruction freezes the main-text recipes and 105-slot observation schema")
        if self.environment.horizon <= 0 or self.episodes_per_round <= 0 or self.evaluation_rounds <= 0:
            raise ValueError("Episode/round counts must be positive")
        if self.num_envs <= 0 or self.rollout_steps <= 0 or self.total_steps <= 0:
            raise ValueError("Training dimensions must be positive")
        if self.ppo.minibatches <= 0 or self.ppo.epochs <= 0 or self.ppo.bptt_steps <= 0:
            raise ValueError("PPO dimensions must be positive")
        if self.rollout_steps % self.ppo.bptt_steps or self.total_steps % self.batch_size:
            raise ValueError("Rollout must divide into BPTT sequences and budget into complete batches")
        if self.ppo.lr <= 0 or self.ppo.clip <= 0 or self.ppo.max_grad_norm <= 0 or self.ppo.entropy < 0 or self.ppo.value_coef < 0:
            raise ValueError("Invalid PPO scale")
        numbers = [value for value in asdict(self.ppo).values() if isinstance(value, (int, float))]
        numbers += [value for value in asdict(self.environment).values() if isinstance(value, (int, float))]
        if not all(math.isfinite(value) for value in numbers):
            raise ValueError("Experiment numbers must be finite")
        if min(self.environment.interact_reward, self.environment.progress_reward, self.environment.complete_reward, self.environment.step_cost) < 0:
            raise ValueError("Reward magnitudes and step cost must be nonnegative")
        if self.hidden_dim != 128 or tuple(self.hidden_dims) != (128, 128):
            raise ValueError("Paper architecture is frozen to MLP[128,128], GRU128")
        if (self.l1_ego_side, self.ego_side) not in ((1, 0), (0, 1)):
            raise ValueError("Nested levels must occupy opposite roles")
        if self.context_reset != "round" or not all((self.previous_action_input, self.episode_boundary_input, self.finite_horizon_returns)):
            raise ValueError("Only the documented history/return contract is implemented")
        groups = [self.construction_seeds, self.validation_seeds, self.test_seeds]
        if not all(groups) or len(set(sum((list(g) for g in groups), []))) != sum(map(len, groups)):
            raise ValueError("L1 construction/validation/test seed partitions must be nonempty and disjoint")
        if not 0 <= self.ppo.gamma <= 1 or not 0 <= self.ppo.gae_lambda <= 1:
            raise ValueError("Invalid discount/GAE coefficient")


def _strict(cls, value: dict):
    unknown = set(value) - {f.name for f in fields(cls)}
    if unknown:
        raise ValueError(f"Unknown {cls.__name__} fields: {sorted(unknown)}")
    return cls(**value)


def load_config(path: Path | str) -> ExperimentConfig:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    data["environment"] = _strict(EnvironmentConfig, data.get("environment", {}))
    data["ppo"] = _strict(PPOConfig, data.get("ppo", {}))
    config = _strict(ExperimentConfig, data)
    config.validate()
    return config
