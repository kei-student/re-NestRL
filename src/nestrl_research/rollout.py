"""Collect recurrent trajectories with frozen partners and five-episode contexts."""

from __future__ import annotations

from dataclasses import asdict

import numpy as np
import torch

from .config import ExperimentConfig, RECIPES
from .environment import CookingEnv
from .peers import ScriptedPeer


class FrozenPartners:
    """Batch frozen inference by partner identity; each environment owns its history."""

    def __init__(self, config: ExperimentConfig, side: int, device, rng, models=None):
        self.config, self.side, self.device, self.rng = config, side, device, rng
        self.models = models or []
        for model in self.models:
            model.eval()
            model.requires_grad_(False)
        count = len(self.models) if self.models else len(RECIPES)
        self.indices = rng.integers(count, size=config.num_envs)
        self.hidden = torch.zeros(1, config.num_envs, config.hidden_dim, device=device)
        self.previous = torch.full((config.num_envs,), -1, dtype=torch.long, device=device)
        self.starts = torch.ones(config.num_envs, dtype=torch.bool, device=device)
        self.resets = self.starts.clone()
        self.scripts = ([] if self.models else [
            ScriptedPeer(RECIPES[int(index)], agent_id=side, seed=int(rng.integers(2**31)))
            for index in self.indices
        ])

    @torch.no_grad()
    def act(self, observations, environments, deterministic=False):
        if not self.models:
            return np.array([peer.act(env) for peer, env in zip(self.scripts, environments)], dtype=np.int64)
        observations = torch.as_tensor(observations, device=self.device)
        actions = torch.empty(self.config.num_envs, dtype=torch.long, device=self.device)
        for identity, model in enumerate(self.models):
            subset = np.flatnonzero(self.indices == identity)
            if not len(subset):
                continue
            ix = torch.as_tensor(subset, device=self.device)
            result = model.act(observations[ix], self.previous[ix], self.starts[ix], self.hidden[:, ix],
                               self.resets[ix], deterministic=deterministic)
            actions[ix], _, _, self.hidden[:, ix] = result
        return actions.cpu().numpy()

    def advance(self, actions, dones, round_dones):
        self.previous.copy_(torch.as_tensor(actions, device=self.device))
        self.starts.copy_(torch.as_tensor(dones, device=self.device))
        self.resets.copy_(torch.as_tensor(round_dones, device=self.device))
        self.previous[self.starts] = -1
        for index in np.flatnonzero(round_dones):
            count = len(self.models) if self.models else len(RECIPES)
            self.indices[index] = self.rng.integers(count)
            if not self.models:
                self.scripts[index] = ScriptedPeer(RECIPES[int(self.indices[index])], agent_id=self.side,
                                                   seed=int(self.rng.integers(2**31)))


class RolloutCollector:
    def __init__(self, config: ExperimentConfig, model, ego_side: int, seed: int, device, partner_models=None):
        self.config, self.model, self.ego_side, self.device = config, model, ego_side, device
        self.rng = np.random.default_rng(seed)
        self.envs = [CookingEnv(asdict(config.environment)) for _ in range(config.num_envs)]
        self.observations = np.stack([env.reset(seed=int(self.rng.integers(2**31))) for env in self.envs])
        self.partners = FrozenPartners(config, 1 - ego_side, device, self.rng, partner_models)
        self.hidden = model.initial_state(config.num_envs, device)
        self.previous = torch.full((config.num_envs,), -1, dtype=torch.long, device=device)
        self.starts = torch.ones(config.num_envs, dtype=torch.bool, device=device)
        self.resets = self.starts.clone()
        self.in_round = np.zeros(config.num_envs, dtype=np.int64)
        self.episode_return = np.zeros(config.num_envs)
        self.episode_steps = np.zeros(config.num_envs, dtype=np.int64)
        self.episodes = []
        # Replay at most one episode to rebuild PACE's dynamically loaded objects.
        # Pickling those objects would depend on process-local module identities.
        self.action_history = [[] for _ in self.envs]

    @torch.no_grad()
    def collect(self):
        data = {key: [] for key in ("observations", "previous_actions", "episode_starts", "resets", "hidden_states",
                                    "actions", "old_log_probs", "old_values", "rewards", "dones")}
        completed = []
        for _ in range(self.config.rollout_steps):
            observation = torch.as_tensor(self.observations[:, self.ego_side], device=self.device)
            for key, value in (("observations", observation), ("previous_actions", self.previous),
                               ("episode_starts", self.starts), ("resets", self.resets), ("hidden_states", self.hidden[0])):
                data[key].append(value.clone())
            actions, log_probs, values, self.hidden = self.model.act(observation, self.previous, self.starts,
                                                                    self.hidden, self.resets)
            data["actions"].append(actions)
            data["old_log_probs"].append(log_probs)
            data["old_values"].append(values)
            partner_actions = self.partners.act(self.observations[:, 1 - self.ego_side], self.envs)
            rewards, dones, round_dones = [], [], []
            for index, env in enumerate(self.envs):
                joint = [0, 0]
                joint[self.ego_side] = int(actions[index])
                joint[1 - self.ego_side] = int(partner_actions[index])
                observation_next, reward, terminated, truncated, info = env.step(joint)
                self.action_history[index].append(joint)
                done = bool(terminated or truncated)
                self.episode_return[index] += reward
                self.episode_steps[index] += 1
                round_done = False
                if done:
                    completed.append({"success": bool(info["success"]), "return": float(self.episode_return[index]),
                                      "length": int(self.episode_steps[index]), "episode_in_round": int(self.in_round[index]),
                                      "partner_index": int(self.partners.indices[index])})
                    self.in_round[index] += 1
                    round_done = self.in_round[index] == self.config.episodes_per_round
                    if round_done:
                        self.in_round[index] = 0
                    self.episode_return[index] = self.episode_steps[index] = 0
                    observation_next = env.reset(seed=int(self.rng.integers(2**31)))
                    self.action_history[index] = []
                self.observations[index] = observation_next
                rewards.append(reward)
                dones.append(done)
                round_dones.append(round_done)
            data["rewards"].append(torch.tensor(rewards, dtype=torch.float32, device=self.device))
            data["dones"].append(torch.tensor(dones, dtype=torch.bool, device=self.device))
            self.previous = actions.clone()
            self.starts = torch.tensor(dones, dtype=torch.bool, device=self.device)
            self.resets = torch.tensor(round_dones, dtype=torch.bool, device=self.device)
            self.previous[self.starts] = -1
            self.partners.advance(partner_actions, dones, round_dones)
        # Evaluate the next state without advancing its carried recurrent history.
        output = self.model(torch.as_tensor(self.observations[:, self.ego_side], device=self.device).unsqueeze(0),
                            self.previous.unsqueeze(0), self.starts.unsqueeze(0), self.hidden, self.resets.unsqueeze(0))
        self.episodes.extend(completed)
        return {key: torch.stack(value) for key, value in data.items()}, output["values"][0], completed

    def state_dict(self):
        """Portable state at a completed rollout/update boundary (no pickle code)."""
        arrays = ("observations", "in_round", "episode_return", "episode_steps")
        tensors = ("hidden", "previous", "starts", "resets")
        return {
            "arrays": {name: torch.from_numpy(getattr(self, name).copy()) for name in arrays},
            "tensors": {name: getattr(self, name).detach().cpu().clone() for name in tensors},
            "rng": self.rng.bit_generator.state,
            "episodes": self.episodes,
            "environments": [{"seed": env.seed, "actions": actions} for env, actions in
                             zip(self.envs, self.action_history)],
            "partners": {
                "indices": torch.from_numpy(self.partners.indices.copy()),
                "tensors": {name: getattr(self.partners, name).detach().cpu().clone() for name in tensors},
                "scripts": [{"recipe": peer.recipe, "agent_id": peer.agent_id, "seed": peer.seed,
                             "random_action_probability": peer.random_action_probability,
                             "rng": peer.rng.bit_generator.state} for peer in self.partners.scripts],
            },
        }

    def load_state_dict(self, state):
        if len(state["environments"]) != len(self.envs):
            raise ValueError("Checkpoint environment count differs")
        for index, (env, saved) in enumerate(zip(self.envs, state["environments"])):
            observation = env.reset(seed=saved["seed"])
            for actions in saved["actions"]:
                observation, _, _, _, _ = env.step(actions)
            if not np.array_equal(observation, state["arrays"]["observations"][index].numpy()):
                raise ValueError("Checkpoint environment replay differs")
        self.action_history = [[list(actions) for actions in saved["actions"]]
                               for saved in state["environments"]]
        for name, value in state["arrays"].items():
            setattr(self, name, value.numpy().copy())
        for name, value in state["tensors"].items():
            setattr(self, name, value.to(self.device).clone())
        self.rng.bit_generator.state = state["rng"]
        self.episodes = state["episodes"]
        self.partners.indices = state["partners"]["indices"].numpy().copy()
        for name, value in state["partners"]["tensors"].items():
            setattr(self.partners, name, value.to(self.device).clone())
        self.partners.scripts = []
        for saved in state["partners"]["scripts"]:
            peer = ScriptedPeer(saved["recipe"], saved["agent_id"], saved["seed"],
                                saved["random_action_probability"])
            peer.rng.bit_generator.state = saved["rng"]
            self.partners.scripts.append(peer)
