#!/usr/bin/env python3
"""Fail-fast checks for this independent checkout and reconstruction contract."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bootstrap_upstream import verify_upstream
from nestrl_research.artifacts import choose_device
from nestrl_research.config import RECIPES, load_config
from nestrl_research.environment import CookingEnv
from nestrl_research.model import RecurrentActorCritic
from nestrl_research.peers import ScriptedPeer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "mps", "cuda"))
    parser.add_argument("--allow-dirty", action="store_true")
    args = parser.parse_args()
    if sys.version_info[:2] != (3, 12) or Path(sys.prefix).resolve() != (ROOT / ".venv").resolve():
        raise RuntimeError("Use Python 3.12 from this repository's activated .venv")
    if not (ROOT / ".git").exists():
        raise RuntimeError("Independent re-NestRL Git repository required; parent repositories are never used")
    git_root = subprocess.check_output(["git", "rev-parse", "--show-toplevel"], cwd=ROOT, text=True).strip()
    if Path(git_root).resolve() != ROOT.resolve():
        raise RuntimeError("Git root is outside re-NestRL")
    dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip()
    if dirty and not args.allow_dirty:
        raise RuntimeError("Working tree is dirty; use --allow-dirty only during development")
    verified = verify_upstream()
    configs = {name: load_config(ROOT / "configs" / f"{name}.json").sha256 for name in ("paper", "smoke")}
    results = {}
    for recipe in RECIPES:
        env = CookingEnv()
        observation = env.reset(seed=0)
        assert observation.shape == (2, 105)
        peers = [ScriptedPeer(recipe, agent_id=side) for side in (0, 1)]
        for step in range(40):
            observation, reward, terminated, truncated, info = env.step([peer.act(env) for peer in peers])
            if terminated or truncated:
                break
        if not info["success"] or info["recipe"] != recipe:
            raise RuntimeError(f"Scripted integration failed: {recipe}")
        results[recipe] = step + 1
    import torch
    torch.set_num_threads(1)
    device = choose_device(args.device)
    model = RecurrentActorCritic().to(device)
    model.act(torch.as_tensor(observation[:1], device=device), torch.tensor([-1], device=device),
              torch.tensor([True], device=device), model.initial_state(1, device))
    print(json.dumps({"status": "ok", "python": sys.version.split()[0], "torch": torch.__version__,
                      "device": str(device), "upstream_source_files_verified": verified,
                      "scripted_recipe_steps": results, "config_sha256": configs}, indent=2))


if __name__ == "__main__":
    main()
