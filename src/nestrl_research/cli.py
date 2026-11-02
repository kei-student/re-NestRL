"""Command-line entry points with explicit profiles and non-overwriting run IDs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .artifacts import choose_device
from .config import ROOT, load_config
from .evaluation import evaluate
from .pipeline import ensure_registry, partition_seeds, train


def main(argv=None):
    parser = argparse.ArgumentParser(description="Fresh NestRL reconstruction from arXiv v2")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("registry", "train", "evaluate", "pipeline"):
        command_parser = subparsers.add_parser(command)
        command_parser.add_argument("--config", required=True, type=Path)
        command_parser.add_argument("--output-root", type=Path)
        command_parser.add_argument("--device", default="auto", choices=("auto", "cpu", "mps", "cuda"))
        command_parser.add_argument("--threads", type=int, default=1)
        if command == "train":
            command_parser.add_argument("--stage", required=True, choices=("l1", "l2", "generalist"))
            command_parser.add_argument("--split", default="construction", choices=("construction", "validation", "test"))
            command_parser.add_argument("--seed", type=int)
            command_parser.add_argument("--run-id")
        if command == "evaluate":
            command_parser.add_argument("--checkpoint", type=Path, required=True)
            command_parser.add_argument("--partner-level", default="l1", choices=("l0", "l1"))
            command_parser.add_argument("--split", default="test", choices=("validation", "test"))
            command_parser.add_argument("--run-id")
            command_parser.add_argument("--deterministic", action="store_true")
    args = parser.parse_args(argv)
    if args.threads < 1:
        parser.error("--threads must be positive")
    torch.set_num_threads(args.threads)
    config = load_config(args.config)
    output_root = (args.output_root or ROOT / "outputs" / config.profile).resolve()
    device = choose_device(args.device)
    if args.command == "registry":
        print(json.dumps({"registry_sha256": ensure_registry(config, output_root), "config_sha256": config.sha256}))
    elif args.command == "train":
        print(train(config, output_root, args.stage, device, split=args.split, seed=args.seed, run_id=args.run_id))
    elif args.command == "evaluate":
        print(json.dumps(evaluate(config, output_root, args.checkpoint, device, partner_level=args.partner_level,
                                  split=args.split, run_id=args.run_id, deterministic=args.deterministic), indent=2))
    else:
        ensure_registry(config, output_root)
        for split in ("construction", "validation", "test"):
            for seed in partition_seeds(config, split):
                train(config, output_root, "l1", device, split=split, seed=seed)
        for method in ("l2", "generalist"):
            checkpoint = train(config, output_root, method, device)
            for split in ("validation", "test"):
                summary = evaluate(config, output_root, checkpoint, device, split=split)
                print(json.dumps({"method": method, "split": split, "success_rate": summary["success_rate"]}), flush=True)
            summary = evaluate(config, output_root, checkpoint, device, partner_level="l0")
            print(json.dumps({"method": method, "fixed_l0_success_rate": summary["success_rate"]}), flush=True)


if __name__ == "__main__":
    main()
