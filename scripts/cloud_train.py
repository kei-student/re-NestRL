#!/usr/bin/env python3
"""Run CPU construction training as detached jobs on the local Docker daemon."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
IMAGE = "python@sha256:aa943ce411423c7582288f18707ff66eddcbaeb510502d747a72aeb34f30d159"
LABEL = "re-nestrl.training-root"


def docker(*args, check=True):
    environment = dict(os.environ)
    for key in ("DOCKER_HOST", "DOCKER_CONTEXT", "DOCKER_TLS", "DOCKER_TLS_VERIFY", "DOCKER_CERT_PATH"):
        environment.pop(key, None)
    return subprocess.run(["docker", "--host=unix:///var/run/docker.sock", *args], env=environment,
                          text=True, capture_output=True, check=check)


def name_for(root, seed):
    identity = hashlib.sha256(str(root).encode()).hexdigest()[:12]
    return f"nestrl-{identity}-l1-{seed}"


def inspect(name, root):
    template = '{"State":{{json .State}},"RestartCount":{{.RestartCount}},"Labels":{{json .Config.Labels}}}'
    result = docker("container", "inspect", "--format", template, name, check=False)
    if result.returncode:
        if "No such" in result.stderr:
            return None
        raise RuntimeError(result.stderr.strip())
    info = json.loads(result.stdout)
    if (info.get("Labels") or {}).get(LABEL) != str(root):
        raise RuntimeError("Container identity does not match this experiment")
    return info


def worker(args, root, config):
    directory = root / f"l1-construction-{args.seed}"
    if (directory / "complete.json").exists():
        print("Training already complete", flush=True)
        return
    command = [sys.executable, "-u", "-m", "nestrl_research", "train", "--config", str(config),
               "--output-root", str(root), "--stage", "l1", "--split", "construction",
               "--seed", str(args.seed), "--device", "cpu", "--threads", "1", "--checkpoint-every", "1"]
    if (directory / "resume/latest.json").exists():
        command.append("--resume")
    elif directory.exists():
        raise RuntimeError("Existing run has no recoverable snapshot; preserve it and choose a new output root")
    if args.max_updates:
        command.extend(["--max-updates", str(args.max_updates)])
    # Persistent append-only log, including across container restarts.
    with (root / f"seed-{args.seed}.log").open("a", buffering=1) as log:
        log.flush()
        os.dup2(log.fileno(), 1)
        os.dup2(log.fileno(), 2)
        os.chdir(ROOT)
        os.execv(command[0], command)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("start", "resume", "status", "stop", "worker"))
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/paper.json")
    parser.add_argument("--seeds", type=int, nargs="+", default=[101, 102, 103])
    parser.add_argument("--seed", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--max-updates", type=int, help="Pause after N updates per invocation (for verification)")
    parser.add_argument("--memory-gib", type=int, help="Memory limit per job in GiB (default: 8)")
    args = parser.parse_args()
    root = args.output_root.resolve()
    config = args.config.resolve()
    if not root.is_relative_to(ROOT / "outputs") or root == ROOT / "outputs":
        parser.error("Choose a dedicated output directory beneath this repository's outputs/")
    if not config.is_relative_to(ROOT) or not config.is_file():
        parser.error("Config must be a file inside this repository")
    if args.max_updates is not None and args.max_updates < 1:
        parser.error("--max-updates must be positive")
    if args.memory_gib is not None and args.memory_gib < 1:
        parser.error("--memory-gib must be positive")
    memory_gib = args.memory_gib or 8
    if args.action == "worker":
        if args.seed is None:
            parser.error("worker requires --seed")
        worker(args, root, config)
        return
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("Duplicate seeds")
    from nestrl_research.config import load_config
    from nestrl_research.pipeline import ensure_registry
    cfg = load_config(config)
    if not set(args.seeds).issubset(cfg.construction_seeds):
        parser.error("Seeds must belong to this config's construction split")
    docker("info", "--format", "{{.ServerVersion}}")
    if args.action in ("start", "resume"):
        memory_limit = Path("/sys/fs/cgroup/memory.max")
        if memory_limit.exists():
            limit = memory_limit.read_text().strip()
            if limit != "max" and memory_gib * 2**30 * len(args.seeds) > int(limit):
                parser.error("Requested job memory limits exceed this environment's budget; reduce --memory-gib")
        root.mkdir(parents=True, exist_ok=True)
        ensure_registry(cfg, root)
        docker("image", "inspect", IMAGE)
    for seed in args.seeds:
        name = name_for(root, seed)
        info = inspect(name, root)
        if args.action == "status":
            directory = root / f"l1-construction-{seed}"
            metrics = directory / "metrics.jsonl"
            latest = None
            if metrics.exists():
                for line in metrics.read_text().splitlines():
                    try:
                        latest = json.loads(line)
                    except json.JSONDecodeError:
                        break
            print(json.dumps({"seed": seed, "container": name, "state": info["State"]["Status"] if info else "absent",
                              "exit_code": info["State"]["ExitCode"] if info else None,
                              "restarts": info["RestartCount"] if info else 0,
                              "complete": (directory / "complete.json").exists(),
                              "recoverable": (directory / "resume/latest.json").exists(), "latest_metrics": latest}))
        elif args.action == "stop":
            if info and info["State"]["Running"]:
                docker("stop", "--time", "120", name)
            print(json.dumps({"seed": seed, "container": name, "stopped": True}))
        elif info:
            if args.action != "resume":
                raise RuntimeError(f"Container {name} already exists; inspect status or use resume")
            # A retained container must not silently change its immutable settings.
            if info["Labels"].get("re-nestrl.config-sha256") != cfg.sha256:
                raise RuntimeError("Container config differs from the requested config")
            if args.max_updates is not None and info["Labels"].get("re-nestrl.max-updates") != str(args.max_updates):
                raise RuntimeError("Retained container uses a different --max-updates limit")
            if args.memory_gib is not None and info["Labels"].get("re-nestrl.memory-gib") != str(memory_gib):
                raise RuntimeError("Retained container uses a different --memory-gib limit")
            if not info["State"]["Running"]:
                docker("start", name)
            print(json.dumps({"seed": seed, "container": name, "started": True}))
        else:
            directory = root / f"l1-construction-{seed}"
            if directory.exists() and not (directory / "resume/latest.json").exists():
                raise RuntimeError("Run has no recoverable snapshot; preserve it and choose a new output root")
            # Use the retained .venv and its exact base interpreter, not an image's
            # possibly different Python/PyTorch. Only experiment outputs are writable.
            command = ["run", "--detach", "--name", name, "--init", "--network", "none",
                       "--restart", "on-failure:3", "--stop-timeout", "120", "--cpus", "1", "--memory", f"{memory_gib}g",
                       "--pids-limit", "128", "--user", f"{os.getuid()}:{os.getgid()}",
                       "--read-only", "--tmpfs", "/tmp:rw,nosuid,nodev,size=256m",
                       "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
                       "--label", f"{LABEL}={root}", "--label", f"re-nestrl.config-sha256={cfg.sha256}",
                       "--label", f"re-nestrl.max-updates={args.max_updates or 0}",
                       "--label", f"re-nestrl.memory-gib={memory_gib}",
                       "--mount", f"type=bind,src={ROOT},dst={ROOT},readonly",
                       "--mount", f"type=bind,src={root},dst={root}",
                       "--mount", f"type=bind,src={sys.base_prefix},dst={sys.base_prefix},readonly",
                       "-e", "OMP_NUM_THREADS=1", "-e", "MKL_NUM_THREADS=1", "-e", "OPENBLAS_NUM_THREADS=1",
                       "-e", "PYTHONDONTWRITEBYTECODE=1", "--workdir", str(ROOT), IMAGE,
                       str(ROOT / ".venv/bin/python"), "-u", str(Path(__file__).resolve()), "worker",
                       "--output-root", str(root), "--config", str(config), "--seed", str(seed)]
            if args.max_updates:
                command.extend(["--max-updates", str(args.max_updates)])
            docker(*command)
            print(json.dumps({"seed": seed, "container": name, "started": True}), flush=True)


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as error:
        raise SystemExit(error.stderr.strip() or str(error))
