#!/usr/bin/env python3
"""Replay immutable evaluation traces and describe each passing L1 policy."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
from itertools import groupby
import json
from pathlib import Path

from nestrl_research.behavior import analyze_episode, summarize
from nestrl_research.environment import CookingEnv

ROOT = Path(__file__).resolve().parents[1]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def analyze_model(request: dict, model: dict, output: str) -> dict:
    seed = model["seed"]
    run = ROOT / request["evaluation_root"] / "runs" / model["run_id"]
    ep_path, trace_path = run / "episodes.json", run / "traces.jsonl"
    inputs = {"episodes.json": sha(ep_path), "traces.jsonl": sha(trace_path),
              "protocol.json": sha(run / "protocol.json")}
    meta = json.loads(ep_path.read_text())
    key = lambda row: (row["partner_id"], row["round"], row["episode_in_round"])
    by_key = {key(e): e for e in meta}
    if len(by_key) != request["episodes_per_model"]:
        raise ValueError("unexpected episode count")
    env = CookingEnv(request["environment"])
    rows = []
    with trace_path.open() as stream:
        for identity, grouped in groupby((json.loads(line) for line in stream), key=key):
            row = analyze_episode(env, by_key.pop(identity), list(grouped))
            rows.append({"model_seed": seed, **row})
    if by_key or len(rows) != request["episodes_per_model"]:
        raise ValueError("missing episodes")
    if inputs != {"episodes.json": sha(ep_path), "traces.jsonl": sha(trace_path),
                  "protocol.json": sha(run / "protocol.json")}:
        raise ValueError("input changed during analysis")
    result = {"seed": seed, "checkpoint_sha256": model["checkpoint_sha256"],
              "input_sha256": inputs, "all": summarize(rows),
              "by_episode": {str(e + 1): summarize([r for r in rows if r["episode_in_round"] == e])
                             for e in range(5)},
              "by_partner": {p: summarize([r for r in rows if r["partner_id"] == p])
                             for p in sorted({r["partner_id"] for r in rows})},
              "replay_verified": True}
    out = Path(output)
    (out / f"episodes-{seed}.jsonl").write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows))
    (out / f"summary-{seed}.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    request = json.loads(args.config.read_text())
    summary_path = ROOT / request["evaluation_root"] / "summary.json"
    if sha(summary_path) != request["evaluation_summary_sha256"]:
        raise ValueError("evaluation summary hash mismatch")
    models = [m for m in json.loads(summary_path.read_text())["models"] if m["passes_all"]]
    if sorted(m["seed"] for m in models) != request["model_seeds"]:
        raise ValueError("passing policy selection mismatch")
    args.output.mkdir(parents=True, exist_ok=False)
    results = []
    with ProcessPoolExecutor(max_workers=request["workers"]) as pool:
        tasks = [pool.submit(analyze_model, request, model, str(args.output)) for model in models]
        for task in as_completed(tasks):
            result = task.result()
            results.append(result)
            print(json.dumps({"seed": result["seed"], "episodes": result["all"]["episodes"],
                              "replay_verified": True}), flush=True)
    results.sort(key=lambda r: r["seed"])
    (args.output / "summary.json").write_text(json.dumps({"analysis_id": request["analysis_id"],
        "models": results, "episodes": sum(r["all"]["episodes"] for r in results),
        "weight_updates": False, "policy_inference": False}, indent=2, sort_keys=True) + "\n")
    manifest = {p.name: sha(p) for p in sorted(args.output.iterdir()) if p.is_file()}
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
