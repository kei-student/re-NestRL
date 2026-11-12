#!/usr/bin/env python3
"""Run a frozen, explicit L1/L0 evaluation plan and aggregate every-partner passes."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import json
from pathlib import Path
import subprocess
import sys

from nestrl_research.artifacts import atomic_json, file_hash
from nestrl_research.config import ROOT, load_config
from nestrl_research.evaluation import passes_each_partner
from nestrl_research.pipeline import ensure_registry


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--models-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    config = load_config(ROOT / plan["training_config"])
    if config.sha256 != plan["training_config_sha256"]:
        raise ValueError("Training config differs from the evaluation plan")
    if (not config.environment.random_start or plan["weight_updates"]
            or plan["deterministic_actions"] or plan["device"] != "cpu"
            or plan["episodes_per_round"] != config.episodes_per_round
            or plan["episodes_per_partner"] != plan["rounds_per_partner"] * config.episodes_per_round
            or len(set(plan["seeds"])) != len(plan["seeds"])):
        raise ValueError("Unsupported or inconsistent evaluation plan")
    root = args.output_root
    root.mkdir(parents=True, exist_ok=False)
    atomic_json(root / "plan.json", plan)
    ensure_registry(config, root / "runs")  # Publish once before concurrent readers.

    def evaluate_seed(seed):
        checkpoint = args.models_root / f"l1-construction-{seed}/checkpoints/final.pt"
        before = file_hash(checkpoint)
        run_id = f"eval-l1-{seed}-l0-{plan['episodes_per_partner']}-s{plan['evaluation_seed']}"
        command = [sys.executable, "-m", "nestrl_research", "evaluate",
                   "--config", plan["training_config"], "--checkpoint", str(checkpoint),
                   "--output-root", str(root / "runs"), "--run-id", run_id,
                   "--partner-level", "l0", "--device", plan["device"],
                   "--threads", str(plan["threads_per_worker"]),
                   "--rounds", str(plan["rounds_per_partner"]),
                   "--evaluation-seed", str(plan["evaluation_seed"])]
        execution = root / "executions" / f"seed-{seed}"
        recorded = [sys.executable, str(ROOT / "scripts/record_run.py"),
                    "--run-id", run_id, "--config", str(args.plan), "--output", str(execution), "--", *command]
        subprocess.run(recorded, cwd=ROOT, check=True)
        record = json.loads((execution / "run.json").read_text())
        if not record["source_unchanged"] or not record["config_unchanged"] or file_hash(checkpoint) != before:
            raise ValueError("Evaluation inputs or checkpoint changed")
        summary = json.loads((root / "runs" / run_id / "summary.json").read_text())
        protocol = summary["protocol"]
        if (protocol["learner_checkpoint_sha256"] != before or protocol["weight_updates"]
                or protocol["rounds"] != plan["rounds_per_partner"]
                or protocol["seed"] != plan["evaluation_seed"] or protocol["deterministic_actions"]):
            raise ValueError("Recorded evaluation protocol differs from the plan")
        passed = passes_each_partner(summary, plan["partner_ids"], plan["episodes_per_partner"],
                                     plan["minimum_successes_per_partner"])
        counts = summary["per_partner_counts"]
        row = {"seed": seed, "checkpoint_sha256": before, "run_id": run_id,
               "counts": counts, "per_partner": summary["per_partner"],
               "minimum_success_rate": min(summary["per_partner"].values()), "passes_all": passed}
        print(json.dumps(row), flush=True)
        return row

    with ThreadPoolExecutor(max_workers=plan["workers"]) as pool:
        futures = [pool.submit(evaluate_seed, seed) for seed in plan["seeds"]]
        rows = sorted((future.result() for future in as_completed(futures)), key=lambda row: row["seed"])
    result = {"evaluation_id": plan["evaluation_id"], "plan": plan, "models": rows,
              "passed_seeds": [row["seed"] for row in rows if row["passes_all"]],
              "passed_count": sum(row["passes_all"] for row in rows), "model_count": len(rows),
              "total_episodes": len(rows) * len(plan["partner_ids"]) * plan["episodes_per_partner"]}
    atomic_json(root / "summary.json", result)
    with (root / "summary.csv").open("x", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["seed", *plan["partner_ids"], "minimum_success_rate", "passes_all"])
        for row in rows:
            writer.writerow([row["seed"], *(row["per_partner"][identity] for identity in plan["partner_ids"]),
                             row["minimum_success_rate"], row["passes_all"]])
    print(json.dumps({"passed_count": result["passed_count"], "passed_seeds": result["passed_seeds"],
                      "models": len(rows), "episodes": result["total_episodes"]}), flush=True)


if __name__ == "__main__":
    main()
