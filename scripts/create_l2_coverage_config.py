#!/usr/bin/env python3
"""Freeze an independent three-L1 draw plus all three L0s; never train."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import secrets

from nestrl_research.artifacts import file_hash, safe_name
from nestrl_research.config import ROOT, RECIPES, L2MixtureConfig, load_config
from nestrl_research.partner_pool import read_catalog, relative_path, select_partner_ids


def fresh_selection_seed():
    return secrets.randbits(32)


def create(study_path, learner_seed, output_root, *, selection_seed=None):
    study_path, output_root = Path(study_path), Path(output_root)
    if output_root.exists():
        raise FileExistsError("Instance exists; reuse its frozen config without redrawing")
    if type(learner_seed) is not int or not 0 <= learner_seed < 2**32:
        raise ValueError("Learner seed must be an integer in [0,2**32)")
    study = json.loads(study_path.read_text())
    if study.get("schema") != "re-nestrl-l2-coverage-study-v1":
        raise ValueError("Use the coverage study")
    if (study["l1_partner_count"] != 3 or tuple(study["l0_recipes"]) != RECIPES
            or study["sampling"] != "uniform_per_partner_per_round"
            or study["updates_per_model"] != 1000 or study["steps_per_model"] != 72_000_000):
        raise ValueError("Coverage study must preserve three L0s, three L1s and 1000 updates")
    base = load_config(relative_path(ROOT, study["base_config"]))
    catalog_path = relative_path(ROOT, study["catalog"])
    if base.sha256 != study["base_config_sha256"] or file_hash(catalog_path) != study["catalog_sha256"]:
        raise ValueError("Base configuration or authenticated catalog changed")
    if (base.l1_pool is None or base.l1_pool.catalog_sha256 != study["catalog_sha256"]
            or base.ego_side != 0 or base.l1_ego_side != 1
            or base.environment.visibility != "full" or base.environment.max_ingredients_per_agent != 1
            or not base.environment.random_start or base.environment.horizon != 40
            or base.episodes_per_round != 5 or base.batch_size != 72_000):
        raise ValueError("Coverage base differs from the environment3 comparison")
    catalog = read_catalog(catalog_path)
    if sorted(m["seed"] for m in catalog["models"]) != study["candidate_seeds"]:
        raise ValueError("Qualified construction cohort differs")
    seed = fresh_selection_seed() if selection_seed is None else selection_seed
    chosen = select_partner_ids([m["id"] for m in catalog["models"]], 3, seed)
    identity = safe_name(output_root.name)
    resolved = replace(base, profile=f"l2-env3-coverage-1000u-{identity}",
                       learner_seed=learner_seed, total_steps=study["steps_per_model"],
                       l1_pool=replace(base.l1_pool, partner_ids=chosen),
                       l2_mixture=L2MixtureConfig())
    resolved.validate()
    l0_ids = [f"l0-1-{recipe}" for recipe in RECIPES]
    record = {
        "schema": "re-nestrl-l2-coverage-selection-v1", "instance_id": identity,
        "learner_seed": learner_seed, "selection_seed": seed,
        "selection_seed_source": "fresh OS entropy" if selection_seed is None else "explicit replay seed",
        "selection_rng": "numpy.PCG64", "l1_partner_ids": list(chosen), "l0_partner_ids": l0_ids,
        "partner_order": [*chosen, *l0_ids], "probability_per_partner": 1/6,
        "level_probabilities": {"l0": 0.5, "l1": 0.5}, "sampling": study["sampling"],
        "membership_lifetime": "fixed for this L2; reuse on resume",
        "study_file_sha256": file_hash(study_path), "base_config_sha256": base.sha256,
        "catalog_sha256": study["catalog_sha256"], "config_sha256": resolved.sha256,
        "updates": 1000, "steps": 72_000_000, "training_started": False}
    output_root.mkdir(parents=True, exist_ok=False)
    with (output_root/"config.json").open("x") as stream:
        stream.write(json.dumps(resolved.to_dict(), indent=2)+"\n")
    record["config_file_sha256"] = file_hash(output_root/"config.json")
    with (output_root/"selection.json").open("x") as stream:
        stream.write(json.dumps(record, indent=2)+"\n")
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, default=ROOT/"configs/l2-env3-coverage-1000u-20261117.json")
    parser.add_argument("--learner-seed", type=int, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--selection-seed", type=int, help="Replay a recorded draw; omit for fresh OS entropy")
    args = parser.parse_args()
    print(json.dumps(create(args.study, args.learner_seed, args.output_root,
                            selection_seed=args.selection_seed), indent=2))


if __name__ == "__main__":
    main()
