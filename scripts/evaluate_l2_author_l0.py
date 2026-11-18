#!/usr/bin/env python3
"""Frozen re-NestRL ego vs byte-pinned author RuleBasedPolicy, on re-NestRL dynamics."""
from __future__ import annotations
import argparse
import ast
from collections import Counter
import contextlib
from dataclasses import asdict
import hashlib
import io
import json
from pathlib import Path
import numpy as np
import torch
from nestrl_research.artifacts import atomic_json, file_hash, freeze_manifest, new_run, runtime_metadata
from nestrl_research.config import load_config
from nestrl_research.environment import CookingEnv, RECIPE_INGREDIENTS, RECIPES, upstream_modules
from nestrl_research.evaluation import PolicyContext, summarize
from nestrl_research.pipeline import ensure_registry, load_policy, seed_all
from nestrl_research.provenance import source_metadata

AUTHOR_COMMIT = "918d33fc831c520545460e7c06d0b6d797ab8be4"
POLICY_SHA = "097d50d5d4e701991dcedaecd34cced473a18809b8ba84f585756eee12c98833"
ZOO_SHA = "f652db4032116b6a5d1f1b15a68b4c00d755ed00d39ad4426b3e206d93dbba40"
# Python3.12 ast.dump; source bytes are separately version-independent pinned.
AST_SHA = "a774a632198a9963ef9197a3982e19bc5c1128caac31219a2182752e6c09d444"
NAMES = {"static_objects","ingredients","dynamic_objects","Action","inf","event","RuleBasedPolicy","generate_policy_pool"}

def selected_module(source):
    tree = ast.parse(source)
    nodes = []
    for node in tree.body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in NAMES:
            nodes.append(node)
        elif isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in NAMES for t in node.targets):
            nodes.append(node)
    return ast.Module(body=nodes, type_ignores=[])

def load_author(policy_path, zoo_path):
    policy_path, zoo_path = Path(policy_path), Path(zoo_path)
    if file_hash(policy_path) != POLICY_SHA or file_hash(zoo_path) != ZOO_SHA:
        raise ValueError("Author source bytes differ from pinned inputs")
    module = selected_module(policy_path.read_text())
    digest = hashlib.sha256(ast.dump(module, include_attributes=False).encode()).hexdigest()
    if digest != AST_SHA:
        raise ValueError("Extracted author AST differs; author methods must remain unchanged")
    zoo = ast.parse(zoo_path.read_text())
    maps = [ast.literal_eval(n.value) for n in zoo.body if isinstance(n, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "Ingred2ID" for t in n.targets)]
    if len(maps) != 1:
        raise ValueError("Author ingredient map is ambiguous")
    modules = upstream_modules()
    namespace = {}
    for name in ("gym_cooking.cooking_world.abstract_classes","gym_cooking.cooking_world.world_objects"):
        namespace.update(vars(modules[name]))
    namespace.update(np=np, Ingred2ID=maps[0], __name__="pinned_author_l0")
    exec(compile(module, str(policy_path), "exec"), namespace)
    return namespace

def author_pool(namespace, p, pool_seed=1):
    if p not in (0.0, 0.25) or type(pool_seed) is not int or not 0 <= pool_seed < 2**32:
        raise ValueError("This protocol permits p=0 or0.25 and an unsigned pool seed")
    with contextlib.redirect_stdout(io.StringIO()):
        pool = namespace["generate_policy_pool"](False,p,"fc_small_divider_test",3,"cross",pool_seed)
    by_recipe = {}
    for peer in pool:
        pair = set(peer.ingredient_support_set)
        recipe = next((r for r, items in RECIPE_INGREDIENTS.items() if set(items)==pair),None)
        if recipe is None or recipe in by_recipe:
            raise ValueError("Author factory did not produce exactly the three original recipes")
        peer.set_id(1)
        by_recipe[recipe] = peer
    if set(by_recipe) != set(RECIPES):
        raise ValueError("Missing author recipe")
    return by_recipe

def tensor_digest(model):
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()

def verify_run(directory, episodes_per_round=5):
    """Independent streaming trace-to-episode and artifact verification."""
    directory = Path(directory)
    episodes = json.loads((directory/"episodes.json").read_text())
    rows = {}
    for e in episodes:
        key = (e["partner_id"],e["round"],e["episode_in_round"])
        if key in rows:
            raise ValueError("Duplicate episode")
        rows[key] = e
    trace_counts = Counter()
    returns = Counter()
    final_success = {}
    with (directory/"traces.jsonl").open() as stream:
        for line in stream:
            t=json.loads(line)
            key=(t["partner_id"],t["round"],t["episode_in_round"])
            if key not in rows or t["step"] != trace_counts[key]:
                raise ValueError("Missing or unordered trace episode/step")
            if len(t["actions"]) != 2 or any(type(a) is not int or not 0 <= a < 6 for a in t["actions"]):
                raise ValueError("Action outside native six-action contract")
            trace_counts[key]+=1
            returns[key]+=t["reward"]
            final_success[key]=t["success"]
    for key,e in rows.items():
        if (trace_counts[key] != e["steps"] or final_success.get(key) != e["success"]
                or not np.isclose(returns[key],e["return"],rtol=0,atol=1e-9)):
            raise ValueError("Trace/episode outcomes differ")
    rebuilt=summarize(episodes,episodes_per_round)
    summary=json.loads((directory/"summary.json").read_text())
    for key in ("episodes","successes","per_partner_counts","per_episode_position"):
        if rebuilt[key] != summary[key]:
            raise ValueError("Independent episode summary differs")
    index=json.loads((directory/"artifacts.json").read_text())
    for name,row in index["files"].items():
        if file_hash(directory/name) != row["sha256"]:
            raise ValueError("Evaluation artifact hash mismatch")
    return {"verified":True,"episodes":len(episodes),"trace_rows":sum(trace_counts.values()),
            "per_partner_counts":rebuilt["per_partner_counts"]}

@torch.no_grad()
def evaluate_author(config, output_root, checkpoint, policy_path, zoo_path, *,
                    p, run_id, rounds=200, evaluation_seed=6001, pool_seed=1, driver_commit=None,
                    expected_updates=400):
    if type(expected_updates) is not int or expected_updates not in (400, 1000):
        raise ValueError("Explicit checkpoint budget must be400 or1000 updates")
    if (type(rounds) is not int or rounds<1 or type(evaluation_seed) is not int
            or not 0 <= evaluation_seed < 2**32 or config.episodes_per_round != 5):
        raise ValueError("Invalid five-episode evaluation budget/seed")
    if evaluation_seed+200000+1000*(rounds-1)+4 >= 2**32:
        raise ValueError("Episode seed overflow")
    if (config.environment.visibility != "full" or config.environment.max_ingredients_per_agent != 1
            or not config.environment.random_start or config.environment.horizon != 40):
        raise ValueError("This frozen protocol is for random-start environment3, horizon40")
    output_root, checkpoint=Path(output_root),Path(checkpoint)
    ensure_registry(config,output_root)
    source=source_metadata()
    ego,manifest=load_policy(config,checkpoint,"cpu")
    expected_steps = expected_updates * config.batch_size
    if (manifest["ego_side"] != 0 or manifest["steps"] != expected_steps
            or config.total_steps != expected_steps):
        raise ValueError("Completed left L2 manifest/config differ from explicit checkpoint budget")
    ego.requires_grad_(False)
    namespace=load_author(policy_path,zoo_path)
    pool=author_pool(namespace,p,pool_seed)
    inputs={Path(policy_path):POLICY_SHA,Path(zoo_path):ZOO_SHA,
            output_root/"config.json":file_hash(output_root/"config.json"),
            output_root/"registry.json":file_hash(output_root/"registry.json"),
            Path(__file__):file_hash(Path(__file__))}
    trained=checkpoint.parent.parent
    inputs.update({path:file_hash(path) for path in trained.rglob("*") if path.is_file()})
    before=tensor_digest(ego)
    partner_metadata=[]
    for recipe in RECIPES:
        peer=pool[recipe]
        partner_metadata.append({"id":"author-l0-1-"+recipe,"recipe":recipe,
            "support":peer.ingredient_support_set,"event_probabilities":peer.event_probs.tolist(),
            "nav_p":peer.nav_p,"tar_p":peer.tar_p,"rand_p":peer.rand_p,
            "plate_take_requires_deliver":peer.plate_take_requires_deliver})
    protocol={**source,"config_sha256":config.sha256,"learner_checkpoint_sha256":file_hash(checkpoint),
        "author_commit":AUTHOR_COMMIT,"author_policy_sha256":POLICY_SHA,"author_zoo_sha256":ZOO_SHA,
        "author_selected_AST_sha256":AST_SHA,"partner_level":"author_l0","partners":partner_metadata,
        "p":p,"pool_seed":pool_seed,"seed":evaluation_seed,"rounds":rounds,"episodes_per_round":5,
        "checkpoint_updates":expected_updates,"checkpoint_steps":expected_steps,
        "episode_seed_rule":"seed+100000*canonical_recipe_index+1000*round+position",
        "ego_side":0,"partner_side":1,"deterministic_actions":False,"weight_updates":False,
        "author_reset":"exact original reset() no-op at each episode; same pool instance retained",
        "partner_binding":"Unmodified author event/RuleBasedPolicy/generate_policy_pool AST; class identities bound to pinned PACE world; author 73-dimensional observation is not used",
        "environment_changed":False,"observation_changed":False,"driver_commit":driver_commit,
        "driver_sha256":file_hash(Path(__file__)),"p0_is_deterministic":False}
    with new_run(output_root,run_id) as directory:
        freeze_manifest(directory/"protocol.json",protocol)
        atomic_json(directory/"runtime.json",runtime_metadata("cpu"))
        atomic_json(directory/"input-hashes-before.json",{str(path.relative_to(output_root)) if path.is_relative_to(output_root) else str(path.name):v for path,v in inputs.items()})
        episodes=[]
        with (directory/"traces.jsonl").open("x") as trace:
            for index,recipe in enumerate(RECIPES):
                peer=pool[recipe]
                identity=partner_metadata[index]["id"]
                for round_index in range(rounds):
                    seed_all(evaluation_seed+100000*index+1000*round_index)
                    context=PolicyContext(ego,"cpu",False)
                    for position in range(5):
                        peer.reset()
                        env=CookingEnv(asdict(config.environment))
                        episode_seed=evaluation_seed+100000*index+1000*round_index+position
                        observation=env.reset(seed=episode_seed)
                        total_return=0.0
                        for step in range(config.environment.horizon):
                            joint=[context.act(observation[0]),int(peer(env.world))]
                            observation,reward,terminated,truncated,info=env.step(joint)
                            total_return+=reward
                            trace.write(json.dumps({"partner_id":identity,"round":round_index,
                                "episode_in_round":position,"step":step,"actions":joint,
                                "reward":reward,"events":info["events"],"success":bool(info["success"])})+"\n")
                            if terminated or truncated:
                                break
                        episodes.append({"partner_id":identity,"round":round_index,"episode_in_round":position,
                            "seed":episode_seed,"steps":step+1,"success":bool(info["success"]),
                            "recipe":info["recipe"],"return":total_return})
                        context.episode_boundary()
                    if (round_index+1)%20==0 or round_index+1==rounds:
                        trace.flush()
                        atomic_json(directory/"progress.json",{"completed_episodes":len(episodes),
                            "target_episodes":3*rounds*5,"partner_id":identity,"completed_partner_rounds":round_index+1})
        if source_metadata()!=source or tensor_digest(ego)!=before:
            raise ValueError("Core source or in-memory ego weights changed")
        if any(file_hash(path)!=expected for path,expected in inputs.items()):
            raise ValueError("Immutable author/learner/config/driver input changed")
        summary={"protocol":protocol,**summarize(episodes,5)}
        atomic_json(directory/"episodes.json",episodes)
        atomic_json(directory/"summary.json",summary)
        atomic_json(directory/"frozen-inputs-verified.json",{"all_input_file_hashes_unchanged":True,
            "in_memory_policy_hash_before":before,"in_memory_policy_hash_after":tensor_digest(ego),
            "source_unchanged":True,"weight_updates":False})
        files=("protocol.json","runtime.json","input-hashes-before.json","traces.jsonl","episodes.json",
               "summary.json","progress.json","frozen-inputs-verified.json")
        atomic_json(directory/"artifacts.json",{"weight_updates":False,
            "files":{name:{"sha256":file_hash(directory/name)} for name in files}})
        verification=verify_run(directory)
        atomic_json(directory/"verification.json",verification)
    return summary

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("config","output-root","checkpoint","author-policy","author-zoo"):
        parser.add_argument("--"+name,type=Path,required=True)
    parser.add_argument("--p",type=float,choices=(0.0,0.25),required=True)
    parser.add_argument("--run-id",required=True)
    parser.add_argument("--rounds",type=int,default=200)
    parser.add_argument("--evaluation-seed",type=int,default=6001)
    parser.add_argument("--pool-seed",type=int,default=1)
    parser.add_argument("--expected-updates",type=int,choices=(400,1000),default=400,
                        help="Authenticate the frozen checkpoint budget; never changes training")
    parser.add_argument("--threads",type=int,default=1)
    parser.add_argument("--driver-commit")
    args=parser.parse_args()
    if args.threads != 1:
        parser.error("One Torch/BLAS thread is required")
    torch.set_num_threads(1)
    result=evaluate_author(load_config(args.config),args.output_root,args.checkpoint,
        args.author_policy,args.author_zoo,p=args.p,run_id=args.run_id,rounds=args.rounds,
        evaluation_seed=args.evaluation_seed,pool_seed=args.pool_seed,driver_commit=args.driver_commit,
        expected_updates=args.expected_updates)
    print(json.dumps(result,indent=2))
if __name__=="__main__":
    main()

