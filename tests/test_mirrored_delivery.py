"""Mirrored geometry, either-side delivery, fair contention and 108-slot recovery."""
from dataclasses import asdict, replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from nestrl_research.artifacts import load_checkpoint
from nestrl_research.config import ExperimentConfig, EnvironmentConfig as RunEnvironment, PPOConfig, load_config
from nestrl_research.environment import CookingEnv, EnvironmentConfig, MIRRORED_LAYOUT, RECIPES, RECIPE_INGREDIENTS, observation_schema_for
from nestrl_research.peers import ScriptedPeer
from nestrl_research.pipeline import train, load_policy
from nestrl_research.evaluation import evaluate

ROOT = Path(__file__).resolve().parents[1]
VARIANTS = (("full", None), ("own_side", 1), ("full", 1))


def config(visibility="full", limit=None, random=True):
    return EnvironmentConfig(layout=MIRRORED_LAYOUT, observation_dim=108, visibility=visibility,
        max_ingredients_per_agent=limit, random_start=random,
        observation_schema=observation_schema_for(visibility, limit, MIRRORED_LAYOUT))


def complete_plate(env, recipe=RECIPES[0]):
    plate = env.world.world_objects["Plate"][0]
    for name in RECIPE_INGREDIENTS[recipe]:
        food = env.world.world_objects[name][0]
        food.chop()
        plate.add_content(food)
        food.move_to(plate.location)
    return plate


def recipe_run(env, recipe, starts, server):
    env.reset(seed=101)
    for a, pos in zip(env.world.agents, starts):
        a.move_to(pos)
    peers = [ScriptedPeer(recipe, i) for i in (0, 1)]
    for step in range(1, 41):
        plate = env.world.world_objects["Plate"][0]
        actions = [p.act(env) for p in peers]
        for i in (0, 1):
            own = env.world.world_objects[RECIPE_INGREDIENTS[recipe][i]][0]
            if i != server and own in plate.content and env.world.agents[i].holding is None:
                actions[i] = 0
        _, _, done, truncated, info = env.step(actions)
        if done or truncated:
            return done, info, step, plate.location
    raise AssertionError("No finite boundary")


class MirroredGeometryTests(unittest.TestCase):
    def test_stations_foods_fixed_starts_and_facing_are_mirrored(self):
        env = CookingEnv(config(random=False))
        self.assertEqual(env.observe().shape, (2, 108))
        self.assertEqual(env.observation_dim, 108)
        self.assertEqual([a.location for a in env.world.agents], [(2, 5), (4, 5)])
        self.assertEqual([a.orientation for a in env.world.agents], [1, 2])
        self.assertEqual([o.location for o in env.world.world_objects["CutBoard"]], [(0, 4), (6, 4)])
        self.assertEqual([o.location for o in env.world.world_objects["DeliverSquare"]], [(0, 5), (6, 5)])
        for left, right in RECIPE_INGREDIENTS.values():
            x, y = env.world.world_objects[left][0].location
            self.assertEqual(env.world.world_objects[right][0].location, (6-x, y))
        for y in range(8):
            for x in range(7):
                a = type(env.world.get_objects_at((x,y), env.objects.StaticObject)[0])
                b = type(env.world.get_objects_at((6-x,y), env.objects.StaticObject)[0])
                self.assertEqual(a,b)
        self.assertEqual(len(env.world.world_objects["Plate"]), 1)
        self.assertEqual(len(env.world.world_objects["Floor"]), 24)

    def test_partial_view_masks_second_port_and_full_view_keeps_both(self):
        partial = CookingEnv(config("own_side", 1, False)).observe()
        np.testing.assert_array_equal(partial[0, 15:18], 0)
        np.testing.assert_array_equal(partial[1, 12:15], 0)
        self.assertEqual(partial[0, 14], 1)
        self.assertEqual(partial[1, 17], 1)
        full = CookingEnv(config("full", 1, False)).observe()
        self.assertTrue(np.all(full[:, [14,17]] == 1))
        np.testing.assert_array_equal(full[:, 96:], 0)

    def test_random_starts_and_action_history_are_reproducible(self):
        seen = [set(),set()]
        a, b = CookingEnv(config()), CookingEnv(config())
        for seed in range(128):
            np.testing.assert_array_equal(a.reset(seed=seed), b.reset(seed=seed))
            for i, agent in enumerate(a.world.agents):
                seen[i].add(agent.location)
                self.assertIn(agent.location[0], (1,2) if i==0 else (4,5))
        self.assertEqual(list(map(len, seen)), [12,12])

    def test_invalid_layout_dimension_and_schema_are_rejected(self):
        for changes in ({"observation_dim":105}, {"layout":"unknown"},
                        {"observation_schema":"pace_dense_full_93_plus_reserved12_v1"}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                replace(config(), **changes)
        with self.assertRaises(ValueError):
            EnvironmentConfig(observation_dim=108)


class EitherSideDeliveryTests(unittest.TestCase):
    def test_each_side_completes_all_144_starts_and_three_recipes_in_every_variant(self):
        cases = 0
        for visibility, limit in VARIANTS:
            env = CookingEnv(config(visibility, limit))
            for recipe in RECIPES:
                for lx in (1,2):
                    for ly in range(1,7):
                        for rx in (4,5):
                            for ry in range(1,7):
                                left = recipe_run(env, recipe, [(lx,ly),(rx,ry)], 0)
                                right = recipe_run(env, recipe, [(6-rx,ry),(6-lx,ly)], 1)
                                for server, result in enumerate((left,right)):
                                    self.assertTrue(result[0], (visibility,limit,recipe,lx,ly,rx,ry,server))
                                    self.assertEqual(result[1]["delivery_agent"], server)
                                    self.assertEqual(result[1]["recipe"], recipe)
                                    self.assertEqual(result[3], (0 if server==0 else 6,5))
                                self.assertEqual(left[2],right[2])
                                cases += 2
        self.assertEqual(cases,2592)

    def test_both_autonomous_experts_can_deliver_and_legacy_left_still_waits(self):
        winners = set()
        for seed in range(32):
            env = CookingEnv(config()); env.reset(seed=seed)
            peers = [ScriptedPeer(RECIPES[0],i) for i in (0,1)]
            for _ in range(40):
                _,_,done,truncated,info=env.step([p.act(env) for p in peers])
                if done or truncated: break
            self.assertTrue(done)
            winners.add(info["delivery_agent"])
        self.assertEqual(winners,{0,1})
        old = CookingEnv(EnvironmentConfig(random_start=False))
        complete_plate(old)
        self.assertEqual(ScriptedPeer(RECIPES[0],0).act(old),0)
        self.assertNotEqual(ScriptedPeer(RECIPES[0],1).act(old),0)

    def test_simultaneous_grab_has_seeded_fair_order_and_replays_exactly(self):
        winners = []
        for seed in range(128):
            outcomes=[]
            for _ in range(2):
                env=CookingEnv(config());env.reset(seed=seed)
                plate=complete_plate(env)
                for i,pos in enumerate(((2,3),(4,3))):
                    env.world.agents[i].move_to(pos)
                    env.world.agents[i].change_orientation(2 if i==0 else 1)
                obs,reward,_,_,info=env.step([5,5])
                owner=next(i for i,a in enumerate(env.world.agents) if a.holding is plate)
                arbitration=next(e for e in info["events"] if e["type"]=="shared_plate_contention")
                self.assertEqual(owner,arbitration["first_agent"])
                outcomes.append((owner,obs.tolist(),reward,info))
            self.assertEqual(outcomes[0],outcomes[1])
            winners.append(outcomes[0][0])
        self.assertTrue(40 <= winners.count(0) <= 88)

    def test_both_ports_reject_wrong_incomplete_and_extra_recipes(self):
        for port in ((0,5),(6,5)):
            for names in (("Lettuce",), ("Lettuce","Carrot"), ("Lettuce","Onion","Carrot")):
                env=CookingEnv(config());plate=env.world.world_objects["Plate"][0]
                for name in names:
                    food=env.world.world_objects[name][0];food.chop();plate.add_content(food);food.move_to(port)
                plate.move_to(port)
                self.assertFalse(env.step([0,0])[2])


class MirroredRunTests(unittest.TestCase):
    def test_three_new_configs_preserve_budgets_roles_and_old_hashes(self):
        hashes=[]
        for name,(visibility,limit) in zip(("full","single-ingredient","full-single-ingredient"),VARIANTS):
            old=load_config(ROOT/f"configs/l1-left-l0-serving-{name}-400u-s101-20261113.json")
            new=load_config(ROOT/f"configs/l1-mirrored-dual-delivery-{name}-400u-s101-20261115.json")
            self.assertEqual(new.environment.layout,MIRRORED_LAYOUT)
            self.assertEqual(new.environment.observation_dim,108)
            self.assertEqual((new.environment.visibility,new.environment.max_ingredients_per_agent),(visibility,limit))
            self.assertEqual(new.total_steps,28_800_000)
            self.assertEqual(new.total_steps//new.batch_size,400)
            self.assertEqual(new.ppo,old.ppo)
            self.assertEqual((new.l1_ego_side,new.ego_side),(0,1))
            self.assertEqual(list(new.construction_seeds),[101,102,103,104])
            self.assertTrue(new.environment.random_start)
            hashes.append(new.sha256)
        self.assertEqual(len(set(hashes)),3)
        expected=("6b50e2f1e0c22ef2f2a4482fd8e1bd409dfdf4fd9477588509c66ee706d34baf",
                  "f0bbdda67e4afe66a7bbc3db059e04198eae74dae963f73e8d030ac645a7e775",
                  "b74a3d2b47b25633d1a9dcc315ed2385fc8981c448ab1380cd26f63b3cff4001")
        for name,digest in zip(("full","single-ingredient","full-single-ingredient"),expected):
            self.assertEqual(load_config(ROOT/f"configs/l1-left-l0-serving-{name}-400u-s101-20261113.json").sha256,digest)

    def test_108_input_cpu_resume_is_bit_exact_and_cross_config_load_rejected(self):
        torch.set_num_threads(1)
        env=RunEnvironment(**asdict(replace(config(),horizon=7)))
        run=ExperimentConfig(environment=env,num_envs=2,rollout_steps=4,total_steps=24,
            ppo=PPOConfig(epochs=1,minibatches=2,bptt_steps=2),construction_seeds=(101,),validation_seeds=(111,),test_seeds=(121,),l1_ego_side=0,ego_side=1)
        with tempfile.TemporaryDirectory() as d,patch("builtins.print"):
            root=Path(d)
            full=train(run,root/"full","l1","cpu",seed=101)
            train(run,root/"resume","l1","cpu",seed=101,max_updates=1)
            resumed=train(run,root/"resume","l1","cpu",seed=101,resume=True)
            a=load_checkpoint(full,"cpu")[0];b=load_checkpoint(resumed,"cpu")[0]
            def equal(x,y):
                if isinstance(x,torch.Tensor):torch.testing.assert_close(x,y,rtol=0,atol=0)
                elif isinstance(x,dict):
                    self.assertEqual(x.keys(),y.keys())
                    for key in x:equal(x[key],y[key])
                elif isinstance(x,(tuple,list)):
                    self.assertEqual(len(x),len(y))
                    for p,q in zip(x,y):equal(p,q)
                else:self.assertEqual(x,y)
            equal(a,b)
            self.assertEqual(json.loads((full.parents[1]/"episodes.json").read_text()),
                             json.loads((resumed.parents[1]/"episodes.json").read_text()))
            policy,_=load_policy(run,full,torch.device("cpu"))
            self.assertEqual(policy.obs_dim,108)
            summary=evaluate(run,root/"evaluation",full,torch.device("cpu"),partner_level="l0",rounds=1,evaluation_seed=6001)
            self.assertEqual(summary["episodes"],15)
            self.assertEqual(set(summary["per_partner"]),{f"l0-1-{r}" for r in RECIPES})
            with self.assertRaises(ValueError):load_policy(replace(run,environment=RunEnvironment()),full,torch.device("cpu"))
