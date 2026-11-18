"""Author source fidelity, lifecycle, action mapping and frozen evaluator integration."""
from dataclasses import asdict
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
from nestrl_research.config import ROOT, load_config
from nestrl_research.environment import CookingEnv, RECIPES
from nestrl_research.peers import ScriptedPeer
from nestrl_research.pipeline import make_model, seed_all
spec=importlib.util.spec_from_file_location("author_l0_driver",ROOT/"scripts/evaluate_l2_author_l0.py")
driver=importlib.util.module_from_spec(spec)
spec.loader.exec_module(driver)

class SourceGuardTests(unittest.TestCase):
    def test_wrong_source_bytes_rejected_before_execution(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/"policy.py"; z=Path(td)/"zoo.py"
            p.write_text("raise RuntimeError('must never run')")
            z.write_text("Ingred2ID={}")
            with self.assertRaisesRegex(ValueError,"source bytes"):
                driver.load_author(p,z)

@unittest.skipUnless(os.environ.get("RE_NESTRL_AUTHOR_SOURCE_ROOT"),"Set pinned author source root for integration")
class AuthorIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root=Path(os.environ["RE_NESTRL_AUTHOR_SOURCE_ROOT"])/"environment/overcooked"
        cls.policy=cls.root/"policy.py"
        cls.zoo=cls.root/"gym_cooking/environment/cooking_zoo.py"
        cls.namespace=driver.load_author(cls.policy,cls.zoo)
        torch.set_num_threads(1)

    def test_factory_matches_recipe_support_weights_and_preserves_global_rng(self):
        np.random.seed(91); state=np.random.get_state()
        pool=driver.author_pool(self.namespace,0.25)
        after=np.random.get_state()
        self.assertTrue(np.array_equal(state[1],after[1]))
        self.assertEqual(state[2:],after[2:])
        self.assertEqual(list(pool),list(RECIPES))
        zero=driver.author_pool(self.namespace,0)
        for recipe,p in pool.items():
            self.assertEqual(p.agent_id,1)
            self.assertTrue(np.array_equal(p.event_probs,zero[recipe].event_probs))
            self.assertAlmostEqual(float(p.event_probs.sum()),1)
            self.assertEqual((p.nav_p,p.tar_p,p.rand_p),(0.25,0.25,0.25))
        with self.assertRaises(ValueError):
            driver.author_pool(self.namespace,0.1)

    def test_author_reset_retains_original_event_instead_of_adapter_refresh(self):
        p=driver.author_pool(self.namespace,0)["LettuceOnionSalad"]
        marker=self.namespace["event"]("Take","Onion")
        p.cur_event=marker;p.reset()
        self.assertIs(p.cur_event,marker)

    def test_all_recipes_work_with_native_world_classes_and_actions(self):
        for recipe,p in driver.author_pool(self.namespace,0).items():
            successes=0
            for index in range(10):
                np.random.seed(6001+index)
                env=CookingEnv({"visibility":"full","max_ingredients_per_agent":1,
                    "observation_schema":"pace_dense_full_93_plus_ingredient_locks12_v1"})
                env.reset(seed=6001+index);p.reset()
                left=ScriptedPeer(recipe,0)
                for step in range(40):
                    a=int(p(env.world))
                    self.assertIn(a,range(6))
                    _,_,term,trunc,info=env.step([left.act(env),a])
                    if term or trunc:
                        break
                successes+=bool(info["success"])
            self.assertEqual(successes,10,recipe)

    def test_evaluator_reproducible_preserves_inputs_and_rejects_corrupt_trace(self):
        config=load_config(ROOT/"configs/l2-env3-400u-k03-20261116.json")
        seed_all(31); model=make_model(config,"cpu").eval()
        manifest={"ego_side":0,"steps":28800000}
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);(root/"learner/checkpoints").mkdir(parents=True)
            checkpoint=root/"learner/checkpoints/final.pt";checkpoint.write_bytes(b"frozen-fixture")
            (root/"config.json").write_text(json.dumps(asdict(config)))
            before=driver.tensor_digest(model)
            with patch.object(driver,"load_policy",return_value=(model,manifest)):
                for rid,p in (("first",0),("repeat",0),("noise",0.25)):
                    driver.evaluate_author(config,root,checkpoint,self.policy,self.zoo,
                        p=p,run_id=rid,rounds=1,evaluation_seed=6001)
            self.assertEqual(driver.tensor_digest(model),before)
            self.assertEqual((root/"first/traces.jsonl").read_bytes(),(root/"repeat/traces.jsonl").read_bytes())
            self.assertEqual((root/"first/episodes.json").read_bytes(),(root/"repeat/episodes.json").read_bytes())
            self.assertNotEqual((root/"first/traces.jsonl").read_bytes(),(root/"noise/traces.jsonl").read_bytes())
            self.assertEqual(driver.verify_run(root/"first")["episodes"],15)
            path=root/"first/traces.jsonl";lines=path.read_text().splitlines()
            row=json.loads(lines[-1]);row["reward"]+=1;lines[-1]=json.dumps(row)
            path.write_text("\n".join(lines)+"\n")
            with self.assertRaisesRegex(ValueError,"Trace/episode"):
                driver.verify_run(root/"first")

if __name__=="__main__":
    unittest.main()

