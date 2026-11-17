"""Mixed frozen L0/L1 dispatch, round history and exact CPU recovery."""
import copy
from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from nestrl_research.config import ROOT, RECIPES, L2MixtureConfig, load_config
from nestrl_research.environment import CookingEnv
from nestrl_research.peers import ScriptedPeer
from nestrl_research.pipeline import make_model
from nestrl_research.rollout import FrozenPartners, RolloutCollector

spec = importlib.util.spec_from_file_location("create_coverage", ROOT/"scripts/create_l2_coverage_config.py")
creator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(creator)
STUDY = ROOT/"configs/l2-env3-coverage-1000u-20261117.json"


def equal(test, a, b):
    if isinstance(a, torch.Tensor):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    elif isinstance(a, np.ndarray):
        np.testing.assert_array_equal(a, b)
    elif isinstance(a, dict):
        test.assertEqual(set(a), set(b))
        for key in a:
            equal(test, a[key], b[key])
    elif isinstance(a, (list, tuple)):
        test.assertEqual(len(a), len(b))
        for left, right in zip(a, b):
            equal(test, left, right)
    else:
        test.assertEqual(a, b)


class CoverageSelectionTests(unittest.TestCase):
    def test_fresh_draws_replay_and_no_redraw_on_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch.object(creator, "fresh_selection_seed", side_effect=[7,8]) as entropy:
                a = creator.create(STUDY, 181, root/"a")
                b = creator.create(STUDY, 181, root/"b")
            self.assertEqual(entropy.call_count, 2)
            self.assertNotEqual(a["l1_partner_ids"], b["l1_partner_ids"])
            with patch.object(creator, "fresh_selection_seed", side_effect=AssertionError("redraw")):
                replay = creator.create(STUDY, 999, root/"replay", selection_seed=7)
                with self.assertRaises(FileExistsError):
                    creator.create(STUDY, 181, root/"a")
            self.assertEqual(a["l1_partner_ids"], replay["l1_partner_ids"])
            cfg = load_config(root/"a/config.json")
            self.assertEqual((cfg.total_steps, cfg.total_steps//cfg.batch_size), (72_000_000,1000))
            self.assertEqual(cfg.l2_mixture.l0_recipes, RECIPES)
            self.assertEqual(len(cfg.l1_pool.partner_ids), 3)
            self.assertEqual(len(a["partner_order"]), 6)
            self.assertEqual(a["level_probabilities"], {"l0":0.5,"l1":0.5})
            self.assertEqual(a["config_sha256"], cfg.sha256)
            self.assertFalse(a["training_started"])
            self.assertEqual(json.loads((root/"a/selection.json").read_text()), a)

    def test_legacy_hashes_and_invalid_mixture(self):
        base = load_config(ROOT/"configs/l2-env3-400u-k03-20261116.json")
        self.assertEqual(base.sha256, "05f5e9a413a8df02f7bf2e4981753a05692724995e125b3313e2b60919661417")
        self.assertNotIn("l2_mixture", base.to_dict())
        with self.assertRaisesRegex(ValueError, "authenticated"):
            replace(base, l1_pool=None, l2_mixture=L2MixtureConfig()).validate()
        for value in [L2MixtureConfig((),), L2MixtureConfig((RECIPES[0],RECIPES[0])),
                      L2MixtureConfig(("wrong",)), L2MixtureConfig(sampling="per_step")]:
            with self.assertRaises(ValueError):
                replace(base, l2_mixture=value).validate()
        with tempfile.TemporaryDirectory() as tmp:
            for learner, selection in [(-1,None),(181,-1)]:
                with self.assertRaises(ValueError):
                    creator.create(STUDY, learner, Path(tmp)/"bad", selection_seed=selection)
                self.assertFalse((Path(tmp)/"bad").exists())


class MixedRolloutTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        base = load_config(ROOT/"configs/l2-env3-400u-k03-20261116.json")
        self.cfg = replace(base, num_envs=6, rollout_steps=10, total_steps=120,
                           environment=replace(base.environment, horizon=2),
                           ppo=replace(base.ppo, epochs=1, minibatches=2, bptt_steps=2),
                           l2_mixture=L2MixtureConfig())
        torch.manual_seed(7)
        self.models = [make_model(self.cfg,"cpu") for _ in range(3)]
        self.learner = make_model(self.cfg,"cpu")

    def test_all_six_dispatch_to_correct_policy_and_only_rounds_redraw(self):
        peers = FrozenPartners(self.cfg,1,"cpu",np.random.default_rng(42),self.models,l0_recipes=RECIPES)
        peers.indices[:] = np.arange(6)
        peers.scripts = [peers._mixed_script(i) for i in peers.indices]
        peers.validate_mixed_state()
        envs = [CookingEnv(self.cfg.environment.__dict__) for _ in range(6)]
        observations = np.stack([e.reset(seed=i)[1] for i,e in enumerate(envs)])
        weights = copy.deepcopy([m.state_dict() for m in self.models])
        state = torch.get_rng_state()
        expected = []
        with torch.no_grad():
            for i, model in enumerate(self.models):
                expected.append(int(model.act(torch.as_tensor(observations[i:i+1]),
                    peers.previous[i:i+1],peers.starts[i:i+1],peers.hidden[:,i:i+1],
                    peers.resets[i:i+1])[0][0]))
        expected += [ScriptedPeer(r,agent_id=1).act(envs[i+3]) for i,r in enumerate(RECIPES)]
        torch.set_rng_state(state)
        actual = peers.act(observations, envs)
        np.testing.assert_array_equal(actual, expected)
        equal(self,weights,[m.state_dict() for m in self.models])
        self.assertTrue(all(not p.requires_grad and p.grad is None for m in self.models for p in m.parameters()))
        scripts = list(peers.scripts)
        peers.advance(actual,np.ones(6,dtype=bool),np.zeros(6,dtype=bool))
        np.testing.assert_array_equal(peers.indices,np.arange(6))
        self.assertEqual(peers.scripts,scripts)
        self.assertTrue(torch.all(peers.previous==-1))
        self.assertFalse(torch.any(peers.resets))
        seen = set()
        for _ in range(80):
            peers.advance(actual,np.ones(6,dtype=bool),np.ones(6,dtype=bool))
            peers.validate_mixed_state()
            seen.update(peers.indices.tolist())
        self.assertEqual(seen,set(range(6)))

    def test_five_episodes_hold_identity_and_mixed_cpu_snapshot_is_exact(self):
        collector = RolloutCollector(self.cfg,self.learner,0,100181,"cpu",self.models)
        initial = collector.partners.indices.copy()
        collector.collect()
        for env_index in range(6):
            episodes = [e for e in collector.episodes if e["env_index"]==env_index]
            self.assertEqual([e["episode_in_round"] for e in episodes],list(range(5)))
            self.assertEqual({e["partner_index"] for e in episodes},{int(initial[env_index])})
            self.assertEqual(len({e["partner_id"] for e in episodes}),1)
        self.assertTrue(torch.all(collector.resets))
        saved = copy.deepcopy(collector.state_dict())
        rng = torch.get_rng_state()
        expected = collector.collect()
        final_state = copy.deepcopy(collector.state_dict())
        restored = RolloutCollector(self.cfg,self.learner,0,999,"cpu",self.models)
        restored.load_state_dict(saved)
        torch.set_rng_state(rng)
        actual = restored.collect()
        equal(self,expected,actual)
        equal(self,final_state,restored.state_dict())
        bad = copy.deepcopy(saved)
        neural_index = next((i for i,v in enumerate(bad["partners"]["scripts"]) if v is None),None)
        if neural_index is not None:
            bad["partners"]["indices"][neural_index] = 3
            with self.assertRaisesRegex(ValueError,"snapshot differs"):
                restored.load_state_dict(bad)


if __name__ == "__main__":
    unittest.main()
