"""Scientific contract for the single-worker environment-3 replication."""
import hashlib
import json
from pathlib import Path
import unittest
import torch
from nestrl_research.config import load_config
from nestrl_research.pipeline import registry_spec
from nestrl_research.pipeline import make_model
from nestrl_research.rollout import RolloutCollector

ROOT = Path(__file__).resolve().parents[1]
NAME = 'l1-left-env3-12models-400u-s152-20261116'

class Env3TwelveModelsTests(unittest.TestCase):
    def test_only_authorized_config_fields_change(self):
        path = ROOT/'configs/l1-left-env3-single-worker-400u-s151-20261116.json'
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), 'ca98399fb95769ed54b7e24863c36a25065a0112e314a58f152be311c515aa9e')
        old = json.loads(path.read_text())
        new = json.loads((ROOT/f'configs/{NAME}.json').read_text())
        self.assertEqual({k for k in old.keys()|new.keys() if old.get(k)!=new.get(k)}, {'profile','construction_seeds'})
        self.assertEqual(new['profile'], NAME)
        self.assertEqual(new['construction_seeds'], list(range(152,164)))
        c = load_config(ROOT/f'configs/{NAME}.json')
        self.assertEqual((c.batch_size,c.total_steps,c.total_steps//c.batch_size),(72000,28800000,400))
        self.assertEqual(list(c.validation_seeds),[111])
        self.assertEqual(list(c.test_seeds),[121,122,123,124,125])
        self.assertEqual((c.l1_ego_side,c.ego_side,c.environment.visibility,c.environment.max_ingredients_per_agent),(0,1,'full',1))
        self.assertEqual((c.environment.random_start,c.environment.horizon,c.episodes_per_round,c.context_reset),(True,40,5,'round'))
        reg=registry_spec(c)
        self.assertEqual([p['seed'] for p in reg['l1'] if p['split']=='construction'],list(range(152,164)))
        self.assertEqual(len([p for p in reg['l0'] if p['side']==1 and p['frozen']]),3)

    def test_actual_collector_is_105_input_left_and_360_right_scripts(self):
        torch.set_num_threads(1)
        c=load_config(ROOT/f'configs/{NAME}.json')
        model=make_model(c,torch.device('cpu'))
        collector=RolloutCollector(c,model,0,100151,torch.device('cpu'))
        self.assertEqual(collector.observations.shape,(360,2,105))
        self.assertEqual(collector.observations[:,0].shape,(360,105))
        self.assertEqual(len(collector.partners.scripts),360)
        for peer in collector.partners.scripts:
            self.assertEqual(peer.agent_id,1)
            self.assertEqual(peer.random_action_probability,0)
        self.assertEqual({p.recipe for p in collector.partners.scripts},set(c.environment.recipes))

    def test_partition_covers_construction_once(self):
        p=json.loads((ROOT/f'docs/experiments/{NAME}-plan.json').read_text())
        assignments=p['task_assignments']
        self.assertEqual(len(assignments),6)
        self.assertTrue(all(len(t['seeds'])==2 for t in assignments))
        seeds=[s for t in assignments for s in t['seeds']]
        self.assertEqual(seeds,list(range(152,164)))
        self.assertEqual(len({t['output_root'] for t in assignments}),6)
        self.assertFalse(set(seeds)&{111,121,122,123,124,125})
