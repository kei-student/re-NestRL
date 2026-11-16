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
NAME = 'l1-left-env3-single-worker-400u-s151-20261116'

class Env3SingleWorkerTests(unittest.TestCase):
    def test_only_authorized_config_fields_change(self):
        path = ROOT/'configs/l1-left-l0-serving-full-single-ingredient-400u-s101-20261113.json'
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), 'f0c502b6e5fb9a83fd51ebb2081a5eab7d3a761eed217daad971a219fe636eb3')
        old = json.loads(path.read_text())
        new = json.loads((ROOT/f'configs/{NAME}.json').read_text())
        self.assertEqual({k for k in old.keys()|new.keys() if old.get(k)!=new.get(k)}, {'profile','construction_seeds'})
        self.assertEqual(new['profile'], NAME)
        self.assertEqual(new['construction_seeds'], [151])
        c = load_config(ROOT/f'configs/{NAME}.json')
        self.assertEqual((c.batch_size,c.total_steps,c.total_steps//c.batch_size),(72000,28800000,400))
        self.assertEqual(list(c.validation_seeds),[111])
        self.assertEqual(list(c.test_seeds),[121,122,123,124,125])
        self.assertEqual((c.l1_ego_side,c.ego_side,c.environment.visibility,c.environment.max_ingredients_per_agent),(0,1,'full',1))
        self.assertEqual((c.environment.random_start,c.environment.horizon,c.episodes_per_round,c.context_reset),(True,40,5,'round'))
        reg=registry_spec(c)
        self.assertEqual([p['seed'] for p in reg['l1'] if p['split']=='construction'],[151])
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
