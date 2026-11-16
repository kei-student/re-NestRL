"""Each L2 gets a recorded independent draw; recovery never draws again."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from nestrl_research.artifacts import file_hash
from nestrl_research.config import ROOT, load_config

spec = importlib.util.spec_from_file_location('create_l2_config', ROOT/'scripts/create_l2_config.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
STUDY = ROOT/'configs/l2-env3-per-agent-20261116.json'


class PerAgentSelectionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_same_condition_and_learner_seed_draw_separately_for_each_l2(self):
        with patch.object(module.secrets,'randbits',side_effect=[7,8]) as entropy:
            a=module.create(STUDY,'k03',181,self.root/'agent-a')
            b=module.create(STUDY,'k03',181,self.root/'agent-b')
        self.assertEqual(entropy.call_count,2)
        self.assertEqual([a['selection_seed'],b['selection_seed']],[7,8])
        self.assertEqual(a['learner_seed'],b['learner_seed'])
        self.assertNotEqual(a['partner_ids'],b['partner_ids'])
        self.assertEqual(a['selection_seed_source'],'fresh OS entropy')
        for folder,record in [('agent-a',a),('agent-b',b)]:
            cfg=load_config(self.root/folder/'config.json')
            self.assertEqual(len(cfg.l1_pool.partner_ids),3)
            self.assertEqual(cfg.sha256,record['config_sha256'])
            self.assertEqual(file_hash(self.root/folder/'config.json'),record['config_file_sha256'])
            self.assertEqual(cfg.total_steps//cfg.batch_size,400)
            self.assertFalse(record['training_started'])

    def test_replay_reproduces_membership_without_consuming_new_entropy(self):
        with patch.object(module.secrets,'randbits',return_value=17):
            a=module.create(STUDY,'k05',181,self.root/'original')
        with patch.object(module.secrets,'randbits',side_effect=AssertionError('new entropy')):
            b=module.create(STUDY,'k05',999,self.root/'replay',selection_seed=a['selection_seed'])
        self.assertEqual(len(set(a['partner_ids'])),5)
        self.assertEqual(a['partner_ids'],b['partner_ids'])
        self.assertEqual(b['selection_seed_source'],'explicit replay seed')
        self.assertNotEqual(a['config_sha256'],b['config_sha256'])

    def test_existing_run_config_is_never_redrawn_or_overwritten(self):
        folder=self.root/'agent'
        module.create(STUDY,'k03',181,folder,selection_seed=7)
        before={p.name:p.read_bytes() for p in folder.iterdir()}
        with patch.object(module.secrets,'randbits',side_effect=AssertionError('must not draw')):
            with self.assertRaisesRegex(FileExistsError,'reuse its saved config'):
                module.create(STUDY,'k03',181,folder)
        self.assertEqual(before,{p.name:p.read_bytes() for p in folder.iterdir()})

    def test_all_partner_case_and_invalid_requests_preserve_experiment_contract(self):
        with patch.object(module.secrets,'randbits',side_effect=AssertionError('all does not draw')):
            record=module.create(STUDY,'k15',182,self.root/'all')
        study=json.loads(STUDY.read_text())
        self.assertEqual(record['partner_seeds'],study['candidate_seeds'])
        self.assertIsNone(record['selection_seed'])
        cfg=load_config(self.root/'all/config.json')
        self.assertEqual((cfg.environment.visibility,cfg.environment.max_ingredients_per_agent),('full',1))
        for condition,learner,selection in [('bad',181,None),('k03',-1,None),('k03',181,-1),('k15',181,7)]:
            path=self.root/'invalid'
            with self.assertRaises(ValueError):
                module.create(STUDY,condition,learner,path,selection_seed=selection)
            self.assertFalse(path.exists())


if __name__ == '__main__':
    unittest.main()
