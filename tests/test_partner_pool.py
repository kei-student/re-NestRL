"""Mixed-cohort provenance, frozen training, recovery and explicit diagnostics."""
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import numpy as np
import torch

from nestrl_research.artifacts import atomic_json, file_hash, save_checkpoint
from nestrl_research.config import ROOT, L1PoolConfig, L2MixtureConfig, canonical_hash, load_config
from nestrl_research.environment import observation_schema_for
from nestrl_research.evaluation import evaluate
from nestrl_research.partner_pool import ARTIFACT_FILES, CATALOG_SCHEMA, load_catalog_pool, select_partner_ids, stage_catalog
from nestrl_research.pipeline import make_model, registry_spec, train
from nestrl_research.provenance import source_metadata
from nestrl_research.rollout import FrozenPartners


class CatalogPoolTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.archives = self.root / 'archives'; self.archives.mkdir()
        base = load_config(ROOT / 'configs/l1-random-start-400u-full-single-ingredient-20agents-s151-20261115.json')
        self.source_configs = []
        rows = []
        for seed in (101, 151):
            cfg = replace(base, profile=f'fixture-source-{seed}', construction_seeds=(seed,))
            self.source_configs.append(cfg)
            folder = self.root / f'source-{seed}' / f'l1-construction-{seed}'
            folder.mkdir(parents=True)
            registry = registry_spec(cfg)
            atomic_json(folder.parent / 'registry.json', registry)
            metadata = {**source_metadata(), 'stage':'l1', 'split':'construction', 'seed':seed,
                        'ego_side':1, 'run_id':folder.name, 'config_sha256':cfg.sha256,
                        'registry_sha256':canonical_hash(registry),
                        'training_partners':[{'id':f'l0-0-{r}','frozen':True} for r in cfg.environment.recipes]}
            atomic_json(folder / 'run.json', metadata)
            atomic_json(folder / 'config.json', cfg.to_dict())
            model = make_model(cfg, 'cpu')
            checkpoint = save_checkpoint(folder, model, torch.optim.Adam(model.parameters()),
                                         {**metadata, 'steps':cfg.total_steps})
            atomic_json(folder / 'complete.json', {'sha256':file_hash(checkpoint), 'steps':cfg.total_steps})
            archive = self.archives / f'{seed}.zip'
            members, files = {}, {}
            with zipfile.ZipFile(archive, 'w') as z:
                for name in ARTIFACT_FILES:
                    source = folder.parent / name if name == 'registry.json' else folder / name
                    member = ('nested/registry.json' if name == 'registry.json' else f'nested/{folder.name}/{name}')
                    z.write(source, member); members[name] = member; files[name] = file_hash(source)
            rows.append({'id':folder.name, 'seed':seed, 'stage':'l1', 'split':'construction', 'ego_side':1,
                         'steps':cfg.total_steps, 'updates':400, 'config_sha256':cfg.sha256, 'files':files,
                         'archive':{'filename':archive.name,'sha256':file_hash(archive),'members':members},
                         'qualification':{'episodes_per_recipe':1000,'successes':{r:1000 for r in cfg.environment.recipes}}})
        self.catalog = self.root / 'catalog.json'
        atomic_json(self.catalog, {'schema':CATALOG_SCHEMA,'models':rows})
        self.cfg = replace(base, profile='fixture-l2', construction_seeds=(101,151),
                           num_envs=2, rollout_steps=4, total_steps=16,
                           ppo=replace(base.ppo,epochs=1,minibatches=2,bptt_steps=2),
                           l1_pool=L1PoolConfig('partners/catalog.json',file_hash(self.catalog),tuple(r['id'] for r in rows)))
        self.target = self.root / 'prepared'

    def stage(self, cfg=None):
        return stage_catalog(cfg or self.cfg,self.catalog,self.archives,self.target)

    def test_original_configs_and_registries_are_preserved_and_models_are_frozen(self):
        self.assertNotEqual(self.source_configs[0].sha256,self.source_configs[1].sha256)
        before = {p:file_hash(p) for p in self.archives.iterdir()}
        models, provenance = self.stage()
        self.assertEqual([p['seed'] for p in provenance],[101,151])
        self.assertEqual([p['source_config_sha256'] for p in provenance],[c.sha256 for c in self.source_configs])
        weights = [{k:v.clone() for k,v in m.state_dict().items()} for m in models]
        peers = FrozenPartners(self.cfg,1,'cpu',np.random.default_rng(4),models)
        actions = peers.act(np.zeros((2,105),dtype=np.float32),[])
        self.assertTrue(np.all((0<=actions)&(actions<6)))
        for model, original in zip(models,weights):
            self.assertFalse(model.training)
            self.assertTrue(all(not p.requires_grad and p.grad is None for p in model.parameters()))
            for name,value in model.state_dict().items():
                torch.testing.assert_close(value,original[name],rtol=0,atol=0)
        self.assertEqual(before,{p:file_hash(p) for p in self.archives.iterdir()})
        self.assertEqual(provenance,self.stage()[1])

    def test_l2_uses_only_selected_partners_and_exact_cpu_recovery_remains_supported(self):
        cfg = replace(self.cfg,l1_pool=replace(self.cfg.l1_pool,partner_ids=('l1-construction-151',)))
        self.stage(cfg)
        source_hashes = {p:file_hash(p) for p in (self.target/'partners').rglob('*') if p.is_file()}
        with patch('builtins.print'):
            train(cfg,self.target,'l2','cpu',seed=181,checkpoint_every=1,max_updates=1)
            checkpoint = train(cfg,self.target,'l2','cpu',seed=181,checkpoint_every=1,resume=True)
        manifest = json.loads(checkpoint.with_suffix('.json').read_text())
        self.assertEqual([x['seed'] for x in manifest['training_partners']],[151])
        self.assertEqual(manifest['ego_side'],0)
        self.assertEqual(source_hashes,{p:file_hash(p) for p in (self.target/'partners').rglob('*') if p.is_file()})
        summary = evaluate(cfg,self.target,checkpoint,'cpu',rounds=1,run_id='common-cohort',
                           partner_pool=self.cfg.l1_pool)
        self.assertEqual(summary['protocol']['partition'],'construction_diagnostic')
        self.assertEqual([p['seen_in_l2_training'] for p in summary['protocol']['partners']],[False,True])
        self.assertEqual(summary['episodes'],10)
        with self.assertRaisesRegex(ValueError,'held-out'):
            evaluate(cfg,self.target,checkpoint,'cpu',rounds=1,run_id='not-held-out')
        with self.assertRaisesRegex(ValueError,'only valid'):
            train(cfg,self.target,'l1','cpu',seed=101)

    def test_different_environment_and_round_history_are_rejected(self):
        self.stage()
        other = replace(self.cfg,environment=replace(self.cfg.environment,visibility='own_side',
                        observation_schema=observation_schema_for('own_side',1)))
        for cfg in [other,replace(self.cfg,episodes_per_round=4)]:
            with self.assertRaisesRegex(ValueError,'contract differs'):
                load_catalog_pool(cfg,self.target,'cpu')

    def test_mixed_frozen_pool_training_metadata_and_exact_resume(self):
        from test_l2_coverage import equal
        cfg = replace(self.cfg, num_envs=6, rollout_steps=10, total_steps=120,
                      l2_mixture=L2MixtureConfig())
        self.stage(cfg)
        before={p:file_hash(p) for p in (self.target/'partners').rglob('*') if p.is_file()}
        full=self.root/'full'
        import importlib.util
        spec=importlib.util.spec_from_file_location('prepare_mixed',ROOT/'scripts/prepare_l2.py')
        prepare_module=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(prepare_module)
        config_path=self.root/'mixed.json';atomic_json(config_path,cfg.to_dict())
        prepared=prepare_module.prepare(config_path,None,self.archives,full,catalog_path=self.catalog)
        self.assertEqual(prepared['partner_count'],5)
        self.assertEqual(prepared['l0_partner_count'],3)
        self.assertEqual(prepared['l1_partner_count'],2)
        self.assertEqual(prepared['real_environment_dispatch'],'passed')
        self.assertFalse(prepared['training_started'])
        with patch('builtins.print'):
            train(cfg,self.target,'l2','cpu',seed=181,checkpoint_every=1,max_updates=1)
            resumed=train(cfg,self.target,'l2','cpu',seed=181,checkpoint_every=1,resume=True)
            train(cfg,full,'l2','cpu',seed=181,checkpoint_every=1)
        manifest=json.loads(resumed.with_suffix('.json').read_text())
        self.assertEqual([p['id'] for p in manifest['training_partners']],
                         ['l1-construction-101','l1-construction-151']+
                         [f'l0-1-{r}' for r in cfg.environment.recipes])
        self.assertEqual(manifest['partner_sampling']['probability_per_partner'],1/5)
        states=[]
        for directory in [self.target,full]:
            folder=directory/'l2-181/resume'
            pointer=json.loads((folder/'latest.json').read_text())
            states.append(torch.load(folder/pointer['file'],map_location='cpu',weights_only=True))
        for key in ['model','optimizer','collector','rng']:
            equal(self,states[0][key],states[1][key])
        self.assertEqual(before,{p:file_hash(p) for p in (self.target/'partners').rglob('*') if p.is_file()})
        with self.assertRaisesRegex(ValueError,'only valid'):
            train(cfg,self.root/'not-l1','l1','cpu',seed=101)

    def test_tampered_artifact_catalog_and_mismatched_source_config_are_rejected(self):
        self.stage()
        path = self.target/'partners/l1-construction-101/checkpoints/final.pt'
        original = path.read_bytes();path.write_bytes(original+b'altered')
        with self.assertRaisesRegex(ValueError,'SHA-256'):
            load_catalog_pool(self.cfg,self.target,'cpu')
        path.write_bytes(original)
        catalog = self.target/'partners/catalog.json';catalog.write_bytes(catalog.read_bytes()+b'\n')
        with self.assertRaisesRegex(ValueError,'catalog SHA-256'):
            load_catalog_pool(self.cfg,self.target,'cpu')

    def test_self_consistent_hashes_do_not_allow_a_test_policy_to_be_relabelled(self):
        self.stage()
        folder = self.target/'partners/l1-construction-101'
        for name in ['run.json','checkpoints/final.json']:
            path = folder/name;value=json.loads(path.read_text());value['split']='test';atomic_json(path,value)
        atomic_json(folder/'checkpoints/latest.json',{'manifest':'final.json','manifest_sha256':file_hash(folder/'checkpoints/final.json')})
        catalog_path=self.target/'partners/catalog.json';catalog=json.loads(catalog_path.read_text())
        for name in ARTIFACT_FILES:catalog['models'][0]['files'][name]=file_hash(folder/name)
        atomic_json(catalog_path,catalog)
        cfg=replace(self.cfg,l1_pool=replace(self.cfg.l1_pool,catalog_sha256=file_hash(catalog_path)))
        with self.assertRaisesRegex(ValueError,'manifest role'):
            load_catalog_pool(cfg,self.target,'cpu')

    def test_archive_corruption_and_existing_conflicts_never_overwrite(self):
        self.stage()
        path = self.target/'partners/l1-construction-101/run.json';path.write_bytes(b'preserve me')
        with self.assertRaisesRegex(ValueError,'will not overwrite'):
            self.stage()
        self.assertEqual(path.read_bytes(),b'preserve me')
        with (self.archives/'101.zip').open('ab') as f:f.write(b'altered')
        with self.assertRaisesRegex(ValueError,'archive SHA-256'):
            stage_catalog(self.cfg,self.catalog,self.archives,self.root/'uncreated')
        self.assertFalse((self.root/'uncreated').exists())


class FrozenStudyContractTests(unittest.TestCase):
    def test_independent_draws_budget_and_legacy_hashes(self):
        study=json.loads((ROOT/'configs/l2-env3-experiment-20261116.json').read_text())
        candidates=[f'l1-construction-{n}' for n in study['candidate_seeds']]
        controls=[]
        for condition in study['conditions']:
            cfg=load_config(ROOT/condition['config']);self.assertEqual(cfg.sha256,condition['config_sha256'])
            self.assertEqual(cfg.total_steps//cfg.batch_size,400)
            self.assertEqual((cfg.environment.visibility,cfg.environment.max_ingredients_per_agent),('full',1))
            self.assertTrue(cfg.environment.random_start)
            self.assertEqual(cfg.learner_seed,181)
            if condition['selection_seed'] is not None:
                self.assertEqual(tuple(cfg.l1_pool.partner_ids),select_partner_ids(candidates,condition['partner_count'],condition['selection_seed']))
            self.assertEqual(len(cfg.l1_pool.partner_ids),condition['partner_count'])
            value=cfg.to_dict();value.pop('profile');value.pop('l1_pool');controls.append(value)
        self.assertEqual(controls[0],controls[1]);self.assertEqual(controls[1],controls[2])
        self.assertEqual(load_config(ROOT/'configs/l1-random-start-400u-full-single-ingredient-20agents-s151-20261115.json').sha256,
                         '9327dbb49f721f6aefc122c04d97c13c2e866acc36fdf4ed5a50eb97e3cc1658')
        with self.assertRaisesRegex(ValueError,'unique'):
            select_partner_ids([candidates[0],candidates[0]],1,7)
        with self.assertRaisesRegex(ValueError,'portable'):
            L1PoolConfig('../escape.json','0'*64,(candidates[0],)).validate()
        with self.assertRaisesRegex(ValueError,'unique'):
            L1PoolConfig('catalog.json','0'*64,()).validate()


if __name__ == '__main__':
    unittest.main()
