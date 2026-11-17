"""Budget-only state fork must match fresh execution with the smaller budget."""
from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import torch
import test_partner_pool as fixtures
from test_l2_coverage import equal
from nestrl_research.artifacts import file_hash
from nestrl_research.config import ROOT, L2MixtureConfig, canonical_hash, load_config
from nestrl_research.partner_pool import stage_catalog
from nestrl_research.pipeline import train
from nestrl_research.recovery import training_run

spec=importlib.util.spec_from_file_location("budget_reduction",ROOT/"scripts/prepare_l2_budget_reduction.py")
driver=importlib.util.module_from_spec(spec);spec.loader.exec_module(driver)


class BudgetReductionTests(unittest.TestCase):
    def setUp(self):
        self.fixture=fixtures.CatalogPoolTests(methodName="runTest")
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        f=self.fixture
        self.cfg=replace(f.cfg,total_steps=48,l2_mixture=L2MixtureConfig())
        f.stage(self.cfg)
        with patch("builtins.print"):
            train(self.cfg,f.target,"l2","cpu",seed=181,checkpoint_every=1,max_updates=2)
        self.parent=f.target/"l2-181"
        self.pointer=json.loads((self.parent/"resume/latest.json").read_text())

    def prepare(self,**changes):
        kwargs={"parent_root":self.fixture.target,"parent_run_id":"l2-181",
                "child_root":self.fixture.root/"child","child_run_id":"l2-181-budget4",
                "target_updates":4,"expected_config_sha256":self.cfg.sha256,
                "expected_snapshot_sha256":self.pointer["sha256"]}
        kwargs.update(changes)
        return driver.prepare(**kwargs)

    def test_budget_change_preserves_parent_and_exact_full_training_state(self):
        f=self.fixture
        before={p:file_hash(p) for p in f.target.rglob("*") if p.is_file()}
        lineage=self.prepare()
        self.assertEqual(lineage["parent_saved_update"],2)
        self.assertEqual(lineage["remaining_updates"],2)
        self.assertTrue(lineage["state_equal_except_config_registry_run_metadata"])
        self.assertEqual(before,{p:file_hash(p) for p in f.target.rglob("*") if p.is_file()})
        child=f.root/"child";cfg=load_config(child/"config.json")
        old,new=self.cfg.to_dict(),cfg.to_dict();old.pop("total_steps");new.pop("total_steps")
        self.assertEqual(canonical_hash(old),canonical_hash(new))
        self.assertEqual(cfg.total_steps,32)
        reference=f.root/"reference"
        stage_catalog(cfg,f.catalog,f.archives,reference)
        with patch("builtins.print"):
            train(cfg,child,"l2","cpu",seed=181,run_id="l2-181-budget4",checkpoint_every=1,resume=True)
            train(cfg,reference,"l2","cpu",seed=181,run_id="l2-181-budget4",checkpoint_every=1)
        states=[]
        for root in [child,reference]:
            folder=root/"l2-181-budget4/resume"
            pointer=json.loads((folder/"latest.json").read_text())
            states.append(torch.load(folder/pointer["file"],map_location="cpu",weights_only=True))
            self.assertEqual(pointer["update"],4)
            self.assertEqual(json.loads((root/"l2-181-budget4/complete.json").read_text())["steps"],32)
        for key in ("model","optimizer","collector","rng"):
            equal(self,states[0][key],states[1][key])

    def test_active_writer_wrong_hash_and_past_target_are_rejected(self):
        with training_run(self.fixture.target,"l2-181",resume=True):
            with self.assertRaisesRegex(RuntimeError,"active writer"):
                self.prepare()
        for values in [{"expected_snapshot_sha256":"0"*64},{"expected_config_sha256":"0"*64},
                       {"target_updates":1},{"target_updates":6}]:
            with self.assertRaises(ValueError):
                self.prepare(**values)
        self.assertFalse((self.fixture.root/"child").exists())

    def test_existing_child_is_never_overwritten_or_redrawn(self):
        self.prepare()
        root=self.fixture.root/"child"
        before={p:file_hash(p) for p in root.rglob("*") if p.is_file()}
        with self.assertRaises(FileExistsError):
            self.prepare()
        self.assertEqual(before,{p:file_hash(p) for p in root.rglob("*") if p.is_file()})

    def test_snapshot_at_target_finalizes_without_an_extra_update(self):
        f=self.fixture
        lineage=self.prepare(target_updates=2)
        self.assertEqual(lineage["remaining_updates"],0)
        child=f.root/"child";cfg=load_config(child/"config.json")
        with patch("builtins.print"):
            train(cfg,child,"l2","cpu",seed=181,run_id="l2-181-budget4",checkpoint_every=1,resume=True)
        pointer=json.loads((child/"l2-181-budget4/resume/latest.json").read_text())
        self.assertEqual(pointer["update"],2)
        self.assertTrue((child/"l2-181-budget4/complete.json").is_file())


if __name__=="__main__":
    unittest.main()
