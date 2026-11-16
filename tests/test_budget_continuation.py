"""Budget extension must equal uninterrupted CPU training without editing its parent."""
from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from nestrl_research import pipeline
from nestrl_research.artifacts import file_hash
from nestrl_research.config import ExperimentConfig, EnvironmentConfig, PPOConfig

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/prepare_budget_continuation.py"
spec = importlib.util.spec_from_file_location("budget_continuation", SCRIPT)
continuation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(continuation)


class BudgetContinuationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.config = ExperimentConfig(environment=EnvironmentConfig(horizon=7), num_envs=2,
            rollout_steps=4, total_steps=16,
            ppo=PPOConfig(epochs=1, minibatches=2, bptt_steps=2),
            construction_seeds=(101,), validation_seeds=(201,), test_seeds=(301,))
        self.parent_root = self.root / "parent"
        with patch("builtins.print"):
            pipeline.train(self.config, self.parent_root, "generalist", torch.device("cpu"),
                           run_id="generalist-401", checkpoint_every=1)
        self.parent = self.parent_root / "generalist-401"
        pointer = json.loads((self.parent / "resume/latest.json").read_text())
        self.args = dict(parent_root=self.parent_root, parent_run_id="generalist-401",
            child_root=self.root / "child", child_run_id="generalist-401-u4-from2",
            target_updates=4, expected_config_sha256=self.config.sha256,
            expected_snapshot_sha256=pointer["sha256"])

    def test_extension_matches_uninterrupted_model_optimizer_gru_collector_and_rng(self):
        before = {p.relative_to(self.parent_root).as_posix():file_hash(p)
                  for p in self.parent_root.rglob("*") if p.is_file()}
        lineage = continuation.prepare(**self.args)
        after = {p.relative_to(self.parent_root).as_posix():file_hash(p)
                 for p in self.parent_root.rglob("*") if p.is_file()}
        self.assertEqual(before, after)
        self.assertEqual(lineage["additional_updates"], 2)
        child_root = self.args["child_root"]
        child_config = continuation.load_config(child_root / "config.json")
        with patch("builtins.print"):
            pipeline.train(child_config, child_root, "generalist", torch.device("cpu"),
                           run_id=self.args["child_run_id"], resume=True, checkpoint_every=1)
            pipeline.train(replace(self.config,total_steps=32), self.root/"full", "generalist",
                           torch.device("cpu"),run_id="full",checkpoint_every=1)
        child = child_root/self.args["child_run_id"]
        full = self.root/"full/full"
        states = []
        for directory in (child, full):
            ptr=json.loads((directory/"resume/latest.json").read_text())
            self.assertEqual(ptr["update"],4)
            states.append(torch.load(directory/"resume"/ptr["file"],weights_only=True))
        for key in ("model","optimizer","collector","rng","update"):
            continuation.assert_same(states[0][key],states[1][key])
        metrics=[json.loads(l) for l in (child/"metrics.jsonl").read_text().splitlines()]
        self.assertEqual([m["update"] for m in metrics],[1,2,3,4])
        self.assertEqual(json.loads((child/"complete.json").read_text())["steps"],32)

    def test_wrong_snapshot_and_config_are_rejected_without_child(self):
        for key in ("expected_snapshot_sha256","expected_config_sha256"):
            with self.subTest(key=key), self.assertRaises(ValueError):
                continuation.prepare(**{**self.args,key:"0"*64})
            self.assertFalse(self.args["child_root"].exists())

    def test_non_increasing_budget_and_existing_destination_are_rejected(self):
        for updates in (0,1,2,True):
            with self.subTest(updates=updates), self.assertRaises(ValueError):
                continuation.prepare(**{**self.args,"target_updates":updates})
        self.args["child_root"].mkdir()
        with self.assertRaises(FileExistsError):
            continuation.prepare(**self.args)

    def test_runtime_mismatch_is_rejected(self):
        runtime=continuation.runtime_metadata("cpu")
        with patch.object(continuation,"runtime_metadata",return_value={**runtime,"threads":9}):
            with self.assertRaisesRegex(ValueError,"runtime mismatch: threads"):
                continuation.prepare(**self.args)
        self.assertFalse(self.args["child_root"].exists())

    def test_active_parent_writer_is_rejected(self):
        with continuation.parent_lock(self.parent):
            with self.assertRaises((RuntimeError,OSError)):
                continuation.prepare(**self.args)
        self.assertFalse(self.args["child_root"].exists())

    def test_l2_catalog_under_partners_and_selected_partner_order_are_preserved(self):
        from test_partner_pool import CatalogPoolTests
        fixture = CatalogPoolTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        cfg = replace(fixture.cfg,l1_pool=replace(fixture.cfg.l1_pool,partner_ids=("l1-construction-151",)))
        fixture.stage(cfg)
        with patch("builtins.print"):
            pipeline.train(cfg,fixture.target,"l2","cpu",seed=181,checkpoint_every=1)
        parent=fixture.target/"l2-181"
        ptr=json.loads((parent/"resume/latest.json").read_text())
        root=fixture.root/"continuation"
        continuation.prepare(fixture.target,"l2-181",root,"l2-181-u4-from2",4,cfg.sha256,ptr["sha256"])
        self.assertEqual(file_hash(root/"partners/catalog.json"),cfg.l1_pool.catalog_sha256)
        new_cfg=continuation.load_config(root/"config.json")
        with patch("builtins.print"):
            checkpoint=pipeline.train(new_cfg,root,"l2","cpu",seed=181,
                run_id="l2-181-u4-from2",resume=True,checkpoint_every=1)
        manifest=json.loads(checkpoint.with_suffix(".json").read_text())
        self.assertEqual([p["seed"] for p in manifest["training_partners"]],[151])
        for p in (fixture.target/"partners").rglob("*"):
            if p.is_file():
                self.assertEqual(file_hash(p),file_hash(root/p.relative_to(fixture.target)))
