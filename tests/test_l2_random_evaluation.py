"""Round sampling and exact single-partner equivalence to the native evaluator."""
import copy
from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from nestrl_research.artifacts import file_hash
from nestrl_research.config import ROOT, L1PoolConfig, load_config
from nestrl_research.evaluation import evaluate
from nestrl_research.pipeline import make_model, seed_all

spec = importlib.util.spec_from_file_location("random_l2_eval", ROOT / "scripts/evaluate_l2_random_partners.py")
driver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(driver)


class RandomPartnerEvaluationTests(unittest.TestCase):
    def test_schedule_is_reproducible_independent_and_allows_replacement(self):
        ids = ("a", "b", "c")
        first = driver.sample_partner_schedule(ids, 100, 71)
        self.assertEqual(first, driver.sample_partner_schedule(ids, 100, 71))
        self.assertNotEqual(first, driver.sample_partner_schedule(ids, 100, 72))
        self.assertEqual(set(first), set(ids))
        self.assertTrue(any(a == b for a, b in zip(first, first[1:])))
        for args in [((), 2, 1), (("a", "a"), 2, 1), (ids, 0, 1), (ids, True, 1), (ids, 2, -1)]:
            with self.assertRaises(ValueError):
                driver.sample_partner_schedule(*args)

    def test_single_partner_matches_native_traces_and_round_history_exactly(self):
        torch.set_num_threads(1)
        config = load_config(ROOT / "configs/l2-env3-400u-k15-20261116.json")
        pool = replace(config.l1_pool, partner_ids=(config.l1_pool.partner_ids[0],))
        seed_all(3)
        ego, peer = make_model(config, "cpu").eval(), make_model(config, "cpu").eval()
        peer.requires_grad_(False)
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            checkpoint, partner_checkpoint = root / "ego.pt", root / "peer.pt"
            checkpoint.write_bytes(b"fixture-ego")
            partner_checkpoint.write_bytes(b"fixture-peer")
            manifest = {"ego_side": 0, "run_id": "fixture", "training_partners": [{"id": pool.partner_ids[0]}]}
            provenance = [{"id": pool.partner_ids[0], "checkpoint": "peer.pt",
                           "checkpoint_sha256": file_hash(partner_checkpoint), "frozen": True}]
            for name in ("native", "random"):
                folder = root / name
                folder.mkdir()
                (folder / "peer.pt").write_bytes(partner_checkpoint.read_bytes())
            def ego_loader(*args):
                return copy.deepcopy(ego), manifest
            def peer_loader(*args, **kwargs):
                return [copy.deepcopy(peer)], provenance
            with patch("nestrl_research.evaluation.load_policy", side_effect=ego_loader), \
                 patch("nestrl_research.partner_pool.load_catalog_pool", side_effect=peer_loader):
                native = evaluate(config, root / "native", checkpoint, "cpu", partner_pool=pool,
                                  rounds=2, evaluation_seed=61, run_id="native")
            with patch.object(driver, "load_policy", side_effect=ego_loader), \
                 patch.object(driver, "load_catalog_pool", side_effect=peer_loader):
                random = driver.evaluate_random_partners(
                    config, root / "random", checkpoint, partner_pool=pool, rounds=2,
                    evaluation_seed=61, selection_seed=91, run_id="random")
            for filename in ("episodes.json", "traces.jsonl"):
                self.assertEqual((root / "native/native" / filename).read_bytes(),
                                 (root / "random/random" / filename).read_bytes())
            self.assertEqual(native["per_partner_counts"], random["per_partner_counts"])
            self.assertEqual(native["per_episode_position"], random["per_episode_position"])
            self.assertEqual(random["episodes"], 10)
            schedule = json.loads((root / "random/random/round_schedule.json").read_text())
            self.assertEqual(schedule["round_partner_ids"], [pool.partner_ids[0]] * 2)
            self.assertEqual(checkpoint.read_bytes(), b"fixture-ego")

if __name__ == "__main__":
    unittest.main()

