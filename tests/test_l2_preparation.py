"""L2 staging preserves L1 artifacts and rejects incompatible or altered inputs."""

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import torch

from nestrl_research.artifacts import file_hash
from nestrl_research.config import ROOT, load_config
from nestrl_research.pipeline import train

spec = importlib.util.spec_from_file_location("prepare_l2", ROOT / "scripts/prepare_l2.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class L2PreparationTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.cfg_path = ROOT / "configs/smoke.json"
        self.cfg = load_config(self.cfg_path)
        source = self.root / "source"
        with patch("builtins.print"):
            self.models = [train(self.cfg, source, "l1", "cpu", seed=s) for s in self.cfg.construction_seeds]
        self.hashes = [file_hash(p) for p in self.models]
        rows = []
        for seed, checkpoint in zip(self.cfg.construction_seeds, self.models):
            archive = self.root / f"seed-{seed}.zip"
            with zipfile.ZipFile(archive, "w") as z:
                for p in (source / f"l1-construction-{seed}").rglob("*"):
                    if p.is_file():
                        z.write(p, Path("nested") / p.relative_to(source))
                z.write(source / "registry.json", "nested/registry.json")
            rows.append({"seed": seed, "stage": "l1", "split": "construction",
                         "archive": archive.name, "archive_sha256": file_hash(archive),
                         "archive_checkpoint_path": f"nested/l1-construction-{seed}/checkpoints/final.pt",
                         "checkpoint_sha256": file_hash(checkpoint), "config_sha256": self.cfg.sha256,
                         "source_commit": "test-fixture"})
        self.index = self.root / "index.json"
        self.index.write_text(json.dumps({"models": rows}))

    def test_repeatable_staging_and_frozen_inference_leave_source_models_unchanged(self):
        target = self.root / "prepared"
        report = module.prepare(self.cfg_path, self.index, self.root, target)
        self.assertTrue(report["all_frozen"])
        self.assertEqual(report["partner_seeds"], list(self.cfg.construction_seeds))
        self.assertEqual(report, module.prepare(self.cfg_path, self.index, self.root, target))
        self.assertEqual(self.hashes, [file_hash(p) for p in self.models])

    def test_existing_different_model_is_never_overwritten(self):
        target = self.root / "prepared"
        module.prepare(self.cfg_path, self.index, self.root, target)
        path = target / "l1-construction-11/checkpoints/final.pt"
        path.write_bytes(b"local work")
        with self.assertRaisesRegex(ValueError, "will not overwrite"):
            module.prepare(self.cfg_path, self.index, self.root, target)
        self.assertEqual(path.read_bytes(), b"local work")

    def test_altered_archive_rejected_before_creating_target(self):
        archive = self.root / "seed-11.zip"
        with archive.open("ab") as stream:
            stream.write(b"altered")
        target = self.root / "not-created"
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            module.prepare(self.cfg_path, self.index, self.root, target)
        self.assertFalse(target.exists())

    def test_different_config_is_rejected_instead_of_relabelling_checkpoints(self):
        with self.assertRaisesRegex(ValueError, "Missing|Incompatible"):
            module.prepare(ROOT / "configs/paper-eight-l1.json", self.index, self.root, self.root / "bad")


if __name__ == "__main__":
    unittest.main()
