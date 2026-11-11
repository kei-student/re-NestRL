"""Opt-in real Docker test: detached progress, abrupt stop, snapshot recovery."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import unittest

from nestrl_research.config import ROOT


@unittest.skipUnless(os.environ.get("NESTRL_DOCKER_TESTS") == "1", "set NESTRL_DOCKER_TESTS=1 for local Docker integration")
class CloudJobTests(unittest.TestCase):
    def test_l2_detached_worker_uses_frozen_l1_pool_and_resumes(self):
        from unittest.mock import patch
        import torch
        from nestrl_research.config import load_config
        from nestrl_research.pipeline import train
        from nestrl_research.artifacts import file_hash
        (ROOT / "outputs").mkdir(exist_ok=True)
        root = Path(tempfile.mkdtemp(prefix="l2-job-test-", dir=ROOT / "outputs"))
        cfg = load_config(ROOT / "configs/smoke.json")
        torch.set_num_threads(1)
        with patch("builtins.print"):
            checkpoints = [train(cfg, root, "l1", "cpu", seed=seed) for seed in cfg.construction_seeds]
        hashes = [file_hash(p) for p in checkpoints]
        base = [str(ROOT / ".venv/bin/python"), str(ROOT / "scripts/cloud_train.py")]
        options = ["--stage", "l2", "--config", str(ROOT / "configs/smoke.json"),
                   "--output-root", str(root), "--seeds", "41", "--memory-gib", "3", "--max-updates", "1"]
        environment = {k: v for k, v in os.environ.items() if k not in
                       ("DOCKER_HOST", "DOCKER_CONTEXT", "DOCKER_TLS", "DOCKER_TLS_VERIFY", "DOCKER_CERT_PATH")}
        name = None
        def call(action):
            return json.loads(subprocess.check_output(base + [action] + options, cwd=ROOT, text=True, timeout=30))
        def exited():
            deadline = time.monotonic() + 40
            while time.monotonic() < deadline:
                info = call("status")
                if info["state"] == "exited":
                    self.assertEqual(info["exit_code"], 0)
                    return info
                time.sleep(.1)
            self.fail("L2 worker did not stop")
        try:
            name = call("start")["container"]
            self.assertIn("-l2-41", name)
            self.assertEqual(exited()["latest_metrics"]["steps"], 80)
            run = root / "l2-41"
            metadata = json.loads((run / "run.json").read_text())
            self.assertEqual(metadata["stage"], "l2")
            self.assertEqual([p["id"] for p in metadata["training_partners"]],
                             [f"l1-construction-{seed}" for seed in cfg.construction_seeds])
            call("resume")
            self.assertTrue(exited()["complete"])
            self.assertEqual([file_hash(p) for p in checkpoints], hashes)
            self.assertTrue((root / "l2-seed-41.log").is_file())
        finally:
            if name:
                subprocess.run(["docker", "--host=unix:///var/run/docker.sock", "rm", "--force", name],
                               env=environment, capture_output=True, timeout=20)
            shutil.rmtree(root)

    def test_detached_progress_after_launcher_exit_and_abrupt_stop_resume(self):
        (ROOT / "outputs").mkdir(exist_ok=True)
        (ROOT / "tmp").mkdir(exist_ok=True)
        root = Path(tempfile.mkdtemp(prefix="recovery-test-", dir=ROOT / "outputs"))
        config = ROOT / "tmp" / f"{root.name}.json"
        value = json.loads((ROOT / "configs/smoke.json").read_text())
        value["total_steps"] = 3200
        config.write_text(json.dumps(value))
        base = [str(ROOT / ".venv/bin/python"), str(ROOT / "scripts/cloud_train.py")]
        options = ["--config", str(config), "--output-root", str(root), "--seeds", "11", "--memory-gib", "3"]
        docker_env = {k: v for k, v in os.environ.items() if k not in
                      ("DOCKER_HOST", "DOCKER_CONTEXT", "DOCKER_TLS", "DOCKER_TLS_VERIFY", "DOCKER_CERT_PATH")}
        docker = ["docker", "--host=unix:///var/run/docker.sock"]
        name = None
        def call(action):
            return subprocess.run(base + [action] + options, cwd=ROOT, check=True, capture_output=True,
                                  text=True, timeout=60).stdout
        def wait_for(predicate):
            deadline = time.monotonic() + 55
            while time.monotonic() < deadline:
                if predicate():
                    return
                time.sleep(0.1)
            self.fail("Docker training did not reach the expected state")
        directory = root / "l1-construction-11"
        pointer = directory / "resume/latest.json"
        try:
            launched = json.loads(call("start"))
            name = launched["container"]
            # The launch subprocess has exited. Only the daemon owns the job.
            wait_for(lambda: pointer.exists() and json.loads(pointer.read_text())["update"] >= 1)
            info = json.loads(call("status"))
            self.assertEqual(info["state"], "running")
            self.assertGreater(info["latest_metrics"]["steps"], 0)
            subprocess.run(docker + ["stop", "--time", "0", name], env=docker_env, check=True,
                           capture_output=True, text=True, timeout=15)
            committed = json.loads(pointer.read_text())["update"]
            self.assertLess(committed, 40)
            self.assertFalse((directory / "complete.json").exists())
            call("resume")
            wait_for(lambda: (directory / "complete.json").exists())
            info = json.loads(call("status"))
            # The final complete marker can appear just before container exit.
            wait_for(lambda: json.loads(call("status"))["state"] == "exited")
            info = json.loads(call("status"))
            self.assertEqual(info["exit_code"], 0)
            self.assertTrue(info["complete"])
            self.assertEqual(info["latest_metrics"]["steps"], 3200)
            events = [json.loads(line) for line in (directory / "resumes.jsonl").read_text().splitlines()]
            self.assertEqual(events[0]["from_update"], committed)
        finally:
            if name:
                subprocess.run(docker + ["rm", "--force", name], env=docker_env, capture_output=True, timeout=20)
            config.unlink(missing_ok=True)
            shutil.rmtree(root)


if __name__ == "__main__":
    unittest.main()
