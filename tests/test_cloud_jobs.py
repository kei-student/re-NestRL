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
