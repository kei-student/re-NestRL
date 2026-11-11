from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class RunRecordingTests(unittest.TestCase):
    def test_records_failure_and_refuses_to_overwrite_original_run(self):
        record = load_script("record_run")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            subprocess.run(["git", "init", "-q"], cwd=root, check=True)
            subprocess.run(["git", "-c", "user.name=Test", "-c", "user.email=test@example.invalid",
                            "commit", "--allow-empty", "-qm", "baseline"], cwd=root, check=True)
            config = root / "experiment.json"
            config.write_text('{"updates": 7}\n')
            output = root / "run"
            code = record.record_run(root, output, "trial-7", config,
                                     [sys.executable, "-c", "print('trial'); raise SystemExit(3)"])
            self.assertEqual(code, 3)
            manifest = json.loads((output / "run.json").read_text())
            self.assertEqual(manifest["state"], "failed")
            self.assertEqual(json.loads((output / "config.json").read_text()), {"updates": 7})
            original = (output / "run.json").read_bytes()
            with self.assertRaises(FileExistsError):
                record.record_run(root, output, "different", config, [sys.executable, "-V"])
            self.assertEqual((output / "run.json").read_bytes(), original)

    def test_detects_configuration_changed_by_the_command(self):
        record = load_script("record_run")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            subprocess.run(["git", "init", "-q"], cwd=root, check=True)
            subprocess.run(["git", "-c", "user.name=Test", "-c", "user.email=test@example.invalid",
                            "commit", "--allow-empty", "-qm", "baseline"], cwd=root, check=True)
            config = root / "experiment.json"
            config.write_text('{"updates": 7}\n')
            output = root / "run"
            command = [sys.executable, "-c", "from pathlib import Path; Path('experiment.json').write_text('{}')"]
            self.assertNotEqual(record.record_run(root, output, "changed", config, command), 0)
            self.assertEqual(json.loads((output / "run.json").read_text())["state"], "inputs_changed")


if __name__ == "__main__":
    unittest.main()
