import json
from pathlib import Path
import shutil
import tempfile
import unittest

from nestrl_research.config import ROOT
from nestrl_research.provenance import (
    UPSTREAM_MANIFEST,
    UPSTREAM_SOURCE_PATHS,
    source_metadata,
    verify_upstream_sources,
)


class SourceProvenanceTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.root = Path(self.temporary_directory.name)
        manifest = self.root / UPSTREAM_MANIFEST
        manifest.parent.mkdir(parents=True)
        shutil.copyfile(ROOT / UPSTREAM_MANIFEST, manifest)
        for relative in UPSTREAM_SOURCE_PATHS:
            destination = self.root / "external" / "PACE" / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / "external" / "PACE" / relative, destination)
        package = self.root / "src" / "nestrl_research"
        package.mkdir(parents=True)
        for source in (ROOT / "src" / "nestrl_research").glob("*.py"):
            shutil.copyfile(source, package / source.name)

    def test_metadata_is_portable_and_identical_for_a_source_copy(self):
        metadata = source_metadata(self.root)
        self.assertEqual(metadata, source_metadata(ROOT))
        encoded = json.dumps(metadata)
        self.assertNotIn(str(ROOT), encoded)
        self.assertNotIn(str(self.root), encoded)
        self.assertIn("src/nestrl_research/provenance.py", metadata["implementation_sources"])
        self.assertEqual(set(metadata["upstream_sources"]), UPSTREAM_SOURCE_PATHS)
        for key in ("environment_source_sha256", "implementation_source_sha256", "upstream_manifest_sha256"):
            self.assertRegex(metadata[key], r"^[0-9a-f]{64}$")

    def test_changed_pace_source_is_rejected(self):
        relative = sorted(UPSTREAM_SOURCE_PATHS)[0]
        source = self.root / "external" / "PACE" / relative
        source.write_bytes(source.read_bytes() + b"\n# altered source\n")
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            source_metadata(self.root)

    def test_missing_pace_source_is_rejected(self):
        relative = sorted(UPSTREAM_SOURCE_PATHS)[0]
        (self.root / "external" / "PACE" / relative).unlink()
        with self.assertRaisesRegex(ValueError, "source is missing"):
            verify_upstream_sources(self.root)

    def test_omitted_source_and_escaping_manifest_paths_are_rejected(self):
        path = self.root / UPSTREAM_MANIFEST
        original = json.loads(path.read_text())
        for extra_path in (None, "../../outside.py", "/absolute.py"):
            manifest = dict(original)
            manifest.pop(next(iter(manifest)))
            if extra_path is not None:
                manifest[extra_path] = "0" * 64
            path.write_text(json.dumps(manifest))
            with self.subTest(extra=extra_path), self.assertRaisesRegex(ValueError, "exactly the six"):
                verify_upstream_sources(self.root)

    def test_manifest_format_changes_do_not_change_source_identity(self):
        before = source_metadata(self.root)
        path = self.root / UPSTREAM_MANIFEST
        manifest = json.loads(path.read_text())
        path.write_text(json.dumps(dict(reversed(list(manifest.items()))), separators=(",", ":")))
        self.assertEqual(before, source_metadata(self.root))

    def test_environment_and_implementation_edits_have_separate_identity(self):
        before = source_metadata(self.root)
        path = self.root / "src" / "nestrl_research" / "environment.py"
        path.write_bytes(path.read_bytes() + b"\n# reconstructed environment revision\n")
        after = source_metadata(self.root)
        self.assertNotEqual(before["environment_source_sha256"], after["environment_source_sha256"])
        self.assertNotEqual(before["implementation_source_sha256"], after["implementation_source_sha256"])
        self.assertEqual(before["upstream_manifest_sha256"], after["upstream_manifest_sha256"])
        unrelated = self.root / "src" / "nestrl_research" / "model.py"
        unrelated.write_bytes(unrelated.read_bytes() + b"\n# model revision\n")
        latest = source_metadata(self.root)
        self.assertEqual(after["environment_source_sha256"], latest["environment_source_sha256"])
        self.assertNotEqual(after["implementation_source_sha256"], latest["implementation_source_sha256"])

    def test_added_and_removed_implementation_sources_change_identity(self):
        before = source_metadata(self.root)
        additional = self.root / "src" / "nestrl_research" / "added.py"
        additional.write_text("# new reconstruction component\n")
        after = source_metadata(self.root)
        self.assertNotEqual(before["implementation_source_sha256"], after["implementation_source_sha256"])
        additional.unlink()
        self.assertEqual(before, source_metadata(self.root))


if __name__ == "__main__":
    unittest.main()
