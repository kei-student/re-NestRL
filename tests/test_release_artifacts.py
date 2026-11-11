from __future__ import annotations

import importlib.util
import io
import tempfile
import unittest
import urllib.error
from pathlib import Path

spec = importlib.util.spec_from_file_location("release_artifacts", Path(__file__).resolve().parents[1] / "scripts/release_artifacts.py")
store = importlib.util.module_from_spec(spec)
spec.loader.exec_module(store)


class ArtifactTransferTests(unittest.TestCase):
    def test_redirect_never_forwards_authorization_to_asset_cdn(self):
        class Opener:
            requests = []

            def open(self, request, timeout):
                self.requests.append(request)
                if len(self.requests) == 1:
                    raise urllib.error.HTTPError(request.full_url, 302, "redirect",
                                                 {"Location": "https://release-assets.githubusercontent.com/example"}, None)
                return io.BytesIO(b"verified")

        opener = Opener()
        self.assertEqual(store.request_bytes("https://api.github.com/repos/test/repo/releases/assets/1",
                                            "test-placeholder", binary=True, opener=opener), b"verified")
        self.assertIsNotNone(opener.requests[0].get_header("Authorization"))
        self.assertIsNone(opener.requests[1].get_header("Authorization"))

    def test_multipart_restoration_rejects_tampering_and_keeps_destination_absent(self):
        parts = {10: b"first", 20: b"second"}
        combined = b"firstsecond"
        manifest = {"schema_version": 1, "bytes": len(combined), "sha256": store.sha256(combined),
                    "parts": [{"order": i, "asset_id": key, "bytes": len(value), "sha256": store.sha256(value)}
                              for i, (key, value) in enumerate(parts.items(), 1)]}
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "archive.zip"
            bad_parts = {**parts, 20: b"edited"}
            with self.assertRaises(ValueError):
                store.reconstruct(manifest, bad_parts.__getitem__, output)
            self.assertFalse(output.exists())
            self.assertEqual(store.reconstruct(manifest, parts.__getitem__, output), store.sha256(combined))
            self.assertEqual(output.read_bytes(), combined)
            output.write_bytes(b"different existing file")
            with self.assertRaises(FileExistsError):
                store.reconstruct(manifest, parts.__getitem__, output)

    def test_does_not_delete_another_downloads_partial_file(self):
        manifest = {"schema_version": 1, "bytes": 1, "sha256": store.sha256(b"x"),
                    "parts": [{"order": 1, "asset_id": 1, "bytes": 1, "sha256": store.sha256(b"x")}]}
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "archive.zip"
            partial = output.with_name(output.name + ".downloading")
            partial.write_bytes(b"owned by another process")
            with self.assertRaises(FileExistsError):
                store.reconstruct(manifest, lambda _: b"x", output)
            self.assertEqual(partial.read_bytes(), b"owned by another process")


if __name__ == "__main__":
    unittest.main()
