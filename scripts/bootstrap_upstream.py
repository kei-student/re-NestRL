#!/usr/bin/env python3
"""Fetch the official pinned world; never overwrite an existing checkout."""

from __future__ import annotations

import hashlib
import json
import shutil
import tarfile
import tempfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COMMIT = "f008c19aa743ec209550375c9ee9c25379ac8b24"
ARCHIVE_SHA256 = "d1abf44f532f58645ffa3f2877a1912c9359c564c4437adf72ed18d46e413828"
URL = f"https://codeload.github.com/bigai-ai/PACE/tar.gz/{COMMIT}"


def verify_upstream(root=ROOT):
    expected = json.loads((root / "scripts" / "upstream_files.json").read_text(encoding="utf-8"))
    for relative, digest in expected.items():
        source = root / "external" / "PACE" / relative
        if not source.is_file() or hashlib.sha256(source.read_bytes()).hexdigest() != digest:
            raise ValueError(f"Pinned upstream source missing or changed: {relative}")
    return len(expected)


def main():
    target = ROOT / "external" / "PACE"
    if target.exists():
        count = verify_upstream()
        print(f"Verified {count} imported sources from PACE {COMMIT}; existing directory preserved.")
        return
    target.parent.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="nestrl-upstream-") as temporary:
        temporary = Path(temporary)
        archive = temporary / "pace.tar.gz"
        urllib.request.urlretrieve(URL, archive)
        if hashlib.sha256(archive.read_bytes()).hexdigest() != ARCHIVE_SHA256:
            raise ValueError("Upstream archive SHA-256 mismatch; no files were installed")
        with tarfile.open(archive) as bundle:
            bundle.extractall(temporary, filter="data")
        shutil.move(str(temporary / f"PACE-{COMMIT}"), str(target))
    count = verify_upstream()
    print(f"Installed and verified PACE {COMMIT}, {count} imported source files.")


if __name__ == "__main__":
    main()
