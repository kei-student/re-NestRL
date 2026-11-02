"""Verify pinned PACE sources and identify the reconstruction used by a run."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from .config import ROOT, UPSTREAM_COMMIT


SOURCE_SCHEMA = "re-nestrl-sources-v1"
UPSTREAM_MANIFEST = "scripts/upstream_files.json"
UPSTREAM_SOURCE_PATHS = frozenset({
    "environment/overcooked/gym_cooking/cooking_world/constants.py",
    "environment/overcooked/gym_cooking/cooking_world/abstract_classes.py",
    "environment/overcooked/gym_cooking/cooking_world/world_objects.py",
    "environment/overcooked/gym_cooking/cooking_world/cooking_world.py",
    "environment/overcooked/gym_cooking/cooking_book/recipe.py",
    "environment/overcooked/gym_cooking/cooking_book/recipe_drawer.py",
})
ENVIRONMENT_SOURCE_PATHS = (
    "src/nestrl_research/environment.py",
    "src/nestrl_research/peers.py",
)


def _map_sha256(value: dict[str, str]) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _file_sha256(root: Path, relative_path: str) -> str:
    path = root / relative_path
    if not path.is_file():
        raise ValueError(f"Required source is missing: {relative_path}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate upstream manifest entry: {key}")
        result[key] = value
    return result


def verify_upstream_sources(root: Path | str | None = None) -> dict[str, str]:
    """Reject missing or altered upstream code before training or evaluation.

    Manifest keys are strictly the six source paths consumed by the local PACE
    loader. They are relative to ``external/PACE``. This function is independent
    of bootstrap scripts and can also be used by workspace validation tools.
    ``root`` is an optional repository root, primarily useful for isolated tests.
    """
    root = Path(root) if root is not None else ROOT
    manifest_path = root / UPSTREAM_MANIFEST
    if not manifest_path.is_file():
        raise ValueError(f"Required source manifest is missing: {UPSTREAM_MANIFEST}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"), object_pairs_hook=_unique_object)
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise ValueError(f"Invalid upstream manifest: {UPSTREAM_MANIFEST}") from error
    if not isinstance(manifest, dict) or set(manifest) != UPSTREAM_SOURCE_PATHS:
        raise ValueError("Upstream manifest must contain exactly the six consumed PACE source paths")
    for relative_path, expected_sha256 in manifest.items():
        if not isinstance(expected_sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", expected_sha256):
            raise ValueError(f"Invalid source SHA-256 for {relative_path}")
        actual_sha256 = _file_sha256(root / "external" / "PACE", relative_path)
        if actual_sha256 != expected_sha256:
            raise ValueError(f"Pinned PACE source hash mismatch: {relative_path}")
    return dict(sorted(manifest.items()))


def source_metadata(root: Path | str | None = None) -> dict[str, Any]:
    """Return portable source identity separately from runtime/device metadata.

    A source-set digest is the SHA-256 of a sorted, compact JSON mapping from
    repository-relative POSIX paths to individual file-byte SHA-256 values.
    The environment identity additionally includes the canonical manifest digest,
    and therefore every verified upstream file digest. The implementation digest
    covers every direct ``src/nestrl_research/*.py`` source, including this file.
    It changes on added/removed source files as well as on byte changes.
    """
    root = Path(root) if root is not None else ROOT
    upstream_sources = verify_upstream_sources(root)
    manifest_sha256 = _map_sha256(upstream_sources)
    environment_sources = {path: _file_sha256(root, path) for path in ENVIRONMENT_SOURCE_PATHS}
    environment_sources[UPSTREAM_MANIFEST] = manifest_sha256
    implementation_sources = {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted((root / "src" / "nestrl_research").glob("*.py"))
        if path.is_file()
    }
    if "src/nestrl_research/provenance.py" not in implementation_sources:
        raise ValueError("Required source is missing: src/nestrl_research/provenance.py")
    return {
        "source_schema": SOURCE_SCHEMA,
        "upstream_commit": UPSTREAM_COMMIT,
        "upstream_manifest_sha256": manifest_sha256,
        "upstream_sources": upstream_sources,
        "environment_source_sha256": _map_sha256(environment_sources),
        "environment_sources": environment_sources,
        "implementation_source_sha256": _map_sha256(implementation_sources),
        "implementation_sources": implementation_sources,
    }
