#!/usr/bin/env python3
"""Store immutable experiment bundles in GitHub Draft Releases and verify transfer."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_sha256(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def request_bytes(url: str, token: str, *, data: bytes | None = None,
                  method: str = "GET", binary: bool = False, opener=None) -> bytes:
    """Create a fresh unauthenticated request after every asset redirect."""
    opener = opener or urllib.request.build_opener(NoRedirect())
    for _ in range(6):
        parsed = urllib.parse.urlsplit(url)
        if parsed.scheme != "https":
            raise ValueError("HTTPS is required")
        headers = {"User-Agent": "NestRL-artifact-store", "Accept":
                   "application/octet-stream" if binary and method == "GET" else "application/vnd.github+json"}
        if parsed.hostname in {"api.github.com", "uploads.github.com"}:
            headers["Authorization"] = "Bearer " + token
        elif not (parsed.hostname or "").endswith(".githubusercontent.com"):
            raise ValueError("Unexpected artifact host")
        if data is not None:
            headers["Content-Type"] = "application/octet-stream" if binary else "application/json"
        request = urllib.request.Request(url, data=data, method=method, headers=headers)
        try:
            with opener.open(request, timeout=120) as response:
                return response.read()
        except urllib.error.HTTPError as error:
            if method == "GET" and error.code in {301, 302, 303, 307, 308} and error.headers.get("Location"):
                url = urllib.parse.urljoin(url, error.headers["Location"])
                continue
            raise RuntimeError(f"GitHub request failed: {method} {parsed.hostname} HTTP {error.code}") from None
    raise RuntimeError("Too many asset redirects")


def reconstruct(manifest: dict, fetch_part, output: Path) -> str:
    """Reject modified, reordered, truncated, or incorrectly combined parts."""
    if manifest.get("schema_version") != 1 or not manifest.get("parts"):
        raise ValueError("Unsupported transfer manifest")
    if output.exists():
        if file_sha256(output) == manifest["sha256"]:
            return manifest["sha256"]
        raise FileExistsError("Existing destination has different contents")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".downloading")
    created = False
    try:
        with temporary.open("xb") as target:
            created = True
            for expected_order, part in enumerate(manifest["parts"], 1):
                if part["order"] != expected_order:
                    raise ValueError("Invalid part ordering")
                block = fetch_part(part["asset_id"])
                if len(block) != part["bytes"] or sha256(block) != part["sha256"]:
                    raise ValueError("Downloaded part hash/size mismatch")
                target.write(block)
        if temporary.stat().st_size != manifest["bytes"] or file_sha256(temporary) != manifest["sha256"]:
            raise ValueError("Reconstructed archive hash/size mismatch")
        os.link(temporary, output)
        temporary.unlink()
    except BaseException:
        if created:
            temporary.unlink(missing_ok=True)
        raise
    return manifest["sha256"]


class GitHubStore:
    def __init__(self, repository: str, token: str):
        parts = repository.split("/")
        if len(parts) != 2 or any(not p or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-" for c in p) for p in parts):
            raise ValueError("Use owner/repository")
        self.repository = repository
        self.token = token
        self.base = "https://api.github.com/repos/" + repository

    def api(self, suffix: str, data: dict | None = None):
        return json.loads(request_bytes(self.base + suffix, self.token,
                                       data=json.dumps(data).encode() if data is not None else None,
                                       method="POST" if data is not None else "GET"))

    def create(self, tag: str, title: str) -> dict:
        return self.api("/releases", {"tag_name": tag, "name": title, "draft": True})

    def fetch_part(self, asset_id: int) -> bytes:
        return request_bytes(self.base + f"/releases/assets/{asset_id}", self.token, binary=True)

    def upload_one(self, release_id: int, name: str, contents: bytes) -> dict:
        page = 1
        while True:
            assets = self.api(f"/releases/{release_id}/assets?per_page=100&page={page}")
            for asset in assets:
                if asset["name"] != name:
                    continue
                if asset.get("digest") not in {None, "sha256:" + sha256(contents)}:
                    raise FileExistsError("Asset name already contains different data")
                if self.fetch_part(asset["id"]) != contents:
                    raise FileExistsError("Existing asset bytes differ")
                return asset
            if len(assets) < 100:
                break
            page += 1
        url = ("https://uploads.github.com/repos/" + self.repository
               + f"/releases/{release_id}/assets?name=" + urllib.parse.quote(name, safe=""))
        asset = json.loads(request_bytes(url, self.token, data=contents, method="POST", binary=True))
        if self.fetch_part(asset["id"]) != contents:
            raise ValueError("Uploaded asset failed read-back verification")
        return asset

    def upload(self, release_id: int, path: Path, part_bytes: int) -> dict:
        if not self.api(f"/releases/{release_id}").get("draft"):
            raise ValueError("Select a Draft Release")
        if part_bytes < 1:
            raise ValueError("part_bytes must be positive")
        manifest = {"schema_version": 1, "repository": self.repository, "release_id": release_id,
                    "filename": path.name, "bytes": path.stat().st_size,
                    "sha256": file_sha256(path), "parts": []}
        with path.open("rb") as source:
            order = 1
            while block := source.read(part_bytes):
                name = path.name if path.stat().st_size <= part_bytes else f"{path.name}.part{order:03d}"
                asset = self.upload_one(release_id, name, block)
                manifest["parts"].append({"order": order, "name": name, "asset_id": asset["id"],
                                          "bytes": len(block), "sha256": sha256(block)})
                order += 1
        if not manifest["parts"]:
            raise ValueError("Do not upload an empty bundle")
        payload = (json.dumps(manifest, sort_keys=True, indent=2) + "\n").encode()
        sidecar = self.upload_one(release_id, path.name + ".parts.json", payload)
        remote_manifest = json.loads(self.fetch_part(sidecar["id"]))
        with tempfile.TemporaryDirectory(prefix="nestrl-transfer-") as directory:
            reconstruct(remote_manifest, self.fetch_part, Path(directory) / path.name)
        return {"repository": self.repository, "release_id": release_id,
                "manifest_asset_id": sidecar["id"], "manifest_sha256": sha256(payload),
                "archive_sha256": manifest["sha256"], "parts": len(manifest["parts"]),
                "read_back_verified": True}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", required=True)
    commands = parser.add_subparsers(dest="action", required=True)
    create = commands.add_parser("create")
    create.add_argument("--tag", required=True)
    create.add_argument("--title", required=True)
    upload = commands.add_parser("upload")
    upload.add_argument("--release-id", type=int, required=True)
    upload.add_argument("--file", type=Path, required=True)
    upload.add_argument("--part-bytes", type=int, default=2_500_000)
    download = commands.add_parser("download")
    download.add_argument("--manifest-asset-id", type=int, required=True)
    download.add_argument("--manifest-sha256", required=True)
    download.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    token = os.environ.get("RELEASE_GITHUB_TOKEN")
    if not token:
        parser.error("Configure RELEASE_GITHUB_TOKEN in the environment; do not pass it in task text")
    store = GitHubStore(args.repository, token)
    if args.action == "create":
        release = store.create(args.tag, args.title)
        result = {"repository": args.repository, "release_id": release["id"], "draft": release["draft"], "url": release["html_url"]}
    elif args.action == "upload":
        result = store.upload(args.release_id, args.file, args.part_bytes)
    else:
        data = store.fetch_part(args.manifest_asset_id)
        if sha256(data) != args.manifest_sha256:
            raise ValueError("Transfer manifest hash mismatch")
        manifest = json.loads(data)
        if manifest["repository"] != args.repository:
            raise ValueError("Transfer manifest belongs to a different repository")
        result = {"archive_sha256": reconstruct(manifest, store.fetch_part, args.output), "verified": True}
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
