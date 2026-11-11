#!/usr/bin/env python3
"""Standalone, experiment-neutral GitHub Draft artifact storage (stdlib only)."""

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import tempfile
import urllib.error
import urllib.parse
import urllib.request
import zipfile


class StorageError(Exception):
    pass


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def safe_name(name):
    if not name or Path(name).name != name or name in ('.', '..') or '/' in name or '\\' in name:
        raise StorageError('Asset name must be a single filename')
    return name


def safe_relative(name):
    p = PurePosixPath(name)
    if not name or p.as_posix() != name or p.is_absolute() or '..' in p.parts or '\\' in name or ':' in name:
        raise StorageError('Unsafe bundle path')
    if any(part.lower() in {'.git', '.venv', '.aws', '.ssh', 'credentials', 'secrets', '.env'}
           or part.lower().startswith('.env.') for part in p.parts):
        raise StorageError('Credential or environment directories cannot be bundled')
    return p


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class GitHub:
    def __init__(self, repo, token=None):
        if not re.fullmatch(r'[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+', repo):
            raise StorageError('Expected owner/repository')
        self.repo = repo
        self.api = 'https://api.github.com/repos/' + repo
        self.token = token or os.environ.get('RELEASE_GITHUB_TOKEN')
        if not self.token:
            raise StorageError('RELEASE_GITHUB_TOKEN is not configured')
        self.opener = urllib.request.build_opener(NoRedirect())

    def request(self, url, method='GET', data=None, destination=None):
        initial = urllib.parse.urlsplit(url)
        if initial.scheme != 'https' or initial.hostname not in {'api.github.com', 'uploads.github.com'}:
            raise StorageError('Unexpected GitHub API destination')
        if isinstance(data, dict):
            data = json.dumps(data).encode()
            content = 'application/json'
        else:
            content = 'application/octet-stream'
        for _ in range(5):
            parsed = urllib.parse.urlsplit(url)
            if parsed.scheme != 'https' or parsed.hostname not in {
                    initial.hostname, 'release-assets.githubusercontent.com', 'objects.githubusercontent.com'}:
                raise StorageError('Unexpected asset redirect destination')
            headers = {'User-Agent': 'verified-release-artifacts', 'Content-Type': content,
                       'Accept': 'application/octet-stream' if destination else 'application/vnd.github+json'}
            # A managed Secret may be a proxy placeholder. Never inspect/extract it.
            # Do not forward Authorization to a redirect host, including the asset CDN.
            if parsed.hostname == initial.hostname:
                headers['Authorization'] = 'Bearer ' + self.token
            req = urllib.request.Request(url, method=method, headers=headers, data=data)
            try:
                response = self.opener.open(req, timeout=120)
            except urllib.error.HTTPError as error:
                response = error
            except urllib.error.URLError:
                raise StorageError('GitHub network request failed') from None
            with response:
                if response.code in (301, 302, 303, 307, 308):
                    if method != 'GET':
                        raise StorageError('Refusing redirect of a write request')
                    url = urllib.parse.urljoin(url, response.headers['Location'])
                    continue
                if not 200 <= response.code < 300:
                    raise StorageError(f'GitHub {method} HTTP {response.code}')
                if destination:
                    with Path(destination).open('xb') as output:
                        shutil.copyfileobj(response, output)
                    return None
                return json.load(response)
        raise StorageError('Too many redirects')

    def release(self, release_id, require_draft=False):
        release = self.request(self.api + '/releases/' + str(release_id))
        if require_draft and not release['draft']:
            raise StorageError('Uploads require a Draft Release')
        return release

    def assets(self, release_id):
        assets, page = {}, 1
        while True:
            batch = self.request(self.api + f'/releases/{release_id}/assets?per_page=100&page={page}')
            for asset in batch:
                if asset['name'] in assets:
                    raise StorageError('Duplicate remote asset names')
                assets[asset['name']] = asset
            if len(batch) < 100:
                return assets
            page += 1

    def fetch(self, asset, destination):
        self.request(self.api + '/releases/assets/' + str(asset['id']), destination=destination)

    def put(self, release, path, existing):
        name = safe_name(path.name)
        expected = sha256(path)
        asset = existing.get(name)
        if asset:
            if asset['size'] != path.stat().st_size:
                raise StorageError('Existing asset differs: ' + name)
            if asset.get('digest'):
                if asset['digest'] != 'sha256:' + expected:
                    raise StorageError('Existing asset differs: ' + name)
            else:
                with tempfile.TemporaryDirectory() as tmp:
                    downloaded = Path(tmp) / name
                    self.fetch(asset, downloaded)
                    if sha256(downloaded) != expected:
                        raise StorageError('Existing asset differs: ' + name)
        else:
            upload = release['upload_url'].split('{', 1)[0]
            if urllib.parse.urlsplit(upload).hostname != 'uploads.github.com':
                raise StorageError('Unexpected upload host')
            data = path.read_bytes()
            if hashlib.sha256(data).hexdigest() != expected:
                raise StorageError('Local input changed: ' + name)
            asset = self.request(upload + '?' + urllib.parse.urlencode({'name': name}), 'POST', data)
            if asset['size'] != len(data) or (asset.get('digest') and asset['digest'] != 'sha256:' + expected):
                raise StorageError('Uploaded asset differs: ' + name)
        # Verification is mandatory for newly uploaded AND reused assets.
        with tempfile.TemporaryDirectory() as tmp:
            downloaded = Path(tmp) / name
            self.fetch(asset, downloaded)
            if sha256(downloaded) != expected:
                raise StorageError('Download verification failed: ' + name)
        existing[name] = asset
        return asset


def prepare_parts(files, directory, part_size):
    if part_size < 1:
        raise StorageError('Part size must be positive')
    records, staged, names = [], {}, set()
    for path in files:
        path = Path(path)
        name = safe_name(path.name)
        if name in names or not path.is_file() or path.is_symlink():
            raise StorageError('Duplicate or invalid input file')
        names.add(name)
        record = {'name': name, 'size': path.stat().st_size, 'sha256': sha256(path), 'parts': []}
        if record['size'] <= part_size:
            staged[name] = path
            record['parts'].append({'name': name, 'size': record['size'], 'sha256': record['sha256']})
        else:
            with path.open('rb') as stream:
                index = 1
                while chunk := stream.read(part_size):
                    part = directory / (name + f'.part{index:03d}')
                    part.write_bytes(chunk)
                    staged[part.name] = part
                    record['parts'].append({'name': part.name, 'size': len(chunk),
                                            'sha256': hashlib.sha256(chunk).hexdigest()})
                    index += 1
            if sha256(path) != record['sha256']:
                raise StorageError('Input changed while splitting')
        records.append(record)
    part_names = [p['name'] for r in records for p in r['parts']]
    if len(part_names) != len(set(part_names)):
        raise StorageError('Part names collide')
    return records, staged


def reassemble(manifest, parts_dir, output_dir):
    for record in manifest['files']:
        name = safe_name(record['name'])
        target = output_dir / name
        if target.exists():
            if not target.is_symlink() and sha256(target) == record['sha256']:
                continue
            raise StorageError('Refusing to overwrite different local file: ' + name)
        with tempfile.TemporaryDirectory(dir=output_dir) as tmp:
            temporary = Path(tmp) / name
            with temporary.open('xb') as stream:
                for part in record['parts']:
                    path = parts_dir / safe_name(part['name'])
                    if path.is_symlink() or path.stat().st_size != part['size'] or sha256(path) != part['sha256']:
                        raise StorageError('Part checksum mismatch')
                    with path.open('rb') as input_stream:
                        shutil.copyfileobj(input_stream, stream)
            if temporary.stat().st_size != record['size'] or sha256(temporary) != record['sha256']:
                raise StorageError('Whole-file checksum mismatch')
            # Hard link is atomic and refuses to replace a concurrently created file.
            os.link(temporary, target)


def upload(client, release_id, files, manifest_name, part_size):
    release = client.release(release_id, require_draft=True)
    existing = client.assets(release_id)
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        records, staged = prepare_parts(files, directory, part_size)
        name = safe_name(manifest_name)
        if name in staged or name in {'parts', 'verified-parts', 'verified-files'}:
            raise StorageError('Manifest name collides with input assets')
        manifest = {'schema': 'github-artifact-parts-v1', 'repo': client.repo,
                    'release_id': release_id, 'files': records}
        metadata = directory / name
        metadata.write_text(json.dumps(manifest, sort_keys=True, indent=2) + '\n')
        # Check ALL conflicting names before starting new uploads.
        for asset_name, path in {**staged, name: metadata}.items():
            asset = existing.get(asset_name)
            if asset and (asset['size'] != path.stat().st_size or
                          (asset.get('digest') and asset['digest'] != 'sha256:' + sha256(path))):
                raise StorageError('Existing asset differs: ' + asset_name)
        for path in staged.values():
            client.put(release, path, existing)
        # Check the whole reconstructed bytes as well as individual downloaded parts.
        verified = directory / 'verified-parts'
        restored = directory / 'verified-files'
        verified.mkdir()
        restored.mkdir()
        for name in staged:
            client.fetch(existing[name], verified / name)
        reassemble(manifest, verified, restored)
        client.put(release, metadata, existing)
        return {'release_id': release_id, 'manifest': metadata.name, 'manifest_sha256': sha256(metadata),
                'files': records, 'download_verification': 'passed', 'draft': True}


def download(client, release_id, manifest_name, expected_manifest_sha256, output):
    assets = client.assets(release_id)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if output.is_symlink():
        raise StorageError('Output symlinks are not supported')
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        metadata = directory / safe_name(manifest_name)
        client.fetch(assets[metadata.name], metadata)
        if sha256(metadata) != expected_manifest_sha256:
            raise StorageError('Parts manifest checksum mismatch')
        manifest = json.loads(metadata.read_text())
        if manifest['schema'] != 'github-artifact-parts-v1' or manifest['repo'] != client.repo or manifest['release_id'] != release_id:
            raise StorageError('Parts manifest belongs to a different release')
        parts = directory / 'parts'
        parts.mkdir()
        seen = set()
        for record in manifest['files']:
            safe_name(record['name'])
            for part in record['parts']:
                name = safe_name(part['name'])
                if name not in seen:
                    client.fetch(assets[name], parts / name)
                    seen.add(name)
        reassemble(manifest, parts, output)
    return {'release_id': release_id, 'download_verification': 'passed', 'output': str(output)}


def bundle(root, includes, output):
    root, output = Path(root).resolve(), Path(output)
    if output.exists():
        raise StorageError('Bundle output already exists')
    files = {}
    for name in includes:
        safe_relative(name)
        selected = root / name
        if selected.is_symlink():
            raise StorageError('Bundle symlinks are not supported')
        if not selected.exists():
            raise StorageError('Bundle input is missing')
        for path in ([selected] if selected.is_file() else selected.rglob('*')):
            relative = path.relative_to(root).as_posix()
            safe_relative(relative)
            if path.is_symlink() or not path.resolve().is_relative_to(root):
                raise StorageError('Bundle symlinks are not supported')
            if path.name == '.writer.lock':
                raise StorageError('Run may have an active writer; bundle only a stopped snapshot')
            if path.name == '.training.lock' or path.is_dir():
                continue
            if relative == 'bundle-files.json':
                raise StorageError('Reserved bundle manifest name')
            files[relative] = sha256(path)
    with zipfile.ZipFile(output, 'x', zipfile.ZIP_DEFLATED) as archive:
        for name in sorted(files):
            archive.write(root / name, name)
        archive.writestr('bundle-files.json', json.dumps({'schema': 'run-bundle-v1', 'files': files}, sort_keys=True, indent=2) + '\n')
    with zipfile.ZipFile(output) as archive:
        for name, digest in files.items():
            if hashlib.sha256(archive.read(name)).hexdigest() != digest:
                raise StorageError('Input changed during bundling')
    return {'file': str(output), 'sha256': sha256(output), 'files': len(files)}


def restore_bundle(archive_path, expected_sha256, output):
    if sha256(archive_path) != expected_sha256:
        raise StorageError('Bundle checksum mismatch')
    output = Path(output)
    if output.exists():
        raise StorageError('Restore into a new directory; existing runs are never overwritten')
    with zipfile.ZipFile(archive_path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise StorageError('Duplicate bundle entries')
        for item in archive.infolist():
            safe_relative(item.filename)
            if stat.S_ISLNK(item.external_attr >> 16):
                raise StorageError('Bundle symlink rejected')
        manifest = json.loads(archive.read('bundle-files.json'))
        if manifest['schema'] != 'run-bundle-v1' or set(names) != set(manifest['files']) | {'bundle-files.json'}:
            raise StorageError('Bundle file inventory differs')
        output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=output.parent) as tmp:
            directory = Path(tmp) / 'restored'
            directory.mkdir()
            for name, digest in manifest['files'].items():
                data = archive.read(name)
                if hashlib.sha256(data).hexdigest() != digest:
                    raise StorageError('Bundle member checksum mismatch')
                target = directory / name
                target.parent.mkdir(parents=True, exist_ok=True)
                with target.open('xb') as stream:
                    stream.write(data)
            if output.exists():
                raise StorageError('Restore destination appeared; refusing overwrite')
            directory.rename(output)
    return {'output': str(output), 'files': len(manifest['files']), 'sha256': expected_sha256}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    p = sub.add_parser('create-draft')
    p.add_argument('--repo', required=True)
    p.add_argument('--tag', required=True)
    p.add_argument('--target', required=True, help='Pinned commit SHA')
    p.add_argument('--title', required=True)
    for action in ('upload', 'download'):
        p = sub.add_parser(action)
        p.add_argument('--repo', required=True)
        p.add_argument('--release-id', required=True, type=int)
        p.add_argument('--manifest-name', required=True)
        if action == 'upload':
            p.add_argument('--file', required=True, action='append', type=Path)
            p.add_argument('--part-size', type=int, default=2500000)
        else:
            p.add_argument('--manifest-sha256', required=True)
            p.add_argument('--output-dir', required=True, type=Path)
    p = sub.add_parser('bundle')
    p.add_argument('--root', required=True, type=Path)
    p.add_argument('--include', required=True, action='append')
    p.add_argument('--output', required=True, type=Path)
    p = sub.add_parser('restore-bundle')
    p.add_argument('--file', required=True, type=Path)
    p.add_argument('--sha256', required=True)
    p.add_argument('--output-dir', required=True, type=Path)
    args = parser.parse_args()
    try:
        if args.action == 'bundle':
            result = bundle(args.root, args.include, args.output)
        elif args.action == 'restore-bundle':
            result = restore_bundle(args.file, args.sha256, args.output_dir)
        else:
            client = GitHub(args.repo)
            if args.action == 'create-draft':
                if not re.fullmatch('[0-9a-fA-F]{40}', args.target):
                    raise StorageError('Target must be a full commit SHA')
                # Reuse a matching draft rather than creating a duplicate.
                matches, page = [], 1
                while True:
                    batch = client.request(client.api + f'/releases?per_page=100&page={page}')
                    matches.extend(r for r in batch if r['tag_name'] == args.tag)
                    if len(batch) < 100:
                        break
                    page += 1
                if len(matches) > 1:
                    raise StorageError('Multiple releases match tag')
                if matches:
                    result = matches[0]
                    if not result['draft'] or result['target_commitish'] != args.target:
                        raise StorageError('Existing release differs')
                else:
                    result = client.request(client.api + '/releases', 'POST',
                        {'tag_name': args.tag, 'target_commitish': args.target, 'name': args.title, 'draft': True})
                result = {k: result[k] for k in ('id', 'html_url', 'draft', 'tag_name', 'target_commitish')}
            elif args.action == 'upload':
                result = upload(client, args.release_id, args.file, args.manifest_name, args.part_size)
            else:
                result = download(client, args.release_id, args.manifest_name, args.manifest_sha256, args.output_dir)
        print(json.dumps(result, indent=2))
    except (StorageError, OSError, KeyError, ValueError, zipfile.BadZipFile) as error:
        if isinstance(error, StorageError):
            parser.exit(1, str(error) + '\n')
        parser.exit(1, 'Artifact operation failed: ' + type(error).__name__ + '\n')


if __name__ == '__main__':
    main()
