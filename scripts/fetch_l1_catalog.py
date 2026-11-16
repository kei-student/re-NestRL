#!/usr/bin/env python3
"""Download only the immutable, hash-pinned bundles needed by an L1 catalog."""
import argparse
import json
import os
from pathlib import Path

from release_artifacts import GitHubStore, file_sha256, reconstruct, sha256
from nestrl_research.partner_pool import read_catalog


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--catalog', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--repository', default='kei-student/re-NestRL')
    args = parser.parse_args()
    token = os.environ.get('RELEASE_GITHUB_TOKEN')
    if not token:
        parser.error('Configure RELEASE_GITHUB_TOKEN in the environment; never pass it in arguments')
    catalog = read_catalog(args.catalog)
    store = GitHubStore(args.repository, token)
    archives = {}
    for model in catalog['models']:
        item = model['archive']
        previous = archives.setdefault(item['filename'], item)
        for key in ('release_id','manifest_asset_id','manifest_sha256','sha256'):
            if previous[key] != item[key]:
                raise ValueError('Catalog archive records disagree')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for filename, item in archives.items():
        if Path(filename).name != filename:
            raise ValueError('Unsafe archive filename')
        payload = store.fetch_part(item['manifest_asset_id'])
        if sha256(payload) != item['manifest_sha256']:
            raise ValueError('Transfer manifest SHA-256 mismatch')
        manifest = json.loads(payload)
        if (manifest['repository'] != args.repository or manifest['release_id'] != item['release_id']
                or manifest['filename'] != filename or manifest['sha256'] != item['sha256']):
            raise ValueError('Transfer manifest differs from catalog')
        reconstruct(manifest, store.fetch_part, args.output_dir / filename)
        receipt = args.output_dir / (filename + '.parts.json')
        if receipt.exists() and receipt.read_bytes() != payload:
            raise ValueError('Existing transfer manifest differs; will not overwrite')
        if not receipt.exists():
            with receipt.open('xb') as f:
                f.write(payload)
        print(json.dumps({'archive':filename,'archive_sha256':file_sha256(args.output_dir/filename),
                          'verified':True}),flush=True)


if __name__ == '__main__':
    main()
