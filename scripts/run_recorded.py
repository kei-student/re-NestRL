#!/usr/bin/env python3
"""Record a fresh invocation, then run its explicit argv; no experiment defaults."""

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import sys

from source_manifest import ROOT, canonical_hash, digest, source_manifest


def utc():
    return datetime.now(timezone.utc).isoformat()


def record_invocation(config, command, record_dir, metadata=None):
    if not command:
        raise ValueError('Supply the exact experiment command after --')
    if any(arg.split('=', 1)[0].lower() in {'--token', '--password', '--secret', '--authorization'} for arg in command):
        raise ValueError('Credentials must not be passed in recorded argv')
    config = config.resolve()
    data = json.loads(config.read_text())
    for index, arg in enumerate(command):
        if arg == '--config' and index + 1 < len(command):
            if (ROOT / command[index + 1]).resolve() != config:
                raise ValueError('Recorded config differs from command config')
        if arg.startswith('--config=') and (ROOT / arg.split('=', 1)[1]).resolve() != config:
            raise ValueError('Recorded config differs from command config')
    files = source_manifest()
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    dirty = subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True).splitlines()
    effective = None
    if '-m' in command and 'nestrl_research' in command:
        sys.path.insert(0, str(ROOT / 'src'))
        from nestrl_research.config import load_config
        effective = load_config(config).to_dict()
    report = {'schema': 'experiment-invocation-v1', 'started_utc': utc(), 'cwd': str(ROOT),
              'argv': command, 'git_head_at_invocation': head, 'dirty_paths_at_invocation': dirty,
              'source_manifest': files, 'source_manifest_sha256': canonical_hash(files),
              'config_path': str(config), 'config_bytes_sha256': digest(config.read_bytes()),
              'declared_config': data, 'effective_config_before_cli_overrides': effective,
              'config_sha256': canonical_hash(effective if effective is not None else data),
              'runtime': {'python': platform.python_version(), 'executable': sys.executable,
                          'platform': platform.platform(),
                          'packages': {d.metadata['Name']: d.version for d in importlib.metadata.distributions()}},
              'experiment_metadata': metadata,
              'effective_runtime_note': 'Exact argv includes overrides. Run-created config.json/run.json and recovery manifests record the applied child runtime and remain authoritative; preserve them unchanged.'}
    record_dir.mkdir(parents=True, exist_ok=False)
    (record_dir / 'invocation.json').write_text(json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + '\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--record-dir', type=Path, required=True, help='New directory for this invocation, including resume invocations')
    parser.add_argument('--metadata', type=Path, help='Optional experiment manifest JSON; never a secret file')
    parser.add_argument('--record-only', action='store_true', help='Write intent only; do not execute the command')
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    record_invocation(args.config, command, args.record_dir,
                      json.loads(args.metadata.read_text()) if args.metadata else None)
    if args.record_only:
        print(json.dumps({'record_dir': str(args.record_dir), 'executed': False}))
        return
    with (args.record_dir / 'command.log').open('xb') as log:
        result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    (args.record_dir / 'exit.json').write_text(json.dumps({'finished_utc': utc(), 'returncode': result.returncode}) + '\n')
    print(json.dumps({'record_dir': str(args.record_dir), 'returncode': result.returncode}))
    raise SystemExit(result.returncode)


if __name__ == '__main__':
    main()
