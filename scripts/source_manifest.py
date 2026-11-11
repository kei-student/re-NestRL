#!/usr/bin/env python3
"""Hash source bytes and verify a named historical runtime without importing it."""

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import subprocess

ROOT = Path(__file__).resolve().parents[1]

def digest(data):
    return hashlib.sha256(data).hexdigest()

def canonical_hash(value):
    return digest(json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=True,
                             allow_nan=False).encode('utf-8'))

def safe_file(root, name):
    path = PurePosixPath(name)
    if path.is_absolute() or '..' in path.parts or '\\' in name:
        raise ValueError('Unsafe source path')
    target = root / name
    if target.is_symlink() or not target.resolve().is_relative_to(root.resolve()):
        raise ValueError('Source symlinks are not supported')
    return target

def source_manifest(root=ROOT):
    result = subprocess.check_output(['git', 'ls-files', '-z', '--cached', '--others',
                                      '--exclude-standard', '--', 'src', 'scripts', 'tests',
                                      'configs', 'pyproject.toml', '.gitattributes', '.gitignore'], cwd=root)
    files = {}
    for name in sorted(set(filter(None, result.decode().split('\0')))):
        p = PurePosixPath(name)
        if any(part in {'.git', '.venv', 'external', 'outputs', '__pycache__', '.env', 'secrets', 'credentials'}
               or part.startswith('.env.') for part in p.parts):
            continue
        selected = ((p.parts[0] in {'src', 'scripts', 'tests'} and p.suffix == '.py')
                    or (p.parts[0] == 'configs' and p.suffix == '.json')
                    or name in {'pyproject.toml', '.gitattributes', '.gitignore'})
        if selected:
            target = safe_file(root, name)
            if not target.is_file():
                raise ValueError(f'Missing tracked source: {name}')
            files[name] = digest(target.read_bytes())
    return files

def verify_runtime(root, reference):
    expected = reference['runtime_files']
    expected_core = {n for n in expected if n.startswith('src/')}
    current_core = {p.relative_to(root).as_posix() for p in (root / 'src').rglob('*.py')}
    if current_core != expected_core:
        raise ValueError('Runtime source file set differs')
    for name, sha256 in expected.items():
        if digest(safe_file(root, name).read_bytes()) != sha256:
            raise ValueError(f'Runtime snapshot differs: {name}')
    return len(expected)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', type=Path, help='Verify this named historical runtime; not a new experiment policy')
    args = parser.parse_args()
    files = source_manifest()
    report = {'manifest': files, 'manifest_sha256': canonical_hash(files)}
    if args.check:
        report['runtime_files_verified'] = verify_runtime(ROOT, json.loads(args.check.read_text()))
    print(json.dumps(report, indent=2))

if __name__ == '__main__':
    main()
