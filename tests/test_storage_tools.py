"""Transfer contract tests: immutable assets, byte recovery, and secret routing."""

import hashlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import zipfile

SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'
spec = importlib.util.spec_from_file_location('release_storage_tools', SCRIPTS / 'release_artifacts.py')
storage = importlib.util.module_from_spec(spec)
spec.loader.exec_module(storage)


class MemoryGitHub(storage.GitHub):
    def __init__(self):
        super().__init__('example/project', token='test-placeholder')
        self.saved = {}
        self.writes = 0

    def request(self, url, method='GET', data=None, destination=None):
        if destination:
            number = int(url.rsplit('/', 1)[1])
            body = next(v['body'] for v in self.saved.values() if v['id'] == number)
            Path(destination).write_bytes(body)
            return
        if method == 'POST':
            name = storage.urllib.parse.parse_qs(storage.urllib.parse.urlsplit(url).query)['name'][0]
            if name in self.saved:
                raise AssertionError('Must not overwrite an existing remote asset')
            self.writes += 1
            asset = {'name': name, 'id': self.writes, 'size': len(data),
                     'digest': 'sha256:' + hashlib.sha256(data).hexdigest()}
            self.saved[name] = {**asset, 'body': data}
            return asset
        if '/assets?' in url:
            return [{k: v for k, v in row.items() if k != 'body'} for row in self.saved.values()]
        return {'id': 7, 'draft': True, 'upload_url': 'https://uploads.github.com/test/assets{?name,label}'}


class StorageTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def test_split_upload_reuse_download_and_corrupt_part_rejection(self):
        original = self.root / 'run.zip'
        original.write_bytes(bytes(range(100)))
        client = MemoryGitHub()
        result = storage.upload(client, 7, [original], 'experiment-parts.json', 19)
        self.assertEqual(result['download_verification'], 'passed')
        self.assertEqual(len(result['files'][0]['parts']), 6)
        writes = client.writes
        storage.upload(client, 7, [original], 'experiment-parts.json', 19)
        self.assertEqual(client.writes, writes)
        output = self.root / 'downloaded'
        storage.download(client, 7, 'experiment-parts.json', result['manifest_sha256'], output)
        self.assertEqual((output / 'run.zip').read_bytes(), original.read_bytes())
        part = client.saved['run.zip.part002']
        part['body'] = b'corrupted'
        with self.assertRaisesRegex(storage.StorageError, 'checksum'):
            storage.download(client, 7, 'experiment-parts.json', result['manifest_sha256'], self.root / 'bad')
        self.assertFalse((self.root / 'bad/run.zip').exists())

    def test_different_remote_name_is_never_overwritten(self):
        original = self.root / 'model.pt'
        original.write_bytes(b'original')
        client = MemoryGitHub()
        storage.upload(client, 7, [original], 'models.json', 100)
        before = {k: v.copy() for k, v in client.saved.items()}
        original.write_bytes(b'different')
        with self.assertRaisesRegex(storage.StorageError, 'differs'):
            storage.upload(client, 7, [original], 'models.json', 100)
        self.assertEqual(client.saved, before)

    def test_complete_run_bundle_preserves_final_and_recovery_bytes(self):
        source = self.root / 'source'
        names = ['run/checkpoints/final.pt', 'run/checkpoints/final.json', 'run/checkpoints/latest.json',
                 'run/config.json', 'run/run.json', 'run/metrics.jsonl', 'run/resume/update-000001.pt',
                 'run/resume/latest.json', 'registry.json', 'worker.log']
        for name in names:
            p = source / name
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(('unchanged:' + name).encode())
        archive = self.root / 'bundle.zip'
        result = storage.bundle(source, ['run', 'registry.json', 'worker.log'], archive)
        destination = self.root / 'restored'
        storage.restore_bundle(archive, result['sha256'], destination)
        self.assertEqual(result['files'], len(names))
        for name in names:
            self.assertEqual((destination / name).read_bytes(), (source / name).read_bytes())
        with self.assertRaises(storage.StorageError):
            storage.restore_bundle(archive, result['sha256'], destination)

    def test_bundle_rejects_credentials_active_writer_and_traversal(self):
        source = self.root / 'source'
        source.mkdir()
        (source / '.env').write_text('must not be read')
        with self.assertRaises(storage.StorageError):
            storage.bundle(source, ['.env'], self.root / 'env.zip')
        (source / '.writer.lock').write_text('{}')
        with self.assertRaises(storage.StorageError):
            storage.bundle(source, ['.writer.lock'], self.root / 'active.zip')
        archive = self.root / 'malicious.zip'
        with zipfile.ZipFile(archive, 'w') as z:
            z.writestr('../escape', b'no')
            z.writestr('bundle-files.json', '{}')
        with self.assertRaises(storage.StorageError):
            storage.restore_bundle(archive, storage.sha256(archive), self.root / 'unsafe')
        self.assertFalse((self.root / 'escape').exists())

    def test_api_secret_is_not_forwarded_to_asset_cdn(self):
        client = storage.GitHub('example/project', token='test-placeholder')
        requests = []
        class Response(io.BytesIO):
            def __init__(self, body, code, headers):
                super().__init__(body)
                self.code, self.headers = code, headers
        class Opener:
            def open(self, request, timeout):
                requests.append(request)
                if len(requests) == 1:
                    return Response(b'', 302, {'Location': 'https://release-assets.githubusercontent.com/file'})
                return Response(b'payload', 200, {})
        client.opener = Opener()
        client.fetch({'id': 1}, self.root / 'download')
        self.assertIsNotNone(requests[0].get_header('Authorization'))
        self.assertIsNone(requests[1].get_header('Authorization'))
        self.assertEqual((self.root / 'download').read_bytes(), b'payload')

    def test_record_only_does_not_execute_and_accepts_experiment_specific_config(self):
        config = self.root / 'experiment.json'
        config.write_text('{"method":"future-method","seed":987,"updates":3}')
        flag = self.root / 'would-execute'
        record = self.root / 'invocation'
        args = [sys.executable, str(SCRIPTS / 'run_recorded.py'), '--config', str(config),
                '--record-dir', str(record), '--record-only', '--', sys.executable, '-c',
                'from pathlib import Path; Path(' + repr(str(flag)) + ').touch()']
        subprocess.run(args, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        self.assertFalse(flag.exists())
        saved = json.loads((record / 'invocation.json').read_text())
        self.assertEqual(saved['declared_config']['updates'], 3)
        self.assertEqual(saved['argv'], args[args.index('--') + 1:])
        before = (record / 'invocation.json').read_bytes()
        second = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        self.assertNotEqual(second.returncode, 0)
        self.assertEqual((record / 'invocation.json').read_bytes(), before)


if __name__ == '__main__':
    unittest.main()
