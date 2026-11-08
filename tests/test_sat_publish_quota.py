import unittest
from unittest.mock import patch, MagicMock
import json
import os
import sys
import tempfile
from pathlib import Path
from datetime import datetime, timezone
import io

try:
    from sat import publish
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from sat import publish

class TestSatPublishFailClosed(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.out_dir = Path(self.tmpdir.name) / 'out'
        self.state_dir = Path(self.tmpdir.name) / 'state'
        self.out_dir.mkdir(parents=True)
        self.state_dir.mkdir(parents=True)

        # Create valid dummy manifest with new observation
        self.manifest_path = self.out_dir / 'manifest.json'
        self.asset_bytes = b'fake png bytes'
        self.asset_hash = '86610c40efe63f0a46c58c4b605c164b4ffa3a3ad3f1dcf13e6ba4c59cb3ce16'
        self.asset_name = f'err_20261105T1200Z_{self.asset_hash[:16]}.png'
        self.manifest_data = {
            'schema_version': '2.0.0',
            'generated_at': '2026-11-05T12:10:00Z',
            'err': {
                'latest': {
                    'asset': self.asset_name,
                    'scan_end': '2026-11-05T12:09:00Z',
                    'slot': '2026-11-05T12:00:00Z'
                },
                'timeline': [{'asset': self.asset_name}],
                'accumulations': {}
            },
            'cloud': {
                'slot': '2026-11-05T12:00:00Z',
                'assets': {},
                'ir_timeline': []
            },
            'assets': {
                self.asset_name: {
                    'sha256': self.asset_hash,
                    'bytes': len(self.asset_bytes)
                }
            }
        }
        self.manifest_path.write_text(json.dumps(self.manifest_data), encoding='utf-8')
        (self.out_dir / 'assets').mkdir(parents=True, exist_ok=True)
        (self.out_dir / 'assets' / self.asset_name).write_bytes(self.asset_bytes)

        os.environ['WORKER_URL'] = 'https://fake-worker.test'
        os.environ['SAT_PUBLISH_TOKEN'] = 'secret-test-token'

    def tearDown(self):
        self.tmpdir.cleanup()
        os.environ.pop('WORKER_URL', None)
        os.environ.pop('SAT_PUBLISH_TOKEN', None)

    @patch('requests.Session')
    def test_fail_closed_on_http_500(self, mock_session_cls):
        mock_session = MagicMock()
        mock_session_cls.return_value = mock_session

        mock_live_m = MagicMock()
        mock_live_m.status_code = 200
        mock_live_m.json.return_value = {'err': {'latest': {'scan_end': '2026-11-05T11:50:00Z'}}}

        mock_quota_resp = MagicMock()
        mock_quota_resp.status_code = 500

        def get_handler(url, **kwargs):
            if 'quota.json' in url:
                return mock_quota_resp
            return mock_live_m

        mock_session.get.side_effect = get_handler
        sys.argv = ['publish.py', '--out', str(self.out_dir), '--state', str(self.state_dir), '--send']

        with self.assertRaises(SystemExit) as cm:
            publish.main()
        self.assertIn('FAIL-CLOSED: Live quota endpoint returned HTTP 500', str(cm.exception))
        mock_session.put.assert_not_called()

    @patch('requests.Session')
    def test_fail_closed_on_network_error(self, mock_session_cls):
        mock_session = MagicMock()
        mock_session_cls.return_value = mock_session

        import requests
        mock_live_m = MagicMock()
        mock_live_m.status_code = 200
        mock_live_m.json.return_value = {'err': {'latest': {'scan_end': '2026-11-05T11:50:00Z'}}}

        def get_handler(url, **kwargs):
            if 'quota.json' in url:
                raise requests.RequestException('DNS lookup failure')
            return mock_live_m

        mock_session.get.side_effect = get_handler
        sys.argv = ['publish.py', '--out', str(self.out_dir), '--state', str(self.state_dir), '--send']

        with self.assertRaises(SystemExit) as cm:
            publish.main()
        self.assertIn('FAIL-CLOSED: Network error querying live quota endpoint', str(cm.exception))
        mock_session.put.assert_not_called()

    @patch('requests.Session')
    def test_fail_closed_on_date_mismatch(self, mock_session_cls):
        mock_session = MagicMock()
        mock_session_cls.return_value = mock_session

        mock_live_m = MagicMock()
        mock_live_m.status_code = 200
        mock_live_m.json.return_value = {'err': {'latest': {'scan_end': '2026-11-05T11:50:00Z'}}}

        mock_quota_resp = MagicMock()
        mock_quota_resp.status_code = 200
        mock_quota_resp.json.return_value = {
            'date_utc': '1999-01-01', # Mismatched date
            'daily_kv_writes': 10,
            'daily_cap': 750
        }

        def get_handler(url, **kwargs):
            if 'quota.json' in url:
                return mock_quota_resp
            return mock_live_m

        mock_session.get.side_effect = get_handler
        sys.argv = ['publish.py', '--out', str(self.out_dir), '--state', str(self.state_dir), '--send']

        with self.assertRaises(SystemExit) as cm:
            publish.main()
        self.assertIn('FAIL-CLOSED: Quota date mismatch', str(cm.exception))
        mock_session.put.assert_not_called()

    @patch('requests.Session')
    def test_fail_closed_on_malformed_json(self, mock_session_cls):
        mock_session = MagicMock()
        mock_session_cls.return_value = mock_session

        mock_live_m = MagicMock()
        mock_live_m.status_code = 200
        mock_live_m.json.return_value = {'err': {'latest': {'scan_end': '2026-11-05T11:50:00Z'}}}

        mock_quota_resp = MagicMock()
        mock_quota_resp.status_code = 200
        mock_quota_resp.json.return_value = {'unexpected': 'format'}

        def get_handler(url, **kwargs):
            if 'quota.json' in url:
                return mock_quota_resp
            return mock_live_m

        mock_session.get.side_effect = get_handler
        sys.argv = ['publish.py', '--out', str(self.out_dir), '--state', str(self.state_dir), '--send']

        with self.assertRaises(SystemExit) as cm:
            publish.main()
        self.assertIn('FAIL-CLOSED: Quota payload invalid or malformed', str(cm.exception))
        mock_session.put.assert_not_called()

    @patch('requests.Session')
    def test_throttled_when_server_cap_exceeded(self, mock_session_cls):
        mock_session = MagicMock()
        mock_session_cls.return_value = mock_session

        mock_live_m = MagicMock()
        mock_live_m.status_code = 200
        mock_live_m.json.return_value = {'err': {'latest': {'scan_end': '2026-11-05T11:50:00Z'}}}

        today = datetime.now(timezone.utc).strftime('%Y-%m-%d')
        mock_quota_resp = MagicMock()
        mock_quota_resp.status_code = 200
        mock_quota_resp.json.return_value = {
            'date_utc': today,
            'daily_kv_writes': 750,
            'daily_cap': 750
        }

        def get_handler(url, **kwargs):
            if 'quota.json' in url:
                return mock_quota_resp
            return mock_live_m

        mock_session.get.side_effect = get_handler
        sys.argv = ['publish.py', '--out', str(self.out_dir), '--state', str(self.state_dir), '--send']

        captured = io.StringIO()
        with patch('sys.stdout', captured):
            publish.main()

        out_json = json.loads(captured.getvalue().strip())
        self.assertEqual(out_json.get('status'), 'THROTTLED_GLOBAL_SERVER_QUOTA_GUARD')
        mock_session.put.assert_not_called()

    @patch('requests.Session')
    def test_passes_and_publishes_when_quota_healthy(self, mock_session_cls):
        mock_session = MagicMock()
        mock_session_cls.return_value = mock_session

        mock_live_m = MagicMock()
        mock_live_m.status_code = 200
        mock_live_m.json.return_value = {'err': {'latest': {'scan_end': '2026-11-05T11:50:00Z'}}}

        today = datetime.now(timezone.utc).strftime('%Y-%m-%d')
        mock_quota_resp = MagicMock()
        mock_quota_resp.status_code = 200
        mock_quota_resp.json.return_value = {
            'date_utc': today,
            'daily_kv_writes': 10,
            'daily_cap': 750
        }

        def get_handler(url, **kwargs):
            if 'quota.json' in url:
                return mock_quota_resp
            return mock_live_m

        mock_session.get.side_effect = get_handler

        mock_put_resp = MagicMock()
        mock_put_resp.status_code = 200
        mock_put_resp.json.return_value = {'published': True}
        mock_session.put.return_value = mock_put_resp

        sys.argv = ['publish.py', '--out', str(self.out_dir), '--state', str(self.state_dir), '--send']

        captured = io.StringIO()
        with patch('sys.stdout', captured):
            publish.main()

        out_json = json.loads(captured.getvalue().strip())
        self.assertEqual(out_json.get('uploaded_assets'), 1)
        self.assertEqual(out_json.get('kv_writes'), 2)
        self.assertEqual(mock_session.put.call_count, 2)

    @patch('requests.Session')
    def test_passes_and_publishes_bundle_with_2_kv_writes(self, mock_session_cls):
        mock_session = MagicMock()
        mock_session_cls.return_value = mock_session

        mock_live_m = MagicMock()
        mock_live_m.status_code = 200
        mock_live_m.json.return_value = {'err': {'latest': {'scan_end': '2026-11-05T11:50:00Z'}}}

        today = datetime.now(timezone.utc).strftime('%Y-%m-%d')
        mock_quota_resp = MagicMock()
        mock_quota_resp.status_code = 200
        mock_quota_resp.json.return_value = {
            'date_utc': today,
            'daily_kv_writes': 20,
            'daily_cap': 750
        }

        def get_handler(url, **kwargs):
            if 'quota.json' in url:
                return mock_quota_resp
            return mock_live_m

        mock_session.get.side_effect = get_handler

        mock_put_resp = MagicMock()
        mock_put_resp.status_code = 200
        mock_put_resp.json.return_value = {'published': True}
        mock_session.put.return_value = mock_put_resp

        import tarfile, hashlib
        assets = {}
        names = []
        for pfx in ['err', 'cloud', 'cloud2', 'ir']:
            b = f'bytes for {pfx}'.encode('ascii')
            h = hashlib.sha256(b).hexdigest()
            n = f'{pfx}_20261105T1200Z_{h[:16]}.png'
            assets[n] = {'bytes': len(b), 'sha256': h}
            names.append(n)
            (self.out_dir / 'assets' / n).write_bytes(b)

        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode='w', format=tarfile.USTAR_FORMAT) as tar:
            for n in names:
                data = (self.out_dir / 'assets' / n).read_bytes()
                ti = tarfile.TarInfo(name=n)
                ti.size = len(data)
                ti.mtime = 1700000000
                ti.mode = 0o644
                tar.addfile(ti, io.BytesIO(data))
        tar_bytes = buf.getvalue()
        tar_hash = hashlib.sha256(tar_bytes).hexdigest()
        bname = f'scan_20261105T1200Z_{tar_hash[:16]}.tar'
        (self.out_dir / 'bundles').mkdir(parents=True, exist_ok=True)
        (self.out_dir / 'bundles' / bname).write_bytes(tar_bytes)

        for n in names:
            assets[n]['bundle'] = bname

        manifest = {
            'schema_version': '2.0.0',
            'generated_at': '2026-11-05T12:10:00Z',
            'err': {'latest': {'asset': names[0], 'slot': '2026-11-05T12:00:00Z', 'scan_end': '2026-11-05T12:09:00Z'}, 'timeline': [{'asset': names[0]}], 'accumulations': {}},
            'cloud': {'slot': '2026-11-05T12:00:00Z', 'assets': {'classes_height_temp': names[1], 'cod_phase_mask': names[2]}, 'ir_timeline': [{'asset': names[3]}]},
            'assets': assets,
            'bundles': {bname: {'sha256': tar_hash, 'bytes': len(tar_bytes), 'files': names}}
        }
        self.manifest_path.write_text(json.dumps(manifest), encoding='utf-8')

        sys.argv = ['publish.py', '--out', str(self.out_dir), '--state', str(self.state_dir), '--send']

        captured = io.StringIO()
        with patch('sys.stdout', captured):
            publish.main()

        out_json = json.loads(captured.getvalue().strip())
        self.assertEqual(out_json.get('uploaded_bundles'), 1)
        self.assertEqual(out_json.get('uploaded_assets'), 0)
        self.assertEqual(out_json.get('kv_writes'), 2)
        self.assertEqual(mock_session.put.call_count, 2)
        first_call = mock_session.put.call_args_list[0]
        self.assertIn(f'/admin/sat/bundle/{bname}', first_call[0][0])
        self.assertEqual(first_call[1]['headers']['Content-Type'], 'application/x-tar')
        second_call = mock_session.put.call_args_list[1]
        self.assertIn('/admin/sat/manifest', second_call[0][0])
        pub_saved = json.loads((self.state_dir / 'published.json').read_text(encoding='utf-8'))
        self.assertIn(bname, pub_saved)
        self.assertNotIn(names[0], pub_saved)

    @patch('requests.Session')
    def test_graceful_throttle_on_http_503_storage_unavailable(self, mock_session_cls):
        mock_session = MagicMock()
        mock_session_cls.return_value = mock_session

        mock_live_m = MagicMock()
        mock_live_m.status_code = 200
        mock_live_m.json.return_value = {'err': {'latest': {'scan_end': '2026-11-05T11:50:00Z'}}}

        today = datetime.now(timezone.utc).strftime('%Y-%m-%d')
        mock_quota_resp = MagicMock()
        mock_quota_resp.status_code = 200
        mock_quota_resp.json.return_value = {
            'date_utc': today,
            'daily_kv_writes': 100,
            'daily_cap': 750
        }

        def get_handler(url, **kwargs):
            if 'quota.json' in url:
                return mock_quota_resp
            return mock_live_m

        mock_session.get.side_effect = get_handler

        mock_put_resp = MagicMock()
        mock_put_resp.status_code = 503
        mock_put_resp.text = '{"error":"publication_storage_unavailable"}'
        mock_session.put.return_value = mock_put_resp

        sys.argv = ['publish.py', '--out', str(self.out_dir), '--state', str(self.state_dir), '--send']

        captured = io.StringIO()
        with patch('sys.stdout', captured):
            publish.main()

        out_json = json.loads(captured.getvalue().strip())
        self.assertEqual(out_json.get('status'), 'THROTTLED_STORAGE_QUOTA_REACHED')
        self.assertTrue((self.state_dir / 'published.json').exists())

if __name__ == '__main__':
    unittest.main()

