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

    @patch('requests.Session')
    def test_regression_failed_run_published_json_no_repeated_writes(self, mock_session_cls):
        """Regression test for the 2026-11-08 historical published.json bug:
        1) Bundled member PNGs must never be written into published.json as standalone keys.
        2) If a run uploads bundles (e.g. acc_*.tar + scan_*.tar) and then fails on
           PUT /admin/sat/manifest (e.g. HTTP 409 asset_not_ready or HTTP 500),
           published.json must still persist the uploaded bundles before exiting.
        3) Subsequent runs (retry or next 10-min cycle) must not re-upload already
           persisted bundles (0 repeated KV writes).
        """
        import tarfile, hashlib

        mock_session = MagicMock()
        mock_session_cls.return_value = mock_session

        today = datetime.now(timezone.utc).strftime('%Y-%m-%d')
        mock_quota_resp = MagicMock()
        mock_quota_resp.status_code = 200
        mock_quota_resp.json.return_value = {
            'date_utc': today,
            'daily_kv_writes': 36,
            'daily_cap': 750
        }
        mock_live_m = MagicMock()
        mock_live_m.status_code = 200
        mock_live_m.json.return_value = {'err': {'latest': {'slot': '2026-11-07T23:50:00Z'}}}

        def get_handler(url, **kwargs):
            if 'quota.json' in url:
                return mock_quota_resp
            return mock_live_m

        mock_session.get.side_effect = get_handler

        def make_bundle(prefix, slot_tag, member_files):
            buf = io.BytesIO()
            with tarfile.open(fileobj=buf, mode='w', format=tarfile.USTAR_FORMAT) as tar:
                for fname, fbytes in sorted(member_files.items()):
                    ti = tarfile.TarInfo(name=fname)
                    ti.size = len(fbytes)
                    ti.mtime = 1700000000
                    ti.mode = 0o644
                    tar.addfile(ti, io.BytesIO(fbytes))
            raw = buf.getvalue()
            h = hashlib.sha256(raw).hexdigest()
            bname = f'{prefix}_{slot_tag}_{h[:16]}.tar'
            (self.out_dir / 'bundles').mkdir(parents=True, exist_ok=True)
            (self.out_dir / 'bundles' / bname).write_bytes(raw)
            return bname, h, len(raw)

        # Build Cycle 1 (00:00Z): 1 hourly acc bundle + 1 scan bundle
        acc_bytes = b'acc1h_png_bytes_0000Z'
        acc_h = hashlib.sha256(acc_bytes).hexdigest()
        acc_png = f'acc1h_20261108T0000Z_{acc_h[:16]}.png'
        (self.out_dir / 'assets' / acc_png).write_bytes(acc_bytes)
        acc_bname, acc_bh, acc_blen = make_bundle('acc', '20261108T0000Z', {acc_png: acc_bytes})

        err0_bytes = b'err_png_bytes_0000Z'
        err0_h = hashlib.sha256(err0_bytes).hexdigest()
        err0_png = f'err_20261108T0000Z_{err0_h[:16]}.png'
        (self.out_dir / 'assets' / err0_png).write_bytes(err0_bytes)
        scan0_bname, scan0_bh, scan0_blen = make_bundle('scan', '20261108T0000Z', {err0_png: err0_bytes})

        manifest_cycle1 = {
            'schema_version': '2.0.0',
            'generated_at': '2026-11-08T00:04:58Z',
            'err': {
                'latest': {'asset': err0_png, 'slot': '2026-11-08T00:00:00Z', 'scan_end': '2026-11-08T00:09:52Z'},
                'timeline': [{'asset': err0_png}],
                'accumulations': {'1': {'complete': True, 'asset': acc_png}}
            },
            'cloud': {'slot': '2026-11-08T00:00:00Z', 'assets': {}, 'ir_timeline': []},
            'assets': {
                acc_png: {'sha256': acc_h, 'bytes': len(acc_bytes), 'bundle': acc_bname},
                err0_png: {'sha256': err0_h, 'bytes': len(err0_bytes), 'bundle': scan0_bname},
            },
            'bundles': {
                acc_bname: {'sha256': acc_bh, 'bytes': acc_blen, 'files': [acc_png]},
                scan0_bname: {'sha256': scan0_bh, 'bytes': scan0_blen, 'files': [err0_png]},
            }
        }
        self.manifest_path.write_text(json.dumps(manifest_cycle1), encoding='utf-8')

        # Simulate Cycle 1: bundle uploads succeed (200), but manifest upload fails with HTTP 409
        def put_cycle1(url, **kwargs):
            resp = MagicMock()
            if '/admin/sat/bundle/' in url:
                resp.status_code = 200
                resp.text = '{"published":true}'
            else:
                resp.status_code = 409
                resp.text = '{"error":"asset_not_ready","asset":"err_20261107T2350Z_3bcf44dd9bd0845a.png"}'
            return resp

        mock_session.put.side_effect = put_cycle1
        sys.argv = ['publish.py', '--out', str(self.out_dir), '--state', str(self.state_dir), '--send']

        with self.assertRaises(SystemExit) as cm:
            publish.main()
        self.assertIn('manifest upload failed HTTP 409', str(cm.exception))

        # Verify published.json saved both uploaded bundles BEFORE exiting and did NOT poison member PNG names
        pub_after_fail = json.loads((self.state_dir / 'published.json').read_text(encoding='utf-8'))
        self.assertIn(acc_bname, pub_after_fail)
        self.assertIn(scan0_bname, pub_after_fail)
        self.assertNotIn(acc_png, pub_after_fail)
        self.assertNotIn(err0_png, pub_after_fail)

        # Simulate Cycle 2 (00:10Z intra-hour cycle): same hourly acc_bname + new scan1_bname bundling timeline
        err1_bytes = b'err_png_bytes_0010Z'
        err1_h = hashlib.sha256(err1_bytes).hexdigest()
        err1_png = f'err_20261108T0010Z_{err1_h[:16]}.png'
        (self.out_dir / 'assets' / err1_png).write_bytes(err1_bytes)
        scan1_bname, scan1_bh, scan1_blen = make_bundle(
            'scan', '20261108T0010Z', {err0_png: err0_bytes, err1_png: err1_bytes}
        )

        manifest_cycle2 = {
            'schema_version': '2.0.0',
            'generated_at': '2026-11-08T00:14:58Z',
            'err': {
                'latest': {'asset': err1_png, 'slot': '2026-11-08T00:10:00Z', 'scan_end': '2026-11-08T00:19:52Z'},
                'timeline': [{'asset': err0_png}, {'asset': err1_png}],
                'accumulations': {'1': {'complete': True, 'asset': acc_png}}
            },
            'cloud': {'slot': '2026-11-08T00:10:00Z', 'assets': {}, 'ir_timeline': []},
            'assets': {
                acc_png: {'sha256': acc_h, 'bytes': len(acc_bytes), 'bundle': acc_bname},
                err0_png: {'sha256': err0_h, 'bytes': len(err0_bytes), 'bundle': scan1_bname},
                err1_png: {'sha256': err1_h, 'bytes': len(err1_bytes), 'bundle': scan1_bname},
            },
            'bundles': {
                acc_bname: {'sha256': acc_bh, 'bytes': acc_blen, 'files': [acc_png]},
                scan1_bname: {'sha256': scan1_bh, 'bytes': scan1_blen, 'files': [err0_png, err1_png]},
            }
        }
        self.manifest_path.write_text(json.dumps(manifest_cycle2), encoding='utf-8')

        mock_session.put.reset_mock()
        mock_ok = MagicMock()
        mock_ok.status_code = 200
        mock_ok.text = '{"published":true}'
        mock_session.put.side_effect = None
        mock_session.put.return_value = mock_ok

        captured = io.StringIO()
        with patch('sys.stdout', captured):
            publish.main()

        out_json = json.loads(captured.getvalue().strip())
        self.assertEqual(out_json.get('status'), 'PUBLISHED_SUCCESS')
        # acc_bname was already uploaded in the failed run and saved in published.json -> NOT re-uploaded!
        self.assertEqual(out_json.get('uploaded_bundles'), 1)
        self.assertEqual(out_json.get('uploaded_assets'), 0)
        self.assertEqual(out_json.get('kv_writes'), 2)
        put_urls = [c[0][0] for c in mock_session.put.call_args_list]
        self.assertEqual(len(put_urls), 2)
        self.assertTrue(any(scan1_bname in u for u in put_urls))
        self.assertFalse(any(acc_bname in u for u in put_urls))

if __name__ == '__main__':
    unittest.main()

