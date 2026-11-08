"""Upload new satellite assets, then the manifest, to the Niño Costero Worker.

Env:
  SAT_PUBLISH_TOKEN or PUBLISH_TOKEN (secret)
  WORKER_URL (https://<worker-host>; only origin is used)

Features:
- FAIL-CLOSED quota check: Refuses publication if live server quota cannot be verified.
- Global server-authoritative KV write quota guard (/v3/sat/quota.json).
- Zero local cache dependency for quota enforcement.
- Skip publication entirely (0 KV writes) when no new data exists.
- Content-addressed asset deduplication via <state>/published.json.
- Atomic publication: all new assets verified before manifest is committed.
- Zero secrets printed in logs or audit outputs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import urlsplit

import requests

TTL_H = 48
DAILY_KV_WRITE_CAP = 750


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='build/sat')
    ap.add_argument('--state', default='.sat_state')
    ap.add_argument('--send', action='store_true')
    a = ap.parse_args()
    out, st = Path(a.out), Path(a.state)
    manifest_path = out / 'manifest.json'
    if not manifest_path.exists():
        sys.exit(f'manifest not found at {manifest_path}')

    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    referenced = set()
    for f in manifest.get('err', {}).get('timeline', []):
        referenced.add(f['asset'])
    for v in manifest.get('err', {}).get('accumulations', {}).values():
        if v.get('asset'):
            referenced.add(v['asset'])
    if manifest.get('cloud'):
        referenced |= set(manifest['cloud'].get('assets', {}).values())
        referenced |= {f['asset'] for f in manifest['cloud'].get('ir_timeline', [])}
    if manifest.get('err', {}).get('latest'):
        referenced.add(manifest['err']['latest']['asset'])

    missing = [n for n in referenced if not (out / 'assets' / n).exists()]
    if missing:
        sys.exit(f'manifest references missing assets: {missing[:5]}')

    for n in referenced:
        b = (out / 'assets' / n).read_bytes()
        spec = manifest.get('assets', {}).get(n, {})
        digest = hashlib.sha256(b).hexdigest()
        if spec.get('sha256') != digest or spec.get('bytes') != len(b) or not n.endswith(f'_{digest[:16]}.png'):
            sys.exit('asset integrity failed: ' + n)

    manifest_bundles = manifest.get('bundles', {})
    for bname, bmeta in manifest_bundles.items():
        bpath = out / 'bundles' / bname
        if not bpath.exists():
            sys.exit(f'manifest references missing bundle: {bname}')
        bb = bpath.read_bytes()
        digest = hashlib.sha256(bb).hexdigest()
        if bmeta.get('sha256') != digest or bmeta.get('bytes') != len(bb) or not bname.endswith(f'_{digest[:16]}.tar'):
            sys.exit(f'bundle integrity failed: {bname}')

    pub_path = st / 'published.json'
    now = datetime.now(timezone.utc)
    today_utc = now.strftime('%Y-%m-%d')
    published = json.loads(pub_path.read_text(encoding='utf-8')) if pub_path.exists() else {}

    # Assets and bundles expire after 48h. Filter out fresh ones (published within 33 hours)
    fresh = {k: v for k, v in published.items()
             if datetime.fromisoformat(v) > now - timedelta(hours=TTL_H - 15)}

    bundles_to_upload = sorted([bn for bn in manifest_bundles if bn not in fresh])
    unbundled_to_upload = sorted([n for n in referenced
                                  if not manifest.get('assets', {}).get(n, {}).get('bundle') and n not in fresh])

    planned_writes = len(bundles_to_upload) + len(unbundled_to_upload) + 1  # bundles + unbundled assets + manifest
    report = {
        'referenced': len(referenced),
        'bundles_to_upload': len(bundles_to_upload),
        'unbundled_to_upload': len(unbundled_to_upload),
        'kv_writes_planned': planned_writes,
        'bytes_planned': sum((out / 'bundles' / bn).stat().st_size for bn in bundles_to_upload) +
                         sum((out / 'assets' / n).stat().st_size for n in unbundled_to_upload),
        'daily_cap': DAILY_KV_WRITE_CAP,
    }

    token = os.environ.get('SAT_PUBLISH_TOKEN') or os.environ.get('PUBLISH_TOKEN')
    url = os.environ.get('WORKER_URL')
    if not token or not url:
        if not a.send:
            print(json.dumps({'dry_run': True, **report}))
            return
        sys.exit('SAT_PUBLISH_TOKEN/PUBLISH_TOKEN and WORKER_URL must be configured')

    u = urlsplit(url)
    if u.scheme != 'https' or not u.hostname:
        sys.exit('WORKER_URL must be https')
    base = f'https://{u.hostname}'

    s = requests.Session()
    s.headers['Authorization'] = f'Bearer {token}'
    s.headers['User-Agent'] = 'NinoCostero-SatPublisher/2.3.2'

    # -------------------------------------------------------------
    # FAIL-CLOSED QUOTA CHECK
    # Must obtain a valid, verifiable, current UTC-date quota response
    # from the Cloudflare Worker. If unreachable, non-200, invalid JSON,
    # or date mismatch, it MUST FAIL-CLOSED (refuse to upload any assets).
    # -------------------------------------------------------------
    server_writes_today = None
    server_cap = DAILY_KV_WRITE_CAP

    try:
        qr = s.get(f'{base}/v3/sat/quota.json', timeout=15)
        if qr.status_code != 200:
            msg = f'FAIL-CLOSED: Live quota endpoint returned HTTP {qr.status_code}'
            print(json.dumps({'status': 'ABORTED_FAIL_CLOSED', 'error': msg, 'http_status': qr.status_code}))
            sys.exit(msg)
        qd = qr.json()
        if not isinstance(qd, dict) or 'daily_kv_writes' not in qd or 'date_utc' not in qd:
            msg = 'FAIL-CLOSED: Quota payload invalid or malformed'
            print(json.dumps({'status': 'ABORTED_FAIL_CLOSED', 'error': msg, 'payload': qd}))
            sys.exit(msg)
        if qd['date_utc'] != today_utc:
            msg = f'FAIL-CLOSED: Quota date mismatch (server {qd["date_utc"]} vs client {today_utc})'
            print(json.dumps({'status': 'ABORTED_FAIL_CLOSED', 'error': msg}))
            sys.exit(msg)

        server_writes_today = int(qd['daily_kv_writes'])
        server_cap = int(qd.get('daily_cap', DAILY_KV_WRITE_CAP))
        report['server_daily_writes_used'] = server_writes_today
        report['server_quota_status'] = qd.get('status')
        report['server_quota_date'] = qd.get('date_utc')
    except requests.RequestException as e:
        msg = f'FAIL-CLOSED: Network error querying live quota endpoint: {e}'
        print(json.dumps({'status': 'ABORTED_FAIL_CLOSED', 'error': str(e)}))
        sys.exit(msg)

    if server_writes_today is None:
        sys.exit('FAIL-CLOSED: Quota verification unconfirmed')

    if server_writes_today + planned_writes > server_cap:
        print(json.dumps({
            'status': 'THROTTLED_GLOBAL_SERVER_QUOTA_GUARD',
            'warning': f'Cloudflare KV daily write limit reached on server ({server_writes_today}/{server_cap}). '
                       f'Skipping publication to guarantee S/ 0 free tier.',
            **report
        }))
        return

    # Check if there is really anything new to publish
    if len(bundles_to_upload) == 0 and len(unbundled_to_upload) == 0:
        try:
            cur_r = s.get(f'{base}/v3/sat/manifest.json', timeout=10)
            if cur_r.status_code == 200:
                cur_m = cur_r.json()
                cur_slot = cur_m.get('err', {}).get('latest', {}).get('slot')
                cur_cloud = cur_m.get('cloud', {}).get('slot')
                new_slot = manifest.get('err', {}).get('latest', {}).get('slot')
                new_cloud = manifest.get('cloud', {}).get('slot')
                if cur_slot == new_slot and cur_cloud == new_cloud:
                    print(json.dumps({
                        'status': 'SKIPPED_NO_NEW_DATA',
                        'note': f'NOAA observation already live ({new_slot}); 0 KV writes performed.',
                        'uploaded_bundles': 0,
                        'uploaded_assets': 0,
                        'kv_writes': 0,
                        **report
                    }))
                    return
        except Exception:
            pass

    if not a.send:
        print(json.dumps({'dry_run': True, **report}))
        return

    t0, sent = time.perf_counter(), 0

    def save_published():
        st.mkdir(parents=True, exist_ok=True)
        pub_path.write_text(json.dumps({k: v for k, v in fresh.items()
                                        if datetime.fromisoformat(v) > now - timedelta(hours=TTL_H)}),
                            encoding='utf-8')

    # 1. Upload new bundles
    for bn in bundles_to_upload:
        bundle_bytes = (out / 'bundles' / bn).read_bytes()
        r = s.put(f'{base}/admin/sat/bundle/{bn}', data=bundle_bytes,
                  headers={'Content-Type': 'application/x-tar'}, timeout=30, allow_redirects=False)
        if r.status_code in (429, 503) and ('publication_storage_unavailable' in r.text or 'quota' in r.text.lower()):
            save_published()
            print(json.dumps({
                'status': 'THROTTLED_STORAGE_QUOTA_REACHED',
                'warning': f'Cloudflare KV storage/write limit reached during bundle upload ({bn}). State preserved.',
                'http_status': r.status_code,
                **report
            }))
            return
        if r.status_code != 200:
            save_published()
            sys.exit(f'bundle upload failed HTTP {r.status_code} for {bn}: {r.text[:200]}')
        fresh[bn] = now.isoformat()
        save_published()
        sent += 1

    # 2. Upload standalone unbundled assets (if any)
    for n in unbundled_to_upload:
        asset_bytes = (out / 'assets' / n).read_bytes()
        r = s.put(f'{base}/admin/sat/asset/{n}', data=asset_bytes,
                  headers={'Content-Type': 'image/png'}, timeout=30, allow_redirects=False)
        if r.status_code in (429, 503) and ('publication_storage_unavailable' in r.text or 'quota' in r.text.lower()):
            save_published()
            print(json.dumps({
                'status': 'THROTTLED_STORAGE_QUOTA_REACHED',
                'warning': f'Cloudflare KV storage/write limit reached during asset upload ({n}). State preserved.',
                'http_status': r.status_code,
                **report
            }))
            return
        if r.status_code != 200:
            save_published()
            sys.exit(f'asset upload failed HTTP {r.status_code} for {n}: {r.text[:200]}')
        fresh[n] = now.isoformat()
        save_published()
        sent += 1

    # 3. Upload manifest with batch write count for server quota tracking
    manifest['processing'] = manifest.get('processing', {})
    manifest['processing']['run_kv_writes'] = sent + 1
    manifest_bytes = json.dumps(manifest, separators=(',', ':')).encode('utf-8')

    r = s.put(f'{base}/admin/sat/manifest', data=manifest_bytes,
              headers={'Content-Type': 'application/json'}, timeout=30, allow_redirects=False)
    if r.status_code == 409 and 'older_manifest_rejected' in r.text:
        save_published()
        print(json.dumps({'status': 'SKIPPED_OLDER_MANIFEST', 'note': 'Remote manifest is newer or identical'}))
        return
    if r.status_code in (429, 503) and ('publication_storage_unavailable' in r.text or 'quota' in r.text.lower()):
        save_published()
        print(json.dumps({
            'status': 'THROTTLED_STORAGE_QUOTA_REACHED',
            'warning': 'Cloudflare KV storage/write limit reached during manifest upload. State preserved.',
            'http_status': r.status_code,
            **report
        }))
        return
    if r.status_code != 200:
        save_published()
        sys.exit(f'manifest upload failed HTTP {r.status_code}: {r.text[:300]}')

    # Update state
    save_published()

    print(json.dumps({
        'status': 'PUBLISHED_SUCCESS',
        'uploaded_bundles': len(bundles_to_upload),
        'uploaded_assets': len(unbundled_to_upload),
        'kv_writes': sent + 1,
        'server_writes_today_est': server_writes_today + sent + 1,
        'seconds': round(time.perf_counter() - t0, 2),
        **report
    }))


if __name__ == '__main__':
    main()
