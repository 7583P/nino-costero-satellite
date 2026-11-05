"""Upload new satellite assets, then the manifest, to the Niño Costero Worker.

Env:
  SAT_PUBLISH_TOKEN or PUBLISH_TOKEN (secret)
  WORKER_URL (https://<worker-host>; only origin is used)

Features:
- Content-addressed asset deduplication via <state>/published.json.
- Daily KV write quota guard (<state>/kv_quota.json) to strictly safeguard S/ 0 budget.
- Atomic publication: all new assets verified before the manifest is committed.
- Replay / out-of-order rejection handled in synchronization with Worker.
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
DAILY_KV_WRITE_CAP = 750  # Hard daily cap to preserve 250 free writes buffer (out of 1000/day free limit)


def load_quota(path: Path, now: datetime) -> dict:
    today = now.strftime('%Y-%m-%d')
    if path.exists():
        try:
            d = json.loads(path.read_text(encoding='utf-8'))
            if d.get('date') == today:
                return d
        except Exception:
            pass
    return {'date': today, 'writes': 0, 'runs': 0, 'updated_at': now.isoformat()}


def save_quota(path: Path, quota: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(quota, indent=2), encoding='utf-8')


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

    pub_path = st / 'published.json'
    now = datetime.now(timezone.utc)
    published = json.loads(pub_path.read_text(encoding='utf-8')) if pub_path.exists() else {}

    # Assets expire after 48h. Filter out fresh ones (published within 33 hours)
    fresh = {k: v for k, v in published.items()
             if datetime.fromisoformat(v) > now - timedelta(hours=TTL_H - 15)}
    new = sorted(n for n in referenced if n not in fresh)

    # Check daily KV quota guard
    quota_path = st / 'kv_quota.json'
    quota = load_quota(quota_path, now)
    planned_writes = len(new) + 1  # assets + manifest

    report = {
        'referenced': len(referenced),
        'to_upload': len(new),
        'kv_writes_planned': planned_writes,
        'bytes_planned': sum((out / 'assets' / n).stat().st_size for n in new),
        'daily_writes_used_before': quota['writes'],
        'daily_cap': DAILY_KV_WRITE_CAP,
    }

    if quota['writes'] + planned_writes > DAILY_KV_WRITE_CAP:
        print(json.dumps({
            'status': 'THROTTLED_QUOTA_GUARD',
            'warning': f'Daily KV write limit reached ({quota["writes"]}/{DAILY_KV_WRITE_CAP}). '
                       f'Skipping publication to guarantee S/ 0 free tier.',
            **report
        }))
        return

    if not a.send:
        print(json.dumps({'dry_run': True, **report}))
        return

    token = os.environ.get('SAT_PUBLISH_TOKEN') or os.environ.get('PUBLISH_TOKEN')
    url = os.environ.get('WORKER_URL')
    if not token or not url:
        sys.exit('SAT_PUBLISH_TOKEN/PUBLISH_TOKEN and WORKER_URL must be configured')

    u = urlsplit(url)
    if u.scheme != 'https' or not u.hostname:
        sys.exit('WORKER_URL must be https')
    base = f'https://{u.hostname}'

    s = requests.Session()
    s.headers['Authorization'] = f'Bearer {token}'
    s.headers['User-Agent'] = 'NinoCostero-SatPublisher/2.3'
    t0, sent = time.perf_counter(), 0

    # 1. Upload new assets
    for n in new:
        asset_bytes = (out / 'assets' / n).read_bytes()
        r = s.put(f'{base}/admin/sat/asset/{n}', data=asset_bytes,
                  headers={'Content-Type': 'image/png'}, timeout=30, allow_redirects=False)
        if r.status_code != 200:
            sys.exit(f'asset upload failed HTTP {r.status_code} for {n}: {r.text[:200]}')
        fresh[n] = now.isoformat()
        sent += 1

    # 2. Upload manifest (commits active version atomically)
    r = s.put(f'{base}/admin/sat/manifest', data=(out / 'manifest.json').read_bytes(),
              headers={'Content-Type': 'application/json'}, timeout=30, allow_redirects=False)
    if r.status_code == 409 and 'older_manifest_rejected' in r.text:
        print(json.dumps({'status': 'SKIPPED_OLDER_MANIFEST', 'note': 'Remote manifest is newer or identical'}))
        return
    if r.status_code != 200:
        sys.exit(f'manifest upload failed HTTP {r.status_code}: {r.text[:300]}')

    # Update state & quota tracking
    st.mkdir(parents=True, exist_ok=True)
    pub_path.write_text(json.dumps({k: v for k, v in fresh.items()
                                    if datetime.fromisoformat(v) > now - timedelta(hours=TTL_H)}),
                        encoding='utf-8')

    quota['writes'] += sent + 1
    quota['runs'] += 1
    quota['updated_at'] = now.isoformat()
    save_quota(quota_path, quota)

    print(json.dumps({
        'status': 'PUBLISHED_SUCCESS',
        'uploaded_assets': sent,
        'kv_writes': sent + 1,
        'daily_writes_used_total': quota['writes'],
        'seconds': round(time.perf_counter() - t0, 2),
        **report
    }))


if __name__ == '__main__':
    main()
