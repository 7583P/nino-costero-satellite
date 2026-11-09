"""Publish validated DWD ICON-EPS hourly forecast to Cloudflare Worker KV.
Validates locally; uploads only with --send and authorized HTTPS secret.
No provisioning, billing, retries or redirects. Never print token or URL values.
"""
import argparse
import hashlib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit
import requests
from common import ROOT, RESULTS, iso, write_json

CITIES = ('tumbes', 'piura', 'chiclayo')

def validate_hourly_bundle(hourly_dir: Path) -> tuple[dict, bytes]:
    payload = {}
    for city in CITIES:
        file_path = hourly_dir / f"{city}.json"
        if not file_path.exists():
            raise FileNotFoundError(f"Missing hourly file for city {city}: {file_path}")
        doc = json.loads(file_path.read_text(encoding='utf-8'))
        
        # Schema and contract checks matching worker expectations
        if doc.get('schema_version') != '1.0.0':
            raise ValueError(f"Unsupported schema_version for {city}: {doc.get('schema_version')}")
        if doc.get('city', {}).get('id') != city:
            raise ValueError(f"City mismatch in {city}.json: expected {city}, got {doc.get('city', {}).get('id')}")
        if doc.get('source', {}).get('model') != 'ICON-EPS':
            raise ValueError(f"Model mismatch in {city}.json: expected ICON-EPS")
        if doc.get('source', {}).get('total_ensemble_members') != 40:
            raise ValueError(f"Member count mismatch in {city}.json: expected 40")
        
        intervals = doc.get('intervals', [])
        if not isinstance(intervals, list) or len(intervals) < 1:
            raise ValueError(f"Intervals list empty or missing in {city}.json")
            
        gen_at = doc.get('generated_at')
        if not gen_at:
            raise ValueError(f"Missing generated_at in {city}.json")
        try:
            datetime.fromisoformat(gen_at.replace('Z', '+00:00'))
        except Exception:
            raise ValueError(f"Invalid generated_at timestamp format in {city}.json: {gen_at}")
            
        payload[city] = doc

    body = json.dumps(payload, ensure_ascii=False, separators=(',', ':'), allow_nan=False).encode('utf-8')
    if len(body) > 64000:
        raise ValueError(f"Hourly bundle exceeds the 64000-byte publication cap: {len(body)} bytes")
    return payload, body

def validated_target(value: str) -> str:
    target = urlsplit(value)
    if target.scheme != 'https' or not target.hostname or target.username or target.password or target.query or target.fragment or target.path != '/admin/hourly':
        raise ValueError('Expected HTTPS /admin/hourly without credentials, query or fragment')
    return value

def main():
    p = argparse.ArgumentParser(description="Publish validated DWD ICON-EPS hourly forecast.")
    p.add_argument('--dir', default=None, help="Directory containing the 3 city JSON files (default: public/v2/hourly)")
    p.add_argument('--tag', default='latest', help="Alphanumeric publication tag (default: latest)")
    p.add_argument('--send', action='store_true', help="Explicit upload; default validates only, with zero HTTP requests")
    args = p.parse_args()

    if not re.fullmatch(r'[A-Za-z0-9_-]+', args.tag):
        p.error('Use an alphanumeric tag')

    hourly_dir = Path(args.dir) if args.dir else (ROOT / 'public/v2/hourly')
    payload, body = validate_hourly_bundle(hourly_dir)

    run_id = payload['tumbes'].get('run', {}).get('run_id', 'unknown')
    output = RESULTS / 'processed' / f'publication_hourly_{args.tag}.json'

    record = {
        'checked_at': iso(),
        'run_id': run_id,
        'body_bytes': len(body),
        'sha256': hashlib.sha256(body).hexdigest(),
        'cities': list(CITIES),
        'hours_count': len(payload['tumbes']['intervals']),
        'schema_valid': True,
        'scientific_timestamps_unchanged': True,
        'upload_requested': args.send,
        'upload_performed': False,
        'billable_resources_enabled': False
    }

    if args.send:
        raw_url = os.getenv('WORKER_URL') or os.getenv('NINO_PUBLISH_URL') or ''
        # Allow baseUrl without path or with /admin/hourly
        if raw_url and not raw_url.endswith('/admin/hourly'):
            raw_url = raw_url.rstrip('/') + '/admin/hourly'
        target = validated_target(raw_url)
        token = os.getenv('PUBLISH_TOKEN') or os.getenv('SAT_PUBLISH_TOKEN') or os.getenv('NINO_PUBLISH_TOKEN')
        if not token:
            raise ValueError('Configure the publication token locally; do not paste it into code or chat')

        response = requests.put(
            target,
            data=body,
            headers={'Authorization': f'Bearer {token}', 'Content-Type': 'application/json'},
            timeout=(10, 30),
            allow_redirects=False
        )
        record.update(upload_performed=True, http_status=response.status_code, finished_at=iso())
        write_json(output, record)
        if response.status_code not in (200, 304):
            raise RuntimeError(f"Hourly publication failed; status {response.status_code}; no retry or redirect performed")
    else:
        write_json(output, record)

    print(json.dumps(record, indent=2))

if __name__ == '__main__':
    main()

