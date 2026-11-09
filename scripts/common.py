"""Small, auditable HTTPS and evidence helpers; never log credentials."""
from __future__ import annotations
import hashlib
import json
import threading
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit, parse_qsl, urlencode
import requests

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / 'results'
for folder in ('raw', 'processed', 'logs'):
    (RESULTS / folder).mkdir(parents=True, exist_ok=True)
CITIES = [
    {'id': 'tumbes', 'name': 'Tumbes', 'latitude': -3.5669, 'longitude': -80.4515},
    {'id': 'piura', 'name': 'Piura', 'latitude': -5.1945, 'longitude': -80.6328},
    {'id': 'chiclayo', 'name': 'Chiclayo', 'latitude': -6.7714, 'longitude': -79.8409},
]
PERU = timezone(timedelta(hours=-5))
LOCK = threading.Lock()

def now():
    return datetime.now(timezone.utc)

def iso(value=None):
    return (value or now()).astimezone(timezone.utc).isoformat().replace('+00:00', 'Z')

def clean_url(url):
    parts = urlsplit(url)
    query = [(k, '[REDACTED]' if any(s in k.lower() for s in ('token', 'secret', 'password', 'code')) else v)
             for k, v in parse_qsl(parts.query, keep_blank_values=True)]
    return urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(query), ''))

def write_json(path, value, compact=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False,
                              indent=None if compact else 2,
                              separators=(',', ':') if compact else None), encoding='utf-8')

def log(event, filename='http.jsonl'):
    with LOCK:
        with (RESULTS / 'logs' / filename).open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(event, ensure_ascii=False, allow_nan=False) + '\n')

def get(url, *, params=None, headers=None, dest=None, max_bytes=80_000_000, timeout=45, auth=None):
    """GET only. Count entity bytes, retain HTTP evidence, and enforce transfer cap."""
    start = now()
    timer = time.perf_counter()
    event = {'started_at': iso(start), 'url': clean_url(url), 'method': 'GET'}
    response = None
    chunks = []
    size = 0
    try:
        response = requests.get(url, params=params, headers={
            'User-Agent': 'NinoCostero-DataValidation/0.1 (research prototype; no cloud deployment)',
            'Accept-Encoding': 'identity', **(headers or {})}, timeout=(15, timeout), stream=True, auth=auth)
        event.update({'url': clean_url(response.url), 'status': response.status_code,
                      'server_date': response.headers.get('Date'),
                      'last_modified': response.headers.get('Last-Modified'),
                      'etag': response.headers.get('ETag'),
                      'content_type': response.headers.get('Content-Type'),
                      'content_range': response.headers.get('Content-Range'),
                      'declared_length': response.headers.get('Content-Length')})
        declared = int(response.headers.get('Content-Length', 0))
        if declared > max_bytes:
            raise ValueError(f'Transfer cap: declared {declared} > {max_bytes}')
        for chunk in response.iter_content(65536):
            size += len(chunk)
            if size > max_bytes:
                raise ValueError(f'Transfer cap exceeded: {size}')
            chunks.append(chunk)
        body = b''.join(chunks)
        event.update({'downloaded_at': iso(), 'entity_bytes': size,
                      'sha256': hashlib.sha256(body).hexdigest()})
        if dest is not None:
            dest = Path(dest)
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(body)
            event['local_path'] = str(dest.relative_to(ROOT))
        return response.status_code, body, event
    except Exception as exc:
        event['error'] = f'{type(exc).__name__}: {exc}'
        event['entity_bytes'] = size
        raise
    finally:
        event['finished_at'] = iso()
        event['elapsed_s'] = round(time.perf_counter() - timer, 6)
        log(event)
        if response is not None:
            response.close()

def parse_time(value):
    return datetime.fromisoformat(value.replace('Z', '+00:00'))
