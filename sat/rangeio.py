"""HTTP Range access to public NOAA NODD buckets (anonymous HTTPS, no AWS account).

Only byte ranges of HDF5/NetCDF4 objects are fetched, so a 68 MB full-disk file costs
a few hundred KB when only the Peru chunks are decoded. Every request is metered.
The object ETag is pinned with If-Match so a file replaced mid-read is rejected.
"""
from __future__ import annotations

import io
import re
import threading
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from urllib.parse import quote

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

ALLOWED_BUCKETS = {
    'noaa-enterprise-rainrate-pds',
    'noaa-goes19',
}
UA = 'NinoCostero-Pipeline/2.0 (+https://nino-costero.pages.dev; public-data research, S/0)'
_S3NS = '{http://s3.amazonaws.com/doc/2006-03-01/}'


@dataclass
class Meter:
    """Thread-safe transfer accounting shared by a whole pipeline run."""
    requests: int = 0
    entity_bytes: int = 0
    seconds: float = 0.0
    by_product: dict = field(default_factory=dict)
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def add(self, product: str, nbytes: int, elapsed: float):
        with self.lock:
            self.requests += 1
            self.entity_bytes += nbytes
            self.seconds += elapsed
            slot = self.by_product.setdefault(product, {'requests': 0, 'entity_bytes': 0})
            slot['requests'] += 1
            slot['entity_bytes'] += nbytes

    def summary(self):
        return {'requests': self.requests, 'entity_bytes': self.entity_bytes,
                'http_seconds_sum': round(self.seconds, 3), 'by_product': self.by_product}


METER = Meter()
_session_local = threading.local()


def session() -> requests.Session:
    s = getattr(_session_local, 's', None)
    if s is None:
        s = requests.Session()
        # Bounded retries for transient transport errors only (connection reset, 5xx).
        retry = Retry(total=3, connect=3, read=3, status=3, backoff_factor=0.5,
                      status_forcelist=(500, 502, 503, 504), allowed_methods=('GET', 'HEAD'))
        s.mount('https://', HTTPAdapter(max_retries=retry, pool_maxsize=16))
        s.headers.update({'User-Agent': UA, 'Accept-Encoding': 'identity'})
        _session_local.s = s
    return s


def bucket_url(bucket: str) -> str:
    if bucket not in ALLOWED_BUCKETS:
        raise ValueError(f'bucket not allowed: {bucket}')
    return f'https://{bucket}.s3.amazonaws.com'


def list_keys(bucket: str, prefix: str, product: str = 'listing') -> list[dict]:
    """ListObjectsV2 (anonymous). Returns [{'key','last_modified','size','etag'}]."""
    out, token = [], None
    while True:
        params = {'list-type': '2', 'prefix': prefix, 'max-keys': '1000'}
        if token:
            params['continuation-token'] = token
        t = time.perf_counter()
        r = session().get(bucket_url(bucket) + '/', params=params, timeout=(10, 30))
        METER.add(product, len(r.content), time.perf_counter() - t)
        r.raise_for_status()
        root = ET.fromstring(r.content)
        for c in root.findall(f'{_S3NS}Contents'):
            out.append({'key': c.findtext(f'{_S3NS}Key'),
                        'last_modified': c.findtext(f'{_S3NS}LastModified'),
                        'size': int(c.findtext(f'{_S3NS}Size')),
                        'etag': (c.findtext(f'{_S3NS}ETag') or '').strip('"')})
        if root.findtext(f'{_S3NS}IsTruncated') == 'true':
            token = root.findtext(f'{_S3NS}NextContinuationToken')
        else:
            return out


class RangeFile(io.RawIOBase):
    """Seekable read-only file object backed by HTTP Range GETs with block cache.

    Refuses: non-206 responses (no silent full download), ETag change, transfer cap.
    """

    def __init__(self, bucket: str, key: str, *, size: int | None = None, etag: str | None = None,
                 product: str = 'object', block_size: int = 65536, max_bytes: int = 6_000_000):
        super().__init__()
        if not re.fullmatch(r'[A-Za-z0-9_./-]+\.nc', key):
            raise ValueError('unexpected object key')
        self.url = bucket_url(bucket) + '/' + quote(key, safe='/')
        self.key, self.product, self.block_size, self.max_bytes = key, product, block_size, max_bytes
        self.pos, self.blocks, self.received = 0, {}, 0
        if size is None or etag is None:
            t = time.perf_counter()
            h = session().head(self.url, timeout=(10, 30))
            METER.add(product, 0, time.perf_counter() - t)
            h.raise_for_status()
            size, etag = int(h.headers['Content-Length']), h.headers['ETag'].strip('"')
        self.size, self.etag = size, etag

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=0):
        self.pos = offset + (self.pos if whence == 1 else self.size if whence == 2 else 0)
        if self.pos < 0:
            raise ValueError('negative seek')
        return self.pos

    def _fetch(self, first: int, last: int):
        """Fetch blocks first..last (inclusive) in a single coalesced request."""
        start = first * self.block_size
        end = min(self.size, (last + 1) * self.block_size) - 1
        n = end - start + 1
        if self.received + n > self.max_bytes:
            raise IOError(f'transfer cap reached for {self.key}')
        t = time.perf_counter()
        r = session().get(self.url, headers={'Range': f'bytes={start}-{end}', 'If-Match': f'"{self.etag}"'},
                          timeout=(10, 45))
        METER.add(self.product, len(r.content), time.perf_counter() - t)
        if r.status_code != 206 or len(r.content) != n:
            raise IOError(f'range read refused HTTP {r.status_code} for {self.key}')
        self.received += n
        data = r.content
        for b in range(first, last + 1):
            off = (b - first) * self.block_size
            self.blocks[b] = data[off:off + self.block_size]

    def read(self, size=-1):
        if size < 0:
            size = self.size - self.pos
        size = min(size, self.size - self.pos)
        if size <= 0:
            return b''
        first, last = self.pos // self.block_size, (self.pos + size - 1) // self.block_size
        missing = [b for b in range(first, last + 1) if b not in self.blocks]
        if missing:
            # coalesce consecutive missing blocks into single requests
            run = [missing[0]]
            for b in missing[1:]:
                if b == run[-1] + 1:
                    run.append(b)
                else:
                    self._fetch(run[0], run[-1])
                    run = [b]
            self._fetch(run[0], run[-1])
        buf = b''.join(self.blocks[b] for b in range(first, last + 1))
        off = self.pos - first * self.block_size
        out = buf[off:off + size]
        self.pos += len(out)
        return out

    def readinto(self, b):
        data = self.read(len(b))
        b[:len(data)] = data
        return len(data)
