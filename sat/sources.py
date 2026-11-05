"""Readers for NOAA ERR (G19), GOES-19 ABI L2 and GLM L2 — regional window only."""
from __future__ import annotations

import io
import re
import time
from datetime import datetime, timedelta, timezone

import h5py
import numpy as np

from . import grid
from .rangeio import METER, RangeFile, list_keys, session, bucket_url

ERR_BUCKET, ERR_PREFIX = 'noaa-enterprise-rainrate-pds', 'G19/ABI-L2-FLDK-RainfallRate'
GOES_BUCKET = 'noaa-goes19'
UTC = timezone.utc


def _attr(v):
    v = np.asarray(v)
    if v.size == 1:
        x = v.reshape(-1)[0]
        if isinstance(x, (bytes, np.bytes_)):
            return bytes(x).decode('utf-8', 'replace')
        return x.item() if isinstance(x, np.generic) else x
    return v.tolist()


def parse_stamp(token: str) -> datetime:
    """'s202611041440214' (ERR, YYYYMMDD) or 's20263081440214' (GOES, YYYYJJJ)."""
    t = token[1:]
    if len(t) == 15:  # YYYYMMDDhhmmss + tenth
        base = datetime.strptime(t[:14], '%Y%m%d%H%M%S')
    elif len(t) == 14:  # YYYYJJJhhmmss + tenth
        base = datetime.strptime(t[:13], '%Y%j%H%M%S')
    else:
        raise ValueError(token)
    return (base + timedelta(seconds=int(t[-1]) / 10)).replace(tzinfo=UTC)


def _stamps(key):
    name = key.rsplit('/', 1)[-1]
    s, e, c = (re.search(rf'_({ch}\d{{14,15}})', name).group(1) for ch in 'sec')
    return parse_stamp(s), parse_stamp(e), parse_stamp(c)


def slot_of(start: datetime) -> datetime:
    """Nominal 10-min full-disk slot (scan starts ~20 s after the slot)."""
    return start.replace(minute=start.minute - start.minute % 10, second=0, microsecond=0)


def iso(d: datetime | None):
    return None if d is None else d.astimezone(UTC).strftime('%Y-%m-%dT%H:%M:%SZ')


def _hours(end: datetime, hours: int):
    end = end.replace(minute=0, second=0, microsecond=0)
    return [end - timedelta(hours=h) for h in range(hours, -1, -1)]


# ----------------------------------------------------------------------------- ERR
def err_catalog(now: datetime, hours: int):
    items = []
    for h in _hours(now, hours):
        for k in list_keys(ERR_BUCKET, f'{ERR_PREFIX}/{h:%Y/%m/%d/%H}/', product='listing'):
            if not k['key'].endswith('.nc') or '_g19_' not in k['key']:
                continue
            s, e, c = _stamps(k['key'])
            items.append({**k, 'start': s, 'end': e, 'created': c, 'slot': slot_of(s)})
    # one file per slot: keep the latest created (reprocessing)
    best = {}
    for it in items:
        if it['slot'] not in best or it['created'] > best[it['slot']]['created']:
            best[it['slot']] = it
    return [best[s] for s in sorted(best)]


def err_read(item):
    t = time.perf_counter()
    f = RangeFile(ERR_BUCKET, item['key'], size=item['size'], etag=item['etag'], product='ERR', block_size=16384,
                  max_bytes=3_000_000)
    with h5py.File(f, 'r') as h:
        v = h['RRQPE']
        if v.shape != (grid.NFULL, grid.NFULL):
            raise ValueError('unexpected ERR shape')
        units = _attr(v.attrs['units'])
        scale = float(_attr(v.attrs['scale_factor']))
        fill = int(_attr(v.attrs['_FillValue']))
        if units not in ('mm h-1', 'mm/h') or abs(scale - 0.1) > 1e-6:
            raise ValueError(f'unexpected ERR units/scale {units} {scale}')
        if _attr(h.attrs['platform']) != 'GOES 19':
            raise ValueError('not GOES 19')
        raw = v[grid.R0:grid.R1, grid.C0:grid.C1].astype(np.int16)
        dqf = h['DQF'][grid.R0:grid.R1, grid.C0:grid.C1].astype(np.int16)
        meta = {'time_coverage_start': _attr(h.attrs['time_coverage_start']),
                'time_coverage_end': _attr(h.attrs['time_coverage_end']),
                'date_created': _attr(h.attrs['date_created']),
                'resolution': _attr(h.attrs['resolution'])}
    valid = (raw != fill) & (raw >= 0) & (raw <= 1000) & (dqf == 0)
    return {'raw': raw, 'valid': valid, 'dqf': dqf, 'meta': meta, 'bytes': f.received,
            'seconds': round(time.perf_counter() - t, 3)}


# ----------------------------------------------------------------------------- ABI
ABI = {
    # product: (variable, quality rule)
    'ABI-L2-ACMF': ('ACM', 'acm'),
    'ABI-L2-ACHA2KMF': ('HT', 'cloud_top'),
    'ABI-L2-ACHTF': ('TEMP', 'cloud_top'),
    'ABI-L2-ACTPF': ('Phase', 'phase'),
    'ABI-L2-COD2KMF': ('COD', 'cod'),
    'ABI-L2-CMIPF': ('CMI', 'cmi'),
}


def abi_catalog(product: str, now: datetime, hours: int, band: str | None = None):
    out = {}
    for h in _hours(now, hours):
        for k in list_keys(GOES_BUCKET, f'{product}/{h:%Y/%j/%H}/', product='listing'):
            if '-M6' not in k['key'] or (band and band not in k['key']):
                continue
            s, e, c = _stamps(k['key'])
            slot = slot_of(s)
            if slot not in out or c > out[slot]['created']:
                out[slot] = {**k, 'start': s, 'end': e, 'created': c, 'slot': slot, 'product': product}
    return [out[s] for s in sorted(out)]


def _usable(rule, dqf):
    if rule == 'acm':
        return dqf == 0
    if rule == 'cloud_top':
        return np.isin(dqf, (0, 1, 4))  # good, marginal, opaque retrieval
    if rule == 'phase':
        return (dqf & 1) == 0
    if rule == 'cod':
        return ((dqf & 1) == 0) & ((dqf & 4) == 0)  # day algorithm and good quality
    if rule == 'cmi':
        return dqf == 0
    raise ValueError(rule)


def abi_read(item):
    product = item['product']
    var, rule = ABI[product]
    t = time.perf_counter()
    f = RangeFile(GOES_BUCKET, item['key'], size=item['size'], etag=item['etag'], product=product,
                  block_size=65536, max_bytes=5_000_000)
    with h5py.File(f, 'r') as h:
        if _attr(h.attrs['platform_ID']) != 'G19':
            raise ValueError('not G19')
        p = h['goes_imager_projection'].attrs
        if abs(float(_attr(p['longitude_of_projection_origin'])) + 75.0) > 1e-6:
            raise ValueError('unexpected projection origin')
        v = h[var]
        if v.shape != (grid.NFULL, grid.NFULL):
            raise ValueError(f'unexpected shape {v.shape} for {product}')
        x0 = float(np.asarray(h['x'][grid.C0]) * _attr(h['x'].attrs['scale_factor']) + _attr(h['x'].attrs['add_offset']))
        if abs(x0 - (grid.X0 + grid.C0 * grid.STEP)) > 1e-7:
            raise ValueError('fixed grid mismatch')
        raw = np.asarray(v[grid.R0:grid.R1, grid.C0:grid.C1])
        if str(_attr(v.attrs.get('_Unsigned', 'false'))).lower() == 'true' and raw.dtype.kind == 'i':
            raw = raw.view(np.dtype('u' + str(raw.dtype.itemsize)))
        fill = v.attrs.get('_FillValue')
        missing = np.zeros(raw.shape, bool)
        if fill is not None:
            missing |= raw == np.asarray(fill).astype(raw.dtype, casting='unsafe').reshape(-1)[0]
        val = raw.astype(np.float32)
        if 'scale_factor' in v.attrs:
            val = val * float(_attr(v.attrs['scale_factor'])) + float(_attr(v.attrs.get('add_offset', 0.0)))
        dqf_raw = np.asarray(h['DQF'][grid.R0:grid.R1, grid.C0:grid.C1]).astype(np.int32)
        dfill = int(_attr(h['DQF'].attrs.get('_FillValue', 255)))
        usable = (~missing) & (dqf_raw != dfill) & _usable(rule, dqf_raw)
        meta = {'product': product, 'variable': var, 'units': _attr(v.attrs.get('units', '1')),
                'time_coverage_start': _attr(h.attrs['time_coverage_start']),
                'time_coverage_end': _attr(h.attrs['time_coverage_end']),
                'resolution': _attr(h.attrs.get('spatial_resolution', '2km at nadir')),
                'file': item['key'].rsplit('/', 1)[-1]}
    val[~usable] = np.nan
    return {'value': val, 'usable': usable, 'meta': meta, 'bytes': f.received,
            'seconds': round(time.perf_counter() - t, 3)}


# ----------------------------------------------------------------------------- GLM
def _decode(v, scaled=True):
    raw = np.asarray(v[:])
    if str(_attr(v.attrs.get('_Unsigned', 'false'))).lower() == 'true' and raw.dtype.kind == 'i':
        raw = raw.view(np.dtype('u' + str(raw.dtype.itemsize)))
    out = raw.astype(float)
    if scaled:
        out = out * float(_attr(v.attrs.get('scale_factor', 1.0))) + float(_attr(v.attrs.get('add_offset', 0.0)))
    return out


def glm_catalog(now: datetime, hours: int):
    out = {}
    for h in _hours(now, hours):
        for k in list_keys(GOES_BUCKET, f'GLM-L2-LCFA/{h:%Y/%j/%H}/', product='listing'):
            if k['key'].endswith('.nc'):
                s, e, c = _stamps(k['key'])
                if (s,e) not in out or c > out[s,e]['created']:
                    out[s,e] = {**k, 'start': s, 'end': e, 'created': c}
    return sorted(out.values(), key=lambda i: i['start'])


def glm_read(item, bbox):
    """Full GET (files are ~220 kB) and keep good-quality flashes inside bbox."""
    if item['size'] > 3_000_000:
        raise ValueError('GLM object unexpectedly large')
    t = time.perf_counter()
    r = session().get(bucket_url(GOES_BUCKET) + '/' + item['key'], timeout=(10, 45))
    METER.add('GLM-L2-LCFA', len(r.content), time.perf_counter() - t)
    r.raise_for_status()
    if r.content[:8] != b'\x89HDF\r\n\x1a\n':
        raise ValueError('GLM not HDF5')
    flashes = []
    with h5py.File(io.BytesIO(r.content), 'r') as h:
        if _attr(h.attrs['platform_ID']) != 'G19':
            raise ValueError('not G19')
        n = h['flash_lat'].shape[0]
        if n:
            lat = _decode(h['flash_lat'])
            lon = _decode(h['flash_lon'])
            q = _decode(h['flash_quality_flag'], scaled=False)
            off_v = h['flash_time_offset_of_first_event']
            off = _decode(off_v)
            en = _decode(h['flash_energy'])
            m = re.match(r'seconds since (\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}(?:\.\d+)?)', _attr(off_v.attrs['units']))
            if not m:
                raise ValueError('unexpected GLM time units')
            base = datetime.fromisoformat(m.group(1)).replace(tzinfo=UTC)
            sel = (q == 0) & np.isfinite(off) & np.isfinite(en) & (lat >= bbox['south'] - 0.5) & (lat <= bbox['north'] + 0.5) & \
                  (lon >= bbox['west'] - 0.5) & (lon <= bbox['east'] + 0.5)
            for i in np.nonzero(sel)[0]:
                event_time = base + timedelta(seconds=float(off[i]))
                if not item['start'] - timedelta(seconds=1) <= event_time <= item['end'] + timedelta(seconds=1):
                    continue
                flashes.append({'t': iso(event_time),
                                'lat': round(float(lat[i]), 4), 'lon': round(float(lon[i]), 4),
                                'energy_fJ': round(float(en[i]) * 1e15, 1)})
    return {'file': item['key'].rsplit('/', 1)[-1], 'start': iso(item['start']), 'end': iso(item['end']),
            'flashes': flashes, 'bytes': len(r.content), 'seconds': round(time.perf_counter() - t, 3)}
