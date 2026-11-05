"""Scientific processing: ERR time integration, cloud classification, spatial statistics.

Rules (documented in the manifest):
* ERR gives instantaneous rates (mm/h) once per 10-min full-disk slot. An accumulation
  for H hours is ESTIMATED as sum(rate_i * 1/6 h) over the 6H consecutive slots ending at
  the latest slot. It is reported only when every slot is present (strict completeness)
  and only for pixels valid (DQF == 0, no fill) in every slot. It is NOT a gauge total.
* Cloud classes are conservative multi-evidence categories, never a confirmed species.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np

SLOT_H = 10 / 60
WINDOWS_H = (1, 3, 6, 12, 24)

# ------------------------------------------------------------------ rain index (mirror of web)
RAIN_LEVELS = [
    (0, 'Sin lluvia'), (1, 'Llovizna'), (2, 'Lluvia débil'), (3, 'Lluvia'), (4, 'Lluvia moderada'),
    (5, 'Lluvia fuerte'), (6, 'Lluvia severa'), (7, 'LLUVIA LEGENDARIA'), (8, 'LLUVIA EXISTENCIAL'),
    (9, 'LLUVIA DE RENACIMIENTO'), (10, 'LLUVIA FINAL')]
_RAIN_EDGES = [1.0, 3.0, 5.0, 8.0, 20.0, 25.0, 50.0, 70.0, 90.0]


def round1(v: float) -> float:
    """Half-up rounding to 0.1 mm, matching Math.round(v*10)/10 in the web client."""
    return float(np.floor(v * 10 + 0.5) / 10)


def rain_level(mm: float | None):
    if mm is None or not np.isfinite(mm) or mm < 0:
        return None
    v = round1(mm)
    if v <= 0.0:
        return 0
    if v < 1.0:
        return 1
    for i, edge in enumerate(_RAIN_EDGES[1:], start=2):
        if v < edge:
            return i
    return 10


def storm_level(mm24: float | None, flashes: int | None, glm_complete: bool):
    """Original 0-8 storm scale. Requires authentic GLM evidence; no flashes => no storm."""
    if flashes is None or flashes < 0:
        return None
    if flashes == 0:
        return 0 if glm_complete else None
    if mm24 is None or not np.isfinite(mm24) or mm24 < 0:
        return None
    v = round1(mm24)
    for lvl, edge in enumerate([5, 10, 15, 25, 50, 70, 90], start=1):
        if v < edge:
            return lvl
    return 8


# ------------------------------------------------------------------ accumulation
def accumulate(frames: dict, latest_slot, hours: int):
    """frames: slot(datetime) -> {'raw': int16 (0.1 mm/h), 'valid': bool}."""
    slots = [latest_slot - timedelta(minutes=10 * i) for i in range(hours * 6)]
    missing = [s for s in slots if s not in frames]
    # Nominal 10-min integration requires a matching full-disk scan in each slot.
    # Small scan scheduling gaps are inherent; this is a rate-sampling estimate.
    for s in slots:
        if s in frames and 'meta' in frames[s]:
            meta = frames[s]['meta']
            try:
                start = datetime.fromisoformat(meta['time_coverage_start'].replace('Z', '+00:00'))
                end = datetime.fromisoformat(meta['time_coverage_end'].replace('Z', '+00:00'))
                valid_time = 0 <= (start-s).total_seconds() <= 120 and 480 <= (end-s).total_seconds() <= 660
            except (KeyError, ValueError, TypeError):
                valid_time = False
            if not valid_time:
                missing.append(s)
    info = {'hours': hours, 'expected_slots': len(slots), 'received_slots': len(slots) - len(missing),
            'missing_slots': [s.strftime('%Y-%m-%dT%H:%MZ') for s in missing[:12]],
            'missing_count': len(missing),
            'valid_from': (slots[-1]).strftime('%Y-%m-%dT%H:%M:%SZ'),
            'valid_to': (latest_slot + timedelta(minutes=10)).strftime('%Y-%m-%dT%H:%M:%SZ'),
            'complete': not missing}
    if missing:
        info['reason'] = 'serie incompleta de escaneos ERR; no se estima acumulado'
        return None, None, info
    acc = np.zeros(frames[slots[0]]['raw'].shape, np.float64)
    ok = np.ones(acc.shape, bool)
    for s in slots:
        f = frames[s]
        ok &= f['valid']
        acc += np.where(f['valid'], f['raw'].astype(np.float64) * 0.1, 0.0) * SLOT_H
    acc[~ok] = np.nan
    return acc, ok, info


def coverage_fraction(covered, t0, t1):
    """Union of covered intervals, so duplicate/overlapping files cannot hide gaps."""
    intervals = sorted((max(s, t0), min(e, t1)) for s, e in covered if e > t0 and s < t1)
    total, end = 0.0, t0
    for s, e in intervals:
        total += max(0.0, (e - max(end, s)).total_seconds())
        end = max(end, e)
    return min(1.0, total / max(1, (t1-t0).total_seconds()))


# ------------------------------------------------------------------ cloud classification
CLOUD_CLASSES = {
    0: ('sin_dato', 'Sin dato de calidad suficiente'),
    1: ('despejado', 'Despejado o probablemente despejado'),
    2: ('nubosidad_baja', 'Nubosidad baja'),
    3: ('estratos', 'Estratos / estratocúmulos probables (capa baja homogénea)'),
    4: ('cumulos', 'Cúmulos probables (textura cumuliforme baja-media)'),
    5: ('nubosidad_media', 'Nubosidad media'),
    6: ('nube_alta', 'Nube alta (cirros / yunque no convectivo)'),
    7: ('conveccion_profunda_probable', 'Convección profunda probable'),
    8: ('cumulonimbus_probable', 'Cumulonimbus probable (no confirmado)'),
    9: ('no_determinada', 'Nube detectada, clasificación no determinada'),
}


def _local_std(a, k=2):
    """Std of finite values in a (2k+1)^2 window (NaN-aware)."""
    pad = np.pad(a, k, mode='edge')
    stack = [pad[i:i + a.shape[0], j:j + a.shape[1]] for i in range(2 * k + 1) for j in range(2 * k + 1)]
    return np.nanstd(np.stack(stack), axis=0)


def _local_min(a, k=2):
    pad = np.pad(a, k, mode='edge')
    stack = [pad[i:i + a.shape[0], j:j + a.shape[1]] for i in range(2 * k + 1) for j in range(2 * k + 1)]
    with np.errstate(all='ignore'):
        return np.nanmin(np.stack(stack), axis=0)


def _smooth(a, k=1):
    pad = np.pad(a, k, mode='edge')
    stack = [pad[i:i + a.shape[0], j:j + a.shape[1]] for i in range(2 * k + 1) for j in range(2 * k + 1)]
    with np.errstate(all='ignore'):
        return np.nanmean(np.stack(stack), axis=0)


def classify(acm, cth, ctt, phase, cod, bt, bt_prev, flash_near):
    """All inputs NR x NC float arrays with NaN where unusable (acm/phase as codes).

    Returns (class uint8, evidence dict of counts).
    """
    shape = acm.shape
    cls = np.zeros(shape, np.uint8)
    have_mask = np.isfinite(acm)
    clear = have_mask & (acm <= 1)
    cloudy = have_mask & (acm >= 2)
    cls[clear] = 1
    cls[cloudy] = 9
    with np.errstate(invalid='ignore'):
        cooling = (_smooth(bt) - _smooth(bt_prev)) if bt_prev is not None else np.full(shape, np.nan)
        texture = _local_std(bt)
        cold_core = _local_min(bt)
        h = np.where(cloudy, cth, np.nan)
        ice = phase == 4
        liquid = (phase == 1) | (phase == 2)
        thick = np.where(np.isfinite(cod), cod >= 20, cold_core <= 225)  # night: cold core proxy
        deep = cloudy & (h >= 8000) & (ctt <= 235) & ice & (bt <= 235) & thick
        strong_growth = cooling <= -4.0
        cb = deep & (ctt <= 225) & (flash_near | strong_growth)
        high = cloudy & (h >= 6000) & ~deep
        mid = cloudy & (h >= 2500) & (h < 6000)
        low = cloudy & (h < 2500)
        stratus = low & liquid & (texture < 0.8)
        cumulus = cloudy & (h < 4000) & ~ice & (texture >= 1.5)
    cls[low] = 2
    cls[mid] = 5
    cls[stratus] = 3
    cls[cumulus] = 4
    cls[high] = 6
    cls[deep] = 7
    cls[cb] = 8
    evidence = {
        'cb_with_lightning_px': int((cb & flash_near).sum()),
        'cb_with_growth_only_px': int((cb & ~flash_near & strong_growth).sum()),
        'deep_without_second_evidence_px': int((deep & ~cb).sum()),
        'cooling_available': bool(bt_prev is not None),
        'cod_daytime_px': int(np.isfinite(cod).sum()),
    }
    return cls, evidence


def flash_proximity(lat, lon, flashes, radius_km=10.0):
    near = np.zeros(lat.shape, bool)
    for f in flashes:
        d2 = ((lat - f['lat']) * 110.6) ** 2 + ((lon - f['lon']) * 111.32 * np.cos(np.radians(f['lat']))) ** 2
        near |= d2 <= radius_km ** 2
    return near


def motion(bt_prev, bt, max_shift=6):
    """Mean cloud-field displacement (pixels) by exhaustive block matching on IR BT."""
    if bt_prev is None:
        return None
    a = np.where(np.isfinite(bt_prev), bt_prev, np.nan)
    b = np.where(np.isfinite(bt), bt, np.nan)
    if np.nanstd(b) < 2.0:
        return {'status': 'campo_uniforme', 'reason': 'contraste IR insuficiente para rastrear movimiento'}
    best = (None, -2)
    m = max_shift
    core_b = b[m:-m, m:-m]
    for dy in range(-m, m + 1):
        for dx in range(-m, m + 1):
            core_a = a[m - dy:a.shape[0] - m - dy, m - dx:a.shape[1] - m - dx]
            ok = np.isfinite(core_a) & np.isfinite(core_b)
            if ok.sum() < 500:
                continue
            x, y = core_a[ok], core_b[ok]
            c = np.corrcoef(x, y)[0, 1]
            if c > best[1]:
                best = ((dy, dx), c)
    if best[0] is None:
        return None
    return {'status': 'estimado' if best[1] >= 0.6 else 'baja_confianza', 'shift_px': best[0],
            'correlation': round(float(best[1]), 3)}


# ------------------------------------------------------------------ spatial statistics
def area_stats(values, valid, mask, thresholds):
    sel = mask
    n_total = int(sel.sum())
    v = values[sel & valid]
    v = v[np.isfinite(v)]
    out = {'pixels_total': n_total, 'pixels_valid': int(v.size),
           'valid_fraction': round(v.size / n_total, 4) if n_total else 0.0}
    if v.size == 0:
        out.update({'max': None, 'mean': None, 'p90': None, 'area_fraction_ge': {str(t): None for t in thresholds}})
        return out
    out.update({'max': round(float(v.max()), 2), 'mean': round(float(v.mean()), 3),
                'p90': round(float(np.percentile(v, 90)), 2),
                'area_fraction_ge': {str(t): round(float((v >= t).sum() / v.size), 4) for t in thresholds}})
    return out


def level_distribution(acc, valid, mask):
    v = acc[mask & valid]
    v = v[np.isfinite(v)]
    if v.size == 0:
        return None
    lv = np.array([rain_level(x) for x in v])
    return {str(k): round(float((lv == k).sum() / v.size), 4) for k in range(11) if (lv == k).any()}
