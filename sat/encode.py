"""Lossless data PNGs on the 0.01 deg display grid (alpha always 255 => no premultiply loss).

err  : R,G = rate uint16 in 0.1 mm/h ; B = 1 valid, 2 invalid quality, 0 outside window
acc  : R,G,B low 6 bits = accumulation in 1/60 mm units (22 bits, exact rate sum)
       B high bits = 64 valid, 128 invalid, 0 outside; A always 255
ir   : R = BT - 150 K (1 K steps, 1..255) ; B = 1 valid / 2 invalid / 0 outside
cloud: R = class code, G = cloud-top height /100 m (255 none), B = CTT - 150 K (0 none)
cloud2: R = COD (rounded, 255 none), G = phase (255 none), B = ACM (255 none)
"""
from __future__ import annotations

import io

import numpy as np
from PIL import Image

from . import grid

_IDX = None


def idx():
    global _IDX
    if _IDX is None:
        _IDX = grid.display_index()
    return _IDX


def _to_display(native, fill):
    r, c = idx()
    inside = r >= 0
    out = np.full(r.shape, fill, dtype=native.dtype)
    out[inside] = native[r[inside], c[inside]]
    return out, inside


def _png(r, g, b):
    a = np.full(r.shape, 255, np.uint8)
    img = Image.fromarray(np.dstack([r, g, b, a]).astype(np.uint8), 'RGBA')
    buf = io.BytesIO()
    img.save(buf, format='PNG', optimize=True)
    return buf.getvalue()


def _flag(valid_native):
    v, inside = _to_display(valid_native.astype(np.uint8), 0)
    return np.where(inside, np.where(v == 1, 1, 2), 0).astype(np.uint8)


def uint16_png(values_native, valid_native, scale):
    q = np.where(valid_native & np.isfinite(values_native), np.clip(np.rint(values_native * scale), 0, 65535), 0)
    q = q.astype(np.uint16)
    d, _ = _to_display(q, 0)
    return _png((d >> 8).astype(np.uint8), (d & 255).astype(np.uint8), _flag(valid_native & np.isfinite(values_native)))


def err_png(raw, valid):
    return uint16_png(raw.astype(np.float64), valid, 1.0)  # raw already in 0.1 mm/h


def acc_png(acc, valid):
    # RRQPE is quantized to 0.1 mm/h; a ten-minute integral is exactly 1/60 mm.
    # 22 bits preserve totals well above 100 mm without clipping at 655.35 mm.
    q = np.where(valid & np.isfinite(acc), np.rint(acc * 60), 0).astype(np.uint32)
    if np.any(q > 0x3fffff):
        raise ValueError('accumulation exceeds lossless encoding range')
    d, _ = _to_display(q, 0)
    flag = _flag(valid & np.isfinite(acc))
    return _png((d >> 8) & 255, d & 255, ((d >> 16) & 63) | np.where(flag == 1, 64, np.where(flag == 2, 128, 0)))


def ir_png(bt):
    ok = np.isfinite(bt)
    q = np.where(ok, np.clip(np.rint(bt - 150), 1, 255), 0).astype(np.uint8)
    d, _ = _to_display(q, 0)
    z = np.zeros_like(d)
    return _png(d, z, _flag(ok))


def cloud_png(cls, cth, ctt):
    c, inside = _to_display(cls.astype(np.uint8), 255)
    h = np.where(np.isfinite(cth), np.clip(np.rint(cth / 100), 0, 254), 255).astype(np.uint8)
    t = np.where(np.isfinite(ctt), np.clip(np.rint(ctt - 150), 1, 255), 0).astype(np.uint8)
    h, _ = _to_display(h, 255)
    t, _ = _to_display(t, 0)
    return _png(c, h, t)


def cloud2_png(cod, phase, acm):
    f = lambda a: np.where(np.isfinite(a), np.clip(np.rint(a), 0, 254), 255).astype(np.uint8)
    return _png(_to_display(f(cod), 255)[0], _to_display(f(phase), 255)[0], _to_display(f(acm), 255)[0])
