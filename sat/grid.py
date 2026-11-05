"""Geometry shared by ERR and ABI 2 km full-disk products (identical fixed grid).

Verified 2026-11-04: ERR G19 Latitude/Longitude arrays equal the GOES-R ABI fixed grid
(lon_0 = -75.0, sweep x, h = 35786023 m) to < 1e-5 deg. The display grid is a regular
0.01 deg lat/lon raster filled by nearest native pixel (no interpolation of values).
"""
from __future__ import annotations

import numpy as np
from pyproj import CRS, Transformer

H = 35786023.0
X0, Y0, STEP = -0.151844, 0.151844, 0.000056
NFULL = 5424
# Native window covering Tumbes, Piura and Lambayeque with margin.
R0, R1, C0, C1 = 2877, 3129, 2347, 2531
NR, NC = R1 - R0, C1 - C0
# Display raster (cell centres): lon [-81.60,-78.30), lat (-3.00,-7.60]
LON_W, LAT_N, DSTEP, DNX, DNY = -81.60, -3.00, 0.01, 330, 460
BBOX = {'west': LON_W, 'east': round(LON_W + DNX * DSTEP, 4), 'north': LAT_N, 'south': round(LAT_N - DNY * DSTEP, 4)}

_CRS = CRS.from_proj4(f'+proj=geos +h={H} +lon_0=-75.0 +a=6378137 +b=6356752.31414 +sweep=x +units=m +no_defs')
_FWD = Transformer.from_crs('EPSG:4326', _CRS, always_xy=True)
_INV = Transformer.from_crs(_CRS, 'EPSG:4326', always_xy=True)


def native_latlon():
    """Lat/lon of every native pixel centre in the regional window (NR x NC)."""
    cols = np.arange(C0, C1)
    rows = np.arange(R0, R1)
    xx, yy = np.meshgrid((X0 + cols * STEP) * H, (Y0 - rows * STEP) * H)
    lon, lat = _INV.transform(xx, yy)
    return np.asarray(lat), np.asarray(lon)


def latlon_to_native(lat, lon):
    """Fractional-free nearest native (row, col) in window coords; -1 when outside."""
    x, y = _FWD.transform(np.asarray(lon, float), np.asarray(lat, float))
    col = np.rint((np.asarray(x) / H - X0) / STEP).astype(int) - C0
    row = np.rint((Y0 - np.asarray(y) / H) / STEP).astype(int) - R0
    bad = (col < 0) | (col >= NC) | (row < 0) | (row >= NR) | ~np.isfinite(x)
    return np.where(bad, -1, row), np.where(bad, -1, col)


def display_index():
    """Index arrays (DNY x DNX) mapping each display cell to a native pixel."""
    lons = LON_W + (np.arange(DNX) + 0.5) * DSTEP
    lats = LAT_N - (np.arange(DNY) + 0.5) * DSTEP
    lo, la = np.meshgrid(lons, lats)
    return latlon_to_native(la, lo)


def pixel_area_km2(lat, lon):
    """Approximate native pixel footprint from neighbour spacing (km^2)."""
    dy = np.abs(np.gradient(lat, axis=0)) * 110.6
    dx = np.abs(np.gradient(lon, axis=1)) * 111.32 * np.cos(np.radians(lat))
    return dx * dy
