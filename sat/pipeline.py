"""Niño Costero satellite pipeline: NOAA ERR + GOES-19 ABI/GLM -> light regional products.

python -I scripts/sat/pipeline.py --state .sat_state --out build/sat [--max-err 160] [--glm-hours 1]

* Stateless w.r.t. the cloud: per-scan regional extracts are kept in --state (restored
  from actions/cache in CI). Missing scans inside the 24 h window are backfilled from S3.
* Writes: <out>/manifest.json and <out>/assets/*.png. Publishing is a separate step.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import sys
import tarfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import psutil

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sat import analysis as A  # noqa: E402
from sat import encode as E  # noqa: E402
from sat import grid as G  # noqa: E402
from sat import sources as S  # noqa: E402
from sat.rangeio import METER  # noqa: E402

UTC = timezone.utc
def _find_geo(filename: str) -> Path:
    candidates = [
        Path(__file__).resolve().parent.parent / 'data/geo' / filename,
        Path(__file__).resolve().parent / 'data/geo' / filename,
        Path(__file__).resolve().parents[2] / 'web/public/data/geo' / filename,
        Path('data/geo') / filename,
    ]
    for p in candidates:
        if p.exists():
            return p
    return candidates[0]

ADMIN = _find_geo('admin_costa_norte.json')
LOCALITIES = _find_geo('localities.json')
TIMELINE_H = 3
SCHEMA = '2.0.0'


def stamp(d):
    return d.strftime('%Y%m%dT%H%MZ')


def from_stamp(s):
    return datetime.strptime(s, '%Y%m%dT%H%MZ').replace(tzinfo=UTC)


def accumulation_anchor(latest_slot):
    """Last fully completed UTC hour, represented by its final HH:50 scan."""
    return (latest_slot + timedelta(minutes=10)).replace(minute=0, second=0, microsecond=0) - timedelta(minutes=10)


def make_tar_bundle(files: dict[str, bytes], mtime: int = 1700000000) -> tuple[bytes, str]:
    """Packs multiple raster bytes into a deterministic standard ustar archive."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode='w', format=tarfile.USTAR_FORMAT) as tar:
        for fname in sorted(files.keys()):
            data = files[fname]
            ti = tarfile.TarInfo(name=fname)
            ti.size = len(data)
            ti.mtime = mtime
            ti.mode = 0o644
            tar.addfile(ti, io.BytesIO(data))
    tar_bytes = buf.getvalue()
    tar_hash = hashlib.sha256(tar_bytes).hexdigest()
    return tar_bytes, tar_hash


def partition_tar_assets(files: dict[str, bytes], max_bytes=4 * 1024 * 1024):
    """Deterministic bounded bundles; never raise the authenticated Worker limit.

    A rare spatially complex rain scene may compress poorly. Splitting preserves
    the exact raster values and the publisher's server-authoritative KV cap.
    """
    batches, batch, padded_bytes = [], {}, 0
    for name in sorted(files):
        data = files[name]
        if len(data) > 700000:
            raise ValueError(f'asset exceeds authenticated PNG limit: {name} ({len(data)} bytes)')
        entry_size = 512 + ((len(data) + 511) // 512) * 512
        # Python tarfile emits complete 10240-byte records and two zero headers.
        total = ((padded_bytes + entry_size + 1024 + 10239) // 10240) * 10240
        if batch and (total > max_bytes or len(batch) >= 45):
            batches.append(batch)
            batch, padded_bytes = {}, 0
        batch[name] = data
        padded_bytes += entry_size
    if batch:
        batches.append(batch)
    return batches


class Peak:
    def __enter__(self):
        self.p = psutil.Process()
        self.t0, self.c0, self.peak = time.perf_counter(), self.p.cpu_times(), self.p.memory_info().rss
        self.stop = threading.Event()

        def run():
            while not self.stop.wait(0.05):
                self.peak = max(self.peak, self.p.memory_info().rss)
        self.th = threading.Thread(target=run, daemon=True)
        self.th.start()
        return self

    def __exit__(self, *a):
        self.stop.set()
        self.th.join()
        c = self.p.cpu_times()
        self.result = {'wall_s': round(time.perf_counter() - self.t0, 2),
                       'cpu_user_s': round(c.user - self.c0.user, 2), 'cpu_system_s': round(c.system - self.c0.system, 2),
                       'peak_rss_mb': round(self.peak / 2 ** 20, 1)}


# ------------------------------------------------------------------ state store
class State:
    def __init__(self, root: Path):
        # A changed extraction window must never reuse old NPZ arrays or hourly
        # rasters. Preserve the older cache intact in its original directory.
        self.root = root / G.GRID_ID
        for d in ('err', 'ir', 'visible', 'glm', 'acc'):
            (self.root / d).mkdir(parents=True, exist_ok=True)

    def err_slots(self):
        return {from_stamp(p.stem): p for p in (self.root / 'err').glob('*.npz')}

    def save_err(self, slot, d):
        np.savez_compressed(self.root / 'err' / f'{stamp(slot)}.npz', raw=d['raw'], valid=d['valid'],
                            meta=json.dumps(d['meta']))

    def load_err(self, path):
        z = np.load(path)
        return {'raw': z['raw'], 'valid': z['valid'], 'meta': json.loads(str(z['meta']))}

    def ir_slots(self):
        return {from_stamp(p.stem): p for p in (self.root / 'ir').glob('*.npz')}

    def save_ir(self, slot, d):
        np.savez_compressed(self.root / 'ir' / f'{stamp(slot)}.npz', bt=d['value'].astype(np.float32),
                            meta=json.dumps(d['meta']))

    def load_ir(self, path):
        z = np.load(path)
        return {'value': z['bt'], 'meta': json.loads(str(z['meta']))}

    def visible_path(self, slot):
        return self.root / 'visible' / f'{stamp(slot)}.npz'

    def save_visible(self, slot, d):
        # Retain calibrated source reflectances, including values >1.0. All colour
        # enhancement and display quantization happen downstream on copies.
        np.savez_compressed(self.visible_path(slot), **d['channels'], meta=json.dumps(d['meta']))

    def load_visible(self, slot):
        z = np.load(self.visible_path(slot))
        return {'channels': {k: z[k] for k in ('C01', 'C02', 'C03')}, 'meta': json.loads(str(z['meta']))}

    def glm_done(self):
        out = {}
        for p in (self.root / 'glm').glob('*.json'):
            out.update(json.loads(p.read_text())['files'])
        return out

    def save_glm(self, results):
        by_hour = {}
        for r in results:
            by_hour.setdefault(r['start'][:13], []).append(r)
        for hour, rs in by_hour.items():
            p = self.root / 'glm' / f'{hour.replace("-", "").replace("T", "")}.json'
            doc = json.loads(p.read_text()) if p.exists() else {'files': {}}
            for r in rs:
                doc['files'][r['file']] = {'start': r['start'], 'end': r['end'], 'flashes': r['flashes']}
            p.write_text(json.dumps(doc, separators=(',', ':')))

    def prune(self, now):
        cut = now - timedelta(hours=26)
        for d in ('err', 'ir', 'visible'):
            for p in (self.root / d).glob('*.npz'):
                if from_stamp(p.stem) < cut:
                    p.unlink()
        for p in (self.root / 'glm').glob('*.json'):
            if datetime.strptime(p.stem, '%Y%m%d%H').replace(tzinfo=UTC) < cut:
                p.unlink()
        for p in (self.root / 'acc').glob('*.png'):
            try:
                ts_str = p.stem.split('_', 1)[1]
                if from_stamp(ts_str) < cut:
                    p.unlink()
            except Exception:
                pass


# ------------------------------------------------------------------ masks
def admin_masks(lat, lon):
    from shapely import contains_xy
    from shapely.geometry import shape
    doc = json.loads(ADMIN.read_text(encoding='utf-8'))
    out = {}
    for f in doc['features']:
        p = f['properties']
        if p['level'] in (1, 3):
            out[p['id']] = (p, contains_xy(shape(f['geometry']), lon, lat))
    return out


def pool_map(fn, items, workers=8):
    res, errs = [], []
    with ThreadPoolExecutor(workers) as ex:
        futs = {ex.submit(fn, it): it for it in items}
        for fu, it in futs.items():
            try:
                res.append((it, fu.result()))
            except Exception as e:  # recorded, never silently replaced by data
                errs.append({'key': it.get('key', '?').rsplit('/', 1)[-1], 'error': f'{type(e).__name__}: {e}'[:300]})
    return res, errs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--state', default='.sat_state')
    ap.add_argument('--out', default='build/sat')
    ap.add_argument('--max-err', type=int, default=160, help='max ERR scans fetched this run')
    ap.add_argument('--glm-hours', type=float, default=1.25, help='GLM look-back fetched this run')
    ap.add_argument('--max-glm', type=int, default=400)
    ap.add_argument('--now', default=None, help='override current UTC time (tests)')
    ap.add_argument('--cadence', default='cadencia operativa GOES-19 cada 10 min')
    args = ap.parse_args()
    now = datetime.fromisoformat(args.now).replace(tzinfo=UTC) if args.now else datetime.now(UTC)
    st = State(Path(args.state))
    out = Path(args.out)
    (out / 'assets').mkdir(parents=True, exist_ok=True)
    errors = {}
    with Peak() as prof:
        lat, lon = G.native_latlon()
        masks = admin_masks(lat, lon)

        # ---------------- ERR: catalog 25 h, backfill newest-first
        cat = S.err_catalog(now, 25)
        have = st.err_slots()
        window_start = now - timedelta(hours=24, minutes=40)
        todo = [it for it in reversed(cat) if it['slot'] not in have and it['slot'] >= window_start][:args.max_err]
        res, errors['err'] = pool_map(S.err_read, todo)
        for it, d in res:
            d['meta'].update({'key': it['key'], 'object_last_modified': it['last_modified'], 'etag': it['etag'],
                              'bytes': d['bytes']})
            st.save_err(it['slot'], d)
        frames = {s: st.load_err(p) for s, p in st.err_slots().items() if s + timedelta(minutes=10) <= now}
        latest_slot = max(frames) if frames else None

        # ---------------- ABI IR (C13) timeline + cloud products (latest common slot)
        ir_cat = S.abi_catalog('ABI-L2-CMIPF', now, TIMELINE_H, band='M6C13')
        ir_have = st.ir_slots()
        ir_todo = [it for it in ir_cat if it['slot'] not in ir_have and it['slot'] >= now - timedelta(hours=TIMELINE_H)]
        res, errors['ir'] = pool_map(S.abi_read, ir_todo, workers=6)
        for it, d in res:
            st.save_ir(it['slot'], d)
        ir = {s: st.load_ir(p) for s, p in st.ir_slots().items() if s + timedelta(minutes=10) <= now}

        prods = ['ABI-L2-ACMF', 'ABI-L2-ACHA2KMF', 'ABI-L2-ACHTF', 'ABI-L2-ACTPF', 'ABI-L2-COD2KMF']
        cats = {p: {it['slot']: it for it in S.abi_catalog(p, now, 1)} for p in prods}
        common = set(cats['ABI-L2-ACMF']) & set(cats['ABI-L2-ACHA2KMF']) & set(cats['ABI-L2-ACTPF']) & \
            set(cats['ABI-L2-ACHTF']) & set(ir)
        cloud_slot = max(common) if common else None
        cloud = {}
        if cloud_slot:
            res, errors['cloud'] = pool_map(S.abi_read, [cats[p][cloud_slot] for p in prods if cloud_slot in cats[p]], 5)
            cloud = {it['product']: d for it, d in res}

        # ---------------- ABI visible, NOAA co-registered 2-km MCMIP product.
        # Download only the latest scene, and only while a part of this domain is
        # above 5 degrees solar elevation. Never fabricate night-time reflectance.
        visible = None
        visible_daylight = None
        visible_reason = 'Producto visible no disponible'
        errors['visible'] = []
        try:
            vis_cat = S.abi_catalog('ABI-L2-MCMIPF', now, 1)
            if vis_cat:
                item = vis_cat[-1]
                visible_daylight = A.solar_cosine(lat, lon, item['end']) > np.sin(np.radians(5))
                scene_age = now - item['end']
                if visible_daylight.any() and timedelta(0) <= scene_age <= timedelta(minutes=30):
                    if st.visible_path(item['slot']).exists():
                        visible = st.load_visible(item['slot'])
                    else:
                        visible = S.abi_visible_read(item)
                        st.save_visible(item['slot'], visible)
                elif not visible_daylight.any():
                    visible_reason = 'Noche / crepúsculo: no se publica visible; usar infrarrojo calibrado'
                else:
                    visible_reason = 'Escena visible fuera de la ventana vigente de 30 minutos'
        except Exception as e:
            errors['visible'].append({'error': f'{type(e).__name__}: {e}'[:300]})
            visible_reason = 'Proveedor visible no disponible: no se sustituye por textura sintética'

        # ---------------- GLM
        glm_cat = S.glm_catalog(now, int(np.ceil(args.glm_hours)) + 1)
        done = st.glm_done()
        g_todo = [it for it in reversed(glm_cat) if it['key'].rsplit('/', 1)[-1] not in done
                  and it['start'] >= now - timedelta(hours=args.glm_hours)][:args.max_glm]
        res, errors['glm'] = pool_map(lambda it: S.glm_read(it, G.BBOX), g_todo, workers=12)
        st.save_glm([d for _, d in res])
        glm_files = {k: v for k, v in st.glm_done().items() if datetime.fromisoformat(v['end'].replace('Z', '+00:00')) <= now}

        st.prune(now)

        # ======================================================== products
        assets, manifest = {}, {'schema_version': SCHEMA, 'generated_at': S.iso(now)}
        cell_area = G.pixel_area_km2(lat, lon)
        regions = {k: v for k, v in masks.items() if v[0]['level'] == 1}
        districts = {k: v for k, v in masks.items() if v[0]['level'] == 3}

        # ERR timeline frames
        timeline = []
        if latest_slot:
            for s in sorted(frames):
                if s < latest_slot - timedelta(hours=TIMELINE_H) + timedelta(minutes=10):
                    continue
                name = f'err_{stamp(s)}.png'
                assets[name] = E.err_png(frames[s]['raw'], frames[s]['valid'])
                m = frames[s]['meta']
                timeline.append({'asset': name, 'slot': S.iso(s), 'scan_start': m['time_coverage_start'],
                                 'scan_end': m['time_coverage_end'], 'created': m['date_created'],
                                 'published_s3': m['object_last_modified']})
        cur = frames.get(latest_slot) if latest_slot else None

        def rate_stats(mask):
            if cur is None:
                return None
            v = cur['raw'].astype(float) * 0.1
            return A.area_stats(v, cur['valid'], mask, (0.1, 1, 5, 10))

        # A single completed-hour window supplies maps, statistics and metadata.
        # Previously a rolling window timestamp was attached to an older raster,
        # causing up to 50 minutes of temporal mislabelling and complete-without-asset.
        accs = {}
        hourly_slot = accumulation_anchor(latest_slot) if latest_slot else None
        for hrs in A.WINDOWS_H:
            acc, ok, info = A.accumulate(frames, hourly_slot, hrs) if hourly_slot else (None, None, {'complete': False, 'hours': hrs, 'reason': 'sin escaneos ERR'})
            if acc is not None:
                name = f'acc{hrs}h_{stamp(hourly_slot)}.png'
                assets[name] = E.acc_png(acc, ok)
                info['asset'] = name
                info['update_cadence'] = 'Última hora UTC completada; mismo intervalo para raster y estadísticas'
            accs[hrs] = (acc, ok, info)

        # GLM flashes and coverage
        all_flashes, covered = [], []
        records = {(rec['start'],rec['end']):rec for fname,rec in sorted(glm_files.items())}
        for rec in records.values():
            covered.append((datetime.fromisoformat(rec['start'][:19]).replace(tzinfo=UTC),
                            datetime.fromisoformat(rec['end'][:19]).replace(tzinfo=UTC)))
            all_flashes.extend(rec['flashes'])

        def glm_coverage(t0, t1):
            return A.coverage_fraction(covered, t0, t1)

        def flashes_between(t0, t1):
            return [f for f in all_flashes if t0 <= datetime.fromisoformat(f['t'][:19]).replace(tzinfo=UTC) < t1]

        # cloud classification
        cloud_info = None
        if cloud_slot and 'ABI-L2-ACMF' in cloud and 'ABI-L2-ACHA2KMF' in cloud:
            nan = np.full(lat.shape, np.nan, np.float32)
            gv = lambda p: cloud[p]['value'] if p in cloud else nan
            bt = ir[cloud_slot]['value']
            prev = ir.get(cloud_slot - timedelta(minutes=10))
            bt_prev = prev['value'] if prev else None
            t_end = cloud_slot + timedelta(minutes=10)
            recent = flashes_between(cloud_slot - timedelta(minutes=10), t_end + timedelta(minutes=10))
            near = A.flash_proximity(lat, lon, recent, 10.0)
            cls, evidence = A.classify(gv('ABI-L2-ACMF'), gv('ABI-L2-ACHA2KMF'), gv('ABI-L2-ACHTF'),
                                       gv('ABI-L2-ACTPF'), gv('ABI-L2-COD2KMF'), bt, bt_prev, near)
            mot = A.motion(bt_prev, bt)
            if mot and mot.get('shift_px') is not None:
                dy, dx = mot['shift_px']
                km_y, km_x = dy * 2.2, dx * 2.05  # native pixel spacing at ~5 S (measured)
                mot.update({'u_kmh': round(km_x * 6, 1), 'v_kmh': round(-km_y * 6, 1),
                            'interval_min': 10, 'method': 'correlación cruzada IR 10.3 µm entre dos escaneos'})
            cname = f'cloud_{stamp(cloud_slot)}.png'
            assets[cname] = E.cloud_png(cls, gv('ABI-L2-ACHA2KMF'), gv('ABI-L2-ACHTF'))
            c2name = f'cloud2_{stamp(cloud_slot)}.png'
            assets[c2name] = E.cloud2_png(gv('ABI-L2-COD2KMF'), gv('ABI-L2-ACTPF'), gv('ABI-L2-ACMF'))
            cloud_info = {
                'slot': S.iso(cloud_slot),
                'products': {p: cloud[p]['meta'] for p in cloud},
                'assets': {'classes_height_temp': cname, 'cod_phase_mask': c2name},
                'classes': {str(k): {'id': v[0], 'label': v[1]} for k, v in A.CLOUD_CLASSES.items()},
                'evidence': evidence, 'motion': mot,
                'region_class_fraction': {
                    rid: {str(k): round(float((cls[m] == k).sum() / max(1, m.sum())), 4) for k in range(10) if (cls[m] == k).any()}
                    for rid, (_, m) in regions.items()},
                'rules': {
                    'conveccion_profunda_probable': 'nube con cima ≥ 8 km, T cima ≤ 235 K, fase hielo, Tb 10.3 µm ≤ 235 K y espesor óptico ≥ 20 (día) o núcleo frío ≤ 225 K (noche)',
                    'cumulonimbus_probable': 'convección profunda probable + T cima ≤ 225 K + segunda evidencia independiente: rayo GLM a ≤ 10 km (±10 min) o enfriamiento ≤ −4 K en 10 min',
                    'estratos': 'cima < 2.5 km, fase líquida, textura IR homogénea (σ5x5 < 0.8 K)',
                    'cumulos': 'cima < 4 km, no hielo, textura IR heterogénea (σ5x5 ≥ 1.5 K); cúmulos pequeños son subpíxel',
                    'nota': 'Clasificación conservadora por píxel de 2 km en nadir. No identifica especies ni confirma cumulonimbus. Sin corrección de paralaje: la posición aparente de nubes altas puede desplazarse varios km, según altura y ángulo de observación.'},
            }
            ir_frames = []
            for s in sorted(ir):
                if s < cloud_slot - timedelta(hours=TIMELINE_H) + timedelta(minutes=10):
                    continue
                n = f'ir_{stamp(s)}.png'
                assets[n] = E.ir_png(ir[s]['value'])
                ir_frames.append({'asset': n, 'slot': S.iso(s), 'scan_start': ir[s]['meta']['time_coverage_start'],
                                  'scan_end': ir[s]['meta']['time_coverage_end']})
            cloud_info['ir_timeline'] = ir_frames
            cloud_info['visible_status'] = visible_reason if visible is None else 'AVAILABLE'
            if visible is not None:
                name = f'visible_{stamp(datetime.fromisoformat(visible["meta"]["time_coverage_start"].replace("Z", "+00:00")))}.png'
                assets[name] = E.visible_png(visible['channels'], visible_daylight)
                cloud_info['assets']['visible_reflectance'] = name
                cloud_info['visible'] = {
                    'asset': name, 'scan_start': visible['meta']['time_coverage_start'],
                    'scan_end': visible['meta']['time_coverage_end'], 'resolution_km': 2,
                    'daylight_fraction': round(float(visible_daylight.mean()), 4),
                    'method': 'ABI C01 azul, C02 rojo y C03 NIR; R=C02,G=C03,B=C01,A=DQF0 y sol>5°. Reflectancia display 1/255; verde sintético se deriva sólo en la vista natural.',
                    'source_file': visible['meta']['file'], 'source_key': visible['meta']['key'],
                    'source_etag': visible['meta']['etag'], 'source_units': '1',
                    'display_encoding': 'RGBA8 reflectance (0..1); R=C02 G=C03 B=C01 A=daylight_quality_mask',
                }

        # ---------------- regional & district indicators
        def zone_block(mask):
            blk = {'rate_now': rate_stats(mask)}
            for hrs in A.WINDOWS_H:
                acc, ok, info = accs[hrs]
                if acc is None:
                    blk[f'acc_{hrs}h'] = None
                    continue
                s = A.area_stats(acc, ok, mask, (0.1, 1, 8, 25))
                s['level_of_max'] = A.rain_level(s['max'])
                s['level_of_p90'] = A.rain_level(s['p90'])
                if hrs == 24:
                    s['level_area_fraction'] = A.level_distribution(acc, ok, mask)
                blk[f'acc_{hrs}h'] = s
            return blk

        manifest['regions'] = {rid: {'name': p['name'], 'area_km2': round(float(cell_area[m].sum())), **zone_block(m)}
                               for rid, (p, m) in regions.items()}
        manifest['districts'] = {}
        for did, (p, m) in districts.items():
            if m.sum() == 0:  # district smaller than a 2 km pixel centre: nearest pixel
                r, c = G.latlon_to_native(p['label_lat'], p['label_lon'])
                m = np.zeros(lat.shape, bool)
                if int(r) >= 0:
                    m[int(r), int(c)] = True
            b = zone_block(m)
            manifest['districts'][did] = {'name': p['name'], 'province': p['province'], 'region': p['region'],
                                          'pixels': int(m.sum()),
                                          'rate_max': b['rate_now']['max'] if b['rate_now'] else None,
                                          'rate_valid_fraction': b['rate_now']['valid_fraction'] if b['rate_now'] else None,
                                          **{f'acc{h}h': (None if b[f'acc_{h}h'] is None else
                                                          {k: b[f'acc_{h}h'][k] for k in ('max', 'mean', 'p90', 'valid_fraction')})
                                             for h in A.WINDOWS_H}}

        # ---------------- localities (nearest native pixel; not street-level)
        locs = json.loads(LOCALITIES.read_text(encoding='utf-8'))['localities']
        acc24, ok24, info24 = accs[24]
        win_end = latest_slot + timedelta(minutes=10) if latest_slot else now
        win24 = (win_end - timedelta(hours=24), win_end)
        cov24 = glm_coverage(*win24)
        fl24 = flashes_between(*win24)
        manifest['localities'] = []
        for L in locs:
            r, c = (int(x) for x in G.latlon_to_native(L['lat'], L['lon']))
            row = {'id': L['id'], 'name': L['name'], 'region': L['region'], 'lat': L['lat'], 'lon': L['lon'],
                   'pixel': {'row': r + G.R0, 'col': c + G.C0, 'lat': round(float(lat[r, c]), 4),
                             'lon': round(float(lon[r, c]), 4), 'area_km2': round(float(cell_area[r, c]), 2)}}
            row['rate_mm_h'] = (float(cur['raw'][r, c]) * 0.1 if cur is not None and cur['valid'][r, c] else None)
            for hrs in A.WINDOWS_H:
                acc, ok, _ = accs[hrs]
                row[f'acc{hrs}h_mm'] = (round(float(acc[r, c]), 2) if acc is not None and ok[r, c] else None)
            n20 = sum(1 for f in fl24 if ((f['lat'] - L['lat']) * 110.6) ** 2 +
                      ((f['lon'] - L['lon']) * 111.32 * np.cos(np.radians(L['lat']))) ** 2 <= 400)
            row['rain_index_observed'] = A.rain_level(row['acc24h_mm'])
            row['flashes_20km_24h'] = n20
            row['storm_index_observed'] = (A.storm_level(row['acc24h_mm'], n20, cov24 == 1)
                                           if (n20 > 0 or row['acc24h_mm'] is not None) else None)
            manifest['localities'].append(row)

        # ---------------- glm block
        t_now_end = max((e for _, e in covered), default=None)
        last60 = flashes_between(t_now_end - timedelta(minutes=60), t_now_end) if t_now_end else []
        manifest['glm'] = {
            'source': 'NOAA GOES-19 GLM L2 LCFA', 'quality_filter': 'flash_quality_flag == 0',
            'latest_data_end': S.iso(t_now_end),
            'coverage_fraction': {'60min': glm_coverage(t_now_end - timedelta(minutes=60), t_now_end) if t_now_end else 0,
                                  '24h': round(cov24, 4)},
            'flashes_60min': sorted(last60, key=lambda f: f['t']),
            'flash_count_24h_bbox': len(fl24),
            'note': 'GLM detecta rayos ópticamente (sin distinguir nube-tierra); eficiencia de detección no medida localmente.'}

        manifest['err'] = {
            'source': 'NOAA/NESDIS Enterprise Rain Rate (ERR), GOES-19 ABI, v1r1',
            'bucket': 's3://noaa-enterprise-rainrate-pds/G19/ABI-L2-FLDK-RainfallRate/',
            'nature': 'ESTIMACIÓN SATELITAL (infrarrojo ABI calibrado con microondas); no es pluviómetro ni radar',
            'units': 'mm/h', 'native_resolution': '2 km en el nadir; huella crece con el ángulo de observación (ver grid.native.pixel_area_km2)',
            'source_cadence_min': 10,
            'latest': timeline[-1] if timeline else None,
            'timeline': timeline,
            'accumulations': {str(h): accs[h][2] for h in A.WINDOWS_H},
            'accumulation_method': 'Σ tasa_i × 10 min sobre todos los escaneos consecutivos de la ventana; solo con serie completa y píxel válido (DQF = 0) en todos; estimación, no total medido',
            'quality_rule': 'DQF == 0 (todos los bits de calidad en cero) y valor distinto de _FillValue',
            'frames_in_state': len(frames),
        }
        manifest['cloud'] = cloud_info
        manifest['grid'] = {'id': G.GRID_ID, 'display': {'bbox': G.BBOX, 'step_deg': G.DSTEP, 'nx': G.DNX, 'ny': G.DNY,
                                        'origin': 'centro de celda = (west + (i+0.5)·step, north − (j+0.5)·step)',
                                        'resampling': 'vecino más cercano desde el píxel nativo; sin interpolación'},
                            'native': {'rows': [G.R0, G.R1], 'cols': [G.C0, G.C1], 'projection': 'GOES-R ABI fixed grid, lon_0 = −75.0°',
                                       'pixel_area_km2': {'min': round(float(np.nanmin(cell_area)), 3), 'max': round(float(np.nanmax(cell_area)), 3)},
                                       'cloud_top_reference': 'height above mean sea level (ABI ACHA retrieval); no terrain correction'},
                            'encoding': E.__doc__}
        manifest['freshness_rules_min'] = {'rate_current_max_age': 75, 'cloud_current_max_age': 75,
                                           'lightning_visual_max_age': 75, 'manifest_stale_after': 150}
        manifest['processing'] = {'cadence': args.cadence,
                                  'source_cadence_min': 10}
        manifest['attribution'] = [
            'NOAA/NESDIS (ERR, GOES-19 ABI y GLM) vía NOAA Open Data Dissemination; datos modificados (recorte, remuestreo y derivados). Sin aval de NOAA.',
            'Límites: IGN Perú / OCHA / geoBoundaries (CC BY 3.0 IGO)']
    # Content-address assets before publishing. A reprocessed slot can never poison
    # browser/Worker immutable caches, and the manifest records exact SHA-256 hashes.
    names = {n: n[:-4] + '_' + hashlib.sha256(b).hexdigest()[:16] + '.png' for n,b in assets.items()}
    def rename(value):
        if isinstance(value, dict): return {k: rename(v) for k,v in value.items()}
        if isinstance(value, list): return [rename(v) for v in value]
        return names.get(value, value) if isinstance(value, str) else value
    manifest = rename(manifest)
    assets = {names[n]: b for n,b in assets.items()}
    manifest['assets'] = {n:{'sha256':hashlib.sha256(b).hexdigest(),'bytes':len(b)} for n,b in assets.items()}

    # ---------------- Tar bundling for KV write reduction (Niño Costero 2.3.6)
    bundles = {}
    # 1. Hourly accumulations bundle: acc1h, acc3h, acc6h, acc12h, acc24h
    acc_files = {}
    for h in A.WINDOWS_H:
        acc_meta = manifest.get('err', {}).get('accumulations', {}).get(str(h), {})
        an = acc_meta.get('asset')
        if an and an in assets:
            acc_files[an] = assets[an]

    if len(acc_files) >= 1:
        acc_slot_dt = hourly_slot if 'hourly_slot' in locals() and hourly_slot else (latest_slot or now)
        mtime_epoch = int(acc_slot_dt.timestamp()) if isinstance(acc_slot_dt, datetime) else 1700000000
        acc_tar, acc_hash = make_tar_bundle(acc_files, mtime=mtime_epoch)
        acc_bname = f"acc_{stamp(acc_slot_dt)}_{acc_hash[:16]}.tar"
        bundles[acc_bname] = {'sha256': acc_hash, 'bytes': len(acc_tar), 'files': list(acc_files.keys()), 'bytes_tar': acc_tar}
        for fn in acc_files:
            manifest['assets'][fn]['bundle'] = acc_bname

    # 2. Scan + timeline bundle: all remaining referenced assets (latest ERR, ERR timeline,
    # cloud top/phase/mask rasters, and IR timeline) so unbundled_to_upload is always 0
    # and historical timeline frames never lose their bundle reference across 10-min cycles.
    scan_files = {fn: b for fn, b in assets.items() if fn not in acc_files}
    if scan_files:
        slot_dt = latest_slot or cloud_slot or now
        mtime_epoch = int(slot_dt.timestamp()) if isinstance(slot_dt, datetime) else 1700000000
        partitions = partition_tar_assets(scan_files)
        for part_index, partition in enumerate(partitions):
            scan_tar, scan_hash = make_tar_bundle(partition, mtime=mtime_epoch)
            if len(scan_tar) > 4 * 1024 * 1024:
                raise ValueError('scan bundle exceeds authenticated 4-MiB limit')
            prefix = 'scan' if len(partitions) == 1 else f'scan{part_index+1}'
            scan_bname = f"{prefix}_{stamp(slot_dt)}_{scan_hash[:16]}.tar"
            bundles[scan_bname] = {'sha256': scan_hash, 'bytes': len(scan_tar), 'files': list(partition.keys()), 'bytes_tar': scan_tar}
            for fn in partition:
                manifest['assets'][fn]['bundle'] = scan_bname

    manifest['bundles'] = {bn: {'sha256': bd['sha256'], 'bytes': bd['bytes'], 'files': bd['files']} for bn, bd in bundles.items()}
    manifest['errors'] = {k: v[:20] for k, v in errors.items() if v}
    manifest['generated_at'] = S.iso(now if args.now else datetime.now(UTC))
    manifest['run'] = {**prof.result, 'transfer': METER.summary(),
                       'fetched': {'err_scans': len(todo), 'ir_scans': len(ir_todo), 'glm_files': len(g_todo)},
                       'assets': len(assets), 'assets_bytes': sum(len(v) for v in assets.values()),
                       'bundles': len(bundles), 'bundles_bytes': sum(bd['bytes'] for bd in bundles.values()),
                       'runner': os.environ.get('RUNNER_OS', 'local')}
    (out / 'bundles').mkdir(parents=True, exist_ok=True)
    for bn, bd in bundles.items():
        (out / 'bundles' / bn).write_bytes(bd['bytes_tar'])
    for n, b in assets.items():
        (out / 'assets' / n).write_bytes(b)
    (out / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, allow_nan=False,
                                                  separators=(',', ':')), encoding='utf-8')
    print(json.dumps({'latest_err': manifest['err']['latest'] and manifest['err']['latest']['slot'],
                      'acc_complete': {h: accs[h][2]['complete'] for h in A.WINDOWS_H},
                      'cloud_slot': cloud_info and cloud_info['slot'], 'run': manifest['run'],
                      'errors': {k: len(v) for k, v in errors.items()}}, indent=1, default=str))


if __name__ == '__main__':
    main()
