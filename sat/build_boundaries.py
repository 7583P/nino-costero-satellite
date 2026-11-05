"""Build the Costa Norte administrative layer (regions, provinces, districts).

Source: Instituto Geográfico Nacional (IGN) Perú via OCHA HDX, redistributed by
geoBoundaries gbHumanitarian (commit 9469f09). License CC BY 3.0 IGO.
Usage: python -I scripts/sat/build_boundaries.py <dir with ADM1/2/3.geojson> <out.json>
Hierarchy (district -> province -> region) is derived spatially with representative points,
because the source carries no parent codes.
"""
import hashlib
import json
import sys
import unicodedata
from pathlib import Path

from shapely.geometry import mapping, shape
from shapely.ops import unary_union

REGIONS = {'Tumbes': 'tumbes', 'Piura': 'piura', 'Lambayeque': 'lambayeque'}
TOL = {1: 0.0015, 2: 0.0012, 3: 0.0008}  # degrees (~170 m, ~130 m, ~90 m)


def slug(text):
    t = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode().lower()
    return ''.join(c if c.isalnum() else '-' for c in t).strip('-')


def rounded(geom, nd=4):
    def r(c):
        if isinstance(c[0], (int, float)):
            return [round(c[0], nd), round(c[1], nd)]
        return [r(x) for x in c]
    g = mapping(geom)
    return {'type': g['type'], 'coordinates': r(g['coordinates'])}


def main(src, out):
    src = Path(src)
    load = lambda n: json.loads((src / f'ADM{n}.geojson').read_text(encoding='utf-8'))['features']
    sha = {n: hashlib.sha256((src / f'ADM{n}.geojson').read_bytes()).hexdigest() for n in (1, 2, 3)}
    regions = {}
    for f in load(1):
        name = f['properties']['shapeName']
        if name in REGIONS:
            regions[REGIONS[name]] = (name, shape(f['geometry']).buffer(0))
    assert set(regions) == set(REGIONS.values()), regions.keys()

    def parent(geom, candidates):
        p = geom.representative_point()
        hits = [k for k, (_, g) in candidates.items() if g.contains(p)]
        return hits[0] if len(hits) == 1 else None

    features, provinces = [], {}
    for rid, (name, g) in regions.items():
        features.append({'type': 'Feature', 'properties': {'level': 1, 'id': rid, 'name': name, 'region': rid},
                         'geometry': rounded(g.simplify(TOL[1], preserve_topology=True))})
    for f in load(2):
        g = shape(f['geometry']).buffer(0)
        rid = parent(g, regions)
        if not rid:
            continue
        name = f['properties']['shapeName']
        pid = f'{rid}/{slug(name)}'
        provinces[pid] = (name, g, rid)
        features.append({'type': 'Feature', 'properties': {'level': 2, 'id': pid, 'name': name, 'region': rid},
                         'geometry': rounded(g.simplify(TOL[2], preserve_topology=True))})
    pmap = {k: (v[0], v[1]) for k, v in provinces.items()}
    for f in load(3):
        g = shape(f['geometry']).buffer(0)
        if not any(r[1].intersects(g) for r in regions.values()):
            continue
        pid = parent(g, pmap)
        if not pid:
            continue
        name = f['properties']['shapeName']
        did = f'{pid}/{slug(name)}'
        c = g.representative_point()
        features.append({'type': 'Feature', 'properties': {
            'level': 3, 'id': did, 'name': name, 'province': provinces[pid][0], 'province_id': pid,
            'region': provinces[pid][2], 'label_lon': round(c.x, 4), 'label_lat': round(c.y, 4),
            'area_km2': round(g.area * 111.32 * 110.6, 1)},
            'geometry': rounded(g.simplify(TOL[3], preserve_topology=True))})
    ids = [f['properties']['id'] for f in features]
    assert len(ids) == len(set(ids)), 'duplicate ids'
    counts = {lvl: sum(1 for f in features if f['properties']['level'] == lvl) for lvl in (1, 2, 3)}
    doc = {'type': 'FeatureCollection', 'metadata': {
        'source': 'Instituto Geográfico Nacional (IGN) del Perú, vía OCHA HDX (COD-AB) y geoBoundaries gbHumanitarian',
        'license': 'CC BY 3.0 IGO', 'source_year': 2015,
        'urls': ['https://www.geoboundaries.org/api/current/gbHumanitarian/PER/ADM3',
                 'https://data.humdata.org/dataset/cod-ab-per'],
        'input_sha256': sha, 'simplification_deg': TOL, 'counts': counts,
        'hierarchy': 'derivada espacialmente (punto representativo); la fuente no incluye códigos padre',
        'attribution': 'Límites: IGN Perú / OCHA / geoBoundaries (CC BY 3.0 IGO)'},
        'features': features}
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(doc, ensure_ascii=False, separators=(',', ':')), encoding='utf-8')
    print(counts, Path(out).stat().st_size)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
