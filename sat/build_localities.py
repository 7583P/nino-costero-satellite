"""One-off geocoding of reference localities with OSM Nominatim (ODbL), 1 request/s.

Each result must fall inside the expected IGN district polygon, otherwise it is rejected.
Usage: python -I scripts/sat/build_localities.py web/public/data/geo/admin_costa_norte.json out.json
"""
import json
import sys
import time

import requests
from shapely.geometry import Point, shape

# (name shown, query, expected district id)
PLACES = [
    ('Chiclayo', 'Chiclayo, Lambayeque, Peru', 'lambayeque/chiclayo/chiclayo'),
    ('José Leonardo Ortiz', 'José Leonardo Ortiz, Chiclayo, Peru', 'lambayeque/chiclayo/jose-leonardo-ortiz'),
    ('La Victoria', 'La Victoria, Chiclayo, Lambayeque, Peru', 'lambayeque/chiclayo/la-victoria'),
    ('Pimentel', 'Pimentel, Chiclayo, Peru', 'lambayeque/chiclayo/pimentel'),
    ('Reque', 'Reque, Chiclayo, Peru', 'lambayeque/chiclayo/reque'),
    ('Monsefú', 'Monsefú, Chiclayo, Peru', 'lambayeque/chiclayo/monsefu'),
    ('Puerto Eten', 'Puerto Eten, Chiclayo, Peru', 'lambayeque/chiclayo/eten-puerto'),
    ('Santa Rosa', 'Santa Rosa, Chiclayo, Lambayeque, Peru', 'lambayeque/chiclayo/santa-rosa'),
    ('Tumán', 'Tumán, Chiclayo, Peru', 'lambayeque/chiclayo/tuman'),
    ('Lambayeque', 'Lambayeque, Lambayeque, Peru', 'lambayeque/lambayeque/lambayeque'),
    ('Mórrope', 'Mórrope, Lambayeque, Peru', 'lambayeque/lambayeque/morrope'),
    ('Motupe', 'Motupe, Lambayeque, Peru', 'lambayeque/lambayeque/motupe'),
    ('Olmos', 'Olmos, Lambayeque, Peru', 'lambayeque/lambayeque/olmos'),
    ('Ferreñafe', 'Ferreñafe, Lambayeque, Peru', 'lambayeque/ferrenafe/ferrenafe'),
    ('Piura', 'Piura, Piura, Peru', 'piura/piura/piura'),
    ('Castilla', 'Castilla, Piura, Peru', 'piura/piura/castilla'),
    ('Veintiséis de Octubre', 'Veintiséis de Octubre, Piura, Peru', 'piura/piura/veintiseis-de-octubre'),
    ('Catacaos', 'Catacaos, Piura, Peru', 'piura/piura/catacaos'),
    ('Sullana', 'Sullana, Piura, Peru', 'piura/sullana/sullana'),
    ('Paita', 'Paita, Piura, Peru', 'piura/paita/paita'),
    ('Talara', 'Talara, Piura, Peru', 'piura/talara/parinas'),
    ('Máncora', 'Máncora, Talara, Peru', 'piura/talara/mancora'),
    ('Sechura', 'Sechura, Piura, Peru', 'piura/sechura/sechura'),
    ('Chulucanas', 'Chulucanas, Morropón, Peru', 'piura/morropon/chulucanas'),
    ('Ayabaca', 'Ayabaca, Piura, Peru', 'piura/ayabaca/ayabaca'),
    ('Huancabamba', 'Huancabamba, Piura, Peru', 'piura/huancabamba/huancabamba'),
    ('Tumbes', 'Tumbes, Tumbes, Peru', 'tumbes/tumbes/tumbes'),
    ('Zarumilla', 'Zarumilla, Tumbes, Peru', 'tumbes/zarumilla/zarumilla'),
    ('Aguas Verdes', 'Aguas Verdes, Zarumilla, Peru', 'tumbes/zarumilla/aguas-verdes'),
    ('Zorritos', 'Zorritos, Tumbes, Peru', 'tumbes/contralmirante-villar/zorritos'),
]


def main(admin, out):
    feats = {f['properties']['id']: f for f in json.load(open(admin, encoding='utf-8'))['features']}
    s = requests.Session()
    s.headers['User-Agent'] = 'NinoCostero/2.0 one-off locality geocoding (+https://nino-costero.pages.dev)'
    res, rejected = [], []
    for name, q, did in PLACES:
        if did not in feats:
            rejected.append({'name': name, 'reason': f'district id not found: {did}'})
            continue
        r = s.get('https://nominatim.openstreetmap.org/search',
                  params={'q': q, 'format': 'jsonv2', 'limit': 5, 'countrycodes': 'pe'}, timeout=30)
        time.sleep(1.1)
        poly = shape(feats[did]['geometry'])
        hit = None
        for c in r.json():
            if c.get('addresstype') in ('city', 'town', 'village', 'suburb', 'hamlet', 'municipality', 'neighbourhood',
                                         'quarter', 'locality') or c.get('category') == 'place':
                if poly.buffer(0.01).contains(Point(float(c['lon']), float(c['lat']))):
                    hit = c
                    break
        if not hit:
            rejected.append({'name': name, 'reason': 'no OSM place inside expected district',
                             'candidates': [(c.get('display_name'), c.get('addresstype')) for c in r.json()[:3]]})
            continue
        p = feats[did]['properties']
        res.append({'id': did.split('/')[-1] if name != 'Talara' else 'talara', 'name': name,
                    'lat': round(float(hit['lat']), 5), 'lon': round(float(hit['lon']), 5),
                    'district_id': did, 'district': p['name'], 'province': p['province'], 'region': p['region'],
                    'osm': f"{hit['osm_type']}/{hit['osm_id']}"})
    doc = {'metadata': {'source': 'OpenStreetMap Nominatim (© colaboradores de OpenStreetMap, ODbL)',
                        'validated_against': 'polígonos distritales IGN (admin_costa_norte.json), tolerancia 0.01°',
                        'rejected': rejected}, 'localities': res}
    json.dump(doc, open(out, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print(len(res), 'ok;', len(rejected), 'rejected', rejected)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
