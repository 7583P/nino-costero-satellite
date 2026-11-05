"""Validate retained/processed satellite assets without downloading anything."""
import argparse
import hashlib
import io
import json
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
from PIL import Image
from . import analysis as A, grid as G

def validate(folder, state=None):
    folder=Path(folder)
    m=json.loads((folder/'manifest.json').read_text(encoding='utf8'))
    assert m['schema_version']=='2.0.0'
    for n,a in m['assets'].items():
        b=(folder/'assets'/n).read_bytes()
        assert len(b)==a['bytes'] and hashlib.sha256(b).hexdigest()==a['sha256'], n
        img=Image.open(io.BytesIO(b));assert img.size==(m['grid']['display']['nx'],m['grid']['display']['ny'])
        assert img.mode=='RGBA'
    for bn, bmeta in m.get('bundles', {}).items():
        bb = (folder / 'bundles' / bn).read_bytes()
        assert len(bb) == bmeta['bytes'] and hashlib.sha256(bb).hexdigest() == bmeta['sha256'], bn
    comparisons={}
    if state:
        frames={}
        for p in (Path(state)/'err').glob('*.npz'):
            z=np.load(p);t=datetime.strptime(p.stem,'%Y%m%dT%H%MZ').replace(tzinfo=timezone.utc)
            frames[t]={'raw':z['raw'],'valid':z['valid'],'meta':json.loads(str(z['meta']))}
        end=datetime.fromisoformat(m['err']['latest']['slot'].replace('Z','+00:00'))
        rr,cc=G.display_index();inside=rr>=0
        for h in A.WINDOWS_H:
            acc,ok,info=A.accumulate(frames,end,h)
            acc_meta = m['err']['accumulations'].get(str(h), {})
            assert info['complete']==acc_meta.get('complete')
            if not acc_meta.get('complete') or not acc_meta.get('asset'):continue
            asset_name = acc_meta['asset']
            try:
                ts_str = asset_name.split('_')[1]
                target_slot = datetime.strptime(ts_str, '%Y%m%dT%H%MZ').replace(tzinfo=timezone.utc)
            except Exception:
                target_slot = end
            acc_v, ok_v, _ = A.accumulate(frames, target_slot, h)
            if acc_v is None:
                acc_v, ok_v = acc, ok
            p=np.asarray(Image.open(folder/'assets'/asset_name),dtype=np.uint32)
            flag=p[:,:,2]&192;decoded=(((p[:,:,2]&63)<<16)|(p[:,:,0]<<8)|p[:,:,1])/60
            v=inside.copy()
            if ok_v is not None:
                v[inside]=ok_v[rr[inside],cc[inside]]
                assert np.array_equal(flag==64,v)
                diff=np.abs(decoded[v]-acc_v[rr[v],cc[v]])
                assert np.max(diff,initial=0)<1e-8
                comparisons[str(h)]={'valid_display_cells':int(v.sum()),'max_error_mm':float(np.max(diff,initial=0))}
        for l in m['localities']:
            r,c=G.latlon_to_native(l['lat'],l['lon']);assert r>=0 and c>=0
    return {'status':'PASS','generated_at':m['generated_at'],'assets':len(m['assets']),'bytes':sum(v['bytes'] for v in m['assets'].values()),'regions':list(m['regions']),'districts':len(m['districts']),'localities':len(m['localities']),'accumulations':comparisons,'run':m['run']}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--state');p.add_argument('--report');a=p.parse_args()
    r=validate(a.out,a.state)
    text=json.dumps(r,indent=2)
    if a.report:Path(a.report).write_text(text,encoding='utf8')
    print(text)
