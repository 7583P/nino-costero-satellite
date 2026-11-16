"""V3.2 tests: real native-grid metadata + explicitly synthetic forecast values."""
from __future__ import annotations
import copy
from datetime import datetime, timedelta, timezone
import importlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from contextlib import redirect_stdout

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.build_province_forecast import DEFAULT_MAPPING, GRID_UUID, GRID_POINTS, PROVINCE_IDS, build_forecast_bundle, load_grid_mapping, strict_hourly_statistics, validate_forecast_bundle
from scripts import extract_icon_hourly as extractor

def evidence(step,run='2026111512'):
    return {'step':step,'url':f'https://opendata.dwd.de/weather/nwp/icon-eps/grib/{run[8:]}/tot_prec/SYNTHETIC_{step}.grib2.bz2','download_bytes':0,'sha256':'a'*64,'grib_sha256':'b'*64,'members_extracted':40,'grib_meta':{'numberOfGridUsed':39,'uuidOfHGrid':GRID_UUID,'numberOfDataPoints':GRID_POINTS,'numberOfForecastsInEnsemble':40,'units':'kg m**-2','stepType':'accum','startStep':0,'endStep':step,'dataDate':int(run[:8]),'dataTime':int(run[8:])*100}}
def fixture():
    mapping=load_grid_mapping();run='2026111512';start=datetime(2026,11,15,12,tzinfo=timezone.utc);intervals={}
    for province in mapping['provinces']:
        intervals[province['target_key']]=[{'step':i+1,'start_utc':(start+timedelta(hours=i)).isoformat().replace('+00:00','Z'),'end_utc':(start+timedelta(hours=i+1)).isoformat().replace('+00:00','Z'),'quality':'VALID','valid_members':40,'probability_percent':2.5,'exceeding_members':1,'mean_ensemble_mm':.005,'max_member_mm':.2} for i in range(24)]
    return build_forecast_bundle(run,intervals,mapping,'2026-11-15T15:00:00Z',[evidence(i) for i in range(1,25)])

class TestProvinceForecast(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        (ROOT/'.tools').mkdir(parents=True,exist_ok=True)

    def test_real_49_grid_mapping_and_original_three_indices(self):
        m=load_grid_mapping();self.assertEqual({p['id'] for p in m['provinces']},PROVINCE_IDS);self.assertEqual(len(m['provinces']),49)
        self.assertTrue(all(p['distance_km']<=25 for p in m['provinces']));self.assertEqual(m['grid']['grid_uuid'],GRID_UUID)
        self.assertTrue(all(c['matches'] for c in m['legacy_index_checks']));self.assertEqual(len(m['coordinate_source']),2)
        self.assertTrue(all(s['url'].startswith('https://opendata.dwd.de/') and len(s['sha256'])==64 and s['metadata']['uuidOfHGrid']==GRID_UUID for s in m['coordinate_source']))
        self.assertGreater(len({(p['latitude'],p['longitude']) for p in m['provinces']}),40)

    def test_exact_40_members_exceedance_probability_without_adjustment(self):
        self.assertEqual(strict_hourly_statistics([0]*40,[0]*40)[0][0],0)
        self.assertEqual(strict_hourly_statistics([0]*40,[.2]*40)[0][0],100)
        self.assertEqual(strict_hourly_statistics([0]*40,[.2]+[0]*39)[0][:3],[2.5,40,1])
        self.assertEqual(strict_hourly_statistics([0]*40,[.1]*17+[0]*23)[0][0],42.5)

    def test_missing_members_nan_and_negative_accumulations_remain_unavailable(self):
        for previous,current in [([0]*39,[1]*39),([0]*40,[None]+[1]*39),([0]*40,[float('nan')]+[1]*39),([0]*40,[9999]+[1]*39),([1]*40,[.5]+[1]*39)]:
            row,reason=strict_hourly_statistics(previous,current);self.assertIsNone(row);self.assertIsNotNone(reason)

    def test_compact_49_by_24_contract_is_bounded_and_preserves_values(self):
        d=fixture();self.assertTrue(validate_forecast_bundle(d));self.assertEqual(len(d['provinces']),49);self.assertEqual(len(d['hours']),24)
        self.assertTrue(all(p['hourly'][0]==[2.5,40,1,.005,.2] for p in d['provinces']));self.assertLess(len(json.dumps(d).encode()),400000)
        for mutation in ['percent','members','uuid','hours','source_boundary']:
            bad=copy.deepcopy(d)
            if mutation=='percent':bad['provinces'][0]['hourly'][0][0]=70
            elif mutation=='members':bad['provinces'][0]['hourly'][0][1]=39
            elif mutation=='uuid':bad['source']['grid_uuid']='0'*32
            elif mutation=='hours':bad['hours'][2]['end_utc']=bad['hours'][0]['end_utc']
            else:bad['source_files'].pop(1)
            with self.assertRaises(ValueError,msg=mutation):validate_forecast_bundle(bad)

    def test_one_shared_24_file_pass_produces_legacy3_and_49_without_extra_downloads(self):
        # Synthetic values, real validated 49-cell mapping; no network is used.
        calls=[]
        def fake(run,step,temp_dir,session,targets):
            calls.append(step)
            if step==5:raise OSError('SYNTHETIC upstream failure')
            return {member:{key:step*.2 if member==1 else 0. for key in targets} for member in range(1,41)},evidence(step,run)
        with tempfile.TemporaryDirectory(dir=ROOT/'.tools',prefix='province-forecast-test-') as temporary:
            base=Path(temporary);out=base/'public/v2/hourly'
            with patch.object(extractor,'ROOT',base),patch.object(extractor,'download_and_extract_step',fake):
                summary=extractor.process_icon_hourly('2026111512',24,.1,out,province_mapping_path=DEFAULT_MAPPING)
            self.assertEqual(calls,list(range(1,25)));self.assertEqual(summary['province_forecast']['provinces'],49)
            self.assertTrue(all((out/f'{c}.json').exists() for c in extractor.CITIES))
            d=json.loads((out/'provinces.json').read_text(encoding='utf-8'));self.assertTrue(validate_forecast_bundle(d))
            for p in d['provinces']:
                self.assertEqual(p['hourly'][3][0],2.5);self.assertEqual(p['hourly'][4],[None,0,None,None,None]);self.assertEqual(p['hourly'][5],[None,0,None,None,None]);self.assertEqual(p['hourly'][6][0],2.5)
            old=json.loads((out/'piura.json').read_text(encoding='utf-8'));self.assertIsNone(old['intervals'][4]['probability_percent']);self.assertEqual(old['intervals'][6]['probability_percent'],2.5)

    @unittest.skipUnless((ROOT/'infra/worker/provinceForecast.mjs').exists(),'Worker integration belongs to application checkout, not satellite-only checkout')
    def test_worker_rejects_tampered_counts_and_auth_then_publishes_one_kv_key(self):
        # Cross-language worker integration against a synthetic forecast, not production.
        code="""import fs from 'node:fs';import {validProvinceForecast,createProvinceForecastService} from './infra/worker/provinceForecast.mjs';
const d=JSON.parse(fs.readFileSync(0,'utf8'));const yes=validProvinceForecast(d);const bad=structuredClone(d);bad.provinces[0].hourly[0][0]=70;const no=!validProvinceForecast(bad);const missing=structuredClone(d);missing.source_files=[];const noEvidence=!validProvinceForecast(missing);
let writes=0,stored=null;const env={PROVINCE_FORECAST_ENABLED:'true',PUBLISH_TOKEN:'test',DATA:{get:async()=>stored,put:async(k,v)=>{writes++;stored=JSON.parse(v);}}};const s=createProvinceForecastService({clock:()=>Date.parse('2026-11-15T20:00:00Z')});
const put=token=>s(new Request('https://test/admin/province-forecast',{method:'PUT',headers:{'Content-Type':'application/json',Authorization:'Bearer '+token},body:JSON.stringify(d)}),env);const unauthorized=(await put('wrong')).status;const publish=await (await put('test')).json();const again=await (await put('test')).json();const get=await (await s(new Request('https://test/v3/province-forecast.json'),env)).json();console.log(JSON.stringify({yes,no,noEvidence,unauthorized,publish,again,writes,availability:get.availability,freshness:get.freshness}));"""
        p=subprocess.run(['node','--input-type=module','-e',code],input=json.dumps(fixture()),text=True,capture_output=True,cwd=ROOT,check=True);r=json.loads(p.stdout)
        self.assertTrue(r['yes'] and r['no'] and r['noEvidence']);self.assertEqual(r['unauthorized'],401);self.assertEqual(r['writes'],1);self.assertTrue(r['publish']['published']);self.assertTrue(r['again']['unchanged']);self.assertEqual(r['availability'],'AVAILABLE');self.assertEqual(r['freshness'],'CURRENT')

    def test_publisher_sends_bulk_after_legacy_once_and_missing_bulk_preserves_legacy(self):
        sys.path.insert(0,str(ROOT/'scripts'))
        publisher=importlib.import_module('publish_hourly_snapshot')
        class Response:
            status_code=200
            def json(self):return {'published':True,'kv_keys_written':1}
        with tempfile.TemporaryDirectory(dir=ROOT/'.tools',prefix='province-publisher-test-') as temporary:
            base=Path(temporary);bulk=fixture()
            for dirname,include in [('with-province',True),('legacy-only',False)]:
                directory=base/dirname;directory.mkdir()
                for city in publisher.CITIES:
                    d={'schema_version':'1.0.0','city':{'id':city},'source':{'model':'ICON-EPS','total_ensemble_members':40},'run':bulk['run'],'generated_at':bulk['generated_at'],'intervals':[{'probability_percent':2.5,'quality':'VALID'}]}
                    (directory/f'{city}.json').write_text(json.dumps(d),encoding='utf-8')
                if include:(directory/'provinces.json').write_text(json.dumps(bulk),encoding='utf-8')
                argv=['publisher','--dir',str(directory),'--tag',dirname,'--send']
                with patch.object(sys,'argv',argv),patch.object(publisher,'RESULTS',base/'results'),patch.dict('os.environ',{'WORKER_URL':'https://test.invalid','PUBLISH_TOKEN':'synthetic-test'}),patch.object(publisher.requests,'put',return_value=Response()) as put,redirect_stdout(io.StringIO()):publisher.main()
                self.assertEqual(put.call_count,2 if include else 1)
                self.assertTrue(put.call_args_list[0].args[0].endswith('/admin/hourly'))
                if include:self.assertTrue(put.call_args_list[1].args[0].endswith('/admin/province-forecast'))

if __name__=='__main__':unittest.main()
