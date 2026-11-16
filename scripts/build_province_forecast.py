"""DWD ICON-EPS province mapping and compact 49-capital forecast contract.

Mapping uses the official time-invariant CLAT/CLON of the very same native grid,
not city surrogates. Building a forecast consumes the per-member values already
decoded by extract_icon_hourly.py; it performs no forecast downloads.
"""
from __future__ import annotations
import argparse
import bz2
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
PROVINCE_IDS={f'{department}{i:02d}' for department,count in [('02',20),('11',5),('14',3),('15',10),('20',8),('24',3)] for i in range(1,count+1)}
GRID_UUID='49c340e2248d11e58105c12d5e3b00a4'
GRID_POINTS=1310720
DEFAULT_MAPPING=ROOT/'scripts/data/icon_province_grid.json'
SOURCE={'model':'ICON-EPS','producer':'Deutscher Wetterdienst (DWD)','license':'CC BY 4.0','license_url':'https://www.dwd.de/EN/service/legal_notice/legal_notice_node.html','total_ensemble_members':40,'grid_type':'unstructured_grid','number_of_grid_used':39,'number_of_data_points':GRID_POINTS,'grid_uuid':GRID_UUID,'grid_mesh':'R02B07','nominal_resolution_km':20}
COLUMNS=['probability_percent','valid_members','exceeding_members','mean_ensemble_mm','max_member_mm']

def finite(value):return isinstance(value,(int,float,np.integer,np.floating)) and not isinstance(value,bool) and math.isfinite(float(value))
def utc_iso(value):return value.astimezone(timezone.utc).isoformat().replace('+00:00','Z')
def distance_km(lat,lon,other_lat,other_lon):
    r=math.pi/180;dlat=(other_lat-lat)*r;dlon=(other_lon-lon)*r
    return 12742*math.asin(min(1,math.sqrt(math.sin(dlat/2)**2+math.cos(lat*r)*math.cos(other_lat*r)*math.sin(dlon/2)**2)))

def build_grid_mapping(catalog_path:Path,cache_dir:Path,session=None):
    import requests
    import eccodes as ec
    session=session or requests.Session();session.headers['User-Agent']='NinoCostero/3.2 (matiasgamesdev.support@gmail.com; ICON native grid metadata)'
    catalog=json.loads(catalog_path.read_text(encoding='utf-8'));provinces=catalog['provinces']
    if {p['id'] for p in provinces}!=PROVINCE_IDS or len(provinces)!=49:raise ValueError('Se requieren las 49 provincias verificadas')
    cache_dir.mkdir(parents=True,exist_ok=True);arrays={};evidence=[]
    for field,unit in [('clat','Degree N'),('clon','Degree E')]:
        base=f'https://opendata.dwd.de/weather/nwp/icon-eps/grib/00/{field}/'
        listing=session.get(base,timeout=(10,25));listing.raise_for_status()
        names=re.findall(r'href="([^"/]+\.bz2)"',listing.text)
        if not names:raise RuntimeError(f'Sin coordenadas oficiales {field}')
        filename=sorted(names)[-1];path=cache_dir/filename
        if not path.exists():
            response=session.get(base+filename,timeout=(15,45));response.raise_for_status()
            if len(response.content)>3000000:raise ValueError('Coordenadas exceden presupuesto de 3 MB/campo')
            path.write_bytes(response.content)
        compressed=path.read_bytes();raw=bz2.decompress(compressed);gid=ec.codes_new_from_message(raw)
        try:
            metadata={k:ec.codes_get(gid,k) for k in ['gridType','numberOfGridUsed','numberOfDataPoints','uuidOfHGrid','units','bitsPerValue','binaryScaleFactor','decimalScaleFactor']}
            if metadata['gridType']!='unstructured_grid' or metadata['numberOfGridUsed']!=39 or metadata['numberOfDataPoints']!=GRID_POINTS or metadata['uuidOfHGrid']!=GRID_UUID or metadata['units']!=unit:raise ValueError('Malla/unidades oficiales incompatibles')
            values=ec.codes_get_values(gid)
            if values.size!=GRID_POINTS or not np.isfinite(values).all():raise ValueError('Coordenadas incompletas')
            arrays[field]=np.asarray(values,dtype=float)
            quantum=2.0**metadata['binaryScaleFactor']*10.0**(-metadata['decimalScaleFactor'])
            evidence.append({'field':field,'url':base+filename,'compressed_bytes':len(compressed),'sha256':hashlib.sha256(compressed).hexdigest(),'grib_sha256':hashlib.sha256(raw).hexdigest(),'quantization_degrees':quantum,'metadata':metadata})
        finally:ec.codes_release(gid)
    lat=np.radians(arrays['clat']);lon=np.radians(arrays['clon']);sinlat=np.sin(lat);coslat=np.cos(lat);targets=[]
    max_position_uncertainty_km=math.hypot(evidence[0]['quantization_degrees']*111.32/2,evidence[1]['quantization_degrees']*111.32/2)
    for province in provinces:
        la=math.radians(province['lat']);lo=math.radians(province['lon'])
        chord=1-(sinlat*math.sin(la)+coslat*math.cos(la)*np.cos(lon-lo))
        first_two=np.argpartition(chord,1)[:2];first_two=sorted(first_two,key=lambda i:chord[i]);idx=int(first_two[0]);second=int(first_two[1])
        d=distance_km(province['lat'],province['lon'],arrays['clat'][idx],arrays['clon'][idx]);d2=distance_km(province['lat'],province['lon'],arrays['clat'][second],arrays['clon'][second])
        if d>25:raise ValueError(f"Celda demasiado alejada: {province['id']}")
        targets.append({'id':province['id'],'target_key':f"province_{province['id']}",'name':province['name'],'capital':province['capital'],'department_id':province['departmentId'],'latitude':province['lat'],'longitude':province['lon'],'elevation_m':province.get('elevationM'),'cell_index':idx,'cell_latitude':float(arrays['clat'][idx]),'cell_longitude':float(arrays['clon'][idx]),'distance_km':round(d,4),'nearest_margin_km':round(d2-d,4),'robust_to_coordinate_quantization':d2-d>2*max_position_uncertainty_km})
    legacy=[]
    for name,la,lo,expected in [('tumbes',-3.5669,-80.4515,539249),('piura',-5.1945,-80.6328,539161),('chiclayo',-6.7714,-79.8409,537055)]:
        chord=1-(sinlat*math.sin(math.radians(la))+coslat*math.cos(math.radians(la))*np.cos(lon-math.radians(lo)));found=int(np.argmin(chord))
        legacy.append({'id':name,'expected_index':expected,'nearest_published_index':found,'matches':expected==found})
    if not all(x['matches'] for x in legacy):raise ValueError('La verificación independiente no reproduce los índices legacy')
    return {'schema_version':'3.2.0','generated_at':utc_iso(datetime.now(timezone.utc)),'grid':dict(SOURCE),'catalog_sha256':hashlib.sha256(catalog_path.read_bytes()).hexdigest(),'coordinate_source':evidence,'coordinate_position_uncertainty_max_km':round(max_position_uncertainty_km,4),'method':'Nearest great-circle distance to official published CLAT/CLON native cell centres; no interpolation. Distances include coordinate quantization uncertainty.','legacy_index_checks':legacy,'provinces':targets}

def load_grid_mapping(path:Path=DEFAULT_MAPPING):
    mapping=json.loads(path.read_text(encoding='utf-8'))
    if mapping.get('schema_version')!='3.2.0' or mapping.get('grid',{}).get('grid_uuid')!=GRID_UUID or mapping.get('grid',{}).get('number_of_grid_used')!=39:raise ValueError('Mapping de malla incompatible')
    targets=mapping.get('provinces',[])
    if len(targets)!=49 or {p.get('id') for p in targets}!=PROVINCE_IDS:raise ValueError('Mapping de provincias incompleto')
    for p in targets:
        if not isinstance(p.get('cell_index'),int) or not 0<=p['cell_index']<GRID_POINTS or not all(finite(p.get(k)) for k in ['latitude','longitude','cell_latitude','cell_longitude','distance_km']) or p['distance_km']>25:raise ValueError('Mapping inválido')
    return mapping

def strict_hourly_statistics(previous,current,threshold=.1):
    """Require all 40 members and nonnegative finite accumulation differences."""
    if len(previous)!=40 or len(current)!=40:return None,'MISSING_MEMBERS'
    if not all(finite(p) and finite(c) and 0<=p<9999 and 0<=c<9999 for p,c in zip(previous,current)):return None,'MISSING_OR_INVALID_VALUE'
    increments=[float(c)-float(p) for p,c in zip(previous,current)]
    if any(x<0 for x in increments):return None,'NEGATIVE_ACCUMULATION'
    count=sum(value>=threshold for value in increments)
    return [count/40*100,40,count,round(sum(increments)/40,4),round(max(increments),4)],None

def validate_forecast_bundle(d):
    if d.get('schema_version')!='3.2.0' or d.get('nature')!='FORECAST' or d.get('columns')!=COLUMNS or d.get('event')!={'operator':'>=','amount_mm':.1,'period_hours':1,'unit':'mm'}:raise ValueError('Contrato provincial/evento incompatible')
    source=d.get('source',{})
    if any(source.get(k)!=SOURCE[k] for k in ['model','total_ensemble_members','number_of_grid_used','number_of_data_points','grid_uuid','nominal_resolution_km']):raise ValueError('Malla o fuente provincial incompatible')
    run_id=d.get('run',{}).get('run_id','')
    if not re.fullmatch(r'\d{10}',run_id):raise ValueError('Corrida provincial inválida')
    run=datetime.strptime(run_id,'%Y%m%d%H').replace(tzinfo=timezone.utc)
    if run.hour not in (0,12) or datetime.fromisoformat(d['run']['run_utc'].replace('Z','+00:00'))!=run:raise ValueError('Timestamp de corrida incompatible')
    datetime.fromisoformat(d['generated_at'].replace('Z','+00:00'))
    hours=d.get('hours',[]);provinces=d.get('provinces',[])
    if len(hours)!=24 or len(provinces)!=49 or {p.get('id') for p in provinces}!=PROVINCE_IDS:raise ValueError('Faltan horas o provincias')
    for i,h in enumerate(hours):
        if h['step']!=i+1 or datetime.fromisoformat(h['start_utc'].replace('Z','+00:00'))!=run+timedelta(hours=i) or datetime.fromisoformat(h['end_utc'].replace('Z','+00:00'))!=run+timedelta(hours=i+1):raise ValueError('Ventana horaria inválida')
    files=set()
    for f in d.get('source_files',[]):
        m=f.get('grib_meta',{});step=f.get('step')
        if type(step) is not int or not 1<=step<=24 or step in files or f.get('members_extracted')!=40 or not re.fullmatch(r'[a-f0-9]{64}',f.get('grib_sha256','')) or not f.get('url','').startswith('https://opendata.dwd.de/weather/nwp/icon-eps/grib/'):raise ValueError('Trazabilidad GRIB inválida')
        if any(m.get(k)!=v for k,v in {'numberOfGridUsed':39,'uuidOfHGrid':GRID_UUID,'numberOfDataPoints':GRID_POINTS,'numberOfForecastsInEnsemble':40,'units':'kg m**-2','stepType':'accum','startStep':0,'endStep':step,'dataDate':int(run_id[:8]),'dataTime':int(run_id[8:])*100}.items()):raise ValueError('Metadatos GRIB incompatibles')
        files.add(step)
    for p in provinces:
        if p.get('department_id')!=p['id'][:2] or not finite(p.get('lat')) or not finite(p.get('lon')) or not -19<=p['lat']<=1 or not -83<=p['lon']<=-68:raise ValueError('Coordenadas provinciales inválidas')
        cell=p.get('grid_cell',{})
        if type(cell.get('index')) is not int or not 0<=cell['index']<GRID_POINTS or not finite(cell.get('distance_km')) or not 0<=cell['distance_km']<=25:raise ValueError('Celda provincial inválida')
        if len(p.get('hourly',[]))!=24:raise ValueError('Serie provincial incompleta')
        for i,row in enumerate(p['hourly']):
            if len(row)!=5:raise ValueError('Fila inválida')
            if row[0] is None:
                if row != [None,0,None,None,None]:raise ValueError('Dato faltante convertido a cero')
                continue
            if row[1]!=40 or type(row[2]) is not int or not 0<=row[2]<=40 or not finite(row[0]) or abs(row[0]-row[2]/40*100)>1e-8 or not finite(row[3]) or not finite(row[4]) or not 0<=row[3]<=row[4]<9999:raise ValueError('Probabilidad alterada o miembros incompletos')
            if i+1 not in files or (i>0 and i not in files):raise ValueError('Faltan límites GRIB originales consecutivos')
    return True

def build_forecast_bundle(run_id,intervals,mapping,generated_at=None,source_files=None):
    if not re.fullmatch(r'\d{10}',run_id):raise ValueError('Corrida inválida')
    start=datetime.strptime(run_id,'%Y%m%d%H').replace(tzinfo=timezone.utc)
    if start.hour not in (0,12):raise ValueError('Ciclo no admitido')
    first=intervals[mapping['provinces'][0]['target_key']]
    if len(first)!=24:raise ValueError('Se requieren exactamente 24 horas')
    hours=[{'step':step,'start_utc':utc_iso(start+timedelta(hours=step-1)),'end_utc':utc_iso(start+timedelta(hours=step))} for step in range(1,25)]
    provinces=[]
    for target in mapping['provinces']:
        entries=intervals.get(target['target_key'],[])
        if len(entries)!=24:raise ValueError('Serie provincial incompleta')
        rows=[];reasons={}
        for i,entry in enumerate(entries):
            if entry['step']!=i+1 or entry['start_utc']!=hours[i]['start_utc'] or entry['end_utc']!=hours[i]['end_utc']:raise ValueError('Ventana temporal incompatible')
            if entry['quality']=='VALID' and entry['valid_members']==40 and finite(entry['probability_percent']):
                count=entry['exceeding_members'];probability=entry['probability_percent']
                if not isinstance(count,int) or not 0<=count<=40 or abs(probability-count/40*100)>1e-9:raise ValueError('Probabilidad modificada o conteo inválido')
                rows.append([probability,40,count,entry['mean_ensemble_mm'],entry['max_member_mm']])
            else:
                rows.append([None,0,None,None,None]);reasons[str(i)]=entry.get('reason') or entry['quality']
        provinces.append({'id':target['id'],'name':target['name'],'capital':target['capital'],'department_id':target['department_id'],'lat':target['latitude'],'lon':target['longitude'],'elevation_m':target.get('elevation_m'),'grid_cell':{'index':target['cell_index'],'lat':target['cell_latitude'],'lon':target['cell_longitude'],'distance_km':target['distance_km']},'hourly':rows,'missing_reasons':reasons})
    result={'schema_version':'3.2.0','nature':'FORECAST','source':dict(SOURCE),'event':{'operator':'>=','amount_mm':.1,'period_hours':1,'unit':'mm'},'run':{'run_id':run_id,'run_utc':utc_iso(start)},'generated_at':generated_at or utc_iso(datetime.now(timezone.utc)),'columns':COLUMNS,'hours':hours,'provinces':provinces,'mapping_sha256':hashlib.sha256(json.dumps(mapping,sort_keys=True,separators=(',',':')).encode()).hexdigest(),'source_files':source_files or [],'uncertainty':'Frecuencia bruta de 40 miembros; sin calibración ni validación local de habilidad. Celda nativa ~20 km de referencia de la capital, no resolución provincial ni observación actual.'}
    validate_forecast_bundle(result)
    return result

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--map-only',action='store_true',required=True);p.add_argument('--catalog',type=Path,default=ROOT/'web/public/data/geo/catalog_v32.json');p.add_argument('--coordinate-cache',type=Path,default=ROOT/'.tools/v32-icon');p.add_argument('--output-mapping',type=Path,default=DEFAULT_MAPPING);args=p.parse_args()
    mapping=build_grid_mapping(args.catalog,args.coordinate_cache);args.output_mapping.parent.mkdir(parents=True,exist_ok=True);args.output_mapping.write_text(json.dumps(mapping,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'provinces':len(mapping['provinces']),'grid_uuid':GRID_UUID,'max_distance_km':max(p['distance_km'] for p in mapping['provinces']),'quantization_ambiguous':sum(not p['robust_to_coordinate_quantization'] for p in mapping['provinces']),'output':str(args.output_mapping)},ensure_ascii=True))

if __name__=='__main__':main()
