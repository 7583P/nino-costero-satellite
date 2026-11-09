"""DWD ICON-EPS hourly precipitation probability extractor.

Extracts authentic 1-hour ensemble precipitation probabilities for Tumbes, Piura, and Chiclayo
from DWD ICON-EPS global icosahedral forecasts (CC BY 4.0, zero cost).
Performs sequential bounded downloads, immediate temporary cleanup, and strict scientific validation.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone, timedelta
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time
import bz2

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import psutil
import eccodes as ec

ROOT = Path(__file__).resolve().parents[1]

CITIES = {
    'tumbes': {
        'id': 'tumbes',
        'name': 'Tumbes',
        'latitude': -3.5669,
        'longitude': -80.4515,
        'cell_index': 539249,
        'cell_latitude': -3.5501,
        'cell_longitude': -80.3685,
        'distance_km': 9.40
    },
    'piura': {
        'id': 'piura',
        'name': 'Piura',
        'latitude': -5.1945,
        'longitude': -80.6328,
        'cell_index': 539161,
        'cell_latitude': -5.2420,
        'cell_longitude': -80.6835,
        'distance_km': 7.70
    },
    'chiclayo': {
        'id': 'chiclayo',
        'name': 'Chiclayo',
        'latitude': -6.7714,
        'longitude': -79.8409,
        'cell_index': 537055,
        'cell_latitude': -6.6892,
        'cell_longitude': -79.8431,
        'distance_km': 9.15
    }
}

GRID_METADATA = {
    'grid_type': 'unstructured_grid',
    'number_of_grid_used': 39,
    'number_of_data_points': 1310720,
    'uuid': '49c340e2248d11e58105c12d5e3b00a4',
    'mesh': 'R02B07',
    'nominal_resolution_km': 20.0
}

USER_AGENT = 'NinoCostero/0.2.0-visual (matiasgamesdev.support@gmail.com; DWD-ICON-EPS-Ingestion)'

def format_pet(dt_utc: datetime) -> str:
    pet_dt = dt_utc.astimezone(timezone(timedelta(hours=-5)))
    return pet_dt.isoformat()

def discover_latest_run(cycle_preference=('12', '00')) -> tuple[str, str, list[int]]:
    base = 'https://opendata.dwd.de/weather/nwp/icon-eps/grib/'
    session = requests.Session()
    session.headers.update({'User-Agent': USER_AGENT})
    session.mount('https://', HTTPAdapter(max_retries=Retry(total=2,backoff_factor=.5,status_forcelist=(429,500,502,503,504),allowed_methods=('GET',))))
    candidates = []
    
    for cycle in cycle_preference:
        url = f"{base}{cycle}/tot_prec/"
        try:
            r = session.get(url, timeout=(10, 20))
            if r.status_code == 200:
                files = re.findall(r'href="([^"]+\.bz2)"', r.text)
                runs = sorted(set(re.findall(r'_(\d{10})_', ' '.join(files))))
                if runs:
                    latest_run = runs[-1]
                    steps = sorted([int(m.group(1)) for f in files if (m := re.search(rf'_{latest_run}_(\d{{3}})_', f))])
                    if set(range(1,25)).issubset(steps):
                        candidates.append((cycle, latest_run, steps))
        except Exception:
            continue
    if candidates:
        # Compare both cycles before choosing. A preferred 12Z directory may
        # still contain yesterday's run while today's 00Z is already available.
        return max(candidates, key=lambda c: c[1])
    raise RuntimeError("No suitable complete ICON-EPS run found on DWD opendata")

def download_and_extract_step(run: str, step: int, temp_dir: Path, session: requests.Session) -> tuple[dict[int, dict[str, float]], dict]:
    cycle = run[8:10]
    filename = f"icon-eps_global_icosahedral_single-level_{run}_{step:03d}_tot_prec.grib2.bz2"
    url = f"https://opendata.dwd.de/weather/nwp/icon-eps/grib/{cycle}/tot_prec/{filename}"
    
    bz2_path = temp_dir / filename
    grib_path = temp_dir / f"step_{step:03d}.grib2"
    
    if grib_path.exists() and grib_path.stat().st_size > 1000000:
        dl_elapsed = 0.0
        bytes_downloaded = 0
        bz2_sha256 = 'cached_local'
        uncompressed_bytes = grib_path.stat().st_size
        dec_elapsed = 0.0
    else:
        t_start = time.perf_counter()
        r = session.get(url, stream=True, timeout=(20, 60))
        if r.status_code != 200:
            raise RuntimeError(f"HTTP {r.status_code} downloading {url}")
        
        bytes_downloaded = 0
        sha256_hasher = hashlib.sha256()
        with bz2_path.open('wb') as f_out:
            for chunk in r.iter_content(chunk_size=65536):
                if chunk:
                    f_out.write(chunk)
                    sha256_hasher.update(chunk)
                    bytes_downloaded += len(chunk)
        
        dl_elapsed = time.perf_counter() - t_start
        bz2_sha256 = sha256_hasher.hexdigest()
        
        # Decompress
        t_dec = time.perf_counter()
        uncompressed_bytes = 0
        with bz2.open(bz2_path, 'rb') as f_in, grib_path.open('wb') as f_out:
            while block := f_in.read(1024 * 1024):
                f_out.write(block)
                uncompressed_bytes += len(block)
        dec_elapsed = time.perf_counter() - t_dec
        
        if bz2_path.exists():
            try:
                bz2_path.unlink()
            except Exception:
                pass
    
    # Read with eccodes
    t_read = time.perf_counter()
    members_data: dict[int, dict[str, float]] = {}
    grib_meta = {}
    
    with grib_path.open('rb') as f_grib:
        msg_count = 0
        while (gid := ec.codes_grib_new_from_file(f_grib)) is not None:
            try:
                mem = ec.codes_get(gid, 'perturbationNumber')
                step_type = ec.codes_get(gid, 'stepType')
                start_step = ec.codes_get(gid, 'startStep')
                end_step = ec.codes_get(gid, 'endStep')
                units = ec.codes_get(gid, 'units')
                num_points = ec.codes_get(gid, 'numberOfDataPoints')
                grid_used = ec.codes_get(gid, 'numberOfGridUsed')
                
                if msg_count == 0:
                    grib_meta = {
                        'shortName': ec.codes_get(gid, 'shortName'),
                        'name': ec.codes_get(gid, 'name'),
                        'units': units,
                        'dataDate': ec.codes_get(gid, 'dataDate'),
                        'dataTime': ec.codes_get(gid, 'dataTime'),
                        'validityDate': ec.codes_get(gid, 'validityDate'),
                        'validityTime': ec.codes_get(gid, 'validityTime'),
                        'stepType': step_type,
                        'startStep': start_step,
                        'endStep': end_step,
                        'numberOfForecastsInEnsemble': ec.codes_get(gid, 'numberOfForecastsInEnsemble'),
                        'numberOfDataPoints': num_points,
                        'numberOfGridUsed': grid_used
                    }
                
                # Check grid invariants
                assert grid_used == 39, f"Unexpected grid used: {grid_used}"
                assert num_points == 1310720, f"Unexpected data points: {num_points}"
                assert units == 'kg m**-2', f"Unexpected units: {units}"
                assert step_type == 'accum', f"Unexpected stepType: {step_type}"
                assert end_step == step, f"Step mismatch: {end_step} != {step}"
                
                vals = ec.codes_get_values(gid)
                members_data[mem] = {city_id: float(vals[c['cell_index']]) for city_id, c in CITIES.items()}
                msg_count += 1
            finally:
                ec.codes_release(gid)
    
    read_elapsed = time.perf_counter() - t_read
    
    # Record evidence entry
    evidence = {
        'step': step,
        'url': url,
        'download_bytes': bytes_downloaded,
        'sha256': bz2_sha256,
        'download_seconds': round(dl_elapsed, 4),
        'uncompressed_bytes': uncompressed_bytes,
        'decompress_seconds': round(dec_elapsed, 4),
        'read_seconds': round(read_elapsed, 4),
        'members_extracted': len(members_data),
        'grib_meta': grib_meta
    }
    
    # Remove files to keep disk bounded
    try:
        bz2_path.unlink(missing_ok=True)
        grib_path.unlink(missing_ok=True)
    except Exception:
        pass
        
    return members_data, evidence

def process_icon_hourly(
    run: str | None = None,
    num_steps: int = 24,
    threshold_mm: float = 0.1,
    out_dir: Path | None = None,
    keep_temp: bool = False
) -> dict:
    process = psutil.Process()
    cpu_start = process.cpu_times()
    t0 = time.perf_counter()
    
    session = requests.Session()
    session.headers.update({'User-Agent': USER_AGENT})
    session.mount('https://', HTTPAdapter(max_retries=Retry(total=2,backoff_factor=.5,status_forcelist=(429,500,502,503,504),allowed_methods=('GET',))))
    
    if run is None or run == 'latest':
        cycle, run, available_steps = discover_latest_run()
    else:
        cycle = run[8:10]
        available_steps = list(range(1, num_steps + 1))
        
    run_dt = datetime.strptime(run, '%Y%m%d%H').replace(tzinfo=timezone.utc)
    
    temp_dir = ROOT / 'results/temp_icon_extraction' / run
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    target_out_dir = (out_dir if out_dir is not None else ROOT / 'public/v2/hourly').resolve()
    target_out_dir.mkdir(parents=True, exist_ok=True)
    
    step_evidences = []
    total_download_bytes = 0
    peak_rss = 0
    negative_truncations_detected = 0
    
    # City hourly intervals accumulator
    # city -> list of interval records
    city_intervals: dict[str, list[dict]] = {c: [] for c in CITIES}
    
    # Track previous step accumulation per member per city: member -> city -> float
    prev_accumulations: dict[int, dict[str, float]] = {m: {c: 0.0 for c in CITIES} for m in range(1, 41)}
    previous_step = 0
    
    for step in range(1, num_steps + 1):
        step_start_utc = run_dt + timedelta(hours=step - 1)
        step_end_utc = run_dt + timedelta(hours=step)
        
        try:
            members_data, ev = download_and_extract_step(run, step, temp_dir, session)
            step_evidences.append(ev)
            total_download_bytes += ev['download_bytes']
            
            # Check RSS
            rss = process.memory_info().rss
            if rss > peak_rss:
                peak_rss = rss
                
            # Process members and compute hourly precipitation
            assert 38 <= len(members_data) <= 40, f"Expected 38-40 members, got {len(members_data)}"
            valid_mems = len(members_data)
            if previous_step != step - 1:
                prev_accumulations = members_data
                previous_step = step
                raise ValueError('Falta acumulado del paso anterior: no se interpreta una diferencia multihoraria como una hora')
            
            for city_id, city_info in CITIES.items():
                member_hourly_precip = []
                exceeding_count = 0
                
                for mem in sorted(members_data.keys()):
                    curr_acc = members_data[mem][city_id]
                    prev_acc = prev_accumulations.get(mem, {}).get(city_id, 0.0)
                    delta = curr_acc - prev_acc
                    
                    if delta < -1e-5:
                        negative_truncations_detected += 1
                        
                    hourly_rain = max(0.0, delta)
                    member_hourly_precip.append(hourly_rain)
                    if hourly_rain >= threshold_mm:
                        exceeding_count += 1
                        
                prob_percent = round((exceeding_count / float(valid_mems)) * 100.0, 1)
                
                interval_record = {
                    'step': step,
                    'start_utc': step_start_utc.isoformat().replace('+00:00', 'Z'),
                    'end_utc': step_end_utc.isoformat().replace('+00:00', 'Z'),
                    'start_pet': format_pet(step_start_utc),
                    'end_pet': format_pet(step_end_utc),
                    'probability_percent': prob_percent,
                    'threshold_mm': threshold_mm,
                    'valid_members': valid_mems,
                    'exceeding_members': exceeding_count,
                    'max_member_mm': round(max(member_hourly_precip), 4),
                    'mean_ensemble_mm': round(sum(member_hourly_precip) / float(valid_mems), 4),
                    'run_id': run,
                    'source': 'DWD ICON-EPS',
                    'quality': 'VALID'
                }
                city_intervals[city_id].append(interval_record)
                
            # Update previous accumulations for next step
            for mem, data_c in members_data.items():
                prev_accumulations[mem] = data_c
            previous_step = step
                    
        except Exception as e:
            print(f'DWD step {step}: unavailable ({type(e).__name__}: {e})', flush=True)
            # Handle missing / invalid step gracefully without inventing data
            for city_id in CITIES:
                interval_record = {
                    'step': step,
                    'start_utc': step_start_utc.isoformat().replace('+00:00', 'Z'),
                    'end_utc': step_end_utc.isoformat().replace('+00:00', 'Z'),
                    'start_pet': format_pet(step_start_utc),
                    'end_pet': format_pet(step_end_utc),
                    'probability_percent': None,
                    'threshold_mm': threshold_mm,
                    'valid_members': 0,
                    'exceeding_members': 0,
                    'max_member_mm': None,
                    'mean_ensemble_mm': None,
                    'run_id': run,
                    'source': 'DWD ICON-EPS',
                    'quality': f'MISSING_STEP: {str(e)}'
                }
                city_intervals[city_id].append(interval_record)
                
    elapsed_total = time.perf_counter() - t0
    cpu_end = process.cpu_times()
    cpu_user = cpu_end.user - cpu_start.user
    cpu_system = cpu_end.system - cpu_start.system
    
    generated_at_utc = datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')
    
    # Build and write contract JSON for each city
    generated_files = []
    for city_id, city_info in CITIES.items():
        doc = {
            'schema_version': '1.0.0',
            'city': {
                'id': city_id,
                'name': city_info['name'],
                'latitude': city_info['latitude'],
                'longitude': city_info['longitude'],
                'grid_cell': {
                    'index': city_info['cell_index'],
                    'cell_latitude': city_info['cell_latitude'],
                    'cell_longitude': city_info['cell_longitude'],
                    'distance_km': city_info['distance_km']
                }
            },
            'source': {
                'model': 'ICON-EPS',
                'producer': 'Deutscher Wetterdienst (DWD)',
                'license': 'CC BY 4.0',
                'license_url': 'https://www.dwd.de/EN/service/legal_notice/legal_notice_node.html',
                'commercial_use_allowed': True,
                'cost': 'S/ 0',
                'total_ensemble_members': 40,
                'grid_type': GRID_METADATA['grid_type'],
                'grid_mesh': GRID_METADATA['mesh'],
                'nominal_resolution_km': GRID_METADATA['nominal_resolution_km'],
                'number_of_grid_used': GRID_METADATA['number_of_grid_used']
            },
            'run': {
                'run_id': run,
                'run_utc': run_dt.isoformat().replace('+00:00', 'Z')
            },
            'parameter': {
                'name': 'precipitation_probability',
                'threshold_mm': threshold_mm,
                'accumulation_period_hours': 1,
                'unit': '%',
                'label': 'Probabilidad de precipitación ≥0,1 mm durante la hora indicada — DWD ICON-EPS'
            },
            'quality': {
                'total_hours': num_steps,
                'complete_hours': sum(1 for iv in city_intervals[city_id] if iv['probability_percent'] is not None),
                'valid_members_per_step': 40,
                'negative_accumulations_detected': negative_truncations_detected > 0,
                'negative_truncations_count': negative_truncations_detected
            },
            'generated_at': generated_at_utc,
            'intervals': city_intervals[city_id]
        }
        
        out_file = target_out_dir / f"{city_id}.json"
        out_file.write_text(json.dumps(doc, indent=2, ensure_ascii=False), encoding='utf-8')
        generated_files.append({
            'city': city_id,
            'path': out_file.resolve().relative_to(ROOT.resolve()).as_posix(),
            'bytes': out_file.stat().st_size,
            'sha256': hashlib.sha256(out_file.read_bytes()).hexdigest(),
            'hours_count': len(doc['intervals']),
            'nonnull_hours': doc['quality']['complete_hours']
        })
        
    summary = {
        'run_id': run,
        'run_utc': run_dt.isoformat().replace('+00:00', 'Z'),
        'extracted_at': generated_at_utc,
        'num_steps_requested': num_steps,
        'total_download_bytes': total_download_bytes,
        'total_download_mb': round(total_download_bytes / (1024 * 1024), 2),
        'elapsed_seconds': round(elapsed_total, 2),
        'cpu_user_seconds': round(cpu_user, 2),
        'cpu_system_seconds': round(cpu_system, 2),
        'peak_rss_mb': round(peak_rss / (1024 * 1024), 2),
        'negative_truncations_detected': negative_truncations_detected,
        'output_files': generated_files,
        'step_evidences': step_evidences
    }
    
    # Save extraction summary
    summary_path = ROOT / 'results/processed' / f'icon_hourly_extraction_{run}.json'
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding='utf-8')
    
    # Also save raw execution evidence
    raw_ev_dir = ROOT / 'results/raw/icon_eps' / run
    raw_ev_dir.mkdir(parents=True, exist_ok=True)
    (raw_ev_dir / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    
    # Clean up temp dir if requested
    if not keep_temp:
        try:
            for item in temp_dir.glob('*'):
                item.unlink(missing_ok=True)
            temp_dir.rmdir()
        except Exception:
            pass
            
    return summary

def main():
    parser = argparse.ArgumentParser(description="Extract DWD ICON-EPS hourly precipitation probabilities.")
    parser.add_argument('--run', type=str, default=None, help="Run ID (e.g. 2026110212 or 2026110200) or 'latest'")
    parser.add_argument('--steps', type=int, default=24, help="Number of hourly steps (default: 24)")
    parser.add_argument('--threshold', type=float, default=0.1, help="Precipitation threshold in mm (default: 0.1)")
    parser.add_argument('--out-dir', type=str, default=None, help="Output directory")
    parser.add_argument('--keep-temp', action='store_true', help="Keep temporary decompressed GRIB files")
    args = parser.parse_args()
    
    out_dir = Path(args.out_dir) if args.out_dir else None
    print(f"Starting DWD ICON-EPS extraction (run={args.run or 'latest'}, steps={args.steps})...")
    res = process_icon_hourly(
        run=args.run,
        num_steps=args.steps,
        threshold_mm=args.threshold,
        out_dir=out_dir,
        keep_temp=args.keep_temp
    )
    print(json.dumps({
        'run_id': res['run_id'],
        'steps': res['num_steps_requested'],
        'download_mb': res['total_download_mb'],
        'elapsed_s': res['elapsed_seconds'],
        'peak_rss_mb': res['peak_rss_mb'],
        'files': [f['path'] for f in res['output_files']]
    }, indent=2))

if __name__ == '__main__':
    main()

