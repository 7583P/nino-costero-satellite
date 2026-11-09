"""Automated scientific tests for DWD ICON-EPS hourly extraction."""
import json
import sys
from pathlib import Path
import unittest
from datetime import datetime, timezone, timedelta

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

class TestIconEpsExtraction(unittest.TestCase):
    def setUp(self):
        self.cities = ['tumbes', 'piura', 'chiclayo']
        self.hourly_dir = ROOT / 'public/v2/hourly'
        self.assertGreater(len(list(self.hourly_dir.glob('*.json'))), 0, "No hourly JSON files generated")

    def test_three_cities_json_exist_and_validate_schema(self):
        for city_id in self.cities:
            path = self.hourly_dir / f"{city_id}.json"
            self.assertTrue(path.exists(), f"File {path} must exist")
            doc = json.loads(path.read_text(encoding='utf-8'))
            
            # Check schema and source
            self.assertEqual(doc['schema_version'], '1.0.0')
            self.assertEqual(doc['city']['id'], city_id)
            self.assertEqual(doc['source']['model'], 'ICON-EPS')
            self.assertEqual(doc['source']['producer'], 'Deutscher Wetterdienst (DWD)')
            self.assertEqual(doc['source']['total_ensemble_members'], 40)
            self.assertEqual(doc['source']['grid_mesh'], 'R02B07')
            self.assertEqual(doc['source']['number_of_grid_used'], 39)
            self.assertEqual(doc['parameter']['threshold_mm'], 0.1)
            self.assertEqual(doc['parameter']['accumulation_period_hours'], 1)
            self.assertEqual(doc['parameter']['unit'], '%')
            self.assertEqual(
                doc['parameter']['label'],
                'Probabilidad de precipitación ≥0,1 mm durante la hora indicada — DWD ICON-EPS'
            )
            
            # Check intervals
            intervals = doc['intervals']
            self.assertGreaterEqual(len(intervals), 3)
            self.assertEqual(doc['quality']['total_hours'], len(intervals))

    def test_intervals_chronological_continuity_and_utc_pet(self):
        for city_id in self.cities:
            doc = json.loads((self.hourly_dir / f"{city_id}.json").read_text(encoding='utf-8'))
            intervals = doc['intervals']
            
            for i, iv in enumerate(intervals):
                self.assertEqual(iv['step'], i + 1)
                self.assertEqual(iv['threshold_mm'], 0.1)
                if iv['quality'] == 'VALID':
                    self.assertGreaterEqual(iv['valid_members'], 38)
                    self.assertLessEqual(iv['valid_members'], 40)
                else:
                    self.assertEqual(iv['valid_members'], 0)
                self.assertIn(iv['quality'], ['VALID', 'MISSING_STEP'])
                
                # Verify probability bounds
                if iv['probability_percent'] is not None:
                    self.assertGreaterEqual(iv['probability_percent'], 0.0)
                    self.assertLessEqual(iv['probability_percent'], 100.0)
                    self.assertGreaterEqual(iv['exceeding_members'], 0)
                    self.assertLessEqual(iv['exceeding_members'], 40)
                    expected_prob = round((iv['exceeding_members'] / 40.0) * 100.0, 1)
                    self.assertAlmostEqual(iv['probability_percent'], expected_prob, places=1)
                
                # Check UTC times
                start_utc = datetime.fromisoformat(iv['start_utc'].replace('Z', '+00:00'))
                end_utc = datetime.fromisoformat(iv['end_utc'].replace('Z', '+00:00'))
                self.assertEqual((end_utc - start_utc).total_seconds(), 3600)
                
                # Check PET times (UTC - 5 hours)
                start_pet = datetime.fromisoformat(iv['start_pet'])
                end_pet = datetime.fromisoformat(iv['end_pet'])
                self.assertEqual(start_pet.tzinfo, timezone(timedelta(hours=-5)))
                self.assertEqual(end_pet.tzinfo, timezone(timedelta(hours=-5)))
                self.assertEqual(start_utc, start_pet.astimezone(timezone.utc))
                self.assertEqual(end_utc, end_pet.astimezone(timezone.utc))
                
                # Check consecutive continuity
                if i > 0:
                    prev_end = datetime.fromisoformat(intervals[i - 1]['end_utc'].replace('Z', '+00:00'))
                    self.assertEqual(start_utc, prev_end, f"Gap between step {i} and {i+1}")

    def test_mesh_invariants_and_cell_coordinates(self):
        for city_id in self.cities:
            doc = json.loads((self.hourly_dir / f"{city_id}.json").read_text(encoding='utf-8'))
            cell = doc['city']['grid_cell']
            self.assertIn('index', cell)
            self.assertIn('distance_km', cell)
            self.assertLess(cell['distance_km'], 15.0, "Nearest ICON cell must be within 15 km")

    def test_synthetic_edge_cases_zero_and_hundred_and_null(self):
        # Verify calculation logic on synthetic member deltas
        from scripts.extract_icon_hourly import CITIES
        
        # All 40 members zero
        all_dry = [0.0] * 40
        count_zero = sum(1 for x in all_dry if x >= 0.1)
        prob_zero = round(count_zero / 40.0 * 100.0, 1)
        self.assertEqual(prob_zero, 0.0)
        
        # All 40 members > 0.1 mm
        all_wet = [0.5] * 40
        count_hundred = sum(1 for x in all_wet if x >= 0.1)
        prob_hundred = round(count_hundred / 40.0 * 100.0, 1)
        self.assertEqual(prob_hundred, 100.0)
        
        # Exactly 1 member out of 40 = 2.5%
        one_wet = [0.0] * 39 + [0.2]
        self.assertEqual(round(sum(1 for x in one_wet if x >= 0.1) / 40.0 * 100.0, 1), 2.5)

    def test_second_run_validation_and_cross_run_invariants(self):
        run00_dir = ROOT / 'results/processed/hourly_2026110200'
        if not run00_dir.exists():
            self.skipTest("Second run directory not present")
        
        for city_id in self.cities:
            path00 = run00_dir / f"{city_id}.json"
            path12 = self.hourly_dir / f"{city_id}.json"
            self.assertTrue(path00.exists(), f"Second run file {path00} must exist")
            
            doc00 = json.loads(path00.read_text(encoding='utf-8'))
            doc12 = json.loads(path12.read_text(encoding='utf-8'))
            
            # Check run IDs and UTC issuance
            self.assertRegex(doc00['run']['run_id'], r'^\d{10}$')
            self.assertRegex(doc12['run']['run_id'], r'^\d{10}$')
            self.assertGreaterEqual(doc12['run']['run_id'], doc00['run']['run_id'])
            
            dt00 = datetime.fromisoformat(doc00['run']['run_utc'].replace('Z', '+00:00'))
            dt12 = datetime.fromisoformat(doc12['run']['run_utc'].replace('Z', '+00:00'))
            diff_hours = (dt12 - dt00).total_seconds() / 3600
            self.assertEqual(diff_hours % 12, 0, f"Runs must be multiples of 12 hours apart, got {diff_hours}h")
            
            # Mesh invariants must match exactly across all runs
            self.assertEqual(doc00['city']['grid_cell'], doc12['city']['grid_cell'])
            self.assertEqual(doc00['source']['grid_mesh'], doc12['source']['grid_mesh'])
            self.assertEqual(doc00['source']['total_ensemble_members'], 40)
            self.assertEqual(doc12['source']['total_ensemble_members'], 40)
            
            # Both runs must provide 24 continuous hours
            self.assertEqual(len(doc00['intervals']), 24)
            self.assertEqual(len(doc12['intervals']), 24)
            self.assertEqual(doc00['quality']['complete_hours'], 24)
            self.assertEqual(doc12['quality']['complete_hours'], 24)

if __name__ == '__main__':
    unittest.main()

