import io
import json
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

try:
    from sat import analysis as A, encode as E, grid as G, sources as S
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from sat import analysis as A, encode as E, grid as G, sources as S


class SatelliteScience(unittest.TestCase):
    def test_canonical_decimal_rain_scale(self):
        for value, level in [(0, 0), (.04, 0), (.05, 1), (.9, 1), (.95, 2), (2.94, 2), (2.95, 3), (4.95, 4),
                             (7.95, 5), (19.95, 6), (24.95, 7), (49.95, 8), (69.95, 9), (89.95, 10), (120, 10)]:
            with self.subTest(value=value):
                self.assertEqual(A.rain_level(value), level)
        for v in [None, np.nan, -1]:
            self.assertIsNone(A.rain_level(v))

    def test_storm_requires_electrical_and_rain_evidence(self):
        self.assertIsNone(A.storm_level(120, None, True))
        self.assertIsNone(A.storm_level(0, 0, False))
        self.assertEqual(A.storm_level(None, 0, True), 0)
        self.assertIsNone(A.storm_level(None, 1, True))
        for v, l in [(0, 1), (5, 2), (10, 3), (15, 4), (25, 5), (50, 6), (70, 7), (90, 8), (150, 8)]:
            self.assertEqual(A.storm_level(v, 1, True), l)

    def test_accumulation_no_missing_or_bad_pixel(self):
        end = datetime(2026, 11, 4, 12, tzinfo=timezone.utc)
        frames = {end - timedelta(minutes=10 * i): {'raw': np.array([[60, 120]]), 'valid': np.array([[True, True]])}
                  for i in range(144)}
        for h in A.WINDOWS_H:
            acc, valid, info = A.accumulate(frames, end, h)
            np.testing.assert_allclose(acc, [[6 * h, 12 * h]])
            self.assertTrue(info['complete'])
        frames[end]['valid'][0, 1] = False
        acc, ok, _ = A.accumulate(frames, end, 24)
        self.assertTrue(np.isnan(acc[0, 1]))
        self.assertFalse(ok[0, 1])
        del frames[end - timedelta(minutes=70)]
        self.assertIsNone(A.accumulate(frames, end, 24)[0])

    def test_wrong_scan_timestamp_rejects_total(self):
        t = datetime(2026, 11, 4, 12, tzinfo=timezone.utc)
        frames = {t - timedelta(minutes=10 * i): {'raw': np.zeros((1, 1)), 'valid': np.ones((1, 1), bool)}
                  for i in range(6)}
        frames[t]['meta'] = {'time_coverage_start': '2026-11-04T11:00Z', 'time_coverage_end': '2026-11-04T11:10Z'}
        self.assertFalse(A.accumulate(frames, t, 1)[2]['complete'])

    def test_coverage_union_does_not_double_count(self):
        t = datetime(2026, 11, 4, tzinfo=timezone.utc)
        covered = [(t, t + timedelta(minutes=30))] * 2
        self.assertEqual(A.coverage_fraction(covered, t, t + timedelta(hours=1)), .5)

    def test_cloud_cb_requires_independent_evidence(self):
        a = np.ones((8, 8))
        near = np.zeros((8, 8), bool)
        inp = [a * 3, a * 12000, a * 220, a * 4, a * 30, a * 220, None, near]
        cls, _ = A.classify(*inp)
        self.assertTrue((cls == 7).all())
        inp[-1] = ~near
        cls, _ = A.classify(*inp)
        self.assertTrue((cls == 8).all())
        inp[-1] = near
        inp[-2] = a * 226
        cls, _ = A.classify(*inp)
        self.assertTrue((cls == 8).all())

    def test_coordinate_native_roundtrip(self):
        lat, lon = G.native_latlon()
        for r, c in [(10, 10), (100, 100), (180, 140)]:
            rr, cc = G.latlon_to_native(float(lat[r, c]), float(lon[r, c]))
            self.assertEqual((int(rr), int(cc)), (r, c))

    def test_quality_rules(self):
        dqf = np.arange(8)
        np.testing.assert_array_equal(S._usable('cmi', dqf), dqf == 0)
        np.testing.assert_array_equal(S._usable('cloud_top', dqf), np.isin(dqf, (0, 1, 4)))
        np.testing.assert_array_equal(S._usable('cod', dqf), ((dqf & 1) == 0) & ((dqf & 4) == 0))


if __name__ == '__main__':
    unittest.main()
