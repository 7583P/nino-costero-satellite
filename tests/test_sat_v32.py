"""V3.2 science checks. Synthetic numerical inputs, not weather skill scores."""
import io
import unittest
from datetime import datetime, timezone
from unittest.mock import patch

import h5py
import numpy as np
from PIL import Image

from sat import analysis as A, encode as E, grid as G, sources as S
from sat.pipeline import accumulation_anchor, make_tar_bundle, partition_tar_assets


class SatelliteV32(unittest.TestCase):
    def test_accumulation_anchor_matches_completed_hour(self):
        for minute, hour, expected_minute in [(0, 22, 50), (10, 22, 50), (40, 22, 50), (50, 23, 50)]:
            latest = datetime(2026, 11, 15, 23, minute, tzinfo=timezone.utc)
            anchor = accumulation_anchor(latest)
            self.assertEqual((anchor.hour, anchor.minute), (hour, expected_minute))

    def test_49_province_extent_and_navigation(self):
        # Extrema from the downloaded INEI provincial WFS, not guessed city centres.
        for lat, lon in [(-3.38177335, -80), (-15.44337498, -75), (-8, -81.32823049), (-14, -74.64624123)]:
            r, c = G.latlon_to_native(lat, lon)
            self.assertGreaterEqual(int(r), 0)
            self.assertGreaterEqual(int(c), 0)
        self.assertEqual((G.DNX, G.DNY), (360, 640))
        self.assertAlmostEqual(G.DSTEP, .02)
        self.assertEqual(G.GRID_ID, 'peru49-2km-v32')

    def test_cmi_calibration_unsigned_fill_and_quality(self):
        with h5py.File(io.BytesIO(), 'w') as h:
            raw = h.create_dataset('cmi', data=np.array([[100, 200, -1, 300]], np.int16))
            raw.attrs.update({'scale_factor': .001, 'add_offset': .05, '_FillValue': -1, '_Unsigned': 'true'})
            dqf = h.create_dataset('dqf', data=np.array([[0, 1, 0, 0]], np.uint8))
            out = S.decode_cmi(raw, dqf, np.s_[:, :])
            self.assertAlmostEqual(float(out[0, 0]), .15, places=6)
            self.assertAlmostEqual(float(out[0, 3]), .35, places=6)
            self.assertTrue(np.isnan(out[0, 1]))
            self.assertTrue(np.isnan(out[0, 2]))
            np.testing.assert_array_equal(raw[:], [[100, 200, -1, 300]])

    def test_no_visible_data_at_night(self):
        noon = datetime(2026, 11, 15, 17, tzinfo=timezone.utc)
        night = datetime(2026, 11, 15, 5, tzinfo=timezone.utc)
        self.assertGreater(float(A.solar_cosine(-12.05, -77.04, noon)), .9)
        self.assertLess(float(A.solar_cosine(-12.05, -77.04, night)), -.7)

    def test_reflectance_display_preserves_source_and_day_mask(self):
        channels = {'C01': np.array([[.2, 1.2]]), 'C02': np.array([[.4, .8]]), 'C03': np.array([[.6, .9]])}
        original = {k: v.copy() for k, v in channels.items()}
        rows, cols = np.array([[0, 0]]), np.array([[0, 1]])
        with patch.object(E, 'idx', return_value=(rows, cols)):
            px = np.asarray(Image.open(io.BytesIO(E.visible_png(channels, np.array([[True, False]])))))
        np.testing.assert_array_equal(px[0, 0], [102, 153, 51, 255])
        np.testing.assert_array_equal(px[0, 1], [0, 0, 0, 0])
        for k in channels:
            np.testing.assert_array_equal(channels[k], original[k])

    def test_invalid_reflectance_stays_missing(self):
        channels = {k: np.array([[np.nan]]) for k in ('C01', 'C02', 'C03')}
        with patch.object(E, 'idx', return_value=(np.array([[0]]), np.array([[0]]))):
            px = np.asarray(Image.open(io.BytesIO(E.visible_png(channels, np.array([[True]])))))
        self.assertEqual(int(px[0, 0, 3]), 0)

    def test_complex_rasters_split_without_altering_values(self):
        # Six large legitimate-sized PNG payloads no longer cause an oversize tar.
        files = {f'fixture{i}.png': bytes([i]) * 699000 for i in range(7)}
        parts = partition_tar_assets(files)
        self.assertEqual(len(parts), 2)
        self.assertEqual({k: v for p in parts for k, v in p.items()}, files)
        for p in parts:
            archive, _ = make_tar_bundle(p)
            self.assertLessEqual(len(archive), 4 * 1024 * 1024)
        with self.assertRaises(ValueError):
            partition_tar_assets({'too-large.png': bytes(700001)})


if __name__ == '__main__':
    unittest.main()
