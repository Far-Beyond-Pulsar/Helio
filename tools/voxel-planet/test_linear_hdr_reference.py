"""CPU-only validation of capture parsing, sample coverage and reference arithmetic."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from linear_hdr_reference import accumulate, compare, read_capture, stratified_jitter, write_capture


class LinearReferenceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.meta = {
            "width": 2, "height": 1, "kind": "sample", "stage": "pre_aa_linear",
            "unresolved_pixels": 0,
            "context": {
                "camera": {"origin_cell": [0, 63710000, 0], "fraction": [0, 0, 0],
                           "forward": [0, 0, -1], "right": [1, 0, 0, 1],
                           "up": [0, 1, 0, 1], "near": 0.1, "far": 30000000},
                "scene": {"identity": "canonical-fixture-1", "generator_revision": 2},
                "lighting": {"sun": 3, "ambient": 0.2, "planet_sky": True,
                             "raytraced_sun": True, "voxel_shadows": False},
                "render_size": [2, 1]},
        }

    def tearDown(self):
        self.tmp.cleanup()

    def sample(self, index, value, **changes):
        meta = copy.deepcopy(self.meta)
        meta.update(changes)
        meta["sample"] = {"index": index, "strata_side": 2, "jitter": stratified_jitter(index, 2)}
        return write_capture(self.root / f"sample-{index:04d}.json", meta, np.full((1, 2, 3), value))

    def reference(self):
        paths = [self.sample(i, 1 + i * 2) for i in range(4)]
        return accumulate(paths, self.root / "reference.json")

    def test_hdr_is_averaged_in_linear_space_without_clipping(self):
        meta, image = read_capture(self.reference())
        np.testing.assert_array_equal(image, np.full((1, 2, 3), 4.0))
        self.assertEqual(meta["samples"], 4)
        self.assertAlmostEqual(meta["sample_variance_mean"], 20 / 3)
        self.assertEqual(len(meta["sources"]), 4)

    def test_incomplete_and_duplicate_sample_sets_are_rejected(self):
        a = self.sample(0, 1)
        with self.assertRaisesRegex(ValueError, "complete"):
            accumulate([a], self.root / "bad.json")
        with self.assertRaisesRegex(ValueError, "duplicate"):
            accumulate([a, a], self.root / "bad.json")

    def test_wrong_jitter_is_rejected(self):
        path = self.sample(0, 1)
        meta = json.loads(path.read_text())
        meta["sample"]["jitter"] = [0, 0]
        path.write_text(json.dumps(meta))
        with self.assertRaisesRegex(ValueError, "Jitter"):
            accumulate([path], self.root / "bad.json")

    def test_camera_and_equal_edit_count_world_changes_are_rejected(self):
        for section, key, replacement in [("camera", "fraction", [0, 0.001, 0]),
                                           ("scene", "identity", "different-edit-content"),
                                           ("lighting", "sun", 4)]:
            a, b = self.sample(0, 1), self.sample(1, 3)
            meta = json.loads(b.read_text())
            meta["context"][section][key] = replacement
            b.write_text(json.dumps(meta))
            with self.assertRaisesRegex(ValueError, "Mismatched context"):
                accumulate([a, b], self.root / "bad.json")

    def test_display_stage_and_unavailable_diagnostics_are_rejected(self):
        for changes in [{"stage": "display_srgb"}, {"unresolved_pixels": None},
                        {"unresolved_pixels": 1}, {"unresolved_pixels": False}]:
            path = self.sample(0, 1, **changes)
            with self.assertRaises(ValueError):
                read_capture(path)

    def test_truncated_and_nonfinite_payload_are_rejected(self):
        path = self.sample(0, 1)
        raw = path.with_suffix(".rgb32f")
        raw.write_bytes(raw.read_bytes()[:-1])
        with self.assertRaisesRegex(ValueError, "byte count"):
            read_capture(path)
        path = self.sample(0, 1)
        np.full((1, 2, 3), np.nan, dtype="<f4").tofile(raw)
        with self.assertRaisesRegex(ValueError, "Nonfinite"):
            read_capture(path)

    def test_comparison_metrics_have_known_linear_units(self):
        reference = self.reference()
        candidate_meta = dict(self.meta, kind="candidate", stage="post_aa_linear")
        a = write_capture(self.root / "candidate-a.json", candidate_meta, np.full((1, 2, 3), 3.0))
        b = write_capture(self.root / "candidate-b.json", candidate_meta, np.full((1, 2, 3), 5.0))
        metrics = compare(reference, [a, b])
        self.assertEqual(metrics["comparisons"][0]["radiance_rmse"], 1)
        self.assertEqual(metrics["comparisons"][0]["relative_l2"], 0.25)
        self.assertEqual(metrics["comparisons"][0]["signed_rgb_bias"], [-1, -1, -1])
        self.assertEqual(metrics["fixed_context_sequence"]["temporal_rms"], 1)
        self.assertEqual(metrics["fixed_context_sequence"]["ordered_sample_change_rms"], 2)

    def test_non_native_resolution_is_rejected(self):
        path = self.sample(0, 1)
        meta = json.loads(path.read_text())
        meta["context"]["render_size"] = [1, 1]
        path.write_text(json.dumps(meta))
        with self.assertRaisesRegex(ValueError, "native"):
            read_capture(path)

    def test_valid_crop_preserves_full_raster_and_checks_bounds(self):
        path = self.sample(0, 1)
        meta = json.loads(path.read_text())
        meta["context"]["render_size"] = [1280, 720]
        meta["context"]["crop"] = [500, 600, 2, 1]
        path.write_text(json.dumps(meta))
        result, values = read_capture(path)
        self.assertEqual(result["context"]["render_size"], [1280, 720])
        self.assertEqual(values.shape, (1, 2, 3))
        for crop in [[1279, 600, 2, 1], [500, 720, 2, 1], [-1, 0, 2, 1],
                     [500, 600, 1, 1], [500, 600, 2.0, 1]]:
            meta["context"]["crop"] = crop
            path.write_text(json.dumps(meta))
            with self.assertRaisesRegex(ValueError, "crop"):
                read_capture(path)


if __name__ == "__main__":
    unittest.main()
