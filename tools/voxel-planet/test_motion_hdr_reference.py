"""Checks that real scene motion is not mistakenly scored as reconstruction noise."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from linear_hdr_reference import write_capture
from motion_hdr_reference import compare_motion


class MotionReferenceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.references = {}
        self.candidates = {}
        for frame, radiance in [(20, 1.0), (21, 4.0)]:
            meta = {
                "width": 2, "height": 1, "kind": "reference", "stage": "pre_aa_linear",
                "unresolved_pixels": 0, "strata_side": 2, "samples": 4,
                "context": {
                    "camera": {"origin_cell": [frame, 63710000, 0], "fraction": [0, 0, 0],
                               "forward": [0, 0, -1], "right": [1, 0, 0, 1],
                               "up": [0, 1, 0, 1], "near": 0.1, "far": 30000000},
                    "scene": {"identity": "fixture", "generator_revision": 2},
                    "lighting": {"sun": 3, "ambient": 0.2, "planet_sky": True,
                                 "raytraced_sun": True, "voxel_shadows": False},
                    "render_size": [2, 1]},
            }
            self.references[frame] = write_capture(self.root / f"ref-{frame}.json", meta, np.full((1, 2, 3), radiance))
            meta = copy.deepcopy(meta)
            meta.update(kind="candidate", stage="post_aa_linear", frame=frame)
            self.candidates[frame] = write_capture(self.root / f"candidate-{frame}.json", meta, np.full((1, 2, 3), radiance + 0.5))
        self.regions = {"all": [0, 0, 2, 1]}

    def tearDown(self):
        self.tmp.cleanup()

    def test_scene_change_is_subtracted_before_temporal_error(self):
        result = compare_motion(self.references, self.candidates, self.regions)
        self.assertEqual(result["regions"]["all"]["median_frame_rmse"], 0.5)
        self.assertEqual(result["regions"]["all"]["residual_change_rms"], 0)

    def test_flickering_error_remains_after_scene_change(self):
        path = self.candidates[21]
        meta = json.loads(path.read_text())
        write_capture(path, meta, np.full((1, 2, 3), 3.5))
        result = compare_motion(self.references, self.candidates, self.regions)
        self.assertEqual(result["regions"]["all"]["median_frame_rmse"], 0.5)
        self.assertEqual(result["regions"]["all"]["residual_change_rms"], 1)

    def test_missing_and_nonconsecutive_frames_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "match exactly"):
            compare_motion(self.references, {20: self.candidates[20]}, self.regions)
        with self.assertRaisesRegex(ValueError, "consecutive"):
            compare_motion({20: self.references[20], 22: self.references[21]}, self.candidates, self.regions)

    def test_wrong_pose_and_unresolved_rays_are_rejected(self):
        path = self.candidates[21]
        original = json.loads(path.read_text())
        for kind in ["pose", "unresolved"]:
            meta = copy.deepcopy(original)
            if kind == "pose":
                meta["context"]["camera"]["origin_cell"][0] += 1
            else:
                meta["unresolved_pixels"] = 1
            path.write_text(json.dumps(meta))
            with self.assertRaises(ValueError):
                compare_motion(self.references, self.candidates, self.regions)

    def test_region_must_stay_inside_crop(self):
        with self.assertRaisesRegex(ValueError, "outside"):
            compare_motion(self.references, self.candidates, {"bad": [1, 0, 2, 1]})


if __name__ == "__main__":
    unittest.main()
