"""Reject damaged measurement files before drawing conclusions about TSR."""
from pathlib import Path
import json
import tempfile
import unittest

import numpy as np

from analyze_tsr_diagnostics import read_diagnostics, analyze, FIELDS
from linear_hdr_reference import write_capture


class DiagnosticRecordTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "records.tsr.f32x8"
        self.values = np.array([[[1, 32, .05, 0, .001, 1.5, .2, .3],
                                 [0, 0, 1, 32, 0, 0, 0, 0]]], dtype="<f4")
        self.values.tofile(self.path)

    def tearDown(self):
        self.tmp.cleanup()

    def test_records_preserve_field_and_pixel_order(self):
        np.testing.assert_array_equal(read_diagnostics(self.path, 2, 1), self.values)

    def test_partial_float_truncated_and_extra_records_are_rejected(self):
        original = self.path.read_bytes()
        for payload in (original[:-1], original[:-32], original + b"\0" * 32):
            self.path.write_bytes(payload)
            with self.assertRaisesRegex(ValueError, "size"):
                read_diagnostics(self.path, 2, 1)

    def test_nonfinite_invalid_flags_counts_blends_and_distances_are_rejected(self):
        for field, value in [(0, np.nan), (7, np.inf), (3, -1), (3, .5),
                             (3, 16), (3, 128), (1, -1), (1, 33),
                             (2, 0), (2, 1.1), (4, -.01), (5, -1)]:
            with self.subTest(field=field, value=value):
                values = self.values.copy()
                values[0, 0, field] = value
                values.tofile(self.path)
                with self.assertRaises(ValueError):
                    read_diagnostics(self.path, 2, 1)

    def test_bad_dimensions_and_frames_are_rejected(self):
        for size in [(0, 1), (2.0, 1), (True, 1)]:
            with self.assertRaises(ValueError):
                read_diagnostics(self.path, *size)
        for frames in ([], [1], [1, 3], [True, 2], [1., 2.]):
            with self.assertRaises(ValueError):
                analyze({"frames": frames, "regions": {"all": [0, 0, 2, 1]}},
                        self.path.parent, self.path.parent)

    def test_descriptors_bind_records_to_exact_capture(self):
        root = self.path.parent
        context = {"camera":dict(origin_cell=[0,0,0],fraction=[0,0,0],forward=[0,0,-1],
            right=[1,0,0],up=[0,1,0],near=.1,far=100),
            "scene":{"identity":"fixture","generator_revision":2},
            "lighting":dict(sun=3,ambient=.2,planet_sky=True,raytraced_sun=True,voxel_shadows=False),
            "render_size":[2,1]}
        for frame in [20, 21]:
            stem = f"candidate-{frame:04d}"
            meta = dict(width=2,height=1,context=context,unresolved_pixels=0,frame=frame,
                        kind="candidate",stage="post_aa_linear",producer_sha256="fixture")
            write_capture(root/f"{stem}.json",meta,np.zeros((1,2,3)))
            self.values.tofile(root/f"{stem}.tsr.f32x8")
            descriptor = dict(schema="helio.tsr_diagnostics.v1",encoding="f32x8_le",fields=FIELDS,
                              source_hdr=f"{stem}.json",data=f"{stem}.tsr.f32x8",**{
                                  k:meta[k] for k in ("width","height","context","frame","producer_sha256")})
            (root/f"{stem}.tsr.json").write_text(json.dumps(descriptor))
        protocol = {"frames":[20,21],"regions":{"all":[0,0,2,1]}}
        result = analyze(protocol, root, root)
        self.assertTrue(result["instrumentation_gate"])
        self.assertEqual(result["regions"]["all"]["valid_history_fraction"], .5)
        path = root/"candidate-0021.tsr.json"
        original = json.loads(path.read_text())
        for key, value in [("frame",20),("encoding","f64x8"),("fields",list(reversed(FIELDS))),
                           ("width",1),("producer_sha256","wrong")]:
            path.write_text(json.dumps(dict(original, **{key:value})))
            with self.subTest(key=key), self.assertRaises(ValueError):
                analyze(protocol,root,root)
        path.unlink()
        self.assertTrue(analyze(protocol,root,root)["comparison"][1]["legacy_layout_without_descriptor"])


if __name__ == "__main__":
    unittest.main()
