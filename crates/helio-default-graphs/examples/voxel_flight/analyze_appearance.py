"""Compare an experimental filter against an unfiltered spatial reference.

python -B analyze_appearance.py CONTROL FILTERED HIGH_SAMPLE_REFERENCE > report.json

Requires schema 2 captures (graphics-stream capture), with identical poses and
control/filter sampling grids. The high-sample mean is an estimate, not an exact
oracle. Two-pose deltas include real parallax, not a temporal flicker measurement.
Uses only the Python standard library. Raw outputs belong outside Git.
"""
import argparse
import csv
import json
import math
from pathlib import Path
import struct

from compare_cache import compare


def rows(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def mse(values):
    values = list(values)
    assert all(math.isfinite(x) for x in values)
    return sum(x*x for x in values) / len(values) if values else None


def rmse(a, b, mask):
    value = mse(a[i][c] - b[i][c] for i in mask for c in range(3))
    return math.sqrt(value) if value is not None else None


def analyze(control, filtered, reference):
    for directory, expected_filter in [(control, False), (filtered, True), (reference, False)]:
        assert (directory / "README.txt").is_file(), "incomplete run"
        metadata = json.loads((directory / "capture.json").read_text())
        assert metadata["schema"] == 2 and metadata["lighting_stream"] == "graphics-current-frame"
        assert metadata["appearance_filter"] == expected_filter
    checks = compare(control, filtered)
    for check in checks:
        assert not check["invalid_primary"], check
        assert all(v == 0 for k, v in check.items()
                   if k.endswith("_differences") and k != "lighting_differences"), check
    poses = [{r["name"]: r for r in rows(d / "poses.csv")} for d in (control, filtered, reference)]
    assert poses[0].keys() == poses[1].keys() == poses[2].keys()
    result = []
    images = {}
    for name, pose in poses[2].items():
        for table in poses[:2]:
            assert {k:v for k,v in table[name].items() if k != "samples_per_pixel"} == {
                k:v for k,v in pose.items() if k != "samples_per_pixel"}
            assert int(table[name]["samples_per_pixel"]) < int(pose["samples_per_pixel"])
        a, b = [list(struct.iter_unpack("<12f", (d / (name + ".center.lighting.bin")).read_bytes()))
                for d in (control, filtered)]
        ref = list(struct.iter_unpack("<3f", (reference / (name + ".linear.f32")).read_bytes()))
        hits = [(d / (name + ".center.hits.bin")).read_bytes() for d in (control, filtered, reference)]
        assert hits[0] == hits[1] == hits[2], name
        decoded = list(struct.iter_unpack("<3iI4f", hits[0]))
        pixels = rows(reference / (name + ".pixels.csv"))
        height = max(int(p["y"]) for p in pixels) + 1
        assert len(a) == len(b) == len(ref) == len(pixels)
        partial = [i for i,p in enumerate(pixels) if 0 < float(p["coverage"]) < 1]
        # The flight harness uses a fixed 45-degree vertical field of view.
        resolved = [i for i,h in enumerate(decoded) if (h[3] & 3) == 1 and
                    h[7] * (2 * math.tan(math.pi/8)) / height <= float(pose["grid_metres"])]
        sky = [i for i,h in enumerate(decoded) if (h[3] & 3) == 0]
        assert all(a[i] == b[i] for i in resolved + sky), name + ": changed resolved voxel or sky"
        r = {"case": name, "changed_center_pixels": sum(x[:3] != y[:3] for x,y in zip(a,b))}
        for group, mask in [("all",range(len(a))), ("partial",partial), ("resolved",resolved), ("sky",sky)]:
            r[group] = dict(count=len(mask), control_rmse=rmse(a,ref,mask), filtered_rmse=rmse(b,ref,mask))
        result.append(r)
        images[name] = a,b,ref
    deltas = []
    for name in images:
        if not name.endswith("move0"):
            continue
        partner = name[:-1] + "1"
        if partner not in images:
            continue
        a,b,ref = images[name]; am,bm,refm = images[partner]
        delta = lambda x,y: [[y[i][c]-x[i][c] for c in range(3)] for i in range(len(x))]
        target = delta(ref,refm)
        deltas.append(dict(case=name[:-6], control_delta_rmse=rmse(delta(a,am),target,range(len(a))),
                           filtered_delta_rmse=rmse(delta(b,bm),target,range(len(a)))))
    return dict(scope="spatial error and two-pose delta error; not continuous-motion acceptance",
                cases=result, pose_deltas=deltas, geometry_and_material_checks=checks)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("control", "filtered", "reference"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    print(json.dumps(analyze(args.control, args.filtered, args.reference), indent=2))
