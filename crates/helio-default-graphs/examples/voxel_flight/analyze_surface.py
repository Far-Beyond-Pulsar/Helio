"""Compare two canonical surface-reference sample grids (stdlib only).

python analyze_surface.py LOWER_SAMPLE_RUN HIGHER_SAMPLE_RUN

Outputs stay in the higher-sample run directory. These are spatial sampling
diagnostics, not a production-filter or performance acceptance test. Camera
translation deltas include real parallax; they are not a flicker metric.
"""
import argparse
import csv
import json
import math
import struct
from pathlib import Path


def rows(path):
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def rms(values):
    values = list(values)
    assert values and all(math.isfinite(v) for v in values)
    return math.sqrt(sum(v * v for v in values) / len(values))


def difference(a, b):
    assert len(a) == len(b)
    return rms(x - y for x, y in zip(a, b))


def read_case(directory, pose):
    name = pose["name"]
    table = rows(directory / (name + ".pixels.csv"))
    count = len(table)
    width = max(int(p["x"]) for p in table) + 1
    height = max(int(p["y"]) for p in table) + 1
    assert count == width * height
    for i, p in enumerate(table):
        assert (int(p["x"]), int(p["y"])) == (i % width, i // width)
        for key in p.keys() - {"x", "y"}:
            if p[key]:
                assert math.isfinite(float(p[key])), (name, key, p[key])
        coverage = float(p["coverage"])
        assert 0 <= coverage <= 1
        faces = [float(p[f"face_{f}_coverage"]) for f in range(6)]
        assert abs(sum(faces) - coverage) < 1e-7
        for f in range(6):
            materials = sum(float(p[f"face_{f}_material_{m}"]) for m in range(4))
            assert abs(materials - faces[f]) < 1e-7
        assert 0 <= float(p["sunlit_coverage"]) <= coverage
        assert bool(p["sampled_depth_min_m"]) == (coverage > 0)
        if coverage > 0:
            assert 0 <= float(p["sampled_depth_min_m"]) <= float(p["sampled_depth_max_m"])
    samples = int(pose["samples_per_pixel"])
    assert (directory / (name + ".samples.bin")).stat().st_size == count * samples * 80
    lighting = (directory / (name + ".center.lighting.bin")).read_bytes()
    hits = (directory / (name + ".center.hits.bin")).read_bytes()
    camera = (directory / (name + ".center.camera.bin")).read_bytes()
    assert len(camera) == 368
    assert len(lighting) == count * 48 and len(hits) == count * 32
    center = [channel for pixel in struct.iter_unpack("<12f", lighting) for channel in pixel[:3]]
    rgb = [float(p[channel]) for p in table for channel in ("r", "g", "b")]
    raw_rgb = [v[0] for v in struct.iter_unpack("<f", (directory / (name + ".linear.f32")).read_bytes())]
    assert len(raw_rgb) == len(rgb) and max(abs(a-b) for a, b in zip(raw_rgb, rgb)) < 1e-6
    mixed = sum(sum(float(p[f"face_{f}_coverage"]) > 0 for f in range(6)) > 1 for p in table)
    partial = sum(0 < float(p["coverage"]) < 1 for p in table)
    return dict(rgb=rgb, center=center, hits=hits, lighting=lighting, camera=camera, table=table,
                count=count, mixed=mixed, partial=partial)


def compare(low, high):
    # README is written only after all fixtures and repeated-frame checks finish.
    assert (low / "README.txt").is_file() and (high / "README.txt").is_file()
    for directory in (low, high):
        capture = json.loads((directory / "capture.json").read_text())
        assert capture["schema"] == 2 and capture["lighting_stream"] == "graphics-current-frame"
        assert not capture["appearance_filter"], "a filtered mean is not an unfiltered reference"

    low_poses = {p["name"]: p for p in rows(low / "poses.csv")}
    high_poses = {p["name"]: p for p in rows(high / "poses.csv")}
    assert low_poses.keys() == high_poses.keys()
    cases = {}
    high_data = {}
    for name, pose in high_poses.items():
        lo_pose = low_poses[name]
        assert int(lo_pose["samples_per_pixel"]) < int(pose["samples_per_pixel"])
        assert {k: v for k, v in pose.items() if k != "samples_per_pixel"} == {
            k: v for k, v in lo_pose.items() if k != "samples_per_pixel"}
        a, b = read_case(low, lo_pose), read_case(high, pose)
        assert a["camera"][:128] == b["camera"][:128] and a["camera"][288:296] == b["camera"][288:296], name + ": camera changed between runs"
        assert a["hits"] == b["hits"] and a["lighting"] == b["lighting"], name + ": center changed between runs"
        pixel_errors = [rms(a["rgb"][i:i+3][c] - b["rgb"][i:i+3][c] for c in range(3))
                        for i in range(0, len(b["rgb"]), 3)]
        pixel_errors.sort()
        cases[name] = dict(
            pixels=b["count"], mixed_face_pixels=b["mixed"], partial_coverage_pixels=b["partial"],
            center_vs_high_linear_rmse=difference(b["center"], b["rgb"]),
            low_vs_high_linear_rmse=difference(a["rgb"], b["rgb"]),
            high_linear_rms=rms(b["rgb"]),
            low_vs_high_pixel_error_p95=pixel_errors[math.ceil(len(pixel_errors) * .95) - 1],
            low_vs_high_pixel_error_max=pixel_errors[-1],
            coverage_rmse=rms(float(x["coverage"]) - float(y["coverage"])
                              for x, y in zip(a["table"], b["table"])),
        )
        # Keep only image arrays for movement diagnostics, not every CSV field.
        high_data[name] = {"rgb": b["rgb"], "center": b["center"]}
    movement = {}
    for name, a in high_data.items():
        if name.endswith("-move0"):
            other = name[:-1] + "1"
            b = high_data[other]
            movement[name[:-6]] = dict(
                center_linear_delta_rmse=difference(a["center"], b["center"]),
                mean_linear_delta_rmse=difference(a["rgb"], b["rgb"]),
                translation_metres=0.025,
                interpretation="Unregistered image delta includes real parallax; not a flicker metric",
            )
    audited = 0
    for path in sorted(high.glob("*.canonical.csv")):
        for p in rows(path):
            assert (p["gpu_status"] == "1") == (p["cpu_hit"] == "1"), path.name
            if p["gpu_status"] == "1":
                assert all(p[f"gpu_{a}"] == p[f"cpu_{a}"] for a in "xyz"), path.name
                assert p["gpu_material"] == p["cpu_material"] == p["canonical_at_gpu"], path.name
            audited += 1
    assert audited == len(high_poses) // 2 * 144
    return dict(lower_samples=int(next(iter(low_poses.values()))["samples_per_pixel"]),
                higher_samples=int(next(iter(high_poses.values()))["samples_per_pixel"]),
                case_count=len(cases), audited_center_rays=audited,
                cases=cases, movement=movement,
                limitation="No infinite-sample convergence proof, conservative depth bound, cached-surface validation or performance qualification")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("lower", type=Path)
    parser.add_argument("higher", type=Path)
    args = parser.parse_args()
    report = compare(args.lower, args.higher)
    output = args.higher / "surface-comparison.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k not in ("cases", "movement")}, indent=2))
    print(f"Per-case results: {output}")
