"""Compare matched engine mesh/control captures; stdlib only.

python -B analyze_mesh_engine.py CONTROL CANDIDATE

Supports spatial references (*.samples.bin) and fixed-jitter flight captures
(*.hits.bin). The control is the existing exact-cache renderer, not an independent
oracle. Status attribution bits are excluded; cells, face, material, depth and
ray bits are compared explicitly. Generated reports belong under target/.
"""
import argparse
import csv
import json
import math
from pathlib import Path
import struct


def compare_hits(a, b):
    assert len(a) == len(b) and len(a) % 32 == 0
    result = dict(rays=len(a) // 32, raster_hits=0, geometry_differences=0,
                  depth_differences=0, invalid_statuses=0, maximum_depth_delta_m=0.0)
    disagreements = []
    for i, (x, y) in enumerate(zip(struct.iter_unpack("<3iI4f", a), struct.iter_unpack("<3iI4f", b))):
        assert a[i*32+16:i*32+28] == b[i*32+16:i*32+28], f"ray bits differ at {i}; use matched camera jitter"
        assert all(math.isfinite(v) for v in x[4:] + y[4:])
        result["raster_hits"] += bool(y[3] & (1 << 26))
        # Low status, material, face. Hierarchy level and backend attribution
        # describe traversal rather than the physical first surface.
        different = x[:3] != y[:3] or (x[3] & 0x70000303) != (y[3] & 0x70000303)
        result["geometry_differences"] += different
        result["depth_differences"] += x[7] != y[7]
        result["maximum_depth_delta_m"] = max(result["maximum_depth_delta_m"], abs(x[7] - y[7]))
        result["invalid_statuses"] += (y[3] & 3) not in (0, 1)
        if different and len(disagreements) < 32:
            disagreements.append(dict(pixel=i, control=x[:4], candidate=y[:4], ray=y[4:7]))
    result["first_geometry_disagreements"] = disagreements
    return result


def compare(control, candidate):
    cases = {}
    samples = {}
    if (candidate / "poses.csv").exists():
        assert (control / "poses.csv").read_bytes() == (candidate / "poses.csv").read_bytes()
        samples = {p["name"]: int(p["samples_per_pixel"]) for p in csv.DictReader((candidate / "poses.csv").open())}
    hit_paths = sorted(candidate.glob("*.hits.bin"))
    assert {p.name for p in hit_paths} == {p.name for p in control.glob("*.hits.bin")}, "capture sets differ"
    if (candidate / "benchmark.json").exists():
        a = json.loads((control / "benchmark.json").read_text()); b = json.loads((candidate / "benchmark.json").read_text())
        for key in ["fixture", "size", "voxel_size_m", "recording", "motion", "fixed_jitter"]:
            assert a[key] == b[key], key
        expected = {"cache_stationary.hits.bin", "cache_motion.hits.bin"}
        if b["recording"]:
            expected |= {f"motion-{i:03}.hits.bin" for i in range(b["frames_per_stage"])}
        assert {p.name for p in hit_paths} == expected, "incomplete capture sequence"
    for path in hit_paths:
        cases[path.name] = compare_hits((control / path.name).read_bytes(), path.read_bytes())
    for name, count in samples.items():
        a = (control / (name + ".samples.bin")).read_bytes()
        b = (candidate / (name + ".samples.bin")).read_bytes()
        assert len(a) == len(b) and len(a) % (count * 80) == 0
        pixels = len(a) // (count * 80)
        ha, hb = bytearray(), bytearray()
        lighting_differences = 0
        for sample in range(count):
            base = sample * pixels * 80
            split = base + pixels * 48
            end = base + pixels * 80
            ha.extend(a[split:end]); hb.extend(b[split:end])
            lighting_differences += sum(a[i:i+48] != b[i:i+48] for i in range(base, split, 48))
        result = compare_hits(ha, hb)
        result["lighting_albedo_sunlight_differences"] = lighting_differences
        cases[name + ".samples.bin"] = result
    assert cases, "no raw hit captures"
    assert not any(c["invalid_statuses"] for c in cases.values()), "invalid primary status"
    report = dict(scope="Matched engine control comparison; not an independent geometry oracle or whole-planet acceptance", lighting_compared=bool(samples), cases=cases)
    (candidate / "mesh-engine-comparison.json").write_text(json.dumps(report, indent=2))
    summary = {k: sum(c.get(k, 0) for c in cases.values()) for k in
               ["rays", "raster_hits", "geometry_differences", "depth_differences"]}
    summary["lighting_albedo_sunlight_differences"] = (
        sum(c.get("lighting_albedo_sunlight_differences", 0) for c in cases.values()) if samples else None)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("control", type=Path)
    parser.add_argument("candidate", type=Path)
    args = parser.parse_args()
    print(json.dumps(compare(args.control, args.candidate), indent=2))
