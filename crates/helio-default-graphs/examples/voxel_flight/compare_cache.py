"""Compare identical spatial-reference captures with the cache off/on.

Reports all primary-cell/material/face differences and linear-light errors.
Only the documented cache-provenance status bit is excluded from status checks.
No temporal matching, image alignment, or error threshold hides differences.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import struct


def compare(control: Path, candidate: Path):
    poses_a = {r["name"]: r for r in csv.DictReader((control / "poses.csv").open())}
    poses_b = {r["name"]: r for r in csv.DictReader((candidate / "poses.csv").open())}
    if poses_a != poses_b:
        raise ValueError("capture names, poses, lights or sampling grids differ")
    results = []
    for name, pose in poses_a.items():
        with (control / f"{name}.pixels.csv").open() as f:
            count = sum(1 for _ in csv.DictReader(f))
        samples = int(pose["samples_per_pixel"])
        paths = [root / f"{name}.samples.bin" for root in (control, candidate)]
        data = [p.read_bytes() for p in paths]
        if any(len(d) != count * 80 * samples for d in data):
            raise ValueError(f"{name}: incomplete sample record")
        r = dict(name=name, primary_samples=count * samples, cached_primary_samples=0,
                 invalid_primary=0, coverage_differences=0, status_differences=0,
                 hierarchy_level_differences=0, cell_differences=0,
                 material_differences=0, face_differences=0, ray_differences=0,
                 depth_differences=0, maximum_depth_error_m=0.0,
                 sunlight_differences=0, albedo_differences=0,
                 lighting_differences=0, lighting_squared_error=0.0,
                 maximum_lighting_error=0.0)
        for sample in range(samples):
            at = sample * count * 80
            lights = [struct.iter_unpack("<12f", d[at:at+count*48]) for d in data]
            hits = [struct.iter_unpack("<3iI4f", d[at+count*48:at+count*80]) for d in data]
            for a, b, la, lb in zip(*hits, *lights):
                sa, sb = a[3] & 3, b[3] & 3
                r["cached_primary_samples"] += bool(b[3] & 0x08000000)
                r["invalid_primary"] += sa > 1 or sb > 1
                r["coverage_differences"] += sa != sb
                r["status_differences"] += (a[3] & ~0x08000000) != (b[3] & ~0x08000000)
                r["ray_differences"] += a[4:7] != b[4:7]
                if sa == sb == 1:
                    r["hierarchy_level_differences"] += (a[3] >> 2 & 31) != (b[3] >> 2 & 31)
                    r["cell_differences"] += a[:3] != b[:3]
                    r["material_differences"] += (a[3] >> 8 & 3) != (b[3] >> 8 & 3)
                    r["face_differences"] += (a[3] >> 28 & 7) != (b[3] >> 28 & 7)
                    error = abs(a[7] - b[7])
                    r["depth_differences"] += error != 0
                    r["maximum_depth_error_m"] = max(r["maximum_depth_error_m"], error)
                    r["sunlight_differences"] += la[8] != lb[8]
                    r["albedo_differences"] += la[4:8] != lb[4:8]
                if not all(math.isfinite(v) for v in la + lb + a[4:] + b[4:]):
                    raise ValueError(f"{name}: non-finite sample")
                r["lighting_differences"] += la[:3] != lb[:3]
                for x, y in zip(la[:3], lb[:3]):
                    r["lighting_squared_error"] += (x-y)**2
                    r["maximum_lighting_error"] = max(r["maximum_lighting_error"], abs(x-y))
        r["linear_sample_rmse"] = math.sqrt(r.pop("lighting_squared_error") / (3*count*samples))
        means = [list(struct.iter_unpack("<f", (root / f"{name}.linear.f32").read_bytes()))
                 for root in (control, candidate)]
        if any(len(v) != count*3 for v in means):
            raise ValueError(f"{name}: incomplete spatial mean")
        r["linear_mean_rmse"] = math.sqrt(sum((a[0]-b[0])**2 for a,b in zip(*means))/(count*3))
        r["input_sha256"] = [hashlib.sha256(d).hexdigest() for d in data]
        results.append(r)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("control", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--strict", action="store_true", help="fail on any geometry, depth, ray or lighting difference")
    args = parser.parse_args()
    results = compare(args.control, args.candidate)
    print(json.dumps(results, indent=2))
    if args.strict and any(r["invalid_primary"] or any(v for k,v in r.items() if k.endswith("_differences")) for r in results):
        raise SystemExit(1)
