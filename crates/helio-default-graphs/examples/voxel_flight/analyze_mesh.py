"""Analyze local exact-face raster experiments, using only the standard library.

python -B analyze_mesh.py RUN_DIRECTORY [--timing-log LOG]

RGBA16 captures use synthetic directional Lambert lighting, not engine shading.
The 8x8 spatial mean is a sampled reference, not a convergence proof. Two-pose
deltas include parallax and are not a continuous-motion/flicker metric.
Outputs are generated evidence and must remain outside version control.
"""
import argparse
import ast
import json
import math
from pathlib import Path
import re
import struct
import zlib


def read(path, width, height):
    data = path.read_bytes()
    assert len(data) == width * height * 8, path
    pixels = list(struct.iter_unpack("<4e", data))
    assert all(math.isfinite(v) and 0 <= v <= 1 for p in pixels for v in p), path
    return pixels


def downsample(pixels, width=128, height=72, scale=8):
    source_width = width * scale
    return [tuple(sum(pixels[(y * scale + sy) * source_width + x * scale + sx][c]
                      for sy in range(scale) for sx in range(scale)) / (scale * scale)
                  for c in range(4)) for y in range(height) for x in range(width)]


def rmse(a, b, mask=None, channels=3):
    indices = range(len(a)) if mask is None else mask
    terms = [(a[i][c] - b[i][c]) ** 2 for i in indices for c in range(channels)]
    return math.sqrt(sum(terms) / len(terms)) if terms else None


def preview(path, images, width=128, height=72, scale=3):
    # Side by side: 1x, 4x independently shaded samples, 64-sample mean.
    rows = []
    for y in range(height):
        line = b"".join(bytes(round(max(0, min(1, p[c])) ** (1 / 2.2) * 255)
                              for c in range(3)) * scale
                        for image in images for p in image[y * width:(y + 1) * width])
        rows.extend([b"\0" + line] * scale)
    def chunk(kind, data):
        return (struct.pack(">I", len(data)) + kind + data
                + struct.pack(">I", zlib.crc32(kind + data)))
    header = struct.pack(">IIBBBBB", width * len(images) * scale, height * scale, 8, 2, 0, 0, 0)
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header)
                     + chunk(b"IDAT", zlib.compress(b"".join(rows))) + chunk(b"IEND", b""))


def analyze(root, timing_log=None):
    rows = []
    all_images = {}
    for fixture in range(5):
        for light in range(2):
            for movement in range(2):
                name = f"f{fixture}-l{light}-m{movement}"
                one = read(root / f"{name}-s1-128x72.rgba16", 128, 72)
                four = read(root / f"{name}-s4-128x72.rgba16", 128, 72)
                trace = read(root / f"{name}-trace-128x72.rgba16", 128, 72)
                high = read(root / f"{name}-s1-1024x576.rgba16", 1024, 576)
                reference = downsample(high)
                constant = [i for i, p in enumerate(reference) if p[3] == 1 and
                            all(high[(i // 128 * 8 + sy) * 1024 + i % 128 * 8 + sx] == p
                                for sy in range(8) for sx in range(8))]
                partial = [i for i, p in enumerate(reference) if 0 < p[3] < 1]
                interior = [i for i, p in enumerate(reference) if p[3] == 1]
                rows.append({"case": name, "partial_pixels": len(partial),
                             "constant_reference_pixels": len(constant),
                             "constant_reference_changed_pixels": sum(one[i] != four[i] for i in constant),
                             "trace_vs_one_rgb_rmse": rmse(trace, one),
                             "trace_vs_one_different_pixels": sum(a != b for a, b in zip(trace, one)),
                             "one_rgb_rmse": rmse(one, reference),
                             "four_rgb_rmse": rmse(four, reference),
                             "one_partial_rmse": rmse(one, reference, partial),
                             "four_partial_rmse": rmse(four, reference, partial),
                             "one_interior_rmse": rmse(one, reference, interior),
                             "four_interior_rmse": rmse(four, reference, interior),
                             "one_coverage_rmse": rmse([p[3:] for p in one], [p[3:] for p in reference], channels=1),
                             "four_coverage_rmse": rmse([p[3:] for p in four], [p[3:] for p in reference], channels=1)})
                preview(root / f"{name}-comparison.png", [one, four, reference])
                all_images[fixture, light, movement] = [one, four, reference]
    motion = []
    for fixture in range(5):
        for light in range(2):
            a = all_images[fixture, light, 0]
            b = all_images[fixture, light, 1]
            delta = [[tuple(q[c] - p[c] for c in range(3)) for p, q in zip(x, y)] for x, y in zip(a, b)]
            motion.append({"fixture": fixture, "light": light,
                           "one_delta_rmse": rmse(delta[0], delta[2]),
                           "four_delta_rmse": rmse(delta[1], delta[2])})
    times = []
    if timing_log:
        for match in re.finditer(r"SURFACE_MESH_GPU fixture=(\d+) samples=(\d+) milliseconds=(\[[^\n]+\])", timing_log.read_text()):
            values = sorted(ast.literal_eval(match[3]))
            assert len(values) == 128 and all(math.isfinite(v) and v >= 0 for v in values)
            times.append({"fixture": int(match[1]), "samples": int(match[2]),
                          "method": "raster",
                          "n": len(values), "median_ms": values[63], "p95_ms": values[121],
                          "p99_ms": values[126], "max_ms": values[-1]})
        for match in re.finditer(r"SURFACE_MESH_TRACE fixture=(\d+) repeat=(\d+) milliseconds=(\[[^\n]+\])", timing_log.read_text()):
            values = sorted(ast.literal_eval(match[3]))
            assert len(values) == 128 and all(math.isfinite(v) and v >= 0 for v in values)
            times.append({"fixture": int(match[1]), "samples": 1, "method": "trace",
                          "repeat": int(match[2]), "n": len(values),
                          "median_ms": values[63], "p95_ms": values[121],
                          "p99_ms": values[126], "max_ms": values[-1]})
    report = {"schema": 1, "scope": __doc__, "cases": rows, "two_pose_delta": motion, "gpu_timings": times}
    (root / "analysis.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--timing-log", type=Path)
    args = parser.parse_args()
    print(json.dumps(analyze(args.run, args.timing_log), indent=2))
