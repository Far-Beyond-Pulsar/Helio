#!/usr/bin/env python3
"""Compare two resolution_bench runs.

Usage:
    scripts/bench_compare.py BEFORE_DIR AFTER_DIR [--passes PassA,PassB] [--top 8]

Prints, per scene and resolution, the median frame GPU wait and render CPU
time before/after, plus the per-pass GPU deltas (all passes named with
--passes, else the --top passes with the largest absolute change).
"""
import argparse
import csv
import os
import statistics
from collections import defaultdict


def frames(path):
    rows = defaultdict(lambda: ([], []))
    with open(os.path.join(path, "frames.csv")) as f:
        for r in csv.DictReader(f):
            key = (r["scene"], int(r["width"]), int(r["height"]))
            rows[key][0].append(float(r["cpu_ms"]))
            rows[key][1].append(float(r["gpu_ms"]))
    return {k: (statistics.median(c), statistics.median(g)) for k, (c, g) in rows.items()}


def passes(path):
    out = defaultdict(dict)
    with open(os.path.join(path, "passes.csv")) as f:
        for r in csv.DictReader(f):
            key = (r["scene"], int(r["width"]), int(r["height"]))
            gpu = float(r["gpu_ms"])
            if gpu == gpu:  # skip NaN
                out[key][r["pass"]] = gpu
    return out


def pct(a, b):
    return f"{100.0 * (b - a) / a:+.1f}%" if a else "n/a"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("before")
    ap.add_argument("after")
    ap.add_argument("--passes", default="")
    ap.add_argument("--top", type=int, default=8)
    args = ap.parse_args()
    fb, fa = frames(args.before), frames(args.after)
    pb, pa = passes(args.before), passes(args.after)

    print("| scene | output | GPU wait before | after | Δ | render CPU before | after | Δ |")
    print("|---|---|---:|---:|---:|---:|---:|---:|")
    for key in sorted(fb.keys() & fa.keys()):
        (cb, gb), (ca, ga) = fb[key], fa[key]
        print(f"| {key[0]} | {key[1]}x{key[2]} | {gb:.1f} | {ga:.1f} | {pct(gb, ga)} | {cb:.2f} | {ca:.2f} | {pct(cb, ca)} |")

    wanted = [p for p in args.passes.split(",") if p]
    print("\n| scene | output | pass | GPU before | after | Δ ms |")
    print("|---|---|---|---:|---:|---:|")
    for key in sorted(pb.keys() & pa.keys()):
        names = set(pb[key]) | set(pa[key])
        names = [n for n in names if not n.startswith("__")]
        if wanted:
            chosen = [n for n in wanted if n in names]
        else:
            chosen = sorted(names, key=lambda n: -abs(pa[key].get(n, 0) - pb[key].get(n, 0)))[: args.top]
        for n in chosen:
            b, a = pb[key].get(n, 0.0), pa[key].get(n, 0.0)
            print(f"| {key[0]} | {key[1]}x{key[2]} | {n} | {b:.2f} | {a:.2f} | {a - b:+.2f} |")


if __name__ == "__main__":
    main()
