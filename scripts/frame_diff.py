#!/usr/bin/env python3
"""Diff two directories of rendered frames (e.g. resolution_bench captures).

Usage:
    scripts/frame_diff.py BEFORE_DIR AFTER_DIR [--tol 2] [--max-bad 0.0001]
                          [--heatmaps OUT_DIR]

For every PNG present in both directories it reports:

* exact:  pixels whose RGB differs at all (count and fraction)
* >tol:   pixels whose largest channel difference exceeds --tol (8-bit units)
* max / mean absolute channel difference, RMSE and PSNR (dB, inf = identical)
* bbox:   bounding box (x0,y0,x1,y1) of the >tol pixels

A frame FAILS when its >tol fraction exceeds --max-bad. The exit status is the
number of failing frames, so the script can gate a change. To learn a scene's
harmless nondeterminism, diff two captures of the *same* build first and use
that as the noise floor before judging a change.
"""
import argparse
import math
import os
import sys

import numpy as np
from PIL import Image


def load(path):
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.int16)


def diff(a, b, tol):
    if a.shape != b.shape:
        return None
    d = np.abs(a - b)
    per_pixel = d.max(axis=2)
    total = per_pixel.size
    exact = int((per_pixel > 0).sum())
    bad_mask = per_pixel > tol
    bad = int(bad_mask.sum())
    mse = float((d.astype(np.float64) ** 2).mean())
    psnr = math.inf if mse == 0 else 10 * math.log10(255.0**2 / mse)
    bbox = None
    if bad:
        ys, xs = np.nonzero(bad_mask)
        bbox = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
    return {
        "total": total,
        "exact": exact,
        "bad": bad,
        "max": int(d.max()),
        "mean": float(d.mean()),
        "rmse": math.sqrt(mse),
        "psnr": psnr,
        "bbox": bbox,
        "per_pixel": per_pixel,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("before")
    ap.add_argument("after")
    ap.add_argument("--tol", type=int, default=2, help="per-channel 8-bit tolerance (default 2)")
    ap.add_argument("--max-bad", type=float, default=1e-4, help="max fraction of pixels above --tol (default 1e-4)")
    ap.add_argument("--heatmaps", help="write amplified difference images here")
    args = ap.parse_args()

    names = sorted(n for n in os.listdir(args.before) if n.endswith(".png") and os.path.exists(os.path.join(args.after, n)))
    if not names:
        print("no common PNGs", file=sys.stderr)
        return 1
    if args.heatmaps:
        os.makedirs(args.heatmaps, exist_ok=True)

    failures = 0
    print(f"| frame | exact px | exact % | >{args.tol} px | >{args.tol} % | max | mean | RMSE | PSNR dB | bbox (>{args.tol}) | verdict |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|")
    for name in names:
        r = diff(load(os.path.join(args.before, name)), load(os.path.join(args.after, name)), args.tol)
        if r is None:
            print(f"| {name} | size mismatch | | | | | | | | | FAIL |")
            failures += 1
            continue
        frac = r["bad"] / r["total"]
        ok = frac <= args.max_bad
        failures += not ok
        psnr = "inf" if math.isinf(r["psnr"]) else f"{r['psnr']:.1f}"
        print(
            f"| {name} | {r['exact']} | {100 * r['exact'] / r['total']:.4f} | {r['bad']} | {100 * frac:.4f} | "
            f"{r['max']} | {r['mean']:.4f} | {r['rmse']:.3f} | {psnr} | {r['bbox'] or '-'} | {'ok' if ok else 'FAIL'} |"
        )
        if args.heatmaps and r["exact"]:
            amp = np.clip(r["per_pixel"].astype(np.int32) * 32, 0, 255).astype(np.uint8)
            Image.fromarray(amp, "L").save(os.path.join(args.heatmaps, name))
    return failures


if __name__ == "__main__":
    sys.exit(main())
