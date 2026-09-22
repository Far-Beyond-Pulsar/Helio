"""Compare paired GPU captures, e.g. cache warmup against uncached traversal.

Usage: python compare_captures.py candidate_directory reference_directory
This measures image differences; it does not establish geometric correctness.
"""
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

candidate, reference = map(Path, sys.argv[1:3])
rows = []
for path in sorted(candidate.glob("frame-*.bmp")):
    other = reference / path.name
    if not other.exists():
        raise ValueError(f"Missing reference frame: {other}")
    meta = json.loads(path.with_suffix(".json").read_text())
    control = json.loads(other.with_suffix(".json").read_text())
    for key in ["origin_cell", "fraction", "forward", "right", "up", "jitter_pixels", "render_size", "generator_revision"]:
        if meta.get(key) != control.get(key):
            raise ValueError(f"Mismatched {key} in frame {path.name}")
    a = np.array(Image.open(path).convert("RGB"), dtype=np.int16)
    b = np.array(Image.open(other).convert("RGB"), dtype=np.int16)
    delta = np.abs(a-b)
    rows.append({"frame": meta["frame"], "mean": float(delta.mean()),
                 "max": int(delta.max()),
                 "pixels_gt8": int((delta.max(axis=2)>8).sum())})
if not rows:
    raise ValueError("No paired captures")
print(json.dumps({"candidate": str(candidate), "reference": str(reference),
                  "frames": rows, "worst_mean": max(rows, key=lambda r: r["mean"])}, indent=2))
