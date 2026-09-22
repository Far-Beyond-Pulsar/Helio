"""Measure temporal variation at a fixed camera, after cache/history warmup.

This measures sampled static-image variation, not motion fidelity or spatial
accuracy. Report the capture cadence: skipped frames can hide fast flicker.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("directories", nargs="+", type=Path)
parser.add_argument("--first-frame", type=int, default=120)
parser.add_argument("--top", type=int, default=180, help="Crop sky from the fixed ground view")
args = parser.parse_args()
rows = []
control = None
for directory in args.directories:
    paths = [p for p in sorted(directory.glob("frame-*.bmp"))
             if int(p.stem.split("-")[1]) >= args.first_frame]
    if len(paths) < 3:
        raise ValueError(f"Need at least three warmed captures in {directory}")
    frames, images, unresolved = [], [], []
    for path in paths:
        meta = json.loads(path.with_suffix(".json").read_text())
        if control is None:
            control = meta
        for key in ("origin_cell", "fraction", "right", "up", "generator_revision",
                    "render_size", "pipeline", "aa", "sun", "ambient", "render_scale",
                    "edits", "planet_sky", "raytraced_sun", "primary_traversal_mode"):
            if meta.get(key) != control.get(key):
                raise ValueError(f"Mismatched {key}: {path}")
        if meta["forward"][:3] != control["forward"][:3]:
            raise ValueError(f"Moving camera: {path}")
        image = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32)
        if not 0 <= args.top < image.shape[0]:
            raise ValueError("Crop is outside the image")
        images.append(image[args.top:])
        frames.append(int(path.stem.split("-")[1]))
        unresolved.append(meta.get("unresolved_pixels"))
    stack = np.stack(images)
    variance = np.var(stack, axis=0, dtype=np.float64)
    changes = np.diff(stack, axis=0)
    rows.append({"directory": str(directory), "frames": frames, "crop_top": args.top,
                 "unresolved": unresolved,
                 "temporal_rms_255": float(np.sqrt(variance.mean())),
                 "sampled_change_rms_255": float(np.sqrt(np.mean(changes**2))),
                 "pixel_std_p95_255": float(np.percentile(np.sqrt(variance.mean(axis=2)), 95)),
                 "spatial_gradient_rms_255": float(np.sqrt(np.mean(np.diff(stack.mean(axis=0), axis=1)**2)))})
print(json.dumps({"scope": __doc__, "measurements": rows}, indent=2))
