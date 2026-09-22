"""Compare moving captures to settled AA at exactly the same route poses.

Example:
  python compare_motion_reference.py target/before target/after \
    --reference 239=target/frozen-239 --reference 335=target/frozen-335

Generate references using --motion-check=1 --motion-pose=239 --frames=256
--capture-every=256. A settled render is an image reconstruction reference,
not a proof of correct distant geometry or a measurement of temporal flicker.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("candidates", nargs="+", type=Path)
parser.add_argument("--reference", action="append", required=True, metavar="FRAME=DIRECTORY")
args = parser.parse_args()
rows = []
for spec in args.reference:
    frame_text, directory = spec.split("=", 1)
    frame = int(frame_text)
    reference = sorted(Path(directory).glob("frame-*.bmp"))[-1]
    control = json.loads(reference.with_suffix(".json").read_text())
    a = np.asarray(Image.open(reference).convert("RGB"), dtype=np.float64)
    for candidate in args.candidates:
        path = candidate / f"frame-{frame:04d}.bmp"
        meta = json.loads(path.with_suffix(".json").read_text())
        for key in ["origin_cell", "fraction", "right", "up", "generator_revision"]:
            if meta[key] != control[key]:
                raise ValueError(f"Mismatched {key}: {path} / {reference}")
        for key in ["render_size", "pipeline", "aa", "sun", "ambient", "render_scale", "edits", "voxel_shadows", "shadow_size", "planet_sky", "raytraced_sun"]:
            if meta.get(key) != control.get(key):
                raise ValueError(f"Mismatched {key}: {path} / {reference}")
        # forward.w is the motion flag, so it must differ for a frozen pose.
        if meta["forward"][:3] != control["forward"][:3]:
            raise ValueError(f"Mismatched camera direction: {path} / {reference}")
        b = np.asarray(Image.open(path).convert("RGB"), dtype=np.float64)
        if a.shape != b.shape:
            raise ValueError(f"Mismatched image size: {path} / {reference}")
        delta = np.abs(a - b)
        rows.append({
            "candidate": str(candidate), "frame": frame, "reference": str(reference),
            "mean_absolute_255": float(delta.mean()),
            "rms_255": float(np.sqrt(np.mean(delta * delta))),
            "pixels_over_8": int((delta.max(axis=2) > 8).sum()),
            "unresolved_pixels": meta.get("unresolved_pixels"),
        })
print(json.dumps({"comparisons": rows}, indent=2))
