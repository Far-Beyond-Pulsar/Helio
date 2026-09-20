"""Read preserved, unmodified captures; requires Pillow and NumPy."""
from pathlib import Path
import csv
import json
import numpy as np
from PIL import Image

root = Path(__file__).parent
report = {"image_differences": {}, "hlfs_only_ms": {}}
for frame in [0, 31, 63, 99]:
    a = np.asarray(Image.open(root / "on" / f"cathedral-{frame:03}.png"))[:, :, :3].astype(float)
    b = np.asarray(Image.open(root / "off" / f"cathedral-{frame:03}.png"))[:, :, :3].astype(float)
    difference = np.abs(a - b)
    report["image_differences"][str(frame)] = {
        "changed_pixels": int(np.any(difference != 0, axis=2).sum()),
        "mean_absolute_rgb_8bit": float(difference.mean()),
        "max_channel_difference_8bit": float(difference.max()),
    }
assert report["image_differences"]["0"]["changed_pixels"] == 0
for mode in ["on", "off", "4k"]:
    with (root / mode / "hlfs-gpu-timings.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    values = [float(row["hlfs_only_ms"]) for row in rows if int(row["frame"]) >= 16]
    report["hlfs_only_ms"][mode] = {
        "samples": len(values), "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
    }
print(json.dumps(report, indent=2))
