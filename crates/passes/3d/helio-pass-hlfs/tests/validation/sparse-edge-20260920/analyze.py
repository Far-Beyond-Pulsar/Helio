from pathlib import Path
import csv
import json
import numpy as np
from PIL import Image

root = Path(__file__).parent
reference = np.asarray(Image.open(root / "reference/cathedral-reference-099.png"))[:, :, :3].astype(float)
report = {}
images = {}
for mode in ["baseline", "candidate"]:
    stack = np.stack([np.asarray(Image.open(root / mode / f"cathedral-{frame:03}.png"))[:, :, :3].astype(float) for frame in [96,97,98,99]])
    images[mode] = stack
    difference = stack[-1] - reference
    with (root / mode / "hlfs-gpu-timings.csv").open() as handle:
        rows = [row for row in csv.DictReader(handle) if int(row["frame"]) >= 16]
    report[mode] = {
        "rgb_nrmse": float(np.sqrt(np.mean(difference**2) / np.mean(reference**2))),
        "bottom_440_rows_nrmse": float(np.sqrt(np.mean(difference[1000:]**2) / np.mean(reference[1000:]**2))),
        "four_frame_mean_rgb_std_8bit": float(np.std(stack,axis=0).mean()),
        "timing": {key: {"median_ms": float(np.median([float(row[key]) for row in rows])), "p95_ms": float(np.percentile([float(row[key]) for row in rows],95))} for key in ["hlfs_only_ms","composite_ms"]},
    }
report["exact_rgb_equal"] = bool(np.array_equal(images["baseline"], images["candidate"]))
print(json.dumps(report,indent=2))
