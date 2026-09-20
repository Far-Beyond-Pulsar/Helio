from pathlib import Path
import csv
import json
import numpy as np
from PIL import Image

root = Path(__file__).parent
report = {"hlfs_only_ms": {}, "daylight_reference_nrmse": {}}
for mode in ["daylight", "clear", "window-control", "reference", "4k", "ssr"]:
    with (root / mode / "hlfs-gpu-timings.csv").open() as handle:
        rows = [row for row in csv.DictReader(handle) if int(row["frame"]) >= 16]
    values = [float(row["hlfs_only_ms"]) for row in rows]
    report["hlfs_only_ms"][mode] = {"median": float(np.median(values)), "p95": float(np.percentile(values,95)), "samples": len(values)}
for frame in [31,63,99]:
    a=np.asarray(Image.open(root / "daylight" / f"cathedral-{frame:03}.png"))[:,:,:3].astype(float)
    b=np.asarray(Image.open(root / "reference" / f"cathedral-reference-{frame:03}.png"))[:,:,:3].astype(float)
    report["daylight_reference_nrmse"][str(frame)]=float(np.sqrt(np.mean((a-b)**2)/np.mean(b*b)))
print(json.dumps(report,indent=2))
