"""Measure stationary temporal changes in GPU captures (requires Pillow, NumPy).
Usage: python tools/voxel-planet/analyze_captures.py target/engine-jitter-static
This measures image changes, not correctness, FPS or motion-compensated error.
"""
import json
import sys
from pathlib import Path
import numpy as np
from PIL import Image

folder=Path(sys.argv[1])
files=sorted(folder.glob("frame-*.bmp"))
if len(files)<2:
    raise SystemExit("At least two captures are required")
metadata=[json.loads(f.with_suffix(".json").read_text()) for f in files]
changes=[]
previous=None
for file, meta in zip(files,metadata):
    current=np.asarray(Image.open(file).convert("RGB"),dtype=np.float32)
    if previous is not None: changes.append(np.abs(current-previous))
    previous=current
changes=np.stack(changes)
def pose(m):
    # A motion flag alone can miss different held viewpoints (and is unused by
    # engine captures). Check the actual high-precision camera and its basis.
    return (m["origin_cell"][:3],m["fraction"][:3],m["forward"][:3],
            m["right"][:3],m["up"][:3])
stationary=all(pose(m)==pose(metadata[0]) for m in metadata)
result={
    "captures":len(files),"stationary":stationary,
    "capture_steps":sorted(set(b["frame"]-a["frame"] for a,b in zip(metadata,metadata[1:]))),
    "mean_channel_change_255":float(changes.mean()),
    "rms_channel_change_255":float(np.sqrt(np.mean(changes*changes))),
    "p95_channel_change_255":float(np.percentile(changes,95)),
    "pixels_changing_more_than_8_percent":float((changes.max(axis=3)>8).mean()*100),
    "unresolved_pixels_total":sum(m["unresolved_pixels"] for m in metadata) if all("unresolved_pixels" in m for m in metadata) else None,
}
print(json.dumps(result,indent=2))
