"""Summarize unique completed GPU frame spans from --timings (not FPS)."""
import argparse
import json
from pathlib import Path
import numpy as np

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("capture", type=Path)
args = parser.parse_args()
data = json.loads(args.capture.read_text())
samples = data["samples"]
frames = [s["gpu_frame"] for s in samples]
if len(set(frames)) != len(frames):
    raise ValueError("Duplicate GPU frames would bias statistics")
values = [s.get("gpu_frame_ms") for s in samples]
if not values or any(v is None for v in values):
    raise ValueError("Complete graph timestamps unavailable; legacy pass sums omit graphics work")
result = {
    "source": str(args.capture), "scope": data["scope"],
    "availability": data["availability"], "display_size": data["display_size"],
    "render_scale": data["render_scale"], "warmup_frames": data["warmup_frames"],
    "samples": len(values), "first_gpu_frame": min(frames), "last_gpu_frame": max(frames),
    "unobserved_frames_between_samples": max(frames)-min(frames)+1-len(frames),
    "readback_drops": data["readback_drops"], "query_overflows": data["query_overflows"],
    "gpu_frame_ms": {"median": float(np.median(values)), "p95": float(np.percentile(values,95)),
                     "maximum": float(np.max(values))},
}
print(json.dumps(result, indent=2))
