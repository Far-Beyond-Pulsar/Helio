"""Summarize measured TSR decisions and validate instrumentation against control HDR."""
import argparse
import json
from pathlib import Path

import numpy as np
from linear_hdr_reference import read_capture, matching_context

FIELDS = ["history_weight", "history_count", "current_blend", "classification",
          "clamp_distance_ycocg", "motion_output_pixels", "current_tonemapped_luma",
          "history_tonemapped_luma"]


def quantiles(values):
    if values.size == 0:
        return None
    return {name: float(value) for name, value in zip(
        ["min", "p10", "median", "p90", "p99", "max"],
        np.quantile(values, [0, .1, .5, .9, .99, 1]))}


def read_diagnostics(path, width, height):
    """Read tight little-endian records; dimensions come from validated HDR metadata."""
    if any(type(v) is not int or v < 1 for v in (width, height)):
        raise ValueError("Invalid diagnostic dimensions")
    path = Path(path)
    if path.stat().st_size != width * height * 32:
        raise ValueError("Invalid diagnostic record size")
    data = np.fromfile(path, dtype="<f4").reshape(height, width, 8)
    if not np.isfinite(data).all():
        raise ValueError("Nonfinite diagnostic records")
    raw_flags = data[..., 3]
    if ((raw_flags < 0) | (raw_flags > 111) | (raw_flags != np.floor(raw_flags))).any():
        raise ValueError("Invalid diagnostic classification")
    flags = raw_flags.astype(np.uint32)
    if (flags & ~np.uint32(111)).any():
        raise ValueError("Invalid diagnostic classification")
    if not ((data[..., 2] >= .03999) & (data[..., 2] <= 1)).all():
        raise ValueError("Invalid blend value")
    if ((data[..., 1] < 0) | (data[..., 1] > 32.0001)).any():
        raise ValueError("Invalid history count")
    if (data[..., 4:6] < 0).any():
        raise ValueError("Negative diagnostic distance")
    return data


def analyze(protocol, control, instrumented):
    frames = protocol["frames"]
    if (not isinstance(frames, list) or len(frames) < 2
            or any(type(f) is not int for f in frames)
            or frames != list(range(frames[0], frames[-1]+1))):
        raise ValueError("Expected at least two consecutive frames")
    if not protocol["regions"]:
        raise ValueError("At least one diagnostic region is required")
    regions = {name: [] for name in protocol["regions"]}
    comparison = []
    for frame in frames:
        stem = f"candidate-{frame:04d}"
        a, baseline = read_capture(control / f"{stem}.json")
        b, measured = read_capture(instrumented / f"{stem}.json")
        matching_context(a, b)
        if any(m["kind"] != "candidate" or m["stage"] != "post_aa_linear" for m in (a,b)):
            raise ValueError("Expected post-AA candidate captures")
        if a.get("frame") != frame or b.get("frame") != frame:
            raise ValueError("Frame identity mismatch")
        description = instrumented / f"{stem}.tsr.json"
        if description.exists():
            diagnostic_meta = json.loads(description.read_text(encoding="utf-8"))
            if (diagnostic_meta.get("schema") != "helio.tsr_diagnostics.v1"
                    or diagnostic_meta.get("encoding") != "f32x8_le"
                    or diagnostic_meta.get("fields") != FIELDS
                    or diagnostic_meta.get("source_hdr") != f"{stem}.json"
                    or diagnostic_meta.get("data") != f"{stem}.tsr.f32x8"
                    or diagnostic_meta.get("frame") != frame
                    or diagnostic_meta.get("producer_sha256") != b.get("producer_sha256")):
                raise ValueError("Mismatched diagnostic metadata")
            matching_context(b, diagnostic_meta)
        data = read_diagnostics(instrumented / f"{stem}.tsr.f32x8", b["width"], b["height"])
        delta = baseline.astype(np.float64) - measured
        comparison.append({"frame": frame, "legacy_layout_without_descriptor": not description.exists(),
            "bit_identical_hdr": bool(np.array_equal(baseline, measured)),
            "rmse": float(np.sqrt(np.mean(delta*delta))), "max_abs": float(np.abs(delta).max())})
        for name, (x, y, w, h) in protocol["regions"].items():
            if any(type(v) is not int for v in (x,y,w,h)):
                raise ValueError("Diagnostic regions must contain integers")
            if min(x,y) < 0 or min(w,h) <= 0 or x+w > b["width"] or y+h > b["height"]:
                raise ValueError("Diagnostic region outside crop")
            regions[name].append(data[y:y+h, x:x+w].reshape(-1,8))
    summaries = {}
    for name, arrays in regions.items():
        v = np.concatenate(arrays)
        flags = v[:,3].astype(np.uint32)
        valid = (flags & np.uint32(2|32|64)) == 0
        summaries[name] = {
            "samples": len(v), "valid_history_fraction": float(valid.mean()),
            "large_motion_fraction": float(((flags&1)!=0).mean()),
            "disocclusion_fraction": float(((flags&2)!=0).mean()),
            "shimmer_fraction": float(((flags&4)!=0).mean()),
            "edge_fraction": float(((flags&8)!=0).mean()),
            "reset_fraction": float(((flags&32)!=0).mean()),
            "outside_fraction": float(((flags&64)!=0).mean()),
            "history_weight_valid": quantiles(v[valid,0]),
            "history_count_valid": quantiles(v[valid,1]),
            "current_blend_valid": quantiles(v[valid,2]),
            "current_blend_all": quantiles(v[:,2]),
            "clip_distance_valid": quantiles(v[valid,4]),
            "clipped_valid_fraction": float((v[valid,4]>1e-5).mean()) if valid.any() else None,
            "motion_pixels_valid": quantiles(v[valid,5]),
        }
    return {"comparison":comparison,"instrumentation_gate":all(r["rmse"]<=1e-5 for r in comparison),
        "regions":summaries,"scope":"Actual shader decisions in the captured crop. Instrumented runs are not performance measurements; valid history is the renderer's decision, not an independent visibility oracle."}


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("protocol",type=Path)
    parser.add_argument("control",type=Path)
    parser.add_argument("instrumented",type=Path)
    parser.add_argument("output",type=Path)
    args=parser.parse_args()
    result=analyze(json.loads(args.protocol.read_text()),args.control,args.instrumented)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    print(json.dumps(result,indent=2))
