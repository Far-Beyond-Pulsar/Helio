"""Compare consecutive moving HDR frames to a separate reference at each pose.

Residual change removes each pose's known image change in screen coordinates.
It is not motion-compensated flicker and does not make a finite reference exact.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from linear_hdr_reference import matching_context, read_capture


def compare_motion(references, candidates, regions):
    frames = sorted(references)
    if len(frames) < 2 or any(type(f) is not int for f in frames):
        raise ValueError("At least two integer frame indices are required")
    if frames != list(range(frames[0], frames[-1] + 1)):
        raise ValueError("Reference frames must be consecutive")
    if set(candidates) != set(frames):
        raise ValueError("Candidate and reference frames must match exactly")
    if not regions:
        raise ValueError("At least one region is required")
    rows = []
    errors = {name: [] for name in regions}
    for frame in frames:
        reference, truth = read_capture(references[frame])
        candidate, image = read_capture(candidates[frame])
        if reference["kind"] != "reference" or reference["stage"] != "pre_aa_linear":
            raise ValueError("Expected accumulated pre-AA references")
        side = reference.get("strata_side")
        if type(side) is not int or side < 1 or reference.get("samples") != side * side:
            raise ValueError("Incomplete reference strata metadata")
        if candidate["kind"] != "candidate" or candidate.get("frame") != frame:
            raise ValueError("Candidate frame identity mismatch")
        matching_context(reference, candidate)
        row = {"frame": frame, "reference": str(references[frame]),
               "candidate": str(candidates[frame]), "regions": {}}
        for name, region in regions.items():
            if len(region) != 4 or any(type(v) is not int for v in region):
                raise ValueError("Regions must contain four integers")
            x, y, w, h = region
            if min(x, y) < 0 or min(w, h) <= 0 or x + w > image.shape[1] or y + h > image.shape[0]:
                raise ValueError("Region outside the captured crop")
            target = truth[y:y+h, x:x+w].astype(np.float64)
            delta = image[y:y+h, x:x+w].astype(np.float64) - target
            errors[name].append(delta)
            row["regions"][name] = {
                "rmse": float(np.sqrt(np.mean(delta * delta))),
                "relative_l2": float(np.linalg.norm(delta) / max(np.linalg.norm(target), 1e-12)),
                "signed_rgb_bias": delta.mean(axis=(0, 1)).tolist(),
            }
        rows.append(row)
    summaries = {}
    for name, values in errors.items():
        stack = np.stack(values)
        rmses = [row["regions"][name]["rmse"] for row in rows]
        summaries[name] = {
            "median_frame_rmse": float(np.median(rmses)),
            "max_frame_rmse": float(max(rmses)),
            "residual_change_rms": float(np.sqrt(np.mean(np.diff(stack, axis=0) ** 2))),
            "region_in_crop": regions[name],
        }
    return {"frames": frames, "comparisons": rows, "regions": summaries,
            "scope": "Consecutive screen-space residual changes against independently integrated matching poses; no warping or display transform. Finite references, not exact truth or motion-compensated flicker."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("protocol", type=Path)
    parser.add_argument("reference_prefix", help="Append FRAME.json to this prefix")
    parser.add_argument("candidate_directory", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    frames = protocol["frames"]
    if not isinstance(frames, list) or any(type(f) is not int for f in frames) or len(set(frames)) != len(frames):
        raise ValueError("Protocol frames must be distinct integers")
    references = {f: Path(f"{args.reference_prefix}{f}.json") for f in frames}
    candidates = {f: args.candidate_directory / f"candidate-{f:04d}.json" for f in frames}
    result = compare_motion(references, candidates, protocol["regions"])
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(result["regions"], indent=2))


if __name__ == "__main__":
    main()
