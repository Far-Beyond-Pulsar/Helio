"""Accumulate linear HDR samples and compare captures without tone mapping.

See LINEAR_HDR_REFERENCE.md for the producer contract. This module never renders,
resamples, smooths geometry, or treats a settled temporal resolve as ground truth.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


SCHEMA = "helio.linear_hdr.v1"
CAMERA_KEYS = ("origin_cell", "fraction", "forward", "right", "up", "near", "far")
LIGHT_KEYS = ("sun", "ambient", "planet_sky", "raytraced_sun", "voxel_shadows")


def stratified_jitter(index, side):
    if type(side) is not int or side < 1 or type(index) is not int or not 0 <= index < side * side:
        raise ValueError("Invalid stratified sample index/side")
    return [(index % side + 0.5) / side - 0.5,
            (index // side + 0.5) / side - 0.5]


def _require(mapping, keys, label):
    if not isinstance(mapping, dict) or any(k not in mapping for k in keys):
        raise ValueError(f"Missing required {label}: {keys}")


def read_capture(path):
    """Read a strictly described RGB32F capture; reject unavailable diagnostics."""
    path = Path(path)
    meta = json.loads(path.read_text(encoding="utf-8"))
    _require(meta, ("schema", "kind", "stage", "encoding", "width", "height", "data",
                    "context", "unresolved_pixels"), "capture fields")
    if meta["schema"] != SCHEMA or meta["encoding"] != "rgb32f_le":
        raise ValueError(f"Unsupported schema/encoding: {path}")
    if meta["stage"] not in ("pre_aa_linear", "post_aa_linear"):
        raise ValueError("Capture must precede tone mapping and display transfer")
    if meta["kind"] not in ("sample", "reference", "candidate"):
        raise ValueError("Unknown capture kind")
    if type(meta["unresolved_pixels"]) is not int or meta["unresolved_pixels"] != 0:
        raise ValueError("Unresolved diagnostics must be present and zero")
    for key in ("width", "height"):
        if type(meta[key]) is not int or meta[key] < 1:
            raise ValueError(f"Invalid {key}")
    context = meta["context"]
    _require(context, ("camera", "scene", "lighting", "render_size"), "context")
    _require(context["camera"], CAMERA_KEYS, "camera")
    _require(context["scene"], ("identity", "generator_revision"), "scene")
    _require(context["lighting"], LIGHT_KEYS, "lighting")
    if not isinstance(context["scene"]["identity"], str) or not context["scene"]["identity"]:
        raise ValueError("Scene identity must describe world/edit content, not only edit count")
    # Reject NaN/Inf also in camera and lighting metadata: NaN equality is unsafe.
    json.dumps(context, allow_nan=False)
    render_size = context["render_size"]
    if (not isinstance(render_size, list) or len(render_size) != 2
            or any(type(v) is not int or v < 1 for v in render_size)):
        raise ValueError("Invalid full render dimensions")
    crop = context.get("crop", [0, 0, *render_size])
    if (not isinstance(crop, list) or len(crop) != 4
            or any(type(v) is not int for v in crop)
            or crop[0] < 0 or crop[1] < 0
            or crop[2:] != [meta["width"], meta["height"]]
            or crop[0] + crop[2] > render_size[0]
            or crop[1] + crop[3] > render_size[1]):
        raise ValueError("Invalid native capture crop or out-of-bounds dimensions")
    data_path = path.parent / meta["data"]
    expected = meta["width"] * meta["height"] * 3 * 4
    if data_path.stat().st_size != expected:
        raise ValueError(f"RGB32F byte count mismatch: {data_path}")
    image = np.fromfile(data_path, dtype="<f4").reshape(meta["height"], meta["width"], 3)
    if not np.isfinite(image).all() or np.any(image < 0):
        raise ValueError(f"Nonfinite or negative linear radiance: {data_path}")
    return meta, image


def matching_context(a, b):
    """Exact producer metadata agreement; no camera tolerance or implicit defaults."""
    for key in ("width", "height", "context"):
        if a[key] != b[key]:
            raise ValueError(f"Mismatched {key}; captures cannot be compared")


def write_capture(path, meta, image):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data_path = path.with_suffix(".rgb32f")
    values = np.asarray(image, dtype="<f4")
    if not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("Output radiance is not representable as nonnegative RGB32F")
    values.tofile(data_path)
    result = dict(meta, schema=SCHEMA, encoding="rgb32f_le", data=data_path.name)
    path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return path


def accumulate(paths, output):
    """Average one complete stratified set using online float64 mean/variance."""
    control = None
    seen = set()
    sources = []
    mean = m2 = None
    side = None
    # Sorting is deterministic; sample indices, not filename/frame order, prove coverage.
    for path in sorted(map(Path, paths)):
        meta, image = read_capture(path)
        if meta["kind"] != "sample" or meta["stage"] != "pre_aa_linear":
            raise ValueError("References require individual pre-AA linear samples")
        _require(meta.get("sample"), ("index", "strata_side", "jitter"), "sample")
        sample = meta["sample"]
        jitter = stratified_jitter(sample["index"], sample["strata_side"])
        if sample["jitter"] != jitter:
            raise ValueError("Jitter does not match the declared stratum centre")
        if control is None:
            control, side = meta, sample["strata_side"]
            mean = np.zeros(image.shape, dtype=np.float64)
            m2 = np.zeros_like(mean)
        matching_context(control, meta)
        if side != sample["strata_side"] or sample["index"] in seen:
            raise ValueError("Mixed strata size or duplicate sample index")
        seen.add(sample["index"])
        delta = image.astype(np.float64) - mean
        mean += delta / len(seen)
        m2 += delta * (image - mean)
        sources.append({"metadata": str(path), "metadata_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                        "data_sha256": hashlib.sha256((path.parent / meta["data"]).read_bytes()).hexdigest()})
    if control is None or seen != set(range(side * side)):
        raise ValueError("A complete, nonempty stratified sample set is required")
    variance = m2 / max(1, len(seen) - 1)
    # Deterministic stratification is not IID: do not label variance/N a confidence interval.
    result = {key: control[key] for key in ("width", "height", "context", "unresolved_pixels")}
    result.update(kind="reference", stage="pre_aa_linear", samples=len(seen), strata_side=side,
                  sample_variance_mean=float(variance.mean()), sources=sources,
                  scope="Finite stratified linear-radiance estimate; not a convergence proof")
    return write_capture(output, result, mean)


def compare(reference_path, candidate_paths):
    reference, truth = read_capture(reference_path)
    if reference["kind"] != "reference" or reference["stage"] != "pre_aa_linear":
        raise ValueError("Expected an accumulated pre-AA reference")
    truth = truth.astype(np.float64)
    rows = []
    candidates = []
    for path in candidate_paths:
        meta, image = read_capture(path)
        if meta["kind"] not in ("candidate", "reference"):
            raise ValueError("Compare candidates or independently accumulated references")
        matching_context(reference, meta)
        values = image.astype(np.float64)
        delta = values - truth
        rows.append({"capture": str(path), "stage": meta["stage"],
                     "radiance_rmse": float(np.sqrt(np.mean(delta ** 2))),
                     "radiance_mae": float(np.mean(np.abs(delta))),
                     "radiance_max_error": float(np.max(np.abs(delta))),
                     "relative_l2": float(np.linalg.norm(delta) / max(np.linalg.norm(truth), 1e-12)),
                     "signed_rgb_bias": delta.mean(axis=(0, 1)).tolist()})
        candidates.append(values)
    result = {"reference": str(reference_path), "comparisons": rows,
              "scope": "Matching native pixel crop and full raster; no tone map, resampling, clipping, or exposure fit"}
    if len(candidates) > 1:
        stack = np.stack(candidates)
        result["fixed_context_sequence"] = {
            "count": len(candidates),
            "temporal_rms": float(np.sqrt(np.var(stack, axis=0).mean())),
            "ordered_sample_change_rms": float(np.sqrt(np.mean(np.diff(stack, axis=0) ** 2))),
            "cadence": "Input order; skipped frames can hide flicker"}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    a = commands.add_parser("accumulate")
    a.add_argument("directory", type=Path)
    a.add_argument("output", type=Path)
    c = commands.add_parser("compare")
    c.add_argument("reference", type=Path)
    c.add_argument("candidates", type=Path, nargs="+")
    args = parser.parse_args()
    if args.command == "accumulate":
        print(accumulate(args.directory.glob("sample-*.json"), args.output))
    else:
        print(json.dumps(compare(args.reference, args.candidates), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
