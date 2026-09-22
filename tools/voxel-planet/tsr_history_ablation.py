"""Approximate frozen-scene TSR colour-history controls from real HDR captures.

This is not an engine measurement: it selects the nearest captured stratum to
each R1/R2 jitter and assumes geometrically valid, exactly aligned history.
It isolates colour clipping and its blend-weight adjustment. The variance
mode is a proposed control, not a validated renderer implementation.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from linear_hdr_reference import matching_context, read_capture


def tonemap(values):
    return values / (1 + values.max(axis=2, keepdims=True))


def ycocg(values):
    red, green, blue = np.moveaxis(values, 2, 0)
    return np.stack((red * .25 + green * .5 + blue * .25,
                     (red - blue) * .5,
                     -red * .25 + green * .5 - blue * .25), axis=2)


def rgb(values):
    y, co, cg = np.moveaxis(values, 2, 0)
    return np.stack((y + co - cg, y + cg, y - co - cg), axis=2)


def run(samples, reference, frames=304, warmup=240):
    reference_meta, truth = read_capture(reference)
    side = reference_meta["strata_side"]
    images = [None] * (side * side)
    for path in sorted(Path(samples).glob("sample-*.json")):
        meta, image = read_capture(path)
        matching_context(reference_meta, meta)
        sample = meta["sample"]
        index = sample["index"]
        if sample["strata_side"] != side or not 0 <= index < len(images) or images[index] is not None:
            raise ValueError("Mixed, duplicate, or out-of-range strata")
        images[index] = image
    if any(image is None for image in images):
        raise ValueError("Complete captured strata required")
    if not 0 <= warmup < frames:
        raise ValueError("Warmup must precede the final frame")
    images = np.stack(images)
    height, width = truth.shape[:2]
    if min(height, width) <= 4:
        raise ValueError("Crop needs a two-pixel halo for the 5x5 neighbourhood")
    truth = truth[2:-2, 2:-2].astype(np.float64)
    modes = ("clip_and_push", "push_only", "neither", "variance_widen")
    states = {mode: None for mode in modes}
    outputs = {mode: [] for mode in modes}
    mean = second = None
    temporal_factor = (1 - np.exp(-1 / 60 * 4)) * .5

    for frame in range(frames):
        # Matches libhelio::temporal::r1_r2_jitter, then chooses an approximate
        # ray sample from the finite recorded stratum grid.
        jitter = np.array([((frame * .7548776662466927 + .5) % 1) - .5,
                           ((frame * .5698402905980539 + .5) % 1) - .5], dtype=np.float32)
        sx, sy = np.clip(np.floor((jitter + .5) * side).astype(int), 0, side - 1)
        image = images[sy * side + sx].astype(np.float64)
        x, y = jitter[0], -jitter[1]
        ix, iy = int(np.floor(x)), int(np.floor(y))
        fx, fy = x - ix, y - iy
        current = sum(
            image[2 + iy + dy:height - 2 + iy + dy,
                  2 + ix + dx:width - 2 + ix + dx]
            * (fx if dx else 1 - fx) * (fy if dy else 1 - fy)
            for dy in range(2) for dx in range(2)
        )
        current_tm = ycocg(tonemap(current))
        window = np.lib.stride_tricks.sliding_window_view(ycocg(tonemap(image)), (5, 5), axis=(0, 1))
        lo, hi = window.min(axis=(-1, -2)), window.max(axis=(-1, -2))
        sigma = np.zeros_like(current_tm) if mean is None else np.sqrt(np.maximum(second - mean * mean, 0))
        for mode in modes:
            if states[mode] is None:
                output = current
            else:
                history = ycocg(tonemap(states[mode]))
                lower, upper = (lo - 3 * sigma, hi + 3 * sigma) if mode == "variance_widen" else (lo, hi)
                bounded = np.clip(history, lower, upper)
                push = 0 if mode == "neither" else np.clip(np.linalg.norm(history - bounded, axis=2, keepdims=True) * 8, 0, .5)
                beta = (.04 + push) * (1 - temporal_factor) + .5 * temporal_factor
                retained = bounded if mode in ("clip_and_push", "variance_widen") else history
                result = np.clip(rgb(retained * (1 - beta) + current_tm * beta), 0, 1)
                output = result / (1 - result.max(axis=2, keepdims=True) + 1e-8)
            states[mode] = output.astype(np.float16).astype(np.float64)
            if frame >= warmup:
                outputs[mode].append(states[mode])
        # Fixed, colour-independent moment weights; parameters frozen before
        # evaluating the proposed variance control (3 sigma, 32-sample cap).
        weight = 1 / min(frame + 1, 32)
        mean = current_tm if mean is None else mean * (1 - weight) + current_tm * weight
        second = current_tm * current_tm if second is None else second * (1 - weight) + current_tm * current_tm * weight
        mean, second = mean.astype(np.float32).astype(np.float64), second.astype(np.float32).astype(np.float64)

    result = {"scope": __doc__.strip(), "samples": str(samples), "reference": str(reference),
              "frames": frames, "warmup": warmup, "strata_side": side,
              "variance_sigma": 3, "moment_sample_cap": 32, "modes": {}}
    for mode, captures in outputs.items():
        values = np.stack(captures)
        delta = values - truth
        result["modes"][mode] = {
            "median_rmse": float(np.median(np.sqrt(np.mean(delta * delta, axis=(1, 2, 3))))),
            "mean_relative_l2": float(np.linalg.norm(values.mean(axis=0) - truth) / np.linalg.norm(truth)),
            "temporal_rms": float(np.sqrt(values.var(axis=0).mean())),
            "signed_rgb_bias": delta.mean(axis=(0, 1, 2)).tolist(),
        }
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("samples", type=Path)
    parser.add_argument("reference", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = run(args.samples, args.reference)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps(result["modes"], indent=2))
