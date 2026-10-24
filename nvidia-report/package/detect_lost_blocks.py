"""Finds captures with lost G-buffer blocks (NVIDIA bug 6939141).

Usage: python detect_lost_blocks.py <capture folder> [--noise]

A lost block holds the G-buffer's clear, which lighting draws as the sky's
haze: a flat 8x8 block, aligned to the 8x8 grid, whose colour differs from
the settled reference (stress_descent_NN_ref.png) by more than the
reference's own frame-to-frame noise (against _ref2.png) and more than the
blocks around it differ. A capture with more than THRESHOLD of its blocks
lost is affected. `--noise` scores each _ref2 against its _ref instead (two
clean frames), the detector's false-positive floor.
Needs numpy and Pillow.
"""
import glob
import os
import sys

import numpy as np
from PIL import Image

THRESHOLD = 0.002


def load(path):
    a = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255
    h, w = a.shape[0] // 8 * 8, a.shape[1] // 8 * 8
    return a[:h, :w].reshape(h // 8, 8, w // 8, 8, 3)


def lost_share(capture, reference, second=None):
    c, r = load(capture), load(reference)
    mc, mr = c.mean(axis=(1, 3)), r.mean(axis=(1, 3))
    flat = c.std(axis=(1, 3)).max(axis=-1) < 0.01
    off = np.abs(mc - mr).max(axis=-1)
    noise = np.abs(mr - load(second).mean(axis=(1, 3))).max(axis=-1) if second else np.zeros_like(off)
    # The blocks around each one: a lost block stands out of its
    # neighbourhood (whole regions that moved or changed do not).
    pad = np.pad(off, 1, mode="edge")
    around = np.median(np.stack([pad[1 + dy:pad.shape[0] - 1 + dy, 1 + dx:pad.shape[1] - 1 + dx] for dy in (-1, 0, 1) for dx in (-1, 0, 1) if dy or dx]), axis=0)
    lost = flat & (off > np.maximum(4 * noise, 0.015)) & (off > 3 * around + 0.008)
    return float(lost.mean())


def main():
    folder = sys.argv[1] if len(sys.argv) > 1 else "out_vulkan"
    noise_mode = "--noise" in sys.argv
    results = []
    for capture in sorted(glob.glob(os.path.join(folder, "stress_descent_[0-9][0-9].png"))):
        ref = capture[:-4] + "_ref.png"
        ref2 = capture[:-4] + "_ref2.png"
        if not os.path.exists(ref):
            continue
        second = ref2 if os.path.exists(ref2) else None
        if noise_mode:
            if second:
                results.append((lost_share(ref2, ref), os.path.basename(ref2)))
        else:
            results.append((lost_share(capture, ref, second), os.path.basename(capture)))
    affected = [r for r in results if r[0] > THRESHOLD]
    top = max(results) if results else (0.0, "-")
    print(f"{len(results)} captures, {len(affected)} affected (highest {top[0] * 100:.3f}% in {top[1]})")
    for share, name in sorted(affected, reverse=True):
        print(f"  {name}: {share * 100:.2f}% of 8x8 blocks lost")


if __name__ == "__main__":
    main()
