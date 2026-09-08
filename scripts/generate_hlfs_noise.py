"""Generate Helio's scalar spatiotemporal blue-noise ranks (no external assets).

Uses the published void-and-cluster principle with separable interactions:
points interact spatially only within their frame, and temporally only at the
same pixel. This script is an independent implementation, not NVIDIA SDK code.
NumPy is a build-time authoring tool only; the renderer embeds the raw R8 data.
"""
from pathlib import Path
import hashlib
import numpy as np

SIZE = 32
FRAMES = 32
SEED = 48


def generate():
    rng = np.random.default_rng(SEED)
    d = np.minimum(np.arange(SIZE), SIZE - np.arange(SIZE))
    spatial = np.exp(-(d[:, None] ** 2 + d[None, :] ** 2) / (2 * 1.5**2))
    time = np.exp(-(d**2) / (2 * 1.5**2))
    time[0] = 0  # self interaction is already present in the spatial kernel
    shifts = np.array([[np.roll(spatial, (y, x), (0, 1)) for x in range(SIZE)] for y in range(SIZE)])
    time_shifts = np.array([np.roll(time, t) for t in range(FRAMES)])
    mask = np.zeros((FRAMES, SIZE, SIZE), dtype=bool)
    energy = np.zeros_like(mask, dtype=np.float64)

    def update(index, present):
        t, y, x = np.unravel_index(index, mask.shape)
        sign = 1 if present else -1
        mask[t, y, x] = present
        energy[t] += sign * shifts[y, x]
        energy[:, y, x] += sign * time_shifts[t]

    def cluster():
        return np.argmax(np.where(mask, energy, -np.inf))

    def void():
        return np.argmin(np.where(mask, np.inf, energy))

    initial_count = mask.size // 10
    for i in rng.choice(mask.size, initial_count, replace=False):
        update(i, True)
    for _ in range(mask.size):
        old = cluster()
        update(old, False)
        new = void()
        update(new, True)
        if new == old:
            break
    initial_mask, initial_energy = mask.copy(), energy.copy()
    ranks = np.zeros(mask.shape, dtype=np.uint32)
    for rank in range(initial_count - 1, -1, -1):
        i = cluster()
        ranks.flat[i] = rank
        update(i, False)
    mask[:], energy[:] = initial_mask, initial_energy
    for rank in range(initial_count, mask.size):
        i = void()
        ranks.flat[i] = rank
        update(i, True)
    return (ranks * 256 // mask.size).astype(np.uint8)


if __name__ == "__main__":
    ranks = generate()
    path = Path(__file__).resolve().parents[1] / "crates/passes/3d/helio-pass-hlfs/assets/stbn_32x32x32.r8"
    path.parent.mkdir(exist_ok=True)
    path.write_bytes(ranks.tobytes())
    assert np.all(np.bincount(ranks.ravel(), minlength=256) == ranks.size // 256)
    print(f"{path}: {ranks.size} bytes; sha256={hashlib.sha256(path.read_bytes()).hexdigest()}")
