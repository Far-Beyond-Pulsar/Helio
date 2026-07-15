# nebula-visibility

**Potentially-visible set (PVS) baking** via GPU occlusion testing.

---

## What is a PVS?

A **Potentially Visible Set** is a pre-computed lookup table that answers the question: *"From cell A, which other cells in the scene could possibly be visible?"*

The scene is divided into a regular 3-D grid of cubic cells. For each pair of cells the baker determines whether a line of sight exists between them (considering all scene geometry). The result is stored as a **bit matrix**: cell A's row has one bit per cell — `1` if potentially visible, `0` if definitely occluded.

At runtime, when the camera is in cell A, the renderer reads A's row of bits and skips drawing any object whose cell has a `0` bit. Objects behind walls, below floors, or in completely separate rooms are **never submitted to the GPU** — not rendered, not even culled by the GPU; they don't exist for that frame.

This is one of the oldest and most effective broad-phase culling techniques in game development, and it is almost entirely free at runtime.

---

## How the bake works in practice

### Step 1 — Grid construction

The baker computes the scene's world-space AABB from all mesh positions (transformed by their world matrices), expands it by one cell in each direction to avoid edge cases, then divides it into a 3-D grid of cells with side length `config.cell_size`.

A 100 m × 50 m × 30 m level with 3 m cells produces a `35 × 18 × 11` grid ≈ 6 930 cells. The bit matrix for this is `6 930 × 6 930` bits ≈ 6 MB — a small, fast-loading asset.

### Step 2 — GPU ray casting

A compute shader fires `config.ray_budget` randomly distributed rays from each cell's centre point. For each ray:

1. The direction is sampled from a uniform sphere distribution (not hemisphere — visibility is symmetric but the test is directional to avoid biases).
2. The ray is tested against all scene triangles using Möller–Trumbore intersection — the same kernel as the AO and lightmap bakers.
3. If the ray reaches the centre of any target cell without hitting geometry, the corresponding bit is set atomically in the output buffer.

Each cell is one thread; the thread fires all `ray_budget` rays for its cell independently. The dispatch is `⌈cell_count / 64⌉` workgroups (1-D, 64 threads per workgroup).

### Step 3 — Conservative dilation (optional)

After the GPU pass, an optional CPU pass expands the visible set by one cell in each of the six axis-aligned directions. Any cell visible from a neighbour of A is also marked visible from A.

This eliminates **false negatives** — cases where a valid line of sight was missed because no ray happened to thread through a narrow gap. The PVS grows slightly larger (a few percent of bits may flip from `0` to `1`), but the guarantee becomes: **nothing that is truly visible will ever be culled**. Over-culling (drawing something that cannot actually be seen) is wasted GPU work; under-culling (not drawing something that *can* be seen, causing pop-in) is a visual artefact and typically unacceptable.

### Step 4 — Bit packing

The binary visibility matrix is stored as a flat `Vec<u64>` with `cell_count × words_per_cell` words, where `words_per_cell = ⌈cell_count / 64⌉`. Each bit corresponds to one cell-pair. Row-major ordering means the entire row for cell A is a contiguous run of `words_per_cell` words — a cache-friendly layout for the runtime bit-test inner loop.

---

## Configuration

```rust
let config = PvsConfig {
    cell_size:            3.0,   // World units per cell (smaller = finer, more memory)
    ray_budget:           256,   // Rays per source cell
    conservative:         true,  // Apply conservative dilation pass
    visibility_threshold: 1,     // Minimum rays to count as visible
    max_ray_distance:     500.0, // World units — rays beyond this are treated as misses
};
```

**Presets:**

| Preset | Cell size | Rays | Use case |
|---|---|---|---|
| `fast()` | 8 m | 32 | Quick preview; large open levels |
| `default()` | 3 m | 256 | Standard indoor levels |
| `ultra()` | 1.5 m | 2048 | Fine-grained indoor + outdoor scenes |

---

## How baking improves performance

PVS culling is uniquely effective because it operates **before the GPU sees any geometry at all**. The typical rendering pipeline is:

1. CPU frustum cull — reject objects outside the view frustum.
2. GPU occlusion cull — reject objects hidden behind other objects.
3. Draw calls for surviving objects.

Step 2 (GPU occlusion culling) is itself expensive and introduces latency. PVS replaces most of step 2 with a bit lookup — one `u64` read and a shift-and-mask — at essentially zero cost.

In a dense indoor level (corridors, rooms, multiple floors), PVS typically culls **70–95% of the scene** before any GPU work begins. Draw call counts drop by the same proportion, GPU memory bandwidth drops, and frame times improve dramatically — all with zero change to visual output.

---

## Output

```rust
pub struct PvsOutput {
    pub world_min:      [f32; 3],
    pub world_max:      [f32; 3],
    pub grid_dims:      [u32; 3],   // Cells per axis (X, Y, Z)
    pub cell_size:      f32,
    pub cell_count:     u32,
    pub words_per_cell: u32,        // u64 words per row in the bit matrix
    pub bits:           Vec<u64>,   // Flat bit matrix, row-major
    pub config_json:    String,
}
```

To query at runtime: `bits[cell_a * words_per_cell + cell_b / 64] >> (cell_b % 64) & 1`.

---

## License

MIT OR Apache-2.0
