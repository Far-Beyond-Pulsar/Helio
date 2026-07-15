# nebula-ao

GPU hemisphere-sampled **ambient occlusion** baking.

---

## What is ambient occlusion?

**Ambient occlusion (AO)** answers a simple question for every point on a surface: *how much of the surrounding hemisphere is blocked by nearby geometry?*

A crease between two walls scores near 0 (almost fully occluded). An open rooftop scores near 1 (nothing blocking the sky). In practice AO is multiplied into the base lighting to darken corners, crevices, and contact points, which greatly increases the perceived depth and weight of static geometry without the cost of full global illumination.

AO is often baked alongside a lightmap. The lightmap captures directional color from real light sources; the AO map captures the ambient shadowing from the geometry itself. Together they produce convincing static lighting.

---

## How the bake works in practice

### Step 1 — Upload scene geometry

Same geometry upload strategy as `nebula-light`: all meshes are packed into flat vertex/index/mesh-info buffers and sent to the GPU once.

### Step 2 — Dispatch the AO compute shader

A 2-D compute shader (8×8 workgroups, one thread per texel) runs for every pixel in the output AO texture:

1. **Texel → world space** — the thread unpacks its `(x, y)` coordinate through the mesh's lightmap UVs to find the corresponding world-space position and surface normal.
2. **Hemisphere sampling** — `ray_count` directions are uniformly distributed over the cosine-weighted hemisphere aligned to the surface normal using a deterministic Hammersley/LCG sequence (seeded per-bake so results are reproducible).
3. **Ray casting** — each direction is tested against the scene using a Möller–Trumbore triangle intersection. If the ray hits geometry within `max_distance` world units, that direction counts as occluded.
4. **Accumulation** — the fraction of rays that *escaped* is written as a single `f32` value in `[0, 1]` into an R32F storage texture.
5. **Optional denoising** — a spatial filter smooths the output.

### Step 3 — Readback

The finished R32F texture is staged and read back to CPU.

---

## Configuration

```rust
let config = AoConfig {
    resolution:   1024,   // Texture side (px); shared with the lightmap if used together
    ray_count:    128,    // Rays per texel — more = less noise
    max_distance: 10.0,  // World units — rays beyond this are ignored
    bias:         0.001, // Nudges rays off the surface to avoid self-intersection
    denoise:      true,
};
```

**Presets:**

| Preset | Resolution | Rays | Use case |
|---|---|---|---|
| `fast()` | 512 | 16 | In-editor quick preview |
| `default()` | 1024 | 128 | Standard production |
| `ultra()` | 4096 | 512 | Final quality |

---

## How baking improves performance

A real-time AO estimate (e.g., SSAO — screen-space ambient occlusion) operates on depth-buffer samples, is view-dependent, misses off-screen occluders, and costs ~0.5–2 ms per frame even with aggressive downsampling.

Baked AO is **view-independent**, catches occluders anywhere in the scene, uses far more rays than SSAO could afford, and costs a single texture fetch at runtime. The visual quality is categorically higher.

The only limitation is that baked AO is static — it cannot respond to moved geometry. This is acceptable for walls, floors, furniture, and other static props that form the majority of a game level.

---

## Output

```rust
pub struct AoOutput {
    pub width:       u32,      // Texture width (px)
    pub height:      u32,      // Texture height (px)
    pub texels:      Vec<u8>,  // R32F, row-major (4 bytes per texel)
    pub config_json: String,
}
```

The single-channel R32F layout keeps memory use low — a 1024² AO map is 4 MB uncompressed, under 1 MB after zstd.

---

## License

MIT OR Apache-2.0
