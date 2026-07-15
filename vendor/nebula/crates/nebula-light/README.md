# nebula-light

GPU path-traced lightmap baking — direct illumination, multi-bounce global illumination, and area-light support.

---

## What is a lightmap?

A **lightmap** is a pre-computed texture that stores how much light hits every point on every static surface in a scene. At runtime the renderer samples this texture and multiplies it by the surface albedo — giving the appearance of full global illumination with essentially zero per-frame cost.

The trade-off is that lightmaps only work for **static geometry**. Moving objects cast no shadows into the baked map, and the light itself cannot move. In exchange you get studio-quality bounce lighting with a single texture fetch per pixel.

---

## How the bake works in practice

### Step 1 — Upload scene geometry

The baker packs every mesh's vertex positions, normals, primary UVs, and lightmap UVs into typed GPU storage buffers. All meshes are batched into a single flat vertex/index buffer so the compute shader can trace rays against the entire scene with one dispatch — no per-mesh draw calls.

### Step 2 — Upload lights

Each `LightSource` is converted to a GPU-friendly 48-byte struct:

| Kind | GPU representation |
|---|---|
| Directional | world-space direction vector |
| Point | position + range |
| Spot | position + direction + inner/outer cone angles |
| Area | centre + right + up + half-extents |

Lights with `bake_enabled = false` are silently skipped.

### Step 3 — Dispatch the compute shader

A 2-D compute shader (one thread per lightmap texel, 8×8 workgroups) runs the path-tracing kernel:

1. **Texel unprojection** — each thread maps its `(x, y)` texel coordinate back to a world-space position and normal via the mesh's lightmap UVs.
2. **Direct illumination** — for each light source, evaluate the analytic BRDF response and cast a shadow ray. Shadow hits mask the contribution.
3. **Multi-bounce GI** — for each indirect bounce, sample a hemisphere direction via cosine-weighted distribution. Trace a ray. If the ray hits a surface, accumulate its radiance and recurse (up to `bounce_count` times). Radiance accumulates into a ping-pong RGBA32F texture.
4. **Denoising** — an optional 3×3 Gaussian spatial filter smooths noise in low-sample-count bakes.

### Step 4 — Readback

The finished texture is copied from GPU memory to CPU via a staging buffer, then handed to the serializer.

---

## Atlas layout

Multiple meshes share one lightmap atlas. The baker tiles the atlas as a `⌈√N⌉ × ⌈√N⌉` grid, one equal-area cell per mesh. Each cell's UV offset and scale are stored in `AtlasRegion` so the runtime shader can sample the correct sub-region per mesh.

---

## Configuration

```rust
let config = LightmapConfig {
    resolution:          1024,   // Atlas side (px); power of 2
    samples_per_texel:   64,     // Path-tracing samples (quality vs. speed)
    bounce_count:        2,      // 0 = direct only; 2–4 = typical production
    max_ray_distance:    1000.0, // World units
    denoise:             true,
    hdr_output:          true,   // false = RGBA16F (half the memory)
    area_light_samples:  16,
    ..Default::default()
};
```

**Presets:**

| Preset | Resolution | Samples | Bounces | Use case |
|---|---|---|---|---|
| `fast()` | 512 | 8 | 1 | Quick in-editor preview |
| `default()` | 1024 | 64 | 2 | Standard production |
| `ultra()` | 4096 | 512 | 4 | Final cinematic quality |

---

## How baking improves performance

Without baking, every frame would need to:
- Fire dozens of shadow rays per pixel per light.
- Accumulate multiple bounces of indirect light.
- Run this for every pixel on screen.

A high-end GPU can trace ~1–2 billion rays per second. A typical 1080p frame has ~2 million pixels; at 64 samples per pixel that is 128 million rays **per frame** — over budget by orders of magnitude for real-time rendering.

With baking, those 128 million rays run **once** during the bake (taking a few seconds on a modern GPU). The runtime replaces the entire path-tracing computation with a single bilinear texture fetch per pixel, which costs a handful of nanoseconds. The rendered image looks identical.

---

## Output

```rust
pub struct LightmapOutput {
    pub width:        u32,       // Atlas width (px)
    pub height:       u32,       // Atlas height (px)
    pub channels:     u32,       // 4 (RGBA)
    pub is_f32:       bool,      // true = RGBA32F, false = RGBA16F
    pub texels:       Vec<u8>,   // Raw texel bytes, row-major
    pub atlas_regions: Vec<AtlasRegion>, // Per-mesh UV offset/scale
    pub config_json:  String,    // Reproducibility metadata
}
```

---

## License

MIT OR Apache-2.0
