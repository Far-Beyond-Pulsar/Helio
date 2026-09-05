# Hierarchical Light-Field Sampling

HLFS shades the deferred GBuffer using visibility-guided light sampling. Its
output is linear HDR; the render graph supplies tone mapping and antialiasing.

## Frame stages

1. A 64×64 screen tile cull builds coarse light lists. An 8×8 depth-aware cull
   refines those lists. Overflow falls back to the complete light population,
   so exceeding a list's capacity does not remove illumination.
2. Each shading pixel evaluates a bounded set of candidates. Separate visible
   and hidden reservoirs combine the previous tile's observed lights with
   uniform discovery candidates. Reservoir and group probabilities are corrected
   in the lighting estimate. The hidden selection budget is normally capped at
   20%, relaxed to 50% on disocclusion. Directional proxy weights are limited
   relative to local lights without limiting the resulting lighting energy.
3. At most four selected lights receive shadow evaluation per shading pixel.
   Duplicate selections reuse visibility. Shadow maps and optional ray queries
   use the scene's existing resources. Each tile retains the strongest observed
   light in each of 16 deduplication slots. Eight bounded contact-trace steps use
   current-frame 8×8 minimum depth followed by full-resolution depth; a confirmed
   contact occluder avoids the shadow-map lookup.
4. Separate demodulated diffuse and specular histories use geometry rejection,
   a 5×5 YCoCg variance clamp and distance-dependent history rejection. Geometry
   history stores octahedral normals, logarithmic depth and age in RGBA16F.
5. A sparse geometry-aware spatial filter restores material response at output
   resolution. Optional half-resolution shading rotates through each 2×2 block;
   unmatched thin surfaces use a bounded direct-sampling fallback.
   Exactly evaluated and low-variance full-resolution pixels skip this filter.

CPU recording does not traverse scene lights. GPU culling scales with light
count; candidate and shadow budgets per pixel are bounded. Total GPU frame time
is therefore **not constant in light count**. Intermediate textures and tile
lists are resolution-dependent and are reused between frames. External bind
groups are rebuilt when their actual GPU resource handles change.

`HlfsConfig` controls sampling and history. `Reference` evaluates every light
without denoising; `Unfiltered` exposes stochastic lighting before denoising.
`output_texture()` supports readback. `allocation_bytes()` reports requested
storage, excluding driver alignment and allocator overhead.

## Validation

Shader validation runs without a GPU:

```text
cargo test -p helio-pass-hlfs --lib
```

GPU regression tests are explicit so ordinary CI does not depend on a display
adapter. Set `HLFS_CAPTURE_DIR` to retain comparison images.

```text
cargo test -p helio-pass-hlfs --test gpu_hlfs -- --ignored --skip benchmark_gpu_light_scaling --nocapture
cargo test -p helio-pass-hlfs --test gpu_hlfs benchmark_gpu_light_scaling -- --ignored --nocapture
cargo run -p examples --bin indoor_cathedral_hlfs -- --capture target/hlfs-captures
```

Set `HLFS_REFERENCE=1` for the cathedral's matching full-light reference captures.
Set `HLFS_FXAA=1` to exercise the FXAA graph variant.
Measured results and retained images are in `docs/validation/hlfs/README.md`.

## Platform choices

The visible lists use bounded workgroup memory rather than requiring subgroup
operations. Point, spot and directional lights use the existing light layout;
there is no rectangular-area-light representation to attach quadrant masks or
barn-door culling to. Existing optional ray-query support remains available.

Lighting and moments use RGBA16F rather than packed R11G11B10F with separate
moment textures. This avoids stochastic storage rounding and keeps the shader
path portable, at a higher full-resolution memory cost. The old fixed 128 MiB
clip stacks are removed, but total allocation depends on shading resolution;
removing them does not guarantee lower total memory at every resolution.

Candidates retain nonzero probability rather than dropping a fixed set of dim
lights. Exposure affects importance without silently deleting their combined
energy. Failed half-resolution reconstruction shades the actual surface with
a bounded fallback instead of retaining unvalidated history.

The benchmark uses GPU timestamps surrounding HLFS on the render encoder.
It compares the stochastic path with the full-light reference at the same
resolution; that comparison does not establish speedup over older renderers or
end-to-end game performance.

The embedded scalar spatiotemporal blue-noise ranks are independently generated
by `scripts/generate_hlfs_noise.py`. NumPy is only needed to regenerate the asset;
the renderer has no runtime Python dependency and includes no third-party
noise texture.
