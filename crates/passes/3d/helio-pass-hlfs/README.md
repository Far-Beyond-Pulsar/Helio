# Hierarchical Light-Field Sampling

HLFS shades the deferred GBuffer using visibility-guided light sampling. Its
output is linear HDR; the render graph supplies tone mapping and antialiasing.

## Frame stages

1. A 64×64 screen tile cull builds coarse light lists. An 8×8 depth-aware cull
   refines those lists. Overflow falls back to the complete light population,
   so exceeding a list's capacity does not remove illumination.
2. Each shading pixel evaluates a bounded set of candidates. Separate visible
   and hidden reservoirs combine the previous tile's observed lights with
   stratified discovery candidates. Reservoir and group probabilities are corrected
   in the lighting estimate. The hidden selection budget is normally capped at
   20%, relaxed to 50% on disocclusion. Directional proxy weights are limited
   relative to local lights without limiting the resulting lighting energy.
3. At most four selected lights receive shadow evaluation per shading pixel.
   Duplicate selections reuse visibility. Shadow maps and optional ray queries
   use the scene's existing resources. Parallel workgroup selection gathers up
   to 16 explicit IDs from 64 deduplication scratch slots, then sorts them for
   binary-search membership. Visibility is binary; partial shadow coverage does
   not reduce a visible light's guiding weight. A current-frame minimum-depth
   pyramid supports cell-based traversal with pixel-depth confirmation. Rays
   adapt to projected length within the configured world distance and a hard
   96-iteration budget; misses continue to the shadow map.
4. Separate demodulated diffuse and specular histories use geometry rejection,
   a 5×5 YCoCg variance clamp and distance-dependent history rejection. Geometry
   history packs octahedral normals, logarithmic depth, age and both signals'
   luminance second moments into eight bytes. Specular demodulation uses the
   engine's analytic EnvBRDF fit, including roughness and view angle.
5. A sparse geometry-aware spatial filter runs at shading resolution and reuses
   the raw-lighting buffer. At least 80% visible-energy coverage reduces its
   footprint when temporal variance permits. Exactly evaluated and stable
   low-variance pixels bypass filtering.
6. Full-resolution composition chooses one geometry-matching bilinear neighbor
   with STBN. Optional half-resolution shading visits each 2x2 block. Unmatched
   surfaces reuse geometry-validated history without color rectification when
   lighting is unchanged; new geometry or changed lights use bounded fresh samples.

CPU recording does not traverse scene lights. GPU culling scales with light
count; candidate and shadow budgets per pixel are bounded. Total GPU frame time
is therefore **not constant in light count**. Intermediate textures and tile
lists are resolution-dependent and are reused between frames. External bind
groups are rebuilt when their actual GPU resource handles change.

`HlfsConfig` controls sampling and history. `Reference` evaluates every light
without denoising; `Unfiltered` exposes stochastic lighting before denoising.
`Confidence` displays the 80% energy-coverage flag.
`HlfsConfig::performance()` uses four samples per 2x2 block.
`preferred_output_format()` selects renderable R11G11B10 HDR or RGBA16F fallback.
`enable_timing()` and `timing_query()` expose six GPU stage durations.
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
cargo test -p helio-pass-hlfs --test gpu_hlfs -- --ignored --skip benchmark --nocapture --test-threads=1
cargo test -p helio-pass-hlfs --test gpu_hlfs benchmark_gpu_light_scaling -- --ignored --nocapture
cargo run -p examples --bin indoor_cathedral_hlfs -- --capture target/hlfs-captures
```

Set `HLFS_REFERENCE=1` for the cathedral's matching full-light reference captures.
Set `HLFS_PERFORMANCE=1` to capture the reduced-resolution preset.
Set `HLFS_FXAA=1` to exercise the FXAA graph variant.
Measured results and retained images are in `docs/validation/hlfs/README.md`.

## Platform choices

The visible lists use bounded workgroup memory rather than requiring subgroup
operations. Point, spot and directional lights use the existing light layout;
there is no rectangular-area-light representation to attach quadrant masks or
barn-door culling to. Existing optional ray-query support remains available.

Diffuse and specular use the R11G11B10 unsigned-float bit representation in
portable RG32Uint storage textures, one packed word per signal. Explicit decode avoids requiring native
R11G11B10 storage-image support. STBN stochastic rounding covers normal and
subnormal values. Packed RG32Uint metadata stores octahedral SNORM8 normals,
FP16 logarithmic depth, two unsigned E6M5 luminance second moments, age and energy confidence.
Six exponent bits cover squared HDR luminance without overflow. Linear first
moments are derived from each signal's RGB; spatial filtering uses relative
variance in the same linear domain. Raw signals and ping-pong
history total 40 bytes per shading pixel; the full-resolution output is separate.
At 1080p with RGBA16F output, full shading requests 109,380,120 bytes
(104.31 MiB), or 43,487,640 bytes (41.47 MiB) at half resolution. Native packed
HDR output saves another 8,294,400 bytes. The performance preset then requests
35,193,240 bytes (33.56 MiB / 35.19 decimal MB), including grids and both history
parities. The default keeps full shading resolution. The approximate 34 MB
proposal budget applies to reduced-resolution shading, not full shading.

Coarse and fine grids pack two 16-bit IDs per word. Populations above 65,535
use a full-width global fallback. Visible IDs remain 32-bit. A dedicated
small-population pipeline skips reservoir setup and shared sorting when every
light fits within the shadow-sample budget.

Very dim candidates are culled only when an exact unshadowed contribution bound
fits a population-divided error budget. Aggregate loss is bounded by 1e-5 per
pre-exposed channel and never exceeds 1e-5 after removing pre-exposure. Bright
specular responses and large populations of dim lights retain support. Failed
half-resolution reconstruction first validates reprojected geometry and light
generation before history reuse, then shades the actual surface when history
is unavailable. Neighbor ray reuse is not part of this pass.

The benchmark uses GPU timestamps surrounding HLFS on the render encoder.
It compares the stochastic path with the full-light reference at the same
resolution. The cathedral capture separately reports serialized CPU + GPU frame
latency, excluding image readback. That completed-frame latency includes CPU
submission and GPU waits; it is not an isolated GPU pass time or asynchronous
throughput measurement. The validation report also retains a matched capture
from the previous renderer.

The embedded scalar spatiotemporal blue-noise ranks are independently generated
by `scripts/generate_hlfs_noise.py`. NumPy is only needed to regenerate the asset;
the renderer has no runtime Python dependency and includes no third-party
noise texture.
