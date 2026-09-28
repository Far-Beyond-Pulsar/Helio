# High-resolution rendering performance

Measured optimizations to the default deferred graph, focused on how cost
scales from 1080p to 4K. Every change was compared frame by frame against the
build before it; all but one are bit-identical, and the one that is not is
explained below.

## How to reproduce

```sh
# Mesa lavapipe (software Vulkan) works on machines without a GPU.
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.json
cargo build --release -p examples --bin resolution_bench

# --no-ray-query: lavapipe's compiler crashes on the radiance-cascades
# ray-query pipeline. Add --pulsar-columns for the editor's SceneDB setup.
target/release/resolution_bench --no-ray-query --frames 7 --warmup 10 \
    --out bench --tag after [--pulsar-columns] [--billboards]

scripts/frame_diff.py bench/before bench/after          # pixel diff, exit code = failures
scripts/bench_compare.py bench/before bench/after --passes DofPass,Sky
```

`resolution_bench` renders three scenes at 1920×1080, 2560×1440 and
3840×2160 with the default graph at the `RendererConfig` default render
scale of 0.75 (so internal resolution is 1440×810, 1920×1080 and 2880×1620).
The scenes are:

- `fog_hall`: a colonnade with a shadowed volumetric sun, local fog, point
  lights, metallic and rough spheres, emissive fixtures (bloom) and
  alpha-blended glass.
- `cathedral_large`: the HLFS cathedral, about 416k triangles, many lights
  and stained glass.
- `sky`: an open floor under the sky.

Runs use a fixed frame delta, so the same build renders bit-identical frames
every time; two runs of the same binary diff to zero pixels in every scene.
Any difference after a change is therefore a real change.

The tool records the time for each frame and the graph profiler's per-pass
CPU and GPU times.

`--pulsar-columns` registers the SceneDB columns that Pulsar-Native's editor
registers before any rows exist: billboards, decals, water volumes and water
hitboxes. Several passes used to size their work to a buffer's existence, so
they paid full cost in scenes with none of that content. That setup is the
one the editor actually runs.

### Measurement caveats

- lavapipe runs "GPU" work on CPU cores. Its times track shader and fill work,
  which is what scales with resolution, but not the bandwidth behaviour of a
  real GPU. Absolute values are much higher than on hardware. Use them for
  before/after comparisons on the same machine.
- Frame totals on lavapipe vary by about ±10% between runs, so per-pass
  timings are the reliable signal for a single change. Timings are medians of
  5–7 measured frames after 10 warm-up frames.
- Profiling is compiled in by default (helio-core's `profiling` feature), and
  it disables render-pass chain fusion. Both builds measured here have it on,
  so the comparison is like for like.

## Optimizations

Summary (ranges across the scenes each change affects, lavapipe):

| optimization | measured | 1080p | 1440p | 4K | frame diff |
|---|---|---:|---:|---:|---|
| 1. DoF CoC/gather gated on DOF enabled | DofPass GPU ms | 55.4–83.5 → 8.64–9.18 | 91.0–126.4 → 14.4–16.8 | 207.8–259.7 → 32.5–36.3 | bit-identical |
| 2. Sky depth-tested after the G-buffer | Sky GPU ms | 12.0–12.6 → 7.48–8.17 | 18.9–19.4 → 9.10–10.3 | 36.4–38.3 → 16.7–19.2 | bit-identical |
| 3. WaterSim screen work skipped without live water (demo columns) | WaterSim GPU ms | 6.54–7.83 → 0.13–0.15 | 10.8–11.8 → 0.15–0.16 | 26.2–28.7 → 0.17–0.18 | blit resampling removed (≥72 dB), see §3 |
| 3. same, editor columns | WaterSim GPU ms | 218.8–224.0 → 79.5–109.0 | 307.8–312.1 → 84.0–94.5 | 561.6–572.4 → 92.4–99.3 | same as above |
| 4. Decals skipped with no live decal (editor columns) | DecalApply GPU ms | 217.2–233.2 → 0.12–0.20 | 392.0–405.2 → 0.16–0.17 | 884.9–905.2 → 0.14–0.19 | bit-identical |
| 5. Fog composite passes through a neutral grid | FogComposite GPU ms | 20.8–20.9 → 5.41–6.21 | 40.0–40.8 → 11.3–11.3 | 85.2–93.0 → 23.5–23.6 | bit-identical |
| 6. Host target clear skipped (render CPU) | render() CPU ms | 3.25–4.38 → 2.61–3.04 | 4.36–4.96 → 3.27–4.01 | 6.75–8.03 → 2.96–3.33 | bit-identical |
| 7. Billboards sized to live rows (editor columns) | Billboard GPU ms | 14.8–17.3 → 0.20–0.26 | 15.4–16.4 → 0.19–0.21 | 15.0–25.7 → 0.21–0.25 | bit-identical |

Each table gives the named pass's GPU time in ms, before → after, per output
resolution.

### 1. DoF: skip CoC and gather while depth of field is off

DofPass dispatched its half-resolution circle-of-confusion and 32-tap bokeh
gather every frame, even though DOF is off by default. In that case the
composite returns the sharp image without reading either result. A
one-thread compute pass now writes the indirect arguments from the same DOF
block the passes read, and zeroes them when `dof_aperture_shape < 0`.

Frame diff: bit-identical in all 9 scene/resolution runs.

| scene | 1080p | 1440p | 4K |
|---|---:|---:|---:|
| fog_hall | 55.4 → 9.18 (-83%) | 91.0 → 15.8 (-83%) | 207.8 → 32.5 (-84%) |
| cathedral_large | 83.5 → 8.64 (-90%) | 125.6 → 14.4 (-89%) | 259.7 → 36.3 (-86%) |
| sky | 65.8 → 8.84 (-87%) | 126.4 → 16.8 (-87%) | 254.1 → 32.8 (-87%) |

### 2. Sky: shade only pixels no geometry covered

This is the cross-pass overdraw case. SkyPass ran with the early passes and
shaded the atmosphere and cloud layer for every pixel of `pre_aa` before any
geometry existed. DeferredLightPass then overwrote every covered pixel.

The deferred graphs now add the sky after every opaque depth writer and
before lighting. `SkyPass::with_depth_test` draws the far-plane triangle with
a read-only `LessEqual` depth test, so the depth buffer acts as the "already
covered" mask and early-Z rejects covered pixels.

SSR and planar reflections sample `pre_aa` between the G-buffer and lighting,
and XR uses a multiview depth target, so those configurations keep the
original order. The radiance-cascades fallback also reads `pre_aa` early, but
nothing in the default graph consumes its output: no pass publishes
`rc_view`, so `has_rc_gi` is always 0.

Frame diff: bit-identical in all 9 runs. The open-sky scene saves less
because most of its pixels are sky.

| scene | 1080p | 1440p | 4K |
|---|---:|---:|---:|
| fog_hall | 12.6 → 8.17 (-35%) | 19.4 → 10.3 (-47%) | 36.4 → 19.2 (-47%) |
| cathedral_large | 12.0 → 7.48 (-38%) | 18.9 → 9.10 (-52%) | 38.3 → 16.7 (-56%) |
| sky | 12.0 → 11.6 (-4%) | 19.9 → 13.5 (-32%) | 37.1 → 30.2 (-19%) |

### 3. WaterSim: skip screen-space water when no volume is live

WaterSim blitted `pre_aa` into `water_output` at full resolution every frame
and republished it as `pre_aa`. With the editor's registered but empty volume
column, it also drew the surface and ran a full-screen underwater tint plus a
second full-resolution blit for all-zero rows.

A new asynchronous readback, `helio_core::SceneBufferLiveness`, reads the rows
only when SceneDB reports new contents or a reallocation. When every row is
zero, the pass skips the screen-space work and leaves `pre_aa` alone. Until
the answer arrives it behaves exactly as before, so newly added water is
never skipped. The fixed-size simulation and caustics keep running, so their
state is unchanged when water appears.

Frame diff: not bit-identical, and this is intended. The removed blit
resampled `pre_aa` through a linear sampler at interpolated UVs, a slight
filter on high-contrast detail: max 16–55/255 on under 0.01% of pixels, mean
error 0.002, PSNR ≥ 72 dB. Replacing that blit with an exact `textureLoad`
copy in the old code reproduces the new frames bit-for-bit in every scene and
resolution, so the difference is entirely the removed filtering error.

Default columns:

| scene | 1080p | 1440p | 4K |
|---|---:|---:|---:|
| fog_hall | 6.9 → 0.15 (-98%) | 11.8 → 0.16 (-99%) | 27.6 → 0.17 (-99%) |
| cathedral_large | 6.5 → 0.14 (-98%) | 10.8 → 0.16 (-98%) | 26.2 → 0.18 (-99%) |
| sky | 7.8 → 0.13 (-98%) | 10.9 → 0.15 (-99%) | 28.7 → 0.18 (-99%) |

Editor columns:

| scene | 1080p | 1440p | 4K |
|---|---:|---:|---:|
| fog_hall | 218.8 → 83.0 (-62%) | 312.1 → 86.3 (-72%) | 561.6 → 92.4 (-84%) |
| cathedral_large | 224.0 → 79.5 (-65%) | 307.8 → 84.0 (-73%) | 572.4 → 95.1 (-83%) |
| sky | 221.1 → 109.0 (-51%) | 309.3 → 94.5 (-69%) | 563.3 → 99.3 (-82%) |

The remaining editor-column cost is the fixed-size simulation of 8 volumes ×
3 cascades. See the follow-ups.

### 4. Decals: skip both full-screen passes while every decal row is empty

With the decal column registered, DecalCollect looped over `MAX_DECALS` rows
for every internal-resolution pixel. DecalApply then copied four G-buffer
targets back. With every row empty, the pair only round-trips the G-buffer
unchanged. The pass now uses `SceneBufferLiveness` and sets `decal_count` to 0
once the rows are known to be empty.

Frame diff: bit-identical in all 9 runs. The before and after runs both use
editor columns.

| scene | 1080p | 1440p | 4K |
|---|---:|---:|---:|
| fog_hall | 228.4 → 0.16 (-100%) | 392.0 → 0.17 (-100%) | 897.8 → 0.19 (-100%) |
| cathedral_large | 217.2 → 0.20 (-100%) | 393.3 → 0.17 (-100%) | 884.9 → 0.17 (-100%) |
| sky | 233.2 → 0.12 (-100%) | 405.2 → 0.16 (-100%) | 905.2 → 0.14 (-100%) |

### 5. Fog composite: pass through when the froxel grid is neutral

FogComposite did a cubic B-spline reconstruction per pixel (four trilinear 3D
taps, 32 texels) even with no participating medium. In that case the
integrated grid is uniformly (0,0,0,1) and the composite is the identity.
Fog classification already knew this, because it dispatches zero integration
groups. It now publishes a zero range in `fog_parameters`, which FogComposite
and TransparentPass already treat as pass-through.

Frame diff: bit-identical in all 9 runs. The fog hall is unchanged by design,
since it has a medium.

| scene | 1080p | 1440p | 4K |
|---|---:|---:|---:|
| fog_hall | 24.4 → 21.7 (-11%) | 39.2 → 40.8 (+4%) | 85.4 → 86.1 (+1%) |
| cathedral_large | 20.9 → 5.41 (-74%) | 40.8 → 11.3 (-72%) | 85.2 → 23.5 (-72%) |
| sky | 20.8 → 6.21 (-70%) | 40.0 → 11.3 (-72%) | 93.0 → 23.6 (-75%) |

### 6. Skip the host's target clear when a pass overwrites the whole target

`Renderer::submit_frame` cleared the full output target in its own submission
every frame. In the default graphs, DofPass's composite is the first pass to
write the target, and it clears the target and covers every pixel itself.
That made the host clear a full output-resolution write nobody could see,
about 33 MB per 4K frame.

On lavapipe, the next submission waited behind that clear, which is why
render-thread CPU time grew with resolution. The new
`RenderPass::initializes_target()` lets a pass declare this, and the renderer
then skips only the clear.

Frame diff: bit-identical in all 9 runs. The table shows median `render()`
CPU time in ms, which is now flat across resolutions.

| scene | 1080p | 1440p | 4K |
|---|---:|---:|---:|
| fog_hall | 4.4 → 2.73 (-38%) | 4.8 → 3.27 (-32%) | 7.2 → 3.28 (-55%) |
| cathedral_large | 3.2 → 2.61 (-20%) | 4.4 → 4.01 (-8%) | 8.0 → 2.96 (-63%) |
| sky | 4.0 → 3.04 (-24%) | 5.0 → 3.34 (-33%) | 6.8 → 3.33 (-51%) |

### 7. Billboards: draw the buffer's rows, none while all are empty

BillboardPass drew 65,536 instances whenever the billboard buffer existed.
The editor's buffer has 1,024 rows, so most instances indexed past its end,
where the result depends on the backend's bounds checking. The pass now draws
the buffer's row capacity, and nothing once `SceneBufferLiveness` finds every
row empty.

Frame diff: bit-identical in both setups. With `--billboards`, the captures
differ from the icon-free ones exactly where the icons are.

Editor columns, no billboards:

| scene | 1080p | 1440p | 4K |
|---|---:|---:|---:|
| fog_hall | 15.4 → 0.26 (-98%) | 15.4 → 0.21 (-99%) | 25.7 → 0.21 (-99%) |
| cathedral_large | 14.8 → 0.20 (-99%) | 15.4 → 0.19 (-99%) | 15.0 → 0.23 (-98%) |
| sky | 17.3 → 0.21 (-99%) | 16.4 → 0.21 (-99%) | 15.9 → 0.25 (-98%) |

Editor columns with a live icon over every light (`--billboards`):

| scene | 1080p | 1440p | 4K |
|---|---:|---:|---:|
| fog_hall | 15.4 → 0.67 (-96%) | 16.0 → 0.72 (-95%) | 15.8 → 0.79 (-95%) |

## Cumulative result

Original code (433d9e5) vs this branch. Medians of 7 frames after 10 warm-up frames. "GPU" is the
wait for the frame on lavapipe; "CPU" is `Renderer::render`. Frame differences against the original
come only from optimization 3; every other step was bit-identical.

**Demo column setup (only the scene's own columns registered)**

| scene | output | GPU before | GPU after | Δ | CPU before | CPU after | Δ |
|---|---|---:|---:|---:|---:|---:|---:|
| fog_hall | 1080p | 422 ms | 365 ms | -14% | 4.43 ms | 3.50 ms | -21% |
| fog_hall | 1440p | 646 ms | 520 ms | -20% | 4.67 ms | 2.78 ms | -40% |
| fog_hall | 4K | 1202 ms | 972 ms | -19% | 7.09 ms | 3.48 ms | -51% |
| cathedral_large | 1080p | 361 ms | 283 ms | -22% | 4.50 ms | 3.59 ms | -20% |
| cathedral_large | 1440p | 566 ms | 399 ms | -30% | 4.74 ms | 3.80 ms | -20% |
| cathedral_large | 4K | 1110 ms | 759 ms | -32% | 6.58 ms | 3.91 ms | -41% |
| sky | 1080p | 225 ms | 137 ms | -39% | 4.33 ms | 4.01 ms | -7% |
| sky | 1440p | 370 ms | 215 ms | -42% | 4.91 ms | 3.15 ms | -36% |
| sky | 4K | 781 ms | 441 ms | -43% | 7.26 ms | 3.64 ms | -50% |

4K ÷ 1080p frame-time ratio (2.25× the internal pixels): fog_hall 2.85× → 2.66×, cathedral_large 3.07× → 2.68×, sky 3.47× → 3.21×

**Editor column setup (`--pulsar-columns`)**

| scene | output | GPU before | GPU after | Δ | CPU before | CPU after | Δ |
|---|---|---:|---:|---:|---:|---:|---:|
| fog_hall | 1080p | 871 ms | 487 ms | -44% | 5.95 ms | 5.54 ms | -7% |
| fog_hall | 1440p | 1309 ms | 625 ms | -52% | 6.52 ms | 5.27 ms | -19% |
| fog_hall | 4K | 2622 ms | 1089 ms | -58% | 8.82 ms | 5.12 ms | -42% |
| cathedral_large | 1080p | 802 ms | 400 ms | -50% | 6.21 ms | 5.15 ms | -17% |
| cathedral_large | 1440p | 1281 ms | 518 ms | -60% | 7.06 ms | 5.56 ms | -21% |
| cathedral_large | 4K | 2513 ms | 863 ms | -66% | 8.30 ms | 4.78 ms | -42% |
| sky | 1080p | 703 ms | 243 ms | -65% | 6.39 ms | 5.42 ms | -15% |
| sky | 1440p | 1088 ms | 332 ms | -69% | 8.03 ms | 5.42 ms | -33% |
| sky | 4K | 2290 ms | 549 ms | -76% | 10.19 ms | 5.13 ms | -50% |

4K ÷ 1080p frame-time ratio (2.25× the internal pixels): fog_hall 3.01× → 2.24×, cathedral_large 3.13× → 2.16×, sky 3.26× → 2.26×

## Remaining bottlenecks and follow-ups

Ranked by measured 4K cost after the changes above.

- **DeferredLight (110–400 ms at 4K, lavapipe).** Every PCF tap samples both
  the dynamic and the static shadow atlas and takes the minimum. When one of
  them has no casters for a layer, that sample is always 1 and could be
  skipped exactly. This needs per-layer occupancy on the GPU, because the CPU
  shadow counts lag a frame behind. Also, sky pixels could be rejected by a
  depth test instead of `discard`.
- **PostProcess (90–112 ms at 4K).** Bloom composites five mips through
  4-tap B-spline upsamples at output resolution, 20 bilinear taps per pixel.
  A progressive upsample pyramid would cut this to about 4 taps, but it
  changes pixels slightly, so it needs a visual sign-off.
- **FXAA (40–54 ms at 4K).** This is expected work for the AA path.
- **HiZ min pyramid (~17 ms of HiZ's ~35 ms at 4K).** It is rebuilt every frame
  but only read by SSR (off by default) and WaterSim's screen path (only with
  live water). Building it on demand needs the consumers' demand before HiZ
  runs, which no pre-frame hook currently provides.
- **DoF composite (≈33 ms at 4K) and FogComposite (≈23 ms at 4K) when their
  effects are off.** Each is now a pure full-resolution copy. Removing the
  copies needs CPU knowledge of GPU-resolved post-process settings and fog
  state.
- **Editor columns: WaterSim simulation (80–110 ms, fixed size).** It still
  simulates 8 volumes × 3 cascades while every volume is empty. Skipping that
  would freeze the simulation state until water appears, which is a visible
  difference on the first frames of new water, so it was left as is.
- **Render-pass fusion is disabled whenever profiling is compiled in**, and
  it is compiled in by default (`helio-core` `profiling` feature), so shipped
  builds never fuse G-buffer, foliage and portal passes. This matters most on
  tile-based GPUs.
- **One-frame-late compute.** The compute encoder is submitted before the
  graphics encoder. DofPass's CoC/gather and PostProcess's exposure/bloom
  therefore read the previous frame's images and settings. This is a
  pre-existing latency behaviour, recorded here but not changed.

## Tests

Tests were run on lavapipe.

- New: `crates/helio-core/tests/scene_liveness.rs` covers unknown, empty and
  rewritten contents.
- Extended: `helio-pass-volumetric-fog`'s `physical_fog` now checks the
  published range both with media and after the media are removed.
- Passing: `helio-pass-volumetric-fog`, `helio-pass-transparent`,
  `helio-pass-postprocess`, `helio-pass-dof`, `helio-pass-sky`,
  `helio-pass-decal` and `helio`.
- Failing before these changes, with the same failures at the base commit:
  - `helio-core`'s `gpu_frame_timing`.
  - `helio-core`'s `wgsl_validation`, on `transparent.wgsl` and the
    `__PP_TAIL_VEC4__` placeholder in `volumetric_fog.wgsl`.
  - `helio-pass-water-sim`'s integration tests, which no longer compile
    against current APIs.
