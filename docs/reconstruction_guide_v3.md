# Helio Render V3 — AI Reconstruction Guide

This document is the complete specification for reconstructing `helio-render-v3` from scratch. Read it top-to-bottom before writing a single line of code.

**V3 vs V2:** Two breaking changes from V2:
1. **Feature System gone** — every capability that was previously a `Feature` is now a self-contained `RenderPass` that owns its own GPU resources. No `Feature` trait, no `FeatureRegistry`, no `FeatureContext`, no `PrepareContext`.
2. **HISM batching replaces pointer-identity batching** — meshes and materials are registered into a `HismRegistry` to get stable `HismHandle` IDs. Instances reference a `HismHandle` rather than raw `Arc<GpuMesh>` / `Arc<GpuMaterial>` pointers. The batch key is `HismHandle` — no pointer hashing, O(1) lookup, LOD-level aware.

---

## Table of Contents

1. [Philosophy and Performance Contract](#1-philosophy-and-performance-contract)
2. [Crate Layout and Dependencies](#2-crate-layout-and-dependencies)
3. [Public API Surface](#3-public-api-surface)
4. [Core Data Structures](#4-core-data-structures)
5. [GPU Resource Layout](#5-gpu-resource-layout)
6. [Vertex Buffer Layouts](#6-vertex-buffer-layouts)
7. [Bind Group Layouts](#7-bind-group-layouts)
8. [Render Graph Architecture](#8-render-graph-architecture)
9. [Frame Execution — CPU Pre-pass](#9-frame-execution--cpu-pre-pass)
10. [Render Pass Specifications](#10-render-pass-specifications)
11. [Shadow System](#11-shadow-system)
12. [PBR Lighting Model](#12-pbr-lighting-model)
13. [Radiance Cascades GI](#13-radiance-cascades-gi)
14. [Sky System](#14-sky-system)
15. [Anti-Aliasing Options](#15-anti-aliasing-options)
16. [Material System](#16-material-system)
17. [Scene and Camera API](#17-scene-and-camera-api)
18. [Debug Draw System](#18-debug-draw-system)
19. [GPU Profiler](#19-gpu-profiler)
20. [Shared State Ownership Map](#20-shared-state-ownership-map)
21. [Pipeline Cache and Variants](#21-pipeline-cache-and-variants)
22. [Reconstruction Checklist](#22-reconstruction-checklist)

---

## 1. Philosophy and Performance Contract

The renderer is a **deferred PBR renderer** built on a custom `wgpu` fork. Every architectural choice flows from five hard constraints:

| Constraint | Mechanism |
|---|---|
| **No mid-frame allocation** | `DrawCall` list filled into a pre-allocated `Vec`; instance buffers re-used across frames when unchanged |
| **No redundant GPU uploads** | Scene fingerprint (FNV-1a hash of mesh+light state) gates all CPU→GPU uploads; RC dynamic uniforms are delta-gated |
| **Minimal CPU-bound work** | Batch loop groups by `(mesh_ptr, mat_ptr)` pointer identity — no string comparisons |
| **Zero-overhead static frames** | Shadow bundles + GBuffer bundles skip re-encoding when geometry hash unchanged |
| **All times tracked** | `GpuProfiler` wraps every pass in timestamp queries; CPU timings measured with `std::time::Instant`; profiling gated behind `debug_printout` flag |

### Additional goals

- Up to 100k draws supported (targeting Unreal Engine-level scene complexity)
- Complex lighting setups with optional GI and AO, or just AO
- Passes are self-contained units — a pass owns its own textures, buffers, bind groups, and pipelines and creates them in its `new()` constructor

### wgpu Fork Requirements

The crate depends on `Far-Beyond-Pulsar/wgpu` at rev `fce5b80e`. It adds:
- `wgpu::Features::EXPERIMENTAL_RAY_QUERY` — hardware ray queries in WGSL via `rayQueryInitialize` / `rayQueryProceed`
- `wgpu::BufferUsages::BLAS_INPUT` — mesh buffers usable as BLAS geometry input
- `immediate_size` field on `PipelineLayoutDescriptor` — push-constant byte count (not used: portability)
- `create_tlas()` / `create_blas()` device methods

**Do not** request `immediate_size > 0` — the `IMMEDIATES` capability is not universally available. Layer index is passed via a per-slot uniform buffer instead.

---

## 2. Crate Layout and Dependencies

```
crates/helio-render-v3/
├── Cargo.toml
├── shaders/
│   ├── common/          # shared WGSL includes (lighting, brdf, shadows…)
│   └── passes/          # per-pass entry points (all shaders live here now)
│       ├── gbuffer.wgsl
│       ├── deferred_lighting.wgsl
│       ├── depth_prepass.wgsl
│       ├── shadow.wgsl
│       ├── transparent.wgsl
│       ├── sky.wgsl / sky_lut.wgsl
│       ├── rc_trace.wgsl
│       ├── billboard.wgsl
│       ├── fxaa.wgsl / smaa_*.wgsl / taa.wgsl
│       └── debug_draw.wgsl
└── src/
    ├── lib.rs            # re-exports, Error/Result types
    ├── renderer.rs       # Renderer, RendererConfig, frame loop
    ├── camera.rs         # Camera uniform struct
    ├── scene.rs          # Scene, HismInstance, SceneLight, SkyAtmosphere…
    ├── hism.rs            # HismHandle, HismRegistry, HismEntry, LodLevel
    ├── material.rs       # Material (CPU), GpuMaterial, MaterialUniform
    ├── mesh.rs           # PackedVertex, GpuMesh, DrawCall, GpuDrawCall
    ├── debug_draw.rs     # DebugShape, DebugDrawBatch
    ├── profiler.rs       # GpuProfiler, PassTiming
    ├── graph/            # RenderGraph, RenderPass trait, PassContext
    ├── pipeline/         # PipelineCache, PipelineVariant, ShaderConfig
    ├── resources/        # ResourceManager (bind group / texture lifecycle)
    └── passes/           # one file per render pass — each owns its GPU resources
        ├── shadow.rs           # ShadowPass { ShadowConfig }
        ├── depth_prepass.rs
        ├── gbuffer.rs
        ├── deferred_lighting.rs
        ├── transparent.rs
        ├── sky.rs
        ├── sky_lut.rs
        ├── radiance_cascades.rs  # RadianceCascadesPass { RcConfig }
        ├── billboards.rs         # BillboardsPass { BillboardConfig }
        ├── debug_draw.rs
        ├── smaa.rs / fxaa.rs / taa.rs
        └── indirect_dispatch.rs
```

Note: there is **no** `features/` directory in V3.

**Key runtime dependencies:**

| Crate | Usage |
|---|---|
| `wgpu` (fork) | GPU API |
| `glam 0.29` | All math, with `bytemuck` feature |
| `bytemuck 1.14` | Pod casts for GPU uploads |
| `petgraph 0.6` | Optional graph traversal |
| `rand 0.8` | SSAO noise kernel |
| `helio-live-portal` | Per-frame telemetry dashboard |
| `helio-core` | Shared `TextureData` |

---

## 3. Public API Surface

```rust
// lib.rs re-exports
pub use renderer::{Renderer, RendererConfig, ShadowConfig, RcConfig, BillboardConfig, BloomConfig, SsaoConfig};
pub use camera::Camera;
pub use profiler::{GpuProfiler, PassTiming};
pub use mesh::{GpuMesh, PackedVertex, DrawCall, GpuDrawCall};
pub use scene::{Scene, HismInstance, SceneLight, SkyAtmosphere, VolumetricClouds, Skylight};
pub use hism::{HismHandle, HismRegistry, HismEntry, LodLevel};
pub use material::{Material, GpuMaterial, TextureData};
pub use debug_draw::DebugShape;

pub enum Error {
    Pipeline(String),
    Graph(String),
    Resource(String),
    Shader(String),
    Wgpu(wgpu::Error),
}
pub type Result<T> = std::result::Result<T, Error>;
```

### `RendererConfig`

The single point of renderer configuration. No feature registration needed — optional capabilities are expressed as `Option<T>` fields.

```rust
pub struct RendererConfig {
    pub width: u32,
    pub height: u32,
    pub surface_format: wgpu::TextureFormat,
    pub anti_aliasing: AntiAliasingMode,
    pub shadows: Option<ShadowConfig>,
    pub radiance_cascades: Option<RcConfig>,
    pub billboards: Option<BillboardConfig>,
    pub bloom: Option<BloomConfig>,
    pub ssao: Option<SsaoConfig>,
    pub gpu_driven: bool,
    pub debug_printout: bool,   // gates GpuProfiler timestamp queries
}

pub struct ShadowConfig {
    pub max_shadow_lights: u32,  // default 40
    pub atlas_size: u32,         // default 2048
}

pub struct RcConfig {
    pub max_instances: u32,      // for TLAS — default 2048
}

pub struct BillboardConfig {
    pub max_instances: u32,      // default 1024
}

pub struct BloomConfig {
    pub threshold: f32,          // default 1.0
    pub intensity: f32,          // default 0.3
}

pub struct SsaoConfig {
    pub radius: f32,             // default 0.5
    pub bias: f32,               // default 0.025
    pub power: f32,              // default 2.0
    pub samples: u32,            // default 16
}
```

### Renderer Lifecycle

```rust
// 1. Build config
let config = RendererConfig {
    width, height,
    surface_format,
    anti_aliasing: AntiAliasingMode::Taa,
    shadows: Some(ShadowConfig::default()),
    radiance_cascades: Some(RcConfig::default()),
    billboards: None,
    bloom: Some(BloomConfig::default()),
    ssao: None,
    gpu_driven: false,
    debug_printout: false,
};

// 2. Construct renderer (device/queue come from wgpu adapter)
let renderer = Renderer::new(&device, &queue, config)?;

// 3. Each frame
renderer.render_scene(&device, &queue, &surface_view, &scene, delta_time)?;

// 4. Resize
renderer.resize(&device, &queue, new_width, new_height);
```

`Renderer::new()` inspects `RendererConfig`, instantiates only the passes that are requested, constructs each pass (which allocates its own GPU resources), builds the render graph, and derives the `ShaderConfig` for pipeline compilation. No second-phase feature registration.

---

## 4. Core Data Structures

### `Camera` — `#[repr(C)]`, Pod, **144 bytes**

Uploaded to `Group 0, Binding 0` every frame.

| Field | Type | Byte Offset |
|---|---|---|
| `view_proj` | `Mat4` | 0 |
| `position` | `Vec3` | 64 |
| `time` | `f32` | 76 |
| `view_proj_inv` | `Mat4` | 80 |

Constructor: `Camera::perspective(eye, target, up, fov_y, aspect, near, far, time)`.

The `view_proj_inv` is used by the CSM algorithm to unproject frustum corners to world space. `time` is fed to sky/cloud animation shaders. The world-forward vector for depth sorting is derived from near/far center-ray unprojection via `view_proj_inv` — do **not** extract it from the view matrix basis (prone to jitter artifacts).

---

### `GlobalsUniform` — `#[repr(C)]`, Pod, **48 bytes**

Uploaded to `Group 0, Binding 1` every frame.

| Field | Type | Notes |
|---|---|---|
| `frame` | `u32` | Monotonically increasing frame counter |
| `delta_time` | `f32` | Seconds since last frame |
| `light_count` | `u32` | Active lights this frame |
| `ambient_intensity` | `f32` | Sky ambient multiplier |
| `ambient_color` | `[f32;4]` | Estimated sky zenith colour (w unused) |
| `rc_world_min` | `[f32;4]` | RC probe grid AABB min (w unused) |
| `rc_world_max` | `[f32;4]` | RC probe grid AABB max (w unused) |
| `csm_splits` | `[f32;4]` | View-space distances for 4 CSM cascade boundaries: `[16, 80, 300, 1400]` |

---

### `PackedVertex` — `#[repr(C)]`, Pod, **32 bytes**

Slot 0 vertex buffer. Matches all geometry shaders.

| Location | Field | Type | Byte Offset | Encoding |
|---|---|---|---|---|
| 0 | `position` | `Float32x3` | 0 | Model-space XYZ |
| 1 | `bitangent_sign` | `Float32` | 12 | `+1.0` or `-1.0` |
| 2 | `tex_coords` | `Float32x2` | 16 | UV [0,1] |
| 3 | `normal` | `Uint32` | 24 | SNORM8×4 packed: `(x*127, y*127, z*127, 0)` |
| 4 | `tangent` | `Uint32` | 28 | SNORM8×4 packed: `(x*127, y*127, z*127, sign*127)` |

Vertices are always **model-space** (local). The instance buffer (Slot 1) holds model-to-world `Mat4` transforms. Do not pre-transform vertices into world space.

Construction: `PackedVertex::new(position, normal, tex_coords)` auto-computes tangent. `PackedVertex::new_with_tangent(…)` for explicit TBN.

---

### `HismHandle` — stable mesh+material registration key

```rust
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct HismHandle(pub u32);
```

A `HismHandle` is obtained by registering a `(GpuMesh, GpuMaterial)` pair into the `HismRegistry`. It is stable for the life of the registration — no raw pointer arithmetic, no realloc churn. The handle IS the batch key.

### `HismRegistry`

```rust
pub struct HismRegistry {
    entries: Vec<HismEntry>,   // indexed by HismHandle.0
}

pub struct HismEntry {
    pub mesh: Arc<GpuMesh>,
    pub material: Arc<GpuMaterial>,
    pub lod_levels: Vec<LodLevel>,  // optional LOD chain, index 0 = highest detail
}

pub struct LodLevel {
    pub mesh: Arc<GpuMesh>,
    pub screen_coverage_threshold: f32,  // switch to next LOD below this coverage
}
```

`HismRegistry::register(mesh, material) -> HismHandle` — appends a new entry and returns its index as a handle. Deduplication (same mesh+mat again) is left to the caller — the registry does not intern.

### `HismInstance` — the unit added to `Scene`

```rust
pub struct HismInstance {
    pub handle: HismHandle,
    pub transform: Mat4,
    pub lod_bias: i8,  // 0 = auto, positive = force coarser LOD
}
```

`Scene::objects` is now `Vec<HismInstance>`. `SceneObject` (mesh+material+transform triplet) is **removed** in V3.

### `DrawCall` — CPU-side per-draw record

| Field | Type | Notes |
|---|---|---|
| `hism_handle` | `HismHandle` | Batch identity key |
| `vertex_buffer` | `Arc<Buffer>` | Slot 0 — PackedVertex data |
| `index_buffer` | `Arc<Buffer>` | Uint32 indices |
| `index_count` | `u32` | |
| `vertex_count` | `u32` | |
| `material_bind_group` | `Arc<BindGroup>` | Group 1 material BG |
| `transparent_blend` | `bool` | Routes to transparent pass |
| `bounds_center` | `[f32;3]` | World-space AABB centre |
| `bounds_radius` | `f32` | Bounding sphere radius |
| `material_id` | `u32` | GPU-indirect path |
| `instance_buffer` | `Arc<Buffer>` | Slot 1 — mat4x4 transforms; always present |
| `instance_count` | `u32` | Number of instances |

`instance_buffer` is **never `Option`** in V3. Single-instance draws still get a one-entry instance buffer. Passes unconditionally bind slot 1.

**Instance buffer layout** — stride 64, `VertexStepMode::Instance`:

| Location | Format | Byte Offset |
|---|---|---|
| 5 | `Float32x4` | 0 — column 0 of model `mat4x4` |
| 6 | `Float32x4` | 16 — column 1 |
| 7 | `Float32x4` | 32 — column 2 |
| 8 | `Float32x4` | 48 — column 3 |

---

### `GpuLight` — `#[repr(C)]`, Pod, **64 bytes**

Stored in a `STORAGE` read-only buffer at Group 2 Binding 0. Maximum 2048 lights (`MAX_LIGHTS`). Four × 16-byte vec4 rows for std140 alignment.

```
[position.xyz, light_type]         — type: 0=dir, 1=point, 2=spot
[direction.xyz, range]
[color.xyz, intensity]
[cos_inner, cos_outer, _pad0, _pad1]
```

---

### `GpuShadowMatrix` — Pod, **64 bytes**

```rust
struct GpuShadowMatrix { mat: [f32; 16] }  // column-major Mat4
```

One per shadow atlas layer. `layer = light_idx * 6 + face`. Unused layers hold identity matrices.

---

### `MaterialUniform` — Pod, **48 bytes**, Group 1 Binding 0

| Field | Byte Offset |
|---|---|
| `base_color: [f32;4]` | 0 |
| `metallic: f32` | 16 |
| `roughness: f32` | 20 |
| `emissive_factor: f32` | 24 |
| `ao: f32` | 28 |
| `emissive_color: [f32;3]` | 32 |
| `alpha_cutoff: f32` | 44 |

---

### `SkyUniform` — Pod, **112 bytes**

| Field | Offset |
|---|---|
| `sun_direction: [f32;3]` | 0 |
| `sun_intensity: f32` | 12 |
| `rayleigh_scatter: [f32;3]` | 16 |
| `rayleigh_h_scale: f32` | 28 |
| `mie_scatter: f32` | 32 |
| `mie_h_scale: f32` | 36 |
| `mie_g: f32` | 40 |
| `sun_disk_cos: f32` | 44 |
| `earth_radius: f32` | 48 |
| `atm_radius: f32` | 52 |
| `exposure: f32` | 56 |
| `clouds_enabled: u32` | 60 |
| `cloud_coverage: f32` | 64 |
| `cloud_density: f32` | 68 |
| `cloud_base: f32` | 72 |
| `cloud_top: f32` | 76 |
| `cloud_wind_x: f32` | 80 |
| `cloud_wind_z: f32` | 84 |
| `cloud_speed: f32` | 88 |
| `time_sky: f32` | 92 |
| `skylight_intensity: f32` | 96 |
| `_pad0/1/2: f32` | 100–111 |

---

## 5. GPU Resource Layout

### Persistent buffers — created in `Renderer::new()` and shared via `Arc`

| Buffer | Format | Binding | Owner |
|---|---|---|---|
| `camera_buffer` | `Camera` (144 B) | Group 0, Binding 0 | Renderer |
| `globals_buffer` | `GlobalsUniform` (48 B) | Group 0, Binding 1 | Renderer |
| `light_buffer` | `GpuLight × MAX_LIGHTS` | Group 2, Binding 0 | Renderer |
| `shadow_matrix_buffer` | `GpuShadowMatrix × max_lights*6` | ShadowPass Group 0 B0 | ShadowPass |

`ShadowPass` creates `shadow_matrix_buffer` in its own `new()` using `max_shadow_lights` from `ShadowConfig`.

### Pass-owned textures (each pass creates its own in `new()`)

**Renderer-owned (resize-sensitive, shared across passes via `Arc<Mutex<>>`):**

| Name | Format | Usage |
|---|---|---|
| `depth_texture` | `Depth32Float` | `RENDER_ATTACHMENT \| TEXTURE_BINDING` |
| `gbuf_albedo` | `Rgba8Unorm` | `RENDER_ATTACHMENT \| TEXTURE_BINDING` |
| `gbuf_normal` | `Rgba16Float` | `RENDER_ATTACHMENT \| TEXTURE_BINDING` |
| `gbuf_orm` | `Rgba8Unorm` | `RENDER_ATTACHMENT \| TEXTURE_BINDING` |
| `gbuf_emissive` | `Rgba16Float` | `RENDER_ATTACHMENT \| TEXTURE_BINDING` |
| `pre_aa_color` *(when AA enabled)* | `surface_format` | `RENDER_ATTACHMENT \| TEXTURE_BINDING` |
| `velocity_texture` *(TAA)* | `Rg16Float` | `RENDER_ATTACHMENT \| TEXTURE_BINDING` |
| `taa_history_a/b` *(TAA)* | `Rgba16Float` | `RENDER_ATTACHMENT \| TEXTURE_BINDING` |

**ShadowPass-owned:**

| Name | Format | Notes |
|---|---|---|
| `shadow_atlas` | `Depth32Float` 2D-array | `2048×2048 × max_lights*6 layers` |

**SkyLutPass-owned:**

| Name | Format | Notes |
|---|---|---|
| `sky_lut` | `Rgba16Float` | `192×108` |
| `env_cube` | `Rgba8UnormSrgb` | `512×512, 6 faces` |

**RadianceCascadesPass-owned:**

| Name | Format | Notes |
|---|---|---|
| `rc_cascade[0..3]` | `Rgba16Float` | cascade atlas textures |
| `rc_hist_a/b[0..3]` | `Rgba16Float` | temporal history ping-pong |

### Shared samplers — created in `Renderer::new()`

| Name | Filter | Address | Comparison |
|---|---|---|---|
| `material_sampler` | LinearMip | Repeat | None |
| `shadow_sampler` | Nearest | ClampToEdge | LessEqual |
| `env_sampler` | Linear | ClampToEdge | None |
| `sky_sampler` | Linear | U=Repeat, V=ClampToEdge | None |

---

## 6. Vertex Buffer Layouts

### Standard geometry — Slot 0 + Slot 1

Used by: `GBufferPass`, `DepthPrepassPass`, `TransparentPass`, `ShadowPass`.

**Slot 0** (`PackedVertex`, stride=32, `VertexStepMode::Vertex`):

| Location | Format | Offset |
|---|---|---|
| 0 | `Float32x3` | 0 (position) |
| 1 | `Float32` | 12 (bitangent_sign) |
| 2 | `Float32x2` | 16 (tex_coords) |
| 3 | `Uint32` | 24 (normal SNORM8×4) |
| 4 | `Uint32` | 28 (tangent SNORM8×4) |

**Slot 1** (Instance transform, stride=64, `VertexStepMode::Instance`):

| Location | Format | Offset |
|---|---|---|
| 5 | `Float32x4` | 0 |
| 6 | `Float32x4` | 16 |
| 7 | `Float32x4` | 32 |
| 8 | `Float32x4` | 48 |

**Shadow pass** uses only Locations 0 and 2 from Slot 0 (depth-only, no normals/tangents needed).

### Billboard — Slot 0 + Slot 1

**Slot 0** (quad vertex, stride=16, Vertex):

| Location | Format | Offset |
|---|---|---|
| 0 | `Float32x2` | 0 (position [-1..1]) |
| 1 | `Float32x2` | 8 (uv [0..1]) |

**Slot 1** (`GpuBillboardInstance`, stride=32, Instance → locations 2–4)

### Debug draw — Slot 0 only

`DebugDrawVertex` stride=28:

| Location | Format | Offset |
|---|---|---|
| 0 | `Float32x3` | 0 (position) |
| 1 | `Float32x4` | 12 (color RGBA) |

---

## 7. Bind Group Layouts

### Group 0 — Global (all geometry passes)

```
Binding 0: UNIFORM, VERTEX|FRAGMENT — Camera (144 bytes)
Binding 1: UNIFORM, VERTEX|FRAGMENT — GlobalsUniform (48 bytes)
```

### Group 1 — Material (geometry passes)

```
Binding 0: UNIFORM, FRAGMENT           — MaterialUniform (48 bytes)
Binding 1: TEXTURE 2d Float, FRAGMENT  — base_color_texture (Rgba8UnormSrgb)
Binding 2: TEXTURE 2d Float, FRAGMENT  — normal_map (Rgba8Unorm)
Binding 3: SAMPLER Filtering           — material_sampler
Binding 4: TEXTURE 2d Float, FRAGMENT  — orm_texture (Rgba8Unorm)
Binding 5: TEXTURE 2d Float, FRAGMENT  — emissive_texture (Rgba8UnormSrgb)
```

### Group 2 — Lighting (geometry/deferred passes)

```
Binding 0: STORAGE read-only, VERTEX|FRAGMENT — GpuLight array
Binding 1: TEXTURE 2d-array Depth, FRAGMENT   — shadow_atlas
Binding 2: SAMPLER Comparison                  — shadow_sampler (LessEqual)
Binding 3: TEXTURE cube Float, FRAGMENT        — env_cube (IBL)
Binding 4: STORAGE read-only, VERTEX|FRAGMENT  — GpuShadowMatrix array
Binding 5: TEXTURE 2d Float, FRAGMENT          — rc_cascade0 (non-filterable)
Binding 6: SAMPLER Filtering                   — env_sampler
```

When shadows are disabled (`RendererConfig::shadows = None`), `shadow_atlas` is a 1×1×1 dummy depth texture and `shadow_sampler` still exists (binding layout must be identical across all pipelines that share Group 2).

When RC is disabled, `rc_cascade0` is a 1×1 dummy texture.

This "stub binding" policy keeps all Group 2 pipeline layouts identical regardless of which passes are active, avoiding pipeline proliferation.

### Group 1 — G-Buffer read (deferred lighting pass only)

```
Binding 0: TEXTURE 2d Float, FRAGMENT — gbuf_albedo   (Rgba8Unorm)
Binding 1: TEXTURE 2d Float, FRAGMENT — gbuf_normal   (Rgba16Float)
Binding 2: TEXTURE 2d Float, FRAGMENT — gbuf_orm      (Rgba8Unorm)
Binding 3: TEXTURE 2d Float, FRAGMENT — gbuf_emissive (Rgba16Float)
Binding 4: TEXTURE 2d Depth, FRAGMENT — gbuf_depth    (Depth32Float, DepthOnly aspect)
```

### Group 0 — Shadow (shadow pass only)

```
Binding 0: STORAGE read-only, VERTEX — GpuShadowMatrix array
Binding 1: UNIFORM, VERTEX           — shadow_layer_idx (u32, 4 bytes)
```

One `BindGroup` is pre-baked per atlas layer at `ShadowPass::new()`.

### Group 1 — Material in shadow pass

```
Binding 0: UNIFORM, FRAGMENT          — MaterialUniform
Binding 1: TEXTURE 2d Float, FRAGMENT — base_color_texture
Binding 3: SAMPLER Filtering          — material_sampler
```

### Group 0 — Sky / SkyLut

```
Binding 0: UNIFORM, VERTEX|FRAGMENT   — SkyUniform (112 bytes)
Binding 1: TEXTURE 2d Float, FRAGMENT — sky_lut (192×108, Rgba16Float)
Binding 2: SAMPLER Filtering          — sky_sampler
```

### TAA Group 0

```
Binding 0: TEXTURE 2d Float  — current frame
Binding 1: TEXTURE 2d Float  — history frame
Binding 2: TEXTURE 2d Float  — velocity (Rg16Float, non-filterable)
Binding 3: TEXTURE 2d Depth  — depth
Binding 4: SAMPLER Filtering
Binding 5: SAMPLER NonFiltering
Binding 6: UNIFORM           — TaaUniform { feedback_min, feedback_max, jitter_offset }
```

### RC Compute Group 0

```
Binding 0: texture_storage_2d<rgba16float, write>  — cascade_out
Binding 1: texture_2d<f32>                          — cascade_parent
Binding 2: UNIFORM  — RCDynamic { world_min, world_max, frame, light_count, _pad, sky_color }
Binding 3: UNIFORM  — CascadeStatic { cascade_index, probe_dim, dir_dim, t_max_bits,
                                       parent_probe_dim, parent_dir_dim, _pad }
Binding 4: acceleration structure — TLAS
Binding 5: STORAGE read — GpuLight array
Binding 6: texture_2d<f32>                          — cascade_history
Binding 7: texture_storage_2d<rgba16float, write>  — cascade_history_write
```

---

## 8. Render Graph Architecture

### `RenderGraph`

```rust
struct RenderGraph {
    passes: Vec<PassNode>,         // Box<dyn RenderPass> + resource annotations
    execution_order: Vec<usize>,   // Kahn's topological sort
    transient_resources: HashMap,
}
```

Topological sort runs **once at construction**. `ResourceHandle::named("name")` tokens are graph edges. Edges are constructed sequentially in pass-registration order to handle read-modify-write chains correctly (e.g. `TransparentPass` reads *and* writes `"color_target"` — it declares both).

### `RenderPass` Trait

```rust
pub trait RenderPass: Send + Sync {
    fn name(&self) -> &str;
    fn declare_resources(&self, builder: &mut PassResourceBuilder);
    fn execute(&mut self, ctx: &mut PassContext) -> Result<()>;
}
```

`PassContext` provides:
- `begin_render_pass(label, color_attachments, depth_attachment)`
- `begin_compute_pass(label)`
- `camera_position: Vec3`
- `scope_begin(label) / scope_end(token)` — timestamp injection

### Resource tokens and their owners

| Token | Written by | Read by |
|---|---|---|
| `"shadow_atlas"` | `ShadowPass` | `DeferredLightingPass` |
| `"sky_lut"` | `SkyLutPass` | `SkyPass`, `DeferredLightingPass` |
| `"sky_layer"` | `SkyPass` | ordering dep for `DeferredLightingPass` |
| `"depth"` | `DepthPrepassPass` | `GBufferPass` (LoadOp::Load) |
| `"gbuffer"` | `GBufferPass` | `DeferredLightingPass` |
| `"rc_cascade0"` | `RadianceCascadesPass` | `DeferredLightingPass` |
| `"color_target"` | `DeferredLightingPass` | `TransparentPass` (R+W), `BillboardsPass` (R+W), `DebugDrawPass` (R+W) |
| `"pre_aa_color"` | `DeferredLightingPass` *(AA enabled)* | AA passes |

---

## 9. Frame Execution — CPU Pre-pass

### Step 1 — Scene fingerprint

FNV-1a hash over `(hism_handle: u32, instance_count: u32, light_position, light_color, …)`. Handles are stable u32 IDs — no pointer hashing. If fingerprint matches and camera hasn't moved > 0.5 m: skip batch rebuild and light upload, re-use cached state.

### Step 2 — HISM batch construction

Group `HismInstance` list by `HismHandle`. For each unique handle:
1. Look up `HismEntry` in `HismRegistry` — O(1) index into `Vec`.
2. Select LOD mesh: compute screen coverage from `bounds_radius` + camera distance, pick `lod_levels` entry, apply `lod_bias`.
3. Collect `Mat4` transforms → write into pre-allocated instance buffer (overwrite via `queue.write_buffer` when count unchanged; re-create only when count grows).
4. Push one `DrawCall` with `hism_handle`, the LOD-selected vertex/index buffers, the material bind group, and `instance_count = N`.
5. Sort **opaque** `DrawCall` list front-to-back by camera depth (`dot(center - cam, forward)`), with `hism_handle` as a deterministic tie-break — stable sort preserves handle-grouped order within the same depth bucket.

**Per-frame allocation policy:** Instance buffers are recycled — `std::mem::take` clears the shape queue. `reserve(n)` not `reserve(n.saturating_sub(capacity()))`. Batch map (`HashMap<HismHandle, InstanceScratch>`) persists as a scratch field — inner `Vec<Mat4>` cleared, not the outer map. `InstanceScratch` also holds the persisted `Arc<Buffer>` and last-count for buffer recycling.

### Step 3 — Light sort and shadow matrix upload

Sort lights by type (directional → point → spot). Build `GpuLight` array → `queue.write_buffer`. Shadow matrix upload is delta-gated: only if world bounds, light count, or sky color changed (the RC `frame` field alone does not trigger an upload).

Scene fingerprint for cache invalidation now hashes `HismHandle` values (u32, cheap) instead of raw pointer addresses. This makes fingerprints stable across frames even when `Arc` allocations move.

For each shadow-casting light:
- **Directional:** `compute_directional_cascades()` — sphere-fit circumscribed sphere, texel-snapped CSM origins, 4 ortho matrices. **Do not AABB-fit** (causes shadow swimming).
- **Point:** 6 perspective matrices (90° FOV, aspect=1)
- **Spot:** 1 perspective matrix

Shadow cache hash: streaming FNV-1a over draw counts + bounds — no temporary `Vec<u64>`.

### Step 4 — Camera and globals upload

```rust
queue.write_buffer(&camera_buffer, 0, bytemuck::bytes_of(&camera_uniform));
queue.write_buffer(&globals_buffer, 0, bytemuck::bytes_of(&globals));
```

`ambient_color` computed by `estimate_sky_ambient(sun_elevation, rayleigh_coefficients)` each frame.

### Step 5 — Debug batch

Drain `debug_shapes` via `std::mem::take` into `DebugDrawBatch` → upload to pre-allocated debug vertex buffer (slice-based GPU write, not clone+clear).

### Step 6 — `graph.execute()`

Topological execution. Shadow bundles and GBuffer bundles are replayed when `draw_list_generation` hash matches. The hash is a commutative structural hash of `(mesh_ptr, mat_ptr, count)` tuples — only bumps on actual topology change, not per-frame.

### Step 7 — AA pass

Runs after graph. Reads `pre_aa_color`, writes to final surface view.

### Step 8 — Portal telemetry

Fire-and-forget snapshot of scene state + GPU timings → `helio-live-portal`. Never blocks rendering.

---

## 10. Render Pass Specifications

Each pass listed below is self-contained: it allocates its GPU resources in `new()`, declares its graph edges in `declare_resources()`, and executes in `execute()`.

---

### `SkyLutPass`

- **Resources created in `new()`:** `sky_lut` (`Rgba16Float` 192×108), sky bind group layout, sky pipeline.
- **Trigger:** Runs only when `sky_state_changed = true`. Result stable for many frames.
- **Pipeline:** Fullscreen triangle, no vertex buffer.
- **Output:** writes `"sky_lut"`.

---

### `SkyPass`

- **Inputs:** reads `"sky_lut"`.
- **Output:** writes `"sky_layer"` (color target). `LoadOp::Clear(0,0,0,1)`.
- **Shader:** Samples sky LUT, adds procedural FBM cloud plane, applies ACES tone mapping.

---

### `ShadowPass`

Full spec in [Section 11](#11-shadow-system). Created only when `config.shadows.is_some()`.

Resources owned by `ShadowPass`:
- `shadow_atlas` (Depth32Float 2D array, `atlas_size × atlas_size × max_lights*6`)
- `shadow_matrix_buffer` (GpuShadowMatrix STORAGE)
- `slot_bind_groups: Vec<BindGroup>` (pre-baked per atlas layer)
- `slot_idx_buffers: Vec<Buffer>` (immutable layer-index uniforms — keep alive for the duration of the pass)
- `layer_views: Vec<TextureView>`
- `bundle_cache: Vec<Option<RenderBundle>>`
- `bundle_geom_hashes: Vec<u64>`

---

### `DepthPrepassPass`

- **Output:** writes `"depth"`.
- **Pipeline:** Slot 0 + Slot 1. Depth-only (`DepthWrite=true, Compare=Less`). No color targets.
- **Groups:** Group 0 (global), Group 1 (material — alpha cutout).
- **RenderBundle:** cached per `draw_list_generation` hash.

---

### `GBufferPass`

- **Input:** reads `"depth"` → `LoadOp::Load` (uses depth prepass output; never clear depth here).
- **Output:** writes `"gbuffer"`.
- **4 MRT color targets:** albedo (`Rgba8Unorm`), normal (`Rgba16Float`), ORM (`Rgba8Unorm`), emissive (`Rgba16Float`). All cleared to zero.
- **Pipeline:** Group 0 (global), Group 1 (material), Group 2 (lighting). 2 vertex slots.
- **RenderBundle:** cached per `draw_list_generation` hash.
- **GBuffer pipeline depth compare:** `LessEqual`, `depth_write_enabled=false` (prepass already wrote depth — do not clear or re-write it).

---

### `DeferredLightingPass`

- **Inputs:** reads `"gbuffer"`, `"shadow_atlas"`, `"rc_cascade0"`, `"sky_layer"`.
- **Output:** writes `"color_target"` (or `"pre_aa_color"` when AA is enabled). `LoadOp::Load`.
- **Pipeline:** Fullscreen triangle. Group 0 (global), Group 1 (gbuf read), Group 2 (lighting).
- **Bloom:** Computed inline: `excess = max(luminance(color) - threshold, 0); color += color * excess * intensity`. Controlled via WGSL override constants derived from `BloomConfig`.
- **Bind group cache:** FXAA/SMAA/TAA bind groups are cached and rebuilt only on resize.

---

### `TransparentPass`

- **Input + Output:** reads and writes `"color_target"`.
- **Sort:** Back-to-front by `dot(center - cam, forward)` with deterministic tie-break. Index-sorted (no `DrawCall` clone).
- **Pipeline:** Same 3-group layout as GBuffer but alpha blend enabled, `DepthWrite=false`, `DepthCompare=Less`.
- **No RenderBundle** — sort order changes each frame.
- **Material bind group updates skipped** when unchanged (pointer identity).

---

### `BillboardsPass`

Created only when `config.billboards.is_some()`.

- **Resources created in `new()`:** screen-space quad VBO, instance buffer (`max_instances` entries), sprite texture atlas, bind group.
- **Input + Output:** reads and writes `"color_target"`.
- **Pipeline:** Slot 0 (quad) + Slot 1 (instance), alpha blend, no depth write.
- Instance buffer updated each frame via slice-based `queue.write_buffer`. Staging vector is reused (no per-frame allocation).

---

### `RadianceCascadesPass` (compute)

Created only when `config.radiance_cascades.is_some()`. Full spec in [Section 13](#13-radiance-cascades-gi).

Resources created in `new()`:
- 12 `Rgba16Float` textures (`rc_cascade[0..3]`, `rc_hist_a[0..3]`, `rc_hist_b[0..3]`)
- TLAS (`max_instances` cap from `RcConfig`)
- 4 × compute bind groups (one per cascade)
- `RCDynamic` uniform buffer, 4 × `CascadeStatic` uniform buffers
- `active_mesh_keys_scratch: Vec<u64>` (persisted, not re-allocated each frame)

TLAS rebuild is skipped unless the active draw-call topology changes (tracked via `last_active_mesh_keys` + count).

`RCDynamic` uploads are **delta-gated**: only write buffer if world bounds, light count, or sky color changed.

---

### `DebugDrawPass`

- **Input + Output:** reads and writes `"color_target"`.
- **Pipeline:** Lines topology, 1 vertex slot, no depth write, `DepthCompare=Always`.
- Borrows from shared mutex guard — does not clone `DebugDrawBatch`.

---

### `IndirectDispatchPass` *(optional, when `gpu_driven=true`)*

- Compute pass: `GpuDrawCall` array → `DrawIndexedIndirect` commands.
- Eliminates per-draw-call CPU→GPU command encoding. ~20–30% CPU savings on large scenes.

---

### SSAO *(optional, when `config.ssao.is_some()`)*

Runs between `DepthPrepass` and `GBuffer`:
- 16-sample hemisphere kernel, 4×4 tiled noise texture for rotation
- Blur pass → composite into GBuffer ORM red channel (AO)
- Config fields from `SsaoConfig`: `radius`, `bias`, `power`, `samples`

---

## 11. Shadow System

### Atlas Layout

```
Texture: Depth32Float, 2D Array (owned by ShadowPass)
Dimensions: atlas_size × atlas_size (default 2048×2048)
Layers: max_shadow_lights * 6   (default 40 → 240 layers)
Usage: RENDER_ATTACHMENT | TEXTURE_BINDING
```

One `TextureView` per layer (D2 view of single slice) in `ShadowPass::layer_views`.

### Per-Light Layer Assignment

```
point  light i → layers [i*6+0 .. i*6+5]  (faces ±X, ±Y, ±Z)
dir    light i → layers [i*6+0 .. i*6+3]  (CSM cascades 0–3)
spot   light i → layer  [i*6+0]
```

### Shadow Pipeline

- **Vertex:** Slot 0 (locations 0 + 2 only) + Slot 1 (locations 5–8)
- **Groups:** Group 0 (shadow matrix storage + layer index uniform), Group 1 (material)
- **Depth:** `Depth32Float, DepthWrite=true, Compare=Less`
- **Cull:** `None` (two-sided — avoids missing shadows from inconsistently-wound meshes)
- **Depth bias:** `slope_scale=1.0, constant=0`
- **No push constants** (not portable; layer index baked into per-slot bind group)

### WGSL Shadow Vertex Contract

```wgsl
// Group 0 B0: shadow matrix array (STORAGE)
// Group 0 B1: shadow_layer_idx (UNIFORM, u32)
// Group 1 B0: MaterialUniform
// Group 1 B1: base_color_texture
// Group 1 B3: material_sampler

fn vs_main(@location(0) position: vec3<f32>,
           @location(2) tex_coords: vec2<f32>,
           @location(5) model_0: vec4<f32>,
           @location(6) model_1: vec4<f32>,
           @location(7) model_2: vec4<f32>,
           @location(8) model_3: vec4<f32>) {
    let model = mat4x4<f32>(model_0, model_1, model_2, model_3);
    let world_pos = model * vec4<f32>(position, 1.0);
    out.clip_position = light_matrices[shadow_layer_idx].mat * world_pos;
}
```

### Per-Slot Bind Groups

Pre-baked at `ShadowPass::new()` for `max_lights * 6` slots:

```rust
for slot in 0..(max_lights * 6) {
    let idx_buf = device.create_buffer_init(&BufferInitDescriptor {
        contents: &(slot as u32).to_le_bytes(),
        usage: BufferUsages::UNIFORM,
    });
    // store in self.slot_idx_buffers (must outlive pass)
    // build bind group: B0 = shadow_matrix_buffer (STORAGE), B1 = idx_buf (UNIFORM)
    self.slot_bind_groups.push(bind_group);
}
```

`slot_idx_buffers` must be stored in the pass struct. Dropping them while bind groups exist is UB.

### RenderBundle Cache

```rust
bundle_cache: Vec<Option<wgpu::RenderBundle>>   // max_lights * 6 entries
bundle_geom_hashes: Vec<u64>                    // one per light (not per face)
```

Hash: streaming FNV-1a over `(hism_handle: u32, instance_count: u32)` tuples — no raw pointer addresses, deterministic across frames. No temporary `Vec<u64>`. If hash unchanged → replay cached bundles for all 6 faces. Shadow matrices are updated via `queue.write_buffer` before graph execution even when bundles are replayed (GPU reads fresh matrices).

### CPU Culling

Per draw call before bundle encode:
1. **Range cull:** Skip if object > `light.range * 5` from light.
2. **Hemisphere cull** (point only): skip if mesh bounding sphere is entirely opposite to this face normal.

Draw-call iteration borrows from the shared lock-guard — no `Vec<DrawCall>` clone per pass.

### PCF Filtering

4-tap rotated Poisson disk. Kernel scale `(2 / ATLAS_SIZE) * (1 + cascade_idx * 1.5)`:
- Cascade 0: 1× (sharp near shadows)
- Cascade 1: 2.5×
- Cascade 2: 4×
- Cascade 3: 5.5× (soft distant shadows)

CSM cascade selection via `globals.csm_splits` with 10% smooth transition zones (`smoothstep`).

---

## 12. PBR Lighting Model

### Cook-Torrance BRDF

```
Lo = ∑_lights [ (kD * albedo/π + kS * Cook-Torrance) * radiance * NdotL * shadow ]
   + indirect_diffuse (RC probes or ambient fallback)
   + indirect_specular (IBL envmap)
```

**GGX NDF:** `D(h) = a² / (π * ((NdH)² * (a²-1) + 1)²)`, `a = roughness²`

**Smith-Schlick-GGX Geometry:** `k = (roughness+1)²/8`

**Schlick Fresnel:** `F(cosθ) = F0 + (1-F0)*(1-cosθ)^5`, `F0 = mix(0.04, albedo, metallic)`

### Attenuation

```wgsl
// Point/spot
let attenuation = 1.0 / (dist * dist);
// Spot cone
let spot_factor = smoothstep(cos_outer, cos_inner, theta);
```

### Indirect Diffuse (RC)

Trilinear lookup into `rc_cascade0`. Eight probe corners sampled, cosine-weighted. Volume fade (5% margin) blends to ambient fallback.

### Indirect Specular (IBL)

```wgsl
let reflect_dir = reflect(-view, normal);
var spec_color = textureSample(env_cube, env_sampler, reflect_dir).rgb;
spec_color *= (1.0 - roughness * roughness);
```

### Ambient Fallback

```wgsl
let ambient_up   = globals.ambient_color.rgb * globals.ambient_intensity;
let ambient_down = ambient_up * 0.15;
let ambient = mix(ambient_down, ambient_up, (normal.y + 1.0) / 2.0);
```

### Tone Mapping (in SkyPass)

ACES: `f(x) = (x*(2.51x+0.03)) / (x*(2.43x+0.59)+0.14)`

### Bloom (inline in DeferredLightingPass)

```wgsl
let lum = dot(color.rgb, vec3(0.2126, 0.7152, 0.0722));
let excess = max(lum - BLOOM_THRESHOLD, 0.0);
color = vec4(color.rgb + color.rgb * excess * BLOOM_INTENSITY, color.a);
```

`BLOOM_THRESHOLD` and `BLOOM_INTENSITY` are WGSL `override` constants. Compiled from `BloomConfig` if present, otherwise defaults (`1.0`, `0.3`). If `config.bloom = None`, `ENABLE_BLOOM=false` preprocesses them out.

---

## 13. Radiance Cascades GI

Hardware ray-traced GI via compute. Requires `EXPERIMENTAL_RAY_QUERY`. All resources are created and owned by `RadianceCascadesPass::new()`.

### Constants

```
CASCADE_COUNT = 4
PROBE_DIMS    = [16, 8, 4, 2]
DIR_DIMS      = [4, 8, 16, 32]
T_MAXS        = [0.5, 1.0, 2.0, ∞]
ATLAS_W       = 64
ATLAS_HEIGHTS = [1024, 512, 256, 128]
```

### Texture Atlas Layout

```
atlas_x = probe_x * dir_dim + dir_x
atlas_y = (probe_y * probe_dim + probe_z) * dir_dim + dir_y
```

Probe data: `rgb = radiance, w = throughput` (0=opaque hit, 1=sky miss).

### Temporal Accumulation

Ping-pong between `hist_a` and `hist_b`. Each frame: `new = mix(old, traced, 0.15)` (~6-frame convergence). Fast camera motion causes momentary GI trailing.

### TLAS

```rust
device.create_tlas(&TlasDescriptor {
    max_instances: config.max_instances,
    flags: AccelerationStructureFlags::PREFER_FAST_TRACE,
})
```

TLAS rebuild is **skipped** when `active_mesh_keys_scratch` matches `last_active_mesh_keys` (same draw-call topology). `active_mesh_keys_scratch` is a persisted `Vec` field — not re-allocated each frame.

### Compute Dispatch

```wgsl
@workgroup_size(8, 8)
dispatch_workgroups(ATLAS_W / 8, ATLAS_H[level] / 8, 1)
```

4 dispatches per frame (coarsest to finest). Can overlap with fragment shading on async compute.

### Camera-Follow Probe Grid

Grid AABB snapped to cascade-0 cell size each frame using **floor-based cell snapping** to avoid half-cell oscillation from camera jitter:

```rust
let cell_size = world_extent / probe_dim as f32;
let snapped = (camera_pos / cell_size).floor() * cell_size;
let world_min = snapped - Vec3::splat(world_extent * 0.5);
```

Written into `GlobalsUniform.rc_world_min/max`.

---

## 14. Sky System

### Two-Pass Architecture

**`SkyLutPass`** — conditional on `sky_state_changed`:
- 192×108 panoramic `Rgba16Float` LUT (u=azimuth 0…2π, v=sin(elevation) -1…1)
- Full Nishita atmosphere integral per texel
- ~46× cheaper than per-pixel ray-marching

**`SkyPass`** — every frame:
- Samples LUT, falls back to inline atmosphere for degenerate angles
- Sun disc: cos-threshold comparison near `sun_direction`
- Procedural cloud plane: `cloud_base` height intersection, FBM × coverage, wind-animated UVs

### Atmosphere Equations

16 primary march steps, 4 depth steps per in-scatter shadow ray.

```
rayleigh_phase(cos_theta) = (3/16π) * (1 + cos_theta²)
mie_phase(cos_theta, g)   = (3/8π) * (1-g²) * (1+cos_theta²) / ((2+g²)*(1+g²-2g*cos_theta)^1.5)
```

### Ambient Estimation (`estimate_sky_ambient`)

| Sun elevation | Behaviour |
|---|---|
| `< -0.05` (night) | `[0.04,0.06,0.15]` → `[0.01,0.01,0.02]` |
| `0..0.15` (dawn/dusk) | → warm `[0.55,0.38,0.20]` |
| `> 0.15` (day) | Rayleigh-derived blue, clamped |

---

## 15. Anti-Aliasing Options

```rust
pub enum AntiAliasingMode { None, Fxaa, Smaa, Taa, Msaa(MsaaSamples) }
pub enum MsaaSamples { X2 = 2, X4 = 4, X8 = 8 }
```

AA passes run **after** `graph.execute()`. They are not in the topological graph. They read `pre_aa_color` and write to the final surface view. AA bind groups are **cached** and rebuilt only on resize.

### FXAA

NVIDIA FXAA 3.11. Luma-edge detection. Constants: `EDGE_THRESHOLD_MIN=0.0312`, `EDGE_THRESHOLD_MAX=0.125`, `SUBPIXEL_QUALITY=0.75`, `ITERATIONS=12`.

### TAA

- 8-entry Halton(2,3) jitter on projection
- History feedback: `mix(0.97, 0.88, saturate(vel_len * 100))`
- YCoCg colour space for neighbourhood clip
- Catmull-Rom 16-tap history filter
- Variance clipping: 3×3 neighbourhood, 1.25σ
- Motion vectors: `Rg16Float` written in GBuffer pass

### SMAA

3-pass: edge detect → blend weights (area + search textures) → neighbourhood blend.

### MSAA

All render pass targets use multisampled variant. Resolve before post-processing.

---

## 16. Material System

### CPU `Material` — builder pattern

| Field | Default |
|---|---|
| `base_color: [f32;4]` | `[1,1,1,1]` |
| `metallic: f32` | `0.0` |
| `roughness: f32` | `0.5` |
| `ao: f32` | `1.0` |
| `emissive_color: [f32;3]` | `[0,0,0]` |
| `emissive_factor: f32` | `0.0` |
| `alpha_cutoff: f32` | `0.0` |
| `transparent_blend: bool` | `false` |
| `base_color_texture` | `None` |
| `normal_map` | `None` |
| `orm_texture` | `None` |
| `emissive_texture` | `None` |

### Texture Formats

- `base_color_texture`: `Rgba8UnormSrgb`
- `normal_map`: `Rgba8Unorm` linear — `n = tex * 2 - 1`
- `orm_texture`: `Rgba8Unorm` — R=occlusion, G=roughness, B=metallic
- `emissive_texture`: `Rgba8UnormSrgb`

### Routing

Routed to `TransparentPass` when `transparent_blend == true` OR (`base_color.a < 1.0` AND `alpha_cutoff == 0.0`). Otherwise: DepthPrepass → GBuffer → DeferredLighting.

### `GpuMaterial`

Owns 4 texture uploads + material bind group. Created via `build_gpu_material(&device, &queue, &material, &material_bgl, &default_views)`. The `material_bgl` is shared and created once by the renderer.

---

## 17. Scene and Camera API

### `Scene`

```rust
pub struct Scene {
    pub instances: Vec<HismInstance>,  // replaces objects: Vec<SceneObject>
    pub lights: Vec<SceneLight>,
    pub sky: Option<SkyAtmosphere>,
    pub skylight: Option<Skylight>,
    pub camera: Camera,
}
```

### `SceneLight`

```rust
pub struct SceneLight {
    pub light_type: LightType,
    pub position: Vec3,
    pub direction: Vec3,
    pub color: Vec3,
    pub intensity: f32,
    pub range: f32,
    pub inner_angle: f32,
    pub outer_angle: f32,
    pub cast_shadows: bool,
}
pub enum LightType { Directional, Point, Spot }
```

Up to `MAX_LIGHTS = 2048`. Stable-sorted by type each frame.

### `SkyAtmosphere` Defaults

Earth-accurate:
- `rayleigh_scatter: [5.8e-3, 13.5e-3, 33.1e-3]` km⁻¹
- `mie_scatter: 2.1e-3`, `mie_g: 0.76`
- `sun_intensity: 22.0`, `exposure: 4.0`
- `earth_radius: 6360.0` km, `atm_radius: 6420.0` km

### `GpuMesh` Construction

```rust
let mesh = Arc::new(GpuMesh::new(&device, &vertices, &indices));
// Adds BLAS_INPUT usage only if EXPERIMENTAL_RAY_QUERY is present
// bounds_center and bounds_radius auto-computed from vertices
```

Primitives: `GpuMesh::unit_cube`, `GpuMesh::unit_rect3d`.

### Registering a HISM

```rust
// Register once at scene setup time — not per frame
let handle: HismHandle = renderer.hism_registry()
    .write()
    .register(mesh, material); // optionally .register_with_lods(mesh, material, lod_levels)

// Each frame — just push instances
scene.instances.push(HismInstance { handle, transform, lod_bias: 0 });
```

---

## 18. Debug Draw System

```rust
renderer.debug_shape(shape, color, transform);
```

Shapes tessellated on the CPU each frame into `DebugDrawBatch`. Queue cleared via `std::mem::take`. Vertex buffer pre-allocated at 64KB; resized only if exceeded.

Available shapes:
- `DebugShape::Sphere { radius }` — 3× great circles (12 segments each)
- `DebugShape::Aabb { min, max }` — 12 edges
- `DebugShape::Line { start, end }`
- `DebugShape::Arrow { start, end }` — line + arrowhead
- `DebugShape::Frustum { view_proj_inv }` — 12 edges unprojected from NDC
- `DebugShape::Capsule { radius, half_height }`
- `DebugShape::Cross { size }` — 3 axes

`DebugDrawPass` borrows from the shared mutex guard — never clones `DebugDrawBatch`.

---

## 19. GPU Profiler

`GpuProfiler` is **gated behind `RendererConfig::debug_printout`**. When `debug_printout = false` (production), no timestamp queries are issued, no readback buffers are allocated, and `queue.submit` does not stall on timestamp resolve. Use non-blocking `poll_results` only when profiling is enabled.

```rust
// Inside passes (no-ops when profiling disabled)
let token = ctx.scope_begin("pass_name");
// ... GPU work ...
ctx.scope_end(token);
```

Results available on next frame as `Vec<PassTiming>` via `renderer.last_pass_timings()`. Also forwarded to `helio-live-portal`.

```rust
pub struct PassTiming {
    pub name: String,
    pub gpu_us: f64,
}
```

---

## 20. Shared State Ownership Map

All cross-pass shared data uses `Arc<Mutex<T>>` or `Arc<AtomicU32>`. Write all shared state and call all `queue.write_buffer` **before** `graph.execute()`. The graph only reads — no cross-pass contention.

| Shared field | Written by | Read by |
|---|---|---|
| `draw_list: Arc<Mutex<Vec<DrawCall>>>` | Renderer batch loop | All geometry passes |
| `debug_batch: Arc<Mutex<Option<DebugDrawBatch>>>` | Renderer pre-graph | `DebugDrawPass` |
| `light_buffer: Arc<Buffer>` | Renderer (delta-gated) | `DeferredLightingPass`, `TransparentPass`, RC compute |
| `shadow_matrix_buffer: Arc<Buffer>` | Renderer (pre-graph) | `ShadowPass`, `DeferredLightingPass` |
| `light_count_arc: Arc<AtomicU32>` | Renderer | `ShadowPass` |
| `light_face_counts: Arc<Mutex<Vec<u8>>>` | Renderer | `ShadowPass` |
| `shadow_cull_lights: Arc<Mutex<Vec<ShadowCullLight>>>` | Renderer | `ShadowPass` |
| `deferred_bg: Arc<Mutex<Arc<BindGroup>>>` | Renderer on resize | `DeferredLightingPass` |
| `gbuffer_targets: Arc<Mutex<GBufferTargets>>` | Renderer on resize | `GBufferPass`, `DeferredLightingPass` |
| `draw_list_generation: Arc<AtomicU32>` | Renderer batch loop | `GBufferPass`, `DepthPrepassPass` |
| `hism_registry: Arc<RwLock<HismRegistry>>` | User / scene setup | Renderer batch loop (read-only during graph exec) |

---

## 21. Pipeline Cache and Variants

### `PipelineCache`

```rust
struct PipelineCache {
    pipelines: HashMap<PipelineKey, Arc<RenderPipeline>>,
}
```

**V3 key:** `(shader_path: &'static str, variant: PipelineVariant, config_hash: u64)`. `config_hash` is derived from `ShaderConfig` (replaces `FeatureFlags` from V2). HISM handle values are never part of the pipeline key — pipelines are shared across all handles that share the same variant.

### `ShaderConfig`

Derived once from `RendererConfig` at `Renderer::new()`. Injected as WGSL `override` constants.

```rust
pub struct ShaderConfig {
    pub enable_shadows: bool,
    pub max_shadow_lights: u32,
    pub enable_rc: bool,
    pub enable_bloom: bool,
    pub bloom_threshold: f32,
    pub bloom_intensity: f32,
}

impl ShaderConfig {
    pub fn from_renderer_config(c: &RendererConfig) -> Self {
        Self {
            enable_shadows: c.shadows.is_some(),
            max_shadow_lights: c.shadows.as_ref().map_or(0, |s| s.max_shadow_lights),
            enable_rc: c.radiance_cascades.is_some(),
            enable_bloom: c.bloom.is_some(),
            bloom_threshold: c.bloom.as_ref().map_or(1.0, |b| b.threshold),
            bloom_intensity: c.bloom.as_ref().map_or(0.3, |b| b.intensity),
        }
    }
}
```

### `PipelineVariant`

```rust
pub enum PipelineVariant {
    GBufferWrite,         // Group 0 + 1; 4 MRT; depth LessEqual, no write
    DepthOnly,            // Group 0 + 1; depth only; depth Less, write
    DeferredLighting,     // Group 0 + 1(gbuf) + 2; fullscreen
    TransparentForward,   // Group 0 + 1 + 2; alpha blend; depth read-only
    ShadowDepth,          // Group 0(shadow) + 1(mat); depth only; two-sided
    BillboardAlpha,       // billboard groups; alpha blend
    DebugDraw,            // lines; no depth write; depth Always
}
```

### WGSL Override Constants

| Override | Source |
|---|---|
| `ENABLE_SHADOWS: bool` | `ShaderConfig::enable_shadows` |
| `MAX_SHADOW_LIGHTS: u32` | `ShaderConfig::max_shadow_lights` |
| `ENABLE_RC: bool` | `ShaderConfig::enable_rc` |
| `ENABLE_BLOOM: bool` | `ShaderConfig::enable_bloom` |
| `BLOOM_THRESHOLD: f32` | `ShaderConfig::bloom_threshold` |
| `BLOOM_INTENSITY: f32` | `ShaderConfig::bloom_intensity` |

### G-Buffer Pipeline — 4 Color Targets

```
Target 0: Rgba8Unorm    — albedo    LoadOp::Clear(0)
Target 1: Rgba16Float   — normal    LoadOp::Clear(0)
Target 2: Rgba8Unorm    — ORM       LoadOp::Clear(0)
Target 3: Rgba16Float   — emissive  LoadOp::Clear(0)
Depth:    Depth32Float  LoadOp::Load
DepthCompare: LessEqual, depth_write_enabled: false
```

---

## 22. Reconstruction Checklist

Follow this order to avoid dependency issues:

1. **Crate scaffold:** `lib.rs` Error/Result types, `Cargo.toml` with exact deps.
2. **Data structs:** `camera.rs`, `mesh.rs` (PackedVertex, GpuMesh, DrawCall), `material.rs`, `scene.rs` (Scene + HismInstance), `hism.rs` (HismHandle, HismRegistry, HismEntry, LodLevel).
3. **Graph:** `graph/mod.rs` — `RenderPass` trait, `PassContext`, `RenderGraph`, `PassResourceBuilder`, topological sort (edges in registration order).
4. **Pipeline cache:** `pipeline/mod.rs` — `PipelineCache`, `PipelineVariant`, `ShaderConfig` (no FeatureFlags).
5. **Resource manager:** `resources/mod.rs` — texture upload helpers, sampler cache, stub texture factory.
6. **Passes** (each creates its own GPU resources in `new()`):
   - `sky_lut.rs` + `sky.rs`
   - `shadow.rs` (if `config.shadows.is_some()`)
   - `depth_prepass.rs`
   - `gbuffer.rs`
   - `deferred_lighting.rs`
   - `transparent.rs`
   - `radiance_cascades.rs` (if `config.radiance_cascades.is_some()`)
   - `billboards.rs` (if `config.billboards.is_some()`)
   - `debug_draw.rs`
   - `smaa.rs`, `fxaa.rs`, `taa.rs`
   - `indirect_dispatch.rs` (if `config.gpu_driven`)
7. **Shaders:** WGSL files in `shaders/passes/` matching all binding layouts from Section 7. Start with `common/` (brdf, shadows, lighting). Use `override` for all `ShaderConfig` constants.
8. **Renderer:** `renderer.rs` — `RendererConfig`, `Renderer::new()` (derive `ShaderConfig` → instantiate passes → build graph), `render_scene()` (batch loop → light sort → matrix upload → `graph.execute()` → AA pass).
9. **Debug draw:** `debug_draw.rs` — shape tessellation + `DebugDrawPass`.
10. **Profiler:** `profiler.rs` — gated on `debug_printout`.

### Critical Correctness Points

- `instance_buffer` in `DrawCall` is never `Option` — always `Arc<Buffer>`. Passes unconditionally bind slot 1.
- Shadow RenderBundle has the layer-index bind group baked in. **Never** share bind groups across bundles for different faces.
- `shadow_matrix_buffer` is written before `graph.execute()` — bundles read fresh matrices on replay.
- `GpuMesh` vertex buffers need `BLAS_INPUT` **only if** `EXPERIMENTAL_RAY_QUERY` device feature is present.
- CSM uses sphere-fit + texel-snapping. **Not AABB-fit** (causes shadow swimming on camera rotation).
- `DeferredLightingPass` uses `LoadOp::Load` on the color target (sky already there from `SkyPass`).
- `GBufferPass` uses `LoadOp::Load` on depth (from DepthPrepass) and `LessEqual` with `depth_write_enabled=false`.
- `draw_list_generation` is a commutative structural hash of `(hism_handle: u32, instance_count: u32)` tuples — no raw pointer addresses, deterministic.
- RC probe grid uses **floor-based cell snapping** for anchor — not plain rounding (avoids half-cell oscillation).
- Depth-sorting world-forward is derived from near/far center-ray unprojection via `view_proj_inv` — not from view matrix basis.
- `GpuProfiler` timestamp queries only issued when `debug_printout = true`.
- Group 2 bind group layout is identical regardless of whether shadows or RC are enabled. Disabled features get stub 1×1 dummy textures to satisfy the layout.

### Known Pitfalls

| Pitfall | Symptom | Fix |
|---|---|---|
| Using raw `Arc` pointer as batch key | Non-deterministic hashes, broken bundle cache | Use `HismHandle` (u32) as the only batch key |
| Re-registering the same mesh+material every frame | Handle list grows unbounded | Register once at scene setup; store handles |
| `var<immediate>` in shadow shader | Device validation panic | Use per-slot uniform buffer for layer index |
| Instance buffer in shadow pipeline absent | Validation error at draw | Add Slot 1 to shadow `VertexState` |
| Shadow vertices in model space without model matrix | All shadows at origin | Apply model matrix before light-space transform |
| `@builtin(instance_index)` as layer selector | All shadows use layer 0 | Use per-slot uniform |
| `device.create_buffer_init` without `DeviceExt` | Compile error | `use wgpu::util::DeviceExt` |
| Slot-idx buffers dropped at end of scope | Use-after-free | Store `slot_idx_buffers: Vec<Buffer>` in `ShadowPass` |
| CSM AABB-fit | Shadow shimmering on camera yaw | Circumscribed sphere + texel-snap origin |
| Vertices pre-transformed to world space | Double-offset artifacts | Vertices always local-space |
| `LoadOp::Clear(1.0)` on depth in GBufferPass | Discards depth prepass output | Use `LoadOp::Load` |
| `draw_list_generation` unconditionally incremented | RenderBundles rebuilt every frame (~9 ms wasted) | Hash `(mesh_ptr, mat_ptr, count)` tuples |
| RC `RCDynamic` uploaded every frame | Redundant buffer writes | Delta-gate on world bounds / light count / sky color |
| RC probe grid rounded (not floored) | GI world-swimming on camera move | `floor(pos / cell_size) * cell_size` |
| `GpuProfiler` always active | Queue submit stalls from timestamp resolve | Gate behind `debug_printout` flag |
| AA bind groups recreated every frame | Per-frame allocation in hot path | Cache; rebuild only on resize |
| `draw_list` cloned per geometry pass | Large `Vec<DrawCall>` copies every frame | Borrow from shared lock-guard |
| Shadow cache hash uses `Vec<u64>` scratch | Per-frame allocation | Streaming FNV-1a, no scratch Vec |
