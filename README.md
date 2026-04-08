<div align="center">

<img src="./branding/Helio.svg" alt="Helio Renderer" width="400"/>

**A production-grade real-time renderer built on [wgpu](https://wgpu.rs/)**

**WIP**

[![Rust](https://img.shields.io/badge/rust-stable-orange?logo=rust)](https://www.rust-lang.org/)
[![wgpu](https://img.shields.io/badge/wgpu-28-blue)](https://wgpu.rs/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20Linux%20%7C%20macOS%20%7C%20Android-lightgrey)](#)
[![glam](https://img.shields.io/badge/glam-math-blue)](https://docs.rs/glam/)
[![bytemuck](https://img.shields.io/badge/bytemuck-serialization-blue)](https://docs.rs/bytemuck/)
[![PBR](https://img.shields.io/badge/feature-PBR-brightgreen)]()
[![GI](https://img.shields.io/badge/feature-Global%20Illumination-brightgreen)]()
[![CSM](https://img.shields.io/badge/feature-Cascaded%20Shadows-brightgreen)]()
[![Sky](https://img.shields.io/badge/feature-Volumetric%20Sky-brightgreen)]()

> Cross-platform, data-driven, physically-based rendering — with a render graph, radiance cascades GI, cascaded shadow maps, volumetric sky, and bloom — all in pure Rust.

Sample model from https://sketchfab.com/mohamedhussien

</div>

Helio is a GPU-driven deferred rendering engine written in Rust on top of `wgpu`. It is designed around a **handle-based scene API** (inspired by Unreal's FScene model) layered over a modular, pass-oriented render graph with zero-copy per-frame contexts. Every CPU-side operation that touches the GPU buffer is bounded — typically O(1) — and the GPU does the heavy lifting for culling, indirect draw dispatch, and light evaluation.

<img width="1262" height="707" alt="image" src="https://github.com/user-attachments/assets/7787f33a-5e82-44ba-a1f8-a5342da5090d" />
<img width="2555" height="1340" alt="image" src="https://github.com/user-attachments/assets/f3f9f878-9a64-4f7b-b4fa-29f14d78250c" />

## Architecture

The workspace is structured in three tiers:

```
crates/helio           ← high-level scene + Renderer (start here)
crates/helio-v3        ← render graph runtime, GpuScene, RenderPass trait
crates/libhelio        ← shared GPU structs (GpuLight, GpuMaterial, GpuCameraUniforms, …)
crates/helio-pass-*    ← one crate per render pass
crates/helio-asset-compat  ← FBX / glTF / OBJ / USD asset loading via SolidRS
crates/examples        ← native example binaries
```

### Core crates

| Crate | Purpose |
|-------|---------|
| `helio` | `Renderer` + `Scene` with stable typed handles (`MeshId`, `MaterialId`, `LightId`, `ObjectId`, …), group visibility, and the default deferred graph |
| `helio-v3` | `RenderGraph`, `RenderPass`, `PassContext`, `GpuScene`, dirty-tracked GPU buffers, automatic CPU/GPU profiling |
| `libhelio` | `GpuLight`, `GpuMaterial`, `GpuCameraUniforms`, `FrameResources`, `BillboardFrameData`, `ShadowQuality` — shared between all pass crates |
| `helio-asset-compat` | `load_scene_file` / `load_scene_bytes` → `ConvertedScene` (meshes, materials, textures, lights, cameras) |

---

## Quick Start

### 1) Verify with a production example

The current Helio workflow has moved to explicit per-binary examples in `crates/examples`. Running an example is now the most reliable and fastest way to confirm the code works locally.

```sh
cargo run -p examples --bin indoor_cathedral --release
cargo run -p examples --bin indoor_server_room --release
cargo run -p examples --bin ship_flight --release
cargo run -p examples --bin load_fbx --release -- path/to/model.fbx
```

### 2) Add Helio as a dependency

```toml
[dependencies]
helio = { path = "crates/helio" }
helio-asset-compat = { path = "crates/helio-asset-compat" }
wgpu = "0.13"
winit = "0.30"
```

### 3) Create and configure a renderer

```rust
use std::sync::Arc;
use helio::{required_wgpu_features, required_wgpu_limits, Camera, GroupMask, Renderer, RendererConfig, ShadowQuality};

// setup wgpu `device`, `queue`, `surface`, `surface_format`, etc.

let config = RendererConfig::new(width, height, surface_format)
    .with_shadow_quality(ShadowQuality::High);
let mut renderer = Renderer::new(device.clone(), queue.clone(), config);

renderer.set_ambient([0.3, 0.4, 0.6], 0.04);
renderer.use_default_graph(); // or `renderer.use_simple_graph()` for lightweight demo mode
```

### 4) Populate scene and render (actor-based API)

```rust
let floor_mesh = renderer
    .scene_mut()
    .insert_actor(helio::SceneActor::mesh(plane_mesh([0.0, 0.0, 0.0], 32.0)))
    .as_mesh()
    .unwrap();
let wall_material = renderer.scene_mut().insert_material(material);

let light_id = renderer
    .scene_mut()
    .insert_actor(helio::SceneActor::light(point_light([0.0, 10.0, 0.0], [1.0, 0.9, 0.8], 8.0, 14.0)))
    .as_light()
    .unwrap();

let object_id = v3_demo_common::insert_object(
    &mut renderer,
    floor_mesh,
    wall_material,
    Mat4::IDENTITY,
    32.0,
)?;

let camera = Camera::perspective_look_at(eye, target, Vec3::Y, fov, aspect, 0.1, 1000.0);
renderer.render(&camera, &surface_view)?;
```

### Required wgpu features / limits

Call these before `request_device`:

```rust
let features = required_wgpu_features(adapter.features());
let limits   = required_wgpu_limits(adapter.limits());
```

Required features: `TEXTURE_BINDING_ARRAY` + `SAMPLED_TEXTURE_AND_STORAGE_BUFFER_ARRAY_NON_UNIFORM_INDEXING`.  
Optional (enabled when the adapter supports them): `MULTI_DRAW_INDIRECT`, `MULTI_DRAW_INDIRECT_COUNT`, `SHADER_PRIMITIVE_INDEX`.  
Limit: `max_sampled_textures_per_shader_stage` = 256.

---

## Renderer API

### Construction & configuration

```rust
Renderer::new(device, queue, config: RendererConfig) -> Self

// RendererConfig builder
RendererConfig::new(width, height, surface_format)
    .with_shadow_quality(ShadowQuality::Ultra)
    .with_gi_config(GiConfig { rc_radius: 120.0, rc_fade_margin: 30.0 })

// Runtime configuration
renderer.set_gi_config(GiConfig)
renderer.set_shadow_quality(ShadowQuality)
renderer.set_debug_mode(u32)          // 0=normal 10=shadow-heatmap 11=light-depth
renderer.set_render_size(width, height)
renderer.set_clear_color([f32; 4])
renderer.set_ambient([f32; 3], intensity: f32)
```

### Meshes, materials, textures

```rust
renderer.insert_mesh(MeshUpload) -> MeshId
renderer.remove_mesh(MeshId) -> Result<()>

renderer.insert_texture(TextureUpload) -> Result<TextureId>
renderer.remove_texture(TextureId) -> Result<()>

renderer.insert_material(GpuMaterial) -> MaterialId
renderer.update_material(MaterialId, GpuMaterial) -> Result<()>
renderer.remove_material(MaterialId) -> Result<()>
```

### Lights

Every inserted light automatically gets a camera-facing `spotlight.png` billboard icon when `GroupId::EDITOR` is visible (toggle with `renderer.hide_group(GroupId::EDITOR)`).

```rust
renderer.insert_light(GpuLight) -> LightId
renderer.update_light(LightId, GpuLight) -> Result<()>
renderer.remove_light(LightId) -> Result<()>
```

### Objects

```rust
renderer.insert_object(ObjectDescriptor) -> Result<ObjectId>
renderer.update_object_transform(ObjectId, Mat4) -> Result<()>
renderer.remove_object(ObjectId) -> Result<()>
```

### Virtual geometry (GPU-driven meshlets)

```rust
renderer.insert_virtual_mesh(VirtualMeshUpload) -> VirtualMeshId
renderer.insert_virtual_object(VirtualObjectDescriptor) -> Result<VirtualObjectId>
renderer.update_virtual_object_transform(VirtualObjectId, Mat4) -> Result<()>
renderer.remove_virtual_object(VirtualObjectId) -> Result<()>
```

### Billboards

```rust
renderer.set_billboard_instances(&[BillboardInstance])
```

User billboards are composited with the auto-generated editor light icons each frame.

### Custom passes

```rust
renderer.add_pass(Box::new(MyPass::new(...)))
renderer.set_graph(render_graph)     // supply a fully custom RenderGraph
renderer.use_default_graph()         // reset to the built-in deferred pipeline
```

### Rendering

```rust
renderer.render(&camera, &wgpu::TextureView) -> Result<()>
```

---

## Scene API (`helio::Scene`)

Access via `renderer.scene()` / `renderer.scene_mut()`. All methods on `Scene` are also available as pass-throughs on `Renderer`.

### Camera

```rust
Camera::perspective_look_at(position, target, up, fov_y, aspect, near, far) -> Camera
Camera::from_matrices(view, proj, position, near, far) -> Camera
```

### Object descriptors

```rust
pub struct ObjectDescriptor {
    pub mesh:      MeshId,
    pub material:  MaterialId,
    pub transform: Mat4,
    pub bounds:    [f32; 4],    // [cx, cy, cz, radius] world-space bounding sphere
    pub flags:     u32,
    pub groups:    GroupMask,   // GroupMask::NONE = always visible
}

pub struct VirtualObjectDescriptor {
    pub virtual_mesh: VirtualMeshId,
    pub material_id:  u32,
    pub transform:    Mat4,
    pub bounds:       [f32; 4],
    pub flags:        u32,
    pub groups:       GroupMask,
}
```

### Lights

```rust
// GpuLight fields (80 bytes)
position_range:  [f32; 4]  // xyz=world pos,    w=range (m)
direction_outer: [f32; 4]  // xyz=direction,    w=outer_cos (spot)
color_intensity: [f32; 4]  // xyz=linear sRGB,  w=intensity (cd/lux)
shadow_index:    u32        // u32::MAX = no shadow
light_type:      u32        // Point=1, Spot=2, Directional=0, Area=3

// Constructor helpers (v3_demo_common)
point_light(position: [f32;3], color: [f32;3], intensity: f32, range: f32) -> GpuLight
spot_light(pos, dir, color, intensity, range, inner_rad, outer_rad)        -> GpuLight
```

### Materials (PBR)

```rust
pub struct GpuMaterial {
    pub base_color:         [f32; 4],  // linear RGBA
    pub emissive:           [f32; 4],  // xyz=color, w=strength
    pub roughness_metallic: [f32; 4],  // x=roughness, y=metallic, z=IOR, w=specular_tint
    pub tex_base_color:  u32,          // u32::MAX = no texture
    pub tex_normal:      u32,
    pub tex_roughness:   u32,
    pub tex_emissive:    u32,
    pub tex_occlusion:   u32,
    pub workflow:        u32,   // 0=Metallic-Roughness, 1=Specular-Gloss
    pub flags:           u32,   // bit0=double-sided, bit1=alpha-blend, bit2=alpha-test
    pub _pad:            u32,
}
```

---

## Groups System

Objects and virtual objects carry a `GroupMask` (64-bit bitmask). Each bit corresponds to a `GroupId` (0–63). The scene maintains a `group_hidden` mask; an object is invisible when any of its groups overlaps the hidden set. Objects with `GroupMask::NONE` are always visible.

### Built-in groups

| Constant | Index | Intended use |
|----------|-------|--------------|
| `GroupId::EDITOR` | 0 | Editor helpers (light icons, gizmos) — hidden at shipping time |
| `GroupId::DEFAULT` | 1 | General user objects |
| `GroupId::STATIC` | 2 | Non-moving world geometry |
| `GroupId::DYNAMIC` | 3 | Animated / simulated objects |
| `GroupId::WORLD_UI` | 4 | World-space UI elements |
| `GroupId::VFX` | 5 | Particles and effects |
| `GroupId::SHADOW_CASTERS` | 6 | Hint to mass-disable shadows for prop layers |
| `GroupId::DEBUG` | 7 | Debug visualisers |

### Group API

```rust
// Visibility
renderer.hide_group(GroupId::EDITOR);
renderer.show_group(GroupId::EDITOR);
renderer.is_group_hidden(GroupId::EDITOR) -> bool
renderer.set_group_visibility(mask: GroupMask, visible: bool)

// Per-object membership
renderer.set_object_groups(id, GroupMask::NONE.with(GroupId::STATIC))
renderer.add_object_to_group(id, GroupId::STATIC)
renderer.remove_object_from_group(id, GroupId::STATIC)

// Mass transforms (GPU-side, O(N objects in group))
renderer.move_group(GroupId::DYNAMIC, Mat4::from_translation(delta))
renderer.translate_group(GroupId::DYNAMIC, Vec3::new(0.0, 1.0, 0.0))
```

### GroupMask operations

```rust
let mask = GroupMask::NONE
    .with(GroupId::STATIC)
    .with(GroupId::SHADOW_CASTERS);

mask.contains(GroupId::STATIC)        // true
mask.intersects(GroupMask::ALL)       // true
mask.without(GroupId::SHADOW_CASTERS)
let combined = mask_a | mask_b;
let overlap  = mask_a & mask_b;
```

---

## Default Render Pipeline

`Renderer::new` constructs the default deferred pipeline. Passes execute in this order:

| # | Pass | Kind | Description |
|---|------|------|-------------|
| 1 | `ShadowMatrixPass` | Compute | One thread per light — writes face view-proj matrices into `shadow_matrices` buffer |
| 2 | `ShadowPass` | Render | Geometry → 512×512×256-layer Depth32Float shadow atlas |
| 3 | `SkyLutPass` | Render | Bakes a 192×108 Hillaire 2020 atmospheric panoramic LUT |
| 4 | `DepthPrepassPass` | Render | Early-Z via `multi_draw_indexed_indirect`; O(1) CPU |
| 5 | `GBufferPass` | Render | GPU-driven → 4 G-buffer targets (albedo / normal+F0 / ORM / emissive) |
| 5b | `VirtualGeometryPass` | Compute + Render | Per-meshlet frustum + backface-cone culling → `multi_draw_indexed_indirect` into same G-buffer; screen-coverage LOD |
| 6 | `DeferredLightPass` | Render (fullscreen) | Cook-Torrance BRDF, PCF/PCSS CSM shadows (4 cascades at 16/80/300/1 400 wu), RC GI, tone mapping |
| 7 | `BillboardPass` | Render | Instanced camera-facing quads (editor light icons + user billboards) with alpha blending |

### Shadow quality presets

| Preset | PCF samples | PCSS | Blocker samples | Filter samples |
|--------|-------------|------|-----------------|----------------|
| `Low` | 8 | off | 8 | 8 |
| `Medium` | 16 | off | 8 | 8 |
| `High` | 16 | on | 8 | 16 |
| `Ultra` | 32 | on | 16 | 32 |

### GI (Radiance Cascades)

The `DeferredLightPass` integrates a Radiance Cascades probe grid. By default the RC volume is a cube of radius 80 wu centred on the camera, with a 20 wu soft fade to ambient at the edges.

```rust
RendererConfig::new(w, h, fmt).with_gi_config(GiConfig {
    rc_radius: 120.0,
    rc_fade_margin: 30.0,
})
// or:
GiConfig::ambient_only()          // disable RC, use flat ambient only
GiConfig::large_radius(radius)    // rc_radius=radius, rc_fade_margin=radius*0.25
```

---

## Virtual Geometry (Nanite-style Meshlets)

`VirtualGeometryPass` implements GPU-driven per-meshlet culling with automatic screen-coverage LOD selection. CPU cost per frame is O(1).

**Pipeline:**
1. CPU: call `renderer.insert_virtual_mesh(VirtualMeshUpload { vertices, indices })` once. Rust generates 3 LOD levels and partitions each into meshlets (≤ 64 triangles each) with bounding spheres and backface cones.
2. GPU (each frame): one compute dispatch (`⌈meshlets/64⌉` workgroups) — write `instance_count=0` for culled meshlets.
3. GPU: `multi_draw_indexed_indirect` — skips culled commands natively.

**LOD transitions:**  
`screen_radius = (bounds_radius × cot(fov/2)) / view_depth`  
LOD 0 → 1 when `screen_radius < lod_s0`; LOD 1 → 2 when `screen_radius < lod_s1`.  
Thresholds are configured per `LodQuality` (Low / Medium / High / Ultra) on `VirtualGeometryPass`.

---

## Pass Crates Reference

Each `helio-pass-*` crate exposes one type implementing `helio_v3::RenderPass`.

| Crate | Pass type | Summary |
|-------|-----------|---------|
| `helio-pass-depth-prepass` | `DepthPrepassPass` | Early-Z, O(1) CPU via indirect draw |
| `helio-pass-gbuffer` | `GBufferPass` | GPU-driven deferred G-buffer fill (albedo/normal/ORM/emissive) |
| `helio-pass-deferred-light` | `DeferredLightPass` | Cook-Torrance BRDF, PCF/PCSS shadows, RC GI, tone mapping |
| `helio-pass-shadow` | `ShadowPass` | 512×512×256 Depth32Float shadow atlas |
| `helio-pass-shadow-matrix` | `ShadowMatrixPass` | Compute: per-light face view-proj matrices |
| `helio-pass-sky-lut` | `SkyLutPass` | Hillaire 2020 atmospheric LUT (192×108); re-bakes on sky state change |
| `helio-pass-sky` | `SkyPass` | Fullscreen atmospheric background from sky LUT |
| `helio-pass-virtual-geometry` | `VirtualGeometryPass` | Nanite-style meshlet cull + LOD + indirect draw |
| `helio-pass-radiance-cascades` | `RadianceCascadesPass` | Cascade-probe GI (8³ probes, 4 direction bins/axis) |
| `helio-pass-sdf` | `SdfClipmapPass` | 8-level toroidal SDF clipmap (16³ brick grid); fullscreen sphere-trace fragment |
| `helio-pass-billboard` | `BillboardPass` | Up to 65 536 instanced camera-facing quads; per-instance position / scale / tint |
| `helio-pass-transparent` | `TransparentPass` | Alpha-blended forward over depth-read-only buffer |
| `helio-pass-fxaa` | `FxaaPass` | Fullscreen FXAA (luma-based edge detection) |
| `helio-pass-smaa` | `SmaaPass` | SMAA 1× (edge detect → blend weights → neighbourhood blend) |
| `helio-pass-taa` | `TaaPass` | Temporal AA with jitter + reprojection |
| `helio-pass-ssao` | `SsaoPass` | Screen-space AO using G-buffer normals + depth |
| `helio-pass-hiz` | `HiZBuildPass` | Min-reduction Hi-Z mip chain for GPU occlusion culling |
| `helio-pass-occlusion-cull` | `OcclusionCullPass` | GPU occlusion culling using the Hi-Z pyramid |
| `helio-pass-debug` | `DebugShapesPass` | Tube-tessellated debug lines, boxes, spheres, capsules, cones |
| `helio-pass-indirect-dispatch` | `IndirectDispatchPass` | Populates `DrawIndexedIndirect` buffers from the scene draw list |
| `helio-pass-light-cull` | `LightCullPass` | Tile/cluster light culling for the deferred lighting pass |
| `helio-pass-simple-cube` | `SimpleCubePass` | Hardcoded debug cube — sanity-check baseline, no scene required |

### Adding a pass to the default graph

Passes are plain Rust types — no registry, no reflection. Add a dependency, construct the pass, push it:

```rust
use helio_pass_sdf::{SdfClipmapPass, TerrainConfig};

let mut sdf = SdfClipmapPass::new(&device, renderer.camera_buffer(), surface_format);
sdf.set_terrain(TerrainConfig::rolling());
renderer.add_pass(Box::new(sdf));
```

### Building a fully custom graph (low-level `helio-v3`)

```rust
use helio_v3::{RenderGraph, GpuScene};
use helio_pass_gbuffer::GBufferPass;
use helio_pass_shadow::ShadowPass;

let scene = GpuScene::new(device.clone(), queue.clone());
let mut graph = RenderGraph::new(&device, &queue);
graph.add_pass(Box::new(ShadowPass::new(&device)));
graph.add_pass(Box::new(GBufferPass::new(&device, width, height)));
graph.execute_with_frame_resources(&scene, &target, &depth, &frame_resources)?;
```

---

## Asset Loading

```rust
use helio_asset_compat::{load_scene_file, load_scene_bytes, LoadConfig};

// From disk (auto-detects FBX / glTF / OBJ / USD by extension)
let scene = load_scene_file("assets/ship.fbx")?;

// From embedded bytes
let scene = load_scene_bytes(include_bytes!("ship.fbx"), "fbx", None)?;

// With DirectX UV convention (flip v)
let scene = load_scene_file_with_config("model.gltf", LoadConfig { flip_uv_y: true })?;
```

`ConvertedScene` contains:
- `meshes: Vec<ConvertedMesh>` — `PackedVertex` arrays + material indices
- `textures: Vec<TextureUpload>`
- `materials: Vec<ConvertedMaterial>`
- `lights: Vec<GpuLight>`
- `cameras: Vec<CameraData>`

---

## Examples

| Binary | What it demonstrates |
|--------|---------------------|
| `simple_graph` | Fly camera + hardcoded debug cube via `SimpleCubePass` |
| `indoor_room` | Furnished room with point lights |
| `indoor_corridor` | Hallway — fluorescents, exit signs, wall sconces |
| `indoor_cathedral` | Gothic nave (60 m, 12 columns, chandeliers, stained-glass shafts, RC GI) |
| `indoor_server_room` | Data-centre with 32 racks, hot/cold aisles, status LEDs; `E` toggles editor light icons |
| `outdoor_night` | Night-time plaza |
| `outdoor_canyon` | Desert canyon; Q/E rotates sun; campfire point lights |
| `outdoor_city` | Dense city block at dusk — 21 buildings, sodium streetlamps, neon signs |
| `outdoor_volcano` | Volcanic island with lava-glow fire lights |
| `outdoor_rocks` | Rock scatter field (3 types × 30) + VG meshlets + FBX ship |
| `space_station` | Massive orbital station (rings, spokes, solar arrays); fly speed 40 m/s |
| `ship_flight` | Pilot an FBX ship through a 10 000-asteroid field (radius 10–14 km) |
| `load_fbx` | Load any FBX/glTF/OBJ/USDC from disk and display it |
| `load_fbx_embedded` | Same with `include_bytes!` |
| `sdf_demo` | Fullscreen SDF clipmap ray march; FBM terrain + CSG sphere/capsule edits |
| `light_benchmark` | 150 simultaneous point lights (10 × 15 grid) over a warehouse floor |
| `rc_benchmark` | Cornell box with coloured walls — multi-bounce RC GI; 3 adjustable lights |
| `debug_shapes` | All debug primitives (lines, boxes, spheres, capsules) |

```sh
cargo run -p examples --bin indoor_server_room
cargo run -p examples --bin ship_flight
cargo run -p examples --bin light_benchmark
cargo run -p examples --bin load_fbx -- path/to/model.fbx

# Compile-check only
cargo check -p helio -p examples --quiet
```

---

## Editor Billboard System

When any light is inserted via `renderer.insert_light()`, Helio automatically composites a `spotlight.png` billboard icon at its world position each frame, tinted by the light's colour. These icons belong to `GroupId::EDITOR` and are visible by default.

```rust
// Hide all editor icons (e.g. for a shipping build)
renderer.hide_group(GroupId::EDITOR);

// Show them again
renderer.show_group(GroupId::EDITOR);

// Add your own object to the editor group
renderer.add_object_to_group(obj_id, GroupId::EDITOR)?;
```

Icons are world-space quads (0.25 × 0.25 m) that shrink naturally with perspective. The screen-scale mode (`BillboardInstance::scale_flags.z > 0.5`) is available for custom billboards that need a fixed screen size instead.

---

## GPU Layout Reference

### `GpuCameraUniforms` — offset table for custom WGSL shaders

| Field | Byte offset | Description |
|-------|-------------|-------------|
| `view` | 0 | World → view (mat4x4) |
| `proj` | 64 | View → clip (mat4x4) |
| `view_proj` | 128 | Combined VP (mat4x4) |
| `inv_view_proj` | 192 | Clip → world (mat4x4) |
| `position_near` | 256 | xyz=camera pos, w=near |
| `forward_far` | 272 | xyz=forward dir, w=far |
| `jitter_frame` | 288 | xy=TAA jitter, z=frame idx |
| `prev_view_proj` | 304 | Previous frame VP for TAA (mat4x4) |

### `BillboardInstance` (48 bytes)

```rust
pub struct BillboardInstance {
    pub world_pos:   [f32; 4],  // xyz=world position
    pub scale_flags: [f32; 4],  // xy=width/height (metres), z>0.5=screen-space mode
    pub color:       [f32; 4],  // linear RGBA tint
}
```

---

## License

[LICENSE](LICENSE)
