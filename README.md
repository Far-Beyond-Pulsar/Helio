<div align="center">

<img src="./branding/Helio.svg" alt="Helio Renderer" width="400"/>

**GPU-driven deferred rendering in pure Rust**

[![Rust](https://img.shields.io/badge/rust-stable-orange?logo=rust)](https://www.rust-lang.org/)
[![wgpu](https://img.shields.io/badge/wgpu-28-blue)](https://wgpu.rs/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

</div>

Helio is a GPU-driven deferred renderer written entirely in Rust on top of `wgpu`. Every CPU-side call is bounded (typically O(1)) while culling, LOD selection, indirect-draw dispatch, and light evaluation happen entirely on the GPU. The scene API is handle-based — every resource (`MeshId`, `MaterialId`, `ObjectId`, `LightId`, …) is a lightweight stable handle backed by a generational arena.

<img width="2476" height="941" alt="image" src="https://github.com/user-attachments/assets/7034e23e-1c0a-4344-b8a3-e3bf36666047" />
<img width="2555" height="1340" alt="image" src="https://github.com/user-attachments/assets/f3f9f878-9a64-4f7b-b4fa-29f14d78250c" />
<img width="1868" height="1017" alt="image" src="https://github.com/user-attachments/assets/46c36b06-1e49-40f4-9bbc-edf4a6d003a7" />

---

## Features

### GPU-driven pipeline
- **GPU-driven GBuffer** — culling, LOD, and indirect-draw dispatch all on the GPU. CPU cost is O(1) regardless of draw-call count.
- **Virtual geometry** — per-meshlet frustum culling, backface-cone culling, and coverage-based LOD selection, entirely on the GPU.
- **Occlusion culling** — Hi-Z min-reduction pyramid + GPU occlusion tests for shadow casters and scene geometry.
- **Indirect multi-draw** — `multi_draw_indexed_indirect` for depth prepass, GBuffer, and shadow maps.

### Lighting & shading
- **Physically-based shading** — Cook-Torrance BRDF with metallic-roughness workflow, IOR, specular tint.
- **Radiance Cascades GI** — multi-bounce probe-based global illumination with dual-tier (near RC, far ambient).
- **Cascaded shadow maps** — 4-split CSM with PCF/PCSS filtering, configurable quality presets.
- **Tile/cluster light culling** — light indices binned per tile for O(tiles) evaluation.
- **Screen-space ambient occlusion** — with baked-AO override support.
- **Volumetric sky** — Hillaire 2020 atmospheric model with cloud layer support.

### Post-processing
- **Anti-aliasing** — TAA (with jitter + reprojection), FXAA, or SMAA 1x.
- **Tone mapping** — integrated into deferred light pass.
- **Debug visualisations** — UV, normals, albedo, shadow heatmap, LOD heatmap, and more.

### Scene management
- **Handle-based API** — every resource is a `Copy` stable handle (`MeshId`, `MaterialId`, `LightId`, `ObjectId`, …).
- **Group system** — 64-bit bitmask per object; per-group hide/show, transform, and culling.
- **Sectioned meshes** — single vertex buffer with N index ranges (Unreal-style multi-material), O(N sections) transform updates.
- **GPU-native scene** — all scene state lives on the GPU with dirty-tracked CPU mirrors. `flush()` uploads only changed data.

### Modular architecture
- **Pluggable render passes** — each pass is a separate crate implementing `RenderPass`. Swap, add, or remove passes at will.
- **Render graph safety** — debug-build tracking catches unwritten resources; static dependency validation prevents ordering bugs.
- **Zero-copy contexts** — passes receive borrowed references to GPU resources, never owned copies.
- **Automatic profiling** — CPU scopes and GPU timestamps injected per pass, zero manual instrumentation.

### Asset pipeline
- **Multi-format import** — FBX, glTF, OBJ, USD via SolidRS.
- **Pre-baked lighting** — baked AO, lightmaps, reflection probes, and irradiance SH for static geometry.
- **PVS** — pre-computed potentially-visible sets for CPU-side culling.

---

## Quick start

```sh
cargo run -p examples --bin indoor_cathedral --release
cargo run -p examples --bin outdoor_city --release
cargo run -p examples --bin load_fbx --release -- path/to/model.fbx
```

### Minimal setup

```rust
use helio::{Camera, Renderer, RendererConfig, required_wgpu_features, required_wgpu_limits};

let features = required_wgpu_features(adapter.features());
let limits   = required_wgpu_limits(adapter.limits());

let mut renderer = Renderer::new(
    device.clone(),
    queue.clone(),
    RendererConfig::new(width, height, surface_format),
);

let camera = Camera::perspective_look_at(
    glam::Vec3::new(0.0, 2.0, 6.0), glam::Vec3::ZERO, glam::Vec3::Y,
    60_f32.to_radians(), width as f32 / height as f32, 0.1, 1000.0,
);

renderer.render(&camera, &surface_view)?;
```

### Populating a scene

```rust
use helio::{GpuLight, GpuMaterial, MeshUpload, ObjectDescriptor, PackedVertex, SceneActor};
use helio::{GroupMask, LightType, Movability};
use glam::Mat4;

// Upload a mesh
let mesh_id = renderer
    .scene_mut()
    .insert_actor(SceneActor::mesh(MeshUpload { vertices, indices }))
    .as_mesh()?;

// Create a material
let mat_id = renderer.scene_mut().insert_material(GpuMaterial {
    base_color:         [0.9, 0.15, 0.15, 1.0],
    roughness_metallic: [0.5, 0.0, 1.5, 0.5],
    ..Default::default()
});

// Place an object
let _obj_id = renderer
    .scene_mut()
    .insert_actor(SceneActor::object(ObjectDescriptor {
        mesh: mesh_id,
        material: mat_id,
        transform: Mat4::from_translation(glam::Vec3::new(0.0, 1.0, -3.0)),
        bounds: [0.0, 1.0, -3.0, 1.5],
        movability: Some(Movability::Movable),
        ..Default::default()
    }));

// Add a point light
let _light_id = renderer
    .scene_mut()
    .insert_actor(SceneActor::light(GpuLight {
        position_range:  [0.0, 4.0, 0.0, 20.0],
        color_intensity: [1.0, 0.9, 0.8, 12.0],
        light_type:      LightType::Point as u32,
        ..Default::default()
    }));
```

---

## Architecture

```
crates/helio               ── Public API: Renderer, Scene, Camera, editor tools
crates/helio-v3            ── Render graph runtime, GpuScene, RenderPass trait
crates/libhelio            ── GPU-shared types (GpuLight, GpuMaterial, uniforms, Tracked<T>)
crates/helio-pass-*        ── One crate per render pass
crates/helio-asset-compat  ── FBX / glTF / OBJ / USD loading
crates/examples            ── Runnables demos
```

The render graph executes passes in order. Each pass receives a zero-copy context with borrowed GPU resources and publishes its outputs into a shared `FrameResources` struct for downstream passes:

```
Renderer::render()

  ├─ Scene::flush()          ── dirty-tracked GPU upload (no-op at steady state)
  ├─ ShadowMatrixPass        ── per-light face VP matrices (compute)
  ├─ ShadowPass              ── depth-only → shadow atlas (render)
  ├─ SkyLutPass              ── atmospheric LUT (render)
  ├─ DepthPrepass            ── early-Z (indirect multi-draw)
  ├─ GBufferPass             ── albedo/normal/ORM/emissive (GPU-driven)
  ├─ VirtualGeometryPass     ── meshlet cull + LOD + indirect draw
  ├─ DeferredLightPass       ── BRDF + CSM + RC GI + tone map (fullscreen)
  ├─ SSAO / HiZ / LightCull  ── auxiliary passes
  ├─ BillboardPass           ── editor icons + user billboards
  ├─ Transparency            ── alpha-blended forward pass
  ├─ Post-process (TAA/FXAA/SMAA)
  └─ Queue submit
```

### Render graph safety

Helio's `Tracked<T>` wrapper (Phase 1) catches unwritten-resource bugs at runtime in debug builds — if a pass skips writing `gbuffer` when `draw_count == 0`, downstream passes immediately panic with the reader's name. Phase 2 adds static dependency declarations: each pass declares `reads()` / `writes()`, and the graph validates the DAG at construction time. Future phases will add automatic GPU barriers and parallel pass execution.

---

## Scene API

### Groups

Every object carries a `GroupMask` (64-bit bitmask). Objects are culled when any group overlaps the hidden set. `GroupMask::NONE` is always visible.

```rust
renderer.hide_group(GroupId::EDITOR);
renderer.show_group(GroupId::STATIC);
renderer.scene_mut().move_group(GroupId::DYNAMIC, Mat4::from_translation(delta));
renderer.scene_mut().set_object_groups(obj_id, GroupMask::NONE.with(GroupId::STATIC));
```

| Constant | Purpose |
|---|---|
| `GroupId::EDITOR` | Editor helpers — light icons, gizmos |
| `GroupId::STATIC` | Non-moving world geometry |
| `GroupId::DYNAMIC` | Animated / physics objects |
| `GroupId::SHADOW_CASTERS` | Mass shadow toggle |
| `GroupId::DEBUG` | Debug visualisers |

### Lights

```rust
// Point light
GpuLight {
    position_range:  [x, y, z, range],
    color_intensity: [r, g, b, intensity],
    light_type:      LightType::Point as u32,
    ..Default::default()
}

// Spot light
GpuLight {
    position_range:  [x, y, z, range],
    direction_outer: [dx, dy, dz, outer_angle.cos()],
    color_intensity: [r, g, b, intensity],
    light_type:      LightType::Spot as u32,
    inner_angle:     inner_angle.cos(),
    ..Default::default()
}

// Directional light
GpuLight {
    position_range:  [0.0, 0.0, 0.0, f32::MAX],
    direction_outer: [dx, dy, dz, 0.0],
    color_intensity: [r, g, b, intensity],
    light_type:      LightType::Directional as u32,
    ..Default::default()
}
```

### Materials

```rust
GpuMaterial {
    base_color:         [0.55, 0.55, 0.55, 1.0],  // linear RGBA
    emissive:           [0.0, 0.0, 0.0, 0.0],     // rgb + strength
    roughness_metallic: [0.8, 0.0, 1.5, 0.5],     // roughness, metallic, ior, specular_tint
    tex_base_color:     GpuMaterial::NO_TEXTURE,
    workflow: 0,  // 0 = Metallic-Roughness
    flags:    0,  // bit0=double-sided  bit1=alpha-blend  bit2=alpha-test
}
```

### Textures

```rust
let tex_id = renderer.scene_mut().insert_texture(TextureUpload {
    data:   rgba_bytes,
    width:  1024,
    height: 1024,
    format: wgpu::TextureFormat::Rgba8UnormSrgb,
})?;
```

### Sectioned meshes

Single vertex buffer, N index ranges, N draw calls per instance. All sections share one transform, one pickable identity.

```rust
use helio_asset_compat::{load_scene_bytes_with_config, upload_sectioned_scene, LoadConfig};

let scene = load_scene_bytes_with_config(
    include_bytes!("model.fbx"), "fbx", None,
    LoadConfig::default().with_merge_meshes(true).with_import_scale(glam::Vec3::splat(0.01)),
)?;
let (mesh_id, mat_ids) = upload_sectioned_scene(&mut renderer, &scene)?;
let inst_id = renderer.scene_mut().insert_sectioned_object(
    mesh_id, &mat_ids, Mat4::IDENTITY, [0.0; 4], Some(Movability::Movable),
)?;
renderer.scene_mut().update_sectioned_object_transform(inst_id, new_transform)?;
```

---

## Shadow quality presets

| Preset | PCF samples | PCSS | Blocker | Filter |
|---|---|---|---|---|
| `Low` | 8 | off | 8 | 8 |
| `Medium` | 16 | off | 8 | 8 |
| `High` | 16 | on | 8 | 16 |
| `Ultra` | 32 | on | 16 | 32 |

---

## Pass reference

| Crate | Pass | Description |
|---|---|---|
| `helio-pass-depth-prepass` | `DepthPrepassPass` | Early-Z, O(1) CPU |
| `helio-pass-gbuffer` | `GBufferPass` | GPU-driven G-buffer fill |
| `helio-pass-deferred-light` | `DeferredLightPass` | BRDF + shadows + RC GI + tone map |
| `helio-pass-shadow` | `ShadowPass` | 512×512×256 shadow atlas |
| `helio-pass-shadow-matrix` | `ShadowMatrixPass` | Per-light face VP matrices |
| `helio-pass-sky-lut` | `SkyLutPass` | Atmospheric LUT bake |
| `helio-pass-sky` | `SkyPass` | Fullscreen atmospheric background |
| `helio-pass-virtual-geometry` | `VirtualGeometryPass` | Meshlet cull + LOD + indirect draw |
| `helio-pass-radiance-cascades` | `RadianceCascadesPass` | Probe-based GI |
| `helio-pass-sdf` | `SdfClipmapPass` | 8-level toroidal SDF clipmap |
| `helio-pass-billboard` | `BillboardPass` | Up to 65K instanced quads |
| `helio-pass-transparent` | `TransparentPass` | Alpha-blended forward |
| `helio-pass-fxaa` | `FxaaPass` | Fullscreen FXAA |
| `helio-pass-smaa` | `SmaaPass` | SMAA 1× |
| `helio-pass-taa` | `TaaPass` | Temporal AA |
| `helio-pass-ssao` | `SsaoPass` | Screen-space ambient occlusion |
| `helio-pass-hiz` | `HiZBuildPass` | Min-reduction Hi-Z mip chain |
| `helio-pass-occlusion-cull` | `OcclusionCullPass` | GPU occlusion culling |
| `helio-pass-debug` | `DebugShapesPass` | Lines, boxes, spheres, capsules |
| `helio-pass-indirect-dispatch` | `IndirectDispatchPass` | Build indirect draw buffers |
| `helio-pass-light-cull` | `LightCullPass` | Tile/cluster light culling |

---

## Debug views

| Mode | Pass | Name | Shows |
|---|---|---|---|
| 1 | GBuffer | UV Visualisation | R=U, G=V |
| 2 | GBuffer | Raw Texture | Raw texels, no material multiply |
| 3 | GBuffer | Geometry Normals | No normal mapping |
| 4 | DeferredLight | Albedo Only | G-buffer albedo without lighting |
| 5 | DeferredLight | World Normals | World-space normals remapped |
| 10 | DeferredLight | Shadow Heatmap | White=lit, black=shadowed |
| 11 | DeferredLight | Light Depth | Light-space depth projection |
| 20 | VirtualGeometry | VG Mesh Triangles | Per-meshlet solid colour |
| 21 | VirtualGeometry | VG LOD Heatmap | Green=LOD0 → red=LOD7 |

Press **F3** / **F4** in `editor_demo_mini` to cycle debug views. Custom passes register views via `debug_views()` on `RenderPass`.

---

## Custom render passes

```rust
renderer.add_pass(Box::new(MyPass::new(&device)));
renderer.use_default_graph();   // reset to built-in pipeline
```

---

## Asset loading

```rust
use helio_asset_compat::{load_scene_file, upload_scene, LoadConfig};

let scene = load_scene_file("assets/prop.fbx")?;
let uploaded = upload_scene(&mut renderer, &scene)?;

// Or one-shot:
let uploaded = helio_asset_compat::load_and_upload_scene(
    "assets/prop.fbx", LoadConfig::default(), &mut renderer,
)?;
```

`ConvertedScene` holds: `meshes`, `sectioned_mesh` (when `merge_meshes = true`), `textures`, `materials`, `lights`, `cameras`.

---

## Examples

| Binary | Description |
|---|---|
| `indoor_cathedral` | Gothic nave with RC GI, stained-glass light shafts |
| `indoor_corridor` | Hallway with fluorescents, exit signs, wall sconces |
| `outdoor_city` | Dense city block at dusk |
| `outdoor_canyon` | Desert canyon, `Q/E` rotates sun |
| `space_station` | Massive orbital station, 40 m/s fly speed |
| `load_fbx` | Drop-in viewer for FBX/glTF/OBJ/USDC |
| `editor_demo` | Interactive scene editor — pick, translate, rotate, scale, duplicate |
| `light_benchmark` | 150 simultaneous point lights |
| `rc_benchmark` | Multi-bounce RC GI Cornell box |
| `sdf_demo` | Live-editable SDF clipmap ray march |
| `debug_shapes` | All debug primitives |

```sh
cargo run -p examples --bin indoor_cathedral --release
cargo run -p examples --bin load_fbx --release -- path/to/model.fbx
cargo check -p helio -p examples --quiet
```

---

## GPU layout reference

### `GpuCameraUniforms` byte offsets

| Offset | Field | Description |
|---|---|---|
| 0 | `view` | World → view (mat4x4) |
| 64 | `proj` | View → clip (mat4x4) |
| 128 | `view_proj` | Combined VP (mat4x4) |
| 192 | `inv_view_proj` | Clip → world (mat4x4) |
| 256 | `position_near` | xyz=camera position, w=near |
| 272 | `forward_far` | xyz=forward, w=far |
| 288 | `jitter_frame` | xy=TAA jitter, z=frame index |
| 304 | `prev_view_proj` | Previous frame VP (mat4x4) |

### `BillboardInstance` (48 bytes)

| Field | Type | Description |
|---|---|---|
| `world_pos` | `[f32; 4]` | xyz = world position |
| `scale_flags` | `[f32; 4]` | xy = width/height (metres), z > 0.5 = screen-space |
| `color` | `[f32; 4]` | Linear RGBA tint |

---

## License

[MIT](LICENSE)
