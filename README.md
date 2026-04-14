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

*GPU-driven deferred rendering in pure Rust — physically-based shading, radiance cascades global illumination, cascaded shadow maps, Nanite-style virtual geometry, and a fully modular render graph.*

Sample model from https://sketchfab.com/mohamedhussien

</div>

Helio is a GPU-driven deferred rendering engine written entirely in Rust on top of `wgpu`. The scene API is handle-based — every resource (`MeshId`, `MaterialId`, `ObjectId`, `LightId`, …) is a lightweight stable handle backed by a generational arena. All CPU-side calls are bounded — typically O(1) — while culling, indirect-draw dispatch, and light evaluation happen entirely on the GPU.

<img width="2476" height="941" alt="image" src="https://github.com/user-attachments/assets/7034e23e-1c0a-4344-b8a3-e3bf36666047" />
<img width="2555" height="1340" alt="image" src="https://github.com/user-attachments/assets/f3f9f878-9a64-4f7b-b4fa-29f14d78250c" />
<img width="1868" height="1017" alt="image" src="https://github.com/user-attachments/assets/46c36b06-1e49-40f4-9bbc-edf4a6d003a7" />

---

## Workspace structure

```
crates/helio               ← public API: Renderer, Scene, Camera, editor tools
crates/helio-v3            ← render graph runtime, GpuScene, RenderPass trait
crates/libhelio            ← GPU-shared structs (GpuLight, GpuMaterial, uniforms…)
crates/helio-pass-*        ← one crate per render pass, plain Rust types
crates/helio-asset-compat  ← FBX / glTF / OBJ / USD loading via SolidRS
crates/examples            ← runnable native demos
```

| Crate | Role |
|---|---|
| `helio` | `Renderer` + `Scene` — typed handles, group visibility, the default deferred graph |
| `helio-v3` | `RenderGraph`, `PassContext`, dirty-tracked GPU buffers, CPU/GPU profiling |
| `libhelio` | Shared GPU types used by all pass crates |
| `helio-asset-compat` | `load_scene_file` → `ConvertedScene` (meshes, materials, textures, lights) |

---

## Quick start

### Run an example

```sh
cargo run -p examples --bin indoor_cathedral --release
cargo run -p examples --bin outdoor_city --release
cargo run -p examples --bin load_fbx --release -- path/to/model.fbx
```

### Add to your project

```toml
[dependencies]
helio = { path = "crates/helio" }
helio-asset-compat = { path = "crates/helio-asset-compat" }
wgpu  = "28"
winit = "0.30"
```

### Initialise the renderer

```rust
use helio::{
    required_wgpu_features, required_wgpu_limits,
    Camera, Renderer, RendererConfig, ShadowQuality,
};

// Pass these to request_device so the required extensions are available.
let features = required_wgpu_features(adapter.features());
let limits   = required_wgpu_limits(adapter.limits());

let mut renderer = Renderer::new(
    device.clone(),
    queue.clone(),
    RendererConfig::new(width, height, surface_format),
);

renderer.set_editor_mode(true);                     // enable grid + editor icons
renderer.set_clear_color([0.08, 0.09, 0.12, 1.0]);
renderer.set_ambient([0.12, 0.14, 0.18], 0.25);
```

Required wgpu features: `TEXTURE_BINDING_ARRAY` + `SAMPLED_TEXTURE_AND_STORAGE_BUFFER_ARRAY_NON_UNIFORM_INDEXING`.  
Optional (used when the adapter supports them): `MULTI_DRAW_INDIRECT`, `MULTI_DRAW_INDIRECT_COUNT`, `SHADER_PRIMITIVE_INDEX`.

### Populate a scene and render

Everything in the scene — meshes, objects, lights — is placed via `insert_actor`:

```rust
use helio::{
    Camera, GpuMaterial, GroupMask, Movability, MeshUpload,
    ObjectDescriptor, PackedVertex, SceneActor,
};
use glam::{Mat4, Vec3};

// 1. Upload a mesh (returns SceneActorId::Mesh — call .as_mesh() for the MeshId).
let mesh_id = renderer
    .scene_mut()
    .insert_actor(SceneActor::mesh(MeshUpload { vertices, indices }))
    .as_mesh()
    .unwrap();

// 2. Create a PBR material.
let mat_id = renderer.scene_mut().insert_material(GpuMaterial {
    base_color:         [0.9, 0.15, 0.15, 1.0],
    emissive:           [0.0, 0.0, 0.0, 0.0],
    roughness_metallic: [0.5, 0.0, 1.5, 0.5],
    tex_base_color:     GpuMaterial::NO_TEXTURE,
    tex_normal:         GpuMaterial::NO_TEXTURE,
    tex_roughness:      GpuMaterial::NO_TEXTURE,
    tex_emissive:       GpuMaterial::NO_TEXTURE,
    tex_occlusion:      GpuMaterial::NO_TEXTURE,
    workflow: 0,
    flags:    0,
    _pad:     0,
});

let transform = Mat4::from_translation(Vec3::new(0.0, 1.0, -3.0));

// 3. Place an object in the scene.
let obj_id = renderer
    .scene_mut()
    .insert_actor(SceneActor::object(ObjectDescriptor {
        mesh:       mesh_id,
        material:   mat_id,
        transform,
        bounds:     [0.0, 1.0, -3.0, 1.5],   // [cx, cy, cz, radius]
        flags:      0,
        groups:     GroupMask::NONE,
        movability: Some(Movability::Movable),
    }))
    .as_object()
    .unwrap();

// 4. Add a point light.
let _light_id = renderer
    .scene_mut()
    .insert_actor(SceneActor::light(GpuLight {
        position_range:  [0.0, 4.0, 0.0, 20.0],
        direction_outer: [0.0, -1.0, 0.0, 0.0],
        color_intensity: [1.0, 0.9, 0.8, 12.0],
        shadow_index:    0,
        light_type:      LightType::Point as u32,
        inner_angle:     0.0,
        _pad:            0,
    }))
    .as_light()
    .unwrap();

// 5. Render.
let camera = Camera::perspective_look_at(eye, target, Vec3::Y, fov_y, aspect, 0.1, 1000.0);
renderer.render(&camera, &surface_view)?;
```

---

## Renderer API

### Configuration

```rust
renderer.set_shadow_quality(ShadowQuality::Ultra);
renderer.set_render_size(width, height);
renderer.set_clear_color([0.05, 0.05, 0.08, 1.0]);
renderer.set_ambient([0.12, 0.14, 0.18], 0.25);
renderer.set_editor_mode(true);    // enables grid + billboarded light icons
renderer.set_debug_mode(0);        // 0=off  10=shadow heatmap  11=light depth
```

### Meshes

```rust
use helio::{MeshUpload, PackedVertex, SceneActor};

// Build a MeshUpload from packed vertices + u32 indices.
let upload = MeshUpload { vertices: vec![...], indices: vec![0, 1, 2] };

// Upload — returns SceneActorId; use .as_mesh() for the stable MeshId.
let mesh_id = renderer
    .scene_mut()
    .insert_actor(SceneActor::mesh(upload))
    .as_mesh()
    .unwrap();

// Remove
renderer.scene_mut().remove_mesh(mesh_id)?;
```

`PackedVertex::from_components(position, normal, uv, tangent, tangent_sign)` is the canonical way to build vertices.

### Materials

```rust
use helio::GpuMaterial;

let mat_id = renderer.scene_mut().insert_material(GpuMaterial {
    base_color:         [0.55, 0.55, 0.55, 1.0],  // linear RGBA
    emissive:           [0.0, 0.0, 0.0, 0.0],     // xyz=color, w=strength
    roughness_metallic: [0.8, 0.0, 1.5, 0.5],     // roughness, metallic, IOR, specular_tint
    tex_base_color:     GpuMaterial::NO_TEXTURE,
    tex_normal:         GpuMaterial::NO_TEXTURE,
    tex_roughness:      GpuMaterial::NO_TEXTURE,
    tex_emissive:       GpuMaterial::NO_TEXTURE,
    tex_occlusion:      GpuMaterial::NO_TEXTURE,
    workflow: 0,   // 0 = Metallic-Roughness
    flags:    0,   // bit0=double-sided  bit1=alpha-blend  bit2=alpha-test
    _pad:     0,
});

renderer.scene_mut().update_material(mat_id, updated_material)?;
renderer.scene_mut().remove_material(mat_id)?;
```

### Textures

```rust
use helio::TextureUpload;

let tex_id = renderer.scene_mut().insert_texture(TextureUpload {
    data:   rgba_bytes,
    width:  1024,
    height: 1024,
    format: wgpu::TextureFormat::Rgba8UnormSrgb,
})?;

renderer.scene_mut().remove_texture(tex_id)?;
```

### Lights

```rust
use helio::{GpuLight, LightType, SceneActor};

let light_id = renderer
    .scene_mut()
    .insert_actor(SceneActor::light(GpuLight {
        position_range:  [2.0, 3.0, 0.0, 15.0],      // xyz=pos, w=range (m)
        direction_outer: [0.0, -1.0, 0.0, 0.0],
        color_intensity: [1.0, 0.85, 0.6, 80.0],     // xyz=linear sRGB, w=intensity
        shadow_index:    0,                            // 0=shadowed, u32::MAX=shadowless
        light_type:      LightType::Point as u32,
        inner_angle:     0.0,
        _pad:            0,
    }))
    .as_light()
    .unwrap();

renderer.scene_mut().update_light(light_id, updated_gpu_light)?;
renderer.scene_mut().remove_light(light_id)?;
```

### Objects

```rust
use helio::{GroupMask, Movability, ObjectDescriptor, SceneActor};

let obj_id = renderer
    .scene_mut()
    .insert_actor(SceneActor::object(ObjectDescriptor {
        mesh:       mesh_id,
        material:   mat_id,
        transform:  Mat4::from_translation(Vec3::new(1.0, 0.0, -2.0)),
        bounds:     [1.0, 0.0, -2.0, 1.2],  // [cx, cy, cz, radius]
        flags:      0,
        groups:     GroupMask::NONE,
        movability: Some(Movability::Movable),
    }))
    .as_object()
    .unwrap();

renderer.scene_mut().update_object_transform(obj_id, new_transform)?;
renderer.scene_mut().remove_object(obj_id)?;
```

### Multi-section meshes (Unreal-style)

One shared vertex buffer, N index ranges, N draw calls per instance — all sections translate, rotate, scale, and are picked as a single unit. The scene stores instances in a generational pool (`SectionedInstanceId`) and maintains a reverse map so the picker resolves any section hit to the parent instance automatically.

```rust
use helio::{Movability, SectionedMeshUpload};
use helio_asset_compat::{
    load_scene_bytes_with_config, upload_sectioned_scene, LoadConfig,
};

// Load an FBX with merge_meshes=true to produce a ConvertedSectionedMesh.
let scene = load_scene_bytes_with_config(
    include_bytes!("model.fbx"),
    "fbx",
    None,
    LoadConfig::default()
        .with_merge_meshes(true)
        .with_import_scale(glam::Vec3::splat(1.0 / 100.0)),
)?;

// Upload textures, materials, and the shared vertex buffer in one call.
let (multi_mesh_id, section_mat_ids) = upload_sectioned_scene(&mut renderer, &scene)?;

// Place an instance — returns a Copy SectionedInstanceId handle.
let sm = scene.sectioned_mesh.as_ref().unwrap();
let inst_id = renderer.scene_mut().insert_sectioned_object(
    multi_mesh_id,
    &section_mat_ids,
    glam::Mat4::IDENTITY,           // world transform
    [0.0, 0.0, 0.0, 2.0],          // bounding sphere [cx, cy, cz, radius]
    Some(Movability::Movable),
)?;

// Move the whole instance — all N sections update atomically (O(N sections)).
renderer.scene_mut().update_sectioned_object_transform(inst_id, new_transform)?;

// Register each section's geometry with the picker for accurate BVH ray-casting.
if let Some(section_ids) = renderer.scene().sectioned_section_mesh_ids(multi_mesh_id) {
    let section_ids: Vec<_> = section_ids.to_vec();
    for (section_mesh_id, sec) in section_ids.iter().zip(sm.sections.iter()) {
        picker.register_mesh(*section_mesh_id, &helio::MeshUpload {
            vertices: sm.vertices.clone(),
            indices:  sec.indices.clone(),
        });
    }
}

// Remove
renderer.scene_mut().remove_sectioned_object(inst_id)?;
```

### Custom render passes

```rust
renderer.add_pass(Box::new(MyPass::new(&device)));
renderer.use_default_graph();   // resets to built-in deferred pipeline
```

---

## Scene API

Access via `renderer.scene()` / `renderer.scene_mut()`.

### Camera

```rust
use helio::Camera;
use glam::Vec3;

let camera = Camera::perspective_look_at(
    Vec3::new(0.0, 2.0, 6.0),  // eye
    Vec3::ZERO,                  // target
    Vec3::Y,                     // up
    60_f32.to_radians(),         // fov_y
    width as f32 / height as f32,
    0.1, 1000.0,
);
```

### Object descriptor

```rust
pub struct ObjectDescriptor {
    pub mesh:       MeshId,
    pub material:   MaterialId,
    pub transform:  Mat4,
    pub bounds:     [f32; 4],            // [cx, cy, cz, radius] world-space bounding sphere
    pub flags:      u32,                 // reserved — pass 0
    pub groups:     GroupMask,           // GroupMask::NONE = always visible
    pub movability: Option<Movability>,  // None = static, Some(Movable) = dynamic
}
```

### Lights

```rust
use helio::{GpuLight, LightType};

// Point light
GpuLight {
    position_range:  [x, y, z, range_metres],
    direction_outer: [0.0, -1.0, 0.0, 0.0],
    color_intensity: [r, g, b, intensity],
    shadow_index:    0,
    light_type:      LightType::Point as u32,
    inner_angle:     0.0,
    _pad:            0,
}

// Spot light
GpuLight {
    position_range:  [x, y, z, range],
    direction_outer: [dx, dy, dz, outer_angle.cos()],
    color_intensity: [r, g, b, intensity],
    shadow_index:    0,
    light_type:      LightType::Spot as u32,
    inner_angle:     inner_angle.cos(),
    _pad:            0,
}

// Directional light
GpuLight {
    position_range:  [0.0, 0.0, 0.0, f32::MAX],
    direction_outer: [dx, dy, dz, 0.0],
    color_intensity: [r, g, b, intensity],
    shadow_index:    0,
    light_type:      LightType::Directional as u32,
    inner_angle:     0.0,
    _pad:            0,
}
```

### Materials (PBR)

```rust
// Metallic-roughness PBR (workflow = 0)
GpuMaterial {
    base_color:         [r, g, b, a],        // linear RGBA
    emissive:           [r, g, b, strength],
    roughness_metallic: [roughness, metallic, ior, specular_tint],
    tex_base_color:     GpuMaterial::NO_TEXTURE,
    tex_normal:         GpuMaterial::NO_TEXTURE,
    tex_roughness:      GpuMaterial::NO_TEXTURE,
    tex_emissive:       GpuMaterial::NO_TEXTURE,
    tex_occlusion:      GpuMaterial::NO_TEXTURE,
    workflow: 0,
    flags:    0,   // bit0=double-sided  bit1=alpha-blend  bit2=alpha-test
    _pad:     0,
}
```

---

## Groups

Every object carries a `GroupMask` (64-bit bitmask). The scene maintains a hidden-group mask; an object is culled when any of its groups overlaps the hidden set. `GroupMask::NONE` is always visible.

### Built-in groups

| Constant | Description |
|---|---|
| `GroupId::EDITOR` | Editor helpers — light icons, gizmos. Hide at ship time. |
| `GroupId::DEFAULT` | General objects |
| `GroupId::STATIC` | Non-moving world geometry |
| `GroupId::DYNAMIC` | Animated / physics objects |
| `GroupId::WORLD_UI` | World-space UI |
| `GroupId::VFX` | Particles and effects |
| `GroupId::SHADOW_CASTERS` | Mass shadow toggle for prop layers |
| `GroupId::DEBUG` | Debug visualisers |

### Group API

```rust
renderer.hide_group(GroupId::EDITOR);
renderer.show_group(GroupId::EDITOR);
renderer.is_group_hidden(GroupId::EDITOR);

renderer.scene_mut().set_object_groups(obj_id, GroupMask::NONE.with(GroupId::STATIC));
renderer.scene_mut().add_object_to_group(obj_id, GroupId::STATIC);
renderer.scene_mut().remove_object_from_group(obj_id, GroupId::STATIC);

// GPU-side mass transform — O(N objects in the group)
renderer.scene_mut().move_group(GroupId::DYNAMIC, Mat4::from_translation(delta));

// Bitmask composition
let mask = GroupMask::NONE
    .with(GroupId::STATIC)
    .with(GroupId::SHADOW_CASTERS);
mask.contains(GroupId::STATIC);
let combined = mask_a | mask_b;
```

---

## Default render pipeline

The built-in deferred pipeline runs these passes in order:

| # | Pass | Kind | What it does |
|---|---|---|---|
| 1 | `ShadowMatrixPass` | Compute | Per-light face view-proj matrices → `shadow_matrices` buffer |
| 2 | `ShadowPass` | Render | Depth-only → 512×512×256 shadow atlas |
| 3 | `SkyLutPass` | Render | Hillaire 2020 atmospheric panoramic LUT (192×108) |
| 4 | `DepthPrepassPass` | Render | Early-Z via `multi_draw_indexed_indirect`; O(1) CPU |
| 5 | `GBufferPass` | Render | GPU-driven → albedo / normal+F0 / ORM / emissive G-buffer |
| 5b | `VirtualGeometryPass` | Compute + Render | Meshlet frustum + backface culling → indirect draw into same G-buffer |
| 6 | `DeferredLightPass` | Fullscreen | Cook-Torrance BRDF, PCF/PCSS CSM (4 cascades), RC GI, tone mapping |
| 7 | `BillboardPass` | Render | Editor light icons + user billboards, alpha-blended |

### Shadow quality

| Preset | PCF samples | PCSS | Blocker | Filter |
|---|---|---|---|---|
| `Low` | 8 | off | 8 | 8 |
| `Medium` | 16 | off | 8 | 8 |
| `High` | 16 | on | 8 | 16 |
| `Ultra` | 32 | on | 16 | 32 |

```rust
renderer.set_shadow_quality(ShadowQuality::Ultra);
```

### Virtual geometry

`VirtualGeometryPass` runs GPU-side per-meshlet frustum culling, backface-cone culling, and screen-coverage LOD selection every frame. CPU cost is O(1).

```rust
use helio::{SceneActor, VirtualObjectDescriptor};

let virt_id = renderer
    .scene_mut()
    .insert_actor(SceneActor::virtual_mesh(virtual_mesh_upload))
    .as_virtual_mesh()
    .unwrap();

let vobj_id = renderer
    .scene_mut()
    .insert_actor(SceneActor::virtual_object(VirtualObjectDescriptor {
        virtual_mesh: virt_id,
        material_id:  0,
        transform:    Mat4::IDENTITY,
        bounds:       [0.0, 0.0, 0.0, 2.0],
        flags:        0,
        groups:       GroupMask::NONE,
    }))
    .as_virtual_object()
    .unwrap();
```

---

## Pass crates reference

| Crate | Pass type | Summary |
|---|---|---|
| `helio-pass-depth-prepass` | `DepthPrepassPass` | Early-Z, O(1) CPU |
| `helio-pass-gbuffer` | `GBufferPass` | GPU-driven G-buffer fill |
| `helio-pass-deferred-light` | `DeferredLightPass` | BRDF + shadows + RC GI + tone mapping |
| `helio-pass-shadow` | `ShadowPass` | 512×512×256 shadow atlas |
| `helio-pass-shadow-matrix` | `ShadowMatrixPass` | Per-light face matrices |
| `helio-pass-sky-lut` | `SkyLutPass` | Atmospheric LUT bake |
| `helio-pass-sky` | `SkyPass` | Fullscreen atmospheric background |
| `helio-pass-virtual-geometry` | `VirtualGeometryPass` | Meshlet cull + LOD + indirect draw |
| `helio-pass-radiance-cascades` | `RadianceCascadesPass` | Probe-based GI |
| `helio-pass-sdf` | `SdfClipmapPass` | 8-level toroidal SDF clipmap + sphere-trace |
| `helio-pass-billboard` | `BillboardPass` | Up to 65 536 instanced camera-facing quads |
| `helio-pass-transparent` | `TransparentPass` | Alpha-blended forward over depth-read-only |
| `helio-pass-fxaa` | `FxaaPass` | Fullscreen FXAA |
| `helio-pass-smaa` | `SmaaPass` | SMAA 1× |
| `helio-pass-taa` | `TaaPass` | Temporal AA with jitter + reprojection |
| `helio-pass-ssao` | `SsaoPass` | Screen-space ambient occlusion |
| `helio-pass-hiz` | `HiZBuildPass` | Min-reduction Hi-Z mip chain |
| `helio-pass-occlusion-cull` | `OcclusionCullPass` | GPU occlusion culling via Hi-Z |
| `helio-pass-debug` | `DebugShapesPass` | Lines, boxes, spheres, capsules, cones |
| `helio-pass-indirect-dispatch` | `IndirectDispatchPass` | Builds `DrawIndexedIndirect` buffers from the draw list |
| `helio-pass-light-cull` | `LightCullPass` | Tile/cluster light culling |
| `helio-pass-simple-cube` | `SimpleCubePass` | Hardcoded debug cube, no scene required |

---

## Asset loading

```rust
use helio_asset_compat::{
    load_scene_file, load_scene_file_with_config,
    load_scene_bytes_with_config,
    upload_scene, upload_sectioned_scene,
    load_and_upload_scene, LoadConfig,
};

// Load from disk — format detected from extension (FBX, glTF, OBJ, USDC).
let scene = load_scene_file("assets/prop.fbx")?;

// Load from embedded bytes.
let scene = load_scene_bytes_with_config(
    include_bytes!("prop.fbx"),
    "fbx",
    None,                               // optional base dir for texture resolution
    LoadConfig::default()
        .with_uv_flip(false)
        .with_import_scale(glam::Vec3::splat(1.0 / 100.0)),
)?;

// Upload all meshes + materials in one pass → UploadedScene { mesh_ids, material_ids }
let uploaded = upload_scene(&mut renderer, &scene)?;

// Convenience: load + upload together.
let uploaded = load_and_upload_scene("assets/prop.fbx", LoadConfig::default(), &mut renderer)?;

// Sectioned (multi-material) upload — requires merge_meshes = true.
let scene = load_scene_file_with_config(
    "assets/prop.fbx",
    LoadConfig::default().with_merge_meshes(true),
)?;
let (multi_mesh_id, section_mat_ids) = upload_sectioned_scene(&mut renderer, &scene)?;
```

`ConvertedScene` holds:
- `meshes: Vec<ConvertedMesh>` — `PackedVertex` arrays + per-mesh material index
- `sectioned_mesh: Option<ConvertedSectionedMesh>` — shared vertices + per-section index lists (present when `merge_meshes = true`)
- `textures: Vec<TextureUpload>`
- `materials: Vec<ConvertedMaterial>`
- `lights: Vec<GpuLight>`
- `cameras: Vec<CameraData>`

---

## Examples

| Binary | What it covers |
|---|---|
| `simple_graph` | Fly camera + hardcoded debug cube |
| `indoor_room` | Furnished room with point lights |
| `indoor_corridor` | Hallway — fluorescents, exit signs, wall sconces |
| `indoor_cathedral` | Gothic nave, RC GI, stained-glass light shafts |
| `indoor_server_room` | Data-centre; `E` toggles editor icons |
| `outdoor_night` | Night-time plaza |
| `outdoor_canyon` | Desert canyon; `Q/E` rotates sun |
| `outdoor_city` | Dense city block at dusk |
| `outdoor_volcano` | Volcanic island with lava-glow lights |
| `outdoor_rocks` | Rock scatter + virtual geometry + FBX ship |
| `space_station` | Massive orbital station, 40 m/s fly speed |
| `load_fbx` | Drop-in viewer for any FBX/glTF/OBJ/USDC |
| `load_fbx_embedded` | Same with `include_bytes!` |
| `sdf_demo` | Live-editable SDF clipmap ray march |
| `light_benchmark` | 150 simultaneous point lights |
| `rc_benchmark` | Cornell box — multi-bounce RC GI |
| `debug_shapes` | All debug primitives |
| `editor_demo` | Interactive scene editor — pick, translate, rotate, scale, duplicate (supports multi-section meshes) |

```sh
cargo run -p examples --bin indoor_cathedral --release
cargo run -p examples --bin editor_demo --release
cargo run -p examples --bin load_fbx --release -- path/to/model.fbx

# Compile-check without running
cargo check -p helio -p examples --quiet
```

---

## GPU layout reference

### `GpuCameraUniforms` — offsets for custom WGSL shaders

| Field | Byte offset | Description |
|---|---|---|
| `view` | 0 | World → view (mat4x4) |
| `proj` | 64 | View → clip (mat4x4) |
| `view_proj` | 128 | Combined VP (mat4x4) |
| `inv_view_proj` | 192 | Clip → world (mat4x4) |
| `position_near` | 256 | xyz=camera pos, w=near |
| `forward_far` | 272 | xyz=forward dir, w=far |
| `jitter_frame` | 288 | xy=TAA jitter, z=frame index |
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


[MIT](LICENSE)
