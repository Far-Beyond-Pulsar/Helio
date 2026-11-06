//! Row components: one SceneDB component per Helio buffer contract, with
//! the contract row's exact fields. Generated from the phase-1 rows; the
//! `same_layout!` assertions fail the build if a row and its component
//! drift apart.

use crate::same_layout;
use pulsar_scenedb_derive::SceneStore;


/// The authoritative SceneDB record consumed by the billboard pass.
///
/// The packed layout is intentional: the shader reads the three fields as one
/// 48-byte instance record. `World::insert`/`get_mut` update this record and
/// SceneDB owns the deferred GPU upload.
#[derive(SceneStore, bytemuck::Pod, bytemuck::Zeroable, Clone, Copy, Debug, PartialEq)]
#[repr(C)]
#[gpu(layout = packed, buffer = "billboard_instances")]
pub struct BillboardComponent {
    /// Where the billboard is, in the frame's space, as every position in a
    /// pass row: metres from the frame's world origin
    /// (`Renderer::world_origin`).
    #[gpu]
    pub world_pos: [f32; 4],
    #[gpu]
    pub scale_flags: [f32; 4],
    #[gpu]
    pub color: [f32; 4],
}
same_layout!(BillboardComponent, helio_pass_billboard::BillboardRow, helio_pass_billboard::BILLBOARD_INSTANCES, "billboard_instances"; world_pos, scale_flags, color);

#[derive(SceneStore, bytemuck::Pod, bytemuck::Zeroable, Clone, Copy, Debug, PartialEq)]
#[repr(C)]
#[gpu(layout = packed, buffer = "reflection_captures")]
pub struct ReflectionCaptureComponent {
    #[gpu]
    pub position_radius: [f32; 4],
    #[gpu]
    pub extents_transition: [f32; 4],
    #[gpu]
    pub world_to_local: [[f32; 4]; 4],
    #[gpu]
    pub cubemap_index: i32,
    #[gpu]
    pub shape: u32,
    #[gpu]
    pub mobility: u32,
    #[gpu]
    pub brightness: f32,
}
same_layout!(ReflectionCaptureComponent, helio_pass_deferred_light::ReflectionCaptureRow, helio_pass_deferred_light::REFLECTION_CAPTURES, "reflection_captures"; position_radius, extents_transition, world_to_local, cubemap_index, shape, mobility, brightness);

/// A placed light, authored as a SceneDB component. Field-for-field
/// identical to `GpuLight` (enforced by the size/align test
/// below) so conversion is a zero-cost `bytemuck::cast`.
#[derive(SceneStore, bytemuck::Pod, bytemuck::Zeroable, Clone, Copy, Debug, PartialEq)]
#[repr(C)]
#[gpu(layout = packed, buffer = "scene_lights")]
pub struct LightComponent {
    #[gpu]
    pub position_range: [f32; 4],
    #[gpu]
    pub direction_outer: [f32; 4],
    #[gpu]
    pub color_intensity: [f32; 4],
    #[gpu]
    pub shadow_index: u32,
    #[gpu]
    pub light_type: u32,
    #[gpu]
    pub inner_angle: f32,
    #[gpu]
    pub _pad: u32,
    #[gpu]
    pub god_rays_enabled: u32,
    #[gpu]
    pub god_rays_density: f32,
    #[gpu]
    pub god_rays_weight: f32,
    #[gpu]
    pub god_rays_decay: f32,
    #[gpu]
    pub god_rays_exposure: f32,
    #[gpu]
    pub flare_enabled: u32,
    #[gpu]
    pub flare_type: u32,
    #[gpu]
    pub flare_intensity: f32,
    #[gpu]
    pub flare_scale: f32,
    #[gpu]
    pub flare_tint_r: f32,
    #[gpu]
    pub flare_tint_g: f32,
    #[gpu]
    pub flare_tint_b: f32,
    #[gpu]
    pub ies_profile_index: i32,
    #[gpu]
    pub light_function_index: i32,
    #[gpu]
    pub ies_angle_scale: f32,
    #[gpu]
    pub ies_angle_offset: f32,
}
same_layout!(LightComponent, helio_pass_forward_lit::LightRow, helio_pass_forward_lit::SCENE_LIGHTS, "scene_lights"; position_range, direction_outer, color_intensity, shadow_index, light_type, inner_angle, _pad, god_rays_enabled, god_rays_density, god_rays_weight, god_rays_decay, god_rays_exposure, flare_enabled, flare_type, flare_intensity, flare_scale, flare_tint_r, flare_tint_g, flare_tint_b, ies_profile_index, light_function_index, ies_angle_scale, ies_angle_offset);

/// A material authored as a SceneDB component.
///
/// The row deliberately keeps the existing G-buffer shader ABI for this
/// migration step, but its lifetime and updates are now owned by the World.
/// Render passes resolve the packed `"materials"` buffer by key; no renderer
/// material table publication is required for the row itself.
#[derive(SceneStore, bytemuck::Pod, bytemuck::Zeroable, Clone, Copy, Debug, PartialEq)]
#[repr(C)]
#[gpu(layout = packed, buffer = "materials")]
pub struct MaterialComponent {
    #[gpu]
    pub base_color: [f32; 4],
    #[gpu]
    pub emissive: [f32; 4],
    #[gpu]
    pub roughness_metallic: [f32; 4],
    #[gpu]
    pub tex_base_color: u32,
    #[gpu]
    pub tex_normal: u32,
    #[gpu]
    pub tex_roughness: u32,
    #[gpu]
    pub tex_emissive: u32,
    #[gpu]
    pub tex_occlusion: u32,
    #[gpu]
    pub workflow: u32,
    #[gpu]
    pub flags: u32,
    #[gpu]
    pub material_class: u32,
    #[gpu]
    pub class_params: [f32; 4],
}
same_layout!(MaterialComponent, helio_pass_gbuffer::MaterialRow, helio_pass_gbuffer::MATERIALS, "materials"; base_color, emissive, roughness_metallic, tex_base_color, tex_normal, tex_roughness, tex_emissive, tex_occlusion, workflow, flags, material_class, class_params);

/// Stable group membership. `group_mask == 0` means always visible.
#[derive(SceneStore, bytemuck::Pod, bytemuck::Zeroable, Clone, Copy, Debug, PartialEq)]
#[repr(C)]
#[gpu(layout = packed, buffer = "render_groups")]
pub struct RenderGroupComponent {
    #[gpu]
    pub group_mask: u64,
}
same_layout!(RenderGroupComponent, helio_pass_gbuffer::RenderGroupRow, helio_pass_gbuffer::RENDER_GROUPS, "render_groups"; group_mask);

/// A single-mesh static-object placement: SceneDB-owned mesh/material row
/// indices, a world transform, a culling bound, and the static draw
/// parameters a GPU-driven instancing/culling pipeline needs to read this
/// row directly with no per-frame CPU involvement.
///
/// `SectionedObjectComponent` above covers the multi-material case, which
/// additionally needs the owning `MultiMeshId`/per-section materials tracked
/// elsewhere.
///
/// # Why the draw parameters are stored, not derived per frame
///
/// `index_count`/`first_index`/`vertex_offset` (the mesh's vertex/index
/// range), and `material_class`/`graph_hash` (the material's
/// pipeline-selection key) are authored alongside the row. They are not
/// resolved through a renderer API; the SceneDB asset/ingestion layer owns
/// that resolution before inserting this component.
///
/// `mesh_slot` and `material_slot` are indices into the keyed SceneDB asset
/// buffers. The draw pipeline consumes those plain indices; no renderer
/// handle type is stored or reconstructed here.
///
/// # What's NOT solved by this component alone
///
/// Building the *grouped* `draw_calls`/`instances`/`aabbs` arrays a
/// GPU-driven culling pass reads (batching same-mesh-same-material
/// instances into one indirect draw call, partitioned into contiguous
/// opaque/transparent/forward ranges by `material_class`) requires sorting
/// every live `StaticObjectComponent` row by `(material_class, graph_hash,
/// mesh_id, material_id)` — a cross-entity operation, not a per-row one.
/// That GPU sort/group/dispatch pipeline is specified but not yet
/// implemented; see the Helio issue tracking it. Until it lands, a
/// consumer must still assemble those arrays itself from a `World` query
/// over this component (a per-frame CPU cost, same category as the
/// pre-existing shadow-atlas scoring — tracked, not silently hidden).
#[derive(SceneStore, bytemuck::Pod, bytemuck::Zeroable, Clone, Copy, Debug, PartialEq)]
#[repr(C)]
#[gpu(layout = packed, buffer = "static_objects")]
pub struct StaticObjectComponent {
    #[gpu]
    pub mesh_slot: u32,
    #[gpu]
    pub mesh_generation: u32,
    #[gpu]
    pub material_slot: u32,
    #[gpu]
    pub material_generation: u32,
    #[gpu]
    pub transform: [[f32; 4]; 4],
    /// Previous frame's `transform`, for TAA/TSR motion vectors. Equal to
    /// `transform` (zero apparent velocity) the frame this row is first
    /// inserted or whenever a caller uses [`Self::with_transform`] without
    /// tracking real per-frame motion -- a quality simplification (a
    /// stationary-looking first frame), never a correctness one.
    #[gpu]
    pub prev_transform: [[f32; 4]; 4],
    #[gpu]
    pub normal_mat: [[f32; 4]; 3],
    #[gpu]
    pub bounds: [f32; 4],
    /// Resolved once at spawn time from `Renderer::mesh_slice` -- see the
    /// struct doc's "Why the draw parameters are stored" section.
    #[gpu]
    pub index_count: u32,
    #[gpu]
    pub first_index: u32,
    #[gpu]
    pub vertex_offset: i32,
    /// Resolved once at spawn time from `Renderer::material_batch_key`.
    #[gpu]
    pub material_class: u32,
    #[gpu]
    pub graph_hash_lo: u32,
    #[gpu]
    pub graph_hash_hi: u32,
    /// See `helio_pass_object_batch::GpuInstanceData::flags`.
    #[gpu]
    pub flags: u32,
    /// Base vertex in the parallel four-influence skin stream (zero for rigid meshes).
    #[gpu]
    pub skin_vertex_offset: u32,
    /// Base joint matrix in the shared palette pool (zero for rigid meshes).
    #[gpu]
    pub bone_palette_offset: u32,
    /// Number of valid joint matrices for this instance.
    #[gpu]
    pub bone_palette_count: u32,
    #[gpu]
    pub _skin_pad: u32,
}
same_layout!(StaticObjectComponent, helio_pass_gbuffer::StaticObjectRow, helio_pass_gbuffer::STATIC_OBJECTS, "static_objects"; mesh_slot, mesh_generation, material_slot, material_generation, transform, prev_transform, normal_mat, bounds, index_count, first_index, vertex_offset, material_class, graph_hash_lo, graph_hash_hi, flags, skin_vertex_offset, bone_palette_offset, bone_palette_count, _skin_pad);

// ---- conversions the rows have, mirrored on their components ----

impl From<helio_pass_deferred_light::GpuReflectionCapture> for ReflectionCaptureComponent {
    fn from(value: helio_pass_deferred_light::GpuReflectionCapture) -> Self {
        helio_pass_deferred_light::ReflectionCaptureRow::from(value).into()
    }
}

impl From<ReflectionCaptureComponent> for helio_pass_deferred_light::GpuReflectionCapture {
    fn from(value: ReflectionCaptureComponent) -> Self {
        helio_pass_deferred_light::ReflectionCaptureRow::from(value).into()
    }
}

impl From<helio_pass_forward_lit::GpuLight> for LightComponent {
    fn from(value: helio_pass_forward_lit::GpuLight) -> Self {
        helio_pass_forward_lit::LightRow::from(value).into()
    }
}

impl From<LightComponent> for helio_pass_forward_lit::GpuLight {
    fn from(value: LightComponent) -> Self {
        helio_pass_forward_lit::LightRow::from(value).into()
    }
}

impl From<helio_mats::GpuMaterial> for MaterialComponent {
    fn from(value: helio_mats::GpuMaterial) -> Self {
        helio_pass_gbuffer::MaterialRow::from(value).into()
    }
}

impl From<MaterialComponent> for helio_mats::GpuMaterial {
    fn from(value: MaterialComponent) -> Self {
        helio_pass_gbuffer::MaterialRow::from(value).into()
    }
}

// ---- row builders, returning the component (see builders.py note) ----

impl StaticObjectComponent {
    pub fn with_transform(self, transform: glam::Mat4, bounds: [f32; 4]) -> Self {
        helio_pass_gbuffer::StaticObjectRow::from(self)
            .with_transform(transform, bounds)
            .into()
    }
}
