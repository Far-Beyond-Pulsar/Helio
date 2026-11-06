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

/// The air around a planet (or above a flat world). Coefficients are per
/// kilometre, heights and radii in kilometres. The first enabled row is
/// used.
#[derive(SceneStore, bytemuck::Pod, bytemuck::Zeroable, Clone, Copy, Debug, PartialEq)]
#[repr(C)]
#[gpu(layout = packed, buffer = "atmospheres")]
pub struct AtmosphereComponent {
    /// Planet centre in world metres (with [`placement::CENTER`]). Planet
    /// centres are world anchors: unlike the other positions in pass rows,
    /// which are in the frame's space, the pass rebases them by the frame's
    /// world origin.
    #[gpu]
    pub center: [f32; 3],
    /// [`placement`].
    #[gpu]
    pub placement: u32,
    #[gpu]
    pub rayleigh_scattering: [f32; 3],
    #[gpu]
    pub rayleigh_scale_height: f32,
    #[gpu]
    pub mie_scattering: [f32; 3],
    #[gpu]
    pub mie_scale_height: f32,
    #[gpu]
    pub mie_absorption: [f32; 3],
    /// Mie phase asymmetry (forward scattering, the haze around the sun).
    #[gpu]
    pub mie_g: f32,
    #[gpu]
    pub ozone_absorption: [f32; 3],
    /// Height of the ozone layer's peak.
    #[gpu]
    pub ozone_center: f32,
    /// Diffuse reflectance of the ground, for light it bounces into the air.
    #[gpu]
    pub ground_albedo: [f32; 3],
    /// Thickness of the ozone layer (a tent around its peak).
    #[gpu]
    pub ozone_width: f32,
    /// Planet radius.
    #[gpu]
    pub bottom_radius: f32,
    /// Radius of the top of the atmosphere.
    #[gpu]
    pub top_radius: f32,
    /// Angular radius of the sun's disk (radians).
    #[gpu]
    pub sun_angular_radius: f32,
    /// Non-zero to use this atmosphere.
    #[gpu]
    pub enabled: u32,
    /// Bottom of the cloud layer above the surface (km).
    #[gpu]
    pub cloud_bottom: f32,
    /// Top of the cloud layer above the surface (km).
    #[gpu]
    pub cloud_top: f32,
    /// The planet's mean share of sky the clouds cover: 0 clear, 1
    /// overcast. The weather varies it from region to region.
    #[gpu]
    pub cloud_coverage: f32,
    /// The clouds' mean form: 0 flat stratus low in the layer, 1 cumulus
    /// towering through it. The weather's regimes vary it.
    #[gpu]
    pub cloud_type: f32,
    /// Extinction inside a cloud at full density (per km).
    #[gpu]
    pub cloud_extinction: f32,
    /// Size of the finest weather: the groups cumulus gather in (km).
    #[gpu]
    pub cloud_weather_scale: f32,
    /// Size of the clouds' billows (km).
    #[gpu]
    pub cloud_shape_scale: f32,
    /// Size of the wisps that erode their edges (km).
    #[gpu]
    pub cloud_detail_scale: f32,
    /// Wind carrying the clouds (world m/s).
    #[gpu]
    pub cloud_wind: [f32; 3],
    /// Non-zero to draw the clouds.
    #[gpu]
    pub clouds_enabled: u32,
    /// Share of the light a cloud scatters rather than absorbs, per channel.
    #[gpu]
    pub cloud_albedo: [f32; 3],
    /// How far the detail erodes the clouds' edges (0..1).
    #[gpu]
    pub cloud_erosion: f32,
    /// The planet's axis (world direction) the circulation's latitudes
    /// are measured from.
    #[gpu]
    pub cloud_pole: [f32; 3],
    /// How strongly the circulation bands the cover by latitude (0..1).
    #[gpu]
    pub cloud_circulation: f32,
    /// Circulation cells from the equator to each pole (Earth's three:
    /// Hadley, Ferrel, polar).
    #[gpu]
    pub cloud_circulation_cells: f32,
    /// Size of the weather systems: cyclones, fronts and highs (km).
    #[gpu]
    pub cloud_synoptic_scale: f32,
    /// How deep and frequent the cyclones are (0..1).
    #[gpu]
    pub cloud_storminess: f32,
    /// Size of the mesoscale cells and clusters (km).
    #[gpu]
    pub cloud_mesoscale_scale: f32,
    /// How much high cirrus the weather brings (0..1).
    #[gpu]
    pub cloud_cirrus: f32,
    /// Height of the cirrus sheet above the surface (km).
    #[gpu]
    pub cloud_cirrus_height: f32,
    /// How fast the weather evolves: 0 still, 1 real time, more a time
    /// lapse.
    #[gpu]
    pub cloud_evolution: f32,
    /// Which weather: each seed lays out other systems.
    #[gpu]
    pub cloud_seed: u32,
    /// How the clouds are drawn: 0 smooth volumes, otherwise voxels of this
    /// side (km), each cell filled or empty ([`CloudRepresentation`]).
    #[gpu]
    pub cloud_voxel_size: f32,
    /// The density (0..1) a voxel cell is filled from.
    #[gpu]
    pub cloud_voxel_fill: f32,
    /// Unused: the row stays a whole number of vec4s.
    #[gpu]
    pub cloud_reserved: [f32; 2],
}
same_layout!(AtmosphereComponent, helio_pass_sky::AtmosphereRow, helio_pass_sky::ATMOSPHERES, "atmospheres"; center, placement, rayleigh_scattering, rayleigh_scale_height, mie_scattering, mie_scale_height, mie_absorption, mie_g, ozone_absorption, ozone_center, ground_albedo, ozone_width, bottom_radius, top_radius, sun_angular_radius, enabled, cloud_bottom, cloud_top, cloud_coverage, cloud_type, cloud_extinction, cloud_weather_scale, cloud_shape_scale, cloud_detail_scale, cloud_wind, clouds_enabled, cloud_albedo, cloud_erosion, cloud_pole, cloud_circulation, cloud_circulation_cells, cloud_synoptic_scale, cloud_storminess, cloud_mesoscale_scale, cloud_cirrus, cloud_cirrus_height, cloud_evolution, cloud_seed, cloud_voxel_size, cloud_voxel_fill, cloud_reserved);

impl Default for AtmosphereComponent {
    fn default() -> Self {
        helio_pass_sky::AtmosphereRow::default().into()
    }
}

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

impl AtmosphereComponent {
    pub fn around_planet(self, center_m: [f32; 3], radius_m: f64) -> Self {
        helio_pass_sky::AtmosphereRow::from(self)
            .around_planet(center_m, radius_m)
            .into()
    }

    pub fn with_clouds(self, clouds: helio_pass_sky::CloudLayer) -> Self {
        helio_pass_sky::AtmosphereRow::from(self)
            .with_clouds(clouds)
            .into()
    }
}
