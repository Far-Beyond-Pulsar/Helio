//! Row components: one SceneDB component per Helio buffer contract, with
//! the contract row's exact fields. Generated from the phase-1 rows; the
//! `same_layout!` assertions fail the build if a row and its component
//! drift apart.

use crate::same_layout;
use pulsar_scenedb_derive::SceneStore;

/// SceneDB-owned persistent sprite row. The packed ABI intentionally matches
/// [`SpriteInstance`] and `shaders/sprite.wgsl`; cull/sort/indirect buffers
/// remain renderer-derived state.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, SceneStore, bytemuck::Pod, bytemuck::Zeroable)]
#[gpu(layout = packed, buffer = "sprite_instances")]
pub struct SpriteComponent {
    #[gpu]
    pub position_x: f32,
    #[gpu]
    pub position_y: f32,
    #[gpu]
    pub size_x: f32,
    #[gpu]
    pub size_y: f32,
    #[gpu]
    pub rotation: f32,
    #[gpu]
    pub depth: f32,
    #[gpu]
    pub _pad_uv_x: f32,
    #[gpu]
    pub _pad_uv_y: f32,
    #[gpu]
    pub uv_rect: [f32; 4],
    #[gpu]
    pub color: [f32; 4],
    #[gpu]
    pub atlas_layer: u32,
    #[gpu]
    pub _pad_tail_0: u32,
    #[gpu]
    pub _pad_tail_1: u32,
    #[gpu]
    pub _pad_tail_2: u32,
}
same_layout!(SpriteComponent, helio_pass_sprite_batch::SpriteRow, helio_pass_sprite_batch::SPRITE_INSTANCES, "sprite_instances"; position_x, position_y, size_x, size_y, rotation, depth, _pad_uv_x, _pad_uv_y, uv_rect, color, atlas_layer, _pad_tail_0, _pad_tail_1, _pad_tail_2);

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

/// The authored Corona emitter record. The pass owns the schema; SceneDB owns
/// its packed CPU column and GPU projection.
#[derive(SceneStore, bytemuck::Pod, bytemuck::Zeroable, Clone, Copy, Debug, PartialEq)]
#[repr(C)]
#[gpu(layout = packed, buffer = "corona_emitters")]
pub struct CoronaEmitterComponent {
    #[gpu]
    pub transform: [[f32; 4]; 4],
    #[gpu]
    pub emit_params: [f32; 4],
    #[gpu]
    pub size_params: [f32; 4],
    #[gpu]
    pub start_color: [f32; 4],
    #[gpu]
    pub end_color: [f32; 4],
    #[gpu]
    pub velocity: [f32; 4],
    #[gpu]
    pub velocity_variation: [f32; 4],
    #[gpu]
    pub extras: [f32; 4],
    #[gpu]
    pub texture_index: i32,
    #[gpu]
    pub particle_offset: u32,
    #[gpu]
    pub particle_count: u32,
    #[gpu]
    pub spawn_cursor: u32,
    #[gpu]
    pub _pad: [f32; 12],
}
same_layout!(CoronaEmitterComponent, helio_pass_corona::CoronaEmitterRow, helio_pass_corona::CORONA_EMITTERS, "corona_emitters"; transform, emit_params, size_params, start_color, end_color, velocity, velocity_variation, extras, texture_index, particle_offset, particle_count, spawn_cursor, _pad);

#[derive(SceneStore, bytemuck::Pod, bytemuck::Zeroable, Clone, Copy, Debug, PartialEq)]
#[repr(C)]
#[gpu(layout = packed, buffer = "decals")]
pub struct DecalComponent {
    #[gpu]
    pub transform: [f32; 16],
    #[gpu]
    pub color: [f32; 4],
    #[gpu]
    pub albedo_texture_index: u32,
    #[gpu]
    pub normal_texture_index: u32,
    #[gpu]
    pub roughness_texture_index: u32,
    #[gpu]
    pub metalness_texture_index: u32,
    #[gpu]
    pub blend_mode: u32,
    #[gpu]
    pub decal_type: u32,
    #[gpu]
    pub fade_time: f32,
    #[gpu]
    pub fade_start_delay: f32,
    #[gpu]
    pub age: f32,
    #[gpu]
    pub normal_adapt: u32,
    #[gpu]
    pub _pad0: f32,
    #[gpu]
    pub _pad1: f32,
}
same_layout!(DecalComponent, helio_pass_decal::DecalRow, helio_pass_decal::DECALS, "decals"; transform, color, albedo_texture_index, normal_texture_index, roughness_texture_index, metalness_texture_index, blend_mode, decal_type, fade_time, fade_start_delay, age, normal_adapt, _pad0, _pad1);

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

#[derive(SceneStore, bytemuck::Pod, bytemuck::Zeroable, Clone, Copy, Debug, PartialEq)]
#[repr(C)]
#[gpu(layout = packed, buffer = "foliage_types")]
pub struct FoliageTypeComponent {
    #[gpu]
    pub density: f32,
    #[gpu]
    pub height_range: [f32; 2],
    #[gpu]
    pub width_range: [f32; 2],
    #[gpu]
    pub slope_range: [f32; 2],
    #[gpu]
    pub altitude_range: [f32; 2],
    #[gpu]
    pub lod_distances: [f32; 4],
    #[gpu]
    pub wind_response: [f32; 3],
    #[gpu]
    pub interaction_stiffness: f32,
    #[gpu]
    pub material_id: u32,
    #[gpu]
    pub density_layer: u32,
    #[gpu]
    pub kind_and_flags: u32,
    #[gpu]
    pub mesh_or_impostor_id: u32,
    #[gpu]
    pub base_color: u32,
    #[gpu]
    pub roughness_metallic: u32,
    #[gpu]
    pub ground_mask: u32,
    #[gpu]
    pub blades_per_tuft: u32,
    #[gpu]
    pub lean_range: [f32; 2],
    #[gpu]
    pub curve: f32,
    #[gpu]
    pub fall_chance: f32,
    #[gpu]
    pub tip_color: u32,
    #[gpu]
    pub patch_scale: f32,
    #[gpu]
    pub patch_strength: f32,
}
same_layout!(FoliageTypeComponent, helio_pass_foliage_place::FoliageTypeRow, helio_pass_foliage_place::FOLIAGE_TYPES, "foliage_types"; density, height_range, width_range, slope_range, altitude_range, lod_distances, wind_response, interaction_stiffness, material_id, density_layer, kind_and_flags, mesh_or_impostor_id, base_color, roughness_metallic, ground_mask, blades_per_tuft, lean_range, curve, fall_chance, tip_color, patch_scale, patch_strength);

#[derive(SceneStore, bytemuck::Pod, bytemuck::Zeroable, Clone, Copy, Debug, PartialEq)]
#[repr(C)]
#[gpu(layout = packed, buffer = "foliage_layers")]
pub struct FoliageLayerComponent {
    #[gpu]
    pub bounds_min: [f32; 4],
    #[gpu]
    pub bounds_max: [f32; 4],
}
same_layout!(FoliageLayerComponent, helio_pass_foliage_place::FoliageLayerRow, helio_pass_foliage_place::FOLIAGE_LAYERS, "foliage_layers"; bounds_min, bounds_max);

/// Something that pushes foliage aside: a character's feet, a vehicle, a thrown rock.
/// Blades within `radius` of it lie over away from it, and stay pressed down along its
/// wake (where it was over the last `wake` seconds, from its velocity) and rise again
/// behind it. World positions in f64 halves (`position_hi + position_lo`), so it
/// holds at planet scale: build rows with [`FoliageInteractorComponent::new`].
#[derive(SceneStore, bytemuck::Pod, bytemuck::Zeroable, Clone, Copy, Debug, PartialEq)]
#[repr(C)]
#[gpu(layout = packed, buffer = "foliage_interactors")]
pub struct FoliageInteractorComponent {
    /// xyz the world position's f32 part, w the radius (metres; 0: inactive).
    #[gpu]
    pub position_hi_radius: [f32; 4],
    /// xyz the rest of the position (world − hi), w how hard it pushes (1: flat at its
    /// centre for a type of stiffness 1).
    #[gpu]
    pub position_lo_strength: [f32; 4],
    /// xyz its velocity (m/s), w its wake (seconds of track that stay pressed).
    #[gpu]
    pub velocity_wake: [f32; 4],
}
same_layout!(FoliageInteractorComponent, helio_pass_foliage_place::FoliageInteractorRow, helio_pass_foliage_place::FOLIAGE_INTERACTORS, "foliage_interactors"; position_hi_radius, position_lo_strength, velocity_wake);

#[derive(SceneStore, bytemuck::Pod, bytemuck::Zeroable, Clone, Copy, Debug, PartialEq)]
#[repr(C)]
#[gpu(layout = packed, buffer = "foliage_wind")]
pub struct FoliageWindComponent {
    #[gpu]
    pub direction_speed: [f32; 4],
    #[gpu]
    pub gust: [f32; 4],
    #[gpu]
    pub time_prev_time: [f32; 2],
    #[gpu]
    pub _pad: [f32; 2],
}
same_layout!(FoliageWindComponent, helio_pass_foliage_place::FoliageWindRow, helio_pass_foliage_place::FOLIAGE_WIND, "foliage_wind"; direction_speed, gust, time_prev_time, _pad);

/// A placed light, authored as a SceneDB component. Field-for-field
/// identical to `GpuLight` (enforced by the size/align test
/// below) so conversion is a zero-cost `bytemuck::cast`, matching
/// `CoronaEmitterComponent`'s relationship to `GpuCoronaEmitter`.
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

/// A camera's lens. One row per view; see the module doc.
#[derive(SceneStore, bytemuck::Pod, bytemuck::Zeroable, Clone, Copy, Debug, PartialEq)]
#[repr(C)]
#[gpu(layout = packed, buffer = "camera_lenses")]
pub struct LensComponent {
    #[gpu]
    pub view_id: u32,
    #[gpu]
    pub enabled: u32,
    /// [`FOCUS_MANUAL`] or [`FOCUS_AUTO_POINT`].
    #[gpu]
    pub focus_mode: u32,
    /// 0 economical, 1 high (more gather taps), 2 ultra. Also the number of
    /// wavelength samples of lateral chromatic aberration: 3, 7, 16.
    #[gpu]
    pub quality: u32,
    #[gpu]
    pub focal_length_mm: f32,
    #[gpu]
    pub f_number: f32,
    /// Transmission stop; the light lost against the f-number is applied by
    /// a [`element_kind::T_STOP_LOSS`] element.
    #[gpu]
    pub t_stop: f32,
    #[gpu]
    pub sensor_width_mm: f32,
    /// Zero derives it from the width at 3:2.
    #[gpu]
    pub sensor_height_mm: f32,
    #[gpu]
    pub focus_distance_m: f32,
    /// Autofocus point in uv (y down).
    #[gpu]
    pub af_u: f32,
    #[gpu]
    pub af_v: f32,
    /// Autofocus ease rate in 1/s (<= 0: instant).
    #[gpu]
    pub af_speed: f32,
    /// Largest circle of confusion diameter in pixels (<= 0: 32).
    #[gpu]
    pub max_coc_px: f32,
    /// First row of this lens's elements in the `lens_elements` buffer.
    #[gpu]
    pub element_first: u32,
    #[gpu]
    pub element_count: u32,
    /// Scheimpflug: focal plane tilt in radians about an axis in the image
    /// plane at `tilt_axis` radians from +x.
    #[gpu]
    pub tilt_angle: f32,
    #[gpu]
    pub tilt_axis: f32,
    /// Petzval field curvature in dioptres at the frame corner; the focus
    /// shifts by `field_curvature * r^2`.
    #[gpu]
    pub field_curvature: f32,
    /// Iris blade count (0 = circular); shapes the flare pupil.
    #[gpu]
    pub aperture_blades: u32,
    /// Iris rotation in radians.
    #[gpu]
    pub aperture_rotation: f32,
    /// Autofocus sampling radius in uv (<= 0: 0.02).
    #[gpu]
    pub af_radius: f32,
    /// The overscan factor (>= 1; 0 reads as 1) the camera rendering this view
    /// widened its field of view by: the image the pass is given shows
    /// `overscan` times the frame's tangent extent about the frame centre, and
    /// the pass maps it back to the frame. Author it with
    /// `ray_map::applied_overscan` from the lens and elements the
    /// camera widened by; see the README.
    #[gpu]
    pub overscan: f32,
    #[gpu]
    pub _pad1: f32,
}
same_layout!(LensComponent, helio_pass_lens::LensRow, helio_pass_lens::CAMERA_LENSES, "camera_lenses"; view_id, enabled, focus_mode, quality, focal_length_mm, f_number, t_stop, sensor_width_mm, sensor_height_mm, focus_distance_m, af_u, af_v, af_speed, max_coc_px, element_first, element_count, tilt_angle, tilt_axis, field_curvature, aperture_blades, aperture_rotation, af_radius, overscan, _pad1);

/// One optical element in front of the sensor. Rows are indexed from
/// [`LensComponent::element_first`]; kind [`element_kind::NONE`] rows are
/// skipped, so a gap in the buffer is inert.
#[derive(SceneStore, bytemuck::Pod, bytemuck::Zeroable, Clone, Copy, Debug, PartialEq)]
#[repr(C)]
#[gpu(layout = packed, buffer = "lens_elements")]
pub struct LensElementComponent {
    /// [`element_kind`].
    #[gpu]
    pub kind: u32,
    /// [`region_kind`].
    #[gpu]
    pub region_kind: u32,
    /// Polygon vertex count.
    #[gpu]
    pub region_count: u32,
    #[gpu]
    pub flags: u32,
    /// Width of the region's soft edge in image-height units. Negative:
    /// derived from the aperture (see [`element_kind::DIOPTER`]).
    #[gpu]
    pub feather: f32,
    #[gpu]
    pub _pad0: f32,
    #[gpu]
    pub _pad1: f32,
    #[gpu]
    pub _pad2: f32,
    /// Kind-specific parameters.
    #[gpu]
    pub params: [f32; 16],
    /// Region-kind-specific geometry.
    #[gpu]
    pub region: [f32; 16],
}
same_layout!(LensElementComponent, helio_pass_lens::LensElementRow, helio_pass_lens::LENS_ELEMENTS, "lens_elements"; kind, region_kind, region_count, flags, feather, _pad0, _pad1, _pad2, params, region);

/// Authored portal relationship and local aperture data.
///
/// This is intentionally distinct from [`PortalViewComponent`] and
/// [`PortalChainComponent`]. Those rows are derived GPU projections consumed by
/// the current render passes; this row is the persistent SceneDB-facing source
/// of truth. A portal can point to a peer in the same sublevel, another
/// sublevel, or a nested actor context without storing a sublevel index of its
/// own.
#[derive(SceneStore, bytemuck::Pod, bytemuck::Zeroable, Clone, Copy, Debug, PartialEq)]
#[repr(C)]
#[gpu(layout = packed, buffer = "portals")]
pub struct PortalComponent {
    #[gpu]
    pub peer: u64,
    #[gpu]
    pub flags: u32,
    #[gpu]
    pub _pad: u32,
    #[gpu]
    pub transform: [[f32; 4]; 4],
    #[gpu]
    pub half_extent: [f32; 2],
}
same_layout!(PortalComponent, helio_pass_portal_cull::PortalRow, helio_pass_portal_cull::PORTALS, "portals"; peer, flags, _pad, transform, half_extent);

#[derive(SceneStore, bytemuck::Pod, bytemuck::Zeroable, Clone, Copy, Debug, PartialEq)]
#[repr(C)]
#[gpu(layout = packed, buffer = "portal_views")]
pub struct PortalViewComponent {
    #[gpu]
    pub transform: [f32; 16],
    #[gpu]
    pub inverse_transform: [f32; 16],
    #[gpu]
    pub half_extent: [f32; 2],
    #[gpu]
    pub coordinate_space: u32,
    /// Reserved compatibility word. Zero means the row is a physical portal
    /// for legacy manually-authored rows. The projection bridge sets
    /// [`Self::FLAG_MASK_HIDDEN`] on virtual nested rows so they can still be
    /// referenced by a chain without being stamped as another opening.
    #[gpu]
    pub _pad: u32,
}
same_layout!(PortalViewComponent, helio_pass_portal_cull::PortalViewRow, helio_pass_portal_cull::PORTAL_VIEWS, "portal_views"; transform, inverse_transform, half_extent, coordinate_space, _pad);

/// Derived frame counts for the portal projection tables.
///
/// The authored [`PortalComponent`] remains the source of truth. This row is
/// published by the projection bridge alongside the dense derived view and
/// chain rows so a renderer integration can distinguish active rows from a
/// growable SceneDB buffer's reserved capacity without changing the authored
/// data model.
#[derive(SceneStore, bytemuck::Pod, bytemuck::Zeroable, Clone, Copy, Debug, PartialEq)]
#[repr(C)]
#[gpu(layout = packed, buffer = "portal_projection_counts")]
pub struct PortalProjectionCountsComponent {
    #[gpu]
    pub portal_view_count: u32,
    #[gpu]
    pub portal_chain_count: u32,
    #[gpu]
    pub coordinate_space_count: u32,
    #[gpu]
    pub _pad: u32,
}
same_layout!(PortalProjectionCountsComponent, helio_pass_portal_cull::PortalProjectionCountsRow, helio_pass_portal_cull::PORTAL_PROJECTION_COUNTS, "portal_projection_counts"; portal_view_count, portal_chain_count, coordinate_space_count, _pad);

#[derive(SceneStore, bytemuck::Pod, bytemuck::Zeroable, Clone, Copy, Debug, PartialEq)]
#[repr(C)]
#[gpu(layout = packed, buffer = "post_process_volumes")]
pub struct PostProcessVolumeComponent {
    #[gpu]
    pub bounds_min: [f32; 4],
    #[gpu]
    pub bounds_max: [f32; 4],
    #[gpu]
    pub priority: f32,
    #[gpu]
    pub blend_radius: f32,
    /// Zero means "inactive" -- `postprocess.wgsl`'s `cs_volume_blend`
    /// skips a row with `blend_weight <= 0.0` before ever evaluating its
    /// (otherwise degenerate, all-zero) bounds, which is what makes an
    /// unused row safely inert. See `MAX_PP_VOLUMES`'s doc (helio-pass-
    /// postprocess) for the fixed-capacity iteration this enables.
    #[gpu]
    pub blend_weight: f32,
    #[gpu]
    pub unbound: u32,
    /// `PostProcessProperty` override bits, words 0..3 (`vec4<u32>` in WGSL).
    /// Scalars rather than `[u32; 4]`: reflection registers only specific
    /// array shapes, and this row derives reflection through `SceneStore`.
    #[gpu]
    pub override_mask_0: u32,
    #[gpu]
    pub override_mask_1: u32,
    #[gpu]
    pub override_mask_2: u32,
    #[gpu]
    pub override_mask_3: u32,
    /// `crate::GpuPostProcessUniforms`'s raw bytes, split at byte 464:
    /// reflection registers `[u32; 116]` and `[f32; 16]` but no 132-word
    /// array. `settings` is bytes 0..464 (including the first eight words of
    /// the lens block); `lens` is the remaining 64 bytes as raw bits, not
    /// numeric floats. The packed derive needs `pulsar_scenedb::Pod` per
    /// field, which `GpuPostProcessUniforms` lacks; `bytemuck::cast` between
    /// the outer structs only needs equal size -- see `layout_matches_gpu_abi`.
    #[gpu]
    pub settings: [u32; 116],
    #[gpu]
    pub lens: [f32; 16],
}
same_layout!(PostProcessVolumeComponent, helio_pass_post_volumes::PostProcessVolumeRow, helio_pass_post_volumes::POST_PROCESS_VOLUMES, "post_process_volumes"; bounds_min, bounds_max, priority, blend_radius, blend_weight, unbound, override_mask_0, override_mask_1, override_mask_2, override_mask_3, settings, lens);

/// Optional per-view baseline: the camera's own post-process settings,
/// including lens flare. The first enabled row matching the camera's view id
/// (u32 bits of `jitter_frame.w`) wins; with none, the resolver's project
/// defaults apply. PP volumes then override on top. Removing the component
/// clears its packed row. `settings`/`lens` split the uniforms exactly as
/// [`PostProcessVolumeComponent`] does.
#[derive(SceneStore, bytemuck::Pod, bytemuck::Zeroable, Clone, Copy, Debug, PartialEq)]
#[repr(C)]
#[gpu(layout = packed, buffer = "camera_postprocess")]
pub struct CameraPostProcessComponent {
    #[gpu]
    pub view_id: u32,
    #[gpu]
    pub enabled: u32,
    #[gpu]
    pub _pad0: u32,
    #[gpu]
    pub _pad1: u32,
    #[gpu]
    pub settings: [u32; 116],
    #[gpu]
    pub lens: [f32; 16],
}
same_layout!(CameraPostProcessComponent, helio_pass_post_volumes::CameraPostProcessRow, helio_pass_post_volumes::CAMERA_POSTPROCESS, "camera_postprocess"; view_id, enabled, _pad0, _pad1, settings, lens);

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

/// World-space participating medium. One world unit is one metre. Overlaps add.
/// `extinction` is sigma_a + sigma_s in m^-1, `albedo` is sigma_s / sigma_t,
/// and `emission` is scene-linear radiance emitted per metre (independent of density).
/// Mode: 0 uniform, 1 exponential height, 2 animated smoke with a height envelope.
/// Heights are along world Y, or above a planet (`planet`, `around_planet`).
/// Zeroed/deleted rows are inert. No camera-distance gates apply to these media.
#[derive(SceneStore, bytemuck::Pod, bytemuck::Zeroable, Clone, Copy, Debug, PartialEq)]
#[repr(C)]
#[gpu(layout = packed, buffer = "global_fog_media")]
pub struct GlobalFogComponent {
    #[gpu]
    pub enabled: u32,
    #[gpu]
    pub mode: u32,
    #[gpu]
    pub extinction: f32,
    #[gpu]
    pub height_falloff: f32,
    #[gpu]
    pub height: f32,
    #[gpu]
    pub anisotropy: f32,
    #[gpu]
    pub _pad: [f32; 2],
    #[gpu]
    pub albedo: [f32; 3],
    #[gpu]
    pub _pad_albedo: f32,
    #[gpu]
    pub emission: [f32; 3],
    #[gpu]
    pub _pad_emission: f32,
    /// The planet heights are measured above: its centre (world metres) and
    /// radius (m), so `height` is an altitude over its surface; radius 0:
    /// heights along world Y. Planet centres are world anchors, which the
    /// pass rebases by the frame's origin.
    #[gpu]
    pub planet: [f32; 4],
}
same_layout!(GlobalFogComponent, helio_pass_volumetric_fog::GlobalFogRow, helio_pass_volumetric_fog::GLOBAL_FOG_MEDIA, "global_fog_media"; enabled, mode, extinction, height_falloff, height, anisotropy, _pad, albedo, _pad_albedo, emission, _pad_emission, planet);

/// Axis-aligned volume, its bounds in the frame's space (relative to
/// `Renderer::world_origin`, as every position in a pass row). `medium` and `medium_planet` store raw bytes in
/// reflection-supported float arrays (not numeric float conversions), with the
/// GlobalFogComponent layout (its first 16 words, then `planet`); use `new` /
/// `set_medium` instead of packing words manually. `edge_fade` is an
/// inward fade distance in metres; zero gives a hard boundary. No PP priority.
#[derive(SceneStore, bytemuck::Pod, bytemuck::Zeroable, Clone, Copy, Debug, PartialEq)]
#[repr(C)]
#[gpu(layout = packed, buffer = "local_fog_media")]
pub struct LocalFogVolumeComponent {
    #[gpu]
    pub bounds_min: [f32; 4],
    #[gpu]
    pub bounds_max: [f32; 4],
    #[gpu]
    pub medium: [f32; 16],
    #[gpu]
    pub medium_planet: [f32; 4],
    #[gpu]
    pub edge_fade: f32,
    #[gpu]
    pub _pad: [f32; 3],
}
same_layout!(LocalFogVolumeComponent, helio_pass_volumetric_fog::LocalFogVolumeRow, helio_pass_volumetric_fog::LOCAL_FOG_MEDIA, "local_fog_media"; bounds_min, bounds_max, medium, medium_planet, edge_fade, _pad);

/// GPU-resolved per-view scalability. Match `view_id` to the u32 bits in
/// Camera.jitter_frame.w; u32::MAX is a fallback for every view. Exact matches
/// beat fallbacks, and the lowest entity row wins ties. `active=0` is a tombstone;
/// `enabled=0` on an active row explicitly disables rendering for that view.
/// Quality 0: 96x54x64; quality 1: 192x108x128 (aspect adjusted within budget).
/// Increase `history_epoch` for a camera cut or explicit history invalidation.
#[derive(SceneStore, bytemuck::Pod, bytemuck::Zeroable, Clone, Copy, Debug, PartialEq)]
#[repr(C)]
#[gpu(layout = packed, buffer = "volumetric_fog_settings")]
pub struct VolumetricFogSettingsComponent {
    #[gpu]
    pub active: u32,
    #[gpu]
    pub view_id: u32,
    #[gpu]
    pub quality: u32,
    #[gpu]
    pub enabled: u32,
    #[gpu]
    pub max_distance: f32,
    #[gpu]
    pub light_max_distance: f32,
    #[gpu]
    pub temporal_blend: f32,
    #[gpu]
    /// Relative density change at which history starts to be discarded (a
    /// medium appearing or vanishing). Lighting changes never reject: samples
    /// are jittered, so they differ by design and the blend integrates them.
    pub history_rejection: f32,
    #[gpu]
    pub light_samples: u32,
    #[gpu]
    pub history_epoch: u32,
    #[gpu]
    pub _pad: [f32; 2],
}
same_layout!(VolumetricFogSettingsComponent, helio_pass_volumetric_fog::VolumetricFogSettingsRow, helio_pass_volumetric_fog::VOLUMETRIC_FOG_SETTINGS, "volumetric_fog_settings"; active, view_id, quality, enabled, max_distance, light_max_distance, temporal_blend, history_rejection, light_samples, history_epoch, _pad);

#[derive(SceneStore, bytemuck::Pod, bytemuck::Zeroable, Clone, Copy, Debug, PartialEq)]
#[repr(C)]
#[gpu(layout = packed, buffer = "water_volumes")]
pub struct WaterVolumeComponent {
    #[gpu]
    pub bounds_min: [f32; 4],
    #[gpu]
    pub bounds_max: [f32; 4],
    #[gpu]
    pub wave_params: [f32; 4],
    #[gpu]
    pub wave_direction: [f32; 4],
    #[gpu]
    pub water_color: [f32; 4],
    #[gpu]
    pub extinction: [f32; 4],
    #[gpu]
    pub reflection_refraction: [f32; 4],
    #[gpu]
    pub caustics_params: [f32; 4],
    #[gpu]
    pub fog_params: [f32; 4],
    #[gpu]
    pub sim_params: [f32; 4],
    #[gpu]
    pub shadow_params: [f32; 4],
    #[gpu]
    pub sun_direction: [f32; 4],
    #[gpu]
    pub ssr_params: [f32; 4],
    #[gpu]
    pub sim_dynamics: [f32; 4],
    #[gpu]
    pub wind_params: [f32; 4],
    #[gpu]
    pub _pad6: [f32; 4],
}
same_layout!(WaterVolumeComponent, helio_pass_water_sim::WaterVolumeRow, helio_pass_water_sim::WATER_VOLUMES, "water_volumes"; bounds_min, bounds_max, wave_params, wave_direction, water_color, extinction, reflection_refraction, caustics_params, fog_params, sim_params, shadow_params, sun_direction, ssr_params, sim_dynamics, wind_params, _pad6);

#[derive(SceneStore, bytemuck::Pod, bytemuck::Zeroable, Clone, Copy, Debug, PartialEq)]
#[repr(C)]
#[gpu(layout = packed, buffer = "water_hitboxes")]
pub struct WaterHitboxComponent {
    #[gpu]
    pub old_min: [f32; 4],
    #[gpu]
    pub old_max: [f32; 4],
    #[gpu]
    pub new_min: [f32; 4],
    #[gpu]
    pub new_max: [f32; 4],
    #[gpu]
    pub params: [f32; 4],
}
same_layout!(WaterHitboxComponent, helio_pass_water_sim::WaterHitboxRow, helio_pass_water_sim::WATER_HITBOXES, "water_hitboxes"; old_min, old_max, new_min, new_max, params);

impl Default for AtmosphereComponent {
    fn default() -> Self {
        helio_pass_sky::AtmosphereRow::default().into()
    }
}

impl Default for GlobalFogComponent {
    fn default() -> Self {
        helio_pass_volumetric_fog::GlobalFogRow::default().into()
    }
}

impl Default for VolumetricFogSettingsComponent {
    fn default() -> Self {
        helio_pass_volumetric_fog::VolumetricFogSettingsRow::default().into()
    }
}

// ---- conversions the rows have, mirrored on their components ----

impl From<helio_pass_sprite_batch::SpriteInstance> for SpriteComponent {
    fn from(value: helio_pass_sprite_batch::SpriteInstance) -> Self {
        helio_pass_sprite_batch::SpriteRow::from(value).into()
    }
}

impl From<SpriteComponent> for helio_pass_sprite_batch::SpriteInstance {
    fn from(value: SpriteComponent) -> Self {
        helio_pass_sprite_batch::SpriteRow::from(value).into()
    }
}

impl From<helio_pass_corona::GpuCoronaEmitter> for CoronaEmitterComponent {
    fn from(value: helio_pass_corona::GpuCoronaEmitter) -> Self {
        helio_pass_corona::CoronaEmitterRow::from(value).into()
    }
}

impl From<CoronaEmitterComponent> for helio_pass_corona::GpuCoronaEmitter {
    fn from(value: CoronaEmitterComponent) -> Self {
        helio_pass_corona::CoronaEmitterRow::from(value).into()
    }
}

impl From<helio_pass_decal::GpuDecal> for DecalComponent {
    fn from(value: helio_pass_decal::GpuDecal) -> Self {
        helio_pass_decal::DecalRow::from(value).into()
    }
}

impl From<DecalComponent> for helio_pass_decal::GpuDecal {
    fn from(value: DecalComponent) -> Self {
        helio_pass_decal::DecalRow::from(value).into()
    }
}

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

impl From<helio_pass_foliage_place::GpuFoliageType> for FoliageTypeComponent {
    fn from(value: helio_pass_foliage_place::GpuFoliageType) -> Self {
        helio_pass_foliage_place::FoliageTypeRow::from(value).into()
    }
}

impl From<FoliageTypeComponent> for helio_pass_foliage_place::GpuFoliageType {
    fn from(value: FoliageTypeComponent) -> Self {
        helio_pass_foliage_place::FoliageTypeRow::from(value).into()
    }
}

impl From<helio_pass_foliage_place::GpuFoliageLayer> for FoliageLayerComponent {
    fn from(value: helio_pass_foliage_place::GpuFoliageLayer) -> Self {
        helio_pass_foliage_place::FoliageLayerRow::from(value).into()
    }
}

impl From<FoliageLayerComponent> for helio_pass_foliage_place::GpuFoliageLayer {
    fn from(value: FoliageLayerComponent) -> Self {
        helio_pass_foliage_place::FoliageLayerRow::from(value).into()
    }
}

impl From<helio_pass_foliage_place::GpuWind> for FoliageWindComponent {
    fn from(value: helio_pass_foliage_place::GpuWind) -> Self {
        helio_pass_foliage_place::FoliageWindRow::from(value).into()
    }
}

impl From<FoliageWindComponent> for helio_pass_foliage_place::GpuWind {
    fn from(value: FoliageWindComponent) -> Self {
        helio_pass_foliage_place::FoliageWindRow::from(value).into()
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

impl From<helio_pass_portal_cull::GpuPortalView> for PortalViewComponent {
    fn from(value: helio_pass_portal_cull::GpuPortalView) -> Self {
        helio_pass_portal_cull::PortalViewRow::from(value).into()
    }
}

impl From<PortalViewComponent> for helio_pass_portal_cull::GpuPortalView {
    fn from(value: PortalViewComponent) -> Self {
        helio_pass_portal_cull::PortalViewRow::from(value).into()
    }
}

impl From<helio_pass_postprocess::GpuPostProcessVolume> for PostProcessVolumeComponent {
    fn from(value: helio_pass_postprocess::GpuPostProcessVolume) -> Self {
        helio_pass_post_volumes::PostProcessVolumeRow::from(value).into()
    }
}

impl From<PostProcessVolumeComponent> for helio_pass_postprocess::GpuPostProcessVolume {
    fn from(value: PostProcessVolumeComponent) -> Self {
        helio_pass_post_volumes::PostProcessVolumeRow::from(value).into()
    }
}

impl From<helio_pass_water_sim::GpuWaterVolume> for WaterVolumeComponent {
    fn from(value: helio_pass_water_sim::GpuWaterVolume) -> Self {
        helio_pass_water_sim::WaterVolumeRow::from(value).into()
    }
}

impl From<WaterVolumeComponent> for helio_pass_water_sim::GpuWaterVolume {
    fn from(value: WaterVolumeComponent) -> Self {
        helio_pass_water_sim::WaterVolumeRow::from(value).into()
    }
}

impl From<helio_pass_water_sim::GpuWaterHitbox> for WaterHitboxComponent {
    fn from(value: helio_pass_water_sim::GpuWaterHitbox) -> Self {
        helio_pass_water_sim::WaterHitboxRow::from(value).into()
    }
}

impl From<WaterHitboxComponent> for helio_pass_water_sim::GpuWaterHitbox {
    fn from(value: WaterHitboxComponent) -> Self {
        helio_pass_water_sim::WaterHitboxRow::from(value).into()
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

impl LensElementComponent {
    pub fn with_anamorphic(self, squeeze: f32) -> Self {
        helio_pass_lens::LensElementRow::from(self)
            .with_anamorphic(squeeze)
            .into()
    }

    pub fn with_aperture_texture(self, layer: u32, amount: f32) -> Self {
        helio_pass_lens::LensElementRow::from(self)
            .with_aperture_texture(layer, amount)
            .into()
    }

    pub fn with_apodization(self, sigma: f32) -> Self {
        helio_pass_lens::LensElementRow::from(self)
            .with_apodization(sigma)
            .into()
    }

    pub fn with_cats_eye(self, strength: f32, barrel_radius: f32) -> Self {
        helio_pass_lens::LensElementRow::from(self)
            .with_cats_eye(strength, barrel_radius)
            .into()
    }

    pub fn with_central_obstruction(self, ratio: f32) -> Self {
        helio_pass_lens::LensElementRow::from(self)
            .with_central_obstruction(ratio)
            .into()
    }

    pub fn with_diffusion_tail(self, tail: f32) -> Self {
        helio_pass_lens::LensElementRow::from(self)
            .with_diffusion_tail(tail)
            .into()
    }

    pub fn with_f_number_rounding(self, straight_from_f_number: f32) -> Self {
        helio_pass_lens::LensElementRow::from(self)
            .with_f_number_rounding(straight_from_f_number)
            .into()
    }

    pub fn with_feather(self, feather: f32) -> Self {
        helio_pass_lens::LensElementRow::from(self)
            .with_feather(feather)
            .into()
    }

    pub fn with_glint(self, contrast: f32) -> Self {
        helio_pass_lens::LensElementRow::from(self)
            .with_glint(contrast)
            .into()
    }

    pub fn with_halation_tint(self, tint: [f32; 3]) -> Self {
        helio_pass_lens::LensElementRow::from(self)
            .with_halation_tint(tint)
            .into()
    }

    pub fn with_iris(self, blades: u32, curvature: f32, rotation: f32) -> Self {
        helio_pass_lens::LensElementRow::from(self)
            .with_iris(blades, curvature, rotation)
            .into()
    }

    pub fn with_kannala_brandt(self, k: [f32; 4]) -> Self {
        helio_pass_lens::LensElementRow::from(self)
            .with_kannala_brandt(k)
            .into()
    }

    pub fn with_mask_transform(self, scale: f32, offset: [f32; 2], tile: bool) -> Self {
        helio_pass_lens::LensElementRow::from(self)
            .with_mask_transform(scale, offset, tile)
            .into()
    }

    pub fn with_mount_distance(self, metres: f32) -> Self {
        helio_pass_lens::LensElementRow::from(self)
            .with_mount_distance(metres)
            .into()
    }

    pub fn with_onion_rings(self, amplitude: f32, rings: f32) -> Self {
        helio_pass_lens::LensElementRow::from(self)
            .with_onion_rings(amplitude, rings)
            .into()
    }

    pub fn with_panini(self, d: f32, s: f32) -> Self {
        helio_pass_lens::LensElementRow::from(self)
            .with_panini(d, s)
            .into()
    }

    pub fn with_prism_falloff(self, ratio: f32) -> Self {
        helio_pass_lens::LensElementRow::from(self)
            .with_prism_falloff(ratio)
            .into()
    }

    pub fn with_prism_ring(self) -> Self {
        helio_pass_lens::LensElementRow::from(self)
            .with_prism_ring()
            .into()
    }

    pub fn with_prism_spin(self, radians: f32) -> Self {
        helio_pass_lens::LensElementRow::from(self)
            .with_prism_spin(radians)
            .into()
    }

    pub fn with_psf_amount(self, amount: f32) -> Self {
        helio_pass_lens::LensElementRow::from(self)
            .with_psf_amount(amount)
            .into()
    }

    pub fn with_seam_darkening(self, amount: f32) -> Self {
        helio_pass_lens::LensElementRow::from(self)
            .with_seam_darkening(amount)
            .into()
    }

    pub fn with_spherical_aberration(self, amount: f32, asymmetry: f32) -> Self {
        helio_pass_lens::LensElementRow::from(self)
            .with_spherical_aberration(amount, asymmetry)
            .into()
    }

    pub fn with_swirl(self, swirl: f32) -> Self {
        helio_pass_lens::LensElementRow::from(self)
            .with_swirl(swirl)
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

impl GlobalFogComponent {
    pub fn around_planet(self, center_m: [f32; 3], radius_m: f64) -> Self {
        helio_pass_volumetric_fog::GlobalFogRow::from(self)
            .around_planet(center_m, radius_m)
            .into()
    }
}
