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
same_layout!(LightComponent, helio_pass_light_cull::LightRow, helio_pass_light_cull::SCENE_LIGHTS, "scene_lights"; position_range, direction_outer, color_intensity, shadow_index, light_type, inner_angle, _pad, god_rays_enabled, god_rays_density, god_rays_weight, god_rays_decay, god_rays_exposure, flare_enabled, flare_type, flare_intensity, flare_scale, flare_tint_r, flare_tint_g, flare_tint_b, ies_profile_index, light_function_index, ies_angle_scale, ies_angle_offset);

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
same_layout!(MaterialComponent, helio_mats::MaterialRow, helio_mats::MATERIALS, "materials"; base_color, emissive, roughness_metallic, tex_base_color, tex_normal, tex_roughness, tex_emissive, tex_occlusion, workflow, flags, material_class, class_params);

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

impl From<helio_pass_light_cull::GpuLight> for LightComponent {
    fn from(value: helio_pass_light_cull::GpuLight) -> Self {
        helio_pass_light_cull::LightRow::from(value).into()
    }
}

impl From<LightComponent> for helio_pass_light_cull::GpuLight {
    fn from(value: LightComponent) -> Self {
        helio_pass_light_cull::LightRow::from(value).into()
    }
}

impl From<helio_mats::GpuMaterial> for MaterialComponent {
    fn from(value: helio_mats::GpuMaterial) -> Self {
        helio_mats::MaterialRow::from(value).into()
    }
}

impl From<MaterialComponent> for helio_mats::GpuMaterial {
    fn from(value: MaterialComponent) -> Self {
        helio_mats::MaterialRow::from(value).into()
    }
}

