struct SourceLight {
    position_range: vec4<f32>, direction_outer: vec4<f32>, color_intensity: vec4<f32>,
    shadow_index: u32, light_type: u32, inner_angle: f32, _pad: u32,
    god_rays_enabled: u32, god_rays_density: f32, god_rays_weight: f32, god_rays_decay: f32,
    god_rays_exposure: f32, flare_enabled: u32, flare_type: u32, flare_intensity: f32,
    flare_scale: f32, flare_tint_r: f32, flare_tint_g: f32, flare_tint_b: f32,
    ies_profile_index: i32, light_function_index: i32, ies_angle_scale: f32, ies_angle_offset: f32,
}
struct CompactLight {
    position_range: vec4<f32>, direction_outer: vec4<f32>, color_intensity: vec4<f32>,
    shadow_index: u32, light_type: u32, inner_angle: f32, _pad: u32,
}
@group(0) @binding(0) var<storage,read> source_lights: array<SourceLight>;
@group(0) @binding(1) var<storage,read_write> compact_lights: array<CompactLight>;

@compute @workgroup_size(256)
fn compact(@builtin(global_invocation_id) gid: vec3<u32>) {
    let index=gid.x;
    if index>=arrayLength(&source_lights) || index>=arrayLength(&compact_lights) { return; }
    let source=source_lights[index];
    compact_lights[index]=CompactLight(source.position_range,source.direction_outer,
        source.color_intensity,source.shadow_index,source.light_type,source.inner_angle,source._pad);
}
