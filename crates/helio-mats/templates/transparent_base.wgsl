//!use pbr_eval

// Transparent pass base shader — shared shader used by the transparent pass.
// Custom transparent templates override `radiant_eval_transparent`.
// Camera struct matches the gbuffer's layout so position_near is correct.

struct Camera {
    view:           mat4x4<f32>,
    proj:           mat4x4<f32>,
    view_proj:      mat4x4<f32>,
    view_proj_inv:  mat4x4<f32>,
    position_near:  vec4<f32>,
    forward_far:    vec4<f32>,
    jitter_frame:   vec4<f32>,
    prev_view_proj: mat4x4<f32>,
}

struct Globals {
    frame:             u32,
    delta_time:        f32,
    light_count:       u32,
    ambient_intensity: f32,
    ambient_color:     vec4<f32>,
    rc_world_min:      vec4<f32>,
    rc_world_max:      vec4<f32>,
    csm_splits:        vec4<f32>,
    num_tiles_x:       u32,
    num_tiles_y:       u32,
    screen_width:      f32,
    screen_height:     f32,
    // 1 when `lights[i]`/`transforms[i]` share the same raw entity index
    // (SceneDB-direct); 0 when `lights` is `ctx.scene.lights`, a freshly
    // rebuilt dense array needing `light_entity_indices[i]`. See that
    // binding's doc.
    light_mode_direct_index: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

struct GpuInstanceData {
    transform:     mat4x4<f32>,
    normal_mat_0:  vec4<f32>,
    normal_mat_1:  vec4<f32>,
    normal_mat_2:  vec4<f32>,
    bounds:        vec4<f32>,
    prev_model:    mat4x4<f32>,
    mesh_id:       u32,
    material_id:   u32,
    flags:         u32,
    _pad:          u32,
}

@group(0) @binding(0) var<storage, read> cameras: array<Camera, 2>;
@group(0) @binding(1) var<uniform>       globals:       Globals;
@group(0) @binding(2) var<storage, read> instance_data: array<GpuInstanceData>;

struct GpuLight {
    position_range:  vec4<f32>,
    direction_outer: vec4<f32>,
    color_intensity: vec4<f32>,
    shadow_index:    u32,
    light_type:      u32,
    inner_angle:     f32,
    _pad:            u32,
    god_rays_enabled:  u32,
    god_rays_density:  f32,
    god_rays_weight:   f32,
    god_rays_decay:    f32,
    god_rays_exposure: f32,
    flare_enabled:      u32,
    flare_type:         u32,
    flare_intensity:    f32,
    flare_scale:        f32,
    flare_tint_r:       f32,
    flare_tint_g:       f32,
    flare_tint_b:       f32,
    ies_profile_index:    i32,
    light_function_index: i32,
    ies_angle_scale:      f32,
    ies_angle_offset:     f32,
}

const TILE_SIZE: u32 = 16u;
const MAX_LIGHTS_PER_TILE: u32 = 64u;

@group(1) @binding(0) var<storage, read> lights:            array<GpuLight>;
@group(1) @binding(1) var<storage, read> tile_light_lists:  array<u32>;
@group(1) @binding(2) var<storage, read> tile_light_counts: array<u32>;

// Parallel to `lights` when `globals.light_mode_direct_index == 0` -- entry
// `i` is the real SceneDB entity index `lights[i]` was built from this frame
// (`ctx.scene.lights` is a freshly-rebuilt dense array every frame, not
// itself entity-indexed). Ignored (identity) when
// `light_mode_direct_index == 1`, since `lights` is then the SceneDB
// `"scene_lights"` buffer directly, already entity-indexed the same way
// `transforms` is.
@group(1) @binding(3) var<storage, read> light_entity_indices: array<u32>;
// Legacy template ABI; the default path uses the world-space position in
// LightComponent.position_range, matching the other SceneDB render passes.
struct Transform {
    position: array<f32, 3>,
    rotation: array<f32, 3>,
    scale:    array<f32, 3>,
}
@group(1) @binding(4) var<storage, read> transforms: array<Transform>;

struct GpuMaterial {
    base_color:         vec4<f32>,
    emissive:           vec4<f32>,
    roughness_metallic: vec4<f32>,
    tex_base_color:     u32,
    tex_normal:         u32,
    tex_roughness:      u32,
    tex_emissive:       u32,
    tex_occlusion:      u32,
    workflow:           u32,
    flags:              u32,
    material_class:     u32,
    class_params:       vec4<f32>,
}
@group(1) @binding(5) var<storage, read> materials: array<GpuMaterial>;

struct Vertex {
    @location(0) position:       vec3<f32>,
    @location(1) bitangent_sign: f32,
    @location(2) tex_coords:     vec2<f32>,
    @location(3) normal:         u32,
    @location(4) tangent:        u32,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_position: vec3<f32>,
    @location(1) world_normal:   vec3<f32>,
    @location(2) tex_coords:     vec2<f32>,
    @location(3) @interpolate(flat) material_id: u32,
}

fn decode_snorm8x4(packed: u32) -> vec3<f32> {
    return unpack4x8snorm(packed).xyz;
}

@vertex
fn vs_main(vertex: Vertex, @builtin(instance_index) slot: u32) -> VertexOutput {
    let inst      = instance_data[slot];
    let world_pos = inst.transform * vec4<f32>(vertex.position, 1.0);
    let normal_mat = mat3x3<f32>(
        inst.normal_mat_0.xyz,
        inst.normal_mat_1.xyz,
        inst.normal_mat_2.xyz,
    );
    var out: VertexOutput;
    out.clip_position  = cameras[0].view_proj * world_pos;
    out.world_position = world_pos.xyz;
    out.world_normal   = normalize(normal_mat * decode_snorm8x4(vertex.normal));
    out.tex_coords     = vertex.tex_coords;
    out.material_id    = inst.material_id;
    return out;
}

fn pbr_direct_light(
    light:     GpuLight,
    light_pos: vec3<f32>,
    world_pos: vec3<f32>,
    N:         vec3<f32>,
    V:         vec3<f32>,
    F0:        vec3<f32>,
    albedo:    vec3<f32>,
    roughness: f32,
    metallic:  f32,
) -> vec3<f32> {
    var L:        vec3<f32>;
    var radiance: vec3<f32>;

    if light.light_type == 0u {
        L = normalize(-light.direction_outer.xyz);
        radiance = light.color_intensity.xyz * light.color_intensity.w;
    } else {
        let to_light = light_pos - world_pos;
        let dist = length(to_light);
        if dist > light.position_range.w { return vec3<f32>(0.0); }
        L = to_light / dist;
        var atten = 1.0 / (dist * dist + 0.0001);
        let normalized_dist = dist / light.position_range.w;
        atten *= max(0.0, 1.0 - normalized_dist * normalized_dist * normalized_dist * normalized_dist);
        if light.light_type == 2u {
            let cos_a = dot(-L, light.direction_outer.xyz);
            atten *= smoothstep(light.direction_outer.w, light.inner_angle, cos_a);
        }
        radiance = light.color_intensity.xyz * light.color_intensity.w * atten;
    }

    let NdL = max(dot(N, L), 0.0);
    if NdL == 0.0 { return vec3<f32>(0.0); }
    if all(radiance < vec3<f32>(0.002)) { return vec3<f32>(0.0); }

    let H = normalize(V + L);
    let F = fresnel_schlick(max(dot(H, V), 0.0), F0);
    let kD = (1.0 - F) * (1.0 - metallic);
    let NdV = max(dot(N, V), 0.0);

    let D = distribution_ggx(N, H, roughness);
    let G = geometry_smith(N, V, L, roughness);
    let specular = D * G * F / (4.0 * NdV * NdL + 0.0001);

    let diffuse_term = kD * albedo / PI;

    return (diffuse_term + specular) * radiance * NdL;
}

fn radiant_eval_transparent(material_id: u32,
                            world_pos: vec3<f32>,
                            world_normal: vec3<f32>,
                            tex_coords: vec2<f32>) -> vec4<f32> {
    let material = materials[material_id];
    let ambient = globals.ambient_color.rgb * globals.ambient_intensity;
    let emission = material.emissive.rgb * material.emissive.a;
    return vec4<f32>(material.base_color.rgb * ambient + emission, clamp(material.base_color.a, 0.0, 1.0));
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    // RADIANT_OVERRIDE_TRANSPARENT
    // RADIANT_OVERRIDE_END

    var surface = radiant_eval_transparent(
        input.material_id,
        input.world_position,
        input.world_normal,
        input.tex_coords,
    );

    let V = normalize(cameras[0].position_near.xyz - input.world_position);
    let material = materials[input.material_id];
    let normal = normalize(input.world_normal);
    let N = select(-normal, normal, dot(normal, V) >= 0.0);
    let albedo = material.base_color.rgb;
    let roughness = clamp(material.roughness_metallic.x, 0.04, 1.0);
    let metallic = clamp(material.roughness_metallic.y, 0.0, 1.0);
    let F0 = mix(vec3<f32>(0.04), albedo, metallic);

    let tile_x = u32(input.clip_position.x) / TILE_SIZE;
    let tile_y = u32(input.clip_position.y) / TILE_SIZE;
    let tile_idx = tile_y * globals.num_tiles_x + tile_x;
    let tile_light_count = min(tile_light_counts[tile_idx], MAX_LIGHTS_PER_TILE);

    var Lo = vec3<f32>(0.0);
    for (var i = 0u; i < tile_light_count; i++) {
        let light_idx = tile_light_lists[tile_idx * MAX_LIGHTS_PER_TILE + i];
        let light = lights[light_idx];
        let light_pos = light.position_range.xyz;
        if light.light_type != 0u {
            let dist = length(light_pos - input.world_position);
            if dist > light.position_range.w { continue; }
        }
        Lo += pbr_direct_light(
            light, light_pos, input.world_position, N, V, F0, albedo,
            roughness, metallic,
        );
    }

    surface = vec4<f32>(surface.rgb + Lo, surface.a);
    return surface;
}