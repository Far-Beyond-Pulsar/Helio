// ── Vertex shader for voxel meshlet rendering ──────────────────────────────
// Reads per-vertex data from storage buffer (vec4: xyz=position, w=material)
// Outputs to G-buffer: albedo @ loc0, normal @ loc1, orm @ loc2, emissive @ loc3

struct Camera {
    view:           mat4x4<f32>,
    proj:           mat4x4<f32>,
    view_proj:      mat4x4<f32>,
    inv_view_proj:  mat4x4<f32>,
    position_near:  vec4<f32>,
    forward_far:    vec4<f32>,
    jitter_frame:   vec4<f32>,
    prev_view_proj: mat4x4<f32>,
}

struct VertexOutput {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) @interpolate(flat) material: u32,
    @location(1) world_pos: vec3<f32>,
    @location(2) world_normal: vec3<f32>,
    @location(3) @interpolate(flat) scene_material: u32,
}

struct VertexInput {
    @location(0) data: vec4<f32>,
    @location(1) normal: vec4<f32>,
}

// GpuLight (64 bytes, matches libhelio::GpuLight — see deferred_lighting.wgsl).
struct GpuLight {
    position_range:  vec4<f32>,
    direction_outer: vec4<f32>,
    color_intensity: vec4<f32>,
    shadow_index:    u32,
    light_type:      u32,
    inner_angle:     f32,
    _pad:            u32,
    // Light shafts — consumed by helio-pass-volumetric-fog. Padding is three
    // scalars, not vec3<u32>, to keep the struct at 96 bytes (vec3 aligns to 16).
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

struct MeshletParams {
    light_count: u32,
    _pad0:       u32,
    _pad1:       u32,
    _pad2:       u32,
}

@group(0) @binding(0) var<storage, read> cameras: array<Camera, 2>;
@group(0) @binding(1) var<storage, read> lights: array<GpuLight>;
@group(0) @binding(2) var<uniform> params: MeshletParams;
@group(0) @binding(3) var<storage, read> material_palette: array<vec4<f32>>;
struct GpuMaterial {
    base_color: vec4<f32>,
    emissive: vec4<f32>,
    roughness_metallic: vec4<f32>,
    tex_base_color: u32,
    tex_normal: u32,
    tex_roughness: u32,
    tex_emissive: u32,
    tex_occlusion: u32,
    workflow: u32,
    flags: u32,
    material_class: u32,
    class_params: vec4<f32>,
}
@group(0) @binding(4) var<storage, read> scene_materials: array<GpuMaterial>;
@group(0) @binding(5) var<storage, read> brick_material_ids: array<u32>;
@group(0) @binding(6) var<storage, read> brick_relative_origins: array<vec4<f32>>;

@vertex
fn vs_main(v: VertexInput, @builtin(instance_index) brick_slot: u32) -> VertexOutput {
    var out: VertexOutput;
    let local_slot = u32(v.data.w);
    let map_base = brick_slot * 256u;
    out.scene_material = brick_material_ids[map_base];
    out.material = select(local_slot, brick_material_ids[map_base + min(local_slot, 255u)], out.scene_material != 0u);
    if out.scene_material != 0u {
        let relative_position = brick_relative_origins[brick_slot].xyz + v.data.xyz;
        let view_relative = cameras[0].view * vec4(relative_position, 0.0);
        out.clip_pos = cameras[0].proj * vec4(view_relative.xyz, 1.0);
        out.world_pos = relative_position;
    } else {
        out.clip_pos = cameras[0].view_proj * vec4(v.data.xyz, 1.0);
        out.world_pos = v.data.xyz;
    }
    out.world_normal = normalize(v.normal.xyz);
    return out;
}

// ── Fragment shader: G-buffer output ──────────────────────────────────────

fn material_color(index: u32, is_scene: bool) -> vec3<f32> {
    if is_scene {
        if index < arrayLength(&scene_materials) {
            return scene_materials[index].base_color.rgb;
        }
        return vec3<f32>(1.0, 0.0, 1.0);
    }
    if index < arrayLength(&material_palette) {
        return material_palette[index].rgb;
    }
    return vec3<f32>(0.72, 0.72, 0.72);
}

fn material_roughness(index: u32, is_scene: bool) -> f32 {
    if is_scene {
        if index < arrayLength(&scene_materials) {
            return clamp(scene_materials[index].roughness_metallic.x, 0.02, 1.0);
        }
        return 0.8;
    }
    if index < arrayLength(&material_palette) {
        return clamp(material_palette[index].a, 0.02, 1.0);
    }
    return 0.8;
}

fn material_metalness(_index: u32) -> f32 {
    return 0.0;
}

fn material_emissive(_index: u32) -> vec3<f32> {
    return vec3<f32>(0.0);
}

// Simple Lambertian contribution from a scene light (no PBR/specular/shadows —
// this pass is a lightweight forward shader).
fn light_contribution(light: GpuLight, world_pos: vec3<f32>, normal: vec3<f32>, scene_relative: bool) -> vec3<f32> {
    var l: vec3<f32>;
    var radiance: vec3<f32>;

    if light.light_type == 0u {
        l = normalize(-light.direction_outer.xyz);
        radiance = light.color_intensity.xyz * light.color_intensity.w;
    } else {
        let light_position = select(light.position_range.xyz, light.position_range.xyz - cameras[0].position_near.xyz, scene_relative);
        let to_light = light_position - world_pos;
        let dist = length(to_light);
        if dist > light.position_range.w {
            return vec3<f32>(0.0);
        }
        l = to_light / max(dist, 0.0001);
        var atten = 1.0 / (dist * dist + 0.0001);
        let normalized_dist = dist / light.position_range.w;
        atten *= max(0.0, 1.0 - normalized_dist * normalized_dist * normalized_dist * normalized_dist);
        if light.light_type == 2u {
            let cos_a = dot(-l, light.direction_outer.xyz);
            atten *= smoothstep(light.direction_outer.w, light.inner_angle, cos_a);
        }
        radiance = light.color_intensity.xyz * light.color_intensity.w * atten;
    }

    let n_dot_l = max(dot(normal, l), 0.0);
    return radiance * n_dot_l;
}

@fragment
fn fs_main(in: VertexOutput, @builtin(front_facing) front: bool) -> @location(0) vec4<f32> {
    let n = normalize(in.world_normal) * select(-1.0, 1.0, front);
    let col = material_color(in.material, in.scene_material != 0u);
    let emissive = material_emissive(in.material);

    let ambient = 0.2;
    var direct = vec3<f32>(0.0);
    for (var i = 0u; i < params.light_count; i++) {
        direct += light_contribution(lights[i], in.world_pos, n, in.scene_material != 0u);
    }
    let lit = col * (ambient + direct) + emissive * 0.1;

    return vec4(lit, 1.0);
}
