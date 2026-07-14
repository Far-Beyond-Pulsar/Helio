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
}

struct MeshletParams {
    light_count: u32,
    _pad0:       u32,
    _pad1:       u32,
    _pad2:       u32,
}

@group(0) @binding(0) var<uniform> camera: Camera;
@group(0) @binding(1) var<storage, read> lights: array<GpuLight>;
@group(0) @binding(2) var<uniform> params: MeshletParams;

@vertex
fn vs_main(v: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.clip_pos = camera.view_proj * vec4(v.data.xyz, 1.0);
    out.material = u32(v.data.w);
    out.world_pos = v.data.xyz;
    out.world_normal = normalize(v.normal.xyz);
    return out;
}

// ── Fragment shader: G-buffer output ──────────────────────────────────────

fn golden_ratio_hue(index: u32) -> f32 {
    return f32(index) * 0.6180339887;
}

fn material_color(index: u32) -> vec3<f32> {
    let h = golden_ratio_hue(index);
    let r = cos(h * 6.28318 + 0.0) * 0.5 + 0.5;
    let g = cos(h * 6.28318 + 2.09439) * 0.5 + 0.5;
    let b = cos(h * 6.28318 + 4.18879) * 0.5 + 0.5;
    return vec3<f32>(r, g, b);
}

fn material_roughness(index: u32) -> f32 {
    return 0.4 + (f32(index % 16u) * 0.04);
}

fn material_metalness(index: u32) -> f32 {
    return select(0.0, 0.9, (index / 8u) % 2u == 1u);
}

fn material_emissive(index: u32) -> vec3<f32> {
    return select(vec3<f32>(0.0), material_color(index) * 2.0, index == 0u);
}

// Simple Lambertian contribution from a scene light (no PBR/specular/shadows —
// this pass is a lightweight forward shader).
fn light_contribution(light: GpuLight, world_pos: vec3<f32>, normal: vec3<f32>) -> vec3<f32> {
    var l: vec3<f32>;
    var radiance: vec3<f32>;

    if light.light_type == 0u {
        l = normalize(-light.direction_outer.xyz);
        radiance = light.color_intensity.xyz * light.color_intensity.w;
    } else {
        let to_light = light.position_range.xyz - world_pos;
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
    let col = material_color(in.material);
    let emissive = material_emissive(in.material);

    let ambient = 0.2;
    var direct = vec3<f32>(0.0);
    for (var i = 0u; i < params.light_count; i++) {
        direct += light_contribution(lights[i], in.world_pos, n);
    }
    let lit = col * (ambient + direct) + emissive * 0.1;

    return vec4(lit, 1.0);
}
