// Fragment shader for voxel meshlets.
// Full deferred G-buffer output matching Helio's conventions:
//   location 0: albedo   (Rgba8Unorm)   — albedo.rgb + 1.0 alpha
//   location 1: normal   (Rgba16Float)  — world normal.xyz + 0.0
//   location 2: orm      (Rgba8Unorm)   — 1.0, roughness, metalness, 0.0
//   location 3: emissive (Rgba16Float)  — emissive.rgb, 0.0

struct VertexOutput {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) @interpolate(flat) material: u32,
    @location(1) world_pos: vec3<f32>,
}

fn material_color(index: u32) -> vec3<f32> {
    let h = f32(index) * 0.6180339887;
    let r = cos(h * 6.28318 + 0.0) * 0.5 + 0.5;
    let g = cos(h * 6.28318 + 2.09439) * 0.5 + 0.5;
    let b = cos(h * 6.28318 + 4.18879) * 0.5 + 0.5;
    return vec3<f32>(r, g, b);
}

fn material_roughness(index: u32) -> f32 {
    return 0.6 + (f32(index) * 0.03) % 0.4;
}

fn material_metalness(index: u32) -> f32 {
    return select(0.0, 0.8, index % 3u == 1u);
}

fn material_emissive(index: u32) -> vec3<f32> {
    return select(vec3<f32>(0.0), material_color(index) * 0.5, index == 0u);
}

fn compute_normal(pos: vec3<f32>) -> vec3<f32> {
    return normalize(cross(dpdx(pos), dpdy(pos)));
}

struct GBufferOutput {
    @location(0) albedo:   vec4<f32>,
    @location(1) normal:   vec4<f32>,
    @location(2) orm:      vec4<f32>,
    @location(3) emissive: vec4<f32>,
}

@fragment
fn fs_main(in: VertexOutput) -> GBufferOutput {
    let col = material_color(in.material);
    let n = compute_normal(in.world_pos);
    let roughness = material_roughness(in.material);
    let metalness = material_metalness(in.material);
    let emissive = material_emissive(in.material);

    var out: GBufferOutput;
    out.albedo = vec4(col, 1.0);
    out.normal = vec4(n, 0.0);
    out.orm = vec4(1.0, roughness, metalness, 0.0);
    out.emissive = vec4(emissive, 0.0);
    return out;
}
