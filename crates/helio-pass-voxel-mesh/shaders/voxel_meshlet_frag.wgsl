// Fragment shader for voxel meshlets — pure deferred G-buffer output.
// Passes through to the deferred lighting pipeline for actual shading.
//   location 0: albedo   (Rgba8Unorm)
//   location 1: normal   (Rgba16Float)
//   location 2: orm      (Rgba8Unorm)
//   location 3: emissive (Rgba16Float)

struct VertexOutput {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) @interpolate(flat) material: u32,
    @location(1) world_pos: vec3<f32>,
    @location(2) world_normal: vec3<f32>,
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

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let col = material_color(in.material);
    return vec4(col, 1.0);
}
