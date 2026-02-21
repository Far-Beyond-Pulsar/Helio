// Radiance propagation compute shader

struct GpuCascade {
    center_and_extent: vec4<f32>,
    resolution_and_type: vec4<f32>,
    texture_layer: u32,
    _pad0: u32, _pad1: u32, _pad2: u32,
}

struct RadianceCascadesUniforms {
    params: vec4<f32>,
    cascades: array<GpuCascade, 4>,
}

@group(0) @binding(0) var<uniform> uniforms: RadianceCascadesUniforms;
@group(0) @binding(1) var output_radiance: texture_storage_2d_array<rgba16float, write>;
@group(0) @binding(2) var input_history: texture_2d_array<f32>;

@compute @workgroup_size(8, 8, 4)
fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let cascade_idx = 0u;
    let cascade = uniforms.cascades[cascade_idx];
    let resolution = u32(cascade.resolution_and_type.x);
    if gid.x >= resolution || gid.y >= resolution || gid.z >= resolution { return; }

    // Simple propagation: blend from fine to coarse cascade
    let fine = textureLoad(input_history, gid.xy, i32(gid.z), 0);
    textureStore(output_radiance, gid.xy, i32(gid.z), fine);
}
