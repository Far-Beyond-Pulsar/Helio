// SSAO (Screen-Space Ambient Occlusion) fragment shader
//
// Samples the G-buffer to compute ambient occlusion in screen space.

struct CameraUniform {
    view: mat4x4<f32>,
    proj: mat4x4<f32>,
    view_proj: mat4x4<f32>,
    inv_view_proj: mat4x4<f32>,
    position: vec3<f32>,
    _pad0: f32,
}

struct Globals {
    frame: u32,
    delta_time: f32,
    light_count: u32,
    ambient_intensity: f32,
    ambient_color: vec4<f32>,
    rc_world_min: vec4<f32>,
    rc_world_max: vec4<f32>,
    csm_splits: vec4<f32>,
}

@group(0) @binding(0) var<uniform> camera: CameraUniform;
@group(0) @binding(1) var<uniform> globals: Globals;

// G-buffer textures (group 1)
@group(1) @binding(0) var gbuf_albedo: texture_2d<f32>;
@group(1) @binding(1) var gbuf_normal: texture_2d<f32>;
@group(1) @binding(2) var gbuf_orm: texture_2d<f32>;
@group(1) @binding(3) var gbuf_emissive: texture_2d<f32>;
@group(1) @binding(4) var gbuf_depth: texture_depth_2d;

// SSAO data (group 2)
struct SsaoUniform {
    radius: f32,
    bias: f32,
    power: f32,
    samples: u32,
    noise_scale: vec2<f32>,
    _pad: vec2<f32>,
}

@group(2) @binding(0) var<uniform> ssao: SsaoUniform;
@group(2) @binding(1) var<storage, read> sample_kernel: array<vec4<f32>>;
@group(2) @binding(2) var noise_tex: texture_2d<f32>;
@group(2) @binding(3) var noise_sampler: sampler;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    var out: VertexOutput;
    // Fullscreen triangle
    let x = f32((vertex_index << 1u) & 2u);
    let y = f32(vertex_index & 2u);
    out.position = vec4<f32>(x * 2.0 - 1.0, 1.0 - y * 2.0, 0.0, 1.0);
    out.uv = vec2<f32>(x, y);
    return out;
}

// Reconstruct view-space position from depth
fn reconstruct_view_pos(uv: vec2<f32>, depth: f32) -> vec3<f32> {
    let ndc = vec4<f32>(uv * 2.0 - 1.0, depth, 1.0);
    let view_pos = camera.inv_view_proj * ndc;
    return view_pos.xyz / view_pos.w;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) f32 {
    let dimensions = textureDimensions(gbuf_depth);
    let texel = vec2<i32>(in.uv * vec2<f32>(dimensions));
    
    // Sample depth
    let depth = textureLoad(gbuf_depth, texel, 0);
    
    // Skip sky pixels (depth = 1.0)
    if depth >= 1.0 {
        return 1.0;
    }
    
    // Sample normal (stored in view space in G-buffer)
    let normal_sample = textureLoad(gbuf_normal, texel, 0);
    let normal = normalize(normal_sample.xyz * 2.0 - 1.0);
    
    // Reconstruct view-space position
    let frag_pos = reconstruct_view_pos(in.uv, depth);
    
    // Sample random rotation from noise texture
    let noise_uv = in.uv * ssao.noise_scale;
    let random_vec = textureSample(noise_tex, noise_sampler, noise_uv).xyz * 2.0 - 1.0;
    
    // Create TBN matrix to transform samples to view space
    let tangent = normalize(random_vec - normal * dot(random_vec, normal));
    let bitangent = cross(normal, tangent);
    let tbn = mat3x3<f32>(tangent, bitangent, normal);
    
    // Accumulate occlusion
    var occlusion = 0.0;
    
    for (var i = 0u; i < ssao.samples; i = i + 1u) {
        // Get sample position in view space
        let sample_pos = tbn * sample_kernel[i].xyz;
        let offset_pos = frag_pos + sample_pos * ssao.radius;
        
        // Project sample position to screen space
        let offset_ndc = camera.proj * vec4<f32>(offset_pos, 1.0);
        var offset_uv = offset_ndc.xy / offset_ndc.w;
        offset_uv = offset_uv * 0.5 + 0.5;
        
        // Sample depth at offset position
        let sample_depth = textureSample(gbuf_depth, noise_sampler, offset_uv);
        let sample_view_pos = reconstruct_view_pos(offset_uv, sample_depth);
        
        // Range check and accumulate
        let range_check = smoothstep(0.0, 1.0, ssao.radius / abs(frag_pos.z - sample_view_pos.z));
        
        if sample_view_pos.z >= offset_pos.z + ssao.bias {
            occlusion = occlusion + range_check;
        }
    }
    
    occlusion = 1.0 - (occlusion / f32(ssao.samples));
    return pow(occlusion, ssao.power);
}
