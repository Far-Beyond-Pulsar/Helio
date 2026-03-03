//! Standard vertex structures

/// Packed vertex format (matches helio-core's PackedVertex)
struct Vertex {
    @location(0) position: vec3<f32>,
    @location(1) bitangent_sign: f32,
    @location(2) tex_coords: vec2<f32>,
    @location(3) normal: u32,      // Packed as SNORM8x4
    @location(4) tangent: u32,     // Packed as SNORM8x4
}

/// Standard vertex output (to fragment shader)
struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_position: vec3<f32>,
    @location(1) world_normal: vec3<f32>,
    @location(2) world_tangent: vec3<f32>,
    @location(3) tex_coords: vec2<f32>,
    @location(4) bitangent_sign: f32,
}

/// Decode packed normal from u32
fn decode_normal(packed: u32) -> vec3<f32> {
    return unpack4x8snorm(packed).xyz;
}

/// Decode packed tangent from u32
fn decode_tangent(packed: u32) -> vec3<f32> {
    return unpack4x8snorm(packed).xyz;
}

/// Compute TBN matrix from vertex data
fn compute_tbn(normal: vec3<f32>, tangent: vec3<f32>, bitangent_sign: f32) -> mat3x3<f32> {
    let N = normalize(normal);
    let T = normalize(tangent);
    let B = cross(N, T) * bitangent_sign;
    return mat3x3<f32>(T, B, N);
}
