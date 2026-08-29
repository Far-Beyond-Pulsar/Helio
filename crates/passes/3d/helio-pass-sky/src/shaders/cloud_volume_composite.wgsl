struct VertexOut { @builtin(position) position: vec4<f32>, @location(0) uv: vec2<f32> };
@group(0) @binding(0) var cloud_layer: texture_2d<f32>;
@group(0) @binding(1) var cloud_sampler: sampler;

@vertex
fn vs_main(@builtin(vertex_index) i: u32) -> VertexOut {
    let p = array<vec2<f32>, 3>(vec2<f32>(-1.0, -1.0), vec2<f32>(3.0, -1.0), vec2<f32>(-1.0, 3.0));
    let q = p[i];
    return VertexOut(vec4<f32>(q, 0.0, 1.0), vec2<f32>(q.x * 0.5 + 0.5, 0.5 - q.y * 0.5));
}

@fragment
fn fs_main(v: VertexOut) -> @location(0) vec4<f32> {
    let size = vec2<f32>(textureDimensions(cloud_layer));
    let texel = 1.0 / size;
    var color = vec4<f32>(0.0);
    var weight = 0.0;
    for (var y: i32 = -1; y <= 1; y = y + 1) {
        for (var x: i32 = -1; x <= 1; x = x + 1) {
            let d = vec2<f32>(f32(x), f32(y));
            let w = exp(-dot(d, d) * 0.5);
            color += textureSampleLevel(cloud_layer, cloud_sampler, v.uv + d * texel, 0.0) * w;
            weight += w;
        }
    }
    return color / max(weight, 0.0001);
}
