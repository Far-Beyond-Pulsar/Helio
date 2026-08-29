struct Camera {
    view: mat4x4<f32>, proj: mat4x4<f32>, view_proj: mat4x4<f32>,
    view_proj_inv: mat4x4<f32>, position_near: vec4<f32>, forward_far: vec4<f32>,
    jitter_frame: vec4<f32>, prev_view_proj: mat4x4<f32>,
};
struct TemporalParams {
    blend: f32, history_valid: f32, cloud_base: f32, cloud_top: f32,
};
@group(0) @binding(0) var current_layer: texture_2d<f32>;
@group(0) @binding(1) var layer_sampler: sampler;
@group(0) @binding(2) var history_layer: texture_2d<f32>;
@group(0) @binding(3) var history_sampler: sampler;
@group(0) @binding(4) var<uniform> params: TemporalParams;
@group(0) @binding(5) var<storage, read> cameras: array<Camera>;
@group(0) @binding(6) var output_layer: texture_storage_2d<rgba16float, write>;
@group(0) @binding(7) var current_data: texture_2d<f32>;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let size = textureDimensions(output_layer);
    if (gid.x >= size.x || gid.y >= size.y) { return; }
    let uv = (vec2<f32>(gid.xy) + vec2<f32>(0.5)) / vec2<f32>(size);
    let current = textureSampleLevel(current_layer, layer_sampler, uv, 0.0);
    let camera = cameras[0];
    let clip = vec4<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, 1.0, 1.0);
    let far_world = camera.view_proj_inv * clip;
    let ro = camera.position_near.xyz;
    let rd = normalize(far_world.xyz / far_world.w - ro);
    let depth = textureSampleLevel(current_data, layer_sampler, uv, 0.0);
    let layer_y = (params.cloud_base + params.cloud_top) * 0.5;
    var t = depth.r * 6000.0;
    if (depth.g < 0.01 || t <= 0.0) { t = (layer_y - ro.y) / rd.y; }
    if (abs(rd.y) < 0.0001 || t < 0.0) { t = 600.0; }
    let cloud_point = ro + rd * min(t, 6000.0);
    let previous_clip = camera.prev_view_proj * vec4<f32>(cloud_point, 1.0);
    let previous_ndc = previous_clip.xy / max(previous_clip.w, 0.0001);
    let previous_uv = vec2<f32>(previous_ndc.x * 0.5 + 0.5, 0.5 - previous_ndc.y * 0.5);
    let valid_uv = all(previous_uv >= vec2<f32>(0.0)) && all(previous_uv <= vec2<f32>(1.0));
    let history = textureSampleLevel(history_layer, history_sampler, previous_uv, 0.0);
    let motion = length(previous_uv - uv);
    let w = clamp(params.blend * (1.0 - smoothstep(0.015, 0.12, motion)), 0.0, 0.92)
        * step(0.5, params.history_valid) * f32(valid_uv);
    let clamped_history = clamp(history, current - vec4<f32>(0.08, 0.08, 0.08, 0.12), current + vec4<f32>(0.08, 0.08, 0.08, 0.12));
    textureStore(output_layer, vec2<i32>(gid.xy), mix(current, clamped_history, w));
}
