//!use helio_prelude
//!use water_shore_surface
// Half-resolution underwater light transport. The green channel stores linear
// camera distance for a depth-guided full-resolution reconstruction.

struct WaterVolume {
    bounds_min: vec4f, bounds_max: vec4f, wave_params: vec4f,
    wave_direction: vec4f, water_color: vec4f, extinction: vec4f,
    reflection_refraction: vec4f, caustics_params: vec4f, fog_params: vec4f,
    sim_params: vec4f, shadow_params: vec4f, sun_direction: vec4f,
    ssr_params: vec4f, sim_dynamics: vec4f, wind_params: vec4f,
    world_phase: vec4f,
}
struct OceanPhase { patches: array<vec4f, 4> }
struct VertexOutput { @builtin(position) position: vec4f, @location(0) uv: vec2f }

@group(0) @binding(0) var<storage, read> cameras: array<Camera, 2>;
@group(0) @binding(1) var<storage, read> volumes: array<WaterVolume>;
@group(0) @binding(2) var depth_texture: texture_depth_2d;
@group(0) @binding(4) var caustics_tex: texture_2d<f32>;
@group(0) @binding(5) var caustics_samp: sampler;
@group(0) @binding(6) var ocean_displacement: texture_2d_array<f32>;
@group(0) @binding(7) var water_samp: sampler;
@group(0) @binding(8) var<uniform> ocean_phase: OceanPhase;
@group(0) @binding(9) var caustics_deep_tex: texture_2d<f32>;
@group(0) @binding(10) var water_interaction: texture_2d_array<f32>;

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> VertexOutput {
    let p = vec2f(f32((vi << 1u) & 2u), f32(vi & 2u));
    var out: VertexOutput;
    out.position = vec4f(p.x * 2.0 - 1.0, 1.0 - p.y * 2.0, 0.0, 1.0);
    out.uv = p;
    return out;
}

fn surface_height(xz: vec2f, vol: WaterVolume) -> f32 {
    // Shafts can query 24 half-resolution ray samples. Keep the shared shore/
    // FFT/interface formula at stable LOD 0, but skip lateral inversion here;
    // fog and surface boundary queries use the full four-step inversion.
    return water_surface_height_at(xz, vol, 0u, 0.0, true, 0u);
}

fn caustics_at(world_pos: vec3f, vol: WaterVolume) -> f32 {
    let depth = surface_height(world_pos.xz, vol) - world_pos.y;
    if depth <= 0.0 || vol.caustics_params.x <= 0.5 { return 0.0; }
    let uv = fract((world_pos.xz + ocean_phase.patches[2].xy) / 12.0);
    let footprint = max(length(dpdx(world_pos.xz)), length(dpdy(world_pos.xz)));
    let near_lod = clamp(log2(max(footprint * 512.0 / 12.0, 1.0)), 0.0, 9.0);
    let deep_lod = clamp(log2(max(footprint * 256.0 / 12.0, 1.0)), 0.0, 8.0);
    var value: f32;
    if depth <= 4.0 {
        let a = textureSampleLevel(caustics_tex, caustics_samp, uv, near_lod).r;
        let b = textureSampleLevel(caustics_deep_tex, caustics_samp, uv, deep_lod).r;
        value = mix(a, b, smoothstep(1.5, 4.0, clamp(depth, 1.5, 4.0)));
    } else {
        let b = textureSampleLevel(caustics_deep_tex, caustics_samp, uv, deep_lod).r;
        let c = textureSampleLevel(caustics_deep_tex, caustics_samp, uv, deep_lod).g;
        value = mix(b, c, smoothstep(4.0, 10.0, clamp(depth, 4.0, 10.0)));
    }
    return max(value - 1.0, 0.0) * exp(-max(depth - 1.5, 0.0) * 0.12);
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4f {
    let dims = vec2i(textureDimensions(depth_texture));
    let pixel = clamp(vec2i(in.position.xy * 2.0), vec2i(0), dims - vec2i(1));
    let depth = textureLoad(depth_texture, pixel, 0);
    let view_pos = helio_world_from_depth(cameras[0].view_proj_inv, in.uv, depth);
    let linear_depth = distance(view_pos, cameras[0].position_near.xyz);
    if arrayLength(&volumes) == 0u || helio_is_sky(depth) {
        return vec4f(0.0, linear_depth, 0.0, 1.0);
    }
    let vol = volumes[0];
    if vol.fog_params.y <= 1e-4 { return vec4f(0.0, linear_depth, 0.0, 1.0); }

    let sun_world = cameras[0].position_near.xyz + normalize(vol.sun_direction.xyz) * 10000.0;
    let sun_clip = cameras[0].view_proj * vec4f(sun_world, 1.0);
    if sun_clip.w <= 0.0 { return vec4f(0.0, linear_depth, 0.0, 1.0); }
    let sun_uv = helio_ndc_to_uv(sun_clip.xy / sun_clip.w);
    let delta = (sun_uv - in.uv) / 24.0;
    var sample_uv = in.uv;
    var shaft = 0.0;
    var decay = 1.0;
    for (var i = 0; i < 24; i++) {
        sample_uv += delta;
        if any(sample_uv < vec2f(0.0)) || any(sample_uv > vec2f(1.0)) { break; }
        let sample_pixel = clamp(vec2i(sample_uv * vec2f(dims)), vec2i(0), dims - vec2i(1));
        let sd = textureLoad(depth_texture, sample_pixel, 0);
        if !helio_is_sky(sd) {
            let p = helio_world_from_depth(cameras[0].view_proj_inv, sample_uv, sd);
            let submerged = surface_height(p.xz, vol) - p.y;
            if submerged > 0.0 {
                shaft += (0.18 + caustics_at(p, vol)) * decay;
            }
        }
        decay *= 0.94;
    }
    let angular = 1.0 - smoothstep(0.0, 1.1, length(sun_uv - in.uv));
    return vec4f(shaft / 24.0 * vol.fog_params.y * angular, linear_depth, 0.0, 1.0);
}
