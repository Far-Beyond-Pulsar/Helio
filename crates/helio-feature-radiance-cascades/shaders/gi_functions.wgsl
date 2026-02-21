// Fragment-shader GI sampling — the sole lighting path in this pipeline.
// RadianceCascades replaces BasicLighting and ProceduralShadows entirely.
// Probe radiance (pre-computed per-voxel from real lights + shadows) is sampled
// here and multiplied by the surface albedo to produce the final lit colour.

struct GpuCascadeGI {
    center_and_extent: vec4<f32>,
    resolution_and_type: vec4<f32>,
    texture_layer: u32,
    _pad0: u32, _pad1: u32, _pad2: u32,
}

struct RCUniformsGI {
    params: vec4<f32>, // x=cascade_count, y=gi_intensity, z=mode, w=blend
    cascades: array<GpuCascadeGI, 4>,
}

@group(2) @binding(0) var radiance_probes: texture_2d_array<f32>;
@group(2) @binding(1) var gi_sampler: sampler;
@group(2) @binding(2) var<uniform> gi_uniforms: RCUniformsGI;

// ACES filmic tone mapping (Narkowicz 2015)
fn aces_tonemap(x: vec3<f32>) -> vec3<f32> {
    let a = 2.51; let b = 0.03; let c = 2.43; let d = 0.59; let e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), vec3<f32>(0.0), vec3<f32>(1.0));
}

fn linear_to_srgb(x: vec3<f32>) -> vec3<f32> {
    return pow(max(x, vec3<f32>(0.0)), vec3<f32>(1.0 / 2.2));
}

// Convert a world position to normalised [0,1]^3 UVW for a given cascade
fn gi_world_to_uvw(world_pos: vec3<f32>, cascade: GpuCascadeGI) -> vec3<f32> {
    let center = cascade.center_and_extent.xyz;
    let extent = cascade.center_and_extent.w;
    return (world_pos - center) / (2.0 * extent) + 0.5;
}

// Trilinear sample of a 2D-array texture where the Z axis is packed as layers
fn gi_sample_2d_array(uvw: vec3<f32>) -> vec3<f32> {
    let num_layers = f32(textureNumLayers(radiance_probes));
    let layer_f    = clamp(uvw.z * num_layers, 0.0, num_layers - 1.001);
    let layer0     = i32(floor(layer_f));
    let layer1     = min(layer0 + 1, i32(num_layers) - 1);
    let t          = fract(layer_f);
    let s0 = textureSampleLevel(radiance_probes, gi_sampler, uvw.xy, layer0, 0.0).rgb;
    let s1 = textureSampleLevel(radiance_probes, gi_sampler, uvw.xy, layer1, 0.0).rgb;
    return mix(s0, s1, t);
}

// Find the finest cascade that contains world_pos, sample its probes.
// Falls back to a constant ambient if no cascade covers the position.
fn sample_probe_radiance(world_pos: vec3<f32>, world_normal: vec3<f32>) -> vec3<f32> {
    let cascade_count = i32(gi_uniforms.params.x);
    if cascade_count == 0 { return vec3<f32>(0.0); }

    for (var i = 0; i < cascade_count; i++) {
        let c   = gi_uniforms.cascades[i];
        let uvw = gi_world_to_uvw(world_pos, c);
        if all(uvw >= vec3<f32>(0.0)) && all(uvw <= vec3<f32>(1.0)) {
            // Small normal offset to reduce light-bleed through thin geometry
            let offset_uvw = clamp(uvw + normalize(world_normal) * 0.008, vec3<f32>(0.001), vec3<f32>(0.999));
            return gi_sample_2d_array(offset_uvw);
        }
    }
    return vec3<f32>(0.0);
}

// Entry point called from gi_sampling.wgsl at FragmentColorCalculation.
// `base_color` is the surface albedo (already set by BasicMaterials).
// Returns the final tone-mapped, gamma-encoded pixel colour.
fn apply_radiance_cascades_gi(base_color: vec3<f32>, world_pos: vec3<f32>, world_normal: vec3<f32>) -> vec3<f32> {
    // Emissive surfaces (sky, light sources) bypass probe lighting entirely
    let mat = get_material_for_fragment(world_pos, camera.position);
    if mat.emissive_strength > 0.0 {
        return base_color;
    }

    let gi_intensity = gi_uniforms.params.y;
    let probe_radiance = sample_probe_radiance(world_pos, world_normal);
    let linear_colour  = base_color * probe_radiance * gi_intensity;

    return linear_to_srgb(aces_tonemap(linear_colour));
}


