// radiance_cascades.wgsl — GPU radiance cascade GI probe update.
// TLAS-free: approximates GI by integrating from screen-space voxelized scene data.
//
// This is a simplified implementation that:
//   1. Sample probes at a grid covering the scene bounds.
//   2. For each probe, accumulate light contributions from scene lights.
//   3. Write results to cascade0 texture.
//
// A full RC implementation would need multiple bounces and cascade merging.
// This provides a practical approximation compatible with zero TLAS overhead.

override PROBE_GRID_X: u32 = 16u;
override PROBE_GRID_Y: u32 = 8u;

struct Globals {
    frame:             u32,
    delta_time:        f32,
    light_count:       u32,
    ambient_intensity: f32,
    ambient_color:     vec3<f32>,
    csm_split_count:   u32,
    rc_world_min:      vec3<f32>,
    _pad0:             u32,
};

struct RCDynamic {
    world_min:   vec3<f32>,
    world_size:  f32,
    sky_color:   vec3<f32>,
    light_count: u32,
};

struct GpuLight {
    position:   vec3<f32>,
    kind:       u32,
    color:      vec3<f32>,
    intensity:  f32,
    direction:  vec3<f32>,
    range:      f32,
    inner_cone: f32,
    outer_cone: f32,
    shadow_idx: i32,
    _pad:       f32,
};

@group(0) @binding(0) var<uniform>  globals:         Globals;
@group(0) @binding(1) var<uniform>  rc_dynamic:      RCDynamic;
@group(0) @binding(2) var<storage, read> lights:     array<GpuLight>;
@group(0) @binding(3) var           cascade0_out:    texture_storage_2d<rgba16float, write>;

const PI: f32 = 3.14159265358979;

// Compute irradiance at `probe_pos` from all lights (no visibility — approximation).
fn compute_irradiance(probe_pos: vec3<f32>) -> vec3<f32> {
    var irr = rc_dynamic.sky_color * 0.2;  // Sky ambient base

    for (var i = 0u; i < rc_dynamic.light_count && i < globals.light_count; i++) {
        let light = lights[i];
        if light.kind == 0u {
            // Directional: uniform irradiance contribution.
            irr += light.color * light.intensity * 0.1;
        } else if light.kind == 1u {
            // Point: fall-off by distance.
            let d  = light.position - probe_pos;
            let r2 = dot(d, d);
            let r  = light.range;
            let atten = max(1.0 - r2 / (r * r), 0.0);
            irr += light.color * light.intensity * atten * (1.0 / PI);
        }
        // Spot: ignore for cascade simplicity
    }
    return irr;
}

@compute @workgroup_size(8, 8, 1)
fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(cascade0_out);
    if gid.x >= u32(dims.x) || gid.y >= u32(dims.y) { return; }

    // Map pixel coordinates to probe position.
    let u = (f32(gid.x) + 0.5) / f32(dims.x);
    let v = (f32(gid.y) + 0.5) / f32(dims.y);

    let probe_pos = rc_dynamic.world_min + vec3<f32>(u, 0.5, v) * rc_dynamic.world_size;

    let irr = compute_irradiance(probe_pos);
    textureStore(cascade0_out, vec2<i32>(i32(gid.x), i32(gid.y)), vec4<f32>(irr, 1.0));
}
