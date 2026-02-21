// Radiance injection compute shader.
// For every probe voxel in a cascade, accumulates direct radiance from all
// scene lights (with shadow map visibility) and stores the result.

// ---- Shared structs ----

struct GpuCascade {
    center_and_extent: vec4<f32>,
    resolution_and_type: vec4<f32>,
    texture_layer: u32,
    _pad0: u32, _pad1: u32, _pad2: u32,
}

struct RadianceCascadesUniforms {
    params: vec4<f32>, // x=cascade_count, y=gi_intensity, z=mode, w=temporal_blend
    cascades: array<GpuCascade, 4>,
}

const MAX_LIGHTS: u32 = 8u;

struct GpuLight {
    view_proj: mat4x4<f32>,
    position_and_type: vec4<f32>,   // xyz=pos, w=type (0=directional,1=point,2=spot)
    direction_and_radius: vec4<f32>, // xyz=dir, w=attenuation_radius
    color_and_intensity: vec4<f32>,  // xyz=color, w=intensity
    params: vec4<f32>,               // x=inner_angle, y=outer_angle, z=falloff, w=base_shadow_layer
}

struct LightingUniforms {
    light_count: vec4<f32>, // x=count, y=ambient, zw=unused
    lights: array<GpuLight, MAX_LIGHTS>,
}

struct CascadeSelector {
    index: u32,
    _p0: u32, _p1: u32, _p2: u32,
}

// ---- Bindings ----

@group(0) @binding(0) var<uniform> rc_uniforms: RadianceCascadesUniforms;
@group(0) @binding(1) var output_radiance: texture_storage_2d_array<rgba16float, write>;
@group(0) @binding(2) var input_history: texture_2d_array<f32>;
@group(0) @binding(3) var<uniform> lighting: LightingUniforms;
@group(0) @binding(4) var shadow_maps: texture_depth_2d_array;
@group(0) @binding(5) var shadow_sampler: sampler_comparison;
@group(0) @binding(6) var<uniform> cascade_sel: CascadeSelector;

// ---- Matrix helpers ----

fn look_at_rh(eye: vec3<f32>, center: vec3<f32>, up: vec3<f32>) -> mat4x4<f32> {
    let f = normalize(center - eye);
    let s = normalize(cross(f, up));
    let u = cross(s, f);
    return mat4x4<f32>(
        vec4<f32>(s.x, u.x, -f.x, 0.0),
        vec4<f32>(s.y, u.y, -f.y, 0.0),
        vec4<f32>(s.z, u.z, -f.z, 0.0),
        vec4<f32>(-dot(s, eye), -dot(u, eye), dot(f, eye), 1.0),
    );
}

// 90-degree perspective for cube face (aspect=1), Vulkan/wgpu NDC depth [0,1]
fn cube_perspective(near: f32, far: f32) -> mat4x4<f32> {
    let r = far / (near - far);
    return mat4x4<f32>(
        vec4<f32>(1.0, 0.0, 0.0,  0.0),
        vec4<f32>(0.0, 1.0, 0.0,  0.0),
        vec4<f32>(0.0, 0.0, r,   -1.0),
        vec4<f32>(0.0, 0.0, r * near, 0.0),
    );
}

fn cube_face_vp(light_pos: vec3<f32>, face: i32, far: f32) -> mat4x4<f32> {
    let proj = cube_perspective(0.1, far);
    var fwd: vec3<f32>;
    var up: vec3<f32>;
    if      face == 0 { fwd = vec3<f32>( 1.0,  0.0,  0.0); up = vec3<f32>(0.0, -1.0,  0.0); }
    else if face == 1 { fwd = vec3<f32>(-1.0,  0.0,  0.0); up = vec3<f32>(0.0, -1.0,  0.0); }
    else if face == 2 { fwd = vec3<f32>( 0.0,  1.0,  0.0); up = vec3<f32>(0.0,  0.0,  1.0); }
    else if face == 3 { fwd = vec3<f32>( 0.0, -1.0,  0.0); up = vec3<f32>(0.0,  0.0, -1.0); }
    else if face == 4 { fwd = vec3<f32>( 0.0,  0.0,  1.0); up = vec3<f32>(0.0, -1.0,  0.0); }
    else              { fwd = vec3<f32>( 0.0,  0.0, -1.0); up = vec3<f32>(0.0, -1.0,  0.0); }
    return proj * look_at_rh(light_pos, light_pos + fwd, up);
}

fn select_cube_face(d: vec3<f32>) -> i32 {
    let a = abs(d);
    if a.x >= a.y && a.x >= a.z {
        if d.x >= 0.0 { return 0; } else { return 1; }
    } else if a.y >= a.z {
        if d.y >= 0.0 { return 2; } else { return 3; }
    } else {
        if d.z >= 0.0 { return 4; } else { return 5; }
    }
}

// Project world_pos through view_proj → texture UV + depth in [0,1]
fn world_to_shadow_coord(world_pos: vec3<f32>, vp: mat4x4<f32>) -> vec3<f32> {
    let clip = vp * vec4<f32>(world_pos, 1.0);
    var ndc = clip.xyz / clip.w;
    ndc.x =  ndc.x * 0.5 + 0.5;
    ndc.y = -ndc.y * 0.5 + 0.5;
    return ndc;
}

fn sample_shadow(coord: vec3<f32>, layer: i32) -> f32 {
    if coord.x < 0.0 || coord.x > 1.0 ||
       coord.y < 0.0 || coord.y > 1.0 ||
       coord.z < 0.0 || coord.z > 1.0 {
        return 1.0; // outside frustum → unoccluded
    }
    return textureSampleCompareLevel(shadow_maps, shadow_sampler, coord.xy, layer, coord.z - 0.002);
}

fn point_light_visibility(world_pos: vec3<f32>, light: GpuLight) -> f32 {
    let light_pos = light.position_and_type.xyz;
    let far       = light.direction_and_radius.w;
    let base      = i32(light.params.w);
    let face      = select_cube_face(world_pos - light_pos);
    let vp        = cube_face_vp(light_pos, face, far);
    let coord     = world_to_shadow_coord(world_pos, vp);
    return sample_shadow(coord, base + face);
}

fn dir_spot_visibility(world_pos: vec3<f32>, light: GpuLight) -> f32 {
    let base  = i32(light.params.w);
    let coord = world_to_shadow_coord(world_pos, light.view_proj);
    return sample_shadow(coord, base);
}

// ---- Main ----

@compute @workgroup_size(8, 8, 4)
fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let ci       = cascade_sel.index;
    let cascade  = rc_uniforms.cascades[ci];
    let res      = u32(cascade.resolution_and_type.x);
    if gid.x >= res || gid.y >= res || gid.z >= res { return; }

    // World-space position of this probe voxel center
    let uv3      = (vec3<f32>(gid) + 0.5) / f32(res);
    let center   = cascade.center_and_extent.xyz;
    let extent   = cascade.center_and_extent.w;
    let world_pos = center + (uv3 - 0.5) * 2.0 * extent;

    let light_count = i32(lighting.light_count.x);
    let ambient     = lighting.light_count.y;

    // Accumulate radiance from all lights
    var total = vec3<f32>(ambient);

    for (var li: i32 = 0; li < light_count; li = li + 1) {
        let light      = lighting.lights[li];
        let light_type = light.position_and_type.w;
        let light_pos  = light.position_and_type.xyz;
        let lcolor     = light.color_and_intensity.xyz;
        let lintensity = light.color_and_intensity.w;
        let radius     = light.direction_and_radius.w;
        let falloff    = light.params.z;

        var attenuation = 1.0;
        var valid = true;

        // Distance attenuation: UE4 windowed inverse-square (physically correct, spatially localised)
        // attenuation = (window²) / (dist² + 1)  where window = saturate(1 - (dist/radius)^(falloff*2))
        // This falls off like 1/r² at close range but hard-clips at the radius boundary.
        if light_type != 0.0 {
            let dist = length(world_pos - light_pos);
            if dist >= radius {
                valid = false;
            } else {
                let d_over_r = dist / radius;
                let window = saturate(1.0 - pow(d_over_r, falloff * 2.0));
                attenuation = (window * window) / (dist * dist + 1.0);
            }
        }

        // Spot cone attenuation
        if valid && light_type == 2.0 {
            let to_probe  = normalize(world_pos - light_pos);
            let ldir      = light.direction_and_radius.xyz;
            let cos_angle = dot(to_probe, ldir);
            let cos_outer = cos(light.params.y);
            let cos_inner = cos(light.params.x);
            if cos_angle < cos_outer {
                valid = false;
            } else {
                attenuation *= smoothstep(cos_outer, cos_inner, cos_angle);
            }
        }

        if valid {
            // Shadow sampling temporarily disabled — always visible
            // (shadow depth maps are being written but visibility query returns wrong results)
            let visibility = 1.0;
            total += lcolor * lintensity * attenuation * visibility;
        }
    }

    // Optional temporal blend with previous frame's history
    let prev  = textureLoad(input_history, gid.xy, i32(gid.z), 0).rgb;
    let blend = rc_uniforms.params.w;
    let result = mix(total, prev, blend);

    textureStore(output_radiance, gid.xy, i32(gid.z), vec4<f32>(result, 1.0));
}

