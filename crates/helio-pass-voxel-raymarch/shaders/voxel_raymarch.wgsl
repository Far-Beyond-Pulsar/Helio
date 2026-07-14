// DDA voxel ray march shader for dynamic mode.
// One thread per pixel, dispatches over the full render target.
// Reads voxel volumes from scene storage, DDA marches through the brick grid.

struct Camera {
    view:           mat4x4<f32>,
    proj:           mat4x4<f32>,
    view_proj:      mat4x4<f32>,
    inv_view_proj:  mat4x4<f32>,
    position_near:  vec4<f32>,
    forward_far:    vec4<f32>,
    jitter_frame:   vec4<f32>,
    prev_view_proj: mat4x4<f32>,
}

struct GpuVoxelVolume {
    local_to_world:  mat4x4<f32>,
    world_to_local:  mat4x4<f32>,
    dimensions:      vec3<u32>,
    brick_grid_dim:  u32,
    voxel_size:      f32,
    palette_offset:  u32,
    volume_id:       u32,
    _pad:            vec2<u32>,
}

struct RayMarchParams {
    width:          f32,
    height:         f32,
    time:           f32,
    volume_count:   u32,
    light_count:    u32,
    _pad0:          u32,
    _pad1:          u32,
    _pad2:          u32,
}

// GpuLight (64 bytes, matches libhelio::GpuLight — see deferred_lighting.wgsl).
struct GpuLight {
    position_range:  vec4<f32>,
    direction_outer: vec4<f32>,
    color_intensity: vec4<f32>,
    shadow_index:    u32,
    light_type:      u32,
    inner_angle:     f32,
    _pad:            u32,
}

struct HitResult {
    hit:      u32,
    material: u32,
    position: vec3<f32>,
    normal:   vec3<f32>,
}

@group(0) @binding(0) var<uniform> camera: Camera;
@group(0) @binding(1) var<uniform> params: RayMarchParams;
@group(0) @binding(2) var<storage, read> volumes: array<GpuVoxelVolume>;
@group(0) @binding(3) var<storage, read> brick_pool: array<u32>;
@group(0) @binding(4) var<storage, read> voxel_data: array<u32>;
@group(0) @binding(5) var out_color: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(6) var out_normal: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(7) var<storage, read> lights: array<GpuLight>;

// Simple Lambertian contribution from a scene light (no PBR/specular/shadows —
// this pass is a lightweight forward shader, not the deferred PBR pipeline).
fn light_contribution(light: GpuLight, world_pos: vec3<f32>, normal: vec3<f32>) -> vec3<f32> {
    var l: vec3<f32>;
    var radiance: vec3<f32>;

    if light.light_type == 0u {
        // Directional
        l = normalize(-light.direction_outer.xyz);
        radiance = light.color_intensity.xyz * light.color_intensity.w;
    } else {
        // Point / spot
        let to_light = light.position_range.xyz - world_pos;
        let dist = length(to_light);
        if dist > light.position_range.w {
            return vec3<f32>(0.0);
        }
        l = to_light / max(dist, 0.0001);
        var atten = 1.0 / (dist * dist + 0.0001);
        let normalized_dist = dist / light.position_range.w;
        atten *= max(0.0, 1.0 - normalized_dist * normalized_dist * normalized_dist * normalized_dist);
        if light.light_type == 2u {
            let cos_a = dot(-l, light.direction_outer.xyz);
            atten *= smoothstep(light.direction_outer.w, light.inner_angle, cos_a);
        }
        radiance = light.color_intensity.xyz * light.color_intensity.w * atten;
    }

    let n_dot_l = max(dot(normal, l), 0.0);
    return radiance * n_dot_l;
}

fn read_voxel(data_offset: u32, x: u32, y: u32, z: u32) -> u32 {
    let linear = z * 64u + y * 8u + x;
    let word_idx = (data_offset + linear / 4u);
    let byte_in_word = linear % 4u;
    return (voxel_data[word_idx] >> (byte_in_word * 8u)) & 0xFFu;
}

fn ray_aabb(ro: vec3<f32>, inv_rd: vec3<f32>, bmin: vec3<f32>, bmax: vec3<f32>) -> vec2<f32> {
    let t1 = (bmin - ro) * inv_rd;
    let t2 = (bmax - ro) * inv_rd;
    let tmin = max(min(t1.x, t2.x), max(min(t1.y, t2.y), min(t1.z, t2.z)));
    let tmax = min(max(t1.x, t2.x), min(max(t1.y, t2.y), max(t1.z, t2.z)));
    return vec2<f32>(tmin, tmax);
}

fn trace_volume(ro: vec3<f32>, rd: vec3<f32>, vol: GpuVoxelVolume) -> HitResult {
    let vs = vol.voxel_size;
    let dims = vec3<f32>(vol.dimensions);
    let local_size = dims * vs;
    let half = local_size * 0.5;

    // Volume bounds are [-half, half] in local space
    let bmin = -half;
    let bmax = half;
    let inv_rd = 1.0 / rd;

    let t = ray_aabb(ro, inv_rd, bmin, bmax);
    var t_entry = t.x;
    let t_exit = t.y;

    if t_entry > t_exit || t_exit < 0.0 {
        return HitResult(0u, 0u, vec3<f32>(0.0), vec3<f32>(0.0));
    }

    t_entry = max(t_entry, 0.0);

    // Starting position in local space. Grid index 0 corresponds to local
    // position `bmin` (= -half), so the `half` offset must be added back
    // before dividing by voxel size — otherwise grid coords come out centered
    // on the origin instead of the volume's corner.
    var pos = ro + rd * t_entry;
    var grid = vec3<i32>(floor((pos + half) / vs));

    // Clamp to volume bounds
    let max_grid = vec3<i32>(vol.dimensions) - 1;
    grid = clamp(grid, vec3<i32>(0), max_grid);

    // DDA setup
    let step = vec3<i32>(sign(rd));
    let t_delta = vec3<f32>(vs / abs(rd));
    let next_border = (vec3<f32>(grid) + vec3<f32>(select(vec3<i32>(1), vec3<i32>(0), step < vec3(0)))) * vs - half;
    var t_max = (next_border - ro) / rd;

    // Axis crossed to reach the current cell (0=x, 1=y, 2=z) — needed for a
    // correct single-axis face normal instead of the constant per-ray
    // -sign(rd) vector, which only depends on the ray's octant and produces
    // screen-space quadrant banding instead of real per-voxel-face shading.
    // Initialized to the AABB entry face.
    var last_axis = 0u;
    var entry_axis_max = min((bmin.x - ro.x) * inv_rd.x, (bmax.x - ro.x) * inv_rd.x);
    let entry_y = min((bmin.y - ro.y) * inv_rd.y, (bmax.y - ro.y) * inv_rd.y);
    if entry_y > entry_axis_max { entry_axis_max = entry_y; last_axis = 1u; }
    let entry_z = min((bmin.z - ro.z) * inv_rd.z, (bmax.z - ro.z) * inv_rd.z);
    if entry_z > entry_axis_max { entry_axis_max = entry_z; last_axis = 2u; }

    var t_hit = t_entry;
    let max_steps = 256u;

    for (var i = 0u; i < max_steps; i++) {
        if t_hit > t_exit { break; }

        // Read voxel material at current grid position
        let brick_x = grid.x / 8;
        let brick_y = grid.y / 8;
        let brick_z = grid.z / 8;

        // Indirection: for now, assume bricks are in a flat list based on grid pos
        // In a real implementation, this would read from an indirection grid
        let brick_idx = brick_z * i32(vol.brick_grid_dim) * i32(vol.brick_grid_dim)
                      + brick_y * i32(vol.brick_grid_dim)
                      + brick_x;

        if brick_idx >= 0 {
            let meta_word = brick_pool[u32(brick_idx) * 2u];
            let data_offset = meta_word & 0xFFFFFFu;
            let occupancy = (meta_word >> 24u) & 0xFFu;

            if occupancy > 0u && data_offset < 0xFFFFFFFEu {
                let lx = u32(grid.x % 8);
                let ly = u32(grid.y % 8);
                let lz = u32(grid.z % 8);
                let mat = read_voxel(data_offset, lx, ly, lz);

                if mat > 0u {
                    let hit_pos = ro + rd * t_hit;
                    var normal = vec3<f32>(0.0);
                    if last_axis == 0u { normal.x = -f32(step.x); }
                    else if last_axis == 1u { normal.y = -f32(step.y); }
                    else { normal.z = -f32(step.z); }
                    return HitResult(1u, mat, hit_pos, normal);
                }
            }
        }

        // DDA step to next voxel
        if t_max.x < t_max.y {
            if t_max.x < t_max.z {
                t_hit = t_max.x;
                grid.x += step.x;
                t_max.x += t_delta.x;
                last_axis = 0u;
            } else {
                t_hit = t_max.z;
                grid.z += step.z;
                t_max.z += t_delta.z;
                last_axis = 2u;
            }
        } else {
            if t_max.y < t_max.z {
                t_hit = t_max.y;
                grid.y += step.y;
                t_max.y += t_delta.y;
                last_axis = 1u;
            } else {
                t_hit = t_max.z;
                grid.z += step.z;
                t_max.z += t_delta.z;
                last_axis = 2u;
            }
        }

        // Stop if we leave volume bounds
        if any(grid < vec3<i32>(0)) || any(grid > max_grid) {
            break;
        }
    }

    return HitResult(0u, 0u, vec3<f32>(0.0), vec3<f32>(0.0));
}

fn golden_ratio_hue(index: u32) -> f32 {
    return f32(index) * 0.6180339887;
}

fn material_color(index: u32) -> vec3<f32> {
    let h = golden_ratio_hue(index);
    let r = cos(h * 6.28318 + 0.0) * 0.5 + 0.5;
    let g = cos(h * 6.28318 + 2.09439) * 0.5 + 0.5;
    let b = cos(h * 6.28318 + 4.18879) * 0.5 + 0.5;
    return vec3<f32>(r, g, b);
}

@compute @workgroup_size(8, 8, 1)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) wgs: vec3<u32>,
) {
    let px = gid.x;
    let py = gid.y;
    if px >= u32(params.width) || py >= u32(params.height) {
        return;
    }

    // Compute ray direction from pixel coordinate
    let uv = vec2<f32>(f32(px) + 0.5, f32(py) + 0.5) / vec2<f32>(params.width, params.height);
    let ndc = vec2<f32>(uv.x * 2.0 - 1.0, -(uv.y * 2.0 - 1.0));

    let ro = camera.position_near.xyz;

    // Reconstruct ray direction from inverse view-projection
    let near_p = camera.inv_view_proj * vec4<f32>(ndc.x, ndc.y, 0.0, 1.0);
    let far_p = camera.inv_view_proj * vec4<f32>(ndc.x, ndc.y, 1.0, 1.0);
    let near_ws = near_p.xyz / near_p.w;
    let far_ws = far_p.xyz / far_p.w;
    let rd = normalize(far_ws - near_ws);

    // Trace all volumes, find nearest hit
    var best_hit = HitResult(0u, 0u, vec3<f32>(0.0), vec3<f32>(0.0));
    var best_t = 1e10;

    for (var vi = 0u; vi < params.volume_count; vi++) {
        let vol = volumes[vi];
        let w2l = vol.world_to_local;
        let local_ro = (w2l * vec4<f32>(ro, 1.0)).xyz;
        let local_rd = (w2l * vec4<f32>(rd, 0.0)).xyz;

        let h = trace_volume(local_ro, local_rd, vol);
        if h.hit > 0u {
            let hit_t = length(h.position - ro);
            if hit_t < best_t {
                best_t = hit_t;
                best_hit = h;
            }
        }
    }

    if best_hit.hit > 0u {
        let col = material_color(best_hit.material);
        let n = best_hit.normal;

        // Sum the scene's real lights (directional/point/spot) instead of a
        // hardcoded sun, so lights added to the Scene actually affect voxels.
        let ambient = 0.2;
        var direct = vec3<f32>(0.0);
        for (var li = 0u; li < params.light_count; li++) {
            direct += light_contribution(lights[li], best_hit.position, n);
        }
        let lit = col * (ambient + direct);

        textureStore(out_color, vec2<i32>(i32(px), i32(py)), vec4<f32>(lit, 1.0));
        textureStore(out_normal, vec2<i32>(i32(px), i32(py)), vec4<f32>(n * 0.5 + 0.5, best_t));
    } else {
        textureStore(out_color, vec2<i32>(i32(px), i32(py)), vec4<f32>(0.0, 0.0, 0.0, 0.0));
        textureStore(out_normal, vec2<i32>(i32(px), i32(py)), vec4<f32>(0.0, 0.0, 0.0, 1e10));
    }
}
