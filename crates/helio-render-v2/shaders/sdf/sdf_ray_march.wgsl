// SDF Ray March Fullscreen Shader
//
// Renders the cached SDF 3D texture via sphere tracing.
// Follows the sky.wgsl fullscreen triangle pattern.

struct Camera {
    view_proj:     mat4x4<f32>,
    position:      vec3<f32>,
    time:          f32,
    view_proj_inv: mat4x4<f32>,
}

struct Globals {
    frame:             u32,
    delta_time:        f32,
    ambient_intensity: f32,
    _padding:          f32,
}

struct SdfGridParams {
    volume_min:     vec3<f32>,
    grid_dim:       u32,
    volume_max:     vec3<f32>,
    edit_count:     u32,
    voxel_size:     f32,
    max_march_dist: f32,
    _pad0:          f32,
    _pad1:          f32,
}

// Group 0: Global (camera + globals) — shared with all passes
@group(0) @binding(0) var<uniform> camera:  Camera;
@group(0) @binding(1) var<uniform> globals: Globals;

// Group 1: SDF-specific
@group(1) @binding(0) var<uniform>   sdf_params: SdfGridParams;
@group(1) @binding(1) var            sdf_volume: texture_3d<f32>;
@group(1) @binding(2) var            sdf_sampler: sampler;

// ─── Vertex Shader (fullscreen triangle) ───────────────────────────────────

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) ndc_xy: vec2<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) vid: u32) -> VertexOutput {
    // Single oversized triangle covering the entire viewport
    let positions = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 3.0, -1.0),
        vec2<f32>(-1.0,  3.0),
    );
    let xy = positions[vid];
    var out: VertexOutput;
    out.clip_position = vec4<f32>(xy, 0.0, 1.0);
    out.ndc_xy = xy;
    return out;
}

// ─── Helper Functions ──────────────────────────────────────────────────────

fn world_to_uvw(world_pos: vec3<f32>) -> vec3<f32> {
    return (world_pos - sdf_params.volume_min) /
           (sdf_params.volume_max - sdf_params.volume_min);
}

fn sample_distance(world_pos: vec3<f32>) -> f32 {
    // DEBUG: analytic sphere at origin, radius 2 — bypasses 3D texture
    return length(world_pos) - 2.0;
    // let uvw = world_to_uvw(world_pos);
    // if any(uvw < vec3<f32>(0.0)) || any(uvw > vec3<f32>(1.0)) {
    //     return sdf_params.max_march_dist;
    // }
    // return textureSampleLevel(sdf_volume, sdf_sampler, uvw, 0.0).r;
}

fn estimate_normal(p: vec3<f32>) -> vec3<f32> {
    let e = sdf_params.voxel_size;
    let n = vec3<f32>(
        sample_distance(p + vec3<f32>(e, 0.0, 0.0)) - sample_distance(p - vec3<f32>(e, 0.0, 0.0)),
        sample_distance(p + vec3<f32>(0.0, e, 0.0)) - sample_distance(p - vec3<f32>(0.0, e, 0.0)),
        sample_distance(p + vec3<f32>(0.0, 0.0, e)) - sample_distance(p - vec3<f32>(0.0, 0.0, e)),
    );
    return normalize(n);
}

// Ray-AABB intersection (returns (tmin, tmax) or tmin > tmax if no hit)
fn intersect_aabb(ray_origin: vec3<f32>, ray_dir_inv: vec3<f32>, box_min: vec3<f32>, box_max: vec3<f32>) -> vec2<f32> {
    let t0 = (box_min - ray_origin) * ray_dir_inv;
    let t1 = (box_max - ray_origin) * ray_dir_inv;
    let tmin = min(t0, t1);
    let tmax = max(t0, t1);
    let t_enter = max(max(tmin.x, tmin.y), tmin.z);
    let t_exit  = min(min(tmax.x, tmax.y), tmax.z);
    return vec2<f32>(t_enter, t_exit);
}

// ─── Fragment Shader ───────────────────────────────────────────────────────

struct FragOutput {
    @location(0) color: vec4<f32>,
    @builtin(frag_depth) depth: f32,
}

@fragment
fn fs_main(in: VertexOutput) -> FragOutput {
    // Reconstruct world-space ray direction (same technique as sky.wgsl)
    let clip = vec4<f32>(in.ndc_xy, 1.0, 1.0);
    let world_h = camera.view_proj_inv * clip;
    let world_pt = world_h.xyz / world_h.w;
    let ray_dir = normalize(world_pt - camera.position);
    let ray_origin = camera.position;

    // Intersect ray with the SDF volume AABB to skip empty space
    let ray_dir_inv = 1.0 / ray_dir;
    let aabb_hit = intersect_aabb(ray_origin, ray_dir_inv, sdf_params.volume_min, sdf_params.volume_max);

    if aabb_hit.x > aabb_hit.y || aabb_hit.y < 0.0 {
        // Ray misses the volume entirely
        discard;
    }

    // Start marching from the AABB entry point (or camera if inside)
    var t = max(aabb_hit.x, 0.0);
    let max_t = aabb_hit.y;
    let min_dist = sdf_params.voxel_size * 0.5;

    var hit = false;
    for (var i = 0u; i < 192u; i++) {
        let p = ray_origin + ray_dir * t;
        let d = sample_distance(p);

        if d < min_dist {
            hit = true;
            break;
        }

        // Advance by the distance, clamped to avoid taking tiny steps
        t += max(d, sdf_params.voxel_size * 0.25);

        if t > max_t {
            break;
        }
    }

    if !hit {
        // No surface found along this ray
        discard;
    }

    let hit_pos = ray_origin + ray_dir * t;
    let normal = estimate_normal(hit_pos);

    // ── Shading ────────────────────────────────────────────────────────────
    // Simple three-light setup + ambient
    let sun_dir = normalize(vec3<f32>(0.4, 0.7, -0.3));
    let fill_dir = normalize(vec3<f32>(-0.3, 0.4, 0.6));

    let n_dot_sun  = max(dot(normal, sun_dir), 0.0);
    let n_dot_fill = max(dot(normal, fill_dir), 0.0);

    // Hemisphere ambient: bright from above, dim from below
    let sky_ambient = mix(0.08, 0.25, normal.y * 0.5 + 0.5);

    let base_color = vec3<f32>(0.75, 0.76, 0.78);
    let color = base_color * (n_dot_sun * 0.8 + n_dot_fill * 0.25 + sky_ambient);

    // ── Depth output ───────────────────────────────────────────────────────
    let clip_pos = camera.view_proj * vec4<f32>(hit_pos, 1.0);
    let ndc_depth = clip_pos.z / clip_pos.w;

    var out: FragOutput;
    out.color = vec4<f32>(color, 1.0);
    out.depth = ndc_depth;
    return out;
}
