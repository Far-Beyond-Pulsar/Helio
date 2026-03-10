// SDF Sparse Brick Ray March Shader
//
// Renders SDF geometry by sphere-tracing through the sparse brick atlas.
// Uses manual trilinear interpolation via textureLoad to avoid hardware
// filtering bleeding across brick boundaries in the atlas texture.

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
    volume_min:           vec3<f32>,
    _pad0:                f32,
    volume_max:           vec3<f32>,
    _pad1:                f32,
    grid_dim:             u32,
    edit_count:           u32,
    voxel_size:           f32,
    max_march_dist:       f32,
    brick_size:           u32,
    brick_grid_dim:       u32,
    active_brick_count:   u32,
    atlas_bricks_per_axis: u32,
    debug_flags:          u32,
    _pad2:                u32,
    _pad3:                u32,
    _pad4:                u32,
}

// Group 0: Global (camera + globals) — shared with all passes
@group(0) @binding(0) var<uniform> camera:  Camera;
@group(0) @binding(1) var<uniform> globals: Globals;

// Group 1: SDF sparse — params + atlas texture + brick index
@group(1) @binding(0) var<uniform>       sdf_params:  SdfGridParams;
@group(1) @binding(1) var                sdf_atlas:   texture_3d<f32>;
@group(1) @binding(2) var<storage, read> brick_index:  array<u32>;

// ─── Vertex Shader (fullscreen triangle) ───────────────────────────────────

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) ndc_xy: vec2<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) vid: u32) -> VertexOutput {
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

// World-space size of one brick (used as conservative skip distance for empty bricks)
fn brick_world_size() -> f32 {
    return sdf_params.voxel_size * f32(sdf_params.brick_size);
}

fn world_to_uvw(world_pos: vec3<f32>) -> vec3<f32> {
    return (world_pos - sdf_params.volume_min) /
           (sdf_params.volume_max - sdf_params.volume_min);
}

/// Load a single voxel from the sparse brick atlas.
/// Returns the SDF distance, or a conservative brick-sized skip distance
/// for empty/out-of-bounds bricks (NOT max_march_dist, which would overshoot).
fn load_voxel(voxel_coord: vec3<i32>) -> f32 {
    let dim = i32(sdf_params.grid_dim);
    if any(voxel_coord < vec3<i32>(0)) || any(voxel_coord >= vec3<i32>(dim, dim, dim)) {
        return brick_world_size();
    }

    let vc = vec3<u32>(voxel_coord);
    let bs = sdf_params.brick_size;
    let bgd = sdf_params.brick_grid_dim;

    // Which brick does this voxel belong to?
    let brick_coord = vc / bs;
    let brick_linear = brick_coord.x + brick_coord.y * bgd + brick_coord.z * bgd * bgd;
    let atlas_slot = brick_index[brick_linear];

    if atlas_slot == 0xFFFFFFFFu {
        // Empty brick — return a skip distance proportional to brick size
        // so the ray advances through empty space at ~1 brick per step
        return brick_world_size();
    }

    // Decode atlas brick position from slot
    let bpa = sdf_params.atlas_bricks_per_axis;
    let atlas_brick = vec3<u32>(
        atlas_slot % bpa,
        (atlas_slot / bpa) % bpa,
        atlas_slot / (bpa * bpa),
    );
    let local_voxel = vc % bs;
    let atlas_texel = vec3<i32>(atlas_brick * bs + local_voxel);
    return textureLoad(sdf_atlas, atlas_texel, 0).r;
}

/// Sample the SDF with manual trilinear interpolation.
/// Uses 8 textureLoad calls to avoid hardware filtering bleeding
/// across brick boundaries in the packed atlas.
fn sample_distance(world_pos: vec3<f32>) -> f32 {
    let uvw = world_to_uvw(world_pos);
    if any(uvw < vec3<f32>(0.0)) || any(uvw > vec3<f32>(1.0)) {
        return brick_world_size();
    }

    // Center-of-texel convention: subtract 0.5 before flooring
    let voxel_f = uvw * f32(sdf_params.grid_dim) - 0.5;
    let base = vec3<i32>(floor(voxel_f));
    let frac = fract(voxel_f);

    // Load 8 corner samples
    let c000 = load_voxel(base + vec3<i32>(0, 0, 0));
    let c100 = load_voxel(base + vec3<i32>(1, 0, 0));
    let c010 = load_voxel(base + vec3<i32>(0, 1, 0));
    let c110 = load_voxel(base + vec3<i32>(1, 1, 0));
    let c001 = load_voxel(base + vec3<i32>(0, 0, 1));
    let c101 = load_voxel(base + vec3<i32>(1, 0, 1));
    let c011 = load_voxel(base + vec3<i32>(0, 1, 1));
    let c111 = load_voxel(base + vec3<i32>(1, 1, 1));

    // Trilinear interpolation
    let c00 = mix(c000, c100, frac.x);
    let c10 = mix(c010, c110, frac.x);
    let c01 = mix(c001, c101, frac.x);
    let c11 = mix(c011, c111, frac.x);
    let c0 = mix(c00, c10, frac.y);
    let c1 = mix(c01, c11, frac.y);
    return mix(c0, c1, frac.z);
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
    // Reconstruct world-space ray direction
    let clip = vec4<f32>(in.ndc_xy, 1.0, 1.0);
    let world_h = camera.view_proj_inv * clip;
    let world_pt = world_h.xyz / world_h.w;
    let ray_dir = normalize(world_pt - camera.position);
    let ray_origin = camera.position;

    // Intersect ray with the SDF volume AABB
    let ray_dir_inv = 1.0 / ray_dir;
    let aabb_hit = intersect_aabb(ray_origin, ray_dir_inv, sdf_params.volume_min, sdf_params.volume_max);

    if aabb_hit.x > aabb_hit.y || aabb_hit.y < 0.0 {
        discard;
    }

    // Start marching from the AABB entry point (or camera if inside)
    var t = max(aabb_hit.x, 0.0);
    let max_t = aabb_hit.y;
    let min_dist = sdf_params.voxel_size * 0.5;

    var hit = false;
    for (var i = 0u; i < 256u; i++) {
        let p = ray_origin + ray_dir * t;
        let d = sample_distance(p);

        if d < min_dist {
            hit = true;
            break;
        }

        t += max(d, sdf_params.voxel_size * 0.25);

        if t > max_t {
            break;
        }
    }

    if !hit {
        discard;
    }

    let hit_pos = ray_origin + ray_dir * t;
    let normal = estimate_normal(hit_pos);

    // ── Debug visualization ──────────────────────────────────────────────
    if sdf_params.debug_flags == 1u {
        // Per-brick hash color — each brick gets a unique color
        let uvw = world_to_uvw(hit_pos);
        let voxel = uvw * f32(sdf_params.grid_dim);
        let brick_coord = vec3<u32>(floor(voxel / f32(sdf_params.brick_size)));
        let hash = brick_coord.x * 73u + brick_coord.y * 157u + brick_coord.z * 311u;
        let brick_color = vec3<f32>(
            f32((hash * 1234567u) % 256u) / 255.0,
            f32((hash * 7654321u) % 256u) / 255.0,
            f32((hash * 2468135u) % 256u) / 255.0,
        ) * 0.6 + 0.2;

        // Brick boundary lines (dark grid edges)
        let local_in_brick = fract(voxel / f32(sdf_params.brick_size));
        let edge_dist = min(
            min(min(local_in_brick.x, local_in_brick.y), local_in_brick.z),
            min(min(1.0 - local_in_brick.x, 1.0 - local_in_brick.y), 1.0 - local_in_brick.z)
        );
        let edge_width = 0.05;
        let edge_factor = smoothstep(0.0, edge_width, edge_dist);

        // Dark edges, colored brick faces
        let debug_color_raw = mix(vec3<f32>(0.15), brick_color, edge_factor);

        // Lighting to preserve shape readability
        let sun_dir_d = normalize(vec3<f32>(0.4, 0.7, -0.3));
        let n_dot_sun_d = max(dot(normal, sun_dir_d), 0.0) * 0.5 + 0.3;

        let debug_color = debug_color_raw * n_dot_sun_d;

        let clip_pos_d = camera.view_proj * vec4<f32>(hit_pos, 1.0);
        let ndc_depth_d = clip_pos_d.z / clip_pos_d.w;
        var out_d: FragOutput;
        out_d.color = vec4<f32>(debug_color, 1.0);
        out_d.depth = ndc_depth_d;
        return out_d;
    }

    // ── Shading ────────────────────────────────────────────────────────────
    let sun_dir = normalize(vec3<f32>(0.4, 0.7, -0.3));
    let fill_dir = normalize(vec3<f32>(-0.3, 0.4, 0.6));

    let n_dot_sun  = max(dot(normal, sun_dir), 0.0);
    let n_dot_fill = max(dot(normal, fill_dir), 0.0);

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
