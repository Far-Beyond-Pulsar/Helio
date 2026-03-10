// SDF Clip Map Ray March Shader
//
// Multi-level sphere tracing through nested clip map volumes.
// Each level has its own atlas texture and brick index in a concatenated buffer.
// LOD selection picks the finest level containing the sample point, with
// smooth blending near level edges.

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

struct GpuClipLevel {
    volume_min:           vec3<f32>,
    _pad0:                f32,
    volume_max:           vec3<f32>,
    _pad1:                f32,
    voxel_size:           f32,
    brick_size:           u32,
    brick_grid_dim:       u32,
    atlas_bricks_per_axis: u32,
}

struct SdfClipMapParams {
    level_count:    u32,
    grid_dim:       u32,
    max_march_dist: f32,
    debug_flags:    u32,
    level0:         GpuClipLevel,
    level1:         GpuClipLevel,
    level2:         GpuClipLevel,
    level3:         GpuClipLevel,
}

// Group 0: Global (camera + globals)
@group(0) @binding(0) var<uniform> camera:  Camera;
@group(0) @binding(1) var<uniform> globals: Globals;

// Group 1: Clip map — params + 4 atlas textures + concatenated brick indices
@group(1) @binding(0) var<uniform>       clip_params:       SdfClipMapParams;
@group(1) @binding(1) var                atlas_0:           texture_3d<f32>;
@group(1) @binding(2) var                atlas_1:           texture_3d<f32>;
@group(1) @binding(3) var                atlas_2:           texture_3d<f32>;
@group(1) @binding(4) var                atlas_3:           texture_3d<f32>;
@group(1) @binding(5) var<storage, read> all_brick_indices: array<u32>;

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

// ─── Helper: Get clip level by index ────────────────────────────────────────

fn get_level(level: u32) -> GpuClipLevel {
    switch level {
        case 0u: { return clip_params.level0; }
        case 1u: { return clip_params.level1; }
        case 2u: { return clip_params.level2; }
        case 3u: { return clip_params.level3; }
        default: { return clip_params.level0; }
    }
}

// ─── Brick Lookup ───────────────────────────────────────────────────────────

fn load_voxel_level(level: u32, voxel_coord: vec3<i32>) -> f32 {
    let dim = i32(clip_params.grid_dim);
    let lvl = get_level(level);
    let brick_skip = lvl.voxel_size * f32(lvl.brick_size);

    if any(voxel_coord < vec3<i32>(0)) || any(voxel_coord >= vec3<i32>(dim, dim, dim)) {
        return brick_skip;
    }

    let vc = vec3<u32>(voxel_coord);
    let bs = lvl.brick_size;
    let bgd = lvl.brick_grid_dim;

    let brick_coord = vc / bs;
    let brick_linear = brick_coord.x + brick_coord.y * bgd + brick_coord.z * bgd * bgd;
    let idx_offset = level * bgd * bgd * bgd;
    let atlas_slot = all_brick_indices[idx_offset + brick_linear];

    if atlas_slot == 0xFFFFFFFFu {
        return brick_skip;
    }

    let bpa = lvl.atlas_bricks_per_axis;
    let atlas_brick = vec3<u32>(
        atlas_slot % bpa,
        (atlas_slot / bpa) % bpa,
        atlas_slot / (bpa * bpa),
    );
    let local_voxel = vc % bs;
    let atlas_texel = vec3<i32>(atlas_brick * bs + local_voxel);

    switch level {
        case 0u: { return textureLoad(atlas_0, atlas_texel, 0).r; }
        case 1u: { return textureLoad(atlas_1, atlas_texel, 0).r; }
        case 2u: { return textureLoad(atlas_2, atlas_texel, 0).r; }
        case 3u: { return textureLoad(atlas_3, atlas_texel, 0).r; }
        default: { return clip_params.max_march_dist; }
    }
}

// ─── Level UVW and Sampling ─────────────────────────────────────────────────

fn level_uvw(level: u32, world_pos: vec3<f32>) -> vec3<f32> {
    let lvl = get_level(level);
    return (world_pos - lvl.volume_min) / (lvl.volume_max - lvl.volume_min);
}

fn sample_level(level: u32, world_pos: vec3<f32>) -> f32 {
    let uvw = level_uvw(level, world_pos);

    let voxel_f = uvw * f32(clip_params.grid_dim) - 0.5;
    let base = vec3<i32>(floor(voxel_f));
    let frac = fract(voxel_f);

    let c000 = load_voxel_level(level, base + vec3<i32>(0, 0, 0));
    let c100 = load_voxel_level(level, base + vec3<i32>(1, 0, 0));
    let c010 = load_voxel_level(level, base + vec3<i32>(0, 1, 0));
    let c110 = load_voxel_level(level, base + vec3<i32>(1, 1, 0));
    let c001 = load_voxel_level(level, base + vec3<i32>(0, 0, 1));
    let c101 = load_voxel_level(level, base + vec3<i32>(1, 0, 1));
    let c011 = load_voxel_level(level, base + vec3<i32>(0, 1, 1));
    let c111 = load_voxel_level(level, base + vec3<i32>(1, 1, 1));

    let c00 = mix(c000, c100, frac.x);
    let c10 = mix(c010, c110, frac.x);
    let c01 = mix(c001, c101, frac.x);
    let c11 = mix(c011, c111, frac.x);
    let c0 = mix(c00, c10, frac.y);
    let c1 = mix(c01, c11, frac.y);
    return mix(c0, c1, frac.z);
}

fn sample_distance(world_pos: vec3<f32>) -> f32 {
    // Find the finest level that contains this point
    var finest = -1i;
    for (var level = 0u; level < clip_params.level_count; level++) {
        let uvw = level_uvw(level, world_pos);
        if all(uvw >= vec3<f32>(0.0)) && all(uvw <= vec3<f32>(1.0)) {
            finest = i32(level);
            break;
        }
    }

    if finest < 0 {
        // Outside all levels — return a conservative skip distance based on coarsest level
        let coarsest = get_level(clip_params.level_count - 1u);
        return coarsest.voxel_size * f32(coarsest.brick_size);
    }

    let d_fine = sample_level(u32(finest), world_pos);

    // Blend with coarser level near edges to prevent popping
    if finest + 1 < i32(clip_params.level_count) {
        let uvw = level_uvw(u32(finest), world_pos);
        let edge_dist = min(
            min(min(uvw.x, uvw.y), uvw.z),
            min(min(1.0 - uvw.x, 1.0 - uvw.y), 1.0 - uvw.z)
        );
        let blend_width = 0.05;
        if edge_dist < blend_width {
            let d_coarse = sample_level(u32(finest + 1), world_pos);
            let t = smoothstep(0.0, blend_width, edge_dist);
            return mix(d_coarse, d_fine, t);
        }
    }

    return d_fine;
}

fn estimate_normal(p: vec3<f32>) -> vec3<f32> {
    // Use the finest level's voxel size for normal estimation epsilon
    let lvl = get_level(0u);
    let e = lvl.voxel_size;
    let n = vec3<f32>(
        sample_distance(p + vec3<f32>(e, 0.0, 0.0)) - sample_distance(p - vec3<f32>(e, 0.0, 0.0)),
        sample_distance(p + vec3<f32>(0.0, e, 0.0)) - sample_distance(p - vec3<f32>(0.0, e, 0.0)),
        sample_distance(p + vec3<f32>(0.0, 0.0, e)) - sample_distance(p - vec3<f32>(0.0, 0.0, e)),
    );
    return normalize(n);
}

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
    let clip = vec4<f32>(in.ndc_xy, 1.0, 1.0);
    let world_h = camera.view_proj_inv * clip;
    let world_pt = world_h.xyz / world_h.w;
    let ray_dir = normalize(world_pt - camera.position);
    let ray_origin = camera.position;

    // Use the outermost level's volume for AABB intersection
    let outer_level = get_level(clip_params.level_count - 1u);
    let ray_dir_inv = 1.0 / ray_dir;
    let aabb_hit = intersect_aabb(ray_origin, ray_dir_inv, outer_level.volume_min, outer_level.volume_max);

    if aabb_hit.x > aabb_hit.y || aabb_hit.y < 0.0 {
        discard;
    }

    var t = max(aabb_hit.x, 0.0);
    let max_t = aabb_hit.y;
    let finest_lvl = get_level(0u);
    let min_dist = finest_lvl.voxel_size * 0.5;

    var hit = false;
    for (var i = 0u; i < 256u; i++) {
        let p = ray_origin + ray_dir * t;
        let d = sample_distance(p);

        if d < min_dist {
            hit = true;
            break;
        }

        t += max(d, finest_lvl.voxel_size * 0.25);

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
    if clip_params.debug_flags == 1u {
        // Level tint (subtle accent on edges)
        let level_tints = array<vec3<f32>, 4>(
            vec3<f32>(1.0, 0.3, 0.3),   // Level 0 - Red (finest)
            vec3<f32>(0.3, 1.0, 0.3),   // Level 1 - Green
            vec3<f32>(0.3, 0.3, 1.0),   // Level 2 - Blue
            vec3<f32>(1.0, 1.0, 0.3),   // Level 3 - Yellow (coarsest)
        );

        // Find which level this hit belongs to
        var hit_level = 0u;
        for (var level = 0u; level < clip_params.level_count; level++) {
            let uvw = level_uvw(level, hit_pos);
            if all(uvw >= vec3<f32>(0.0)) && all(uvw <= vec3<f32>(1.0)) {
                hit_level = level;
                break;
            }
        }

        // Per-brick hash color — each brick gets a unique color
        let lvl = get_level(hit_level);
        let uvw_d = level_uvw(hit_level, hit_pos);
        let voxel_d = uvw_d * f32(clip_params.grid_dim);
        let brick_coord = vec3<u32>(floor(voxel_d / f32(lvl.brick_size)));
        // Simple hash for distinct per-brick colors
        let hash = brick_coord.x * 73u + brick_coord.y * 157u + brick_coord.z * 311u;
        let brick_color = vec3<f32>(
            f32((hash * 1234567u) % 256u) / 255.0,
            f32((hash * 7654321u) % 256u) / 255.0,
            f32((hash * 2468135u) % 256u) / 255.0,
        ) * 0.6 + 0.2;  // keep in [0.2, 0.8] range for visibility

        // Brick boundary lines (dark grid edges)
        let local_in_brick = fract(voxel_d / f32(lvl.brick_size));
        let edge_dist = min(
            min(min(local_in_brick.x, local_in_brick.y), local_in_brick.z),
            min(min(1.0 - local_in_brick.x, 1.0 - local_in_brick.y), 1.0 - local_in_brick.z)
        );
        let edge_width = 0.05;
        let edge_factor = smoothstep(0.0, edge_width, edge_dist);

        // Level tint on edges, brick color on faces
        let level_tint = level_tints[hit_level];
        let face_color = brick_color;
        let edge_color = level_tint * 0.8;
        let combined = mix(edge_color, face_color, edge_factor);

        // Lighting to preserve shape readability
        let sun_dir_d = normalize(vec3<f32>(0.4, 0.7, -0.3));
        let n_dot_sun_d = max(dot(normal, sun_dir_d), 0.0) * 0.5 + 0.3;

        let debug_color = combined * n_dot_sun_d;

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
