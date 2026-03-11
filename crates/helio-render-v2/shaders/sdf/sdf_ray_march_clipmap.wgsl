// SDF Clip Map Ray March Shader
//
// Multi-level sphere tracing through nested clip map volumes.
// Each level has its own atlas texture and brick index in a concatenated buffer.
// LOD selection picks the finest level containing the sample point, with
// smooth blending near level edges.
//
// Ghost voxels: each brick is stored as 9^3 in the atlas (PADDED_SIZE),
// enabling seamless intra-brick trilinear interpolation without sentinels.

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
    grid_origin:          vec3<f32>,
    _pad2:                f32,
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
    level4:         GpuClipLevel,
    level5:         GpuClipLevel,
    level6:         GpuClipLevel,
    level7:         GpuClipLevel,
}

// Group 0: Global (camera + globals)
@group(0) @binding(0) var<uniform> camera:  Camera;
@group(0) @binding(1) var<uniform> globals: Globals;

// Group 1: Clip map — params + 8 atlas buffers + concatenated brick indices
@group(1) @binding(0) var<uniform>       clip_params:       SdfClipMapParams;
@group(1) @binding(1) var<storage, read> atlas_0:           array<u32>;
@group(1) @binding(2) var<storage, read> atlas_1:           array<u32>;
@group(1) @binding(3) var<storage, read> atlas_2:           array<u32>;
@group(1) @binding(4) var<storage, read> atlas_3:           array<u32>;
@group(1) @binding(5) var<storage, read> atlas_4:           array<u32>;
@group(1) @binding(6) var<storage, read> atlas_5:           array<u32>;
@group(1) @binding(7) var<storage, read> atlas_6:           array<u32>;
@group(1) @binding(8) var<storage, read> atlas_7:           array<u32>;
@group(1) @binding(9) var<storage, read> all_brick_indices: array<u32>;

const PADDED_SIZE: u32 = 9u;

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
        case 4u: { return clip_params.level4; }
        case 5u: { return clip_params.level5; }
        case 6u: { return clip_params.level6; }
        case 7u: { return clip_params.level7; }
        default: { return clip_params.level0; }
    }
}

// ─── Atlas dequantization helpers ────────────────────────────────────────
// Atlas stores u8 packed 4-per-u32. Dequantize back to signed distance.

fn atlas_half_diag_for_level(lvl: GpuClipLevel) -> f32 {
    return lvl.voxel_size * f32(lvl.brick_size) * 0.866025403; // sqrt(3)/2
}

fn read_atlas_word(level: u32, word_idx: u32) -> u32 {
    switch level {
        case 0u: { return atlas_0[word_idx]; }
        case 1u: { return atlas_1[word_idx]; }
        case 2u: { return atlas_2[word_idx]; }
        case 3u: { return atlas_3[word_idx]; }
        case 4u: { return atlas_4[word_idx]; }
        case 5u: { return atlas_5[word_idx]; }
        case 6u: { return atlas_6[word_idx]; }
        case 7u: { return atlas_7[word_idx]; }
        default: { return 0u; }
    }
}

fn load_atlas(level: u32, texel: vec3<i32>) -> f32 {
    let lvl = get_level(level);
    let dim = lvl.atlas_bricks_per_axis * PADDED_SIZE;
    let ut = vec3<u32>(texel);
    let linear = ut.x + ut.y * dim + ut.z * dim * dim;
    let word_idx = linear / 4u;
    let byte_idx = linear % 4u;
    let word = read_atlas_word(level, word_idx);
    let byte_val = (word >> (byte_idx * 8u)) & 0xFFu;
    let normalized = f32(byte_val) / 255.0;
    let hd = atlas_half_diag_for_level(lvl);
    return (normalized - 0.5) * 2.0 * hd;
}

// ─── Level UVW ──────────────────────────────────────────────────────────────

fn level_uvw(level: u32, world_pos: vec3<f32>) -> vec3<f32> {
    let lvl = get_level(level);
    return (world_pos - lvl.volume_min) / (lvl.volume_max - lvl.volume_min);
}

// ─── Intra-brick trilinear sampling with ghost voxels ───────────────────────

fn sample_level(level: u32, world_pos: vec3<f32>) -> f32 {
    let lvl = get_level(level);
    let vs = lvl.voxel_size;
    let bs = lvl.brick_size;
    let bgd = lvl.brick_grid_dim;
    let bpa = lvl.atlas_bricks_per_axis;
    let brick_skip = vs * f32(bs);

    if any(world_pos < lvl.volume_min) || any(world_pos > lvl.volume_max) {
        return brick_skip;
    }

    // 1. Continuous voxel coordinate (world_pos / vs).
    //    This is the fundamental coordinate — floor gives the integer voxel,
    //    the fractional part gives interpolation weights.
    let voxel_f = world_pos / vs;
    let world_voxel = vec3<i32>(floor(voxel_f));

    // 2. Toroidal wrapping — must match the CPU's world_to_grid() exactly.
    //    CPU: grid_idx = ((world_brick % bgd) + bgd) % bgd for each axis.
    //    world_brick = floor(world_pos / brick_world_size) = floor(voxel_f / bs).
    //    Using the continuous voxel coordinate avoids integer floor-division issues
    //    with negative coordinates (WGSL / truncates toward zero, not -inf).
    let bgdi = i32(bgd);
    let world_brick = vec3<i32>(floor(voxel_f / f32(bs)));
    let brick_coord = vec3<u32>(((world_brick % bgdi) + bgdi) % bgdi);

    // Voxel's position within the brick — offset from the brick's world-voxel base
    let local_voxel = vec3<u32>(world_voxel - world_brick * i32(bs));
    let brick_linear = brick_coord.x + brick_coord.y * bgd + brick_coord.z * bgd * bgd;
    let idx_offset = level * bgd * bgd * bgd;
    let atlas_slot = all_brick_indices[idx_offset + brick_linear];

    if atlas_slot == 0xFFFFFFFFu {
        return brick_skip;
    }

    // 4. Atlas origin for this brick (padded stride)
    let atlas_brick = vec3<u32>(
        atlas_slot % bpa,
        (atlas_slot / bpa) % bpa,
        atlas_slot / (bpa * bpa),
    );
    let atlas_origin = vec3<f32>(atlas_brick * PADDED_SIZE);

    // 5. Continuous local position within the brick [0.0, 8.0+).
    //    Corner convention: texel k stores SDF at (brick_world_base + k) * vs.
    //    local_voxel = position within the brick (integer part from grid coords),
    //    sub_voxel = sub-voxel fractional part from the continuous world coords.
    let sub_voxel = voxel_f - vec3<f32>(world_voxel);
    let local_f = vec3<f32>(local_voxel) + sub_voxel;

    // 6. Trilinear interpolation — direct floor/fract, no offset needed.
    //    All texel indices stay in [0..8], fully within the padded 9^3 region.
    let base = vec3<i32>(floor(local_f));
    let frac = local_f - vec3<f32>(vec3<i32>(floor(local_f)));
    let b = base + vec3<i32>(atlas_origin);

    // All 8 corners within padded [0..8] range — no sentinel possible
    let c000 = load_atlas(level, b);
    let c100 = load_atlas(level, b + vec3<i32>(1, 0, 0));
    let c010 = load_atlas(level, b + vec3<i32>(0, 1, 0));
    let c110 = load_atlas(level, b + vec3<i32>(1, 1, 0));
    let c001 = load_atlas(level, b + vec3<i32>(0, 0, 1));
    let c101 = load_atlas(level, b + vec3<i32>(1, 0, 1));
    let c011 = load_atlas(level, b + vec3<i32>(0, 1, 1));
    let c111 = load_atlas(level, b + vec3<i32>(1, 1, 1));

    let c00 = mix(c000, c100, frac.x);
    let c10 = mix(c010, c110, frac.x);
    let c01 = mix(c001, c101, frac.x);
    let c11 = mix(c011, c111, frac.x);
    let c0 = mix(c00, c10, frac.y);
    let c1 = mix(c01, c11, frac.y);
    return mix(c0, c1, frac.z);
}

fn sample_distance(world_pos: vec3<f32>) -> f32 {
    // Walk from finest to coarsest, picking the first level that both
    // contains the point AND has a populated brick at this position.
    // This prevents EMPTY_BRICK gaps at the finest level from causing
    // the ray march to skip over real geometry in coarser levels.
    var finest = -1i;
    var d_fine = 0.0;
    for (var level = 0u; level < clip_params.level_count; level++) {
        let uvw = level_uvw(level, world_pos);
        if all(uvw >= vec3<f32>(0.0)) && all(uvw <= vec3<f32>(1.0)) {
            let d = sample_level(level, world_pos);
            let lvl = get_level(level);
            let brick_skip = lvl.voxel_size * f32(lvl.brick_size);
            // If this level returned brick_skip, the brick is EMPTY — try coarser
            if d < brick_skip - 0.01 {
                finest = i32(level);
                d_fine = d;
                break;
            }
        }
    }

    if finest < 0 {
        // Outside all levels or all bricks empty
        let coarsest = get_level(clip_params.level_count - 1u);
        return coarsest.voxel_size * f32(coarsest.brick_size);
    }

    // Blend with coarser level near edges to prevent popping.
    if finest + 1 < i32(clip_params.level_count) {
        let uvw = level_uvw(u32(finest), world_pos);
        let edge_dist = min(
            min(min(uvw.x, uvw.y), uvw.z),
            min(min(1.0 - uvw.x, 1.0 - uvw.y), 1.0 - uvw.z)
        );
        let blend_width = 0.1;
        if edge_dist < blend_width {
            let d_coarse = sample_level(u32(finest + 1), world_pos);
            let lvl_c = get_level(u32(finest + 1));
            let brick_skip_c = lvl_c.voxel_size * f32(lvl_c.brick_size);
            // Only blend if the coarser level has a valid brick
            if d_coarse < brick_skip_c - 0.01 {
                let t = smoothstep(0.0, blend_width, edge_dist);
                return mix(d_coarse, d_fine, t);
            }
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
    let step_size = finest_lvl.voxel_size * 0.5;

    // Handle camera inside geometry: if the initial sample is negative (inside
    // solid), march forward at fixed steps until we exit the surface.  Without
    // this, the very first sample triggers `d < min_dist` and creates a false
    // hit at the camera position, causing the visual artifacts.
    let d_start = sample_distance(ray_origin + ray_dir * t);
    var inside_solid = d_start < 0.0;
    if inside_solid {
        for (var skip = 0u; skip < 128u; skip++) {
            t += step_size;
            if t > max_t { break; }
            let d_skip = sample_distance(ray_origin + ray_dir * t);
            if d_skip >= min_dist {
                inside_solid = false;
                break;
            }
        }
    }

    var hit = false;
    if !inside_solid {
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
    }

    if !hit {
        discard;
    }

    let hit_pos = ray_origin + ray_dir * t;
    let normal = estimate_normal(hit_pos);

    // ── Debug visualization ──────────────────────────────────────────────
    if clip_params.debug_flags == 1u {
        // Level tint colors (8 levels)
        let level_tints = array<vec3<f32>, 8>(
            vec3<f32>(1.0, 0.3, 0.3),   // Level 0 - Red (finest)
            vec3<f32>(0.3, 1.0, 0.3),   // Level 1 - Green
            vec3<f32>(0.3, 0.3, 1.0),   // Level 2 - Blue
            vec3<f32>(1.0, 1.0, 0.3),   // Level 3 - Yellow
            vec3<f32>(1.0, 0.3, 1.0),   // Level 4 - Magenta
            vec3<f32>(0.3, 1.0, 1.0),   // Level 5 - Cyan
            vec3<f32>(1.0, 0.6, 0.2),   // Level 6 - Orange
            vec3<f32>(0.7, 0.3, 1.0),   // Level 7 - Purple (coarsest)
        );

        // Find which level this hit belongs to (matching sample_distance logic)
        var hit_level = 0u;
        for (var level = 0u; level < clip_params.level_count; level++) {
            let uvw = level_uvw(level, hit_pos);
            if all(uvw >= vec3<f32>(0.0)) && all(uvw <= vec3<f32>(1.0)) {
                let d = sample_level(level, hit_pos);
                let lvl_check = get_level(level);
                let brick_skip_check = lvl_check.voxel_size * f32(lvl_check.brick_size);
                if d < brick_skip_check - 0.01 {
                    hit_level = level;
                    break;
                }
            }
        }

        // Per-brick hash color — use world-space brick coordinate for stable colors
        let lvl = get_level(hit_level);
        let voxel_f_d = hit_pos / lvl.voxel_size;
        let world_brick_d = vec3<i32>(floor(voxel_f_d / f32(lvl.brick_size)));
        // Use world-space brick coords directly for hashing — stable across scrolls
        let hash_coord = vec3<u32>(((world_brick_d % 256) + 256) % 256);
        let hash = hash_coord.x * 73u + hash_coord.y * 157u + hash_coord.z * 311u;
        let brick_color = vec3<f32>(
            f32((hash * 1234567u) % 256u) / 255.0,
            f32((hash * 7654321u) % 256u) / 255.0,
            f32((hash * 2468135u) % 256u) / 255.0,
        ) * 0.6 + 0.2;

        // Brick boundary lines (dark grid edges)
        let local_in_brick = fract(voxel_f_d / f32(lvl.brick_size));
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
