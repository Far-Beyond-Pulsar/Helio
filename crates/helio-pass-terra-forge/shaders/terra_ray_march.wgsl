// Terra Forge — 3-level chunk/brick/voxel DDA ray marcher (compute shader).
//
// Level 0 (chunk):  DDA through chunk grid via indirection table
// Level 1 (brick):  DDA through 32³ brick grid within a chunk
// Level 2 (voxel):  DDA through 8³ voxels within a brick
//
// Continuous parametric t across all levels (no restart at boundaries).

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

struct Uniforms {
    width:            u32,
    height:           u32,
    brick_dim:        u32,
    chunk_dim_bricks: u32,
    voxel_size:       f32,
    planet_radius:    f32,
    indir_grid_dim:   u32,
    edit_count:       u32,
    indir_origin:     vec3<i32>,
    ff_cell_size:     f32,
    camera_offset:    vec3<f32>,
    _pad_cam:         f32,
    jitter:           vec2<f32>,
    _jitter_pad:      vec2<f32>,
}

struct BrickMeta {
    data_offset: u32,
    occupancy:   u32,
}

struct ChunkInfo {
    pos_x:             i32,
    pos_y:             i32,
    pos_z:             i32,
    status:            u32,
    brick_pool_offset: u32,
    voxel_pool_offset: u32,
    _pad0:             u32,
    _pad1:             u32,
}

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<uniform> camera: Camera;
@group(0) @binding(2) var<storage, read> chunk_table: array<ChunkInfo>;
@group(0) @binding(3) var<storage, read> indir_grid:  array<u32>;
@group(0) @binding(4) var<storage, read> brick_pool:  array<BrickMeta>;
@group(0) @binding(5) var<storage, read> voxel_pool:  array<u32>;
@group(0) @binding(6) var out_material: texture_storage_2d<r32uint,     write>;
@group(0) @binding(7) var out_normal:   texture_storage_2d<rgba16float, write>;

struct EditOp {
    shape_type: u32,
    op_type:    u32,
    material:   u32,
    blend_k:    f32,
    position:   vec3<f32>,
    _pad0:      f32,
    size:       vec3<f32>,
    _pad1:      f32,
}

@group(0) @binding(8) var<storage, read> edit_buffer: array<EditOp>;

const INDIR_EMPTY:    u32 = 0xFFFFFFFFu;
const BRICK_EMPTY:    u32 = 0xFFFFFFFFu;
const BRICK_SOLID:    u32 = 0xFFFFFFFEu;
const WORDS_PER_BRICK: u32 = 128u;

// ── Utility ──────────────────────────────────────────────────────────────────

fn ray_aabb(ro: vec3<f32>, inv_rd: vec3<f32>,
            bmin: vec3<f32>, bmax: vec3<f32>) -> vec2<f32> {
    let t0 = (bmin - ro) * inv_rd;
    let t1 = (bmax - ro) * inv_rd;
    let tmin = min(t0, t1);
    let tmax = max(t0, t1);
    return vec2<f32>(max(max(tmin.x, tmin.y), tmin.z),
                     min(min(tmax.x, tmax.y), tmax.z));
}

fn wrap_coord(v: i32, dim: i32) -> u32 {
    return u32(((v % dim) + dim) % dim);
}

// ── Level 2: Voxel DDA within a single brick (8³) ───────────────────────────

fn read_voxel(data_offset: u32, lx: u32, ly: u32, lz: u32) -> u32 {
    let bd = u.brick_dim;
    let local_idx = lx + ly * bd + lz * bd * bd;
    let word_idx = data_offset * WORDS_PER_BRICK + (local_idx >> 2u);
    let byte_shift = (local_idx & 3u) * 8u;
    return (voxel_pool[word_idx] >> byte_shift) & 0xFFu;
}

fn trace_brick(
    ro: vec3<f32>, rd: vec3<f32>, inv_rd: vec3<f32>,
    data_offset: u32, brick_origin: vec3<f32>,
    t_enter: f32, t_exit: f32,
    coarse_normal: vec3<f32>,
    hit_mat: ptr<function, u32>,
) -> vec4<f32> {
    let bd = u.brick_dim;
    let vs = u.voxel_size;

    let entry_pos = ro + rd * t_enter;
    let local_f = (entry_pos - brick_origin) / vs;
    var voxel = vec3<i32>(clamp(vec3<i32>(floor(local_f)),
                                vec3<i32>(0), vec3<i32>(i32(bd) - 1)));

    let step_dir = vec3<i32>(sign(rd));
    let delta = abs(vec3<f32>(vs) * inv_rd);

    var t_max_v = vec3<f32>(
        (brick_origin.x + f32(select(voxel.x, voxel.x + 1, step_dir.x > 0)) * vs - ro.x) * inv_rd.x,
        (brick_origin.y + f32(select(voxel.y, voxel.y + 1, step_dir.y > 0)) * vs - ro.y) * inv_rd.y,
        (brick_origin.z + f32(select(voxel.z, voxel.z + 1, step_dir.z > 0)) * vs - ro.z) * inv_rd.z,
    );

    var normal = coarse_normal;
    var t_cell_enter = t_enter;
    let bd_i = i32(bd);

    for (var i = 0u; i < bd * 3u; i++) {
        if voxel.x < 0 || voxel.y < 0 || voxel.z < 0 ||
           voxel.x >= bd_i || voxel.y >= bd_i || voxel.z >= bd_i {
            break;
        }

        let vmat = read_voxel(data_offset, u32(voxel.x), u32(voxel.y), u32(voxel.z));
        if vmat != 0u {
            *hit_mat = vmat;
            return vec4<f32>(normal, t_cell_enter);
        }

        if t_max_v.x <= t_max_v.y && t_max_v.x <= t_max_v.z {
            normal = vec3<f32>(f32(-step_dir.x), 0.0, 0.0);
            t_cell_enter = t_max_v.x;
            voxel.x += step_dir.x;
            t_max_v.x += delta.x;
        } else if t_max_v.y <= t_max_v.z {
            normal = vec3<f32>(0.0, f32(-step_dir.y), 0.0);
            t_cell_enter = t_max_v.y;
            voxel.y += step_dir.y;
            t_max_v.y += delta.y;
        } else {
            normal = vec3<f32>(0.0, 0.0, f32(-step_dir.z));
            t_cell_enter = t_max_v.z;
            voxel.z += step_dir.z;
            t_max_v.z += delta.z;
        }

        if t_cell_enter > t_exit { break; }
    }

    return vec4<f32>(0.0);
}

// ── Level 1: Brick DDA within a loaded chunk (32³) ──────────────────────────

fn trace_chunk_bricks(
    ro: vec3<f32>, rd: vec3<f32>, inv_rd: vec3<f32>,
    chunk_origin: vec3<f32>,
    brick_pool_offset: u32,
    t_chunk_enter: f32, t_chunk_exit: f32,
    chunk_entry_normal: vec3<f32>,
    hit_mat: ptr<function, u32>,
) -> vec4<f32> {
    let cd = u.chunk_dim_bricks;
    let bd = u.brick_dim;
    let vs = u.voxel_size;
    let brick_world = f32(bd) * vs;

    let entry_pos = ro + rd * t_chunk_enter;
    let bf = (entry_pos - chunk_origin) / brick_world;
    var brick = vec3<i32>(clamp(vec3<i32>(floor(bf)),
                                vec3<i32>(0), vec3<i32>(i32(cd) - 1)));

    let step_dir = vec3<i32>(sign(rd));
    let delta_brick = abs(vec3<f32>(brick_world) * inv_rd);

    var t_max_brick = vec3<f32>(
        (chunk_origin.x + f32(select(brick.x, brick.x + 1, step_dir.x > 0)) * brick_world - ro.x) * inv_rd.x,
        (chunk_origin.y + f32(select(brick.y, brick.y + 1, step_dir.y > 0)) * brick_world - ro.y) * inv_rd.y,
        (chunk_origin.z + f32(select(brick.z, brick.z + 1, step_dir.z > 0)) * brick_world - ro.z) * inv_rd.z,
    );

    var brick_normal = chunk_entry_normal;
    var t_brick_enter = t_chunk_enter;
    let cd_i = i32(cd);

    for (var ci = 0u; ci < cd * 3u; ci++) {
        if brick.x < 0 || brick.y < 0 || brick.z < 0 ||
           brick.x >= cd_i || brick.y >= cd_i || brick.z >= cd_i {
            break;
        }

        let brick_idx = brick_pool_offset
                       + u32(brick.x)
                       + u32(brick.y) * cd
                       + u32(brick.z) * cd * cd;
        let bmeta = brick_pool[brick_idx];
        let t_brick_exit = min(min(t_max_brick.x, t_max_brick.y), t_max_brick.z);

        if bmeta.occupancy > 0u && bmeta.data_offset != BRICK_EMPTY {
            let brick_origin_world = chunk_origin
                + vec3<f32>(f32(brick.x), f32(brick.y), f32(brick.z)) * brick_world;

            if bmeta.data_offset == BRICK_SOLID || t_brick_enter > 200.0 {
                *hit_mat = 1u;
                let n = select(brick_normal, vec3<f32>(0.0, 1.0, 0.0),
                               all(brick_normal == vec3<f32>(0.0)));
                return vec4<f32>(n, t_brick_enter);
            }

            let result = trace_brick(
                ro, rd, inv_rd,
                bmeta.data_offset, brick_origin_world,
                t_brick_enter, t_brick_exit,
                brick_normal, hit_mat,
            );
            if *hit_mat != 0u {
                return result;
            }
        }

        // Step brick DDA.
        if t_max_brick.x <= t_max_brick.y && t_max_brick.x <= t_max_brick.z {
            brick_normal = vec3<f32>(f32(-step_dir.x), 0.0, 0.0);
            t_brick_enter = t_max_brick.x;
            brick.x += step_dir.x;
            t_max_brick.x += delta_brick.x;
        } else if t_max_brick.y <= t_max_brick.z {
            brick_normal = vec3<f32>(0.0, f32(-step_dir.y), 0.0);
            t_brick_enter = t_max_brick.y;
            brick.y += step_dir.y;
            t_max_brick.y += delta_brick.y;
        } else {
            brick_normal = vec3<f32>(0.0, 0.0, f32(-step_dir.z));
            t_brick_enter = t_max_brick.z;
            brick.z += step_dir.z;
            t_max_brick.z += delta_brick.z;
        }
    }

    return vec4<f32>(0.0);
}

// ── Far-field: coarse voxel DDA (world-anchored, no smooth SDF) ──────────────
// Instead of smooth sphere tracing, we step through a coarse world-aligned grid
// and test SDF < 0 at each cell center. This gives stable voxel appearance at
// all distances — no shimmer, no smooth surfaces, always blocky/voxelized.

fn ff_hash3(p: vec3<f32>) -> f32 {
    var q = fract(p * vec3<f32>(0.1031, 0.1030, 0.0973));
    q += dot(q, q.yxz + 33.33);
    return fract((q.x + q.y) * q.z);
}

fn ff_noise3(p: vec3<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let uu = f * f * (3.0 - 2.0 * f);
    return mix(mix(mix(ff_hash3(i + vec3<f32>(0.0, 0.0, 0.0)),
                       ff_hash3(i + vec3<f32>(1.0, 0.0, 0.0)), uu.x),
                   mix(ff_hash3(i + vec3<f32>(0.0, 1.0, 0.0)),
                       ff_hash3(i + vec3<f32>(1.0, 1.0, 0.0)), uu.x), uu.y),
               mix(mix(ff_hash3(i + vec3<f32>(0.0, 0.0, 1.0)),
                       ff_hash3(i + vec3<f32>(1.0, 0.0, 1.0)), uu.x),
                   mix(ff_hash3(i + vec3<f32>(0.0, 1.0, 1.0)),
                       ff_hash3(i + vec3<f32>(1.0, 1.0, 1.0)), uu.x), uu.y), uu.z);
}

fn ff_fbm(p: vec3<f32>) -> f32 {
    var val = 0.0;
    var amp = 0.5;
    var fp = p;
    for (var i = 0; i < 4; i++) {
        val += amp * (ff_noise3(fp) - 0.5);
        fp *= 2.0;
        amp *= 0.5;
    }
    return val;
}

fn ff_fbm2(p: vec3<f32>) -> f32 {
    var val = 0.0;
    var amp = 0.5;
    var fp = p;
    for (var i = 0; i < 2; i++) {
        val += amp * (ff_noise3(fp) - 0.5);
        fp *= 2.0;
        amp *= 0.5;
    }
    return val;
}

fn ff_sdf_terrain(pos: vec3<f32>) -> f32 {
    let base = length(pos) - u.planet_radius;
    let dir = normalize(pos + vec3<f32>(0.001));
    let terrain_noise = ff_fbm(dir * 8.0) * u.planet_radius * 0.05;
    return base - terrain_noise;
}

/// Determine material for a far-field solid cell given its CENTER position.
/// The center is guaranteed to be inside the terrain (SDF < 0 was already checked).
fn ff_material_at_center(center: vec3<f32>) -> u32 {
    let d = ff_sdf_terrain(center);
    let depth = -d;
    let dir = normalize(center + vec3<f32>(0.001));
    let slope_noise = ff_fbm2(dir * 16.0);
    // Scale thresholds by cell_size so material bands stay proportional at all LODs.
    let cs = u.ff_cell_size;
    if depth < cs * 2.0 && slope_noise > -0.1 { return 2u; }
    if depth < cs * 4.0 { return 3u; }
    return 1u;
}

// Two-phase far-field renderer:
// Phase 1: Sphere-trace the SDF to skip empty space and find the surface vicinity.
// Phase 2: Short DDA through a virtual coarse voxel grid near the surface for
//          proper voxel face hits with axis-aligned normals.
//
// This avoids the halo-circle artifacts of a full-length uniform DDA while
// producing a voxelized appearance (flat faces, cube normals) at all distances.

fn trace_far_field(ro: vec3<f32>, rd: vec3<f32>, t_start: f32, t_end: f32,
                   out_mat: ptr<function, u32>) -> vec4<f32> {
    // ro is camera-relative; planet center is at -camera_offset in this space.
    let r_outer = u.planet_radius * 1.03;
    // Offset from planet center to ray origin (world-space direction/magnitude).
    let oc = ro + u.camera_offset;
    let b_outer = dot(oc, rd);
    let c_outer = dot(oc, oc) - r_outer * r_outer;
    let disc_outer = b_outer * b_outer - c_outer;
    if disc_outer < 0.0 { return vec4<f32>(0.0); }

    let sq_outer = sqrt(disc_outer);
    let sphere_t0 = max(-b_outer - sq_outer, 0.0);
    let sphere_t1 = -b_outer + sq_outer;
    if sphere_t1 < 0.0 { return vec4<f32>(0.0); }

    let t_lo = max(t_start, sphere_t0);
    let t_hi = min(t_end, sphere_t1);
    if t_lo >= t_hi { return vec4<f32>(0.0); }

    let cell_size = u.ff_cell_size;

    // Phase 1: Sphere-trace to find vicinity of the surface.
    var t_sdf = t_lo;
    var near_surface = false;
    for (var i = 0; i < 192; i++) {
        // SDF needs world-space position (offset from planet center).
        let pos_world = ro + rd * t_sdf + u.camera_offset;
        let d = ff_sdf_terrain(pos_world);
        if d < cell_size * 3.0 {
            near_surface = true;
            break;
        }
        t_sdf += max(d * 0.7, max(0.5, 0.0005 * t_sdf));
        if t_sdf > t_hi { break; }
    }

    if !near_surface { return vec4<f32>(0.0); }

    // Phase 2: Back up and do a short DDA through a world-aligned virtual grid.
    let dda_start = max(t_sdf - cell_size * 4.0, t_lo);
    // World-space entry for cell index computation (grid is world-aligned).
    let entry_world = ro + rd * dda_start + u.camera_offset;

    var cell = vec3<i32>(floor(entry_world / cell_size));
    let step_dir = vec3<i32>(sign(rd));
    let ff_eps = vec3<f32>(1e-12);
    let safe_rd = select(rd, select(-ff_eps, ff_eps, rd >= vec3<f32>(0.0)),
                         abs(rd) < ff_eps);
    let inv = 1.0 / safe_rd;
    let delta = abs(vec3<f32>(cell_size) * inv);

    // t_max: cell boundaries are world-aligned; ro is camera-relative.
    // t = (cell_boundary_world - ro_world) / rd
    //   = (cell_boundary_world - camera_offset - ro) / rd
    let ro_world = ro + u.camera_offset;
    var t_max_dda = vec3<f32>(
        (f32(select(cell.x, cell.x + 1, step_dir.x > 0)) * cell_size - ro_world.x) * inv.x,
        (f32(select(cell.y, cell.y + 1, step_dir.y > 0)) * cell_size - ro_world.y) * inv.y,
        (f32(select(cell.z, cell.z + 1, step_dir.z > 0)) * cell_size - ro_world.z) * inv.z,
    );

    var normal = vec3<f32>(0.0, 1.0, 0.0);
    var t_cell_enter = dda_start;

    for (var i = 0; i < 128; i++) {
        // Cell center in world space for SDF evaluation.
        let voxel_center = (vec3<f32>(cell) + 0.5) * cell_size;
        // Conservative occupancy: treat a cell as solid if the surface might
        // pass through it (SDF < half-diagonal), not just if center is inside.
        if ff_sdf_terrain(voxel_center) < cell_size * 0.866 {
            *out_mat = ff_material_at_center(voxel_center);
            let n = select(normal, vec3<f32>(0.0, 1.0, 0.0),
                           all(normal == vec3<f32>(0.0)));
            return vec4<f32>(n, t_cell_enter);
        }

        // DDA step to next cell.
        if t_max_dda.x <= t_max_dda.y && t_max_dda.x <= t_max_dda.z {
            normal = vec3<f32>(f32(-step_dir.x), 0.0, 0.0);
            t_cell_enter = t_max_dda.x;
            cell.x += step_dir.x;
            t_max_dda.x += delta.x;
        } else if t_max_dda.y <= t_max_dda.z {
            normal = vec3<f32>(0.0, f32(-step_dir.y), 0.0);
            t_cell_enter = t_max_dda.y;
            cell.y += step_dir.y;
            t_max_dda.y += delta.y;
        } else {
            normal = vec3<f32>(0.0, 0.0, f32(-step_dir.z));
            t_cell_enter = t_max_dda.z;
            cell.z += step_dir.z;
            t_max_dda.z += delta.z;
        }

        if t_cell_enter > t_hi { break; }
    }

    return vec4<f32>(0.0);
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let px = gid.x;
    let py = gid.y;
    if px >= u.width || py >= u.height { return; }

    // Generate ray from camera — direct computation avoids the ill-conditioned
    // inv_view_proj matrix (condition number ~10^7 with near=0.01, far=100000).
    // Apply TAA Halton subpixel jitter so each frame covers a different subpixel,
    // enabling TaaPass to reconstruct high-quality temporal supersampling.
    let uv = (vec2<f32>(f32(px) + 0.5 + u.jitter.x, f32(py) + 0.5 + u.jitter.y)
              / vec2<f32>(f32(u.width), f32(u.height))) * 2.0 - 1.0;
    let ndc = vec2<f32>(uv.x, -uv.y);

    // Camera-relative ray origin (≈ zero for maximum f32 precision).
    let ro = camera.position_near.xyz - u.camera_offset;

    // View-space direction from projection parameters (exact, no matrix inverse).
    let d_view = vec3<f32>(ndc.x / camera.proj[0][0],
                           ndc.y / camera.proj[1][1],
                           -1.0);
    // Inverse view rotation = transpose of view's 3x3 rotation (orthonormal).
    let inv_rot = transpose(mat3x3<f32>(
        camera.view[0].xyz, camera.view[1].xyz, camera.view[2].xyz));
    let rd = normalize(inv_rot * d_view);

    let coord = vec2<i32>(i32(px), i32(py));
    let eps_rd = vec3<f32>(1e-12);
    let safe_rd = select(rd, select(-eps_rd, eps_rd, rd >= vec3<f32>(0.0)),
                         abs(rd) < eps_rd);
    let inv_rd = 1.0 / safe_rd;

    // Compute indirection grid AABB in camera-relative space.
    let chunk_world = f32(u.chunk_dim_bricks * u.brick_dim) * u.voxel_size;
    let ig_dim = i32(u.indir_grid_dim);
    let grid_min = vec3<f32>(f32(u.indir_origin.x), f32(u.indir_origin.y),
                             f32(u.indir_origin.z)) * chunk_world - u.camera_offset;
    let grid_max = grid_min + vec3<f32>(f32(ig_dim)) * chunk_world;

    let hit = ray_aabb(ro, inv_rd, grid_min, grid_max);
    var dda_entered = hit.x <= hit.y && hit.y >= 0.0;

    // Track the farthest t reached by the grid DDA so the full-planet far-field
    // trace doesn't waste iterations re-tracing already-covered regions.
    var t_grid_exit = 0.0;

    // ── Chunk-level DDA (only if ray enters indirection grid) ─────────
    if dda_entered {
        let t_enter = max(hit.x, 0.0) + 0.0001;
        let entry_pos = ro + rd * t_enter;

        let cf = (entry_pos - grid_min) / chunk_world;
        var chunk_cell = vec3<i32>(clamp(vec3<i32>(floor(cf)),
                                         vec3<i32>(0), vec3<i32>(ig_dim - 1)));

        let step_c = vec3<i32>(sign(rd));
        let delta_chunk = abs(vec3<f32>(chunk_world) * inv_rd);

        var t_max_chunk = vec3<f32>(
            (grid_min.x + f32(select(chunk_cell.x, chunk_cell.x + 1, step_c.x > 0)) * chunk_world - ro.x) * inv_rd.x,
            (grid_min.y + f32(select(chunk_cell.y, chunk_cell.y + 1, step_c.y > 0)) * chunk_world - ro.y) * inv_rd.y,
            (grid_min.z + f32(select(chunk_cell.z, chunk_cell.z + 1, step_c.z > 0)) * chunk_world - ro.z) * inv_rd.z,
        );

        var chunk_normal = vec3<f32>(0.0);
        var t_chunk_enter = t_enter;
        let max_chunk_steps = u32(ig_dim) * 3u;

        for (var ci = 0u; ci < max_chunk_steps; ci++) {
            if chunk_cell.x < 0 || chunk_cell.y < 0 || chunk_cell.z < 0 ||
               chunk_cell.x >= ig_dim || chunk_cell.y >= ig_dim || chunk_cell.z >= ig_dim {
                break;
            }

            let chunk_pos = u.indir_origin + chunk_cell;
            let ix = wrap_coord(chunk_pos.x, ig_dim);
            let iy = wrap_coord(chunk_pos.y, ig_dim);
            let iz = wrap_coord(chunk_pos.z, ig_dim);
            let indir_idx = ix + iy * u.indir_grid_dim + iz * u.indir_grid_dim * u.indir_grid_dim;
            let slot = indir_grid[indir_idx];

            let t_chunk_exit = min(min(t_max_chunk.x, t_max_chunk.y), t_max_chunk.z);

            if slot != INDIR_EMPTY {
                let ci_info = chunk_table[slot];
                let chunk_origin = vec3<f32>(f32(ci_info.pos_x), f32(ci_info.pos_y),
                                             f32(ci_info.pos_z)) * chunk_world - u.camera_offset;

                var found_mat: u32 = 0u;
                let result = trace_chunk_bricks(
                    ro, rd, inv_rd,
                    chunk_origin,
                    ci_info.brick_pool_offset,
                    t_chunk_enter, t_chunk_exit,
                    chunk_normal, &found_mat,
                );

                if found_mat != 0u {
                    textureStore(out_material, coord, vec4<u32>(found_mat, 0u, 0u, 0u));
                    textureStore(out_normal, coord, vec4<f32>(result.xyz, result.w));
                    return;
                }
            } else {
                // Unloaded chunk — use far-field DDA scoped to this chunk region.
                var ff_mat: u32 = 1u;
                let ff_chunk = trace_far_field(ro, rd, t_chunk_enter, t_chunk_exit, &ff_mat);
                if ff_chunk.w > 0.0 {
                    textureStore(out_material, coord, vec4<u32>(ff_mat, 0u, 0u, 0u));
                    textureStore(out_normal, coord, vec4<f32>(ff_chunk.xyz, ff_chunk.w));
                    return;
                }
            }

            // Step chunk DDA.
            if t_max_chunk.x <= t_max_chunk.y && t_max_chunk.x <= t_max_chunk.z {
                chunk_normal = vec3<f32>(f32(-step_c.x), 0.0, 0.0);
                t_chunk_enter = t_max_chunk.x;
                chunk_cell.x += step_c.x;
                t_max_chunk.x += delta_chunk.x;
            } else if t_max_chunk.y <= t_max_chunk.z {
                chunk_normal = vec3<f32>(0.0, f32(-step_c.y), 0.0);
                t_chunk_enter = t_max_chunk.y;
                chunk_cell.y += step_c.y;
                t_max_chunk.y += delta_chunk.y;
            } else {
                chunk_normal = vec3<f32>(0.0, 0.0, f32(-step_c.z));
                t_chunk_enter = t_max_chunk.z;
                chunk_cell.z += step_c.z;
                t_max_chunk.z += delta_chunk.z;
            }
            t_grid_exit = t_chunk_enter;
        }
    }

    // No DDA hit — try far-field for the planet beyond the grid region.
    var ff_mat: u32 = 1u;
    let ff_result = trace_far_field(ro, rd, t_grid_exit, 1e20, &ff_mat);
    if ff_result.w > 0.0 {
        textureStore(out_material, coord, vec4<u32>(ff_mat, 0u, 0u, 0u));
        textureStore(out_normal, coord, vec4<f32>(ff_result.xyz, ff_result.w));
        return;
    }

    // No hit — sky.
    textureStore(out_material, coord, vec4<u32>(0u));
    textureStore(out_normal, coord, vec4<f32>(0.0));
}
