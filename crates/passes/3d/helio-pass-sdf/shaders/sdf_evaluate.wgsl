// SDF Evaluate Sparse — compute shader (GPU-native, indirect dispatch)
//
// Dispatched via dispatch_workgroups_indirect for each clip level.
// The indirect x-count is the number of dirty bricks for that level,
// written atomically by the classify pass — zero CPU involvement after
// initial setup.
//
// One workgroup = one dirty brick.  64 threads stride over the padded voxels
// of that brick (9^3 = 729 voxels, covered by 64 threads in ceil(729/64) = 12 iterations).
//
// Bind group 0:
//   b0: uniform   SdfGridParams     — per-level config (voxel_size, brick dims, level_idx)
//   b1: storage r  array<GpuSdfEdit> — global edit list
//   b2: storage rw array<atomic<u32>>— per-level brick atlas
//   b3: storage r  array<u32>        — global dirty_bricks (GPU-built by classify)
//   b4: storage r  array<u32>        — global per_brick_edit_lists (GPU-built)
//   b5: uniform   TerrainParams      — global terrain config

// ── Uniforms ──────────────────────────────────────────────────────────────────

struct SdfGridParams {
    volume_min:            vec3<f32>,
    _pad0:                 f32,
    volume_max:            vec3<f32>,
    _pad1:                 f32,
    grid_dim:              u32,
    edit_count:            u32,
    voxel_size:            f32,
    max_march_dist:        f32,
    brick_size:            u32,
    brick_grid_dim:        u32,
    level_idx:             u32,    // clip-map level index (offsets into global buffers)
    atlas_bricks_per_axis: u32,
    grid_origin:           vec3<f32>,
    debug_flags:           u32,
    bricks_per_level:      u32,    // = brick_grid_dim^3
    _pad2: u32, _pad3: u32, _pad4: u32,
};

struct TerrainParams {
    enabled:     u32,
    style:       u32,
    height:      f32,
    amplitude:   f32,
    frequency:   f32,
    octaves:     u32,
    lacunarity:  f32,
    persistence: f32,
    warp_amount: f32,
    _pad0: u32, _pad1: u32, _pad2: u32,
    _pad3: u32, _pad4: u32, _pad5: u32, _pad6: u32,
};

struct GpuSdfEdit {
    inv_transform_0: vec4<f32>,
    inv_transform_1: vec4<f32>,
    inv_transform_2: vec4<f32>,
    inv_transform_3: vec4<f32>,
    shape_type:   u32,
    boolean_op:   u32,
    blend_radius: f32,
    _pad:         f32,
    params:       vec4<f32>,
};

// ── Constants ─────────────────────────────────────────────────────────────────

// Must match sdf_classify.wgsl exactly.
const EDIT_LIST_STRIDE:     u32 = 65u;   // 1 (count) + 64 (indices)
const MAX_BRICKS_PER_LEVEL: u32 = 4096u; // dirty_bricks stride per level

// ── Bindings ──────────────────────────────────────────────────────────────────

@group(0) @binding(0) var<uniform>            params:               SdfGridParams;
@group(0) @binding(1) var<storage, read>      edits:                array<GpuSdfEdit>;
@group(0) @binding(2) var<storage, read_write> atlas:               array<atomic<u32>>;
@group(0) @binding(3) var<storage, read>      dirty_bricks:         array<u32>;
@group(0) @binding(4) var<storage, read>      per_brick_edit_lists: array<u32>;
@group(0) @binding(5) var<uniform>            terrain:              TerrainParams;

// ── Noise (FBM — must match CPU noise.rs exactly) ─────────────────────────────

fn hash3(px: f32, py: f32, pz: f32) -> f32 {
    var qx = fract(px * 0.3183099 + 0.1);
    var qy = fract(py * 0.3183099 + 0.1);
    var qz = fract(pz * 0.3183099 + 0.1);
    qx = abs(qx); qy = abs(qy); qz = abs(qz);
    qx *= 17.0; qy *= 17.0; qz *= 17.0;
    return fract(abs(qx * qy * qz * (qx + qy + qz)));
}

fn noise3(p: vec3<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);
    let a  = hash3(i.x,       i.y,       i.z      );
    let b  = hash3(i.x + 1.0, i.y,       i.z      );
    let c  = hash3(i.x,       i.y + 1.0, i.z      );
    let d  = hash3(i.x + 1.0, i.y + 1.0, i.z      );
    let e  = hash3(i.x,       i.y,       i.z + 1.0);
    let ff = hash3(i.x + 1.0, i.y,       i.z + 1.0);
    let g  = hash3(i.x,       i.y + 1.0, i.z + 1.0);
    let h  = hash3(i.x + 1.0, i.y + 1.0, i.z + 1.0);
    return mix(mix(mix(a, b, u.x), mix(c, d, u.x), u.y),
               mix(mix(e, ff, u.x), mix(g, h, u.x), u.y), u.z) * 2.0 - 1.0;
}

fn fbm_rotate(p: vec3<f32>, lac: f32) -> vec3<f32> {
    return vec3<f32>(
        lac * ( 0.00 * p.x + 0.80 * p.y + 0.60 * p.z),
        lac * (-0.80 * p.x + 0.36 * p.y - 0.48 * p.z),
        lac * (-0.60 * p.x - 0.48 * p.y + 0.64 * p.z),
    );
}

fn fbm2(p_in: vec3<f32>, octaves: u32, lac: f32, gain: f32) -> f32 {
    var p = p_in; var a = 1.0; var total = 0.0; var weight = 0.0;
    for (var i = 0u; i < octaves; i++) {
        total += a * noise3(p); weight += a; a *= gain; p = fbm_rotate(p, lac);
    }
    return total / weight;
}

fn fbm2_at(x: f32, z: f32) -> f32 {
    return fbm2(vec3<f32>(x, 0.0, z), terrain.octaves, terrain.lacunarity, terrain.persistence);
}

fn warped_fbm2(x: f32, z: f32) -> f32 {
    let wa = terrain.warp_amount;
    let w1x = fbm2_at(x, z); let w1z = fbm2_at(x + 5.2, z + 1.3);
    let p1x = x + wa * w1x;  let p1z = z + wa * w1z;
    let w2x = fbm2_at(p1x, p1z); let w2z = fbm2_at(p1x + 1.7, p1z + 9.2);
    return fbm2_at(x + wa * w2x, z + wa * w2z);
}

fn terrain_sdf(pos: vec3<f32>) -> f32 {
    if terrain.enabled == 0u { return 1e10; }
    let fx = pos.x * terrain.frequency;
    let fz = pos.z * terrain.frequency;
    var th: f32;
    if terrain.style == 0u {
        th = fbm2(vec3<f32>(fx, 0.0, fz), terrain.octaves, terrain.lacunarity, terrain.persistence) * terrain.amplitude;
    } else if terrain.style == 1u {
        th = fbm2(vec3<f32>(fx * 1.5, 0.0, fz * 1.5), terrain.octaves, terrain.lacunarity, terrain.persistence) * terrain.amplitude * 1.3;
    } else if terrain.style == 2u {
        let n = fbm2(vec3<f32>(fx, 0.0, fz), terrain.octaves, terrain.lacunarity, terrain.persistence);
        th = n * terrain.amplitude + fbm2(vec3<f32>(fx * 3.0, 0.0, fz * 3.0), 3u, terrain.lacunarity, 0.4) * 3.0;
    } else if terrain.style == 3u {
        th = fbm2(vec3<f32>(fx * 3.0, 0.0, fz), terrain.octaves, terrain.lacunarity, terrain.persistence) * terrain.amplitude;
    } else if terrain.style == 4u {
        th = terrain.amplitude * warped_fbm2(fx, fz);
    } else {
        th = fbm2(vec3<f32>(fx, 0.0, fz), terrain.octaves, terrain.lacunarity, terrain.persistence) * terrain.amplitude;
    }
    return pos.y - (terrain.height + th);
}

// ── SDF primitives ────────────────────────────────────────────────────────────

fn sdf_sphere(p: vec3<f32>, r: f32) -> f32 { return length(p) - r; }

fn sdf_cube(p: vec3<f32>, half_ext: vec3<f32>) -> f32 {
    let d = abs(p) - half_ext;
    return length(max(d, vec3<f32>(0.0))) + min(max(d.x, max(d.y, d.z)), 0.0);
}

fn sdf_capsule(p: vec3<f32>, r: f32, half_h: f32) -> f32 {
    var q = p; q.y -= clamp(q.y, -half_h, half_h); return length(q) - r;
}

fn sdf_torus(p: vec3<f32>, major_r: f32, minor_r: f32) -> f32 {
    return length(vec2<f32>(length(p.xz) - major_r, p.y)) - minor_r;
}

fn sdf_cylinder(p: vec3<f32>, r: f32, half_h: f32) -> f32 {
    let d = abs(vec2<f32>(length(p.xz), p.y)) - vec2<f32>(r, half_h);
    return min(max(d.x, d.y), 0.0) + length(max(d, vec2<f32>(0.0)));
}

fn evaluate_shape(p: vec3<f32>, edit: GpuSdfEdit) -> f32 {
    if      edit.shape_type == 0u { return sdf_sphere(p, edit.params.x); }
    else if edit.shape_type == 1u { return sdf_cube(p, edit.params.xyz); }
    else if edit.shape_type == 2u { return sdf_capsule(p, edit.params.x, edit.params.y); }
    else if edit.shape_type == 3u { return sdf_torus(p, edit.params.x, edit.params.y); }
    else                          { return sdf_cylinder(p, edit.params.x, edit.params.y); }
}

fn apply_boolean(d1: f32, d2: f32, op: u32, k: f32) -> f32 {
    let use_blend = k > 0.001;
    if op == 0u {
        if use_blend { let h = clamp(0.5 + 0.5*(d2-d1)/k, 0.0, 1.0); return mix(d2,d1,h) - k*h*(1.0-h); }
        return min(d1, d2);
    } else if op == 1u {
        if use_blend { let h = clamp(0.5 - 0.5*(d2+d1)/k, 0.0, 1.0); return mix(d1,-d2,h) + k*h*(1.0-h); }
        return max(d1, -d2);
    } else {
        if use_blend { let h = clamp(0.5 - 0.5*(d2-d1)/k, 0.0, 1.0); return mix(d2,d1,h) + k*h*(1.0-h); }
        return max(d1, d2);
    }
}

fn transform_point(edit: GpuSdfEdit, p: vec3<f32>) -> vec3<f32> {
    let m = mat4x4<f32>(edit.inv_transform_0, edit.inv_transform_1,
                        edit.inv_transform_2, edit.inv_transform_3);
    return (m * vec4<f32>(p, 1.0)).xyz;
}

// ── Brick coordinate helpers ───────────────────────────────────────────────────

fn unpack_world_brick_coord(packed: u32) -> vec3<i32> {
    return vec3<i32>(
        i32(packed          & 0x3FFu) - 512,
        i32((packed >> 10u) & 0x3FFu) - 512,
        i32((packed >> 20u) & 0x3FFu) - 512,
    );
}

fn world_to_grid_flat(wc: vec3<i32>, bgd: i32) -> u32 {
    let gx = ((wc.x % bgd) + bgd) % bgd;
    let gy = ((wc.y % bgd) + bgd) % bgd;
    let gz = ((wc.z % bgd) + bgd) % bgd;
    return u32(gz * bgd * bgd + gy * bgd + gx);
}

fn atlas_index(atlas_id: u32, local: vec3<u32>) -> u32 {
    // Direct atlas mapping: atlas_id == brick_flat_index.
    // Position in the atlas 3D grid via linear decomposition.
    let aba = params.atlas_bricks_per_axis;
    let ps  = params.brick_size + 1u;   // padded brick size
    let bx  = atlas_id % aba;
    let by  = (atlas_id / aba) % aba;
    let bz  = atlas_id / (aba * aba);
    let gx  = bx * ps + local.x;
    let gy  = by * ps + local.y;
    let gz  = bz * ps + local.z;
    let dim = aba * ps;
    return gz * dim * dim + gy * dim + gx;
}

fn quantize_distance(d: f32, vs: f32) -> u32 {
    let range = 4.0 * vs;
    return u32(clamp((d + range) / (2.0 * range), 0.0, 1.0) * 255.0);
}

// ── Main compute kernel ────────────────────────────────────────────────────────
//
// workgroup_id.x  = dirty brick index within this level's dirty list
// 64 threads      = 8×8 local group that strides over 9^3 = 729 padded voxels

@compute @workgroup_size(8, 8, 1)
fn cs_evaluate_sparse(
    @builtin(workgroup_id)        workgroup_id: vec3<u32>,
    @builtin(local_invocation_id) lid:          vec3<u32>,
) {
    let dirty_idx = workgroup_id.x;

    // ── Look up which brick we are from the GPU-built dirty list ─────────
    //   dirty_bricks layout: [level_idx * MAX_BRICKS_PER_LEVEL + dirty_idx]
    let packed      = dirty_bricks[params.level_idx * MAX_BRICKS_PER_LEVEL + dirty_idx];
    let world_coord = unpack_world_brick_coord(packed);

    // Convert packed world brick coords → toroidal grid flat index.
    let bgd  = i32(params.brick_grid_dim);
    let flat = world_to_grid_flat(world_coord, bgd);

    // Direct atlas mapping: atlas_id == flat (no free-list allocation needed).
    let atlas_id = flat;

    // ── Retrieve GPU-built per-brick edit list ────────────────────────────
    //   per_brick_edit_lists layout:
    //     (level_idx * bricks_per_level + flat) * EDIT_LIST_STRIDE
    //       [+ 0]     = edit_count
    //       [+ 1 + e] = edit_index[e]
    let el_base          = (params.level_idx * params.bricks_per_level + flat) * EDIT_LIST_STRIDE;
    let edit_count_local = per_brick_edit_lists[el_base];

    let ps           = params.brick_size + 1u;  // padded brick size (9 for brick_size=8)
    let thread_id    = lid.y * 8u + lid.x;
    let total_voxels = ps * ps * ps;             // 9^3 = 729

    for (var i = thread_id; i < total_voxels; i += 64u) {
        let voxel = vec3<u32>(i % ps, (i / ps) % ps, i / (ps * ps));

        // Reconstruct world-space position from world brick coordinate.
        let vs             = params.voxel_size;
        let bs             = f32(params.brick_size);
        let brick_world_min = vec3<f32>(vec3<i32>(world_coord)) * bs * vs;
        let world_pos      = brick_world_min + vec3<f32>(voxel) * vs;

        // Terrain SDF.
        var dist = terrain_sdf(world_pos);

        // Apply each edit in the GPU-built per-brick edit list.
        for (var e = 0u; e < edit_count_local; e++) {
            let edit_idx = per_brick_edit_lists[el_base + 1u + e];
            if edit_idx >= params.edit_count { break; }
            let edit      = edits[edit_idx];
            let local_pos = transform_point(edit, world_pos);
            let d         = evaluate_shape(local_pos, edit);
            dist          = apply_boolean(dist, d, edit.boolean_op, edit.blend_radius);
        }

        // Quantize and write to the per-level atlas (atomic byte write).
        let q           = quantize_distance(dist, vs);
        let ai          = atlas_index(atlas_id, voxel);
        let word_idx    = ai / 4u;
        let byte_offset = (ai % 4u) * 8u;
        let mask        = 0xFFu << byte_offset;
        atomicAnd(&atlas[word_idx], ~mask);
        atomicOr( &atlas[word_idx],  q << byte_offset);
    }
}
