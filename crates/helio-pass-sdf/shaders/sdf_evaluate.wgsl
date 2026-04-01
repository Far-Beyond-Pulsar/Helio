// SDF Evaluate Sparse — compute shader
//
// One workgroup per dirty brick. Evaluates the SDF (terrain + edits) at each
// voxel inside the brick and writes a u8-quantized distance to the atlas storage.
//
// Bind group 0:
//   b0: uniform  SdfGridParams
//   b1: storage  edit list (array<GpuSdfEdit>)
//   b2: storage  atlas (read_write, u32-packed u8 distances)
//   b3: storage  active_bricks (read-only, per-workgroup brick coords)
//   b4: storage  brick_index (read-only, 3D → atlas mapping)
//   b5: uniform  TerrainParams
//   b6: storage  edit_list_offsets (per-brick offset+count into edit_list_data)
//   b7: storage  edit_list_data (packed edit indices)

// ── Uniforms ────────────────────────────────────────────────────────────────

struct SdfGridParams {
    volume_min: vec3<f32>,
    _pad0: f32,
    volume_max: vec3<f32>,
    _pad1: f32,
    grid_dim: u32,
    edit_count: u32,
    voxel_size: f32,
    max_march_dist: f32,
    brick_size: u32,
    brick_grid_dim: u32,
    active_bricks_count: u32,
    atlas_bricks_per_axis: u32,
    grid_origin: vec3<f32>,
    debug_flags: u32,
};

struct TerrainParams {
    enabled: u32,
    style: u32,
    height: f32,
    amplitude: f32,
    // offset 16
    frequency: f32,
    octaves: u32,
    lacunarity: f32,
    persistence: f32,
    // offset 32
    warp_amount: f32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
    // offset 48
    _pad3: u32,
    _pad4: u32,
    _pad5: u32,
    _pad6: u32,
    // offset 64 - WGSL uniform buffer requires minimum 64 bytes
};

struct GpuSdfEdit {
    inv_transform_0: vec4<f32>,
    inv_transform_1: vec4<f32>,
    inv_transform_2: vec4<f32>,
    inv_transform_3: vec4<f32>,
    shape_type: u32,
    boolean_op: u32,
    blend_radius: f32,
    _pad: f32,
    params: vec4<f32>,
};

@group(0) @binding(0) var<uniform> params: SdfGridParams;
@group(0) @binding(1) var<storage, read> edits: array<GpuSdfEdit>;
@group(0) @binding(2) var<storage, read_write> atlas: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read> active_bricks: array<u32>; // packed xyz
@group(0) @binding(4) var<storage, read> brick_index: array<u32>;
@group(0) @binding(5) var<uniform> terrain: TerrainParams;
@group(0) @binding(6) var<storage, read> edit_list_offsets: array<u32>; // [offset, count] pairs
@group(0) @binding(7) var<storage, read> edit_list_data: array<u32>;

// ── Noise (matching CPU noise.rs exactly) ───────────────────────────────────

fn hash3(px: f32, py: f32, pz: f32) -> f32 {
    var qx = fract(px * 0.3183099 + 0.1);
    var qy = fract(py * 0.3183099 + 0.1);
    var qz = fract(pz * 0.3183099 + 0.1);
    qx = abs(qx);
    qy = abs(qy);
    qz = abs(qz);
    qx *= 17.0;
    qy *= 17.0;
    qz *= 17.0;
    let v = qx * qy * qz * (qx + qy + qz);
    return fract(abs(v));
}

fn noise3(p: vec3<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);
    // Quintic interpolation (matching CPU)
    let u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);

    let a = hash3(i.x,       i.y,       i.z      );
    let b = hash3(i.x + 1.0, i.y,       i.z      );
    let c = hash3(i.x,       i.y + 1.0, i.z      );
    let d = hash3(i.x + 1.0, i.y + 1.0, i.z      );
    let e = hash3(i.x,       i.y,       i.z + 1.0);
    let ff = hash3(i.x + 1.0, i.y,       i.z + 1.0);
    let g = hash3(i.x,       i.y + 1.0, i.z + 1.0);
    let h = hash3(i.x + 1.0, i.y + 1.0, i.z + 1.0);

    let nx00 = mix(a, b, u.x);
    let nx10 = mix(c, d, u.x);
    let nx01 = mix(e, ff, u.x);
    let nx11 = mix(g, h, u.x);
    let nxy0 = mix(nx00, nx10, u.y);
    let nxy1 = mix(nx01, nx11, u.y);
    let val = mix(nxy0, nxy1, u.z);
    return val * 2.0 - 1.0; // range [-1, 1] matching CPU
}

// FBM rotation basis - must match CPU noise.rs fbm_rotate() exactly.
// CPU uses row-wise equations:
//   rx = lac * (0.00*px + 0.80*py + 0.60*pz)
//   ry = lac * (-0.80*px + 0.36*py - 0.48*pz)
//   rz = lac * (-0.60*px - 0.48*py + 0.64*pz)
// We use explicit scalar math (not matrix multiply) to ensure exact parity.
fn fbm_rotate(p: vec3<f32>, lac: f32) -> vec3<f32> {
    let rx = lac * (0.00 * p.x + 0.80 * p.y + 0.60 * p.z);
    let ry = lac * (-0.80 * p.x + 0.36 * p.y - 0.48 * p.z);
    let rz = lac * (-0.60 * p.x - 0.48 * p.y + 0.64 * p.z);
    return vec3<f32>(rx, ry, rz);
}

fn fbm2(p_in: vec3<f32>, octaves: u32, lac: f32, gain: f32) -> f32 {
    var p = p_in;
    var a = 1.0;
    var total = 0.0;
    var weight = 0.0;
    for (var i = 0u; i < octaves; i++) {
        let n = noise3(p);
        total += a * n;
        weight += a;
        a *= gain;
        p = fbm_rotate(p, lac);
    }
    return total / weight;
}

// ── Domain warping (Inigo Quilez two-layer warp) ─────────────────────────────

/// FBM sampled at a 2D point (x, z) in the y=0 plane — used for domain warp.
fn fbm2_at(x: f32, z: f32) -> f32 {
    let p = vec3<f32>(x, 0.0, z);
    return fbm2(p, terrain.octaves, terrain.lacunarity, terrain.persistence);
}

/// Two-layer domain-warped FBM matching the CPU warped_fbm3() in noise.rs.
///
/// warp1 = vec2(fbm(p), fbm(p + (5.2, 1.3)))
/// warp2 = vec2(fbm(p + warp * warp1), fbm(p + warp * warp1 + (1.7, 9.2)))
/// result = fbm(p + warp * warp2)
fn warped_fbm2(x: f32, z: f32) -> f32 {
    let wa = terrain.warp_amount;

    // First warp layer
    let w1x = fbm2_at(x,       z      );
    let w1z = fbm2_at(x + 5.2, z + 1.3);

    let p1x = x + wa * w1x;
    let p1z = z + wa * w1z;

    // Second warp layer
    let w2x = fbm2_at(p1x,       p1z      );
    let w2z = fbm2_at(p1x + 1.7, p1z + 9.2);

    let p2x = x + wa * w2x;
    let p2z = z + wa * w2z;

    // Final sample
    return fbm2_at(p2x, p2z);
}

fn terrain_sdf(pos: vec3<f32>) -> f32 {
    if terrain.enabled == 0u {
        return 1e10;
    }
    let fx = pos.x * terrain.frequency;
    let fz = pos.z * terrain.frequency;
    var terrain_h: f32;

    // Style-specific terrain evaluation - MUST match CPU noise.rs terrain_sdf_styled()
    if terrain.style == 0u {
        // Rolling: Gentle hills with balanced FBM
        let n = fbm2(vec3<f32>(fx, 0.0, fz), terrain.octaves, terrain.lacunarity, terrain.persistence);
        terrain_h = n * terrain.amplitude;
    } else if terrain.style == 1u {
        // Mountains: Taller, sharper mountains with tighter ridges (1.5x freq, 1.3x amp)
        let n = fbm2(vec3<f32>(fx * 1.5, 0.0, fz * 1.5), terrain.octaves, terrain.lacunarity, terrain.persistence);
        terrain_h = n * terrain.amplitude * 1.3;
    } else if terrain.style == 2u {
        // Canyons: Eroded canyon shapes with extra carved detail
        let n = fbm2(vec3<f32>(fx, 0.0, fz), terrain.octaves, terrain.lacunarity, terrain.persistence);
        let detail = fbm2(vec3<f32>(fx * 3.0, 0.0, fz * 3.0), 3u, terrain.lacunarity, 0.4);
        terrain_h = n * terrain.amplitude + detail * 3.0;
    } else if terrain.style == 3u {
        // Dunes: Wind-swept dunes with elongated directional structure (3x X stretch)
        let stretch = 3.0;
        let n = fbm2(vec3<f32>(fx * stretch, 0.0, fz), terrain.octaves, terrain.lacunarity, terrain.persistence);
        terrain_h = n * terrain.amplitude;
    } else if terrain.style == 4u {
        // Warped: Domain-warped organic terrain (IQ two-layer warp)
        terrain_h = terrain.amplitude * warped_fbm2(fx, fz);
    } else {
        // Fallback to standard FBM for unknown styles
        let n = fbm2(vec3<f32>(fx, 0.0, fz), terrain.octaves, terrain.lacunarity, terrain.persistence);
        terrain_h = n * terrain.amplitude;
    }

    let h = terrain.height + terrain_h;
    return pos.y - h;
}

// ── SDF primitives ──────────────────────────────────────────────────────────

fn sdf_sphere(p: vec3<f32>, r: f32) -> f32 {
    return length(p) - r;
}

fn sdf_cube(p: vec3<f32>, half_ext: vec3<f32>) -> f32 {
    let d = abs(p) - half_ext;
    return length(max(d, vec3<f32>(0.0))) + min(max(d.x, max(d.y, d.z)), 0.0);
}

fn sdf_capsule(p: vec3<f32>, r: f32, half_h: f32) -> f32 {
    var q = p;
    q.y -= clamp(q.y, -half_h, half_h);
    return length(q) - r;
}

fn sdf_torus(p: vec3<f32>, major_r: f32, minor_r: f32) -> f32 {
    let q = vec2<f32>(length(p.xz) - major_r, p.y);
    return length(q) - minor_r;
}

fn sdf_cylinder(p: vec3<f32>, r: f32, half_h: f32) -> f32 {
    let d = abs(vec2<f32>(length(p.xz), p.y)) - vec2<f32>(r, half_h);
    return min(max(d.x, d.y), 0.0) + length(max(d, vec2<f32>(0.0)));
}

fn evaluate_shape(p: vec3<f32>, edit: GpuSdfEdit) -> f32 {
    let shape = edit.shape_type;
    if shape == 0u { // Sphere
        return sdf_sphere(p, edit.params.x);
    } else if shape == 1u { // Cube
        return sdf_cube(p, edit.params.xyz);
    } else if shape == 2u { // Capsule
        return sdf_capsule(p, edit.params.x, edit.params.y);
    } else if shape == 3u { // Torus
        return sdf_torus(p, edit.params.x, edit.params.y);
    } else { // Cylinder
        return sdf_cylinder(p, edit.params.x, edit.params.y);
    }
}

// ── Boolean operations (with smooth blending) ───────────────────────────────

fn apply_boolean(d1: f32, d2: f32, op: u32, k: f32) -> f32 {
    let use_blend = k > 0.001;
    if op == 0u { // Union
        if use_blend {
            let h = clamp(0.5 + 0.5 * (d2 - d1) / k, 0.0, 1.0);
            return mix(d2, d1, h) - k * h * (1.0 - h);
        }
        return min(d1, d2);
    } else if op == 1u { // Subtraction
        if use_blend {
            let h = clamp(0.5 - 0.5 * (d2 + d1) / k, 0.0, 1.0);
            return mix(d1, -d2, h) + k * h * (1.0 - h);
        }
        return max(d1, -d2);
    } else { // Intersection
        if use_blend {
            let h = clamp(0.5 - 0.5 * (d2 - d1) / k, 0.0, 1.0);
            return mix(d2, d1, h) + k * h * (1.0 - h);
        }
        return max(d1, d2);
    }
}

fn transform_point(edit: GpuSdfEdit, p: vec3<f32>) -> vec3<f32> {
    // Columns stored as 4 vec4<f32> in column-major order (matching glam to_cols_array)
    let m = mat4x4<f32>(
        edit.inv_transform_0,
        edit.inv_transform_1,
        edit.inv_transform_2,
        edit.inv_transform_3
    );
    return (m * vec4<f32>(p, 1.0)).xyz;
}

// ── Brick coordinate utilities ──────────────────────────────────────────────

// active_bricks stores packed WORLD brick coordinates (10 bits each, bias 512).
fn unpack_world_brick_coord(packed: u32) -> vec3<i32> {
    let x = i32(packed & 0x3FFu) - 512;
    let y = i32((packed >> 10u) & 0x3FFu) - 512;
    let z = i32((packed >> 20u) & 0x3FFu) - 512;
    return vec3<i32>(x, y, z);
}

fn world_to_grid_flat(wc: vec3<i32>, bgd: i32) -> u32 {
    let gx = ((wc.x % bgd) + bgd) % bgd;
    let gy = ((wc.y % bgd) + bgd) % bgd;
    let gz = ((wc.z % bgd) + bgd) % bgd;
    return u32(gz * bgd * bgd + gy * bgd + gx);
}

fn atlas_index(brick_atlas_id: u32, local: vec3<u32>) -> u32 {
    let aba = params.atlas_bricks_per_axis;
    let ps = params.brick_size + 1u; // padded brick size
    let bx = brick_atlas_id % aba;
    let by = (brick_atlas_id / aba) % aba;
    let bz = brick_atlas_id / (aba * aba);
    let gx = bx * ps + local.x;
    let gy = by * ps + local.y;
    let gz = bz * ps + local.z;
    let dim = aba * ps;
    return gz * dim * dim + gy * dim + gx;
}

fn quantize_distance(d: f32, vs: f32) -> u32 {
    // Map from [-4*vs, 4*vs] to [0, 255]
    let range = 4.0 * vs;
    let t = clamp((d + range) / (2.0 * range), 0.0, 1.0);
    return u32(t * 255.0);
}

// ── Main compute kernel ─────────────────────────────────────────────────────
// Workgroup = 1 dirty brick.  Threads within workgroup iterate over padded voxels.

@compute @workgroup_size(8, 8, 1)
fn cs_evaluate_sparse(
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let brick_idx = workgroup_id.x;
    if brick_idx >= params.active_bricks_count {
        return;
    }

    let packed = active_bricks[brick_idx];
    let world_coord = unpack_world_brick_coord(packed);
    let bgd = i32(params.brick_grid_dim);
    let flat = world_to_grid_flat(world_coord, bgd);
    let atlas_id = brick_index[flat];
    if atlas_id == 0xFFFFFFFFu {
        return;
    }

    let ps = params.brick_size + 1u; // padded brick size (9)
    // 8x8 = 64 threads must cover ps^3 = 729 voxels — use flat index striding
    let thread_id = lid.y * 8u + lid.x;
    let total_voxels = ps * ps * ps;

    for (var i = thread_id; i < total_voxels; i += 64u) {
        let voxel = vec3<u32>(i % ps, (i / ps) % ps, i / (ps * ps));

        // World position from world brick coordinate (NOT grid coordinate)
        let vs = params.voxel_size;
        let bs = f32(params.brick_size);
        let brick_world_min = vec3<f32>(f32(world_coord.x), f32(world_coord.y), f32(world_coord.z)) * bs * vs;
        let world_pos = brick_world_min + vec3<f32>(f32(voxel.x), f32(voxel.y), f32(voxel.z)) * vs;

        // Evaluate terrain
        var dist = terrain_sdf(world_pos);

        // Apply per-brick edit list (indexed by dirty brick workgroup index)
        let data_offset = edit_list_offsets[brick_idx];
        let edit_count_local = edit_list_data[data_offset];
        for (var e = 0u; e < edit_count_local; e++) {
            let edit_idx = edit_list_data[data_offset + 1u + e];
            if edit_idx >= params.edit_count {
                break;
            }
            let edit = edits[edit_idx];
            let local_pos = transform_point(edit, world_pos);
            let d = evaluate_shape(local_pos, edit);
            dist = apply_boolean(dist, d, edit.boolean_op, edit.blend_radius);
        }

        // Quantize and store
        let q = quantize_distance(dist, params.voxel_size);
        let ai = atlas_index(atlas_id, voxel);
        let word_idx = ai / 4u;
        let byte_offset = (ai % 4u) * 8u;
        // Atomic byte write: clear then OR
        let mask = 0xFFu << byte_offset;
        atomicAnd(&atlas[word_idx], ~mask);
        atomicOr(&atlas[word_idx], q << byte_offset);
    }
}
