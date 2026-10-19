// Composes each object's world transform from its parent chain (see
// `hierarchy_resolve.rs`). Prepended with scene_join_common.wgsl, which
// supplies `ObjectTransform` and `object_rotation`.

const MAX_DEPTH: u32 = 64u;

struct ResolveUniforms {
    rows: u32,
    // 1.0, unknown to the compiler: `pair_add`'s guard.
    one: f32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<uniform> u: ResolveUniforms;
/// Parent-relative transforms: the frontend's `Transform` column.
@group(0) @binding(1) var<storage, read> local_transforms: array<ObjectTransform>;
/// Per object: the parent's entity index plus one, zero for none.
@group(0) @binding(2) var<storage, read> parents: array<u32>;
@group(0) @binding(3) var<storage, read_write> world_transforms: array<ObjectTransform>;

fn parent_of(index: u32) -> u32 {
    if index >= arrayLength(&parents) {
        return 0u;
    }
    return parents[index];
}

/// The linear part of a transform: rotation * scale.
fn linear_part(t: ObjectTransform) -> mat3x3<f32> {
    let r = object_rotation(t);
    let s = object_scale(t);
    return mat3x3<f32>(r[0] * s.x, r[1] * s.y, r[2] * s.z);
}

/// `a + b` for f32 pairs (`hi + lo` each), renormalised: the sum's f32 and
/// what that drops (two-sum), so world positions keep their precision.
/// Each rounding is kept through `u.one`: a driver reassociating floats
/// folds `(s + lo) - s` to `lo` and the dropped part to zero (NVIDIA's
/// Vulkan driver did, half a metre on a planet).
fn pair_add(a_hi: vec3<f32>, a_lo: vec3<f32>, b_hi: vec3<f32>, b_lo: vec3<f32>) -> array<vec3<f32>, 2> {
    let s = (a_hi + b_hi) * u.one;
    let b_virtual = (s - a_hi) * u.one;
    let error = (a_hi - (s - b_virtual)) + (b_hi - b_virtual);
    let lo = error + a_lo + b_lo;
    let hi = (s + lo) * u.one;
    return array<vec3<f32>, 2>(hi, lo - (hi - s));
}

/// The nearest shear-free `ObjectTransform` to `lin` placed at `hi + lo`,
/// with `rotation` as YXZ Euler angles in degrees (the inverse of
/// `object_rotation`).
fn decompose(lin: mat3x3<f32>, hi: vec3<f32>, lo: vec3<f32>) -> ObjectTransform {
    var sx = length(lin[0]);
    let sy = length(lin[1]);
    let sz = length(lin[2]);
    // A reflection shows as a negative determinant: carry it on x.
    if determinant(lin) < 0.0 {
        sx = -sx;
    }
    var r = mat3x3<f32>(vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(0.0, 0.0, 1.0));
    if sx != 0.0 && sy != 0.0 && sz != 0.0 {
        r = mat3x3<f32>(lin[0] / sx, lin[1] / sy, lin[2] / sz);
    }
    // r = Ry(y) * Rx(x) * Rz(z), indexed r[column][row].
    let to_deg = 57.29577951308232;
    let pitch = asin(clamp(-r[2][1], -1.0, 1.0));
    var yaw = 0.0;
    var roll = 0.0;
    if abs(r[2][1]) < 0.9999999 {
        yaw = atan2(r[2][0], r[2][2]);
        roll = atan2(r[0][1], r[1][1]);
    } else {
        // Looking straight up or down: yaw and roll are the same turn.
        yaw = atan2(-r[0][2], r[0][0]);
    }
    var out: ObjectTransform;
    out.position_hi = array<f32, 3>(hi.x, hi.y, hi.z);
    out.position_lo = array<f32, 3>(lo.x, lo.y, lo.z);
    out.rotation = array<f32, 3>(pitch * to_deg, yaw * to_deg, roll * to_deg);
    out.scale = array<f32, 3>(sx, sy, sz);
    return out;
}

@compute @workgroup_size(64)
fn cs_resolve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let index = gid.x;
    if index >= u.rows {
        return;
    }
    let own = local_transforms[index];
    var parent = parent_of(index);
    if parent == 0u {
        world_transforms[index] = own;
        return;
    }
    var lin = linear_part(own);
    // The position so far, as an f32 pair: a parent on a planet is far from
    // the world's origin, its children near it.
    var hi = position_hi(own);
    var lo = position_lo(own);
    for (var depth = 0u; depth < MAX_DEPTH; depth = depth + 1u) {
        if parent == 0u {
            break;
        }
        let at = parent - 1u;
        if at >= u.rows {
            break;
        }
        let up = local_transforms[at];
        let up_lin = linear_part(up);
        let sum = pair_add(up_lin * hi, up_lin * lo, position_hi(up), position_lo(up));
        hi = sum[0];
        lo = sum[1];
        lin = up_lin * lin;
        parent = parent_of(at);
    }
    world_transforms[index] = decompose(lin, hi, lo);
}
