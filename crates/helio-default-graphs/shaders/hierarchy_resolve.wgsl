// Composes each object's world transform from its parent chain (see
// `hierarchy_resolve.rs`). Prepended with scene_join_common.wgsl, which
// supplies `ObjectTransform` and `object_rotation`.

const MAX_DEPTH: u32 = 64u;

struct ResolveUniforms {
    rows: u32,
    _pad0: u32,
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

/// The nearest shear-free `ObjectTransform` to `lin` placed at `offset`, with
/// `rotation` as YXZ Euler angles in degrees (the inverse of
/// `object_rotation`).
fn decompose(lin: mat3x3<f32>, offset: vec3<f32>) -> ObjectTransform {
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
    out.position = array<f32, 3>(offset.x, offset.y, offset.z);
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
    var offset = object_position(own);
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
        offset = up_lin * offset + object_position(up);
        lin = up_lin * lin;
        parent = parent_of(at);
    }
    world_transforms[index] = decompose(lin, offset);
}
