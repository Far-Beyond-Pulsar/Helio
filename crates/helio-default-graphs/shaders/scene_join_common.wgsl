// Shared by scene_join_meshes.wgsl and scene_join_lights.wgsl: the
// frontend's object rows and the owner join (see `scene_join.rs`).

const NO_OWNER: u32 = 0xFFFFFFFFu;

/// A component instance's owner link (`ComponentOwner`, 12 bytes), keyed by
/// the instance entity. A zero row (no owner, detached, removed) has
/// `enabled == 0`.
struct Owner {
    owner_index: u32,
    owner_generation: u32,
    enabled: u32,
}

/// An object's world transform (`Transform`, 36 bytes), keyed by the
/// object entity. `rotation` is YXZ Euler angles in degrees.
struct ObjectTransform {
    position: array<f32, 3>,
    rotation: array<f32, 3>,
    scale: array<f32, 3>,
}

/// `Ry(yaw) * Rx(pitch) * Rz(roll)` with yaw = rotation[1], pitch =
/// rotation[0], roll = rotation[2], in degrees: the rotation glam's
/// `Quat::from_euler(EulerRot::YXZ, yaw, pitch, roll)` describes.
fn object_rotation(t: ObjectTransform) -> mat3x3<f32> {
    let to_rad = 0.017453292519943295;
    let yaw = t.rotation[1] * to_rad;
    let pitch = t.rotation[0] * to_rad;
    let roll = t.rotation[2] * to_rad;
    let cy = cos(yaw);
    let sy = sin(yaw);
    let cx = cos(pitch);
    let sx = sin(pitch);
    let cz = cos(roll);
    let sz = sin(roll);
    let ry = mat3x3<f32>(vec3<f32>(cy, 0.0, -sy), vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(sy, 0.0, cy));
    let rx = mat3x3<f32>(vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(0.0, cx, sx), vec3<f32>(0.0, -sx, cx));
    let rz = mat3x3<f32>(vec3<f32>(cz, sz, 0.0), vec3<f32>(-sz, cz, 0.0), vec3<f32>(0.0, 0.0, 1.0));
    return ry * rx * rz;
}

fn object_position(t: ObjectTransform) -> vec3<f32> {
    return vec3<f32>(t.position[0], t.position[1], t.position[2]);
}

fn object_scale(t: ObjectTransform) -> vec3<f32> {
    return vec3<f32>(t.scale[0], t.scale[1], t.scale[2]);
}
