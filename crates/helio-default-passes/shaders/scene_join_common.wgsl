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

/// An object's world transform (`TransformRow`, 48 bytes), keyed by the
/// object entity: its f64 position as an f32 pair (`position_hi`, the
/// nearest f32; `position_lo`, what that drops), so a position on a
/// planet keeps its millimetres; `rotation` is YXZ Euler angles in degrees.
struct ObjectTransform {
    position_hi: array<f32, 3>,
    position_lo: array<f32, 3>,
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

fn position_hi(t: ObjectTransform) -> vec3<f32> {
    return vec3<f32>(t.position_hi[0], t.position_hi[1], t.position_hi[2]);
}

fn position_lo(t: ObjectTransform) -> vec3<f32> {
    return vec3<f32>(t.position_lo[0], t.position_lo[1], t.position_lo[2]);
}

/// The object's world position, rounded to f32 (for passes that draw in
/// world space).
fn object_position(t: ObjectTransform) -> vec3<f32> {
    return position_hi(t) + position_lo(t);
}

/// The object's position relative to the world origin `hi + lo` (the f64
/// origin's f32 pair), part by part: near the origin `position_hi - hi` is
/// exact, so the offset keeps the f64 position's precision.
fn object_relative(t: ObjectTransform, hi: vec3<f32>, lo: vec3<f32>) -> vec3<f32> {
    return (position_hi(t) - hi) + (position_lo(t) - lo);
}

fn object_scale(t: ObjectTransform) -> vec3<f32> {
    return vec3<f32>(t.scale[0], t.scale[1], t.scale[2]);
}
