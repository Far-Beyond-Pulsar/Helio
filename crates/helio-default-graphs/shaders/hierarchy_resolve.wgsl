// Composes each object's world transform from its parent chain (see
// `hierarchy_resolve.rs`). Prepended with scene_join_common.wgsl, which
// supplies `ObjectTransform`, `object_rotation`, `object_position`,
// `object_scale` and `relative_position`.
//
// Output rows are relative to the frame's world origin (`origin_hi + origin_lo`,
// zero outside camera-relative frames): the topmost ancestor's position has
// the origin removed *before* anything is added to it, so a nested object far
// from the origin keeps the precision of its offset, not of its distance.
//
// Two results per row:
// - world: the transform composed through every ancestor, relative to the
//   frame's world origin. What lights, environment volumes and anything else
//   that is not frame-aware read.
// - placement: the transform composed only up to the nearest *rigid frame*
//   ancestor (not including it), plus that frame's slot (the object's space).
//   Objects drawn through the coordinate-space table read this: a frame moving
//   does not change it. For objects with no frame above, placement == world.
//
// Entry points:
// - `cs_resolve`: every row (a full resolve).
// - `cs_mark_transform` / `cs_mark_parent` / `cs_mark_frame`: for each row of a
//   flushed-row list `[count, rows..]`, set the changed bit for what was
//   written (1 = transform, 2 = parent link, 4 = frame slot).
// - `cs_resolve_moved`: world rows are recomposed where the row or any ancestor
//   changed; placement rows (and the moved list `[count, rows..]`) where the row
//   or any ancestor up to its frame changed. The rest keep last frame's result.
// - `cs_finish`: sizes the moved-owner join's dispatch from the moved list.

const MAX_DEPTH: u32 = 64u;
const WORKGROUP: u32 = 64u;

const BIT_TRANSFORM: u32 = 1u;
const BIT_PARENT: u32 = 2u;
const BIT_FRAME: u32 = 4u;

struct ResolveUniforms {
    rows: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
    origin_hi: vec4<f32>,
    origin_lo: vec4<f32>,
}

@group(0) @binding(0) var<uniform> u: ResolveUniforms;
/// Parent-relative transforms: the frontend's `Transform` column.
@group(0) @binding(1) var<storage, read> local_transforms: array<ObjectTransform>;
/// Per object: the parent's entity index plus one, zero for none.
@group(0) @binding(2) var<storage, read> parents: array<u32>;
/// World-space (origin-relative) result, for readers that are not frame-aware.
@group(0) @binding(3) var<storage, read_write> world_transforms: array<ObjectTransform>;
/// `[count, rows..]`: the rows a column's last flush wrote.
@group(0) @binding(4) var<storage, read> flushed: array<u32>;
/// One word per row: which columns changed (`BIT_*`).
@group(0) @binding(5) var<storage, read_write> changed: array<atomic<u32>>;
/// `[count, rows..]`: the rows whose placement was recomposed this frame.
@group(0) @binding(6) var<storage, read_write> moved: array<atomic<u32>>;
/// Dispatch arguments for the moved-owner join.
@group(0) @binding(7) var<storage, read_write> args: array<u32>;
/// Per object: its frame slot when it is a rigid frame, else zero.
@group(0) @binding(8) var<storage, read> frames: array<u32>;
/// Frame-relative result: composed up to (not including) the nearest frame.
@group(0) @binding(9) var<storage, read_write> placement: array<ObjectTransform>;
/// Per object: the slot of its nearest frame ancestor, zero for the world.
@group(0) @binding(10) var<storage, read_write> spaces: array<u32>;

fn parent_of(index: u32) -> u32 {
    if index >= arrayLength(&parents) {
        return 0u;
    }
    return parents[index];
}

fn frame_slot(index: u32) -> u32 {
    if index >= arrayLength(&frames) {
        return 0u;
    }
    return frames[index];
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

struct Composed {
    transform: ObjectTransform,
    /// The frame slot composition stopped at, zero if it reached the world.
    space: u32,
}

/// Row `index`'s transform composed up its parent chain. With `stop_at_frames`
/// the chain ends below the nearest rigid frame (the result is relative to it,
/// no origin removed); otherwise it runs to the world and is relative to the
/// world origin.
fn compose_row(index: u32, stop_at_frames: bool) -> Composed {
    let own = local_transforms[index];
    var parent = parent_of(index);
    if parent == 0u {
        var root = own;
        let p = relative_position(object_position(own), u.origin_hi.xyz, u.origin_lo.xyz);
        root.position = array<f32, 3>(p.x, p.y, p.z);
        return Composed(root, 0u);
    }
    var lin = linear_part(own);
    var offset = object_position(own);
    var reached_top = false;
    var space = 0u;
    for (var depth = 0u; depth < MAX_DEPTH; depth = depth + 1u) {
        if parent == 0u {
            break;
        }
        let at = parent - 1u;
        if at >= u.rows {
            break;
        }
        if stop_at_frames {
            let slot = frame_slot(at);
            if slot != 0u {
                // Relative to this frame: its own transform lives in the frame
                // table, and nothing here is near the world origin.
                space = slot;
                break;
            }
        }
        let up = local_transforms[at];
        let up_lin = linear_part(up);
        let next_parent = parent_of(at);
        var up_position = object_position(up);
        if next_parent == 0u {
            // The topmost ancestor: its position is the only large term.
            up_position = relative_position(up_position, u.origin_hi.xyz, u.origin_lo.xyz);
            reached_top = true;
        }
        offset = up_lin * offset + up_position;
        lin = up_lin * lin;
        parent = next_parent;
    }
    if space == 0u && !reached_top {
        offset = relative_position(offset, u.origin_hi.xyz, u.origin_lo.xyz);
    }
    return Composed(decompose(lin, offset), space);
}

@compute @workgroup_size(64)
fn cs_resolve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let index = gid.x;
    if index >= u.rows {
        return;
    }
    world_transforms[index] = compose_row(index, false).transform;
    let local = compose_row(index, true);
    placement[index] = local.transform;
    spaces[index] = local.space;
}

fn mark(row_slot: u32, bit: u32) {
    if row_slot >= flushed[0] {
        return;
    }
    let row = flushed[1u + row_slot];
    if row < u.rows && row < arrayLength(&changed) {
        atomicOr(&changed[row], bit);
    }
}

@compute @workgroup_size(64)
fn cs_mark_transform(@builtin(global_invocation_id) gid: vec3<u32>) {
    mark(gid.x, BIT_TRANSFORM);
}

@compute @workgroup_size(64)
fn cs_mark_parent(@builtin(global_invocation_id) gid: vec3<u32>) {
    mark(gid.x, BIT_PARENT);
}

@compute @workgroup_size(64)
fn cs_mark_frame(@builtin(global_invocation_id) gid: vec3<u32>) {
    mark(gid.x, BIT_FRAME);
}

/// Whether row `index` or any ancestor (through frames) changed: the world row
/// depends on all of them.
fn chain_changed_world(index: u32) -> bool {
    if atomicLoad(&changed[index]) != 0u {
        return true;
    }
    var parent = parent_of(index);
    for (var depth = 0u; depth < MAX_DEPTH; depth = depth + 1u) {
        if parent == 0u {
            break;
        }
        let at = parent - 1u;
        if at >= u.rows {
            break;
        }
        if atomicLoad(&changed[at]) != 0u {
            return true;
        }
        parent = parent_of(at);
    }
    return false;
}

/// Whether the placement row of `index` changed: the row itself, or an
/// ancestor below its nearest frame. A frame's own transform and parent link do
/// not matter (they live in the frame table); whether it is a frame does.
fn chain_changed_local(index: u32) -> bool {
    if atomicLoad(&changed[index]) != 0u {
        return true;
    }
    var parent = parent_of(index);
    for (var depth = 0u; depth < MAX_DEPTH; depth = depth + 1u) {
        if parent == 0u {
            break;
        }
        let at = parent - 1u;
        if at >= u.rows {
            break;
        }
        let bits = atomicLoad(&changed[at]);
        if frame_slot(at) != 0u {
            return (bits & BIT_FRAME) != 0u;
        }
        if bits != 0u {
            return true;
        }
        parent = parent_of(at);
    }
    return false;
}

@compute @workgroup_size(64)
fn cs_resolve_moved(@builtin(global_invocation_id) gid: vec3<u32>) {
    let index = gid.x;
    if index >= u.rows {
        return;
    }
    if !chain_changed_world(index) {
        return;
    }
    world_transforms[index] = compose_row(index, false).transform;
    if !chain_changed_local(index) {
        return;
    }
    let local = compose_row(index, true);
    placement[index] = local.transform;
    spaces[index] = local.space;
    let at = atomicAdd(&moved[0], 1u);
    if 1u + at < arrayLength(&moved) {
        atomicStore(&moved[1u + at], index);
    }
}

@compute @workgroup_size(1)
fn cs_finish() {
    let count = min(atomicLoad(&moved[0]), arrayLength(&moved) - 1u);
    atomicStore(&moved[0], count);
    args[0] = (count + WORKGROUP - 1u) / WORKGROUP;
    args[1] = 1u;
    args[2] = 1u;
}
