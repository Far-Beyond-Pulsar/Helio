// SDF Scroll Detection — single-workgroup compute shader.
//
// One thread per clip level (8 levels max).  Each thread:
//   1. Reads the camera position from the engine camera uniform.
//   2. Computes the new brick-snapped level origin.
//   3. Compares against the persistent origin stored in scroll_state.
//   4. If origin changed OR edit generation changed → sets dirty_flags[level].
//   5. Updates scroll_state with the new origin and edit generation.
//
// CPU side dispatches exactly:  dispatch_workgroups(1, 1, 1)
// → O(1) CPU cost every frame.

// ── Structs ─────────────────────────────────────────────────────────────────

struct CameraUniform {
    view:          mat4x4<f32>,
    proj:          mat4x4<f32>,
    view_proj:     mat4x4<f32>,
    inv_view_proj: mat4x4<f32>,
    position_near: vec4<f32>,   // xyz = world pos, w = near
    forward_far:   vec4<f32>,
    jitter_frame:  vec4<f32>,
    prev_view_proj: mat4x4<f32>,
};

struct ClipConfig {
    level_count:          u32,
    grid_dim:             u32,
    brick_size:           u32,
    brick_grid_dim:       u32,
    bricks_per_level:     u32,
    atlas_bricks_per_axis: u32,
    base_voxel_size:      f32,
    edit_count:           u32,
    bvh_node_count:       u32,
    terrain_enabled:      u32,
    terrain_y_min:        f32,
    terrain_y_max:        f32,
    _pad0:                u32,
    _pad1:                u32,
    _pad2:                u32,
    _pad3:                u32,
    // Per-level voxel sizes packed into two vec4s (levels 0-3, 4-7).
    voxel_sizes_lo:       vec4<f32>,   // levels 0,1,2,3
    voxel_sizes_hi:       vec4<f32>,   // levels 4,5,6,7
};

struct ScrollState {
    // Current snapped-brick origin per level (xyz integer brick coords, w unused).
    snap_origins:   array<vec4<i32>, 8>,
    // Monotonically-increasing edit generation written by CPU.
    edit_gen:       u32,
    // Last edit_gen seen by GPU; GPU updates this here.
    prev_edit_gen:  u32,
    _pad0:          u32,
    _pad1:          u32,
};

// ── Bindings ─────────────────────────────────────────────────────────────────

@group(0) @binding(0) var<uniform>            camera:       CameraUniform;
@group(0) @binding(1) var<uniform>            clip_config:  ClipConfig;
@group(0) @binding(2) var<storage, read_write> scroll_state: ScrollState;
@group(0) @binding(3) var<storage, read_write> dirty_flags:  array<u32>;

// ── Helpers ──────────────────────────────────────────────────────────────────

fn level_voxel_size(level: u32) -> f32 {
    let lo = clip_config.voxel_sizes_lo;
    let hi = clip_config.voxel_sizes_hi;
    if level == 0u { return lo.x; }
    else if level == 1u { return lo.y; }
    else if level == 2u { return lo.z; }
    else if level == 3u { return lo.w; }
    else if level == 4u { return hi.x; }
    else if level == 5u { return hi.y; }
    else if level == 6u { return hi.z; }
    else { return hi.w; }
}

// ── Kernel ───────────────────────────────────────────────────────────────────

// One thread per clip level.  workgroup_size=8 covers all 8 levels in one workgroup.
@compute @workgroup_size(8, 1, 1)
fn cs_scroll(@builtin(local_invocation_id) lid: vec3<u32>) {
    let level = lid.x;
    if level >= clip_config.level_count { return; }

    let cam_pos    = camera.position_near.xyz;
    let vs         = level_voxel_size(level);
    let brick_step = vs * f32(clip_config.brick_size);

    // Snap camera to integer brick coordinates.
    let new_snap = vec3<i32>(
        i32(floor(cam_pos.x / brick_step)),
        i32(floor(cam_pos.y / brick_step)),
        i32(floor(cam_pos.z / brick_step)),
    );

    let old_snap = scroll_state.snap_origins[level].xyz;
    let cam_moved  = any(new_snap != old_snap);
    let edit_dirty = scroll_state.edit_gen != scroll_state.prev_edit_gen;

    if cam_moved || edit_dirty {
        dirty_flags[level] = 1u;
        scroll_state.snap_origins[level] = vec4<i32>(new_snap, 0);
    }

    // Thread 0 acknowledges the edit generation after all level threads have
    // had a chance to detect the change (all run in the same workgroup).
    workgroupBarrier();
    if level == 0u && edit_dirty {
        scroll_state.prev_edit_gen = scroll_state.edit_gen;
    }
}
