//!use helio_prelude
// ── DOF Dispatch Arguments (Compute, 1 thread) ──────────────────────────────
//
// Writes the indirect dispatch arguments for the CoC and gather passes from
// the same DOF block they read. With DOF disabled (aperture_shape < 0) both
// dispatches get zero workgroups: the composite returns the sharp image
// without reading their outputs, so skipping them changes no pixel.
//
// Writes:  args[0..3] = CoC workgroups, args[3..6] = gather workgroups

struct DofUniforms {
    dof_focal_distance:     f32,
    dof_focal_region:       f32,
    dof_aperture_shape:     f32,
    dof_aperture_rotation:  f32,
    dof_near_transition:    f32,
    dof_far_transition:     f32,
    dof_max_bokeh_size:     f32,
    dof_sensor_diagonal:    f32,
}

// Full-size workgroup counts: (coc.x, coc.y, gather.x, gather.y).
struct Groups {
    counts: vec4<u32>,
}

@group(0) @binding(0) var<uniform> dof: DofUniforms;
@group(0) @binding(1) var<uniform> groups: Groups;
@group(0) @binding(2) var<storage, read_write> args: array<u32, 6>;

@compute @workgroup_size(1)
fn cs_args() {
    let on = dof.dof_aperture_shape >= 0.0;
    args[0] = select(0u, groups.counts.x, on);
    args[1] = select(0u, groups.counts.y, on);
    args[2] = select(0u, 1u, on);
    args[3] = select(0u, groups.counts.z, on);
    args[4] = select(0u, groups.counts.w, on);
    args[5] = select(0u, 1u, on);
}
