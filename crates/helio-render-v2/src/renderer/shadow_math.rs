//! Shadow light-space matrix computation (CSM, point-light cubemaps, spot).

use glam::{Mat4, Vec3, Vec4Swizzles};

/// World-space distance thresholds for the 4 CSM cascade boundaries.
/// Cascade i covers [CSM_SPLITS[i-1], CSM_SPLITS[i]] (cascade 0 starts at 0).
pub(crate) const CSM_SPLITS: [f32; 4] = [16.0, 80.0, 300.0, 1400.0];

/// Compute 6 cube-face view-proj matrices for a point light (±X, ±Y, ±Z).
pub(crate) fn compute_point_light_matrices(position: [f32; 3], range: f32) -> [Mat4; 6] {
    let pos = Vec3::from(position);
    let proj = Mat4::perspective_rh(std::f32::consts::FRAC_PI_2, 1.0, 0.05, range.max(0.1));
    let views = [
        Mat4::look_at_rh(pos, pos + Vec3::X,  -Vec3::Y), // +X
        Mat4::look_at_rh(pos, pos - Vec3::X,  -Vec3::Y), // -X
        Mat4::look_at_rh(pos, pos + Vec3::Y,   Vec3::Z), // +Y
        Mat4::look_at_rh(pos, pos - Vec3::Y,  -Vec3::Z), // -Y
        Mat4::look_at_rh(pos, pos + Vec3::Z,  -Vec3::Y), // +Z
        Mat4::look_at_rh(pos, pos - Vec3::Z,  -Vec3::Y), // -Z
    ];
    views.map(|view| proj * view)
}

/// Compute 4 cascaded ortho light-space matrices for a directional light.
///
/// Uses **sphere-fit + texel snapping** — the standard "stable CSM" algorithm
/// used by UE4, Unity HDRP, and id Tech 7:
///
/// 1. Fit a bounding **sphere** (not AABB) to each camera-frustum slice.
///    A sphere is rotation-invariant, so the ortho-box dimensions never change
///    as the camera turns → zero shadow swimming.
/// 2. **Snap** the light-view origin to shadow-map texel boundaries so the
///    projected shadow never crawls as the camera translates.
/// 3. Pull the light camera back by `SCENE_DEPTH` and extend the far plane by
///    the same amount so off-screen casters (ceilings, terrain…) always cast.
///
/// Slots 0-3 hold the four cascade matrices; slots 4-5 are identity (reserved
/// for point-light cube faces which are not used for directional lights).
pub(crate) fn compute_directional_cascades(
    cam_pos: Vec3,
    view_proj_inv: Mat4,
    direction: [f32; 3],
) -> [Mat4; 4] {
    /// Maximum world-space distance from any frustum centre that shadow
    /// casters are guaranteed to be pulled in from.
    const SCENE_DEPTH: f32 = 4000.0;
    /// Shadow-atlas resolution per cascade (must match ShadowsFeature atlas_size).
    const ATLAS_TEXELS: f32 = 2048.0;

    let dir = Vec3::from(direction).normalize();
    let up  = if dir.dot(Vec3::Y).abs() > 0.99 { Vec3::Z } else { Vec3::Y };

    // Un-project 8 NDC corners to world space (wgpu depth: 0 = near, 1 = far)
    let ndc: [[f32; 4]; 8] = [
        [-1.0,-1.0, 0.0, 1.0], [1.0,-1.0, 0.0, 1.0],
        [-1.0, 1.0, 0.0, 1.0], [1.0, 1.0, 0.0, 1.0],
        [-1.0,-1.0, 1.0, 1.0], [1.0,-1.0, 1.0, 1.0],
        [-1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0],
    ];
    let world: Vec<Vec3> = ndc.iter().map(|c| {
        let v = view_proj_inv * glam::Vec4::from(*c);
        v.xyz() / v.w
    }).collect();
    // world[0..4] = near plane corners, world[4..8] = far plane corners

    let near_dist: f32 = world[..4].iter().map(|c| (*c - cam_pos).length()).sum::<f32>() / 4.0;
    let far_dist:  f32 = world[4..].iter().map(|c| (*c - cam_pos).length()).sum::<f32>() / 4.0;
    let depth = (far_dist - near_dist).max(1.0);

    let prev_d = [0.0_f32, CSM_SPLITS[0], CSM_SPLITS[1], CSM_SPLITS[2]];
    let mut matrices = [Mat4::IDENTITY; 4];

    for i in 0..4 {
        let t0 = ((prev_d[i]     - near_dist) / depth).clamp(0.0, 1.0);
        let t1 = ((CSM_SPLITS[i] - near_dist) / depth).clamp(0.0, 1.0);

        // 8 world-space corners of this frustum slice
        let mut cc = [Vec3::ZERO; 8];
        for j in 0..4 {
            cc[j * 2]     = world[j].lerp(world[j + 4], t0);
            cc[j * 2 + 1] = world[j].lerp(world[j + 4], t1);
        }

        // ── Step 1: bounding sphere of the 8 corners ──────────────────────────
        // Centre = centroid; radius = max distance from centre to any corner.
        // A sphere is rotation-invariant → ortho extents never change with yaw/pitch.
        let centroid = cc.iter().copied().fold(Vec3::ZERO, |a, b| a + b) / 8.0;
        let radius   = cc.iter().map(|c| (*c - centroid).length()).fold(0.0_f32, f32::max);
        // Round radius up to the nearest texel to eliminate sub-texel size jitter
        let texel_size  = (2.0 * radius) / ATLAS_TEXELS;
        let radius_snap = (radius / texel_size).ceil() * texel_size;

        // ── Step 2: texel-snapped light-view origin ────────────────────────────
        // Project centroid onto the light's view plane (XY), then quantise to
        // integer texel offsets so the shadow grid never sub-texel-crawls.
        let light_view_raw = Mat4::look_at_rh(centroid - dir * SCENE_DEPTH, centroid, up);
        let centroid_ls    = light_view_raw.transform_point3(centroid);
        let snap           = texel_size;
        let snapped_x      = (centroid_ls.x / snap).round() * snap;
        let snapped_y      = (centroid_ls.y / snap).round() * snap;

        // Reconstruct the right/up axes of the light view to apply the snap in world space
        let right_ws = light_view_raw.row(0).truncate().normalize();
        let up_ws    = light_view_raw.row(1).truncate().normalize();
        let snap_offset = right_ws * (snapped_x - centroid_ls.x)
                        + up_ws   * (snapped_y - centroid_ls.y);
        let stable_centroid = centroid + snap_offset;

        // Final light view: look from far behind the stable centroid
        let light_view = Mat4::look_at_rh(
            stable_centroid - dir * SCENE_DEPTH,
            stable_centroid,
            up,
        );

        // ── Step 3: ortho projection from the sphere radius ────────────────────
        // Z: near = 0.1 (camera is SCENE_DEPTH behind centroid),
        //    far  = SCENE_DEPTH * 2 (covers SCENE_DEPTH in front of centroid).
        // Any caster within SCENE_DEPTH of the scene is guaranteed to cast.
        let proj = Mat4::orthographic_rh(
            -radius_snap, radius_snap,
            -radius_snap, radius_snap,
            0.1, SCENE_DEPTH * 2.0,
        );
        matrices[i] = proj * light_view;
    }

    matrices
}

/// Compute the light-space matrix for a spot light (perspective projection).
pub(crate) fn compute_spot_matrix(position: [f32; 3], direction: [f32; 3], range: f32, outer_angle: f32) -> Mat4 {
    let pos = Vec3::from(position);
    let dir = Vec3::from(direction).normalize();
    let up = if dir.dot(Vec3::Y).abs() > 0.99 { Vec3::Z } else { Vec3::Y };
    let view = Mat4::look_at_rh(pos, pos + dir, up);
    let fov = (outer_angle * 2.0).clamp(std::f32::consts::FRAC_PI_4, std::f32::consts::PI - 0.01);
    let proj = Mat4::perspective_rh(fov, 1.0, 0.05, range.max(0.1));
    proj * view
}
