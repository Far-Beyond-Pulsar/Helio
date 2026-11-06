// The frame's integrated participating medium (`helio_frame_api::media`):
// the grid's depth mapping, which the publisher writes slices by and every
// reader samples by, and how a surface is seen through it. A mismatch does
// not error, it puts the medium at the wrong distance: hence here, once.
//
// The grid is a view-space 3D texture with exponentially spaced depth
// slices, dense near the camera, sparse far away. Its X and Y are the
// unjittered screen.

/// Near plane of the grid. Deliberately not the camera's near plane:
/// exponential spacing from 0.1 m would spend most slices within a couple of
/// metres of the camera and leave almost none for the distance the medium is
/// read at.
const MEDIA_NEAR: f32 = 0.5;

/// View depth (not radial distance) to the grid's normalized depth in
/// [0, 1], for a grid reaching `far`.
fn media_slice_from_view_depth(view_depth: f32, far: f32) -> f32 {
    let d = max(view_depth, MEDIA_NEAR);
    return log(d / MEDIA_NEAR) / log(max(far, MEDIA_NEAR * 2.0) / MEDIA_NEAR);
}

/// Normalized grid depth in [0, 1] to view depth. Inverse of the above.
fn media_view_depth_from_slice(slice_norm: f32, far: f32) -> f32 {
    return MEDIA_NEAR * pow(max(far, MEDIA_NEAR * 2.0) / MEDIA_NEAR, slice_norm);
}

/// The grid's texture `w` coordinate for a surface at `view_depth`, with
/// `slices` slices: each slice holds its cumulative value at its far face,
/// so the coordinate is offset half a slice for texel-centre sampling.
fn media_grid_w(view_depth: f32, far: f32, slices: u32) -> f32 {
    return clamp(media_slice_from_view_depth(view_depth, far) - 0.5 / f32(slices), 0.0, 1.0);
}

/// A surface of radiance `surface` seen through a grid sample `medium`
/// (`rgb` in-scattered light, `a` transmittance).
fn media_seen(surface: vec3<f32>, medium: vec4<f32>) -> vec3<f32> {
    return surface * clamp(medium.a, 0.0, 1.0) + max(medium.rgb, vec3<f32>(0.0));
}
