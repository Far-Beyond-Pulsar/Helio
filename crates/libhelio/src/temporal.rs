//! Shared temporal sampling conventions for raster passes and ray tracers.

/// Express a previous view-projection in a new floating origin. `origin_shift`
/// is new origin minus old origin; current local points therefore need this
/// translation before the previous camera can project them. Compose in f64 so
/// a large origin change does not discard the retained small camera offset.
pub fn rebase_previous_projection(previous: glam::Mat4, origin_shift: glam::DVec3) -> glam::Mat4 {
    assert!(
        origin_shift.is_finite(),
        "camera origin shift must be finite"
    );
    (previous.as_dmat4() * glam::DMat4::from_translation(origin_shift)).as_mat4()
}

/// R1/R2 projection jitter in render pixels, before the NDC conversion.
/// Kept identical to Helio's existing renderer/TSR sequence.
pub fn r1_r2_jitter(frame: u64) -> [f32; 2] {
    const INV_R1: f64 = 0.7548776662466927;
    const INV_R2: f64 = 0.5698402905980539;
    let fx = frame as f64 * INV_R1 + 0.5;
    let fy = frame as f64 * INV_R2 + 0.5;
    [(fx.fract() - 0.5) as f32, (fy.fract() - 0.5) as f32]
}

/// Convert `GpuCameraUniforms::jitter_frame.xy` (NDC projection translation)
/// to ray sample offsets in pixels, with image Y pointing down. Applying a
/// positive projection translation moves geometry right/up, so its inverse
/// ray offsets are left/down. Supply an unjittered ray basis with this offset.
pub fn ray_jitter_from_ndc(jitter: [f32; 2], render_size: [u32; 2]) -> [f32; 4] {
    [
        -jitter[0] * render_size[0] as f32 * 0.5,
        jitter[1] * render_size[1] as f32 * 0.5,
        0.0,
        0.0,
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::{Mat4, Vec3, Vec4};
    #[test]
    fn origin_changes_preserve_previous_projection_without_inventing_motion() {
        let projection = Mat4::perspective_rh(1.0, 16.0 / 9.0, 0.1, 30_000_000.0);
        for shift in [
            glam::DVec3::ZERO,
            glam::DVec3::new(1024.0, -1024.0, 0.0),
            glam::DVec3::new(0.0, 0.0, 8_000_000.0),
        ] {
            let old_camera = shift + glam::DVec3::new(0.031, 0.023, 0.047);
            let previous = projection
                * glam::DMat4::look_to_rh(old_camera, glam::DVec3::NEG_Z, glam::DVec3::Y).as_mat4();
            let rebased = rebase_previous_projection(previous, shift);
            for distance in [1.0, 32.0, 1024.0, 8_000_000.0] {
                let new_point = glam::DVec3::new(0.25, -0.1, -distance);
                let old_point = new_point + shift;
                let reference = previous.as_dmat4() * old_point.extend(1.0);
                // Evaluate f64 here to isolate matrix rebasing from a separate
                // large-local-coordinate precision loss. The host bounds locals.
                let actual = rebased.as_dmat4() * new_point.extend(1.0);
                let a = actual.truncate() / actual.w;
                let b = reference.truncate() / reference.w;
                assert!(
                    (a.x - b.x).abs() < 0.0001 && (a.y - b.y).abs() < 0.0001,
                    "origin shift changed projected XY"
                );
            }
        }
    }
    #[test]
    fn ray_offsets_match_the_engines_jittered_projection() {
        for size in [[960, 540], [1280, 720], [513, 701]] {
            let projection =
                Mat4::perspective_rh(1.0, size[0] as f32 / size[1] as f32, 0.1, 10000.0);
            for frame in [0, 1, 2, 63, 127, 10000] {
                let j = r1_r2_jitter(frame);
                let ndc = [j[0] * 2.0 / size[0] as f32, j[1] * 2.0 / size[1] as f32];
                let shifted = Mat4::from_translation(Vec3::new(ndc[0], ndc[1], 0.0)) * projection;
                let ray = ray_jitter_from_ndc(ndc, size);
                for uv in [[0.5, 0.5], [0.1, 0.8], [0.9, 0.2]] {
                    let actual = shifted.inverse()
                        * Vec4::new(uv[0] * 2.0 - 1.0, 1.0 - uv[1] * 2.0, 0.5, 1.0);
                    let expected = projection.inverse()
                        * Vec4::new(
                            (uv[0] + ray[0] / size[0] as f32) * 2.0 - 1.0,
                            1.0 - (uv[1] + ray[1] / size[1] as f32) * 2.0,
                            0.5,
                            1.0,
                        );
                    assert!(
                        actual
                            .truncate()
                            .normalize()
                            .distance(expected.truncate().normalize())
                            < 0.000001
                    );
                }
            }
        }
    }
}
