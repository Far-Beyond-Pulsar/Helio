//! GPU camera uniform types.

use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec3};

/// Return the canonical temporal-AA jitter for `frame`, in pixel units.
///
/// The R1/R2 low-discrepancy sequence does not repeat on a short fixed cycle.
/// Every producer of jittered geometry or ray-marched samples must use this
/// function; temporal resolves consume the exact resulting camera offset rather
/// than generating another sequence independently.
#[inline]
pub fn temporal_jitter(frame: u64) -> [f32; 2] {
    const INV_R1: f64 = 0.7548776662466927;
    const INV_R2: f64 = 0.5698402905980539;
    const PHASE: f64 = 0.5;

    let fx = frame as f64 * INV_R1 + PHASE;
    let fy = frame as f64 * INV_R2 + PHASE;
    [(fx.fract() - 0.5) as f32, (fy.fract() - 0.5) as f32]
}

/// Convert the canonical pixel-space jitter to the NDC translation applied to
/// a projection matrix for a render target of `width` x `height`.
#[inline]
pub fn temporal_jitter_ndc(frame: u64, width: u32, height: u32) -> [f32; 2] {
    let [x, y] = temporal_jitter(frame);
    [
        x * 2.0 / width.max(1) as f32,
        y * 2.0 / height.max(1) as f32,
    ]
}

/// Per-frame camera uniforms uploaded to GPU every frame.
///
/// Layout matches the WGSL `Camera` struct in all shaders.
/// 256 bytes total (one full uniform buffer row for alignment).
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct GpuCameraUniforms {
    /// View matrix (world → view space)
    pub view: [f32; 16],
    /// Projection matrix (view → clip space)
    pub proj: [f32; 16],
    /// Combined view-projection matrix
    pub view_proj: [f32; 16],
    /// Inverse view-projection (clip → world space, for reconstruction)
    pub inv_view_proj: [f32; 16],
    /// Camera world position (xyz) + near plane (w)
    pub position_near: [f32; 4],
    /// Camera forward direction (xyz) + far plane (w)
    pub forward_far: [f32; 4],
    /// Jitter offset for TAA (xy) + frame index (z) + padding (w)
    pub jitter_frame: [f32; 4],
    /// Previous frame view-projection (for TAA motion vectors)
    pub prev_view_proj: [f32; 16],
}

impl GpuCameraUniforms {
    /// Creates a new camera uniform from decomposed matrices.
    pub fn new(
        view: Mat4,
        proj: Mat4,
        position: Vec3,
        near: f32,
        far: f32,
        frame: u32,
        jitter: [f32; 2],
        prev_view_proj: Mat4,
    ) -> Self {
        let view_proj = proj * view;
        let inv_view_proj = view_proj.inverse();
        let forward = (-view.z_axis.truncate()).normalize();
        Self {
            view: view.to_cols_array(),
            proj: proj.to_cols_array(),
            view_proj: view_proj.to_cols_array(),
            inv_view_proj: inv_view_proj.to_cols_array(),
            position_near: [position.x, position.y, position.z, near],
            forward_far: [forward.x, forward.y, forward.z, far],
            jitter_frame: [jitter[0], jitter[1], frame as f32, 0.0],
            prev_view_proj: prev_view_proj.to_cols_array(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{temporal_jitter, temporal_jitter_ndc};

    #[test]
    fn temporal_jitter_is_bounded_and_non_repeating() {
        let mut samples = std::collections::HashSet::new();
        for frame in 0..256 {
            let [x, y] = temporal_jitter(frame);
            assert!((-0.5..0.5).contains(&x));
            assert!((-0.5..0.5).contains(&y));
            assert!(samples.insert(((x * 1_000_000.0) as i32, (y * 1_000_000.0) as i32)));
        }
    }

    #[test]
    fn ndc_jitter_round_trips_to_pixel_offset() {
        for (width, height) in [(1, 1), (960, 540), (1920, 1080)] {
            for frame in 0..64 {
                let pixel = temporal_jitter(frame);
                let ndc = temporal_jitter_ndc(frame, width, height);
                assert!((ndc[0] * width as f32 * 0.5 - pixel[0]).abs() < 1.0e-6);
                assert!((ndc[1] * height as f32 * 0.5 - pixel[1]).abs() < 1.0e-6);
            }
        }
    }
}
