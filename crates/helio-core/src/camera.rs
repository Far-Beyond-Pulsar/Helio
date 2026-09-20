//! GPU camera uniform types.
//!
//! Generic: every pass in the graph shares exactly one camera per frame, so
//! this is a first-class `helio-core` shape (like `PassContext::camera`),
//! not a pass-owned resource.

use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec3};

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
    /// Projection translation in NDC (xy), frame index (z), padding (w).
    /// Ray tracers convert xy with `temporal::ray_jitter_from_ndc`.
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
        // Transform camera-local forward back to world space. A column of the
        // world-to-view matrix is not a world-space camera axis after rotation.
        let forward = view.inverse().transform_vector3(Vec3::NEG_Z).normalize();
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

    /// Upload left and right eye camera data into a storage buffer.
    ///
    /// The buffer must be sized for at least two `GpuCameraUniforms` elements.
    pub fn upload_stereo(queue: &wgpu::Queue, buffer: &wgpu::Buffer, left: &Self, right: &Self) {
        let data = [*left, *right];
        queue.write_buffer(buffer, 0, bytemuck::cast_slice(&data));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn forward_matches_look_direction_under_yaw_pitch_and_translation() {
        let eye = Vec3::new(43.0, 11.0, 78.0);
        for target in [Vec3::new(0.0, 24.0, 0.0), Vec3::new(90.0, -8.0, 110.0)] {
            let view = glam::camera::rh::view::look_at_mat4(eye, target, Vec3::Y);
            let proj = glam::camera::rh::proj::directx::perspective(0.85, 16.0 / 9.0, 0.1, 350.0);
            let camera = GpuCameraUniforms::new(view, proj, eye, 0.1, 350.0, 0, [0.0; 2], proj * view);
            let forward = Vec3::from_slice(&camera.forward_far);
            assert!(forward.distance((target - eye).normalize()) < 1e-5);
            // A centre-ray view depth must also equal its world-space distance.
            let centre = eye + forward * 50.0;
            assert!((-view.transform_point3(centre).z - 50.0).abs() < 1e-4);
        }
    }
}
