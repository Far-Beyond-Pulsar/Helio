//! GPU camera uniform types.

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
