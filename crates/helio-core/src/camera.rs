//! GPU camera uniform types.
//!
//! Generic: every pass in the graph shares exactly one camera per frame, so
//! this is a first-class `helio-core` shape (like `PassContext::camera`),
//! not a pass-owned resource.

use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec3};

/// Per-frame camera uniforms uploaded to GPU every frame: one per view the
/// frame renders (the host's camera, its XR eyes, and the secondary views of
/// [`crate::view`]).
///
/// Layout matches the prelude's WGSL `Camera` struct (464 bytes).
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
    /// Camera forward direction (xyz) + effect range (w). The projection has
    /// no far plane (see [`crate::depth`]); `w` only bounds distance-limited
    /// effects such as froxel fog.
    pub forward_far: [f32; 4],
    /// Projection translation in NDC (xy), frame index (z), view identity (w,
    /// stored as u32 bits rather than numerically converted to float).
    /// Ray tracers convert xy with `temporal::ray_jitter_from_ndc`.
    pub jitter_frame: [f32; 4],
    /// Previous frame view-projection (for TAA motion vectors)
    pub prev_view_proj: [f32; 16],
    /// The view's clip plane in frame space (`xyz` normal, `w` offset):
    /// what lies on its negative side is not drawn (a planar reflection's
    /// water plane, a portal's destination). `0` for none; drawers test it
    /// with the prelude's `helio_view_clipped` in pipelines built for a
    /// clipped view ([`crate::view::ViewDescription::clipped`]).
    pub clip_plane: [f32; 4],
    /// What view this is ([`crate::view::ViewInfo`]): `x` the inline-view
    /// slot + 1 (`0` for a view drawn into its own G-buffer), `y` flags
    /// (`VIEW_MIRRORED`, `VIEW_CLIPPED`), `z` the view's depth, `w` unused.
    pub view_info: [u32; 4],
    /// For an inline view, the rigid map from the view's content space into
    /// the space of the G-buffer it draws into (identity otherwise): drawers
    /// write G-buffer normals through it (the prelude's `helio_view_normal`),
    /// so every pixel of a G-buffer holds normals in one space and the
    /// lighting maps an inline view's back (`VIEW_REMAPS`).
    pub content_to_drawn: [f32; 16],
}

impl GpuCameraUniforms {
    /// Attach a generic view identity without changing the shared camera ABI.
    pub fn with_view_id(mut self, view_id: u32) -> Self {
        self.jitter_frame[3] = f32::from_bits(view_id);
        self
    }

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
            clip_plane: [0.0; 4],
            view_info: [0; 4],
            content_to_drawn: Mat4::IDENTITY.to_cols_array(),
        }
    }

    /// This camera seen through a rigid map `content_to_drawn` (and its
    /// previous frame's `prev_content_to_drawn`): the camera that, looking at
    /// content in its own space, puts it exactly where drawing it through the
    /// map with this camera would (`view * content_to_drawn`, the same
    /// projection, so depth stays the frame's convention). Used for inline
    /// views ([`crate::view::ViewKind::Inline`]): a portal's destination seen
    /// from its source.
    pub fn through(&self, content_to_drawn: Mat4, prev_content_to_drawn: Mat4) -> Self {
        let view = Mat4::from_cols_array(&self.view) * content_to_drawn;
        let proj = Mat4::from_cols_array(&self.proj);
        let view_proj = proj * view;
        let drawn_to_content = content_to_drawn.inverse();
        let position =
            drawn_to_content.transform_point3(Vec3::from_slice(&self.position_near[..3]));
        let forward = drawn_to_content
            .transform_vector3(Vec3::from_slice(&self.forward_far[..3]))
            .normalize_or_zero();
        let prev = Mat4::from_cols_array(&self.prev_view_proj) * prev_content_to_drawn;
        Self {
            view: view.to_cols_array(),
            proj: self.proj,
            view_proj: view_proj.to_cols_array(),
            inv_view_proj: view_proj.inverse().to_cols_array(),
            position_near: [position.x, position.y, position.z, self.position_near[3]],
            forward_far: [forward.x, forward.y, forward.z, self.forward_far[3]],
            jitter_frame: self.jitter_frame,
            prev_view_proj: prev.to_cols_array(),
            clip_plane: [0.0; 4],
            view_info: self.view_info,
            content_to_drawn: (Mat4::from_cols_array(&self.content_to_drawn) * content_to_drawn)
                .to_cols_array(),
        }
    }

    /// This camera with clip plane `plane` (frame space; see
    /// [`Self::clip_plane`]).
    pub fn with_clip_plane(mut self, plane: glam::Vec4) -> Self {
        self.clip_plane = plane.to_array();
        self
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

    /// A camera seen through a map draws content where drawing it through
    /// the map with the camera would, at the same depth: an inline view's
    /// pixels and depth are the frame's (`crate::view`).
    #[test]
    fn a_camera_through_a_map_puts_content_where_the_map_would() {
        let eye = Vec3::new(1.0, 2.0, 3.0);
        let view = glam::camera::rh::view::look_at_mat4(eye, Vec3::ZERO, Vec3::Y);
        let proj = crate::depth::perspective(0.9, 1.5, 0.1);
        let camera = GpuCameraUniforms::new(view, proj, eye, 0.1, 500.0, 3, [0.0; 2], proj * view);
        let map = Mat4::from_rotation_y(1.2) * Mat4::from_translation(Vec3::new(-40.0, 0.0, 9.0));
        let through = camera.through(map, map);
        let content = Vec3::new(41.0, 0.5, -10.0);
        let direct =
            Mat4::from_cols_array(&camera.view_proj) * map.transform_point3(content).extend(1.0);
        let seen = Mat4::from_cols_array(&through.view_proj) * content.extend(1.0);
        let (direct, seen) = (direct.truncate() / direct.w, seen.truncate() / seen.w);
        assert!(direct.distance(seen) < 1e-4, "{direct} vs {seen}");
        // The eye is where it is in the content's space, and the map is kept
        // for the G-buffer's normals.
        let eye_content = Vec3::from_slice(&through.position_near[..3]);
        assert!(map.transform_point3(eye_content).distance(eye) < 1e-4);
        assert_eq!(through.content_to_drawn, map.to_cols_array());
    }

    #[test]
    fn forward_matches_look_direction_under_yaw_pitch_and_translation() {
        let eye = Vec3::new(43.0, 11.0, 78.0);
        for target in [Vec3::new(0.0, 24.0, 0.0), Vec3::new(90.0, -8.0, 110.0)] {
            let view = glam::camera::rh::view::look_at_mat4(eye, target, Vec3::Y);
            let proj = crate::depth::perspective(0.85, 16.0 / 9.0, 0.1);
            let camera =
                GpuCameraUniforms::new(view, proj, eye, 0.1, 350.0, 0, [0.0; 2], proj * view);
            let forward = Vec3::from_slice(&camera.forward_far);
            assert!(forward.distance((target - eye).normalize()) < 1e-5);
            // A centre-ray view depth must also equal its world-space distance.
            let centre = eye + forward * 50.0;
            assert!((-view.transform_point3(centre).z - 50.0).abs() < 1e-4);
        }
    }
}
