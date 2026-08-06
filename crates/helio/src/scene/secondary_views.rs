//! Per-frame publish of active sublevel/portal camera views — the bridge
//! between the CPU registries in [`super::sublevels`] / [`super::portals`]
//! and the GPU passes (`SecondaryGBufferPass`, `ProxyCompositePass`) that
//! read [`libhelio::FrameResources::secondary`].
//!
//! Call order each frame (see `Scene::update_secondary_views`'s doc comment
//! for why): resolve any portal teleport crossing *before* building this
//! frame's camera, then publish views *after* `update_camera`.

use std::sync::Arc;

use glam::{Mat4, Vec3};
use helio_portal_core::{oblique_clip_plane_view, PortalPair, PortalPose};
use helio_secondary_core::{
    portal_eye_camera, screen_rect_for_points, sublevel_camera, GpuSecondaryView, MAX_PORTAL_DEPTH,
    MAX_PORTAL_VIEWS, NO_PARENT, SECONDARY_RESOLUTION_DIVISOR, SECONDARY_VIEW_FLAG_QUAD_DISCARD,
};
use libhelio::{GpuCameraUniforms, SecondaryFrameData};

use crate::handles::PortalId;

use super::core::Scene;

/// First camera slot reserved for portal eyes (slots `2..5`).
const FIRST_PORTAL_SLOT: u32 = helio_secondary_core::FIRST_SECONDARY_SLOT;
/// First camera slot reserved for sublevel cameras (slots `5..7`).
const FIRST_SUBLEVEL_SLOT: u32 = FIRST_PORTAL_SLOT + helio_secondary_core::MAX_PORTAL_VIEWS;

/// A portal crossing detected by [`Scene::take_portal_teleport`].
#[derive(Debug, Clone, Copy)]
pub struct PortalTeleport {
    /// The portal crossed.
    pub portal: PortalId,
    /// The camera's new world position (`pair_map` applied to the old one).
    pub new_position: Vec3,
    /// Pure rotation to pre-multiply onto the camera's current orientation
    /// (`pair_map`'s rotation part).
    pub rotation: Mat4,
}

impl Scene {
    /// Test the camera's motion since the last call against every registered
    /// portal's plane and, on a crossing, return the teleported pose.
    ///
    /// **Call this before building this frame's `Camera`** — the caller is
    /// expected to apply the returned position/rotation to the camera it is
    /// about to pass to `Renderer::render`, then call
    /// [`Scene::update_camera`]. Portal rendering itself
    /// ([`Scene::update_secondary_views`]) assumes the camera it's given has
    /// already been teleported this frame, so a crossing is never rendered
    /// and taken in the same frame — matching the design doc §7.4's "the
    /// camera never double-traverses in one frame".
    ///
    /// Returns `None` (and just records `camera_world_pos` for next frame)
    /// on the very first call — there is no previous position to have
    /// crossed from.
    pub fn take_portal_teleport(&mut self, camera_world_pos: Vec3) -> Option<PortalTeleport> {
        let prev = self.portal_prev_camera_pos.replace(camera_world_pos);
        let Some(prev) = prev else {
            return None;
        };
        let crossing = self.portals.iter().find_map(|(id, record)| {
            if helio_portal_core::crossing_detected(prev, camera_world_pos, &record.a, record.half_extent) {
                let pair = PortalPair { a: record.a, b: record.b };
                Some((id, pair))
            } else {
                None
            }
        });
        let (id, pair) = crossing?;
        let new_position = pair.map_point(camera_world_pos);
        let rotation = rotation_part(pair.pair_map());
        // Record the post-teleport position so next frame's crossing test
        // starts from where the camera actually ended up, not where it
        // would have been without the portal.
        self.portal_prev_camera_pos = Some(new_position);
        self.taa_reset_pending = true;
        Some(PortalTeleport { portal: id, new_position, rotation })
    }

    /// Consume the "a portal teleport happened this/last frame, reset TAA
    /// history" signal. Returns `true` at most once per crossing.
    pub fn take_taa_reset_pending(&mut self) -> bool {
        std::mem::take(&mut self.taa_reset_pending)
    }

    /// Rebuild this frame's secondary (sublevel + portal) camera views:
    /// writes each active view's `GpuCameraUniforms` into its camera slot and
    /// stages the `GpuSecondaryView` array for [`Scene::secondary_frame_data`]
    /// to publish.
    ///
    /// **Call this after [`Scene::update_camera`]** with that same frame's
    /// uniforms — sublevel cameras and portal eyes are both derived from the
    /// main camera — and **before** building `FrameResources` for the frame.
    /// Split from the actual publish (`secondary_frame_data`, `&self`) for
    /// the same reason `Scene::foliage_frame_data` is a separate `&self`
    /// call from whatever mutates foliage state: `FrameResources` borrows
    /// from `Scene` for the rest of the frame, so the work that needs
    /// `&mut self` (writing camera slots here) has to finish first.
    ///
    /// Portal visibility (which of the registered portals get a camera slot
    /// this frame) is re-decided every call from `main_camera` — a portal
    /// that isn't currently on screen doesn't consume a slot, unlike
    /// sublevels, which hold their slot for as long as they're active
    /// (design doc §5: "Portals take the whole scene... and instead rely on
    /// the oblique clip plane"; sublevels have persistent membership, so
    /// their slot assignment is likewise persistent — see
    /// `Scene::add_sublevel`).
    pub fn refresh_secondary_views(&mut self, main_camera: &GpuCameraUniforms, viewport: [u32; 2]) {
        self.secondary_cpu_views.clear();
        let queue = self.gpu_scene.queue.clone();
        let main_vp = Mat4::from_cols_array(&main_camera.view_proj);
        println!(
            "view_proj[2]=({:.2},{:.2},{:.2},{:.2}) [3]=({:.2},{:.2},{:.2},{:.2})",
            main_camera.view_proj[8], main_camera.view_proj[9], main_camera.view_proj[10], main_camera.view_proj[11],
            main_camera.view_proj[12], main_camera.view_proj[13], main_camera.view_proj[14], main_camera.view_proj[15],
        );
        println!(
            "cam pos=({:.1},{:.1},{:.1}) fwd=({:.1},{:.1},{:.1}) near={:.2} far={:.2}",
            main_camera.position_near[0], main_camera.position_near[1], main_camera.position_near[2],
            main_camera.forward_far[0], main_camera.forward_far[1], main_camera.forward_far[2],
            main_camera.position_near[3], main_camera.forward_far[3],
        );

        // ── Sublevels ────────────────────────────────────────────────────
        for i in 0..self.sublevels.dense_len() {
            let Some(record) = self.sublevels.get_dense(i) else {
                continue;
            };
            let Some(view_slot) = record.view_slot else {
                continue;
            };
            let camera_slot = FIRST_SUBLEVEL_SLOT + view_slot;
            let cam = sublevel_camera(main_camera, record.placement);
            self.gpu_scene.camera.update_slot(&queue, camera_slot, &cam);
            self.secondary_cpu_views.push(GpuSecondaryView {
                camera_slot,
                view_flags: 0,
                clip_plane: [0.0; 4],
                // v1 simplification: full-viewport region rather than the
                // sublevel's actual on-screen AABB (design doc's
                // `region_rect` is a fill-cost optimisation, not a
                // correctness requirement — the composite's fixed-function
                // depth test is correct over the whole screen too).
                region_rect: [0.0, 0.0, viewport[0] as f32, viewport[1] as f32],
                space_transform: Mat4::IDENTITY.to_cols_array(),
                resolution_scale: 1.0,
                parent_index: NO_PARENT,
                _pad: [0.0; 2],
            });
        }

        // ── Portals (recursive, depth ≤ MAX_PORTAL_DEPTH, budget MAX_PORTAL_VIEWS) ──
        //
        // A depth-0 (seen by the main camera) portal composites straight into
        // the main G-buffer chain (`parent_index = NO_PARENT`). A portal seen
        // *from within* an already-active eye's own rendered world composites
        // into that eye's pooled secondary G-buffer instead
        // (`parent_index = Some(that eye's view index)`) — `ProxyCompositePass`
        // processes children before parents so the chain resolves
        // innermost-first, exactly Portal 2's own recursion technique.
        let main_viewer_pose = PortalPose {
            transform: Mat4::from_cols_array(&main_camera.view).inverse(),
        };
        let near = main_camera.position_near[3];
        let far = main_camera.forward_far[3];
        let fov_y = fov_y_from_proj(&main_camera.proj);
        let aspect = viewport[0] as f32 / viewport[1].max(1) as f32;
        // Every pooled secondary-view slot shares one fixed resolution
        // regardless of recursion depth (`SecondaryGBufferPass` sizes its
        // whole pool once) — a nested view's `region_rect` must be expressed
        // against *that* resolution, since its composite destination is the
        // parent's pooled slot, not the main frame.
        let pool_viewport = [
            (viewport[0] / SECONDARY_RESOLUTION_DIVISOR).max(1),
            (viewport[1] / SECONDARY_RESOLUTION_DIVISOR).max(1),
        ];

        let mut next_portal_slot = 0u32;
        self.recurse_portal_views(
            &main_viewer_pose,
            &main_vp,
            viewport,
            pool_viewport,
            NO_PARENT,
            0,
            main_camera,
            near,
            far,
            fov_y,
            aspect,
            &queue,
            &mut next_portal_slot,
        );
    }

    /// DFS portal-visibility recursion — see `refresh_secondary_views`'s
    /// portal section for the model. One call handles one recursion level:
    /// for every portal visible from `viewer_pose` (tested against
    /// `viewer_vp`/`dest_viewport`, the *destination* this level composites
    /// into), allocate a camera slot + view-array entry, then recurse one
    /// level deeper from the new eye's own pose/projection, now always
    /// against `pool_viewport` (every deeper level's destination is a pooled
    /// secondary-G-buffer slot, not the main frame).
    #[allow(clippy::too_many_arguments)]
    fn recurse_portal_views(
        &mut self,
        viewer_pose: &PortalPose,
        viewer_vp: &Mat4,
        dest_viewport: [u32; 2],
        pool_viewport: [u32; 2],
        parent_index: u32,
        depth: u32,
        main_camera: &GpuCameraUniforms,
        near: f32,
        far: f32,
        fov_y: f32,
        aspect: f32,
        queue: &Arc<wgpu::Queue>,
        next_slot: &mut u32,
    ) {
        if depth >= MAX_PORTAL_DEPTH {
            return;
        }
        for i in 0..self.portals.dense_len() {
            if *next_slot >= MAX_PORTAL_VIEWS {
                return;
            }
            let Some(record) = self.portals.get_dense(i) else {
                continue;
            };
            let corners = portal_quad_corners(&record.a, record.half_extent);
            let c0_clip = viewer_vp * corners[0].extend(1.0);
            println!(
                "portal #{} corner0 clip=({:.1},{:.1},{:.1},{:.1})",
                i, c0_clip.x, c0_clip.y, c0_clip.z, c0_clip.w,
            );
            let region_rect = screen_rect_for_points(viewer_vp, &corners, &dest_viewport);
            if region_rect[2] <= 0.0 || region_rect[3] <= 0.0 {
                println!("portal #{} NOT VISIBLE (w={:.2})", i, c0_clip.w);
                continue;
            }

            let pair = PortalPair { a: record.a, b: record.b };
            let mut eye = pair.eye_pose(viewer_pose);
                println!(
                "portal view #{:?}: a_pos=({:.1},{:.1},{:.1}) a_fwd=({:.1},{:.1},{:.1}), \
                 eye pre-flip: pos=({:.1},{:.1},{:.1}) fwd=({:.1},{:.1},{:.1}), \
                 b_pos=({:.1},{:.1},{:.1}) b_fwd=({:.1},{:.1},{:.1}), \
                 region_rect=({:.0},{:.0},{:.0},{:.0})",
                i, record.a.position().x, record.a.position().y, record.a.position().z,
                record.a.forward().x, record.a.forward().y, record.a.forward().z,
                eye.position().x, eye.position().y, eye.position().z,
                eye.forward().x, eye.forward().y, eye.forward().z,
                record.b.position().x, record.b.position().y, record.b.position().z,
                record.b.forward().x, record.b.forward().y, record.b.forward().z,
                region_rect[0], region_rect[1], region_rect[2], region_rect[3],
            );
            // When both portals in a pair face the same direction, the
            // pair_map is a pure translation and eye_pose places the camera
            // on the *destination* side of B looking away from its scene.
            // Detect this and reflect the eye across B's plane to the entry
            // side so it looks toward B (and thus its destination scene).
            let b_fwd = record.b.forward();
            let side = (eye.position() - record.b.position()).dot(b_fwd);
            if side > 0.0 {
                // Reflect the eye position across B's plane to the entry side
                // while preserving the orientation (forward direction stays the
                // same — it now points toward B instead of away).
                let reflect_pos = eye.position() - 2.0 * side * b_fwd;
                let rot = rotation_part(eye.transform);
                eye = PortalPose { transform: Mat4::from_translation(reflect_pos) * rot };
            println!(
                    "  -> flipped eye to entry side: pos=({:.1},{:.1},{:.1}) fwd=({:.1},{:.1},{:.1})",
                    eye.position().x, eye.position().y, eye.position().z,
                    eye.forward().x, eye.forward().y, eye.forward().z,
                );
            }
            // XXX: diagnostic — skip oblique clip to test if frustum cull is
            // discarding geometry.
            //let clip = oblique_clip_plane_view(&eye, &record.b, near.max(0.01));
            //let cam = portal_eye_camera(main_camera, &eye, fov_y, aspect, near, far, Some(clip));
            let cam = portal_eye_camera(main_camera, &eye, fov_y, aspect, near, far, None);

            let camera_slot = FIRST_PORTAL_SLOT + *next_slot;
            let view_index = self.secondary_cpu_views.len() as u32;
            self.gpu_scene.camera.update_slot(queue, camera_slot, &cam);
            self.secondary_cpu_views.push(GpuSecondaryView {
                camera_slot,
                view_flags: SECONDARY_VIEW_FLAG_QUAD_DISCARD,
                clip_plane: clip.to_array(),
                region_rect,
                // Always *this* portal's own inverse map, never composed
                // across depth — see `GpuSecondaryView::space_transform`'s
                // doc comment for why nesting doesn't accumulate a transform.
                space_transform: pair.pair_map_inverse().to_cols_array(),
                resolution_scale: 1.0,
                parent_index,
                _pad: [0.0; 2],
            });
            *next_slot += 1;

            let eye_vp = Mat4::from_cols_array(&cam.view_proj);
            self.recurse_portal_views(
                &eye,
                &eye_vp,
                pool_viewport,
                pool_viewport,
                view_index,
                depth + 1,
                main_camera,
                near,
                far,
                fov_y,
                aspect,
                queue,
                next_slot,
            );
        }
    }

    /// Number of secondary (sublevel + portal) views staged by the last
    /// [`Scene::refresh_secondary_views`] call. Exposed for diagnostics/demos
    /// (e.g. an on-screen "N portal views active" counter) and tests.
    pub fn secondary_view_count(&self) -> usize {
        self.secondary_cpu_views.len()
    }

    /// `parent_index` of every staged secondary view, in publish order — see
    /// [`GpuSecondaryView::parent_index`]. A value other than
    /// `helio_secondary_core::NO_PARENT` at some index means that view is a
    /// *nested* portal recursion, compositing into the view at that parent
    /// index instead of the main frame. Exposed for diagnostics/tests.
    pub fn secondary_view_parent_indices(&self) -> Vec<u32> {
        self.secondary_cpu_views.iter().map(|v| v.parent_index).collect()
    }

    /// This frame's staged secondary-view data, built by
    /// [`Scene::refresh_secondary_views`]. `None` when nothing is active —
    /// the zero-cost contract `FrameResources::secondary` documents; leave
    /// the slot unwritten rather than publishing an empty struct.
    pub fn secondary_frame_data(&self) -> Option<SecondaryFrameData<'_>> {
        if self.secondary_cpu_views.is_empty() {
            None
        } else {
            Some(SecondaryFrameData {
                view_bytes: bytemuck::cast_slice(&self.secondary_cpu_views),
                view_count: self.secondary_cpu_views.len() as u32,
            })
        }
    }
}

/// The four world-space corners of a portal's local XY rectangle
/// (`±half_extent`), transformed into world space.
fn portal_quad_corners(pose: &PortalPose, half_extent: glam::Vec2) -> [Vec3; 4] {
    let (hx, hy) = (half_extent.x, half_extent.y);
    [
        pose.transform.transform_point3(Vec3::new(-hx, -hy, 0.0)),
        pose.transform.transform_point3(Vec3::new(hx, -hy, 0.0)),
        pose.transform.transform_point3(Vec3::new(hx, hy, 0.0)),
        pose.transform.transform_point3(Vec3::new(-hx, hy, 0.0)),
    ]
}

/// Recover vertical FOV (radians) from a wgpu-convention perspective
/// projection's `[1][1]` entry (`1 / tan(fovy / 2)`).
fn fov_y_from_proj(proj: &[f32; 16]) -> f32 {
    // Column-major: entry (row=1, col=1) is at index col*4 + row = 5.
    let y_scale = proj[5].max(1e-4);
    2.0 * (1.0 / y_scale).atan()
}

/// The pure rotation part of a rigid transform (see `helio-portal-core`'s
/// private `rotation_of` — duplicated here since it isn't exposed publicly
/// and this is the only other call site that needs it).
fn rotation_part(t: Mat4) -> Mat4 {
    Mat4::from_cols_array_2d(&[
        [t.x_axis.x, t.x_axis.y, t.x_axis.z, 0.0],
        [t.y_axis.x, t.y_axis.y, t.y_axis.z, 0.0],
        [t.z_axis.x, t.z_axis.y, t.z_axis.z, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ])
}
