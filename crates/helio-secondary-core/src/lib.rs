//! Secondary-view camera machinery for portals and sublevels.
//!
//! Helio generalizes its GPU camera set to `array<Camera, CAMERA_SLOTS>`; the
//! main (and XR eyes) occupy slots 0..[`XR_EYE_SLOTS`] and secondary views —
//! portal eyes and sublevel cameras — start at [`FIRST_SECONDARY_SLOT`]. This
//! crate owns that slot bookkeeping, the per-view uniform that accompanies each
//! secondary camera, and the CPU mirrors of the math the secondary passes run on
//! the GPU (region projection, sublevel camera derivation, portal eye build).
//!
//! The layout of [`GpuSecondaryView`] is pinned by a `const _` size assert the
//! same way `libhelio` pins its GPU structs.

#![warn(missing_docs)]
#![forbid(unsafe_code)]

use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec3, Vec4};
use helio_portal_core::{oblique_near_projection, PortalPose};
use libhelio::GpuCameraUniforms;

// ── Camera-set layout ─────────────────────────────────────────────────────────

/// Maximum simultaneously-active portal views per frame.
pub const MAX_PORTAL_VIEWS: u32 = 3;
/// Maximum simultaneously-active sublevel views per frame.
pub const MAX_SUBLEVEL_VIEWS: u32 = 2;
/// Camera slots reserved for the main camera / XR stereo eyes.
pub const XR_EYE_SLOTS: u32 = 2;
/// Total GPU camera slots. Must equal `CAMERA_SLOTS` in `libhelio::camera` and
/// `prelude.wgsl`.
pub const CAMERA_SLOTS: u32 = XR_EYE_SLOTS + MAX_PORTAL_VIEWS + MAX_SUBLEVEL_VIEWS;
/// First slot that holds a secondary view (portal or sublevel).
pub const FIRST_SECONDARY_SLOT: u32 = XR_EYE_SLOTS;
/// Total secondary-view slots available in the camera set.
pub const MAX_SECONDARY_VIEWS: u32 = MAX_PORTAL_VIEWS + MAX_SUBLEVEL_VIEWS;

/// Maximum portal recursion depth — a camera (or an already-recursed eye)
/// traverses at most this many portals in one chain before recursion stops
/// (design doc §7.3, matching Portal 2's own ~3-level cap). Bounded by
/// [`MAX_PORTAL_VIEWS`] itself: one deep recursive chain and zero independent
/// side-by-side portals is the extreme case the camera-slot budget allows.
pub const MAX_PORTAL_DEPTH: u32 = MAX_PORTAL_VIEWS;

/// The secondary G-buffer pool's resolution relative to the internal render
/// resolution (`ResourceSize::ScaledInternal { divisor: SECONDARY_RESOLUTION_DIVISOR }`
/// in `SecondaryGBufferPass::declare_resources`). Every pooled slot shares
/// this one fixed resolution regardless of recursion depth — a nested
/// (`parent_index != NO_PARENT`) view's `region_rect` is expressed against
/// this resolution, not the main output resolution, because its composite
/// destination is the parent's pooled slot, not the main frame.
pub const SECONDARY_RESOLUTION_DIVISOR: u32 = 2;

// ── GpuSecondaryView flags ────────────────────────────────────────────────────

/// Bit 0: the view's projection has an oblique near plane from [`GpuSecondaryView::clip_plane`].
pub const SECONDARY_VIEW_FLAG_CLIP_PLANE: u32 = 1 << 0;
/// Bit 1: the composite must discard pixels outside the proxy's on-screen quad
/// (portals); sublevels never set this.
pub const SECONDARY_VIEW_FLAG_QUAD_DISCARD: u32 = 1 << 1;

/// Sentinel for [`GpuSecondaryView::parent_index`]: this view composites
/// directly into the main G-buffer chain (depth 0/1 — seen by the main
/// camera), not into another secondary view's off-screen G-buffer.
pub const NO_PARENT: u32 = u32::MAX;

// ── GpuSecondaryView ──────────────────────────────────────────────────────────

/// Per-view CPU→GPU description of a secondary render.
///
/// Companion to the `Camera` the view reads from `cameras[camera_slot]`; the
/// shaders cannot infer these from the camera alone.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct GpuSecondaryView {
    /// Index into the `cameras` storage array.
    pub camera_slot: u32,
    /// Bitfield of the `SECONDARY_VIEW_FLAG_*` constants.
    pub view_flags: u32,
    /// Oblique near plane (view space) when `SECONDARY_VIEW_FLAG_CLIP_PLANE` is set.
    pub clip_plane: [f32; 4],
    /// Region the composite scissors its draw to, in **destination** pixels:
    /// xy = min, zw = size. The destination is the main frame when
    /// `parent_index == NO_PARENT`, or the parent view's own pooled
    /// secondary-G-buffer slot otherwise (both share one fixed resolution —
    /// see `helio_secondary_core::SECONDARY_RESOLUTION_DIVISOR` — so this is
    /// never the main-frame viewport for a nested view).
    pub region_rect: [f32; 4],
    /// Transform applied to reconstructed world positions at composite time:
    /// identity for sublevels, that portal's own `pair_map_inverse()` for
    /// portals — always that *specific* portal's own inverse map, never
    /// composed across recursion depth (every portal pose lives in the same
    /// single absolute world, so nesting doesn't accumulate a transform; see
    /// `helio::scene::secondary_views` for the full reasoning).
    pub space_transform: [f32; 16],
    /// Resolution of the off-screen target relative to the region (1.0 = full).
    pub resolution_scale: f32,
    /// Recursion parent: [`NO_PARENT`], or another active view's **index in
    /// this same frame's `GpuSecondaryView` array** (not a camera slot) whose
    /// own pooled secondary G-buffer this view composites into instead of the
    /// main chain. `ProxyCompositePass` processes views with a parent before
    /// the view they point at, so a chain composites innermost-first.
    pub parent_index: u32,
    /// Reserved for future use; keeps the struct size pinned.
    pub _pad: [f32; 2],
}

impl Default for GpuSecondaryView {
    fn default() -> Self {
        Self {
            camera_slot: 0,
            view_flags: 0,
            clip_plane: [0.0; 4],
            region_rect: [0.0; 4],
            space_transform: [0.0; 16],
            resolution_scale: 1.0,
            parent_index: NO_PARENT,
            _pad: [0.0; 2],
        }
    }
}

const _: () = assert!(std::mem::size_of::<GpuSecondaryView>() == 120);

// ── Screen-region math ────────────────────────────────────────────────────────

/// Project `points` through `view_proj` and return the screen-space rect they
/// cover as `[min_x, min_y, width, height]`, clamped to `viewport`.
///
/// Uses the wgpu UV convention (y down, origin top-left). Points behind the
/// camera are skipped. Returns a zero rect when nothing projects.
pub fn screen_rect_for_points(view_proj: &Mat4, points: &[Vec3], viewport: &[u32; 2]) -> [f32; 4] {
    let vpw = viewport[0] as f32;
    let vph = viewport[1] as f32;
    let mut min = [f32::INFINITY; 2];
    let mut max = [-f32::INFINITY; 2];
    for &p in points {
        let clip = view_proj * p.extend(1.0);
        if clip.w <= 0.0 {
            continue;
        }
        let ndc = clip.truncate() / clip.w;
        let px = (ndc.x * 0.5 + 0.5) * vpw;
        let py = (0.5 - ndc.y * 0.5) * vph;
        min[0] = min[0].min(px);
        min[1] = min[1].min(py);
        max[0] = max[0].max(px);
        max[1] = max[1].max(py);
    }
    if !min[0].is_finite() || !min[1].is_finite() {
        return [0.0, 0.0, 0.0, 0.0];
    }
    clamp_rect([min[0], min[1], max[0] - min[0], max[1] - min[1]], viewport)
}

/// Clamp a `[min_x, min_y, width, height]` rect into `viewport`.
pub fn clamp_rect(rect: [f32; 4], viewport: &[u32; 2]) -> [f32; 4] {
    let max_x = viewport[0] as f32;
    let max_y = viewport[1] as f32;
    let min_x = rect[0].clamp(0.0, max_x);
    let min_y = rect[1].clamp(0.0, max_y);
    let width = (rect[2]).clamp(0.0, max_x - min_x);
    let height = (rect[3]).clamp(0.0, max_y - min_y);
    [min_x, min_y, width, height]
}

// ── Sublevel camera derivation ────────────────────────────────────────────────

/// The view-projection a sublevel renders with.
///
/// Objects in a sublevel keep **local** transforms; the unchanged gbuffer vertex
/// shader computes `view_proj * model * vertex`, so baking the placement into
/// the camera (`main_view_proj * placement`) renders those local transforms at
/// their placed world positions. Moving the whole structure changes only
/// `placement` — no instance transform is ever touched.
pub fn sublevel_view_proj(main_view_proj: Mat4, placement: Mat4) -> Mat4 {
    main_view_proj * placement
}

/// A full camera uniform for a sublevel view, derived from the main camera.
///
/// The returned uniform renders sublevel-local geometry at its placed world
/// position while keeping the camera in sublevel-local space. G-buffer values it
/// produces are placed world positions, so lighting and the composite need no
/// extra transform.
pub fn sublevel_camera(main: &GpuCameraUniforms, placement: Mat4) -> GpuCameraUniforms {
    let placement_inv = placement.inverse();
    let main_pos = Vec3::new(main.position_near[0], main.position_near[1], main.position_near[2]);
    let main_fwd = Vec3::new(main.forward_far[0], main.forward_far[1], main.forward_far[2]);
    let local_pos = placement_inv.transform_point3(main_pos);
    let local_fwd = placement_inv.transform_vector3(main_fwd).normalize();
    let view_proj = sublevel_view_proj(Mat4::from_cols_array(&main.view_proj), placement);
    let view = Mat4::from_cols_array(&main.view) * placement;
    GpuCameraUniforms {
        view: view.to_cols_array(),
        proj: main.proj,
        view_proj: view_proj.to_cols_array(),
        inv_view_proj: view_proj.inverse().to_cols_array(),
        position_near: [local_pos.x, local_pos.y, local_pos.z, main.position_near[3]],
        forward_far: [local_fwd.x, local_fwd.y, local_fwd.z, main.forward_far[3]],
        // No TAA jitter for secondary views (xy); frame index (z) carried
        // through from the main camera, matching `portal_eye_camera`.
        jitter_frame: [0.0, 0.0, main.jitter_frame[2], 0.0],
        prev_view_proj: view_proj.to_cols_array(),
    }
}

// ── Portal eye camera build (fixed-FOV v1) ────────────────────────────────────

/// View matrix of the portal eye.
///
/// v1 uses a fixed FOV matching the main camera; the region-matched off-axis
/// projection that frames the portal quad exactly is a later refinement.
pub fn eye_view(eye_position: Vec3, eye_forward: Vec3, eye_up: Vec3) -> Mat4 {
    glam::camera::rh::view::look_at_mat4(eye_position, eye_position + eye_forward, eye_up)
}

/// Perspective projection in the wgpu [0,1] depth convention.
pub fn perspective_rh(fov_y_radians: f32, aspect: f32, near: f32, far: f32) -> Mat4 {
    glam::camera::rh::proj::directx::perspective(fov_y_radians, aspect, near, far)
}

/// A full camera uniform for a portal eye, derived from the main camera and an
/// eye pose (`PortalPair::eye_pose`).
///
/// v1 uses a fixed FOV matching the main camera (region-matched off-axis
/// framing of the on-screen quad is a later refinement, design doc §7.2) and
/// applies the oblique near-clip **CPU-side**, by folding
/// `helio_portal_core::oblique_near_projection` into the projection matrix
/// before it's baked into `view_proj`, when `clip_view` is `Some`. This is
/// mathematically identical to a GPU per-vertex clip term and means the
/// secondary G-buffer's vertex shader needs no portal-specific branch: it is
/// byte-identical for sublevels and portals alike.
///
/// `clip_view` is the oblique clip plane in the **eye's own view space**
/// (`helio_portal_core::oblique_clip_plane_view(&eye, &portal, near)`) — the
/// caller builds it, this function only applies it to the projection.
///
/// Like [`sublevel_camera`], `prev_view_proj` is set equal to `view_proj`:
/// secondary views are not TAA'd directly, so there is no meaningful "last
/// frame" for them.
pub fn portal_eye_camera(
    main: &GpuCameraUniforms,
    eye: &PortalPose,
    fov_y_radians: f32,
    aspect: f32,
    near: f32,
    far: f32,
    clip_view: Option<Vec4>,
) -> GpuCameraUniforms {
    let eye_pos = eye.position();
    let eye_fwd = eye.forward();
    let view = eye_view(eye_pos, eye_fwd, eye.up());
    let mut proj = perspective_rh(fov_y_radians, aspect, near, far);
    if let Some(clip) = clip_view {
        proj = oblique_near_projection(proj, clip);
    }
    let view_proj = proj * view;
    GpuCameraUniforms {
        view: view.to_cols_array(),
        proj: proj.to_cols_array(),
        view_proj: view_proj.to_cols_array(),
        inv_view_proj: view_proj.inverse().to_cols_array(),
        position_near: [eye_pos.x, eye_pos.y, eye_pos.z, near],
        forward_far: [eye_fwd.x, eye_fwd.y, eye_fwd.z, far],
        // No TAA jitter for secondary views (xy); frame index (z) carried
        // through from the main camera for any shader that keys noise/dither
        // off it, matching `GpuCameraUniforms::new`'s layout (`jitter_frame`
        // = jitter.xy, frame index z, padding w).
        jitter_frame: [0.0, 0.0, main.jitter_frame[2], 0.0],
        prev_view_proj: view_proj.to_cols_array(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::Vec2;

    fn test_camera() -> Mat4 {
        let view = glam::camera::rh::view::look_at_mat4(Vec3::new(0.0, 5.0, 10.0), Vec3::ZERO, Vec3::Y);
        let proj = glam::camera::rh::proj::directx::perspective(60f32.to_radians(), 16.0 / 9.0, 0.1, 1000.0);
        proj * view
    }

    #[test]
    fn screen_rect_covers_a_visible_cube() {
        let vp = test_camera();
        let cube = [
            Vec3::new(-1.0, -1.0, -1.0),
            Vec3::new(1.0, -1.0, -1.0),
            Vec3::new(1.0, 1.0, -1.0),
            Vec3::new(-1.0, 1.0, -1.0),
        ];
        let rect = screen_rect_for_points(&vp, &cube, &[1600, 900]);
        assert!(rect[2] > 0.0, "width should be positive: {rect:?}");
        assert!(rect[3] > 0.0, "height should be positive: {rect:?}");
        assert!(rect[0] >= 0.0 && rect[0] + rect[2] <= 1600.0);
        assert!(rect[1] >= 0.0 && rect[1] + rect[3] <= 900.0);
    }

    #[test]
    fn screen_rect_is_zero_for_points_behind_camera() {
        let vp = test_camera();
        let behind = [Vec3::new(0.0, 5.0, 20.0)];
        assert_eq!(screen_rect_for_points(&vp, &behind, &[1600, 900]), [0.0; 4]);
    }

    #[test]
    fn clamp_rect_stays_in_viewport() {
        assert_eq!(clamp_rect([-5.0, -5.0, 2000.0, 2000.0], &[1600, 900]), [0.0, 0.0, 1600.0, 900.0]);
        assert_eq!(clamp_rect([10.0, 20.0, 100.0, 100.0], &[1600, 900]), [10.0, 20.0, 100.0, 100.0]);
    }

    #[test]
    fn sublevel_view_proj_bakes_placement_into_camera() {
        let main = GpuCameraUniforms::new(
            glam::camera::rh::view::look_at_mat4(Vec3::new(0.0, 5.0, 10.0), Vec3::ZERO, Vec3::Y),
            glam::camera::rh::proj::directx::perspective(60f32.to_radians(), 16.0 / 9.0, 0.1, 1000.0),
            Vec3::new(0.0, 5.0, 10.0),
            0.1,
            1000.0,
            0,
            [0.0, 0.0],
            Mat4::IDENTITY,
        );
        let placement = Mat4::from_translation(Vec3::new(100.0, 0.0, 0.0));
        let sub = sublevel_camera(&main, placement);

        // A sublevel-local point at the origin renders at the same screen
        // position as the placed world point (100,0,0) in the main camera.
        let local = Vec3::ZERO;
        let world = placement.transform_point3(local);
        let main_vp = Mat4::from_cols_array(&main.view_proj);
        let sub_vp = Mat4::from_cols_array(&sub.view_proj);

        let clip_local = sub_vp * local.extend(1.0);
        let clip_world = main_vp * world.extend(1.0);
        let ndc_local = clip_local.truncate() / clip_local.w;
        let ndc_world = clip_world.truncate() / clip_world.w;
        assert!((ndc_local.x - ndc_world.x).abs() < 1e-3);
        assert!((ndc_local.y - ndc_world.y).abs() < 1e-3);
    }

    #[test]
    fn eye_view_looks_along_forward() {
        let v = eye_view(Vec3::ZERO, Vec3::Z, Vec3::Y);
        let origin = v.transform_point3(Vec3::ZERO);
        let target = v.transform_point3(Vec3::new(0.0, 0.0, 10.0));
        // A view matrix looks down -Z: the forward world point lands on -Z.
        let dir = (target - origin).normalize();
        assert!((dir - (-Vec3::Z)).length() < 1e-4);
    }

    #[test]
    fn perspective_projection_is_wgpu_depth() {
        let proj = perspective_rh(60f32.to_radians(), 1.0, 0.1, 1000.0);
        let clip = proj * Vec3::new(0.0, 0.0, -10.0).extend(1.0);
        let ndc_z = clip.z / clip.w;
        assert!(ndc_z > 0.0 && ndc_z < 1.0, "NDC z should be in [0,1], got {ndc_z}");

        // Sanity: Vec2 type is in scope for the API shape.
        let _ = Vec2::ZERO;
    }

    fn main_camera_uniforms() -> GpuCameraUniforms {
        GpuCameraUniforms::new(
            glam::camera::rh::view::look_at_mat4(Vec3::new(0.0, 5.0, 10.0), Vec3::ZERO, Vec3::Y),
            glam::camera::rh::proj::directx::perspective(60f32.to_radians(), 16.0 / 9.0, 0.1, 1000.0),
            Vec3::new(0.0, 5.0, 10.0),
            0.1,
            1000.0,
            42,
            [0.0, 0.0],
            Mat4::IDENTITY,
        )
    }

    #[test]
    fn portal_eye_camera_places_the_camera_at_the_eye_pose() {
        use helio_portal_core::PortalPair;

        let a = PortalPose::from_look_at(Vec3::ZERO, Vec3::new(0.0, 0.0, 1.0), Vec3::Y);
        let b = PortalPose::from_look_at(
            Vec3::new(50.0, 0.0, 0.0),
            Vec3::new(50.0, 0.0, 1.0),
            Vec3::Y,
        );
        let pair = PortalPair { a, b };
        let eye = pair.eye_pose(&a);

        let main = main_camera_uniforms();
        let cam = portal_eye_camera(&main, &eye, 60f32.to_radians(), 16.0 / 9.0, 0.1, 1000.0, None);

        assert_eq!(cam.position_near[0], eye.position().x);
        assert_eq!(cam.position_near[1], eye.position().y);
        assert_eq!(cam.position_near[2], eye.position().z);
        // Not TAA'd directly: prev_view_proj is this frame's view_proj.
        assert_eq!(cam.prev_view_proj, cam.view_proj);
        // Frame index carried through from the main camera.
        assert_eq!(cam.jitter_frame[2], main.jitter_frame[2]);
    }

    #[test]
    fn portal_eye_camera_sees_the_destination_side_and_clips_the_player_side() {
        use helio_portal_core::{oblique_clip_plane_view, PortalPair};

        // Portal A at the origin; portal B 50 units away, same orientation
        // (pair_map is a pure translation). Viewer stands 3 units behind A,
        // looking through it — same offset the portal-core oblique-clip test
        // uses, so the eye lands 3 units behind B, not coincident with it.
        let a = PortalPose::from_look_at(Vec3::ZERO, Vec3::new(0.0, 0.0, 1.0), Vec3::Y);
        let b = PortalPose::from_look_at(
            Vec3::new(50.0, 0.0, 0.0),
            Vec3::new(50.0, 0.0, 1.0),
            Vec3::Y,
        );
        let viewer = PortalPose::from_look_at(
            Vec3::new(0.0, 0.0, -3.0),
            Vec3::new(0.0, 0.0, 1.0),
            Vec3::Y,
        );
        let pair = PortalPair { a, b };
        let eye = pair.eye_pose(&viewer);
        let clip = oblique_clip_plane_view(&eye, &b, 0.1);

        let main = main_camera_uniforms();
        let cam = portal_eye_camera(
            &main,
            &eye,
            60f32.to_radians(),
            16.0 / 9.0,
            0.1,
            1000.0,
            Some(clip),
        );
        let vp = Mat4::from_cols_array(&cam.view_proj);

        // A point on the far (destination) side of B projects with NDC z in [0,1).
        let far_side = b.position() + b.forward() * 5.0;
        let far_clip = vp * far_side.extend(1.0);
        let far_ndc_z = far_clip.z / far_clip.w;
        assert!(far_ndc_z >= 0.0 && far_ndc_z < 1.0, "far side should be visible: {far_ndc_z}");

        // A point on B's player side is clipped away (NDC z < 0).
        let near_side = b.position() - b.forward() * 1.0;
        let near_clip = vp * near_side.extend(1.0);
        assert!(near_clip.z < 0.0, "player side should be clipped: {}", near_clip.z / near_clip.w);

        // The eye's projection stays invertible so `inv_view_proj` reconstruction works.
        let round = (vp * vp.inverse()).to_cols_array();
        let id = Mat4::IDENTITY.to_cols_array();
        for i in 0..16 {
            assert!((round[i] - id[i]).abs() < 1e-3);
        }
    }
}
