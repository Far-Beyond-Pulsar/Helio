use glam::{Mat3, Mat4, Vec3};

use super::{ring_frame, GizmoAxis, GizmoMode};
use super::gizmo::{
    draw_rotate_gizmo, draw_scale_gizmo, draw_translate_gizmo, gizmo_world_size,
    hit_gizmo, object_gizmo_info, ray_plane_hit, ray_sphere_intersect, ray_to_axis_t,
    sectioned_gizmo_info,
};
use crate::handles::ObjectId;
use crate::renderer::Renderer;
use crate::scene::{Scene, SceneActorId};

// ─────────────────────────────────────────────────────────────────────────────
// Internal drag state
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Clone, Copy)]
enum DragState {
    Idle,
    Active {
        axis: GizmoAxis,
        /// Full object transform captured when the drag started.
        initial_transform: Mat4,
        /// Gizmo centre frozen at drag-start (transform origin, not bounds centroid).
        gizmo_center: Vec3,
        /// Object-local axes frozen at drag-start so the drag constraint stays
        /// consistent even as the object rotates underneath the cursor.
        local_axes: [Vec3; 3],
        /// For Translate/Scale: signed t along the axis at drag-start.
        /// For Rotate: angle in radians at drag-start in the ring plane.
        axis_t_start: f32,
    },
}

/// Per-frame editor state: selection, gizmo mode, hover, and drag.
pub struct EditorState {
    selected: Option<SceneActorId>,
    gizmo_mode: GizmoMode,
    /// Set by `update_hover` — the gizmo axis the cursor is currently over.
    hovered_axis: Option<GizmoAxis>,
    drag: DragState,
}

impl Default for EditorState {
    fn default() -> Self {
        Self::new()
    }
}

impl EditorState {
    /// Create a new editor state with no selection and `GizmoMode::Translate`.
    pub fn new() -> Self {
        Self {
            selected: None,
            gizmo_mode: GizmoMode::default(),
            hovered_axis: None,
            drag: DragState::Idle,
        }
    }

    // ── Selection ─────────────────────────────────────────────────────────────

    /// Explicitly select a scene actor by handle.
    pub fn select(&mut self, id: SceneActorId) {
        self.selected     = Some(id);
        self.hovered_axis = None;
        self.drag         = DragState::Idle;
    }

    /// Clear the current selection.
    pub fn deselect(&mut self) {
        self.selected     = None;
        self.hovered_axis = None;
        self.drag         = DragState::Idle;
    }

    /// Returns the currently selected scene actor, if any.
    pub fn selected(&self) -> Option<SceneActorId> {
        self.selected
    }

    /// Returns the selected scene object, if the current selection is an object.
    pub fn selected_object(&self) -> Option<ObjectId> {
        self.selected.and_then(|id| id.as_object())
    }

    // ── Gizmo mode ────────────────────────────────────────────────────────────

    /// Returns the active gizmo mode.
    pub fn gizmo_mode(&self) -> GizmoMode {
        self.gizmo_mode
    }

    /// Switch the active gizmo mode.
    pub fn set_gizmo_mode(&mut self, mode: GizmoMode) {
        self.gizmo_mode   = mode;
        self.hovered_axis = None;
        self.drag         = DragState::Idle;
    }

    // ── Hover & drag accessors ────────────────────────────────────────────────

    /// The gizmo axis currently under the cursor (set by `update_hover`).
    pub fn hovered_axis(&self) -> Option<GizmoAxis> {
        self.hovered_axis
    }

    /// Whether an axis drag is currently in progress.
    pub fn is_dragging(&self) -> bool {
        matches!(self.drag, DragState::Active { .. })
    }

    // ── Ray unprojection ──────────────────────────────────────────────────────

    /// Convert a screen-space pixel coordinate to a world-space ray.
    ///
    /// Returns `(ray_origin, ray_direction)` both in world space, direction normalized.
    pub fn ray_from_screen(
        px: f32,
        py: f32,
        width: f32,
        height: f32,
        view_proj_inv: Mat4,
    ) -> (Vec3, Vec3) {
        let ndc_x = (px / width) * 2.0 - 1.0;
        let ndc_y = 1.0 - (py / height) * 2.0;
        let near  = view_proj_inv.project_point3(Vec3::new(ndc_x, ndc_y, 0.0));
        let far   = view_proj_inv.project_point3(Vec3::new(ndc_x, ndc_y, 1.0));
        let dir   = (far - near).normalize_or_zero();
        (near, dir)
    }

    // ── Hover ─────────────────────────────────────────────────────────────────

    /// Update the hovered-axis state from the current cursor ray.
    ///
    /// Call on every `CursorMoved` event when the cursor is not grabbed.
    /// Returns `true` if the cursor is over any gizmo handle.
    ///
    /// The gizmo hit-test size is derived from the camera info stored in the
    /// renderer (set via [`Renderer::set_gizmo_camera`]) so that hit zones
    /// remain a consistent screen-space size.
    pub fn update_hover(&mut self, ray_o: Vec3, ray_d: Vec3, renderer: &Renderer) -> bool {
        if self.is_dragging() {
            return true; // keep hover alive while dragging
        }
        let Some((gizmo_camera, viewport_height)) = renderer.gizmo_camera_info() else {
            self.hovered_axis = None;
            return false;
        };
        let scene = renderer.scene();
        match self.selected {
            Some(SceneActorId::Object(id)) => {
                let Some((center, _size, local_axes)) = object_gizmo_info(id, scene) else {
                    self.hovered_axis = None;
                    return false;
                };
                let world_size = gizmo_world_size(center, gizmo_camera, viewport_height);
                self.hovered_axis = hit_gizmo(ray_o, ray_d, center, world_size, self.gizmo_mode, local_axes);
                self.hovered_axis.is_some()
            }
            Some(SceneActorId::SectionedObject(id)) => {
                let Some((center, _size, local_axes)) = sectioned_gizmo_info(id, scene) else {
                    self.hovered_axis = None;
                    return false;
                };
                let world_size = gizmo_world_size(center, gizmo_camera, viewport_height);
                self.hovered_axis = hit_gizmo(ray_o, ray_d, center, world_size, self.gizmo_mode, local_axes);
                self.hovered_axis.is_some()
            }
            Some(SceneActorId::Light(id)) => {
                let Some(light) = scene.get_light(id) else {
                    self.hovered_axis = None;
                    return false;
                };
                let center = Vec3::new(
                    light.position_range[0],
                    light.position_range[1],
                    light.position_range[2],
                );
                let world_size = gizmo_world_size(center, gizmo_camera, viewport_height);
                self.hovered_axis = hit_gizmo(ray_o, ray_d, center, world_size, GizmoMode::Translate, [Vec3::X, Vec3::Y, Vec3::Z]);
                self.hovered_axis.is_some()
            }
            _ => {
                self.hovered_axis = None;
                false
            }
        }
    }

    // ── Drag start / update / end ─────────────────────────────────────────────

    /// Try to begin a gizmo drag.  Call on left-click press.
    ///
    /// Returns `true` if the cursor was over a gizmo handle and the drag has
    /// started (the caller should skip normal scene picking in that case).
    /// Returns `false` when the cursor was not over any handle.
    pub fn try_start_drag(&mut self, ray_o: Vec3, ray_d: Vec3, scene: &Scene) -> bool {
        let axis = match self.hovered_axis {
            Some(a) => a,
            None    => return false,
        };
        match self.selected {
            Some(SceneActorId::Object(id)) => {
                let Some((center, _, local_axes)) = object_gizmo_info(id, scene) else { return false };
                let Ok(initial_transform) = scene.get_object_transform(id) else { return false };

                let axis_dir     = local_axes[axis.col()];
                let axis_t_start = match self.gizmo_mode {
                    GizmoMode::Translate | GizmoMode::Scale => {
                        match ray_to_axis_t(ray_o, ray_d, center, axis_dir) {
                            Some(t) => t,
                            None    => return false,
                        }
                    }
                    GizmoMode::Rotate => {
                        let hit = match ray_plane_hit(ray_o, ray_d, center, axis_dir) {
                            Some(h) => h,
                            None    => return false,
                        };
                        let (tan, bitan) = ring_frame(axis, local_axes);
                        let to_hit = hit - center;
                        to_hit.dot(bitan).atan2(to_hit.dot(tan))
                    }
                };

                self.drag = DragState::Active {
                    axis, initial_transform, gizmo_center: center, local_axes, axis_t_start,
                };
                true
            }
            Some(SceneActorId::SectionedObject(id)) => {
                let Some((center, _, local_axes)) = sectioned_gizmo_info(id, scene) else { return false };
                let Some(initial_transform) = scene.get_sectioned_instance_transform(id) else { return false };

                let axis_dir     = local_axes[axis.col()];
                let axis_t_start = match self.gizmo_mode {
                    GizmoMode::Translate | GizmoMode::Scale => {
                        match ray_to_axis_t(ray_o, ray_d, center, axis_dir) {
                            Some(t) => t,
                            None    => return false,
                        }
                    }
                    GizmoMode::Rotate => {
                        let hit = match ray_plane_hit(ray_o, ray_d, center, axis_dir) {
                            Some(h) => h,
                            None    => return false,
                        };
                        let (tan, bitan) = ring_frame(axis, local_axes);
                        let to_hit = hit - center;
                        to_hit.dot(bitan).atan2(to_hit.dot(tan))
                    }
                };

                self.drag = DragState::Active {
                    axis, initial_transform, gizmo_center: center, local_axes, axis_t_start,
                };
                true
            }
            Some(SceneActorId::Light(id)) => {
                let Some(light) = scene.get_light(id) else { return false };
                let center = Vec3::new(
                    light.position_range[0],
                    light.position_range[1],
                    light.position_range[2],
                );
                let local_axes = [Vec3::X, Vec3::Y, Vec3::Z];
                let axis_dir   = local_axes[axis.col()];
                let Some(t)    = ray_to_axis_t(ray_o, ray_d, center, axis_dir) else { return false };

                self.drag = DragState::Active {
                    axis,
                    initial_transform: Mat4::from_translation(center),
                    gizmo_center: center,
                    local_axes,
                    axis_t_start: t,
                };
                true
            }
            _ => false,
        }
    }

    /// Apply the gizmo drag to the selected object given the current ray.
    ///
    /// Call on every `CursorMoved` event while the left button is held.
    ///
    /// The scale-drag sensitivity is normalised by the screen-space gizmo size
    /// so that scaling feels consistent at any camera distance.
    pub fn update_drag(&mut self, ray_o: Vec3, ray_d: Vec3, renderer: &mut Renderer) {
        let DragState::Active { axis, initial_transform, gizmo_center, local_axes, axis_t_start } = self.drag else {
            return;
        };

        let axis_dir = local_axes[axis.col()];
        let center   = gizmo_center;

        let (camera, viewport_height) = match renderer.gizmo_camera_info() {
            Some(c) => c,
            None    => return,
        };
        let world_size = gizmo_world_size(center, camera, viewport_height);

        let scene = renderer.scene_mut();

        match self.selected {
            Some(SceneActorId::Object(object_id)) => {
                let new_transform = match self.gizmo_mode {
                    GizmoMode::Translate => {
                        let t_now = match ray_to_axis_t(ray_o, ray_d, center, axis_dir) {
                            Some(t) => t,
                            None    => return,
                        };
                        let delta = t_now - axis_t_start;
                        Mat4::from_translation(axis_dir * delta) * initial_transform
                    }

                    GizmoMode::Scale => {
                        let t_now = match ray_to_axis_t(ray_o, ray_d, center, axis_dir) {
                            Some(t) => t,
                            None    => return,
                        };
                        let delta        = t_now - axis_t_start;
                        let sensitivity  = 1.5 / world_size.max(0.01);
                        let scale_factor = (1.0 + delta * sensitivity).max(0.01_f32);

                        let ci      = axis.col();
                        let col     = initial_transform.col(ci);
                        let old_len = col.truncate().length();
                        let new_len = (old_len * scale_factor).max(0.001);
                        let col_n   = if old_len > 1e-8 { col / old_len } else { col };
                        let new_col = col_n * new_len;

                        let cols = [
                            if ci == 0 { new_col } else { initial_transform.col(0) },
                            if ci == 1 { new_col } else { initial_transform.col(1) },
                            if ci == 2 { new_col } else { initial_transform.col(2) },
                            initial_transform.col(3),
                        ];
                        Mat4::from_cols(cols[0], cols[1], cols[2], cols[3])
                    }

                    GizmoMode::Rotate => {
                        let hit = match ray_plane_hit(ray_o, ray_d, center, axis_dir) {
                            Some(h) => h,
                            None    => return,
                        };
                        let (tan, bitan) = ring_frame(axis, local_axes);
                        let to_hit      = hit - center;
                        let angle_now   = to_hit.dot(bitan).atan2(to_hit.dot(tan));
                        let angle_delta = angle_now - axis_t_start;

                        let rot       = Mat3::from_axis_angle(axis_dir, angle_delta);
                        let upper     = Mat3::from_mat4(initial_transform);
                        let new_upper = rot * upper;
                        Mat4::from_cols(
                            new_upper.col(0).extend(0.0),
                            new_upper.col(1).extend(0.0),
                            new_upper.col(2).extend(0.0),
                            initial_transform.col(3),
                        )
                    }
                };

                let _ = scene.update_object_transform(object_id, new_transform);
            }
            Some(SceneActorId::SectionedObject(inst_id)) => {
                let new_transform = match self.gizmo_mode {
                    GizmoMode::Translate => {
                        let t_now = match ray_to_axis_t(ray_o, ray_d, center, axis_dir) {
                            Some(t) => t,
                            None    => return,
                        };
                        let delta = t_now - axis_t_start;
                        Mat4::from_translation(axis_dir * delta) * initial_transform
                    }

                    GizmoMode::Scale => {
                        let t_now = match ray_to_axis_t(ray_o, ray_d, center, axis_dir) {
                            Some(t) => t,
                            None    => return,
                        };
                        let delta        = t_now - axis_t_start;
                        let sensitivity  = 1.5 / world_size.max(0.01);
                        let scale_factor = (1.0 + delta * sensitivity).max(0.01_f32);

                        let ci      = axis.col();
                        let col     = initial_transform.col(ci);
                        let old_len = col.truncate().length();
                        let new_len = (old_len * scale_factor).max(0.001);
                        let col_n   = if old_len > 1e-8 { col / old_len } else { col };
                        let new_col = col_n * new_len;

                        let cols = [
                            if ci == 0 { new_col } else { initial_transform.col(0) },
                            if ci == 1 { new_col } else { initial_transform.col(1) },
                            if ci == 2 { new_col } else { initial_transform.col(2) },
                            initial_transform.col(3),
                        ];
                        Mat4::from_cols(cols[0], cols[1], cols[2], cols[3])
                    }

                    GizmoMode::Rotate => {
                        let hit = match ray_plane_hit(ray_o, ray_d, center, axis_dir) {
                            Some(h) => h,
                            None    => return,
                        };
                        let (tan, bitan) = ring_frame(axis, local_axes);
                        let to_hit      = hit - center;
                        let angle_now   = to_hit.dot(bitan).atan2(to_hit.dot(tan));
                        let angle_delta = angle_now - axis_t_start;

                        let rot       = Mat3::from_axis_angle(axis_dir, angle_delta);
                        let upper     = Mat3::from_mat4(initial_transform);
                        let new_upper = rot * upper;
                        Mat4::from_cols(
                            new_upper.col(0).extend(0.0),
                            new_upper.col(1).extend(0.0),
                            new_upper.col(2).extend(0.0),
                            initial_transform.col(3),
                        )
                    }
                };

                let _ = scene.update_sectioned_object_transform(inst_id, new_transform);
            }
            Some(SceneActorId::Light(light_id)) => {
                let Some(t_now) = ray_to_axis_t(ray_o, ray_d, center, axis_dir) else { return };
                let delta    = t_now - axis_t_start;
                let new_pos  = initial_transform.col(3).truncate() + axis_dir * delta;

                let Some(mut light) = scene.get_light(light_id) else { return };
                light.position_range[0] = new_pos.x;
                light.position_range[1] = new_pos.y;
                light.position_range[2] = new_pos.z;
                let _ = scene.update_light(light_id, light);
            }
            _ => {}
        }
    }

    /// Finish the current drag (call on left-button release).
    pub fn end_drag(&mut self) {
        self.drag = DragState::Idle;
    }

    // ── Internal helpers for sibling modules ──────────────────────────

    pub(crate) fn clear_interaction_state(&mut self) {
        self.hovered_axis = None;
        self.drag = DragState::Idle;
    }

    pub(crate) fn take_selected(&mut self) -> Option<SceneActorId> {
        self.selected.take()
    }

    pub(crate) fn replace_selected(&mut self, id: Option<SceneActorId>) {
        self.selected = id;
    }

    // ── Legacy sphere-based pick ──────────────────────────────────────────────

    /// Cast `(ray_origin, ray_dir)` against all objects' bounding spheres.
    ///
    /// Selects the closest intersected object, or clears the selection.
    /// Prefer using `ScenePicker::cast_ray` + `select()` for accurate BVH picking.
    pub fn pick(&mut self, scene: &Scene, ray_origin: Vec3, ray_dir: Vec3) {
        let mut best_t  = f32::MAX;
        let mut best_id: Option<SceneActorId> = None;

        for (id, _transform, bounds, _tag) in scene.iter_objects_for_editor() {
            let center = Vec3::new(bounds[0], bounds[1], bounds[2]);
            let radius = bounds[3];
            if let Some(t) = ray_sphere_intersect(ray_origin, ray_dir, center, radius) {
                if t < best_t {
                    best_t  = t;
                    best_id = Some(SceneActorId::Object(id));
                }
            }
        }

        self.selected     = best_id;
        self.hovered_axis = None;
        self.drag         = DragState::Idle;
    }

    // ── Gizmo rendering ───────────────────────────────────────────────────────

    /// Draw the selection highlight and active transform gizmo.
    ///
    /// Call every frame after [`Renderer::debug_clear`] and before frame submission.
    /// No-op when nothing is selected.
    ///
    /// The gizmo is sized to a constant screen-space footprint (~80 px) using the
    /// camera and viewport height previously set via [`Renderer::set_gizmo_camera`].
    pub fn draw_gizmos(&self, renderer: &mut Renderer) {
        let Some((gizmo_camera, viewport_height)) = renderer.gizmo_camera_info() else { return };
        match self.selected {
            Some(SceneActorId::Object(id)) => {
                let Some((center, _gizmo_size, local_axes)) =
                    object_gizmo_info(id, renderer.scene()) else { return };

                let world_size = gizmo_world_size(center, gizmo_camera, viewport_height);
                let sphere_radius = world_size * 0.5;
                let hov = self.hovered_axis;
                let mode = self.gizmo_mode;
                renderer.debug_batch(|dbg| {
                    dbg.sphere(center.to_array(), sphere_radius, [1.0, 0.95, 0.0, 1.0], 24);
                    match mode {
                        GizmoMode::Translate => draw_translate_gizmo(dbg, center, world_size, hov, local_axes),
                        GizmoMode::Rotate    => draw_rotate_gizmo   (dbg, center, world_size, hov, local_axes),
                        GizmoMode::Scale     => draw_scale_gizmo    (dbg, center, world_size, hov, local_axes),
                    }
                });
            }
            Some(SceneActorId::SectionedObject(id)) => {
                let Some((center, _gizmo_size, local_axes)) =
                    sectioned_gizmo_info(id, renderer.scene()) else { return };
                let world_size = gizmo_world_size(center, gizmo_camera, viewport_height);
                let sphere_radius = world_size * 0.5;
                let hov  = self.hovered_axis;
                let mode = self.gizmo_mode;
                renderer.debug_batch(|dbg| {
                    dbg.sphere(center.to_array(), sphere_radius, [1.0, 0.95, 0.0, 1.0], 24);
                    match mode {
                        GizmoMode::Translate => draw_translate_gizmo(dbg, center, world_size, hov, local_axes),
                        GizmoMode::Rotate    => draw_rotate_gizmo   (dbg, center, world_size, hov, local_axes),
                        GizmoMode::Scale     => draw_scale_gizmo    (dbg, center, world_size, hov, local_axes),
                    }
                });
            }
            Some(SceneActorId::Light(id)) => {
                let Some(light) = renderer.scene().get_light(id) else { return };
                let center = Vec3::new(
                    light.position_range[0],
                    light.position_range[1],
                    light.position_range[2],
                );
                let world_size = gizmo_world_size(center, gizmo_camera, viewport_height);
                let local_axes = [Vec3::X, Vec3::Y, Vec3::Z];
                let hovered_axis = self.hovered_axis;
                renderer.debug_batch(|dbg| {
                    dbg.sphere(center.to_array(), world_size * 0.5, [1.0, 0.95, 0.0, 1.0], 24);
                    draw_translate_gizmo(dbg, center, world_size, hovered_axis, local_axes);
                });
            }
            _ => {}
        }
    }
}
