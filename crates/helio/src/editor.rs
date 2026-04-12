//! Editor-mode utilities: object selection and interactive transform gizmos.
//!
//! # Overview
//!
//! [`EditorState`] tracks selection, gizmo mode, hover, and drag state.
//! Each frame call [`EditorState::draw_gizmos`] to overlay filled transform
//! handles on the selected object. On cursor move call [`EditorState::update_hover`]
//! so handles highlight when the cursor is near them. On left-click call
//! [`EditorState::try_start_drag`] first — if it returns `true` the user
//! clicked a gizmo handle; otherwise fall through to normal object picking.
//! While the left button is held call [`EditorState::update_drag`] every frame
//! to move / rotate / scale the object. On release call [`EditorState::end_drag`].
//!
//! # Gizmo modes
//!
//! | Mode | Key (demo) | Visual |
//! |------|-----------|--------|
//! | [`GizmoMode::Translate`] | G | XYZ arrows — shaft + filled cone tip |
//! | [`GizmoMode::Rotate`]    | R | XYZ rings — line ring, hover highlights |
//! | [`GizmoMode::Scale`]     | S | XYZ axes — shaft + filled box end-cap |
//!
//! # Interactive controls (wired up in editor_demo.rs)
//!
//! 1. `update_hover(ray_o, ray_d, scene)` on every `CursorMoved` when cursor free.
//! 2. On left-click press: `try_start_drag(ray_o, ray_d, scene)`.
//!    - Returns `true`  → drag started, suppress normal pick.
//!    - Returns `false` → forward to `ScenePicker::cast_ray` for object selection.
//! 3. On `CursorMoved` while dragging: `update_drag(ray_o, ray_d, scene_mut)`.
//! 4. On left-click release: `end_drag()`.

use glam::{Mat3, Mat4, Vec3};

use crate::handles::ObjectId;
use crate::renderer::Renderer;
use crate::scene::Scene;

// ─────────────────────────────────────────────────────────────────────────────
// Public types
// ─────────────────────────────────────────────────────────────────────────────

/// Which transform handle to display for the selected object.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum GizmoMode {
    /// Show location (translation) arrows.
    #[default]
    Translate,
    /// Show rotation rings.
    Rotate,
    /// Show scale handles (box end-caps).
    Scale,
}

/// Which axis of the active transform gizmo is hovered or being dragged.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GizmoAxis {
    X,
    Y,
    Z,
}

impl GizmoAxis {
    /// Column index in a matrix / local_axes array (0 = X, 1 = Y, 2 = Z).
    fn col(self) -> usize {
        match self {
            GizmoAxis::X => 0,
            GizmoAxis::Y => 1,
            GizmoAxis::Z => 2,
        }
    }
}

/// Given a set of local axes and the axis we are rotating around, return the
/// two tangent vectors that span the ring plane.
fn ring_frame(axis: GizmoAxis, local_axes: [Vec3; 3]) -> (Vec3, Vec3) {
    // Each pair must satisfy (tan × bitan) == axis_dir to keep right-handed winding.
    //   X: Y × Z = X  ✓
    //   Y: Z × X = Y  ✓  (reversed from X/Z order — X × Z = −Y which inverts drag)
    //   Z: X × Y = Z  ✓
    match axis {
        GizmoAxis::X => (local_axes[1], local_axes[2]),
        GizmoAxis::Y => (local_axes[2], local_axes[0]),
        GizmoAxis::Z => (local_axes[0], local_axes[1]),
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Internal drag state
// ─────────────────────────────────────────────────────────────────────────────

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
    selected: Option<ObjectId>,
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

    /// Explicitly select an object by handle.
    pub fn select(&mut self, id: ObjectId) {
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

    /// Returns the currently selected object, if any.
    pub fn selected(&self) -> Option<ObjectId> {
        self.selected
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
    pub fn update_hover(&mut self, ray_o: Vec3, ray_d: Vec3, scene: &Scene) -> bool {
        if self.is_dragging() {
            return true; // keep hover alive while dragging
        }
        let Some(id) = self.selected else {
            self.hovered_axis = None;
            return false;
        };
        let Some((center, size, local_axes)) = object_gizmo_info(id, scene) else {
            self.hovered_axis = None;
            return false;
        };
        self.hovered_axis = hit_gizmo(ray_o, ray_d, center, size, self.gizmo_mode, local_axes);
        self.hovered_axis.is_some()
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
        let Some(id) = self.selected else { return false };
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

        // Freeze centre and local axes so update_drag always measures against the
        // same reference regardless of how the object moves.
        self.drag = DragState::Active {
            axis, initial_transform, gizmo_center: center, local_axes, axis_t_start,
        };
        true
    }

    /// Apply the gizmo drag to the selected object given the current ray.
    ///
    /// Call on every `CursorMoved` event while the left button is held.
    pub fn update_drag(&mut self, ray_o: Vec3, ray_d: Vec3, scene: &mut Scene) {
        let (axis, initial_transform, gizmo_center, local_axes, axis_t_start, gizmo_size) =
            match &self.drag {
                DragState::Active {
                    axis, initial_transform, gizmo_center, local_axes, axis_t_start,
                } => {
                    let id   = match self.selected { Some(id) => id, None => return };
                    let size = scene.get_object_bounds(id)
                        .map(|b| (b[3].max(0.3) * 1.8).max(0.8))
                        .unwrap_or(1.0);
                    (*axis, *initial_transform, *gizmo_center, *local_axes, *axis_t_start, size)
                }
                DragState::Idle => return,
            };
        let Some(id) = self.selected else { return };

        // Use the frozen local axis direction and frozen gizmo centre so the
        // constraint doesn't shift as the object moves.
        let axis_dir = local_axes[axis.col()];
        let center   = gizmo_center;

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
                // World-unit drag → scale fraction relative to gizmo size.
                let delta        = t_now - axis_t_start;
                let sensitivity  = 1.5 / gizmo_size.max(0.01);
                let scale_factor = (1.0 + delta * sensitivity).max(0.01_f32);

                // Re-scale the column that corresponds to this axis.
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

                // Rotate orientation/scale columns, keep translation.
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

        let _ = scene.update_object_transform(id, new_transform);
    }

    /// Finish the current drag (call on left-button release).
    pub fn end_drag(&mut self) {
        self.drag = DragState::Idle;
    }

    // ── Legacy sphere-based pick ──────────────────────────────────────────────

    /// Cast `(ray_origin, ray_dir)` against all objects' bounding spheres.
    ///
    /// Selects the closest intersected object, or clears the selection.
    /// Prefer using `ScenePicker::cast_ray` + `select()` for accurate BVH picking.
    pub fn pick(&mut self, scene: &Scene, ray_origin: Vec3, ray_dir: Vec3) {
        let mut best_t  = f32::MAX;
        let mut best_id = None;

        for (id, _transform, bounds) in scene.iter_objects_for_editor() {
            let center = Vec3::new(bounds[0], bounds[1], bounds[2]);
            let radius = bounds[3];
            if let Some(t) = ray_sphere_intersect(ray_origin, ray_dir, center, radius) {
                if t < best_t {
                    best_t  = t;
                    best_id = Some(id);
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
    pub fn draw_gizmos(&self, renderer: &mut Renderer) {
        let Some(id) = self.selected else { return };

        let Some((center, gizmo_size, local_axes)) =
            object_gizmo_info(id, renderer.scene()) else { return };

        // Selection highlight: wire sphere at the transform origin (center), radius from bounds.
        let sphere_radius = renderer.scene().get_object_bounds(id)
            .map(|b| b[3].max(0.3_f32))
            .unwrap_or(0.3_f32);
        renderer.debug_sphere(center.to_array(), sphere_radius * 1.08_f32, [1.0, 0.95, 0.0, 1.0], 24);

        let hov = self.hovered_axis;
        match self.gizmo_mode {
            GizmoMode::Translate => draw_translate_gizmo(renderer, center, gizmo_size, hov, local_axes),
            GizmoMode::Rotate    => draw_rotate_gizmo   (renderer, center, gizmo_size, hov, local_axes),
            GizmoMode::Scale     => draw_scale_gizmo    (renderer, center, gizmo_size, hov, local_axes),
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Color helpers
// ─────────────────────────────────────────────────────────────────────────────

const RED:   [f32; 4] = [1.00, 0.15, 0.15, 1.0];
const GREEN: [f32; 4] = [0.15, 1.00, 0.15, 1.0];
const BLUE:  [f32; 4] = [0.15, 0.35, 1.00, 1.0];

/// Bright gold-yellow used for hovered/active handles (Blender convention).
const HOVER: [f32; 4] = [1.0, 0.85, 0.05, 1.0];

/// Solid color for a handle, switching to HOVER when hovered.
fn line_col(base: [f32; 4], hovered: bool) -> [f32; 4] {
    if hovered { HOVER } else { base }
}

/// Semi-transparent fill color for a handle (alpha 0.35 normal, 0.65 hovered).
fn fill_col(base: [f32; 4], hovered: bool) -> [f32; 4] {
    if hovered {
        [HOVER[0], HOVER[1], HOVER[2], 0.65]
    } else {
        [base[0], base[1], base[2], 0.35]
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Gizmo drawing helpers
// ─────────────────────────────────────────────────────────────────────────────

fn draw_translate_gizmo(
    renderer: &mut Renderer,
    center: Vec3,
    size: f32,
    hovered: Option<GizmoAxis>,
    local_axes: [Vec3; 3],
) {
    let shaft  = size * 0.75;
    let cone_h = size * 0.22;
    let cone_r = size * 0.06;
    const SEGS: u32 = 16;

    for (axis, base) in [(GizmoAxis::X, RED), (GizmoAxis::Y, GREEN), (GizmoAxis::Z, BLUE)] {
        let h      = local_axes[axis.col()];
        let tip    = center + h * shaft;
        let apex   = tip + h * cone_h;
        let is_hov = hovered == Some(axis);
        let lc     = line_col(base, is_hov);
        let fc     = fill_col(base, is_hov);

        renderer.debug_line(center.to_array(), tip.to_array(), lc);
        renderer.debug_filled_cone(apex.to_array(), (-h).to_array(), cone_h, cone_r, fc, SEGS);
        renderer.debug_cone       (apex.to_array(), (-h).to_array(), cone_h, cone_r, lc, SEGS);
    }
}

fn draw_rotate_gizmo(
    renderer: &mut Renderer,
    center: Vec3,
    size: f32,
    hovered: Option<GizmoAxis>,
    local_axes: [Vec3; 3],
) {
    const SEGS: u32 = 64;
    const BAND: f32 = 0.055; // annulus half-width as fraction of gizmo size

    for (axis, base) in [(GizmoAxis::X, RED), (GizmoAxis::Y, GREEN), (GizmoAxis::Z, BLUE)] {
        let (tan, bitan) = ring_frame(axis, local_axes);
        let is_hov       = hovered == Some(axis);
        let lc           = line_col(base, is_hov);
        let fc           = fill_col(base, is_hov);

        draw_ring(renderer, center, tan, bitan, size, lc, SEGS);

        let inner = size * (1.0 - BAND);
        let outer = size * (1.0 + BAND);
        draw_annulus(renderer, center, tan, bitan, inner, outer, fc, SEGS);
    }
}

fn draw_scale_gizmo(
    renderer: &mut Renderer,
    center: Vec3,
    size: f32,
    hovered: Option<GizmoAxis>,
    local_axes: [Vec3; 3],
) {
    let shaft    = size * 0.82;
    let box_half = size * 0.07;

    for (axis, base) in [(GizmoAxis::X, RED), (GizmoAxis::Y, GREEN), (GizmoAxis::Z, BLUE)] {
        let h      = local_axes[axis.col()];
        let tip    = center + h * shaft;
        let is_hov = hovered == Some(axis);
        let lc     = line_col(base, is_hov);
        let fc     = fill_col(base, is_hov);

        renderer.debug_line(center.to_array(), tip.to_array(), lc);
        renderer.debug_filled_box(tip.to_array(), box_half, fc);
        draw_box_wire(renderer, tip, box_half, lc);
    }
}

// ── Ring / annulus helpers ───────────────────────────────────────────────────

fn draw_ring(
    renderer: &mut Renderer,
    center: Vec3,
    tangent: Vec3,
    bitangent: Vec3,
    radius: f32,
    color: [f32; 4],
    segs: u32,
) {
    let step = std::f32::consts::TAU / segs as f32;
    let mut prev = center + tangent * radius;
    for i in 1..=segs {
        let theta = i as f32 * step;
        let next  = center + (tangent * theta.cos() + bitangent * theta.sin()) * radius;
        renderer.debug_line(prev.to_array(), next.to_array(), color);
        prev = next;
    }
}

/// Filled flat annulus (ring band) using triangle quads per sector.
fn draw_annulus(
    renderer: &mut Renderer,
    center: Vec3,
    tangent: Vec3,
    bitangent: Vec3,
    inner: f32,
    outer: f32,
    color: [f32; 4],
    segs: u32,
) {
    let step = std::f32::consts::TAU / segs as f32;
    let mut pi = center + tangent * inner;
    let mut po = center + tangent * outer;
    for i in 1..=segs {
        let theta = i as f32 * step;
        let dir   = tangent * theta.cos() + bitangent * theta.sin();
        let ci    = center + dir * inner;
        let co    = center + dir * outer;
        renderer.debug_tri(po.to_array(), pi.to_array(), ci.to_array(), color);
        renderer.debug_tri(po.to_array(), ci.to_array(), co.to_array(), color);
        pi = ci;
        po = co;
    }
}

// ── Box helpers ──────────────────────────────────────────────────────────────

/// 12-edge wireframe cube.
fn draw_box_wire(renderer: &mut Renderer, center: Vec3, half: f32, color: [f32; 4]) {
    let c = [
        center + Vec3::new(-half, -half, -half),
        center + Vec3::new( half, -half, -half),
        center + Vec3::new( half,  half, -half),
        center + Vec3::new(-half,  half, -half),
        center + Vec3::new(-half, -half,  half),
        center + Vec3::new( half, -half,  half),
        center + Vec3::new( half,  half,  half),
        center + Vec3::new(-half,  half,  half),
    ];
    for i in 0..4 {
        renderer.debug_line(c[i].to_array(),     c[(i + 1) % 4].to_array(),     color);
        renderer.debug_line(c[i + 4].to_array(), c[(i + 1) % 4 + 4].to_array(), color);
        renderer.debug_line(c[i].to_array(),     c[i + 4].to_array(),            color);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Gizmo hit-testing
// ─────────────────────────────────────────────────────────────────────────────

/// Returns `(gizmo_center, size, local_axes)` for the given object.
///
/// * `gizmo_center` — world-space transform origin (`col(3).xyz`), i.e. the
///   object's pivot point.  Deliberately *not* the bounding-sphere centroid so
///   the gizmo sits exactly at the object's transform origin.
/// * `size` — uniform visual scale derived from the bounding-sphere radius.
/// * `local_axes` — normalized columns 0-2 of the object's world transform.
///   These match the directions the gizmo handles point in so that all drawing,
///   hit-testing, and drag math automatically follows the object's orientation.
fn object_gizmo_info(id: ObjectId, scene: &Scene) -> Option<(Vec3, f32, [Vec3; 3])> {
    let transform = scene.get_object_transform(id).ok()?;
    let bounds    = scene.get_object_bounds(id).ok()?;
    // Pivot: world-space position of the object's local origin.
    let center = transform.col(3).truncate();
    // Size from the bounding-sphere radius (independent of orientation).
    let size = (bounds[3].max(0.3) * 1.8).max(0.8);
    // Normalized local axes — these are the directions the gizmo arms point.
    let local_axes = [
        transform.col(0).truncate().normalize_or_zero(),
        transform.col(1).truncate().normalize_or_zero(),
        transform.col(2).truncate().normalize_or_zero(),
    ];
    Some((center, size, local_axes))
}

/// Return which gizmo axis handle (if any) the cursor ray intersects.
fn hit_gizmo(
    ray_o: Vec3,
    ray_d: Vec3,
    center: Vec3,
    size: f32,
    mode: GizmoMode,
    local_axes: [Vec3; 3],
) -> Option<GizmoAxis> {
    let threshold = size * 0.12;

    match mode {
        GizmoMode::Translate | GizmoMode::Scale => {
            let len = if matches!(mode, GizmoMode::Translate) {
                size * (0.75 + 0.22)
            } else {
                size * (0.82 + 0.14)
            };

            let mut best_d = threshold;
            let mut best   = None;
            for axis in [GizmoAxis::X, GizmoAxis::Y, GizmoAxis::Z] {
                let end = center + local_axes[axis.col()] * len;
                let d   = ray_to_segment_dist(ray_o, ray_d, center, end);
                if d < best_d {
                    best_d = d;
                    best   = Some(axis);
                }
            }
            best
        }

        GizmoMode::Rotate => {
            let ring_r    = size;
            let ring_band = size * 0.15;

            let mut best_d = ring_band;
            let mut best   = None;
            for axis in [GizmoAxis::X, GizmoAxis::Y, GizmoAxis::Z] {
                let normal = local_axes[axis.col()];
                let denom  = ray_d.dot(normal);
                if denom.abs() < 1e-6 { continue; }
                let t = (center - ray_o).dot(normal) / denom;
                if t < 0.001 { continue; }
                let hit  = ray_o + ray_d * t;
                let dist = ((hit - center).length() - ring_r).abs();
                if dist < best_d {
                    best_d = dist;
                    best   = Some(axis);
                }
            }
            best
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Math helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Minimum distance from a world ray to a finite line segment.
fn ray_to_segment_dist(ray_o: Vec3, ray_d: Vec3, seg_a: Vec3, seg_b: Vec3) -> f32 {
    // Dan Sunday's line-to-line closest-approach algorithm.
    // w  = ray_o − seg_a  (vector between the two starting points)
    // sc = parameter along segment [0,1]
    // tc = parameter along ray     [0,∞)
    let d     = seg_b - seg_a;
    let w     = ray_o - seg_a;
    let a     = d.dot(d);       // squared length of segment
    let b     = d.dot(ray_d);   // projection of ray_d onto segment dir
    let e     = d.dot(w);       // projection of w onto segment dir
    let f     = ray_d.dot(w);   // projection of w onto ray dir
    let denom = a - b * b;      // = |d|²·|ray_d|²·sin²θ ≥ 0

    let (sc, tc) = if denom.abs() > 1e-8 {
        // sc = (e − b·f) / denom,  tc = (b·e − a·f) / denom  (Sunday 2001)
        let sc = ((e - b * f) / denom).clamp(0.0, 1.0);
        let tc = ((b * e - a * f) / denom).max(0.0);
        (sc, tc)
    } else {
        // Nearly parallel — project ray origin onto segment as fallback.
        (e / a.max(1e-8), 0.0)
    };

    let closest_seg = seg_a + d * sc;
    let closest_ray = ray_o + ray_d * tc;
    (closest_ray - closest_seg).length()
}

/// Signed position along an infinite axis line at the closest approach to the ray.
///
/// Returns `None` when the ray is nearly parallel to the axis.
fn ray_to_axis_t(ray_o: Vec3, ray_d: Vec3, axis_origin: Vec3, axis_dir: Vec3) -> Option<f32> {
    // Returns the parameter s along the axis line such that (axis_origin + s·axis_dir)
    // is the point on the axis closest to the ray.  Derivation: minimise the distance
    // between P = axis_origin + s·axis_dir  and  Q = ray_o + t·ray_d.
    // From ∂/∂s = 0: s·(1 − b²) = d − b·e   (Dan Sunday 2001)
    let w     = ray_o - axis_origin;
    let b     = axis_dir.dot(ray_d);
    let d     = axis_dir.dot(w);
    let e     = ray_d.dot(w);
    let denom = 1.0 - b * b;
    if denom.abs() < 1e-8 { return None; }
    Some((d - b * e) / denom)
}

/// Ray–plane intersection; returns world hit point or `None` if parallel/behind.
fn ray_plane_hit(ray_o: Vec3, ray_d: Vec3, plane_pt: Vec3, plane_n: Vec3) -> Option<Vec3> {
    let denom = ray_d.dot(plane_n);
    if denom.abs() < 1e-6 { return None; }
    let t = (plane_pt - ray_o).dot(plane_n) / denom;
    if t < 0.001 { return None; }
    Some(ray_o + ray_d * t)
}

/// Ray vs sphere — returns first positive `t`, or `None`.
fn ray_sphere_intersect(origin: Vec3, dir: Vec3, center: Vec3, radius: f32) -> Option<f32> {
    let oc   = origin - center;
    let b    = oc.dot(dir);
    let c    = oc.dot(oc) - radius * radius;
    let disc = b * b - c;
    if disc < 0.0 { return None; }
    let sq = disc.sqrt();
    let t0 = -b - sq;
    let t1 = -b + sq;
    if t1 < 0.0 { return None; }
    Some(if t0 >= 0.0 { t0 } else { t1 })
}
