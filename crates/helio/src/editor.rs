//! Editor-mode utilities: object selection and transform gizmos.
//!
//! # Overview
//!
//! [`EditorState`] is a lightweight runtime struct you keep in your application.
//! Each frame call [`EditorState::draw_gizmos`] to overlay transform handles on
//! the selected object. On left-click call [`EditorState::pick`] to select the
//! nearest object under the cursor.
//!
//! # Gizmo modes
//!
//! | Mode | Key (demo) | Visual |
//! |------|-----------|--------|
//! | [`GizmoMode::Translate`] | G | XYZ arrows with cone tips |
//! | [`GizmoMode::Rotate`]    | R | XYZ circles (rings) |
//! | [`GizmoMode::Scale`]     | S | XYZ axes with box end-caps |
//!
//! # Picking
//!
//! [`EditorState::pick`] performs a CPU-side ray vs bounding-sphere sweep over
//! all scene objects.  It is O(N) where N is the live object count; do not call
//! it every frame — only call it on click events.
//!
//! # Example
//!
//! ```ignore
//! use helio::{EditorState, GizmoMode};
//!
//! // In your AppState:
//! let mut editor = EditorState::new();
//!
//! // On left-click (cursor not grabbed):
//! let (ray_o, ray_d) = EditorState::ray_from_screen(
//!     mouse_x, mouse_y, width, height, view_proj_inv,
//! );
//! editor.pick(renderer.scene(), ray_o, ray_d);
//!
//! // Switch gizmo with keyboard:
//! if keys.contains(&KeyCode::KeyG) { editor.set_gizmo_mode(GizmoMode::Translate); }
//! if keys.contains(&KeyCode::KeyR) { editor.set_gizmo_mode(GizmoMode::Rotate); }
//! if keys.contains(&KeyCode::KeyS) { editor.set_gizmo_mode(GizmoMode::Scale); }
//!
//! // Every frame, after debug_clear():
//! editor.draw_gizmos(&mut renderer);
//! ```

use glam::{Mat4, Vec3};

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

/// Per-frame editor state.
///
/// Holds the currently-selected object and active gizmo mode. Call
/// [`draw_gizmos`](EditorState::draw_gizmos) every frame and
/// [`pick`](EditorState::pick) on click events.
pub struct EditorState {
    selected: Option<ObjectId>,
    gizmo_mode: GizmoMode,
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
            gizmo_mode: GizmoMode::Translate,
        }
    }

    // ── Selection ────────────────────────────────────────────────────────────

    /// Explicitly select an object by handle.
    pub fn select(&mut self, id: ObjectId) {
        self.selected = Some(id);
    }

    /// Clear the current selection.
    pub fn deselect(&mut self) {
        self.selected = None;
    }

    /// Returns the currently selected object, if any.
    pub fn selected(&self) -> Option<ObjectId> {
        self.selected
    }

    // ── Gizmo mode ───────────────────────────────────────────────────────────

    /// Returns the active gizmo mode.
    pub fn gizmo_mode(&self) -> GizmoMode {
        self.gizmo_mode
    }

    /// Switch the active gizmo mode.
    pub fn set_gizmo_mode(&mut self, mode: GizmoMode) {
        self.gizmo_mode = mode;
    }

    // ── Ray unprojection ─────────────────────────────────────────────────────

    /// Convert a screen-space pixel coordinate to a world-space ray.
    ///
    /// # Parameters
    /// - `px`, `py` — pixel coordinate in `[0 .. width)` × `[0 .. height)`
    /// - `width`, `height` — framebuffer size
    /// - `view_proj_inv` — inverse of `proj * view` for the current frame
    ///
    /// # Returns
    /// `(ray_origin, ray_direction)` — both in world space, direction normalized.
    pub fn ray_from_screen(
        px: f32,
        py: f32,
        width: f32,
        height: f32,
        view_proj_inv: Mat4,
    ) -> (Vec3, Vec3) {
        let ndc_x = (px / width) * 2.0 - 1.0;
        let ndc_y = 1.0 - (py / height) * 2.0;
        let near = view_proj_inv.project_point3(Vec3::new(ndc_x, ndc_y, 0.0));
        let far  = view_proj_inv.project_point3(Vec3::new(ndc_x, ndc_y, 1.0));
        let dir = (far - near).normalize_or_zero();
        (near, dir)
    }

    // ── Picking ───────────────────────────────────────────────────────────────

    /// Cast `(ray_origin, ray_dir)` against all objects' bounding spheres.
    ///
    /// Selects the closest intersected object, or clears the selection if no
    /// object was hit.  The scene's bounding sphere is in world space
    /// (`[cx, cy, cz, radius]`).
    ///
    /// Call this on click events only — it is O(N) over live objects.
    pub fn pick(&mut self, scene: &Scene, ray_origin: Vec3, ray_dir: Vec3) {
        let mut best_t = f32::MAX;
        let mut best_id = None;

        for (id, _transform, bounds) in scene.iter_objects_for_editor() {
            let center = Vec3::new(bounds[0], bounds[1], bounds[2]);
            let radius = bounds[3];
            if let Some(t) = ray_sphere_intersect(ray_origin, ray_dir, center, radius) {
                if t < best_t {
                    best_t = t;
                    best_id = Some(id);
                }
            }
        }

        self.selected = best_id;
    }

    // ── Gizmo rendering ───────────────────────────────────────────────────────

    /// Draw the selection highlight and active transform gizmo for the selected object.
    ///
    /// Call this every frame after [`Renderer::debug_clear`] and before
    /// submitting the frame.  No-op when nothing is selected.
    pub fn draw_gizmos(&self, renderer: &mut Renderer) {
        let Some(id) = self.selected else { return };

        // Gather data under an immutable borrow that ends before any debug draw.
        let bounds = match renderer.scene().get_object_bounds(id) {
            Ok(b) => b,
            Err(_) => return,
        };
        let center = Vec3::new(bounds[0], bounds[1], bounds[2]);
        let radius = bounds[3].max(0.3_f32);

        let gizmo_size: f32 = (radius * 1.8_f32).max(0.8_f32);

        // Selection highlight: thin yellow sphere wrapped around the object.
        renderer.debug_sphere(center.to_array(), radius * 1.08_f32, [1.0, 0.95, 0.0, 1.0], 24);

        match self.gizmo_mode {
            GizmoMode::Translate => draw_translate_gizmo(renderer, center, gizmo_size),
            GizmoMode::Rotate    => draw_rotate_gizmo(renderer, center, gizmo_size),
            GizmoMode::Scale     => draw_scale_gizmo(renderer, center, gizmo_size),
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Internal math helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Ray vs sphere intersection.  Returns the first positive `t` along the ray,
/// or `None` if the ray misses or the sphere is fully behind the origin.
fn ray_sphere_intersect(origin: Vec3, dir: Vec3, center: Vec3, radius: f32) -> Option<f32> {
    let oc = origin - center;
    let b = oc.dot(dir);
    let c = oc.dot(oc) - radius * radius;
    let disc = b * b - c;
    if disc < 0.0 {
        return None;
    }
    let sqrt_disc = disc.sqrt();
    let t0 = -b - sqrt_disc;
    let t1 = -b + sqrt_disc;
    if t1 < 0.0 {
        return None;
    }
    Some(if t0 >= 0.0 { t0 } else { t1 })
}

// ─────────────────────────────────────────────────────────────────────────────
// Gizmo drawing helpers
// ─────────────────────────────────────────────────────────────────────────────

const RED:   [f32; 4] = [1.00, 0.15, 0.15, 1.0];
const GREEN: [f32; 4] = [0.15, 1.00, 0.15, 1.0];
const BLUE:  [f32; 4] = [0.15, 0.35, 1.00, 1.0];

/// Translate gizmo: three coloured axes with cone arrow-heads.
///
/// ```text
/// Red   arrow → +X
/// Green arrow → +Y
/// Blue  arrow → +Z
/// ```
fn draw_translate_gizmo(renderer: &mut Renderer, center: Vec3, size: f32) {
    let shaft   = size * 0.75;
    let cone_h  = size * 0.22;
    let cone_r  = size * 0.06;

    // +X  (red)
    let x_end = center + Vec3::X * shaft;
    renderer.debug_line(center.to_array(), x_end.to_array(), RED);
    renderer.debug_cone(
        (x_end + Vec3::X * cone_h).to_array(),
        [-1.0, 0.0, 0.0],
        cone_h, cone_r, RED, 12,
    );

    // +Y  (green)
    let y_end = center + Vec3::Y * shaft;
    renderer.debug_line(center.to_array(), y_end.to_array(), GREEN);
    renderer.debug_cone(
        (y_end + Vec3::Y * cone_h).to_array(),
        [0.0, -1.0, 0.0],
        cone_h, cone_r, GREEN, 12,
    );

    // +Z  (blue)
    let z_end = center + Vec3::Z * shaft;
    renderer.debug_line(center.to_array(), z_end.to_array(), BLUE);
    renderer.debug_cone(
        (z_end + Vec3::Z * cone_h).to_array(),
        [0.0, 0.0, -1.0],
        cone_h, cone_r, BLUE, 12,
    );
}

/// Rotate gizmo: one ring per axis, drawn in the plane perpendicular to that axis.
///
/// ```text
/// Red   ring → rotates around X (ring lies in YZ plane)
/// Green ring → rotates around Y (ring lies in XZ plane)
/// Blue  ring → rotates around Z (ring lies in XY plane)
/// ```
fn draw_rotate_gizmo(renderer: &mut Renderer, center: Vec3, size: f32) {
    const SEGS: u32 = 48;
    // Ring around X: tangent=Y, bitangent=Z
    draw_ring(renderer, center, Vec3::Y, Vec3::Z, size, RED,   SEGS);
    // Ring around Y: tangent=X, bitangent=Z
    draw_ring(renderer, center, Vec3::X, Vec3::Z, size, GREEN, SEGS);
    // Ring around Z: tangent=X, bitangent=Y
    draw_ring(renderer, center, Vec3::X, Vec3::Y, size, BLUE,  SEGS);
}

/// Draw a planar ring using two perpendicular tangent vectors.
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
        let next = center + (tangent * theta.cos() + bitangent * theta.sin()) * radius;
        renderer.debug_line(prev.to_array(), next.to_array(), color);
        prev = next;
    }
}

/// Scale gizmo: three coloured axes with cube end-caps.
///
/// ```text
/// Red   axis → +X
/// Green axis → +Y
/// Blue  axis → +Z
/// ```
fn draw_scale_gizmo(renderer: &mut Renderer, center: Vec3, size: f32) {
    let shaft    = size * 0.82;
    let box_half = size * 0.07;

    let x_end = center + Vec3::X * shaft;
    renderer.debug_line(center.to_array(), x_end.to_array(), RED);
    draw_box_marker(renderer, x_end, box_half, RED);

    let y_end = center + Vec3::Y * shaft;
    renderer.debug_line(center.to_array(), y_end.to_array(), GREEN);
    draw_box_marker(renderer, y_end, box_half, GREEN);

    let z_end = center + Vec3::Z * shaft;
    renderer.debug_line(center.to_array(), z_end.to_array(), BLUE);
    draw_box_marker(renderer, z_end, box_half, BLUE);
}

/// Draw a small wire-frame cube centred at `center` with the given `half` size.
fn draw_box_marker(renderer: &mut Renderer, center: Vec3, half: f32, color: [f32; 4]) {
    let c = [
        center + Vec3::new(-half, -half, -half), // 0 lbf
        center + Vec3::new( half, -half, -half), // 1 rbf
        center + Vec3::new( half,  half, -half), // 2 rtf
        center + Vec3::new(-half,  half, -half), // 3 ltf
        center + Vec3::new(-half, -half,  half), // 4 lbn
        center + Vec3::new( half, -half,  half), // 5 rbn
        center + Vec3::new( half,  half,  half), // 6 rtn
        center + Vec3::new(-half,  half,  half), // 7 ltn
    ];
    // Four bottom edges, four top edges, four vertical pillars.
    for i in 0..4 {
        renderer.debug_line(c[i].to_array(),     c[(i + 1) % 4].to_array(),     color);
        renderer.debug_line(c[i + 4].to_array(), c[(i + 1) % 4 + 4].to_array(), color);
        renderer.debug_line(c[i].to_array(),     c[i + 4].to_array(),            color);
    }
}
