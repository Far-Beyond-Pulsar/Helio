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

mod commands;
mod gizmo;
mod state;

pub use state::EditorState;

use glam::Vec3;

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
    pub(crate) fn col(self) -> usize {
        match self {
            GizmoAxis::X => 0,
            GizmoAxis::Y => 1,
            GizmoAxis::Z => 2,
        }
    }
}

/// Given a set of local axes and the axis we are rotating around, return the
/// two tangent vectors that span the ring plane.
pub(crate) fn ring_frame(axis: GizmoAxis, local_axes: [Vec3; 3]) -> (Vec3, Vec3) {
    match axis {
        GizmoAxis::X => (local_axes[1], local_axes[2]),
        GizmoAxis::Y => (local_axes[2], local_axes[0]),
        GizmoAxis::Z => (local_axes[0], local_axes[1]),
    }
}
