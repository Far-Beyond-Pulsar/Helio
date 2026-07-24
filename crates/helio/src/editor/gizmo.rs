use glam::Vec3;

use super::{ring_frame, GizmoAxis, GizmoMode};
use crate::handles::{ObjectId, SectionedInstanceId};
use crate::renderer::DebugBatch;
use crate::scene::{Camera, Scene};

// ─────────────────────────────────────────────────────────────────────────────
// Color helpers
// ─────────────────────────────────────────────────────────────────────────────

const RED:   [f32; 4] = [1.00, 0.15, 0.15, 1.0];
const GREEN: [f32; 4] = [0.15, 1.00, 0.15, 1.0];
const BLUE:  [f32; 4] = [0.15, 0.35, 1.00, 1.0];

/// Bright gold-yellow used for hovered/active handles (Blender convention).
const HOVER: [f32; 4] = [1.0, 0.85, 0.05, 1.0];

/// Semi-transparent black used for drop-shadow geometry behind gizmo handles.
const SHADOW: [f32; 4] = [0.0, 0.0, 0.0, 0.55];

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

/// World-space offset of ~2 screen pixels at the gizmo center's depth,
/// projected along the camera's right+up vectors.
pub(super) fn shadow_offset(center: Vec3, camera: &Camera, viewport_height: f32) -> Vec3 {
    let pixel_world = gizmo_world_size(center, camera, viewport_height) / GIZMO_PIXELS;
    let off = pixel_world * 2.0;
    camera.view.x_axis.truncate() * off + camera.view.y_axis.truncate() * off
}

pub(super) fn draw_translate_gizmo(
    renderer: &mut DebugBatch<'_>,
    center: Vec3,
    size: f32,
    hovered: Option<GizmoAxis>,
    local_axes: [Vec3; 3],
    off: Vec3,
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

        renderer.line((center + off).to_array(), (tip + off).to_array(), SHADOW);
        renderer.filled_cone((apex + off).to_array(), (-h).to_array(), cone_h, cone_r, SHADOW, SEGS);
        renderer.cone       ((apex + off).to_array(), (-h).to_array(), cone_h, cone_r, SHADOW, SEGS);

        renderer.line(center.to_array(), tip.to_array(), lc);
        renderer.filled_cone(apex.to_array(), (-h).to_array(), cone_h, cone_r, fc, SEGS);
        renderer.cone       (apex.to_array(), (-h).to_array(), cone_h, cone_r, lc, SEGS);
    }
}

pub(super) fn draw_rotate_gizmo(
    renderer: &mut DebugBatch<'_>,
    center: Vec3,
    size: f32,
    hovered: Option<GizmoAxis>,
    local_axes: [Vec3; 3],
    off: Vec3,
) {
    const SEGS: u32 = 64;
    const BAND: f32 = 0.055; // annulus half-width as fraction of gizmo size

    for (axis, base) in [(GizmoAxis::X, RED), (GizmoAxis::Y, GREEN), (GizmoAxis::Z, BLUE)] {
        let (tan, bitan) = ring_frame(axis, local_axes);
        let is_hov       = hovered == Some(axis);
        let lc           = line_col(base, is_hov);
        let fc           = fill_col(base, is_hov);

        draw_ring(renderer, center, tan, bitan, size, SHADOW, SEGS, off);
        let inner = size * (1.0 - BAND);
        let outer = size * (1.0 + BAND);
        draw_annulus(renderer, center, tan, bitan, inner, outer, SHADOW, SEGS, off);

        draw_ring(renderer, center, tan, bitan, size, lc, SEGS, Vec3::ZERO);
        draw_annulus(renderer, center, tan, bitan, inner, outer, fc, SEGS, Vec3::ZERO);
    }
}

pub(super) fn draw_scale_gizmo(
    renderer: &mut DebugBatch<'_>,
    center: Vec3,
    size: f32,
    hovered: Option<GizmoAxis>,
    local_axes: [Vec3; 3],
    off: Vec3,
) {
    let shaft    = size * 0.82;
    let box_half = size * 0.07;

    for (axis, base) in [(GizmoAxis::X, RED), (GizmoAxis::Y, GREEN), (GizmoAxis::Z, BLUE)] {
        let h      = local_axes[axis.col()];
        let tip    = center + h * shaft;
        let is_hov = hovered == Some(axis);
        let lc     = line_col(base, is_hov);
        let fc     = fill_col(base, is_hov);

        renderer.line((center + off).to_array(), (tip + off).to_array(), SHADOW);
        renderer.filled_box((tip + off).to_array(), box_half, SHADOW);
        draw_box_wire(renderer, tip + off, box_half, SHADOW, Vec3::ZERO);

        renderer.line(center.to_array(), tip.to_array(), lc);
        renderer.filled_box(tip.to_array(), box_half, fc);
        draw_box_wire(renderer, tip, box_half, lc, Vec3::ZERO);
    }
}

// ── Ring / annulus helpers ───────────────────────────────────────────────────

#[allow(clippy::too_many_arguments)]
fn draw_ring(
    renderer: &mut DebugBatch<'_>,
    center: Vec3,
    tangent: Vec3,
    bitangent: Vec3,
    radius: f32,
    color: [f32; 4],
    segs: u32,
    offset: Vec3,
) {
    let step = std::f32::consts::TAU / segs as f32;
    let mut prev = center + tangent * radius + offset;
    for i in 1..=segs {
        let theta = i as f32 * step;
        let next  = center + (tangent * theta.cos() + bitangent * theta.sin()) * radius + offset;
        renderer.line(prev.to_array(), next.to_array(), color);
        prev = next;
    }
}

/// Filled flat annulus (ring band) using triangle quads per sector.
#[allow(clippy::too_many_arguments)]
fn draw_annulus(
    renderer: &mut DebugBatch<'_>,
    center: Vec3,
    tangent: Vec3,
    bitangent: Vec3,
    inner: f32,
    outer: f32,
    color: [f32; 4],
    segs: u32,
    offset: Vec3,
) {
    let step = std::f32::consts::TAU / segs as f32;
    let mut pi = center + tangent * inner + offset;
    let mut po = center + tangent * outer + offset;
    for i in 1..=segs {
        let theta = i as f32 * step;
        let dir   = tangent * theta.cos() + bitangent * theta.sin();
        let ci    = center + dir * inner + offset;
        let co    = center + dir * outer + offset;
        renderer.tri(po.to_array(), pi.to_array(), ci.to_array(), color);
        renderer.tri(po.to_array(), ci.to_array(), co.to_array(), color);
        pi = ci;
        po = co;
    }
}

// ── Box helpers ──────────────────────────────────────────────────────────────

/// 12-edge wireframe cube.
fn draw_box_wire(renderer: &mut DebugBatch<'_>, center: Vec3, half: f32, color: [f32; 4], offset: Vec3) {
    let c = [
        center + Vec3::new(-half, -half, -half) + offset,
        center + Vec3::new( half, -half, -half) + offset,
        center + Vec3::new( half,  half, -half) + offset,
        center + Vec3::new(-half,  half, -half) + offset,
        center + Vec3::new(-half, -half,  half) + offset,
        center + Vec3::new( half, -half,  half) + offset,
        center + Vec3::new( half,  half,  half) + offset,
        center + Vec3::new(-half,  half,  half) + offset,
    ];
    for i in 0..4 {
        renderer.line(c[i].to_array(),     c[(i + 1) % 4].to_array(),     color);
        renderer.line(c[i + 4].to_array(), c[(i + 1) % 4 + 4].to_array(), color);
        renderer.line(c[i].to_array(),     c[i + 4].to_array(),            color);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Screen-space gizmo sizing
// ─────────────────────────────────────────────────────────────────────────────

/// Desired gizmo size in screen pixels (translate shaft length, rotate ring
/// radius, scale shaft length).
const GIZMO_PIXELS: f32 = 80.0;

/// Convert a fixed pixel size into a world-space length at `center`'s depth
/// so that gizmos appear identical on screen regardless of camera distance.
pub(super) fn gizmo_world_size(center: Vec3, camera: &Camera, viewport_height: f32) -> f32 {
    let distance = (center - camera.position).length().max(0.001);
    let fy       = camera.proj.col(1)[1]; // 1.0 / tan(fov_y/2)
    GIZMO_PIXELS * 2.0 * distance / (fy * viewport_height)
}

// ─────────────────────────────────────────────────────────────────────────────
// Gizmo info
// ─────────────────────────────────────────────────────────────────────────────

/// Returns `(gizmo_center, size, local_axes)` for the given sectioned instance.
pub(super) fn sectioned_gizmo_info(id: SectionedInstanceId, scene: &Scene) -> Option<(Vec3, f32, [Vec3; 3])> {
    let transform = scene.get_sectioned_instance_transform(id)?;
    let center    = transform.col(3).truncate();
    let local_axes = [
        transform.col(0).truncate().normalize_or_zero(),
        transform.col(1).truncate().normalize_or_zero(),
        transform.col(2).truncate().normalize_or_zero(),
    ];
    Some((center, 1.0, local_axes))
}

/// Returns `(gizmo_center, size, local_axes)` for the given object.
pub(super) fn object_gizmo_info(id: ObjectId, scene: &Scene) -> Option<(Vec3, f32, [Vec3; 3])> {
    let transform = scene.get_object_transform(id).ok()?;
    let center = transform.col(3).truncate();
    let local_axes = [
        transform.col(0).truncate().normalize_or_zero(),
        transform.col(1).truncate().normalize_or_zero(),
        transform.col(2).truncate().normalize_or_zero(),
    ];
    Some((center, 1.0, local_axes))
}

// ─────────────────────────────────────────────────────────────────────────────
// Gizmo hit-testing
// ─────────────────────────────────────────────────────────────────────────────

/// Return which gizmo axis handle (if any) the cursor ray intersects.
pub(super) fn hit_gizmo(
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
    let d     = seg_b - seg_a;
    let w     = ray_o - seg_a;
    let a     = d.dot(d);
    let b     = d.dot(ray_d);
    let e     = d.dot(w);
    let f     = ray_d.dot(w);
    let denom = a - b * b;

    let (sc, tc) = if denom.abs() > 1e-8 {
        let sc = ((e - b * f) / denom).clamp(0.0, 1.0);
        let tc = ((b * e - a * f) / denom).max(0.0);
        (sc, tc)
    } else {
        (e / a.max(1e-8), 0.0)
    };

    let closest_seg = seg_a + d * sc;
    let closest_ray = ray_o + ray_d * tc;
    (closest_ray - closest_seg).length()
}

/// Signed position along an infinite axis line at the closest approach to the ray.
pub(super) fn ray_to_axis_t(ray_o: Vec3, ray_d: Vec3, axis_origin: Vec3, axis_dir: Vec3) -> Option<f32> {
    let w     = ray_o - axis_origin;
    let b     = axis_dir.dot(ray_d);
    let d     = axis_dir.dot(w);
    let e     = ray_d.dot(w);
    let denom = 1.0 - b * b;
    if denom.abs() < 1e-8 { return None; }
    Some((d - b * e) / denom)
}

/// Ray–plane intersection; returns world hit point or `None` if parallel/behind.
pub(super) fn ray_plane_hit(ray_o: Vec3, ray_d: Vec3, plane_pt: Vec3, plane_n: Vec3) -> Option<Vec3> {
    let denom = ray_d.dot(plane_n);
    if denom.abs() < 1e-6 { return None; }
    let t = (plane_pt - ray_o).dot(plane_n) / denom;
    if t < 0.001 { return None; }
    Some(ray_o + ray_d * t)
}

/// Ray vs sphere — returns first positive `t`, or `None`.
pub(super) fn ray_sphere_intersect(origin: Vec3, dir: Vec3, center: Vec3, radius: f32) -> Option<f32> {
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
