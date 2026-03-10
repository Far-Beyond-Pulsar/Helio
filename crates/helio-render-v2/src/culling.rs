//! CPU-side frustum culling utilities.
//!
//! Implements proper bounds-based culling using the Gribb-Hartmann method for
//! frustum plane extraction and conservative sphere / AABB tests against all
//! 6 frustum planes.  The planes are derived directly from the combined
//! view-projection matrix, so they work for both perspective and orthographic
//! cameras with no extra parameters.
//!
//! # Convention
//!
//! Each plane is stored as `(nx, ny, nz, d)` where the plane equation is:
//!
//! ```text
//! nx·x + ny·y + nz·z + d >= 0   (point is on the INSIDE)
//! ```
//!
//! Planes are **not** pre-normalized; the sphere test accounts for the scale
//! of the normal by dividing out its length.

use glam::{Mat4, Vec3, Vec4};

/// Axis-aligned bounding box.
///
/// Stored as `(min, max)` in the coordinate frame of the caller (local or world).
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Aabb {
    pub min: Vec3,
    pub max: Vec3,
}

impl Aabb {
    #[inline]
    pub fn new(min: Vec3, max: Vec3) -> Self {
        Self { min, max }
    }

    /// Compute the world-space AABB for a local AABB after an arbitrary transform.
    ///
    /// The result is the tightest axis-aligned box that encloses all 8 transformed
    /// corners — always conservative, never under-estimates.
    pub fn transform(&self, m: &Mat4) -> Self {
        // Arvo / Graphics Gems method: cheaper than transforming all 8 corners.
        let mut out_min = Vec3::splat(f32::MAX);
        let mut out_max = Vec3::splat(f32::MIN);
        // Decompose: the transformed box is the sum of 8 half-extents ± each axis.
        let translation = m.w_axis.truncate();
        for axis in [m.x_axis.truncate(), m.y_axis.truncate(), m.z_axis.truncate()] {
            let a = axis * self.min;
            let b = axis * self.max;
            let lo = a.min(b);
            let hi = a.max(b);
            out_min += lo;
            out_max += hi;
        }
        Self {
            min: out_min + translation,
            max: out_max + translation,
        }
    }

    /// World-space center of the AABB.
    #[inline]
    pub fn center(&self) -> Vec3 {
        (self.min + self.max) * 0.5
    }

    /// Bounding sphere radius (half the diagonal of the AABB).
    #[inline]
    pub fn bounding_sphere_radius(&self) -> f32 {
        (self.max - self.min).length() * 0.5
    }
}

/// Six-plane camera frustum extracted from a combined view-projection matrix.
///
/// All arithmetic uses the Gribb-Hartmann method, which works with both
/// perspective and orthographic projections and with any handedness / depth
/// convention.
#[derive(Clone, Debug)]
pub struct Frustum {
    /// Planes in order: left, right, bottom, top, near, far.
    /// Each `Vec4` is `(nx, ny, nz, d)` with the inside of the frustum on the
    /// **positive** side: `dot(normal, point) + d >= 0`.
    planes: [Vec4; 6],
}

impl Frustum {
    /// Extract frustum planes from a combined view-projection matrix (P × V).
    ///
    /// Handles wgpu / Vulkan NDC conventions (z ∈ [0,1]).
    pub fn from_view_proj(m: &Mat4) -> Self {
        // Row i of the matrix (column-major storage: m.col(j)[i]).
        let row = |i: usize| -> Vec4 {
            Vec4::new(m.col(0)[i], m.col(1)[i], m.col(2)[i], m.col(3)[i])
        };
        let r0 = row(0);
        let r1 = row(1);
        let r2 = row(2);
        let r3 = row(3);

        // Gribb-Hartmann formulae for wgpu/Vulkan (NDC z ∈ [0,1]).
        Self {
            planes: [
                r3 + r0, // left   (x >= -w  →  r3+r0 ≥ 0)
                r3 - r0, // right  (x <=  w  →  r3-r0 ≥ 0)
                r3 + r1, // bottom (y >= -w)
                r3 - r1, // top    (y <=  w)
                r2,      // near   (z >=  0  in Vulkan NDC)
                r3 - r2, // far    (z <=  w)
            ],
        }
    }

    /// Returns `true` if the sphere is **potentially** visible (may have false positives
    /// near corners, but **never** culls a truly visible object).
    #[inline]
    pub fn test_sphere(&self, center: Vec3, radius: f32) -> bool {
        for plane in &self.planes {
            let normal = plane.truncate();
            // Signed distance (un-normalized) of the center from the plane.
            let dist = normal.dot(center) + plane.w;
            // If the center is farther than `radius` behind this plane, the sphere
            // is entirely outside the frustum.
            if dist < -(radius * normal.length()) {
                return false;
            }
        }
        true
    }

    /// Returns `true` if the AABB is **potentially** visible.
    ///
    /// Uses the "positive vertex" method: for each plane find the corner of the
    /// AABB that is most in the direction of the plane normal (the positive
    /// vertex p+) and test only that corner against the plane.  If even p+ is
    /// behind the plane the whole AABB is outside.
    #[inline]
    pub fn test_aabb(&self, aabb: &Aabb) -> bool {
        for plane in &self.planes {
            let n = plane.truncate();
            // Select the positive vertex: per-axis max if normal component is
            // positive, min otherwise.
            let p = Vec3::new(
                if n.x >= 0.0 { aabb.max.x } else { aabb.min.x },
                if n.y >= 0.0 { aabb.max.y } else { aabb.min.y },
                if n.z >= 0.0 { aabb.max.z } else { aabb.min.z },
            );
            if n.dot(p) + plane.w < 0.0 {
                return false;
            }
        }
        true
    }

    /// The raw planes `[(nx,ny,nz,d); 6]` as `[[f32;4]; 6]`.
    ///
    /// Suitable for uploading directly to a GPU uniform/storage buffer.
    pub fn as_raw(&self) -> [[f32; 4]; 6] {
        self.planes.map(|p| [p.x, p.y, p.z, p.w])
    }
}
