//! Simple procedural mesh builders shared by all v3 examples.
//!
//! v3 has no GpuMesh::cube / GpuMesh::plane helpers, so we build them here.

use std::sync::Arc;
use helio_render_v3::{GpuMesh, PackedVertex};

// ─── Box ─────────────────────────────────────────────────────────────────────

/// Return an axis-aligned box centred at `(cx, cy, cz)` with half-extents
/// `(hx, hy, hz)`.  It is equivalent to the old `GpuMesh::rect3d` / `GpuMesh::cube`.
pub fn build_box(
    device: &wgpu::Device,
    queue:  &wgpu::Queue,
    cx: f32, cy: f32, cz: f32,
    hx: f32, hy: f32, hz: f32,
) -> Arc<GpuMesh> {
    // Each face: 4 verts, winding so outward normals face away from centre.
    // (n = outward normal, t = tangent, bt_sign = +1)
    let (x0, x1) = (cx - hx, cx + hx);
    let (y0, y1) = (cy - hy, cy + hy);
    let (z0, z1) = (cz - hz, cz + hz);

    let mut verts: Vec<PackedVertex> = Vec::with_capacity(24);
    let mut idx:   Vec<u32>          = Vec::with_capacity(36);

    // Helper: add one quad, push 4 verts + 2 triangles.
    // Corners in CCW order when viewed from the outside.
    let mut quad = |corners: [[f32; 3]; 4], uvs: [[f32; 2]; 4],
                    n: [f32; 3], t: [f32; 3]| {
        let base = verts.len() as u32;
        for (&c, &uv) in corners.iter().zip(uvs.iter()) {
            verts.push(PackedVertex::new(c, uv, n, t, 1.0));
        }
        idx.extend_from_slice(&[base, base+1, base+2,  base, base+2, base+3]);
    };

    // +Y (top)
    quad(
        [[x0,y1,z1],[x1,y1,z1],[x1,y1,z0],[x0,y1,z0]],
        [[0.0,1.0],[1.0,1.0],[1.0,0.0],[0.0,0.0]],
        [0.0,1.0,0.0], [1.0,0.0,0.0],
    );
    // -Y (bottom)
    quad(
        [[x0,y0,z0],[x1,y0,z0],[x1,y0,z1],[x0,y0,z1]],
        [[0.0,0.0],[1.0,0.0],[1.0,1.0],[0.0,1.0]],
        [0.0,-1.0,0.0], [1.0,0.0,0.0],
    );
    // +Z (front)
    quad(
        [[x0,y0,z1],[x1,y0,z1],[x1,y1,z1],[x0,y1,z1]],
        [[0.0,1.0],[1.0,1.0],[1.0,0.0],[0.0,0.0]],
        [0.0,0.0,1.0], [1.0,0.0,0.0],
    );
    // -Z (back)
    quad(
        [[x1,y0,z0],[x0,y0,z0],[x0,y1,z0],[x1,y1,z0]],
        [[0.0,1.0],[1.0,1.0],[1.0,0.0],[0.0,0.0]],
        [0.0,0.0,-1.0], [-1.0,0.0,0.0],
    );
    // +X (right)
    quad(
        [[x1,y0,z1],[x1,y0,z0],[x1,y1,z0],[x1,y1,z1]],
        [[0.0,1.0],[1.0,1.0],[1.0,0.0],[0.0,0.0]],
        [1.0,0.0,0.0], [0.0,0.0,-1.0],
    );
    // -X (left)
    quad(
        [[x0,y0,z0],[x0,y0,z1],[x0,y1,z1],[x0,y1,z0]],
        [[0.0,1.0],[1.0,1.0],[1.0,0.0],[0.0,0.0]],
        [-1.0,0.0,0.0], [0.0,0.0,1.0],
    );

    GpuMesh::upload(device, queue, &verts, &idx, Some("box"))
}

// ─── Plane ───────────────────────────────────────────────────────────────────

/// A flat Y-plane centred at `(cx, y, cz)` with half-extents `(hw, hd)`.
/// Normal points +Y.  Equivalent to old `GpuMesh::plane`.
pub fn build_plane(
    device: &wgpu::Device,
    queue:  &wgpu::Queue,
    cx: f32, y: f32, cz: f32,
    hw: f32, hd: f32,
) -> Arc<GpuMesh> {
    let (x0, x1) = (cx - hw, cx + hw);
    let (z0, z1) = (cz - hd, cz + hd);
    let n = [0.0_f32, 1.0, 0.0];
    let t = [1.0_f32, 0.0, 0.0];
    let verts = [
        PackedVertex::new([x0, y, z1], [0.0, 1.0], n, t, 1.0),
        PackedVertex::new([x1, y, z1], [1.0, 1.0], n, t, 1.0),
        PackedVertex::new([x1, y, z0], [1.0, 0.0], n, t, 1.0),
        PackedVertex::new([x0, y, z0], [0.0, 0.0], n, t, 1.0),
    ];
    let idx: [u32; 6] = [0, 1, 2,  0, 2, 3];
    GpuMesh::upload(device, queue, &verts, &idx, Some("plane"))
}
