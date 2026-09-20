//! Shared procedural architecture primitives, batched by material.
use crate::v3_demo_common::box_mesh;
use glam::Vec3;
use helio::PackedVertex;

#[derive(Default)]
pub(crate) struct Mesh {
    pub(crate) vertices: Vec<PackedVertex>,
    pub(crate) indices: Vec<u32>,
}
impl Mesh {
    pub(crate) fn triangle(&mut self, a: Vec3, b: Vec3, c: Vec3) {
        let normal = (b - a).cross(c - a).normalize();
        let tangent = (b - a).normalize();
        let base = self.vertices.len() as u32;
        for (p, uv) in [(a, [0., 0.]), (b, [1., 0.]), (c, [0., 1.])] {
            self.vertices.push(PackedVertex::from_components(
                p.to_array(),
                normal.to_array(),
                uv,
                tangent.to_array(),
                1.,
            ));
        }
        self.indices.extend_from_slice(&[base, base + 1, base + 2]);
    }
    pub(crate) fn quad(&mut self, a: Vec3, b: Vec3, c: Vec3, d: Vec3) {
        // Both triangles share one UV rectangle. Calling triangle twice maps
        // the whole texture onto each half and disagrees along the diagonal.
        // Preserve per-triangle face normals for non-planar architectural quads.
        for (points, uvs, tangent) in [
            ([a, b, c], [[0., 0.], [1., 0.], [1., 1.]], (b-a).normalize()),
            ([a, c, d], [[0., 0.], [1., 1.], [0., 1.]], (c-d).normalize()),
        ] {
            let normal=(points[1]-points[0]).cross(points[2]-points[0]).normalize();
            let base=self.vertices.len() as u32;
            for (point,uv) in points.into_iter().zip(uvs) {
                self.vertices.push(PackedVertex::from_components(
                    point.to_array(),normal.to_array(),uv,tangent.to_array(),1.0,
                ));
            }
            self.indices.extend_from_slice(&[base,base+1,base+2]);
        }
    }
    pub(crate) fn block(&mut self, center: [f32; 3], half: [f32; 3]) {
        let mesh = box_mesh(center, half);
        let base = self.vertices.len() as u32;
        self.vertices.extend(mesh.vertices);
        self.indices
            .extend(mesh.indices.into_iter().map(|i| i + base));
    }
    pub(crate) fn rod(&mut self, a: Vec3, b: Vec3, radius: f32, sides: usize) {
        let axis = (b - a).normalize();
        let helper = if axis.y.abs() < 0.9 { Vec3::Y } else { Vec3::X };
        let u = axis.cross(helper).normalize() * radius;
        let v = axis.cross(u).normalize() * radius;
        for i in 0..sides {
            let t = i as f32 * std::f32::consts::TAU / sides as f32;
            let t1 = (i + 1) as f32 * std::f32::consts::TAU / sides as f32;
            let p = u * t.cos() + v * t.sin();
            let q = u * t1.cos() + v * t1.sin();
            self.quad(a + p, a + q, b + q, b + p);
            self.triangle(a, a + q, a + p);
            self.triangle(b, b + p, b + q);
        }
    }
    /// Smooth cylindrical sides with a duplicated UV seam and flat end caps.
    pub(crate) fn smooth_rod(&mut self, a: Vec3, b: Vec3, radius: f32, sides: usize) {
        assert!(sides >= 3 && radius > 0.0 && a.distance_squared(b) > 0.0);
        let axis = (b - a).normalize();
        let helper = if axis.y.abs() < 0.9 { Vec3::Y } else { Vec3::X };
        let u = axis.cross(helper).normalize();
        let v = axis.cross(u).normalize();
        let base = self.vertices.len() as u32;
        for i in 0..=sides {
            let uv_x = i as f32 / sides as f32;
            // Make the seam geometrically identical instead of relying on sin(TAU).
            let angle = (i % sides) as f32 * std::f32::consts::TAU / sides as f32;
            let radial = u * angle.cos() + v * angle.sin();
            let tangent = -u * angle.sin() + v * angle.cos();
            for (center, uv_y) in [(a, 0.0), (b, 1.0)] {
                self.vertices.push(PackedVertex::from_components(
                    (center + radial * radius).to_array(), radial.to_array(),
                    [uv_x, uv_y], tangent.to_array(), 1.0,
                ));
            }
        }
        for i in 0..sides {
            let first = base + (2 * i) as u32;
            self.indices.extend_from_slice(&[first, first+2, first+3, first, first+3, first+1]);
            let angle = i as f32 * std::f32::consts::TAU / sides as f32;
            let next = ((i+1) % sides) as f32 * std::f32::consts::TAU / sides as f32;
            let p = (u * angle.cos() + v * angle.sin()) * radius;
            let q = (u * next.cos() + v * next.sin()) * radius;
            for (center, normal, offsets, sign) in [
                (a, -axis, [Vec3::ZERO, q, p], -1.0),
                (b, axis, [Vec3::ZERO, p, q], 1.0),
            ] {
                let cap_base = self.vertices.len() as u32;
                for offset in offsets {
                    self.vertices.push(PackedVertex::from_components(
                        (center + offset).to_array(), normal.to_array(),
                        [0.5 + 0.5 * offset.dot(u) / radius,
                         0.5 + sign * 0.5 * offset.dot(v) / radius],
                        u.to_array(), 1.0,
                    ));
                }
                self.indices.extend_from_slice(&[cap_base, cap_base+1, cap_base+2]);
            }
        }
    }
    pub(crate) fn ring(&mut self, center: Vec3, u: Vec3, v: Vec3, radius: f32, thickness: f32) {
        for i in 0..48 {
            let t = i as f32 * std::f32::consts::TAU / 48.;
            let t1 = (i + 1) as f32 * std::f32::consts::TAU / 48.;
            self.rod(
                center + (u * t.cos() + v * t.sin()) * radius,
                center + (u * t1.cos() + v * t1.sin()) * radius,
                thickness,
                8,
            );
        }
    }
    pub(crate) fn arch(&mut self, a: Vec3, b: Vec3, rise: f32, radius: f32) {
        // Two curved halves meet at a pointed crown.
        let mid = (a + b) * 0.5 + Vec3::Y * rise;
        for (start, end) in [(a, mid), (b, mid)] {
            let mut previous = start;
            for i in 1..=24 {
                let t = i as f32 / 24.;
                let mut p = start.lerp(end, t);
                p.y += rise * 0.24 * (t * std::f32::consts::PI).sin();
                self.rod(previous, p, radius, 10);
                previous = p;
            }
        }
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quad_texture_covers_one_rectangle_without_a_diagonal_seam() {
        let mut mesh=Mesh::default();
        mesh.quad(Vec3::ZERO,Vec3::X*2.0,Vec3::new(2.0,3.0,0.0),Vec3::Y*3.0);
        let uv:Vec<_>=mesh.vertices.iter().map(|v|v.tex_coords0).collect();
        assert_eq!(uv,vec![[0.,0.],[1.,0.],[1.,1.],[0.,0.],[1.,1.],[0.,1.]]);
        // Duplicated diagonal vertices agree in both position and UV. A
        // checkerboard therefore has no discontinuity at the triangle split.
        for (a,b) in [(0,3),(2,4)] {
            assert_eq!(mesh.vertices[a].position,mesh.vertices[b].position);
            assert_eq!(uv[a],uv[b]);
            assert_eq!(mesh.vertices[a].normal,mesh.vertices[b].normal);
            assert_eq!(mesh.vertices[a].tangent,mesh.vertices[b].tangent);
        }
        assert_eq!(mesh.indices,vec![0,1,2,3,4,5]);
    }
}
