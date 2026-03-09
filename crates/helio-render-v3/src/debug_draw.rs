use glam::{Vec3, Vec4};

/// All shapes the debug draw system knows about.
///
/// The batch builder converts these to coloured line segments on the CPU,
/// then uploads once per frame via `queue.write_buffer`.
#[derive(Clone, Debug)]
pub enum DebugShape {
    Line {
        start: Vec3,
        end:   Vec3,
        color: Vec4,
    },
    Sphere {
        center: Vec3,
        radius: f32,
        color:  Vec4,
    },
    Aabb {
        min:   Vec3,
        max:   Vec3,
        color: Vec4,
    },
    /// Oriented box specified by center + half-extents + rotation axes.
    Obb {
        center:  Vec3,
        axes:    [Vec3; 3],  // columns, NOT normalised (encode half-extent in length)
        color:   Vec4,
    },
    Arrow {
        start:     Vec3,
        end:       Vec3,
        color:     Vec4,
        head_size: f32,
    },
    Cross {
        center: Vec3,
        size:   f32,
        color:  Vec4,
    },
    Frustum {
        corners: [[f32; 3]; 8],  // near-plane 0..3, far-plane 4..7 (winding order matches clip space)
        color:   Vec4,
    },
}

/// Ready-to-upload batch: flat list of (position, color) pairs forming line list.
pub struct DebugDrawBatch {
    /// Interleaved: position (vec3), _pad (f32=0), color (vec4).  16+16 = 32 bytes/vertex.
    pub vertices: Vec<DebugVertex>,
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable, Debug)]
pub struct DebugVertex {
    pub position: [f32; 3],
    pub _pad:     f32,
    pub color:    [f32; 4],
}

impl DebugDrawBatch {
    pub fn build(shapes: Vec<DebugShape>) -> Self {
        let mut verts = Vec::with_capacity(shapes.len() * 24);
        for shape in shapes {
            shape.tesselate(&mut verts);
        }
        DebugDrawBatch { vertices: verts }
    }

    pub fn is_empty(&self) -> bool { self.vertices.is_empty() }
    pub fn vertex_count(&self) -> u32 { self.vertices.len() as u32 }
}

impl DebugShape {
    fn tesselate(self, out: &mut Vec<DebugVertex>) {
        match self {
            DebugShape::Line { start, end, color } => {
                push_line(out, start, end, color);
            }
            DebugShape::Sphere { center, radius, color } => {
                // 3 great circles, 32 segments each
                let n = 32usize;
                for &(ax, ay) in &[(Vec3::X, Vec3::Y), (Vec3::X, Vec3::Z), (Vec3::Y, Vec3::Z)] {
                    for i in 0..n {
                        let a0 = (i as f32 / n as f32) * std::f32::consts::TAU;
                        let a1 = ((i + 1) as f32 / n as f32) * std::f32::consts::TAU;
                        let p0 = center + (ax * a0.cos() + ay * a0.sin()) * radius;
                        let p1 = center + (ax * a1.cos() + ay * a1.sin()) * radius;
                        push_line(out, p0, p1, color);
                    }
                }
            }
            DebugShape::Aabb { min, max, color } => {
                let corners = aabb_corners(min, max);
                push_box_lines(out, corners, color);
            }
            DebugShape::Obb { center, axes, color } => {
                // Build 8 corners from ±axis[0] ±axis[1] ±axis[2]
                let mut corners = [[0f32; 3]; 8];
                let mut idx = 0;
                for sx in [-1.0f32, 1.0] {
                    for sy in [-1.0f32, 1.0] {
                        for sz in [-1.0f32, 1.0] {
                            let p = center + axes[0]*sx + axes[1]*sy + axes[2]*sz;
                            corners[idx] = p.into();
                            idx += 1;
                        }
                    }
                }
                push_box_lines(out, corners, color);
            }
            DebugShape::Arrow { start, end, color, head_size } => {
                push_line(out, start, end, color);
                let dir = (end - start).normalize();
                let perp = dir.any_orthonormal_vector();
                let perp2 = dir.cross(perp);
                let tip = end - dir * head_size;
                push_line(out, end, tip + perp  * head_size * 0.4, color);
                push_line(out, end, tip - perp  * head_size * 0.4, color);
                push_line(out, end, tip + perp2 * head_size * 0.4, color);
                push_line(out, end, tip - perp2 * head_size * 0.4, color);
            }
            DebugShape::Cross { center, size, color } => {
                let h = size * 0.5;
                push_line(out, center - Vec3::X*h, center + Vec3::X*h, color);
                push_line(out, center - Vec3::Y*h, center + Vec3::Y*h, color);
                push_line(out, center - Vec3::Z*h, center + Vec3::Z*h, color);
            }
            DebugShape::Frustum { corners, color } => {
                // near face
                for i in 0..4 {
                    let a: Vec3 = corners[i].into();
                    let b: Vec3 = corners[(i+1)%4].into();
                    push_line(out, a, b, color);
                }
                // far face
                for i in 0..4 {
                    let a: Vec3 = corners[i+4].into();
                    let b: Vec3 = corners[(i+1)%4+4].into();
                    push_line(out, a, b, color);
                }
                // connecting edges
                for i in 0..4 {
                    let a: Vec3 = corners[i].into();
                    let b: Vec3 = corners[i+4].into();
                    push_line(out, a, b, color);
                }
            }
        }
    }
}

fn push_line(out: &mut Vec<DebugVertex>, a: Vec3, b: Vec3, col: Vec4) {
    let col: [f32; 4] = col.into();
    out.push(DebugVertex { position: a.into(), _pad: 0.0, color: col });
    out.push(DebugVertex { position: b.into(), _pad: 0.0, color: col });
}

fn aabb_corners(min: Vec3, max: Vec3) -> [[f32; 3]; 8] {
    [
        [min.x, min.y, min.z], [max.x, min.y, min.z],
        [max.x, max.y, min.z], [min.x, max.y, min.z],
        [min.x, min.y, max.z], [max.x, min.y, max.z],
        [max.x, max.y, max.z], [min.x, max.y, max.z],
    ]
}

fn push_box_lines(out: &mut Vec<DebugVertex>, c: [[f32; 3]; 8], col: Vec4) {
    let edges = [
        (0,1),(1,2),(2,3),(3,0),   // near face
        (4,5),(5,6),(6,7),(7,4),   // far face
        (0,4),(1,5),(2,6),(3,7),   // edges
    ];
    for (a,b) in edges {
        push_line(out, c[a].into(), c[b].into(), col);
    }
}
