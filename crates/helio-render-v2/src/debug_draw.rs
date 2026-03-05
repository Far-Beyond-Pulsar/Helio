use std::sync::Arc;

use glam::{Quat, Vec3};
use wgpu::util::DeviceExt;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct DebugDrawVertex {
    pub position: [f32; 3],
    pub color: [f32; 4],
}

#[derive(Clone)]
pub(crate) struct DebugDrawBatch {
    pub vertex_buffer: Arc<wgpu::Buffer>,
    pub index_buffer: Arc<wgpu::Buffer>,
    pub index_count: u32,
}

#[derive(Clone, Debug)]
pub enum DebugShape {
    Line {
        start: Vec3,
        end: Vec3,
        color: [f32; 4],
        thickness: f32,
    },
    Cone {
        apex: Vec3,
        direction: Vec3,
        height: f32,
        radius: f32,
        color: [f32; 4],
        thickness: f32,
    },
    Box {
        center: Vec3,
        half_extents: Vec3,
        rotation: Quat,
        color: [f32; 4],
        thickness: f32,
    },
    Sphere {
        center: Vec3,
        radius: f32,
        color: [f32; 4],
        thickness: f32,
    },
    Capsule {
        start: Vec3,
        end: Vec3,
        radius: f32,
        color: [f32; 4],
        thickness: f32,
    },
}

pub(crate) fn build_batch(device: &wgpu::Device, shapes: &[DebugShape]) -> Option<DebugDrawBatch> {
    if shapes.is_empty() {
        return None;
    }

    let mut vertices = Vec::<DebugDrawVertex>::new();
    let mut indices = Vec::<u32>::new();

    for shape in shapes {
        match shape {
            DebugShape::Line { start, end, color, thickness } => {
                add_segment_tube(&mut vertices, &mut indices, *start, *end, half_thickness(*thickness), *color, 8);
            }
            DebugShape::Cone { apex, direction, height, radius, color, thickness } => {
                add_cone(
                    &mut vertices,
                    &mut indices,
                    *apex,
                    *direction,
                    *height,
                    *radius,
                    *color,
                    half_thickness(*thickness),
                );
            }
            DebugShape::Box { center, half_extents, rotation, color, thickness } => {
                add_box(
                    &mut vertices,
                    &mut indices,
                    *center,
                    *half_extents,
                    *rotation,
                    *color,
                    half_thickness(*thickness),
                );
            }
            DebugShape::Sphere { center, radius, color, thickness } => {
                add_sphere(
                    &mut vertices,
                    &mut indices,
                    *center,
                    *radius,
                    *color,
                    half_thickness(*thickness),
                );
            }
            DebugShape::Capsule { start, end, radius, color, thickness } => {
                add_capsule(
                    &mut vertices,
                    &mut indices,
                    *start,
                    *end,
                    *radius,
                    *color,
                    half_thickness(*thickness),
                );
            }
        }
    }

    if indices.is_empty() {
        return None;
    }

    let vertex_buffer = Arc::new(device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Debug Draw Vertex Buffer"),
        contents: bytemuck::cast_slice(&vertices),
        usage: wgpu::BufferUsages::VERTEX,
    }));
    let index_buffer = Arc::new(device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Debug Draw Index Buffer"),
        contents: bytemuck::cast_slice(&indices),
        usage: wgpu::BufferUsages::INDEX,
    }));

    Some(DebugDrawBatch {
        vertex_buffer,
        index_buffer,
        index_count: indices.len() as u32,
    })
}

fn half_thickness(thickness: f32) -> f32 {
    (thickness.max(0.0005)) * 0.5
}

fn add_segment_tube(
    vertices: &mut Vec<DebugDrawVertex>,
    indices: &mut Vec<u32>,
    start: Vec3,
    end: Vec3,
    radius: f32,
    color: [f32; 4],
    sides: u32,
) {
    let axis = end - start;
    let len = axis.length();
    if len <= 1e-6 {
        return;
    }

    let dir = axis / len;
    let (u, v) = basis_from_dir(dir);
    let side_count = sides.max(3);

    for i in 0..side_count {
        let t0 = (i as f32 / side_count as f32) * std::f32::consts::TAU;
        let t1 = ((i + 1) as f32 / side_count as f32) * std::f32::consts::TAU;

        let o0 = u * t0.cos() * radius + v * t0.sin() * radius;
        let o1 = u * t1.cos() * radius + v * t1.sin() * radius;

        let a0 = start + o0;
        let a1 = start + o1;
        let b0 = end + o0;
        let b1 = end + o1;

        add_quad(vertices, indices, a0, a1, b1, b0, color);
    }
}

fn add_polyline_tube(
    vertices: &mut Vec<DebugDrawVertex>,
    indices: &mut Vec<u32>,
    points: &[Vec3],
    color: [f32; 4],
    thickness_radius: f32,
) {
    if points.len() < 2 {
        return;
    }
    for segment in points.windows(2) {
        add_segment_tube(vertices, indices, segment[0], segment[1], thickness_radius, color, 8);
    }
}

fn add_circle(
    vertices: &mut Vec<DebugDrawVertex>,
    indices: &mut Vec<u32>,
    center: Vec3,
    axis_u: Vec3,
    axis_v: Vec3,
    radius: f32,
    color: [f32; 4],
    thickness_radius: f32,
    segments: u32,
) {
    let seg_count = segments.max(8);
    let mut ring = Vec::with_capacity((seg_count + 1) as usize);
    for i in 0..=seg_count {
        let t = (i as f32 / seg_count as f32) * std::f32::consts::TAU;
        ring.push(center + axis_u * t.cos() * radius + axis_v * t.sin() * radius);
    }
    add_polyline_tube(vertices, indices, &ring, color, thickness_radius);
}

fn add_cone(
    vertices: &mut Vec<DebugDrawVertex>,
    indices: &mut Vec<u32>,
    apex: Vec3,
    direction: Vec3,
    height: f32,
    base_radius: f32,
    color: [f32; 4],
    thickness_radius: f32,
) {
    let h = height.max(0.001);
    let r = base_radius.max(0.001);
    let dir = safe_dir(direction, Vec3::Y);
    let base_center = apex + dir * h;
    let (u, v) = basis_from_dir(dir);

    let segments = 18;
    add_circle(vertices, indices, base_center, u, v, r, color, thickness_radius, segments);

    for i in 0..segments {
        let t = (i as f32 / segments as f32) * std::f32::consts::TAU;
        let p = base_center + u * t.cos() * r + v * t.sin() * r;
        add_segment_tube(vertices, indices, apex, p, thickness_radius, color, 8);
    }
}

fn add_box(
    vertices: &mut Vec<DebugDrawVertex>,
    indices: &mut Vec<u32>,
    center: Vec3,
    half_extents: Vec3,
    rotation: Quat,
    color: [f32; 4],
    thickness_radius: f32,
) {
    let hx = half_extents.x.max(0.0001);
    let hy = half_extents.y.max(0.0001);
    let hz = half_extents.z.max(0.0001);

    let corners_local = [
        Vec3::new(-hx, -hy, -hz),
        Vec3::new(hx, -hy, -hz),
        Vec3::new(hx, hy, -hz),
        Vec3::new(-hx, hy, -hz),
        Vec3::new(-hx, -hy, hz),
        Vec3::new(hx, -hy, hz),
        Vec3::new(hx, hy, hz),
        Vec3::new(-hx, hy, hz),
    ];
    let mut corners = [Vec3::ZERO; 8];
    for (i, c) in corners_local.iter().enumerate() {
        corners[i] = center + rotation * *c;
    }

    let edges = [
        (0usize, 1usize), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ];

    for (a, b) in edges {
        add_segment_tube(vertices, indices, corners[a], corners[b], thickness_radius, color, 8);
    }
}

fn add_sphere(
    vertices: &mut Vec<DebugDrawVertex>,
    indices: &mut Vec<u32>,
    center: Vec3,
    radius: f32,
    color: [f32; 4],
    thickness_radius: f32,
) {
    let r = radius.max(0.001);
    let seg = 32;

    add_circle(vertices, indices, center, Vec3::X, Vec3::Y, r, color, thickness_radius, seg);
    add_circle(vertices, indices, center, Vec3::X, Vec3::Z, r, color, thickness_radius, seg);
    add_circle(vertices, indices, center, Vec3::Y, Vec3::Z, r, color, thickness_radius, seg);
}

fn add_capsule(
    vertices: &mut Vec<DebugDrawVertex>,
    indices: &mut Vec<u32>,
    start: Vec3,
    end: Vec3,
    radius: f32,
    color: [f32; 4],
    thickness_radius: f32,
) {
    let r = radius.max(0.001);
    let dir = safe_dir(end - start, Vec3::Y);
    let (u, v) = basis_from_dir(dir);

    add_circle(vertices, indices, start, u, v, r, color, thickness_radius, 24);
    add_circle(vertices, indices, end, u, v, r, color, thickness_radius, 24);

    add_segment_tube(vertices, indices, start + u * r, end + u * r, thickness_radius, color, 8);
    add_segment_tube(vertices, indices, start - u * r, end - u * r, thickness_radius, color, 8);
    add_segment_tube(vertices, indices, start + v * r, end + v * r, thickness_radius, color, 8);
    add_segment_tube(vertices, indices, start - v * r, end - v * r, thickness_radius, color, 8);
}

fn add_quad(
    vertices: &mut Vec<DebugDrawVertex>,
    indices: &mut Vec<u32>,
    p0: Vec3,
    p1: Vec3,
    p2: Vec3,
    p3: Vec3,
    color: [f32; 4],
) {
    let base = vertices.len() as u32;
    vertices.push(DebugDrawVertex { position: p0.to_array(), color });
    vertices.push(DebugDrawVertex { position: p1.to_array(), color });
    vertices.push(DebugDrawVertex { position: p2.to_array(), color });
    vertices.push(DebugDrawVertex { position: p3.to_array(), color });
    indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
}

fn safe_dir(v: Vec3, fallback: Vec3) -> Vec3 {
    let len = v.length();
    if len > 1e-6 {
        v / len
    } else {
        fallback
    }
}

fn basis_from_dir(dir: Vec3) -> (Vec3, Vec3) {
    let helper = if dir.y.abs() < 0.99 { Vec3::Y } else { Vec3::X };
    let u = dir.cross(helper).normalize_or_zero();
    let u = if u.length_squared() > 0.0 { u } else { Vec3::Z };
    let v = dir.cross(u).normalize_or_zero();
    (u, if v.length_squared() > 0.0 { v } else { Vec3::X })
}