//! Editor-mode wireframe bounds for every volume in the scene.
//!
//! A volume's extent is the part you cannot see while authoring it: a light's
//! falloff, a capture's influence, a post-process blend shell. In editor mode
//! each one draws its own bounds so placing them stops being guesswork.

use glam::{Mat4, Vec3};
use libhelio::{LightType, ReflectionCaptureShape};

use crate::renderer::DebugVertex;

// One colour per volume class, so overlapping volumes stay tellable apart.
const COLOR_LIGHT: [f32; 4] = [1.0, 0.85, 0.2, 1.0]; // amber
const COLOR_REFLECTION: [f32; 4] = [0.3, 0.8, 1.0, 1.0]; // cyan
const COLOR_POST_PROCESS: [f32; 4] = [0.75, 0.4, 1.0, 1.0]; // violet
const COLOR_POST_PROCESS_BLEND: [f32; 4] = [0.45, 0.25, 0.6, 1.0]; // dim violet
const COLOR_WATER: [f32; 4] = [0.2, 0.55, 1.0, 1.0]; // blue
const COLOR_WATER_HITBOX: [f32; 4] = [0.2, 0.9, 0.8, 1.0]; // teal
const COLOR_DECAL: [f32; 4] = [1.0, 0.5, 0.2, 1.0]; // orange

const SPHERE_SEGMENTS: u32 = 24;
const CONE_SEGMENTS: u32 = 16;

/// Accumulates line-list vertices. The debug pipeline is a `LineList`, so every
/// segment contributes an independent pair.
#[derive(Default)]
struct LineSink {
    verts: Vec<DebugVertex>,
}

impl LineSink {
    fn line(&mut self, a: Vec3, b: Vec3, color: [f32; 4]) {
        self.verts.push(DebugVertex {
            position: a.to_array(),
            _pad: 0.0,
            color,
        });
        self.verts.push(DebugVertex {
            position: b.to_array(),
            _pad: 0.0,
            color,
        });
    }

    /// Twelve edges of a box, transformed out of local space. Passing an
    /// oriented transform is what lets rotated volumes draw correctly rather
    /// than collapsing to their axis-aligned hull.
    fn wire_box(&mut self, local_to_world: Mat4, half: Vec3, color: [f32; 4]) {
        let mut c = [Vec3::ZERO; 8];
        for (i, corner) in c.iter_mut().enumerate() {
            // Bit 0/1/2 select the sign on x/y/z.
            let sign = Vec3::new(
                if i & 1 == 0 { -1.0 } else { 1.0 },
                if i & 2 == 0 { -1.0 } else { 1.0 },
                if i & 4 == 0 { -1.0 } else { 1.0 },
            );
            *corner = local_to_world.transform_point3(sign * half);
        }
        // Corner pairs differing in exactly one axis bit.
        const EDGES: [(usize, usize); 12] = [
            (0, 1),
            (2, 3),
            (4, 5),
            (6, 7),
            (0, 2),
            (1, 3),
            (4, 6),
            (5, 7),
            (0, 4),
            (1, 5),
            (2, 6),
            (3, 7),
        ];
        for (a, b) in EDGES {
            self.line(c[a], c[b], color);
        }
    }

    fn aabb(&mut self, min: Vec3, max: Vec3, color: [f32; 4]) {
        let center = (min + max) * 0.5;
        let half = (max - min) * 0.5;
        self.wire_box(Mat4::from_translation(center), half, color);
    }

    /// Three orthogonal rings — enough to read a radius without the vertex cost
    /// of a full wire sphere.
    fn wire_sphere(&mut self, center: Vec3, radius: f32, color: [f32; 4]) {
        if radius <= 0.0 {
            return;
        }
        for plane in 0..3 {
            let mut prev = Vec3::ZERO;
            for i in 0..=SPHERE_SEGMENTS {
                let t = i as f32 / SPHERE_SEGMENTS as f32 * std::f32::consts::TAU;
                let (s, c) = (t.sin() * radius, t.cos() * radius);
                let p = center
                    + match plane {
                        0 => Vec3::new(c, s, 0.0),
                        1 => Vec3::new(c, 0.0, s),
                        _ => Vec3::new(0.0, c, s),
                    };
                if i > 0 {
                    self.line(prev, p, color);
                }
                prev = p;
            }
        }
    }

    fn wire_cone(&mut self, apex: Vec3, dir: Vec3, height: f32, radius: f32, color: [f32; 4]) {
        if height <= 0.0 {
            return;
        }
        let dir = dir.normalize_or_zero();
        if dir == Vec3::ZERO {
            return;
        }
        let base = apex + dir * height;
        let up = if dir.cross(Vec3::Y).length_squared() < 1e-8 {
            Vec3::X
        } else {
            Vec3::Y
        };
        let tangent = dir.cross(up).normalize_or_zero();
        let bitangent = dir.cross(tangent).normalize_or_zero();
        let mut prev = base + tangent * radius;
        for i in 1..=CONE_SEGMENTS {
            let t = i as f32 / CONE_SEGMENTS as f32 * std::f32::consts::TAU;
            let cur = base + (tangent * t.cos() + bitangent * t.sin()) * radius;
            self.line(prev, cur, color); // base ring
            self.line(cur, apex, color); // side
            prev = cur;
        }
    }
}

fn v3(a: &[f32; 4]) -> Vec3 {
    Vec3::new(a[0], a[1], a[2])
}

impl super::Scene {
    /// Wireframe bounds for every bounded volume in the scene.
    ///
    /// Rebuilt from scene records rather than GPU buffers, since the editor
    /// overlay needs CPU-side geometry and volumes number in the dozens.
    pub(crate) fn editor_volume_debug_lines(&self) -> Vec<DebugVertex> {
        let mut sink = LineSink::default();

        // Lights — the attenuation volume, which is the invisible part.
        for (_, rec) in self.lights.iter_with_handles() {
            let g = &rec.gpu;
            let pos = v3(&g.position_range);
            let range = g.position_range[3];
            if g.light_type == LightType::Spot as u32 {
                let outer_cos = g.direction_outer[3].clamp(-1.0, 1.0);
                // Radius of the cone's base at the far end of its range.
                let radius = range * outer_cos.acos().tan().abs();
                sink.wire_cone(pos, v3(&g.direction_outer), range, radius, COLOR_LIGHT);
            } else if g.light_type == LightType::Point as u32
                || g.light_type == LightType::Area as u32
            {
                sink.wire_sphere(pos, range, COLOR_LIGHT);
            }
            // Directional lights are unbounded — nothing meaningful to outline.
        }

        // Reflection captures — sphere or oriented box, matching the shader's
        // own influence test.
        for (_, rec) in self.reflection_captures.iter_with_handles() {
            let g = &rec.gpu;
            if g.shape == ReflectionCaptureShape::Box as u32 {
                let local_to_world = Mat4::from_cols_array_2d(&g.world_to_local).inverse();
                sink.wire_box(local_to_world, v3(&g.extents_transition), COLOR_REFLECTION);
            } else {
                sink.wire_sphere(v3(&g.position_radius), g.position_radius[3], COLOR_REFLECTION);
            }
        }

        // Post-process volumes — bounds, plus the blend shell they fade across.
        for (_, rec) in self.pp_volumes.iter_with_handles() {
            let g = &rec.gpu;
            if g.unbound != 0 {
                continue; // applies everywhere; no bounds exist to draw
            }
            let min = v3(&g.bounds_min);
            let max = v3(&g.bounds_max);
            sink.aabb(min, max, COLOR_POST_PROCESS);
            if g.blend_radius > 0.0 {
                let b = Vec3::splat(g.blend_radius);
                sink.aabb(min - b, max + b, COLOR_POST_PROCESS_BLEND);
            }
        }

        for (_, rec) in self.water_volumes.iter_with_handles() {
            sink.aabb(
                v3(&rec.gpu.bounds_min),
                v3(&rec.gpu.bounds_max),
                COLOR_WATER,
            );
        }

        // Hitboxes move every frame; the new bounds are the live ones.
        for (_, rec) in self.water_hitboxes.iter_with_handles() {
            sink.aabb(
                v3(&rec.gpu.new_min),
                v3(&rec.gpu.new_max),
                COLOR_WATER_HITBOX,
            );
        }

        // Decals project through a [-1,1] local cube, and GpuDecal.transform is
        // world→local (decal_collect.wgsl:111), so it inverts to draw.
        for (_, rec) in self.decals.iter_with_handles() {
            let world_to_local = Mat4::from_cols_array(&rec.gpu.transform);
            sink.wire_box(world_to_local.inverse(), Vec3::ONE, COLOR_DECAL);
        }

        sink.verts
    }
}
