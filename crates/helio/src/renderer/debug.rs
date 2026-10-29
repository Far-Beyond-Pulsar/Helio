use std::sync::{Arc, Mutex};

use bytemuck::{Pod, Zeroable};

use helio_core::{PassContext, PrepareContext, RenderCmds, RenderPass, Result as HelioResult};

use super::renderer_impl::Renderer;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct DebugCameraUniform {
    pub view_proj: [[f32; 4]; 4],
}

#[repr(C)]
#[derive(Clone, Copy, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct DebugVertex {
    pub position: [f32; 3],
    pub _pad: f32,
    pub color: [f32; 4],
}

pub struct DebugDrawState {
    pub editor_enabled: bool,
    /// Double-precision world position of the frame's coordinate origin (see
    /// `Renderer::origin_for`). Overlays defined in world space (the grid
    /// plane at world y = 0, debug lines) are placed relative to it.
    pub world_origin: glam::DVec3,
    pub user_lines: Vec<WorldVertex>,
    pub user_lines_generation: u64,
    pub user_tris: Vec<WorldVertex>,
    pub user_tris_generation: u64,
    /// Wireframe bounds of every scene volume, rebuilt by the renderer while
    /// the editor overlay is on. Kept separate from `user_lines` because the
    /// editor path deliberately ignores those.
    pub editor_volume_lines: Vec<WorldVertex>,
    /// Bumped whenever `editor_volume_lines` changes, so the pass can keep
    /// caching its uploads instead of re-sending the set every frame.
    pub editor_volume_generation: u64,
    /// Named sets of world-space editor lines, each owned by one producer
    /// (e.g. `"splines"` from SceneDB spline components). Drawn with the
    /// volume bounds, so they sit in the scene and follow the camera.
    pub editor_line_layers: std::collections::BTreeMap<String, Vec<WorldVertex>>,
    /// Bumped whenever any of `editor_line_layers` changes.
    pub editor_layers_generation: u64,
    /// Color-blind mode for axis/gizmo colors:
    /// 0=None, 1=Protanopia, 2=Deuteranopia, 3=Tritanopia, 4=Achromatopsia
    pub color_blind_mode: u8,
}

/// A world-space position for the debug-draw API, in f32 or f64. Debug
/// geometry is kept in double precision and drawn relative to the frame's
/// world origin, so a line on a planet's surface lands where it is.
pub trait WorldPoint {
    fn world(self) -> glam::DVec3;
}

impl WorldPoint for glam::DVec3 {
    fn world(self) -> glam::DVec3 {
        self
    }
}

impl WorldPoint for glam::Vec3 {
    fn world(self) -> glam::DVec3 {
        self.as_dvec3()
    }
}

impl WorldPoint for [f64; 3] {
    fn world(self) -> glam::DVec3 {
        glam::DVec3::from_array(self)
    }
}

impl WorldPoint for [f32; 3] {
    fn world(self) -> glam::DVec3 {
        glam::Vec3::from_array(self).as_dvec3()
    }
}

/// A debug-geometry vertex in world space.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WorldVertex {
    pub position: glam::DVec3,
    pub color: [f32; 4],
}

impl WorldVertex {
    pub fn new(position: impl WorldPoint, color: [f32; 4]) -> Self {
        Self {
            position: position.world(),
            color,
        }
    }

    /// This vertex in the frame of `origin`, for the GPU.
    fn in_frame(&self, origin: glam::DVec3) -> DebugVertex {
        DebugVertex {
            position: (self.position - origin).as_vec3().to_array(),
            _pad: 0.0,
            color: self.color,
        }
    }
}

/// A unit vector perpendicular to `n` and another perpendicular to both.
fn basis(n: glam::Vec3) -> (glam::Vec3, glam::Vec3) {
    let up = if n.cross(glam::Vec3::Y).length_squared() < 1e-8 {
        glam::Vec3::X
    } else {
        glam::Vec3::Y
    };
    let tangent = n.cross(up).normalize_or_zero();
    (tangent, n.cross(tangent).normalize_or_zero())
}

/// Debug geometry submitted under one lock. Shapes are placed at a world
/// anchor (`impl WorldPoint`) and built from f32 offsets to it, so they keep
/// their shape however far from the world's origin they are.
pub struct DebugBatch<'a> {
    pub(crate) state: &'a mut DebugDrawState,
    pub(crate) lines_changed: bool,
    pub(crate) tris_changed: bool,
}

impl<'a> DebugBatch<'a> {
    pub fn line(&mut self, from: impl WorldPoint, to: impl WorldPoint, color: [f32; 4]) {
        if self.state.user_lines.len() + 2 > MAX_DEBUG_VERTS as usize {
            return;
        }
        self.state.user_lines.push(WorldVertex::new(from, color));
        self.state.user_lines.push(WorldVertex::new(to, color));
        self.lines_changed = true;
    }

    pub fn tri(
        &mut self,
        v0: impl WorldPoint,
        v1: impl WorldPoint,
        v2: impl WorldPoint,
        color: [f32; 4],
    ) {
        if self.state.user_tris.len() + 3 > MAX_DEBUG_TRIS as usize {
            return;
        }
        self.state.user_tris.push(WorldVertex::new(v0, color));
        self.state.user_tris.push(WorldVertex::new(v1, color));
        self.state.user_tris.push(WorldVertex::new(v2, color));
        self.tris_changed = true;
    }

    /// A line between two offsets from `anchor`.
    fn line_at(&mut self, anchor: glam::DVec3, a: glam::Vec3, b: glam::Vec3, color: [f32; 4]) {
        self.line(anchor + a.as_dvec3(), anchor + b.as_dvec3(), color);
    }

    /// A triangle of three offsets from `anchor`.
    fn tri_at(
        &mut self,
        anchor: glam::DVec3,
        a: glam::Vec3,
        b: glam::Vec3,
        c: glam::Vec3,
        color: [f32; 4],
    ) {
        self.tri(
            anchor + a.as_dvec3(),
            anchor + b.as_dvec3(),
            anchor + c.as_dvec3(),
            color,
        );
    }

    /// The circle of `radius` about `center` in the plane spanned by
    /// `tangent` and `bitangent`, as offsets from `center`.
    fn ring(
        tangent: glam::Vec3,
        bitangent: glam::Vec3,
        radius: f32,
        segments: u32,
    ) -> impl Iterator<Item = (glam::Vec3, glam::Vec3)> {
        let at = move |i: u32| {
            let theta = i as f32 / segments as f32 * std::f32::consts::TAU;
            (tangent * theta.cos() + bitangent * theta.sin()) * radius
        };
        (1..=segments).map(move |i| (at(i - 1), at(i)))
    }

    pub fn circle(&mut self, center: impl WorldPoint, radius: f32, color: [f32; 4], segments: u32) {
        if segments < 3 {
            return;
        }
        let c = center.world();
        for (a, b) in Self::ring(glam::Vec3::X, glam::Vec3::Z, radius, segments) {
            self.line_at(c, a, b, color);
        }
    }

    pub fn sphere(&mut self, center: impl WorldPoint, radius: f32, color: [f32; 4], segments: u32) {
        if segments < 4 {
            return;
        }
        let c = center.world();
        for (u, v) in [
            (glam::Vec3::X, glam::Vec3::Y),
            (glam::Vec3::X, glam::Vec3::Z),
            (glam::Vec3::Y, glam::Vec3::Z),
        ] {
            for (a, b) in Self::ring(u, v, radius, segments) {
                self.line_at(c, a, b, color);
            }
        }
    }

    pub fn filled_disk(
        &mut self,
        center: impl WorldPoint,
        normal: [f32; 3],
        radius: f32,
        color: [f32; 4],
        segments: u32,
    ) {
        if segments < 3 {
            return;
        }
        let c = center.world();
        let (tangent, bitangent) = basis(glam::Vec3::from(normal).normalize_or_zero());
        for (a, b) in Self::ring(tangent, bitangent, radius, segments) {
            self.tri_at(c, glam::Vec3::ZERO, a, b, color);
        }
    }

    pub fn cone(
        &mut self,
        apex: impl WorldPoint,
        axis: [f32; 3],
        height: f32,
        base_radius: f32,
        color: [f32; 4],
        segments: u32,
    ) {
        if segments < 3 {
            return;
        }
        let apex = apex.world();
        let dir = glam::Vec3::from(axis).normalize_or_zero();
        let base = dir * height;
        let (tangent, bitangent) = basis(dir);
        for (a, b) in Self::ring(tangent, bitangent, base_radius, segments) {
            self.line_at(apex, base + a, base + b, color);
            self.line_at(apex, base + b, glam::Vec3::ZERO, color);
        }
    }

    pub fn filled_cone(
        &mut self,
        apex: impl WorldPoint,
        axis: [f32; 3],
        height: f32,
        base_radius: f32,
        color: [f32; 4],
        segments: u32,
    ) {
        if segments < 3 {
            return;
        }
        let apex = apex.world();
        let dir = glam::Vec3::from(axis).normalize_or_zero();
        let base = dir * height;
        let (tangent, bitangent) = basis(dir);
        for (a, b) in Self::ring(tangent, bitangent, base_radius, segments) {
            self.tri_at(apex, glam::Vec3::ZERO, base + a, base + b, color);
            self.tri_at(apex, base, base + b, base + a, color);
        }
    }

    pub fn cylinder(
        &mut self,
        base_center: impl WorldPoint,
        axis: [f32; 3],
        height: f32,
        radius: f32,
        color: [f32; 4],
        segments: u32,
    ) {
        if segments < 3 {
            return;
        }
        let base = base_center.world();
        let dir = glam::Vec3::from(axis).normalize_or_zero();
        let top = dir * height;
        let (tangent, bitangent) = basis(dir);
        for (a, b) in Self::ring(tangent, bitangent, radius, segments) {
            self.line_at(base, a, b, color);
            self.line_at(base, top + a, top + b, color);
            self.line_at(base, a, top + a, color);
        }
    }

    pub fn torus(
        &mut self,
        center: impl WorldPoint,
        normal: [f32; 3],
        major_radius: f32,
        minor_radius: f32,
        color: [f32; 4],
        major_segments: u32,
        minor_segments: u32,
    ) {
        if major_segments < 3 || minor_segments < 3 {
            return;
        }
        let c = center.world();
        let n = glam::Vec3::from(normal).normalize_or_zero();
        let (tangent, bitangent) = basis(n);
        let ring_point = |theta: f32, phi: f32| {
            let radial = tangent * theta.cos() + bitangent * theta.sin();
            radial * major_radius + (n * phi.cos() + radial * phi.sin()) * minor_radius
        };
        let angle = |i: u32, of: u32| i as f32 / of as f32 * std::f32::consts::TAU;
        for j in 0..major_segments {
            let (theta0, theta1) = (angle(j, major_segments), angle(j + 1, major_segments));
            for i in 0..minor_segments {
                let (phi0, phi1) = (angle(i, minor_segments), angle(i + 1, minor_segments));
                self.line_at(c, ring_point(theta0, phi0), ring_point(theta0, phi1), color);
                self.line_at(c, ring_point(theta0, phi0), ring_point(theta1, phi0), color);
            }
        }
    }

    pub fn filled_box(&mut self, center: impl WorldPoint, half: f32, color: [f32; 4]) {
        let c = center.world();
        let h = half;
        let corners = [
            glam::Vec3::new(-h, -h, -h),
            glam::Vec3::new(h, -h, -h),
            glam::Vec3::new(h, h, -h),
            glam::Vec3::new(-h, h, -h),
            glam::Vec3::new(-h, -h, h),
            glam::Vec3::new(h, -h, h),
            glam::Vec3::new(h, h, h),
            glam::Vec3::new(-h, h, h),
        ];
        const QUADS: [[usize; 4]; 6] = [
            [0, 3, 2, 1],
            [4, 5, 6, 7],
            [0, 4, 7, 3],
            [1, 2, 6, 5],
            [0, 1, 5, 4],
            [3, 7, 6, 2],
        ];
        for [a, b, cc, d] in QUADS {
            self.tri_at(c, corners[a], corners[b], corners[cc], color);
            self.tri_at(c, corners[a], corners[cc], corners[d], color);
        }
    }

    pub fn frustum(
        &mut self,
        origin: impl WorldPoint,
        forward: [f32; 3],
        up: [f32; 3],
        fov_y: f32,
        aspect: f32,
        near: f32,
        far: f32,
        color: [f32; 4],
    ) {
        let o = origin.world();
        let fwd = glam::Vec3::from(forward).normalize_or_zero();
        let upv = glam::Vec3::from(up).normalize_or_zero();
        let rightv = fwd.cross(upv).normalize_or_zero();
        let corners = |distance: f32| {
            let h = (fov_y * 0.5).tan() * distance;
            let w = h * aspect;
            let centre = fwd * distance;
            [
                centre + upv * h - rightv * w,
                centre + upv * h + rightv * w,
                centre - upv * h + rightv * w,
                centre - upv * h - rightv * w,
            ]
        };
        let (n, f) = (corners(near), corners(far));
        for i in 0..4 {
            self.line_at(o, n[i], n[(i + 1) % 4], color);
            self.line_at(o, f[i], f[(i + 1) % 4], color);
            self.line_at(o, n[i], f[i], color);
        }
    }

    pub(crate) fn finish(self) {
        if self.lines_changed {
            self.state.user_lines_generation = self.state.user_lines_generation.wrapping_add(1);
        }
        if self.tris_changed {
            self.state.user_tris_generation = self.state.user_tris_generation.wrapping_add(1);
        }
    }
}

impl Default for DebugDrawState {
    fn default() -> Self {
        Self {
            editor_enabled: false,
            world_origin: glam::DVec3::ZERO,
            user_lines: Vec::new(),
            user_lines_generation: 0,
            user_tris: Vec::new(),
            user_tris_generation: 0,
            editor_volume_lines: Vec::new(),
            editor_volume_generation: 0,
            editor_line_layers: std::collections::BTreeMap::new(),
            editor_layers_generation: 0,
            color_blind_mode: 0,
        }
    }
}

const MAX_DEBUG_VERTS: u32 = 65536;
const MAX_DEBUG_TRIS: u32 = 65536;
/// What the vertex buffers start with; they grow with the geometry.
const MIN_DEBUG_VERTS: usize = 256;

fn debug_vertex_buffer(device: &wgpu::Device, label: &str, vertices: usize) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: (vertices * std::mem::size_of::<DebugVertex>()) as u64,
        usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

/// Replaces `buffer` with a larger one (a power of two of vertices) when
/// `vertices` do not fit. The contents are not kept: callers upload all of
/// them after.
fn grow_debug_buffer(
    device: &wgpu::Device,
    buffer: &mut wgpu::Buffer,
    label: &str,
    vertices: usize,
) {
    let bytes = (vertices * std::mem::size_of::<DebugVertex>()) as u64;
    if bytes > buffer.size() {
        *buffer = debug_vertex_buffer(device, label, vertices.next_power_of_two());
    }
}

const MAX_EDITOR_LINE_LAYERS: usize = 32;
const MAX_EDITOR_LAYER_VERTS: usize = 65536;

pub struct DebugPass {
    pipeline_depth: wgpu::RenderPipeline,
    pipeline_no_depth: wgpu::RenderPipeline,
    pipeline_tri_depth: wgpu::RenderPipeline,
    pipeline_tri_no_depth: wgpu::RenderPipeline,
    #[allow(dead_code)]
    bgl: wgpu::BindGroupLayout,
    camera_buf: wgpu::Buffer,
    bind_group: Option<wgpu::BindGroup>,
    bind_group_key: Option<wgpu::Buffer>,
    vertex_buf: wgpu::Buffer,
    pub vertex_count: u32,
    tri_buf: wgpu::Buffer,
    pub tri_count: u32,
    depth_test_enabled: bool,
    /// Test against the internal-res scene depth (and therefore draw into the
    /// internal-res lit image) rather than the output-res dummy depth.
    /// Required for real occlusion — see `render_pass_descriptor`.
    use_scene_depth: bool,
}

impl DebugPass {
    pub fn new(
        device: &wgpu::Device,
        camera_buf: &wgpu::Buffer,
        target_format: wgpu::TextureFormat,
        depth_test: bool,
    ) -> Self {
        let shader = helio_core::shader::module(
            device,
            "Debug Draw Shader",
            helio_core::include_wgsl!("../../shaders/debug_draw.wgsl"),
        );

        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Debug Draw BGL"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Debug Draw PL"),
            bind_group_layouts: &[Some(&bgl)],
            immediate_size: 0,
        });

        let vertex_buf = debug_vertex_buffer(device, "Debug Vertex Buffer", MIN_DEBUG_VERTS);

        let pipeline_depth = helio_core::shader::create_render_pipeline(
            &device,
            wgpu::RenderPipelineDescriptor {
                label: Some("Debug Draw Pipeline Depth"),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs_main"),
                    compilation_options: Default::default(),
                    buffers: &[Some(wgpu::VertexBufferLayout {
                        array_stride: std::mem::size_of::<DebugVertex>() as u64,
                        step_mode: wgpu::VertexStepMode::Vertex,
                        attributes: &[
                            wgpu::VertexAttribute {
                                format: wgpu::VertexFormat::Float32x3,
                                offset: 0,
                                shader_location: 0,
                            },
                            wgpu::VertexAttribute {
                                format: wgpu::VertexFormat::Float32x4,
                                offset: 16,
                                shader_location: 1,
                            },
                        ],
                    })],
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some("fs_main"),
                    compilation_options: Default::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: target_format,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::LineList,
                    ..Default::default()
                },
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: wgpu::TextureFormat::Depth32Float,
                    depth_write_enabled: Some(false),
                    depth_compare: Some(helio_core::depth::NEARER_EQUAL),
                    stencil: wgpu::StencilState::default(),
                    bias: wgpu::DepthBiasState::default(),
                }),
                multisample: wgpu::MultisampleState::default(),
                multiview_mask: None,
                cache: None,
            },
        );

        let pipeline_no_depth = helio_core::shader::create_render_pipeline(
            &device,
            wgpu::RenderPipelineDescriptor {
                label: Some("Debug Draw Pipeline NoDepth"),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs_main"),
                    compilation_options: Default::default(),
                    buffers: &[Some(wgpu::VertexBufferLayout {
                        array_stride: std::mem::size_of::<DebugVertex>() as u64,
                        step_mode: wgpu::VertexStepMode::Vertex,
                        attributes: &[
                            wgpu::VertexAttribute {
                                format: wgpu::VertexFormat::Float32x3,
                                offset: 0,
                                shader_location: 0,
                            },
                            wgpu::VertexAttribute {
                                format: wgpu::VertexFormat::Float32x4,
                                offset: 16,
                                shader_location: 1,
                            },
                        ],
                    })],
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some("fs_main"),
                    compilation_options: Default::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: target_format,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::LineList,
                    ..Default::default()
                },
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                multiview_mask: None,
                cache: None,
            },
        );

        let tri_attribs = [
            wgpu::VertexAttribute {
                format: wgpu::VertexFormat::Float32x3,
                offset: 0,
                shader_location: 0,
            },
            wgpu::VertexAttribute {
                format: wgpu::VertexFormat::Float32x4,
                offset: 16,
                shader_location: 1,
            },
        ];
        let tri_vbl = wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<DebugVertex>() as u64,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &tri_attribs,
        };
        let tri_target = wgpu::ColorTargetState {
            format: target_format,
            blend: Some(wgpu::BlendState::ALPHA_BLENDING),
            write_mask: wgpu::ColorWrites::ALL,
        };
        let tri_prim = wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            cull_mode: None,
            ..Default::default()
        };

        let pipeline_tri_depth = helio_core::shader::create_render_pipeline(
            &device,
            wgpu::RenderPipelineDescriptor {
                label: Some("Debug Tri Pipeline Depth"),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs_main"),
                    compilation_options: Default::default(),
                    buffers: &[Some(tri_vbl.clone())],
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some("fs_main"),
                    compilation_options: Default::default(),
                    targets: &[Some(tri_target.clone())],
                }),
                primitive: tri_prim,
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: wgpu::TextureFormat::Depth32Float,
                    depth_write_enabled: Some(false),
                    depth_compare: Some(helio_core::depth::NEARER_EQUAL),
                    stencil: wgpu::StencilState::default(),
                    bias: wgpu::DepthBiasState::default(),
                }),
                multisample: wgpu::MultisampleState::default(),
                multiview_mask: None,
                cache: None,
            },
        );

        let pipeline_tri_no_depth = helio_core::shader::create_render_pipeline(
            &device,
            wgpu::RenderPipelineDescriptor {
                label: Some("Debug Tri Pipeline NoDepth"),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs_main"),
                    compilation_options: Default::default(),
                    buffers: &[Some(tri_vbl)],
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some("fs_main"),
                    compilation_options: Default::default(),
                    targets: &[Some(tri_target)],
                }),
                primitive: tri_prim,
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                multiview_mask: None,
                cache: None,
            },
        );

        let tri_buf = debug_vertex_buffer(device, "Debug Tri Buffer", MIN_DEBUG_VERTS);

        Self {
            pipeline_depth,
            pipeline_no_depth,
            pipeline_tri_depth,
            pipeline_tri_no_depth,
            bgl,
            camera_buf: camera_buf.clone(),
            bind_group: None,
            bind_group_key: None,
            vertex_buf,
            vertex_count: 0,
            tri_buf,
            tri_count: 0,
            depth_test_enabled: depth_test,
            use_scene_depth: false,
        }
    }

    /// Occlude against real scene geometry instead of drawing over everything.
    /// Only meaningful with depth testing on.
    pub fn set_use_scene_depth(&mut self, enabled: bool) {
        self.use_scene_depth = enabled;
    }

    /// Uploads line vertices (pairs), growing the buffer to fit up to
    /// `MAX_DEBUG_VERTS`.
    pub fn update_lines(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        verts: &[DebugVertex],
    ) {
        let count = verts.len().min(MAX_DEBUG_VERTS as usize);
        grow_debug_buffer(device, &mut self.vertex_buf, "Debug Vertex Buffer", count);
        if count > 0 {
            helio_core::upload::write_buffer(
                queue,
                &self.vertex_buf,
                0,
                bytemuck::cast_slice(&verts[..count]),
            );
        }
        self.vertex_count = count as u32;
    }

    /// Overwrites vertices from `first_vertex` within what the last
    /// [`update_lines`](Self::update_lines) made room for.
    pub fn update_lines_at(
        &mut self,
        queue: &wgpu::Queue,
        first_vertex: usize,
        verts: &[DebugVertex],
    ) {
        let capacity = (self.vertex_buf.size() as usize) / std::mem::size_of::<DebugVertex>();
        if verts.is_empty() || first_vertex >= capacity {
            return;
        }
        let max_writable = capacity.saturating_sub(first_vertex);
        let count = verts.len().min(max_writable);
        if count == 0 {
            return;
        }
        let offset_bytes = (first_vertex * std::mem::size_of::<DebugVertex>()) as u64;
        helio_core::upload::write_buffer(
            queue,
            &self.vertex_buf,
            offset_bytes,
            bytemuck::cast_slice(&verts[..count]),
        );
    }

    pub fn set_line_vertex_count(&mut self, count: usize) {
        self.vertex_count = count.min(MAX_DEBUG_VERTS as usize) as u32;
    }

    /// Uploads triangle vertices (triples), growing the buffer to fit up to
    /// `MAX_DEBUG_TRIS`.
    pub fn update_tris(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        verts: &[DebugVertex],
    ) {
        let count = verts.len().min(MAX_DEBUG_TRIS as usize);
        grow_debug_buffer(device, &mut self.tri_buf, "Debug Tri Buffer", count);
        if count > 0 {
            helio_core::upload::write_buffer(
                queue,
                &self.tri_buf,
                0,
                bytemuck::cast_slice(&verts[..count]),
            );
        }
        self.tri_count = count as u32;
    }

    pub fn clear(&mut self) {
        self.vertex_count = 0;
        self.tri_count = 0;
    }

    /// Drops the geometry and shrinks the buffers back to their starting
    /// size.
    fn release(&mut self, device: &wgpu::Device) {
        self.vertex_buf = debug_vertex_buffer(device, "Debug Vertex Buffer", MIN_DEBUG_VERTS);
        self.tri_buf = debug_vertex_buffer(device, "Debug Tri Buffer", MIN_DEBUG_VERTS);
        self.clear();
    }

    pub fn set_depth_test(&mut self, enabled: bool) {
        self.depth_test_enabled = enabled;
    }
}

impl DebugPass {
    fn draw_commands(&self, rp: &mut RenderCmds<'_>) {
        rp.set_bind_group(0, self.bind_group.as_ref().unwrap(), &[]);

        if self.vertex_count > 0 {
            if self.depth_test_enabled {
                rp.set_pipeline(&self.pipeline_depth);
            } else {
                rp.set_pipeline(&self.pipeline_no_depth);
            }
            rp.set_vertex_buffer(0, self.vertex_buf.slice(..));
            rp.draw(0..self.vertex_count, 0..1);
        }

        if self.tri_count > 0 {
            if self.depth_test_enabled {
                rp.set_pipeline(&self.pipeline_tri_depth);
            } else {
                rp.set_pipeline(&self.pipeline_tri_no_depth);
            }
            rp.set_vertex_buffer(0, self.tri_buf.slice(..));
            rp.draw(0..self.tri_count, 0..1);
        }
    }

    fn ensure_bind_group(&mut self, device: &wgpu::Device) {
        if self.bind_group_key.as_ref() != Some(&self.camera_buf) {
            self.bind_group = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Debug Draw BG"),
                layout: &self.bgl,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.camera_buf.as_entire_binding(),
                }],
            }));
            self.bind_group_key = Some(self.camera_buf.clone());
        }
    }

    pub fn execute_on_target(
        &mut self,
        ctx: &mut PassContext,
        target: &wgpu::TextureView,
    ) -> HelioResult<()> {
        if self.vertex_count == 0 && self.tri_count == 0 {
            return Ok(());
        }

        self.ensure_bind_group(ctx.device);

        let depth_attachment = if self.depth_test_enabled {
            let depth_view = if let Some(frd) = ctx
                .registry
                .get(helio_core::ResourceKey::new("full_res_depth"))
            {
                frd
            } else {
                ctx.depth
            };
            Some(wgpu::RenderPassDepthStencilAttachment {
                view: depth_view,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            })
        } else {
            None
        };

        let color_attachment = wgpu::RenderPassColorAttachment {
            view: target,
            resolve_target: None,
            depth_slice: None,
            ops: wgpu::Operations {
                load: wgpu::LoadOp::Load,
                store: wgpu::StoreOp::Store,
            },
        };
        let color_attachments = [Some(color_attachment)];
        let desc = wgpu::RenderPassDescriptor {
            label: Some("DebugDraw"),
            color_attachments: &color_attachments,
            depth_stencil_attachment: depth_attachment,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        };

        let mut cmds = ctx.graphics_cmds();
        let mut pass = cmds.begin_render_pass(&desc);
        self.draw_commands(&mut pass);
        Ok(())
    }
}

impl RenderPass for DebugPass {
    fn name(&self) -> &'static str {
        "DebugDraw"
    }

    fn render_pass_descriptor_with_storage<'a>(
        &'a self,
        target: &'a wgpu::TextureView,
        depth: &'a wgpu::TextureView,
        resources: &'a helio_core::ResourceRegistry<'a>,
        storage: &'a mut helio_core::RenderFrameStorage,
    ) -> Option<wgpu::RenderPassDescriptor<'a>> {
        // `pre_aa` is internal-resolution (render-scaled); `target` is full
        // output resolution. `use_scene_depth` wants internal-res depth
        // paired with internal-res color, but that pairing only holds when
        // `pre_aa` is actually available -- if it isn't, color must fall
        // back to full-res `target` and depth must follow it to
        // `full_res_depth`, or wgpu rejects the pass for mismatched
        // attachment extents whenever render_scale < 1.0.
        let pre_aa = if self.use_scene_depth {
            resources.get(helio_core::ResourceKey::new("pre_aa"))
        } else {
            None
        };
        let depth_attachment = if self.depth_test_enabled {
            // `full_res_depth` is a dummy: it is allocated so that passes
            // drawing at output resolution have a correctly-sized depth
            // attachment, but no pass ever renders scene depth into it. Testing
            // against it rejects every fragment. Real occlusion has to use the
            // internal-res scene depth, which means drawing at internal res too
            // — see `use_scene_depth` and BillboardPass::occluded_by_geometry,
            // which makes the same trade.
            let depth_view = if self.use_scene_depth && pre_aa.is_some() {
                depth
            } else if let Some(frd) = resources.get(helio_core::ResourceKey::new("full_res_depth"))
            {
                frd
            } else {
                depth
            };
            Some(wgpu::RenderPassDepthStencilAttachment {
                view: depth_view,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            })
        } else {
            None
        };
        // Scene depth is internal-res, so the colour target must be too, or the
        // attachments disagree in size.
        let color_view = pre_aa.unwrap_or(target);
        let color_attachments: &'a [Option<wgpu::RenderPassColorAttachment<'a>>] = storage
            .retain_boxed_slice(Box::new([Some(wgpu::RenderPassColorAttachment {
                view: color_view,
                resolve_target: None,
                depth_slice: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
            })]));
        Some(wgpu::RenderPassDescriptor {
            label: Some("DebugDraw"),
            color_attachments,
            depth_stencil_attachment: depth_attachment,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        })
    }

    fn prepare(&mut self, _ctx: &PrepareContext) -> HelioResult<()> {
        Ok(())
    }

    fn execute(&mut self, ctx: &mut PassContext) -> HelioResult<()> {
        if self.vertex_count == 0 && self.tri_count == 0 {
            return Ok(());
        }
        self.ensure_bind_group(ctx.device);
        let Some(mut rp) = ctx.render_cmds() else {
            // A parallel worker can legitimately reach an empty/omitted
            // render pass when the debug targets are unavailable for this
            // frame. Treat it as a no-op instead of poisoning the worker.
            return Ok(());
        };
        self.draw_commands(&mut rp);
        Ok(())
    }
}

pub struct DebugDrawPass {
    pass: DebugPass,
    state: Arc<Mutex<DebugDrawState>>,
    editor_mode: bool,
    grid_pipeline: wgpu::RenderPipeline,
    grid_uniform: wgpu::Buffer,
    grid_bind_group: wgpu::BindGroup,
    grid_depth_test: bool,
    cached_line_gen: u64,
    cached_tri_gen: u64,
    editor_volume_cache: Vec<WorldVertex>,
    /// Frame origin the uploaded editor overlay lines were rebased to.
    editor_origin: Option<glam::DVec3>,
    /// Frame origin the uploaded user geometry was rebased to.
    user_origin: Option<glam::DVec3>,
    /// `(editor_volume_generation, editor_layers_generation)` the cache was
    /// last built from.
    editor_last_volume_gen: Option<(u64, u64)>,
}

impl DebugDrawPass {
    pub fn new(
        device: &wgpu::Device,
        camera_buf: &wgpu::Buffer,
        surface_format: wgpu::TextureFormat,
        state: Arc<Mutex<DebugDrawState>>,
        depth_test: bool,
        editor_mode: bool,
    ) -> Self {
        let mut pass = DebugPass::new(device, camera_buf, surface_format, depth_test);
        // The editor overlay is meant to sit inside the scene, so it occludes
        // against real geometry rather than painting over it.
        pass.set_use_scene_depth(editor_mode && depth_test);

        let grid_shader = helio_core::shader::module(
            device,
            "Infinite Debug Grid Shader",
            helio_core::include_wgsl!("../../shaders/infinite_grid.wgsl"),
        );
        let grid_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Infinite Debug Grid BGL"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
        let grid_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Infinite Debug Grid Pipeline Layout"),
            bind_group_layouts: &[Some(&grid_bgl)],
            immediate_size: 0,
        });
        let grid_pipeline = helio_core::shader::create_render_pipeline(
            &device,
            wgpu::RenderPipelineDescriptor {
                label: Some("Infinite Debug Grid Pipeline"),
                layout: Some(&grid_layout),
                vertex: wgpu::VertexState {
                    module: &grid_shader,
                    entry_point: Some("vs_main"),
                    compilation_options: Default::default(),
                    buffers: &[],
                },
                fragment: Some(wgpu::FragmentState {
                    module: &grid_shader,
                    entry_point: Some("fs_main"),
                    compilation_options: Default::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: surface_format,
                        blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                primitive: wgpu::PrimitiveState::default(),
                // `infinite_grid.wgsl` writes `frag_depth`, so this pipeline must
                // always declare a depth attachment. It is only submitted by the
                // editor pass, whose render-pass descriptor supplies the matching
                // depth view. The ordinary debug pass constructs this pipeline but
                // never uses it.
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: wgpu::TextureFormat::Depth32Float,
                    depth_write_enabled: Some(false),
                    depth_compare: Some(helio_core::depth::NEARER_EQUAL),
                    stencil: wgpu::StencilState::default(),
                    bias: helio_core::depth::bias_toward_camera(1, 1.0, 0.0),
                }),
                multisample: wgpu::MultisampleState::default(),
                multiview_mask: None,
                cache: None,
            },
        );
        let grid_uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Infinite Debug Grid Uniform"),
            size: std::mem::size_of::<InfiniteGridUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let grid_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Infinite Debug Grid Bind Group"),
            layout: &grid_bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: grid_uniform.as_entire_binding(),
            }],
        });
        Self {
            pass,
            state,
            editor_mode,
            grid_pipeline,
            grid_uniform,
            grid_bind_group,
            grid_depth_test: depth_test,
            cached_line_gen: u64::MAX,
            cached_tri_gen: u64::MAX,
            editor_volume_cache: Vec::new(),
            editor_origin: None,
            user_origin: None,
            editor_last_volume_gen: None,
        }
    }

    fn update_infinite_grid(&self, ctx: &PrepareContext, origin: glam::DVec3) {
        let camera = ctx.camera_data;
        // Frame coordinates are relative to `origin`; the grid lies on
        // world y = 0. Line positions only need the origin modulo the largest
        // grid period, which keeps them exact in f32 far from the origin.
        let period = 1.0e7;
        let uniform = InfiniteGridUniform {
            inv_view_proj: camera.inv_view_proj,
            view_proj: camera.view_proj,
            camera_position: camera.position_near,
            viewport: [ctx.width as f32, ctx.height as f32, 0.0, 0.0],
            origin: [
                origin.x.rem_euclid(period) as f32,
                (-origin.y) as f32,
                origin.z.rem_euclid(period) as f32,
                0.0,
            ],
            axes: [(-origin.x) as f32, (-origin.z) as f32, 0.0, 0.0],
        };
        ctx.write_buffer(&self.grid_uniform, 0, bytemuck::bytes_of(&uniform));
    }

    fn draw_infinite_grid(&self, ctx: &mut PassContext) {
        let Some(mut rp) = ctx.render_cmds() else {
            return;
        };
        rp.set_pipeline(&self.grid_pipeline);
        rp.set_bind_group(0, &self.grid_bind_group, &[]);
        rp.draw(0..3, 0..1);
    }

    pub fn set_depth_test(&mut self, enabled: bool) {
        self.pass.set_depth_test(enabled);
    }

    fn rebuild_editor_volume_cache(&mut self, volume_lines: &[WorldVertex]) {
        // The grid is generated only by `infinite_grid.wgsl`. This cache is
        // exclusively for finite scene-volume bounds, which remain ordinary
        // debug geometry and must not be folded into the grid draw.
        self.editor_volume_cache.clear();
        self.editor_volume_cache.extend_from_slice(volume_lines);
    }
}

impl RenderPass for DebugDrawPass {
    fn name(&self) -> &'static str {
        "DebugDraw"
    }

    /// The editor overlay is a member while the editor draws its overlay;
    /// the user instance while there is user debug geometry.
    fn is_member(&mut self, _scene: &mut helio_core::SceneContent<'_>) -> bool {
        let Ok(state) = self.state.lock() else {
            return false;
        };
        if self.editor_mode {
            state.editor_enabled
        } else {
            !state.user_lines.is_empty() || !state.user_tris.is_empty()
        }
    }

    fn on_leave(&mut self, device: &wgpu::Device) {
        // Uploaded again, into small buffers, when the pass joins.
        self.pass.release(device);
        self.cached_line_gen = u64::MAX;
        self.cached_tri_gen = u64::MAX;
        self.editor_origin = None;
        self.user_origin = None;
        self.editor_last_volume_gen = None;
    }

    fn render_pass_descriptor<'a>(
        &'a self,
        target: &'a wgpu::TextureView,
        depth: &'a wgpu::TextureView,
        resources: &'a helio_core::ResourceRegistry<'a>,
    ) -> Option<wgpu::RenderPassDescriptor<'a>> {
        self.pass.render_pass_descriptor(target, depth, resources)
    }

    fn render_pass_descriptor_with_storage<'a>(
        &'a self,
        target: &'a wgpu::TextureView,
        depth: &'a wgpu::TextureView,
        resources: &'a helio_core::ResourceRegistry<'a>,
        storage: &'a mut helio_core::RenderFrameStorage,
    ) -> Option<wgpu::RenderPassDescriptor<'a>> {
        self.pass
            .render_pass_descriptor_with_storage(target, depth, resources, storage)
    }

    fn prepare(&mut self, ctx: &PrepareContext) -> HelioResult<()> {
        let state_arc = Arc::clone(&self.state);
        let state = state_arc.lock().unwrap();

        if self.editor_mode {
            let editor_enabled = state.editor_enabled;
            let volume_gen = state.editor_volume_generation;

            if !editor_enabled {
                if self.cached_line_gen != 0 {
                    self.pass.update_lines(ctx.device, ctx.queue, &[]);
                    self.cached_line_gen = 0;
                }
                self.editor_origin = None;
                self.pass.update_tris(ctx.device, ctx.queue, &[]);
                self.cached_tri_gen = 0;
                return Ok(());
            }

            // The grid itself is procedural and is drawn as a single
            // fullscreen triangle.
            self.update_infinite_grid(ctx, state.world_origin);

            // Overlay lines (volume bounds, editor line layers) are world
            // space, drawn relative to the frame origin (subtracted in double
            // precision) and re-uploaded when the origin moves.
            let origin = state.world_origin;
            let layers_gen = state.editor_layers_generation;
            let rebuilt = self.editor_last_volume_gen != Some((volume_gen, layers_gen));
            if rebuilt {
                self.rebuild_editor_volume_cache(&state.editor_volume_lines);
                for layer in state.editor_line_layers.values() {
                    self.editor_volume_cache.extend_from_slice(layer);
                }
                self.editor_last_volume_gen = Some((volume_gen, layers_gen));
            }
            drop(state);
            if rebuilt || self.editor_origin != Some(origin) || self.cached_line_gen == u64::MAX {
                let lines: Vec<DebugVertex> = self
                    .editor_volume_cache
                    .iter()
                    .map(|v| v.in_frame(origin))
                    .collect();
                self.pass.update_lines(ctx.device, ctx.queue, &lines);
                self.editor_origin = Some(origin);
                self.cached_line_gen = self.cached_line_gen.wrapping_add(1);
                if self.cached_line_gen == u64::MAX {
                    self.cached_line_gen = 1;
                }
            }

            if self.cached_tri_gen != 0 {
                self.pass.update_tris(ctx.device, ctx.queue, &[]);
                self.cached_tri_gen = 0;
            }
            return Ok(());
        }

        let user_lines_generation = state.user_lines_generation;
        let user_tris_generation = state.user_tris_generation;
        // User geometry is world space too (see the editor overlay above).
        let origin = state.world_origin;
        let moved = self.user_origin != Some(origin);
        let rebase = |vertices: &[WorldVertex]| -> Vec<DebugVertex> {
            vertices.iter().map(|v| v.in_frame(origin)).collect()
        };
        if user_lines_generation != self.cached_line_gen || moved {
            self.pass
                .update_lines(ctx.device, ctx.queue, &rebase(&state.user_lines));
            self.cached_line_gen = user_lines_generation;
        }
        if user_tris_generation != self.cached_tri_gen || moved {
            self.pass
                .update_tris(ctx.device, ctx.queue, &rebase(&state.user_tris));
            self.cached_tri_gen = user_tris_generation;
        }
        self.user_origin = Some(origin);
        Ok(())
    }

    fn execute(&mut self, ctx: &mut PassContext) -> HelioResult<()> {
        // No target override here: the graph has already begun the render pass
        // from render_pass_descriptor() by this point, so reassigning
        // ctx.target would change nothing. Where this draws is decided purely
        // by where the pass sits in the graph.
        self.pass.execute(ctx)?;
        if self.editor_mode {
            let enabled = self
                .state
                .lock()
                .map(|state| state.editor_enabled)
                .unwrap_or(false);
            if enabled && self.grid_depth_test {
                self.draw_infinite_grid(ctx);
            }
        }
        Ok(())
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct InfiniteGridUniform {
    inv_view_proj: [f32; 16],
    view_proj: [f32; 16],
    camera_position: [f32; 4],
    viewport: [f32; 4],
    /// x, z: world origin modulo the grid period; y: frame-local height of
    /// the world y = 0 plane.
    origin: [f32; 4],
    /// x, z: frame-local position of the world x = 0 and z = 0 axes.
    axes: [f32; 4],
}

impl Renderer {
    /// Rebuild the editor-only wireframe cache for bounded post-process volumes
    /// (world-space boxes, `[min, max]`).
    ///
    /// The cache is consumed by `DebugDrawPass` only in editor mode, so this is
    /// intentionally kept separate from transient user debug geometry.
    pub fn debug_set_editor_volume_bounds(&mut self, bounds: &[[glam::DVec3; 2]]) {
        const COLOR: [f32; 4] = [0.15, 0.8, 1.0, 1.0];
        const EDGES: [(usize, usize); 12] = [
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 0),
            (4, 5),
            (5, 6),
            (6, 7),
            (7, 4),
            (0, 4),
            (1, 5),
            (2, 6),
            (3, 7),
        ];
        let mut lines = Vec::with_capacity(bounds.len() * 24);
        for [min, max] in bounds {
            let corner = |x: bool, y: bool, z: bool| {
                glam::DVec3::new(
                    if x { max.x } else { min.x },
                    if y { max.y } else { min.y },
                    if z { max.z } else { min.z },
                )
            };
            let corners = [
                corner(false, false, false),
                corner(true, false, false),
                corner(true, true, false),
                corner(false, true, false),
                corner(false, false, true),
                corner(true, false, true),
                corner(true, true, true),
                corner(false, true, true),
            ];
            for (a, b) in EDGES {
                lines.push(WorldVertex::new(corners[a], COLOR));
                lines.push(WorldVertex::new(corners[b], COLOR));
            }
        }

        if let Ok(mut state) = self.debug_state.lock() {
            if state.editor_volume_lines != lines {
                state.editor_volume_lines = lines;
                state.editor_volume_generation = state.editor_volume_generation.wrapping_add(1);
            }
        }
    }

    /// Replace the editor line layer `layer` with `lines` (world space, as
    /// pairs). An empty set removes the layer. Consumed by `DebugDrawPass`
    /// in editor mode only, and re-uploaded only when a layer changes.
    pub fn debug_set_editor_lines(&mut self, layer: &str, lines: Vec<WorldVertex>) {
        let Ok(mut state) = self.debug_state.lock() else {
            return;
        };
        let mut lines = lines;
        if !lines.is_empty() {
            if !state.editor_line_layers.contains_key(layer)
                && state.editor_line_layers.len() >= MAX_EDITOR_LINE_LAYERS
            {
                return;
            }
            let other_vertices: usize = state
                .editor_line_layers
                .iter()
                .filter(|(name, _)| name.as_str() != layer)
                .map(|(_, vertices)| vertices.len())
                .sum();
            let available = MAX_EDITOR_LAYER_VERTS.saturating_sub(other_vertices);
            lines.truncate(available - available % 2);
        }
        let changed = if lines.is_empty() {
            state.editor_line_layers.remove(layer).is_some()
        } else if state.editor_line_layers.get(layer) != Some(&lines) {
            state.editor_line_layers.insert(layer.to_string(), lines);
            true
        } else {
            false
        };
        if changed {
            state.editor_layers_generation = state.editor_layers_generation.wrapping_add(1);
        }
    }

    pub fn debug_clear(&mut self) {
        if let Ok(mut s) = self.debug_state.lock() {
            if !s.user_lines.is_empty() {
                s.user_lines_generation = s.user_lines_generation.wrapping_add(1);
            }
            s.user_lines.clear();
            if !s.user_tris.is_empty() {
                s.user_tris_generation = s.user_tris_generation.wrapping_add(1);
            }
            s.user_tris.clear();
        }
    }

    /// Submit debug geometry under one lock (see [`DebugBatch`]). The
    /// `debug_*` shape methods each submit one shape this way.
    pub fn debug_batch<F>(&mut self, f: F)
    where
        F: FnOnce(&mut DebugBatch<'_>),
    {
        if let Ok(mut s) = self.debug_state.lock() {
            let mut batch = DebugBatch {
                state: &mut s,
                lines_changed: false,
                tris_changed: false,
            };
            f(&mut batch);
            batch.finish();
        }
    }

    pub fn debug_line(&mut self, from: impl WorldPoint, to: impl WorldPoint, color: [f32; 4]) {
        self.debug_batch(|b| b.line(from, to, color));
    }

    pub fn debug_tri(
        &mut self,
        v0: impl WorldPoint,
        v1: impl WorldPoint,
        v2: impl WorldPoint,
        color: [f32; 4],
    ) {
        self.debug_batch(|b| b.tri(v0, v1, v2, color));
    }

    pub fn debug_filled_disk(
        &mut self,
        center: impl WorldPoint,
        normal: [f32; 3],
        radius: f32,
        color: [f32; 4],
        segments: u32,
    ) {
        self.debug_batch(|b| b.filled_disk(center, normal, radius, color, segments));
    }

    pub fn debug_filled_cone(
        &mut self,
        apex: impl WorldPoint,
        axis: [f32; 3],
        height: f32,
        base_radius: f32,
        color: [f32; 4],
        segments: u32,
    ) {
        self.debug_batch(|b| b.filled_cone(apex, axis, height, base_radius, color, segments));
    }

    pub fn debug_filled_box(&mut self, center: impl WorldPoint, half: f32, color: [f32; 4]) {
        self.debug_batch(|b| b.filled_box(center, half, color));
    }

    pub fn debug_circle(
        &mut self,
        center: impl WorldPoint,
        radius: f32,
        color: [f32; 4],
        segments: u32,
    ) {
        self.debug_batch(|b| b.circle(center, radius, color, segments));
    }

    pub fn debug_sphere(
        &mut self,
        center: impl WorldPoint,
        radius: f32,
        color: [f32; 4],
        segments: u32,
    ) {
        self.debug_batch(|b| b.sphere(center, radius, color, segments));
    }

    pub fn debug_torus(
        &mut self,
        center: impl WorldPoint,
        normal: [f32; 3],
        major_radius: f32,
        minor_radius: f32,
        color: [f32; 4],
        major_segments: u32,
        minor_segments: u32,
    ) {
        self.debug_batch(|b| {
            b.torus(
                center,
                normal,
                major_radius,
                minor_radius,
                color,
                major_segments,
                minor_segments,
            )
        });
    }

    pub fn debug_cylinder(
        &mut self,
        base_center: impl WorldPoint,
        axis: [f32; 3],
        height: f32,
        radius: f32,
        color: [f32; 4],
        segments: u32,
    ) {
        self.debug_batch(|b| b.cylinder(base_center, axis, height, radius, color, segments));
    }

    pub fn debug_cone(
        &mut self,
        apex: impl WorldPoint,
        axis: [f32; 3],
        height: f32,
        base_radius: f32,
        color: [f32; 4],
        segments: u32,
    ) {
        self.debug_batch(|b| b.cone(apex, axis, height, base_radius, color, segments));
    }

    pub fn debug_frustum(
        &mut self,
        origin: impl WorldPoint,
        forward: [f32; 3],
        up: [f32; 3],
        fov_y: f32,
        aspect: f32,
        near: f32,
        far: f32,
        color: [f32; 4],
    ) {
        self.debug_batch(|b| b.frustum(origin, forward, up, fov_y, aspect, near, far, color));
    }
}
