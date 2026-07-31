//! GPU-instanced 2D sprite batcher with texture atlasing.
//!
//! One unit quad drawn instanced against a per-frame instance buffer — the
//! CPU cost of a frame is `instances.len()` pushes into a `Vec`, not one draw
//! call per sprite. Every frame, [`SpriteBatchPass::prepare`]:
//!
//! 1. Culls instances whose bounding circle doesn't intersect the camera's
//!    view rect (see [`SpriteBatchPass::set_camera`]).
//! 2. Sorts the survivors back-to-front by [`SpriteInstance::depth`], since
//!    the sprites are alpha-blended (see the note on [`SpriteInstance::depth`]
//!    for why this is a CPU sort rather than a hardware depth test).
//! 3. Uploads the result to the GPU instance buffer.
//!
//! The atlas is a texture array (see [`SpriteBatchPass::add_atlas_layer`]),
//! not a single texture, so one draw call can span many sprite sheets.
//!
//! This pass has no dependency on `helio` / `helio-default-graphs` — it only
//! needs `helio-core` (for the [`RenderPass`] trait and graph plumbing) and
//! `libhelio` (for [`FrameResources`](libhelio::FrameResources)), matching
//! every other pass crate. It never touches `PassContext::scene`, so it is
//! usable inside a `RenderGraph` that carries no 3D scene data at all.

use bytemuck::{Pod, Zeroable};
use helio_core::{PassContext, PrepareContext, RenderPass, Result};
use wgpu::util::DeviceExt;

/// A single sprite's transform, atlas region, and tint, uploaded verbatim as
/// one per-instance vertex-buffer record.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct SpriteInstance {
    /// World-space center, in world units (1 unit == 1 pixel at zoom 1.0).
    pub position: [f32; 2],
    /// Width/height in world units, before rotation.
    pub size: [f32; 2],
    /// Rotation about the sprite's own center, radians.
    pub rotation: f32,
    /// CPU-side back-to-front sort key (larger draws later, i.e. on top).
    /// Not a hardware depth value: alpha-blended sprites can't rely on a
    /// depth test for correct compositing (blending is order-dependent
    /// regardless of what the depth buffer says), so `prepare()` sorts the
    /// instance list by this field instead of enabling `DepthStencilState`.
    /// A common convention is "world Y" for a top-down/2.5D look (larger Y
    /// = further back = drawn first), but any scalar works.
    pub depth: f32,
    /// Atlas UV rectangle: `[u0, v0, u1, v1]`, each in `0..1`, within
    /// `atlas_layer`.
    pub uv_rect: [f32; 4],
    /// Straight-alpha RGBA tint, multiplied against the sampled atlas texel.
    pub color: [f32; 4],
    /// Index into the atlas texture array — see
    /// [`SpriteBatchPass::add_atlas_layer`].
    pub atlas_layer: u32,
    _pad1: [u32; 3],
}

impl SpriteInstance {
    pub fn new(position: [f32; 2], size: [f32; 2]) -> Self {
        Self {
            position,
            size,
            rotation: 0.0,
            depth: 0.0,
            uv_rect: [0.0, 0.0, 1.0, 1.0],
            color: [1.0, 1.0, 1.0, 1.0],
            atlas_layer: 0,
            _pad1: [0; 3],
        }
    }

    pub fn with_rotation(mut self, radians: f32) -> Self {
        self.rotation = radians;
        self
    }

    pub fn with_depth(mut self, depth: f32) -> Self {
        self.depth = depth;
        self
    }

    pub fn with_uv_rect(mut self, uv_rect: [f32; 4]) -> Self {
        self.uv_rect = uv_rect;
        self
    }

    pub fn with_color(mut self, color: [f32; 4]) -> Self {
        self.color = color;
        self
    }

    pub fn with_atlas_layer(mut self, layer: u32) -> Self {
        self.atlas_layer = layer;
        self
    }

    /// Conservative bounding-circle radius (half the quad's diagonal), used
    /// for view-frustum culling. Covers the sprite at any rotation.
    fn cull_radius(&self) -> f32 {
        0.5 * (self.size[0] * self.size[0] + self.size[1] * self.size[1]).sqrt()
    }
}

const INSTANCE_STRIDE: u64 = std::mem::size_of::<SpriteInstance>() as u64;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct QuadVertex {
    pos: [f32; 2],
    uv: [f32; 2],
}

// Y-up world space; v=0 at the top of the quad to match texture-space UVs.
const QUAD_VERTICES: [QuadVertex; 4] = [
    QuadVertex { pos: [-0.5, -0.5], uv: [0.0, 1.0] },
    QuadVertex { pos: [0.5, -0.5], uv: [1.0, 1.0] },
    QuadVertex { pos: [-0.5, 0.5], uv: [0.0, 0.0] },
    QuadVertex { pos: [0.5, 0.5], uv: [1.0, 0.0] },
];
const QUAD_INDICES: [u16; 6] = [0, 1, 2, 2, 1, 3];

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct CameraUniform {
    view_proj: [[f32; 4]; 4],
}

const ATLAS_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8UnormSrgb;

/// GPU-instanced 2D sprite batch pass.
///
/// Owns its own orthographic camera (recomputed from the render target size
/// every frame — call [`SpriteBatchPass::set_camera`] before `prepare()` to
/// override framing/zoom/pan) and starts with a 1×1 white fallback atlas
/// layer, so it renders solid-colored quads out of the box; call
/// [`SpriteBatchPass::add_atlas_layer`] to load real sprite sheets.
pub struct SpriteBatchPass {
    pipeline: wgpu::RenderPipeline,
    bgl: wgpu::BindGroupLayout,
    bind_group: Option<wgpu::BindGroup>,
    sampler: wgpu::Sampler,

    atlas_texture: wgpu::Texture,
    atlas_view: wgpu::TextureView,
    /// `true` until the first [`SpriteBatchPass::add_atlas_layer`] call — the
    /// pass is still serving the 1×1 white fallback and hasn't committed to
    /// a real atlas width/height yet.
    atlas_using_fallback: bool,
    atlas_width: u32,
    atlas_height: u32,
    atlas_layer_count: u32,
    atlas_layer_capacity: u32,

    camera_buf: wgpu::Buffer,
    quad_vertex_buf: wgpu::Buffer,
    quad_index_buf: wgpu::Buffer,
    instance_buf: wgpu::Buffer,
    instance_capacity: usize,

    /// Raw per-frame push list, in caller-supplied order. Cleared by
    /// [`SpriteBatchPass::clear`], not by `prepare()`.
    instances: Vec<SpriteInstance>,
    /// Culled + depth-sorted subset of `instances`, rebuilt every `prepare()`
    /// and reused across frames to avoid a fresh allocation each time.
    visible_scratch: Vec<SpriteInstance>,
    instance_count: u32,

    /// Half-extent of the orthographic view, in world units. `None` means
    /// "derive from the render target's pixel size" (1 world unit = 1 pixel).
    camera_half_extent: Option<[f32; 2]>,
    camera_center: [f32; 2],
    clear_color: Option<wgpu::Color>,
}

impl SpriteBatchPass {
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue, surface_format: wgpu::TextureFormat) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Sprite Batch Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/sprite.wgsl").into()),
        });

        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Sprite Batch BGL"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2Array,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Sprite Batch PL"),
            bind_group_layouts: &[Some(&bgl)],
            immediate_size: 0,
        });

        let vertex_layout = wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<QuadVertex>() as u64,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x2, offset: 0, shader_location: 0 },
                wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x2, offset: 8, shader_location: 1 },
            ],
        };
        let instance_layout = wgpu::VertexBufferLayout {
            array_stride: INSTANCE_STRIDE,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &[
                wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x2, offset: 0, shader_location: 2 },
                wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x2, offset: 8, shader_location: 3 },
                wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32, offset: 16, shader_location: 4 },
                wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32, offset: 20, shader_location: 5 },
                wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x4, offset: 24, shader_location: 6 },
                wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x4, offset: 40, shader_location: 7 },
                wgpu::VertexAttribute { format: wgpu::VertexFormat::Uint32, offset: 56, shader_location: 8 },
            ],
        };

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Sprite Batch Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[Some(vertex_layout), Some(instance_layout)],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_format,
                    // Straight-alpha atlas texels over an opaque or transparent
                    // target; premultiplied atlases should premultiply at import
                    // time and use `BlendState::REPLACE` via a future variant.
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        let camera_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Sprite Camera Uniform"),
            size: std::mem::size_of::<CameraUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let quad_vertex_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Sprite Quad Vertices"),
            contents: bytemuck::cast_slice(&QUAD_VERTICES),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let quad_index_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Sprite Quad Indices"),
            contents: bytemuck::cast_slice(&QUAD_INDICES),
            usage: wgpu::BufferUsages::INDEX,
        });

        let initial_capacity = 256usize;
        let instance_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Sprite Instance Buffer"),
            size: INSTANCE_STRIDE * initial_capacity as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Sprite Atlas Sampler"),
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            ..Default::default()
        });

        let atlas_texture = create_atlas_array_texture(device, 1, 1, 1);
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &atlas_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &[255u8, 255, 255, 255],
            wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(4), rows_per_image: Some(1) },
            wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
        );
        let atlas_view = atlas_texture.create_view(&wgpu::TextureViewDescriptor {
            dimension: Some(wgpu::TextureViewDimension::D2Array),
            ..Default::default()
        });

        Self {
            pipeline,
            bgl,
            bind_group: None,
            sampler,
            atlas_texture,
            atlas_view,
            atlas_using_fallback: true,
            atlas_width: 1,
            atlas_height: 1,
            atlas_layer_count: 1,
            atlas_layer_capacity: 1,
            camera_buf,
            quad_vertex_buf,
            quad_index_buf,
            instance_buf,
            instance_capacity: initial_capacity,
            instances: Vec::new(),
            visible_scratch: Vec::new(),
            instance_count: 0,
            camera_half_extent: None,
            camera_center: [0.0, 0.0],
            clear_color: Some(wgpu::Color::BLACK),
        }
    }

    /// Uploads `rgba8` (must be exactly `width * height * 4` bytes, straight
    /// alpha) as a new layer in the atlas array, growing the array (and
    /// invalidating the cached bind group, rebuilt lazily on the next
    /// `execute()`) if it's out of capacity. Returns the new layer's index
    /// for [`SpriteInstance::with_atlas_layer`].
    ///
    /// Every layer in a texture array must share one width/height — the
    /// first call after construction fixes that size for the pass's
    /// lifetime; subsequent calls with a different `width`/`height` panic.
    /// Pack sprites that don't already share a sheet size into same-size
    /// atlas pages before calling this (a texture-array layer boundary is
    /// not a UV-rect boundary — `uv_rect` still addresses within one layer).
    pub fn add_atlas_layer(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, width: u32, height: u32, rgba8: &[u8]) -> u32 {
        assert_eq!(
            rgba8.len() as u32,
            width * height * 4,
            "rgba8 length {} does not match width*height*4 ({width}*{height}*4 = {})",
            rgba8.len(),
            width * height * 4
        );

        if self.atlas_using_fallback {
            self.atlas_width = width;
            self.atlas_height = height;
            self.atlas_layer_capacity = 4;
            self.atlas_layer_count = 0;
            self.atlas_texture = create_atlas_array_texture(device, width, height, self.atlas_layer_capacity);
            self.atlas_view = self.atlas_texture.create_view(&wgpu::TextureViewDescriptor {
                dimension: Some(wgpu::TextureViewDimension::D2Array),
                ..Default::default()
            });
            self.atlas_using_fallback = false;
            self.bind_group = None;
        } else {
            assert_eq!(
                width, self.atlas_width,
                "atlas layer width {width} does not match the array's existing width {} \
                 (all layers in one SpriteBatchPass must share dimensions)",
                self.atlas_width
            );
            assert_eq!(
                height, self.atlas_height,
                "atlas layer height {height} does not match the array's existing height {} \
                 (all layers in one SpriteBatchPass must share dimensions)",
                self.atlas_height
            );
        }

        if self.atlas_layer_count >= self.atlas_layer_capacity {
            self.grow_atlas_array(device, queue);
        }

        let layer = self.atlas_layer_count;
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &self.atlas_texture,
                mip_level: 0,
                origin: wgpu::Origin3d { x: 0, y: 0, z: layer },
                aspect: wgpu::TextureAspect::All,
            },
            rgba8,
            wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(width * 4), rows_per_image: Some(height) },
            wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
        );
        self.atlas_layer_count += 1;
        layer
    }

    fn grow_atlas_array(&mut self, device: &wgpu::Device, queue: &wgpu::Queue) {
        let new_capacity = self.atlas_layer_capacity * 2;
        let new_texture = create_atlas_array_texture(device, self.atlas_width, self.atlas_height, new_capacity);
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Sprite Atlas Array Grow"),
        });
        encoder.copy_texture_to_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &self.atlas_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyTextureInfo {
                texture: &new_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::Extent3d { width: self.atlas_width, height: self.atlas_height, depth_or_array_layers: self.atlas_layer_count },
        );
        queue.submit([encoder.finish()]);

        self.atlas_texture = new_texture;
        self.atlas_view = self.atlas_texture.create_view(&wgpu::TextureViewDescriptor {
            dimension: Some(wgpu::TextureViewDimension::D2Array),
            ..Default::default()
        });
        self.atlas_layer_capacity = new_capacity;
        self.bind_group = None;
    }

    /// Escape hatch: replace atlas management entirely with a caller-owned
    /// `D2Array` view (e.g. loaded via a dedicated atlas-packing tool).
    /// `view`/`sampler` must outlive every subsequent `execute()` call until
    /// the next `set_atlas_array`/`add_atlas_layer`.
    pub fn set_atlas_array(&mut self, device: &wgpu::Device, view: &wgpu::TextureView, sampler: &wgpu::Sampler) {
        self.bind_group = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Sprite Batch BG"),
            layout: &self.bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: self.camera_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(view) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(sampler) },
            ],
        }));
    }

    /// Overrides the orthographic view: `center` and `half_extent` in world
    /// units. `half_extent = None` re-derives it from the render target's
    /// pixel dimensions each frame (1 world unit = 1 pixel, origin centered).
    pub fn set_camera(&mut self, center: [f32; 2], half_extent: Option<[f32; 2]>) {
        self.camera_center = center;
        self.camera_half_extent = half_extent;
    }

    /// `None` disables the clear (loads the existing target contents);
    /// `Some(color)` clears to that color every frame. Defaults to opaque black.
    pub fn set_clear_color(&mut self, color: Option<wgpu::Color>) {
        self.clear_color = color;
    }

    pub fn clear(&mut self) {
        self.instances.clear();
    }

    pub fn push_sprite(&mut self, instance: SpriteInstance) {
        self.instances.push(instance);
    }

    /// Total sprites pushed this frame, before culling.
    pub fn sprite_count(&self) -> usize {
        self.instances.len()
    }

    /// Sprites that survived view-frustum culling last `prepare()` and were
    /// actually uploaded/drawn.
    pub fn visible_count(&self) -> usize {
        self.visible_scratch.len()
    }
}

fn create_atlas_array_texture(device: &wgpu::Device, width: u32, height: u32, layers: u32) -> wgpu::Texture {
    device.create_texture(&wgpu::TextureDescriptor {
        label: Some("Sprite Atlas Array"),
        size: wgpu::Extent3d { width, height, depth_or_array_layers: layers },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: ATLAS_FORMAT,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    })
}

impl RenderPass for SpriteBatchPass {
    fn name(&self) -> &'static str {
        "SpriteBatch"
    }

    fn render_pass_descriptor<'a>(
        &'a self,
        target: &'a wgpu::TextureView,
        _depth: &'a wgpu::TextureView,
        _resources: &'a libhelio::FrameResources<'a>,
    ) -> Option<wgpu::RenderPassDescriptor<'a>> {
        // 2D sprites are alpha-blended and CPU-sorted (see `SpriteInstance::depth`)
        // — no depth attachment. `Box::leak` here matches the convention used by
        // every other executor-managed pass: the descriptor only needs to live
        // for this frame's `execute()` call, and the executor drops it before
        // the next `render_pass_descriptor()`.
        let load = match self.clear_color {
            Some(color) => wgpu::LoadOp::Clear(color),
            None => wgpu::LoadOp::Load,
        };
        let attachments: &'a [Option<wgpu::RenderPassColorAttachment<'a>>] =
            Box::leak(Box::new([Some(wgpu::RenderPassColorAttachment {
                view: target,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations { load, store: wgpu::StoreOp::Store },
            })]));
        Some(wgpu::RenderPassDescriptor {
            label: Some("Sprite Batch Pass"),
            color_attachments: attachments,
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        })
    }

    fn prepare(&mut self, ctx: &PrepareContext) -> Result<()> {
        let half_extent = self.camera_half_extent.unwrap_or([ctx.width as f32 * 0.5, ctx.height as f32 * 0.5]);
        let [cx, cy] = self.camera_center;
        let [hx, hy] = half_extent;
        // Y-up world space; a Y-down convention would swap `cy - hy`/`cy + hy`.
        let view_proj = glam::Mat4::orthographic_rh(cx - hx, cx + hx, cy - hy, cy + hy, -1.0, 1.0);
        let uniform = CameraUniform { view_proj: view_proj.to_cols_array_2d() };
        ctx.write_buffer(&self.camera_buf, 0, bytemuck::bytes_of(&uniform));

        // ── Cull ────────────────────────────────────────────────────────────
        // Circle-vs-AABB: clamp the sprite's center into the view rect, then
        // compare the distance to that clamped point against the sprite's
        // bounding radius. Conservative (uses the quad's full diagonal, so a
        // rotated sprite is never culled early) but cheap — one clamp, one
        // sqrt, per instance.
        let view_min = [cx - hx, cy - hy];
        let view_max = [cx + hx, cy + hy];
        self.visible_scratch.clear();
        self.visible_scratch.extend(self.instances.iter().copied().filter(|inst| {
            let clamped = [
                inst.position[0].clamp(view_min[0], view_max[0]),
                inst.position[1].clamp(view_min[1], view_max[1]),
            ];
            let dx = inst.position[0] - clamped[0];
            let dy = inst.position[1] - clamped[1];
            (dx * dx + dy * dy).sqrt() <= inst.cull_radius()
        }));

        // ── Sort ────────────────────────────────────────────────────────────
        // Back-to-front by depth; see `SpriteInstance::depth` for why this is
        // a CPU sort rather than a hardware depth test.
        self.visible_scratch
            .sort_by(|a, b| a.depth.partial_cmp(&b.depth).unwrap_or(std::cmp::Ordering::Equal));

        // ── Upload ──────────────────────────────────────────────────────────
        if self.visible_scratch.len() > self.instance_capacity {
            self.instance_capacity = (self.visible_scratch.len() * 2).max(self.instance_capacity * 2);
            self.instance_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Sprite Instance Buffer"),
                size: INSTANCE_STRIDE * self.instance_capacity as u64,
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
        }
        if !self.visible_scratch.is_empty() {
            ctx.write_buffer(&self.instance_buf, 0, bytemuck::cast_slice(&self.visible_scratch));
        }
        self.instance_count = self.visible_scratch.len() as u32;

        Ok(())
    }

    fn execute(&mut self, ctx: &mut PassContext) -> Result<()> {
        if self.instance_count == 0 {
            return Ok(());
        }

        if self.bind_group.is_none() {
            self.bind_group = Some(ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Sprite Batch BG"),
                layout: &self.bgl,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: self.camera_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&self.atlas_view) },
                    wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(&self.sampler) },
                ],
            }));
        }

        let Some(rp_ptr) = ctx.active_render_pass_ptr() else {
            return Ok(());
        };
        let rp = unsafe { &mut *rp_ptr };
        rp.set_pipeline(&self.pipeline);
        rp.set_bind_group(0, self.bind_group.as_ref().unwrap(), &[]);
        rp.set_vertex_buffer(0, self.quad_vertex_buf.slice(..));
        rp.set_vertex_buffer(1, self.instance_buf.slice(..));
        rp.set_index_buffer(self.quad_index_buf.slice(..), wgpu::IndexFormat::Uint16);
        rp.draw_indexed(0..6, 0, 0..self.instance_count);

        Ok(())
    }
}
