//! GPU-instanced 2D sprite batcher with texture atlasing.
//!
//! Sprites live in a persistent, handle-addressed pool
//! ([`SpriteBatchPass::insert_sprite`] / [`update_sprite`](SpriteBatchPass::update_sprite) /
//! [`remove_sprite`](SpriteBatchPass::remove_sprite)) — not a per-frame push
//! list. This matters for what gets uploaded to the GPU each frame, matching
//! the dirty-range convention the rest of the engine uses for its scene
//! buffers (`helio_core::GrowableBuffer`): instance data (and the parallel
//! alive-flags buffer) is only re-uploaded for the byte range actually
//! touched by `insert_sprite`/`update_sprite`/`remove_sprite` since the last
//! `prepare()` — `O(1)` (no upload at all) if nothing changed.
//!
//! Culling and depth-sorting the pool into a draw order is **not** done
//! here, and not on the CPU at all — pair this pass with
//! `helio-pass-sprite-cull`'s `SpriteCullPass`, added to the graph *before*
//! this one, and wire its outputs in via [`SpriteBatchPass::use_gpu_culling`].
//! That pass culls + radix-sorts the whole pool on the GPU every frame
//! (regardless of pool size — 10, 10 thousand, or 10 million sprites cost
//! the CPU the same: zero), and this pass's `execute()` issues one
//! `draw_indexed_indirect` reading the GPU-computed instance count, never
//! learning the visible count on the CPU at all. See that crate's module doc
//! comment for the full design (and why it's a separate crate: no Cargo
//! dependency either way, just `Arc<wgpu::Buffer>` handles passed between
//! them, matching how `helio-pass-shadow-cull`/`helio-pass-shadow` are wired).
//!
//! The pass renders via vertex-pulling (a `var<storage, read> instances` array
//! indexed through a separate `draw_order` array), not
//! `VertexStepMode::Instance` — see the shader's module doc comment for why.
//!
//! This pass has no dependency on `helio` / `helio-default-graphs` — it only
//! needs `helio-core` (for the [`RenderPass`] trait and graph plumbing) and
//! `libhelio` (for [`FrameResources`](libhelio::FrameResources)), matching
//! every other pass crate. It never touches `PassContext::scene`, so it is
//! usable inside a `RenderGraph` that carries no 3D scene data at all.

use bytemuck::{Pod, Zeroable};
use helio_core::{PassContext, PrepareContext, RenderPass, Result};
use std::sync::Arc;
use wgpu::util::DeviceExt;

/// A stable reference to a sprite in a [`SpriteBatchPass`]'s pool. Returned by
/// [`SpriteBatchPass::insert_sprite`]; opaque, not orderable across passes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SpriteHandle(u32);

/// A single sprite's transform, atlas region, and tint.
///
/// Field order/padding here is not incidental: it must byte-match the
/// `SpriteInstance` struct in `shaders/sprite.wgsl` exactly, including WGSL's
/// storage-buffer alignment rules (`vec4<f32>` requires 16-byte alignment,
/// which Rust's `[f32; 4]` does *not* impose on its own — the `_pad_*` fields
/// exist purely to reproduce that padding on the CPU side).
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
    /// regardless of what a depth buffer says), so `prepare()` radix-sorts
    /// the draw-order index list by this field instead of enabling
    /// `DepthStencilState`. A common convention is "world Y" for a
    /// top-down/2.5D look (larger Y = further back = drawn first), but any
    /// scalar works.
    pub depth: f32,
    _pad_uv: [f32; 2],
    /// Atlas UV rectangle: `[u0, v0, u1, v1]`, each in `0..1`, within
    /// `atlas_layer`.
    pub uv_rect: [f32; 4],
    /// Straight-alpha RGBA tint, multiplied against the sampled atlas texel.
    pub color: [f32; 4],
    /// Index into the atlas texture array — see
    /// [`SpriteBatchPass::add_atlas_layer`].
    pub atlas_layer: u32,
    _pad_tail: [u32; 3],
}

impl SpriteInstance {
    pub fn new(position: [f32; 2], size: [f32; 2]) -> Self {
        Self {
            position,
            size,
            rotation: 0.0,
            depth: 0.0,
            _pad_uv: [0.0; 2],
            uv_rect: [0.0, 0.0, 1.0, 1.0],
            color: [1.0, 1.0, 1.0, 1.0],
            atlas_layer: 0,
            _pad_tail: [0; 3],
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

/// Monotonic ascending-order-preserving `f32 -> u32` transform, standard for
/// radix-sorting floats: flip the sign bit for positives, flip every bit for
/// negatives. NaN depths are not meaningfully orderable and are not handled
/// specially — don't feed them in.
fn depth_to_radix_key(depth: f32) -> u32 {
    let bits = depth.to_bits();
    if bits & 0x8000_0000 != 0 {
        !bits
    } else {
        bits | 0x8000_0000
    }
}

/// GPU-instanced 2D sprite batch pass.
///
/// Owns its own orthographic camera (recomputed from the render target size
/// every frame — call [`SpriteBatchPass::set_camera`] to override
/// framing/zoom/pan) and starts with a 1×1 white fallback atlas layer, so it
/// renders solid-colored quads out of the box; call
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

    quad_vertex_buf: wgpu::Buffer,
    quad_index_buf: wgpu::Buffer,

    // ── Persistent, handle-addressed instance pool (delta-uploaded) ───────
    slots: Vec<SpriteInstance>,
    slot_alive: Vec<bool>,
    free_list: Vec<u32>,
    /// `[start, end)` slot range touched since the last upload, or `None` if
    /// clean. `prepare()` uploads exactly this byte range and nothing else.
    dirty_range: Option<(usize, usize)>,
    instances_buf: wgpu::Buffer,
    instances_capacity: usize,

    // ── Per-frame draw order (culled + radix-sorted slot indices) ─────────
    /// Set whenever visibility-relevant state changes (sprite inserted/
    /// updated/removed, camera moved, render target resized). Cleared once
    /// `prepare()` has rebuilt and uploaded `draw_order_buf`.
    index_dirty: bool,
    draw_order_buf: wgpu::Buffer,
    draw_order_capacity: usize,
    visible_count: u32,
    // Radix-sort scratch, reused every rebuild — no per-frame allocation.
    radix_keys: Vec<u32>,
    radix_a: Vec<u32>,
    radix_b: Vec<u32>,

    camera_buf: wgpu::Buffer,
    camera_dirty: bool,
    last_width: u32,
    last_height: u32,
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
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
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

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Sprite Batch Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[Some(vertex_layout)],
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
        let instances_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Sprite Instance Storage"),
            size: INSTANCE_STRIDE * initial_capacity as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let draw_order_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Sprite Draw Order"),
            size: 4 * initial_capacity as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
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
            quad_vertex_buf,
            quad_index_buf,
            slots: Vec::new(),
            slot_alive: Vec::new(),
            free_list: Vec::new(),
            dirty_range: None,
            instances_buf,
            instances_capacity: initial_capacity,
            index_dirty: true,
            draw_order_buf,
            draw_order_capacity: initial_capacity,
            visible_count: 0,
            radix_keys: Vec::new(),
            radix_a: Vec::new(),
            radix_b: Vec::new(),
            camera_buf,
            camera_dirty: true,
            last_width: 0,
            last_height: 0,
            camera_half_extent: None,
            camera_center: [0.0, 0.0],
            clear_color: Some(wgpu::Color::BLACK),
        }
    }

    fn mark_data_dirty(&mut self, slot: usize) {
        self.dirty_range = Some(match self.dirty_range {
            Some((s, e)) => (s.min(slot), e.max(slot + 1)),
            None => (slot, slot + 1),
        });
    }

    /// Adds a new sprite to the pool. `O(1)` amortized (reuses a freed slot
    /// if one exists). The returned handle stays valid until
    /// [`remove_sprite`](Self::remove_sprite).
    pub fn insert_sprite(&mut self, instance: SpriteInstance) -> SpriteHandle {
        let slot = if let Some(free) = self.free_list.pop() {
            self.slots[free as usize] = instance;
            self.slot_alive[free as usize] = true;
            free
        } else {
            self.slots.push(instance);
            self.slot_alive.push(true);
            (self.slots.len() - 1) as u32
        };
        self.mark_data_dirty(slot as usize);
        self.index_dirty = true;
        SpriteHandle(slot)
    }

    /// Overwrites a sprite's data in place. `O(1)`; marks only this slot's
    /// byte range dirty for the next `prepare()`.
    pub fn update_sprite(&mut self, handle: SpriteHandle, instance: SpriteInstance) {
        let slot = handle.0 as usize;
        self.slots[slot] = instance;
        self.mark_data_dirty(slot);
        self.index_dirty = true;
    }

    /// Frees a sprite's slot for reuse. Its stale GPU bytes are never read
    /// again (excluded from `draw_order` on the next rebuild), so this does
    /// not need to touch the instance data buffer — only the draw order.
    pub fn remove_sprite(&mut self, handle: SpriteHandle) {
        let slot = handle.0 as usize;
        if std::mem::replace(&mut self.slot_alive[slot], false) {
            self.free_list.push(handle.0);
            self.index_dirty = true;
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
                wgpu::BindGroupEntry { binding: 3, resource: self.instances_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: self.draw_order_buf.as_entire_binding() },
            ],
        }));
    }

    /// Overrides the orthographic view: `center` and `half_extent` in world
    /// units. `half_extent = None` re-derives it from the render target's
    /// pixel dimensions each frame (1 world unit = 1 pixel, origin centered).
    pub fn set_camera(&mut self, center: [f32; 2], half_extent: Option<[f32; 2]>) {
        self.camera_center = center;
        self.camera_half_extent = half_extent;
        self.camera_dirty = true;
        self.index_dirty = true;
    }

    /// `None` disables the clear (loads the existing target contents);
    /// `Some(color)` clears to that color every frame. Defaults to opaque black.
    pub fn set_clear_color(&mut self, color: Option<wgpu::Color>) {
        self.clear_color = color;
    }

    /// Sprites currently alive in the pool (inserted, not yet removed).
    pub fn sprite_count(&self) -> usize {
        self.slots.len() - self.free_list.len()
    }

    /// Sprites that survived culling and were actually drawn last `prepare()`.
    pub fn visible_count(&self) -> usize {
        self.visible_count as usize
    }

}

/// 4-pass LSD radix sort (stable, `O(n)`): sorts `a[..count]` ascending by
/// `keys[a[i]]`, using `b[..count]` as ping-pong scratch. `a`/`b` are meant
/// to be reused across calls (e.g. `SpriteBatchPass::radix_a`/`radix_b`) —
/// no allocation here once both are warmed up to `count` capacity.
fn radix_sort_indices(a: &mut Vec<u32>, b: &mut Vec<u32>, keys: &[u32], count: usize) {
    if count <= 1 {
        return;
    }
    b.clear();
    b.resize(count, 0);

    let mut src_is_a = true;
    for pass in 0..4u32 {
        let shift = pass * 8;
        let mut counts = [0u32; 257];
        if src_is_a {
            for &idx in &a[..count] {
                let bucket = ((keys[idx as usize] >> shift) & 0xFF) as usize;
                counts[bucket + 1] += 1;
            }
        } else {
            for &idx in &b[..count] {
                let bucket = ((keys[idx as usize] >> shift) & 0xFF) as usize;
                counts[bucket + 1] += 1;
            }
        }
        for i in 0..256 {
            counts[i + 1] += counts[i];
        }
        if src_is_a {
            let (src, dst) = (&a[..count], &mut b[..count]);
            for &idx in src {
                let bucket = ((keys[idx as usize] >> shift) & 0xFF) as usize;
                dst[counts[bucket] as usize] = idx;
                counts[bucket] += 1;
            }
        } else {
            let (src, dst) = (&b[..count], &mut a[..count]);
            for &idx in src {
                let bucket = ((keys[idx as usize] >> shift) & 0xFF) as usize;
                dst[counts[bucket] as usize] = idx;
                counts[bucket] += 1;
            }
        }
        src_is_a = !src_is_a;
    }
    // 4 (even) passes: the final write always lands back in `a` — each pass
    // writes into whichever buffer is *not* `src`, and `src` alternates
    // a,b,a,b, so the 4th (last) write targets `a`.
    debug_assert!(src_is_a);
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
        if ctx.width != self.last_width || ctx.height != self.last_height {
            self.last_width = ctx.width;
            self.last_height = ctx.height;
            self.camera_dirty = true;
            // A `None` half_extent derives the view rect from the render
            // target size, so a resize can change what's visible.
            self.index_dirty = true;
        }

        if self.camera_dirty {
            self.camera_dirty = false;
            let half_extent = self.camera_half_extent.unwrap_or([ctx.width as f32 * 0.5, ctx.height as f32 * 0.5]);
            let [cx, cy] = self.camera_center;
            let [hx, hy] = half_extent;
            // Y-up world space; a Y-down convention would swap `cy - hy`/`cy + hy`.
            let view_proj = glam::Mat4::orthographic_rh(cx - hx, cx + hx, cy - hy, cy + hy, -1.0, 1.0);
            let uniform = CameraUniform { view_proj: view_proj.to_cols_array_2d() };
            ctx.write_buffer(&self.camera_buf, 0, bytemuck::bytes_of(&uniform));
        }

        // ── Delta-upload instance data ─────────────────────────────────────
        if self.slots.len() > self.instances_capacity {
            self.instances_capacity = (self.slots.len() * 2).max(self.instances_capacity * 2);
            self.instances_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Sprite Instance Storage"),
                size: INSTANCE_STRIDE * self.instances_capacity as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            self.bind_group = None; // buffer identity changed
            self.dirty_range = (!self.slots.is_empty()).then_some((0, self.slots.len()));
        }
        if let Some((start, end)) = self.dirty_range.take() {
            ctx.write_buffer(
                &self.instances_buf,
                start as u64 * INSTANCE_STRIDE,
                bytemuck::cast_slice(&self.slots[start..end]),
            );
        }

        // ── Rebuild + upload draw order, only if something visibility-
        //    relevant changed since the last frame ─────────────────────────
        if self.index_dirty {
            self.index_dirty = false;

            let half_extent = self.camera_half_extent.unwrap_or([ctx.width as f32 * 0.5, ctx.height as f32 * 0.5]);
            let [cx, cy] = self.camera_center;
            let [hx, hy] = half_extent;
            let view_min = [cx - hx, cy - hy];
            let view_max = [cx + hx, cy + hy];

            // Circle-vs-AABB: clamp the sprite's center into the view rect,
            // then compare the distance to that clamped point against the
            // sprite's bounding radius. Conservative (uses the quad's full
            // diagonal, so a rotated sprite is never culled early).
            self.radix_keys.clear();
            self.radix_keys.resize(self.slots.len(), 0);
            self.radix_a.clear();
            for (i, inst) in self.slots.iter().enumerate() {
                if !self.slot_alive[i] {
                    continue;
                }
                let clamped = [
                    inst.position[0].clamp(view_min[0], view_max[0]),
                    inst.position[1].clamp(view_min[1], view_max[1]),
                ];
                let dx = inst.position[0] - clamped[0];
                let dy = inst.position[1] - clamped[1];
                if (dx * dx + dy * dy).sqrt() > inst.cull_radius() {
                    continue;
                }
                self.radix_keys[i] = depth_to_radix_key(inst.depth);
                self.radix_a.push(i as u32);
            }

            let count = self.radix_a.len();
            radix_sort_indices(&mut self.radix_a, &mut self.radix_b, &self.radix_keys, count);
            self.visible_count = count as u32;

            if count > self.draw_order_capacity {
                self.draw_order_capacity = (count * 2).max(self.draw_order_capacity * 2).max(1);
                self.draw_order_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("Sprite Draw Order"),
                    size: 4 * self.draw_order_capacity as u64,
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
                self.bind_group = None;
            }
            if count > 0 {
                ctx.write_buffer(&self.draw_order_buf, 0, bytemuck::cast_slice(&self.radix_a[..count]));
            }
        }

        Ok(())
    }

    fn execute(&mut self, ctx: &mut PassContext) -> Result<()> {
        if self.visible_count == 0 {
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
                    wgpu::BindGroupEntry { binding: 3, resource: self.instances_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 4, resource: self.draw_order_buf.as_entire_binding() },
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
        rp.set_index_buffer(self.quad_index_buf.slice(..), wgpu::IndexFormat::Uint16);
        rp.draw_indexed(0..6, 0, 0..self.visible_count);

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn radix_key_preserves_float_order() {
        let mut values = [-100.0f32, -1.0, -0.001, 0.0, 0.001, 1.0, 100.0, f32::MIN, f32::MAX];
        let sorted_by_key = {
            let mut v = values;
            v.sort_by_key(|f| depth_to_radix_key(*f));
            v
        };
        values.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert_eq!(values, sorted_by_key);
    }

    #[test]
    fn radix_sort_matches_stable_comparison_sort() {
        let depths = [3.0f32, -1.0, 0.0, 2.5, -2.5, 0.0, 1.0, -1.0, 5.0, -5.0];
        let keys: Vec<u32> = depths.iter().map(|d| depth_to_radix_key(*d)).collect();
        let mut a: Vec<u32> = (0..depths.len() as u32).collect();
        let mut b: Vec<u32> = Vec::new();
        radix_sort_indices(&mut a, &mut b, &keys, depths.len());

        let expected: Vec<u32> = {
            let mut idx: Vec<u32> = (0..depths.len() as u32).collect();
            // `sort_by` is stable, matching what a correct radix sort must
            // also do for the tied `0.0` / `-1.0` entries above.
            idx.sort_by(|&x, &y| depths[x as usize].partial_cmp(&depths[y as usize]).unwrap());
            idx
        };
        assert_eq!(a, expected);
    }

    #[test]
    fn radix_sort_handles_empty_and_singleton() {
        let keys: Vec<u32> = vec![];
        let mut a: Vec<u32> = vec![];
        let mut b: Vec<u32> = vec![];
        radix_sort_indices(&mut a, &mut b, &keys, 0); // must not panic

        let keys = vec![depth_to_radix_key(1.0)];
        let mut a = vec![0u32];
        let mut b = vec![];
        radix_sort_indices(&mut a, &mut b, &keys, 1);
        assert_eq!(a, vec![0]);
    }
}
