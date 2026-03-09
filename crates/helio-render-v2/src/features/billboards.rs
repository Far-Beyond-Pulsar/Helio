//! Billboard rendering feature

use super::{FeatureContext, PrepareContext};
use crate::features::{Feature, ShaderDefine};
use crate::passes::BillboardPass;
use crate::gpu_transfer;
use crate::scene::BillboardId;
use crate::Result;
use std::collections::HashMap;
use std::sync::Arc;

/// A single billboard to render
#[derive(Clone, Debug)]
pub struct BillboardInstance {
    /// World-space position
    pub position: [f32; 3],
    /// Width and height in world units (or angular units when `screen_scale` is true)
    pub scale: [f32; 2],
    /// RGBA tint (multiplied with billboard texture / color)
    pub color: [f32; 4],
    /// When true the size is kept constant in screen-space
    pub screen_scale: bool,
}

impl BillboardInstance {
    pub fn new(position: [f32; 3], scale: [f32; 2]) -> Self {
        Self {
            position,
            scale,
            color: [1.0, 1.0, 1.0, 1.0],
            screen_scale: false,
        }
    }

    pub fn with_color(mut self, color: [f32; 4]) -> Self {
        self.color = color;
        self
    }

    pub fn with_screen_scale(mut self, enabled: bool) -> Self {
        self.screen_scale = enabled;
        self
    }
}

/// GPU instance data (must match WGSL struct)
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct GpuBillboardInstance {
    pub position: [f32; 3],
    pub _pad1: f32,
    pub scale: [f32; 2],
    pub screen_scale: u32,
    pub _pad2: u32,
    pub color: [f32; 4],
}

/// Billboard quad vertex (local-space, unit size)
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct BillboardVertex {
    pub position: [f32; 2],
    pub uv: [f32; 2],
}

/// Internal proxy for a registered billboard slot.
struct BillboardProxy {
    /// Index into the dense instance arrays.
    index: u32,
    data: BillboardInstance,
}

/// Billboard rendering feature
pub struct BillboardsFeature {
    enabled: bool,
    // ── Persistent registry ────────────────────────────────────────────────
    proxies: HashMap<u32, BillboardProxy>,
    next_billboard_id: u32,
    /// Dense CPU-side list; matches layout of gpu_staging / instance_buffer.
    instances: Vec<BillboardInstance>,
    /// True when any instance was added / removed / updated since last upload.
    instances_dirty: bool,
    max_instances: u32,
    // ── Sprite / texture state ─────────────────────────────────────────────
    /// Optional sprite image: raw RGBA8 bytes + dimensions
    sprite_rgba: Option<(Vec<u8>, u32, u32)>,
    /// Live sprite texture (stored so we can re-upload data at runtime)
    sprite_texture: Option<Arc<wgpu::Texture>>,
    /// Pixel dimensions of the sprite texture (set after register())
    sprite_dims: (u32, u32),
    /// New RGBA8 pixels waiting to be written to the GPU texture (set by set_sprite)
    pending_sprite: Option<Vec<u8>>,
    // ── Shared GPU resources ───────────────────────────────────────────────
    vertex_buffer: Option<Arc<wgpu::Buffer>>,
    index_buffer: Option<Arc<wgpu::Buffer>>,
    instance_buffer: Option<Arc<wgpu::Buffer>>,
    instance_count: Arc<std::sync::atomic::AtomicU32>,
    /// Reused staging buffer (never reallocated at steady state).
    gpu_staging: Vec<GpuBillboardInstance>,
}

impl BillboardsFeature {
    pub fn new() -> Self {
        Self {
            enabled: true,
            proxies: HashMap::new(),
            next_billboard_id: 1,
            instances: Vec::new(),
            instances_dirty: false,
            max_instances: 1024,
            sprite_rgba: None,
            sprite_texture: None,
            sprite_dims: (0, 0),
            pending_sprite: None,
            vertex_buffer: None,
            index_buffer: None,
            instance_buffer: None,
            instance_count: Arc::new(std::sync::atomic::AtomicU32::new(0)),
            gpu_staging: Vec::new(),
        }
    }

    /// Set a sprite texture from raw RGBA8 pixel data
    pub fn with_sprite(mut self, rgba: Vec<u8>, width: u32, height: u32) -> Self {
        self.sprite_rgba = Some((rgba, width, height));
        self
    }

    /// Set the maximum number of billboard instances (must be called before renderer init)
    pub fn with_max_instances(mut self, n: u32) -> Self {
        self.max_instances = n;
        self
    }

    /// Returns the pixel dimensions of the registered sprite texture, or `None` before init.
    pub fn sprite_dims(&self) -> Option<(u32, u32)> {
        if self.sprite_dims.0 > 0 { Some(self.sprite_dims) } else { None }
    }

    /// Queue a new sprite image to be uploaded to the GPU on the next frame.
    /// `width` and `height` must match the existing sprite texture dimensions.
    pub fn set_sprite(&mut self, rgba: Vec<u8>, width: u32, height: u32) {
        let (sw, sh) = self.sprite_dims;
        if sw == width && sh == height {
            self.pending_sprite = Some(rgba);
        } else {
            log::warn!(
                "BillboardsFeature::set_sprite: dims {}×{} don't match existing sprite {}×{} — ignored",
                width, height, sw, sh
            );
        }
    }

    // ── Persistent API (preferred) ─────────────────────────────────────────

    /// Register a billboard and return a stable [`BillboardId`].  O(1).
    pub fn add_billboard_persistent(&mut self, instance: BillboardInstance) -> BillboardId {
        let id = BillboardId(self.next_billboard_id);
        self.next_billboard_id += 1;

        let index = self.instances.len() as u32;
        self.instances.push(instance.clone());
        self.proxies.insert(id.0, BillboardProxy { index, data: instance });
        self.instances_dirty = true;
        id
    }

    /// Remove a billboard by id.  Swap-removes to keep the array dense.  O(1).
    pub fn remove_billboard_persistent(&mut self, id: BillboardId) {
        let proxy = match self.proxies.remove(&id.0) {
            Some(p) => p,
            None => return,
        };
        let slot = proxy.index as usize;
        let last = self.instances.len() - 1;
        if slot != last {
            self.instances.swap(slot, last);
            // Update the swapped proxy's index.
            for p in self.proxies.values_mut() {
                if p.index as usize == last {
                    p.index = slot as u32;
                    p.data = self.instances[slot].clone();
                    break;
                }
            }
        }
        self.instances.pop();
        self.instances_dirty = true;
    }

    /// Update an existing billboard.  O(1).
    pub fn update_billboard_persistent(&mut self, id: BillboardId, instance: BillboardInstance) {
        let proxy = match self.proxies.get_mut(&id.0) {
            Some(p) => p,
            None => return,
        };
        let slot = proxy.index as usize;
        proxy.data = instance.clone();
        self.instances[slot] = instance;
        self.instances_dirty = true;
    }

    /// Update only the world-space position of a billboard.  O(1).
    pub fn move_billboard(&mut self, id: BillboardId, position: [f32; 3]) {
        let proxy = match self.proxies.get_mut(&id.0) {
            Some(p) => p,
            None => return,
        };
        let slot = proxy.index as usize;
        proxy.data.position = position;
        self.instances[slot].position = position;
        self.instances_dirty = true;
    }

    // ── Legacy snapshot API ────────────────────────────────────────────────

    /// Replace the whole billboard list from a slice.
    /// Prefer the persistent `add_billboard_persistent` / `remove_billboard_persistent`
    /// API to avoid per-frame allocations.
    pub fn set_billboards_slice(&mut self, instances: &[BillboardInstance]) {
        // Only mark dirty if the content actually changed.
        if self.instances.len() != instances.len()
            || self.instances.iter().zip(instances.iter()).any(|(a, b)| {
                a.position != b.position
                    || a.scale != b.scale
                    || a.color != b.color
                    || a.screen_scale != b.screen_scale
            })
        {
            self.instances.clear();
            self.instances.extend_from_slice(instances);
            // Clear the proxy map since the dense array was fully replaced.
            self.proxies.clear();
            self.instances_dirty = true;
        }
    }

    pub fn set_billboards(&mut self, instances: Vec<BillboardInstance>) {
        self.set_billboards_slice(&instances);
    }

    pub fn add_billboard(&mut self, instance: BillboardInstance) {
        self.instances.push(instance);
        self.instances_dirty = true;
    }

    pub fn clear_billboards(&mut self) {
        if !self.instances.is_empty() {
            self.instances.clear();
            self.proxies.clear();
            self.instances_dirty = true;
        }
    }

    pub fn billboards(&self) -> &[BillboardInstance] {
        &self.instances
    }
}

impl Default for BillboardsFeature {
    fn default() -> Self {
        Self::new()
    }
}

impl Feature for BillboardsFeature {
    fn name(&self) -> &str {
        "billboards"
    }

    fn register(&mut self, ctx: &mut FeatureContext) -> Result<()> {
        use wgpu::util::DeviceExt;

        // Unit quad in local space (-0.5 .. +0.5), Y-up
        let vertices = [
            BillboardVertex { position: [-0.5, -0.5], uv: [0.0, 1.0] },
            BillboardVertex { position: [ 0.5, -0.5], uv: [1.0, 1.0] },
            BillboardVertex { position: [ 0.5,  0.5], uv: [1.0, 0.0] },
            BillboardVertex { position: [-0.5,  0.5], uv: [0.0, 0.0] },
        ];
        let indices: [u32; 6] = [0, 1, 2, 0, 2, 3];

        let vertex_buffer = Arc::new(ctx.device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("Billboard Quad Vertices"),
                contents: bytemuck::cast_slice(&vertices),
                usage: wgpu::BufferUsages::VERTEX,
            },
        ));

        let index_buffer = Arc::new(ctx.device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("Billboard Quad Indices"),
                contents: bytemuck::cast_slice(&indices),
                usage: wgpu::BufferUsages::INDEX,
            },
        ));

        let instance_buffer = Arc::new(ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Billboard Instance Buffer"),
            size: std::mem::size_of::<GpuBillboardInstance>() as u64 * self.max_instances as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));

        // ── Sprite bind group layout (group 1) ─────────────────────────────────
        let sprite_layout = Arc::new(ctx.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Billboard Sprite Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        }));

        // Build the sprite texture (or a 1×1 white fallback)
        let (sprite_tex_arc, tex_w, tex_h) = if let Some((ref rgba, w, h)) = self.sprite_rgba {
            let tex = ctx.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("Billboard Sprite"),
                size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
                mip_level_count: 1, sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8UnormSrgb,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            });
            ctx.queue.write_texture(
                wgpu::TexelCopyTextureInfo { texture: &tex, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
                rgba,
                wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(w * 4), rows_per_image: Some(h) },
                wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
            );
            (Arc::new(tex), w, h)
        } else {
            // 1×1 white fallback
            let tex = ctx.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("Billboard Sprite (white fallback)"),
                size: wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
                mip_level_count: 1, sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8UnormSrgb,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            });
            ctx.queue.write_texture(
                wgpu::TexelCopyTextureInfo { texture: &tex, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
                &[255u8, 255, 255, 255],
                wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(4), rows_per_image: Some(1) },
                wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
            );
            (Arc::new(tex), 1u32, 1u32)
        };
        self.sprite_dims = (tex_w, tex_h);

        let sprite_view = sprite_tex_arc.create_view(&wgpu::TextureViewDescriptor::default());
        let sprite_sampler = ctx.device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Billboard Sprite Sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        let sprite_bg = Arc::new(ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Billboard Sprite Bind Group"),
            layout: &sprite_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&sprite_view) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&sprite_sampler) },
            ],
        }));

        ctx.graph.add_pass(BillboardPass::new(
            vertex_buffer.clone(),
            index_buffer.clone(),
            instance_buffer.clone(),
            self.instance_count.clone(),
            ctx.surface_format,
            sprite_bg,
            sprite_layout,
        ));

        self.vertex_buffer = Some(vertex_buffer);
        self.index_buffer = Some(index_buffer);
        self.instance_buffer = Some(instance_buffer);
        self.sprite_texture = Some(sprite_tex_arc);

        // Mark dirty so the first frame uploads whatever was pre-loaded.
        self.instances_dirty = true;

        log::info!("Billboards feature registered: max {} instances", self.max_instances);
        Ok(())
    }

    fn prepare(&mut self, ctx: &PrepareContext) -> Result<()> {
        // Re-upload sprite pixels if set_sprite() was called since last frame.
        if let Some(pending) = self.pending_sprite.take() {
            if let Some(tex) = &self.sprite_texture {
                let (sw, sh) = self.sprite_dims;
                ctx.queue.write_texture(
                    wgpu::TexelCopyTextureInfo { texture: tex, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
                    &pending,
                    wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(sw * 4), rows_per_image: Some(sh) },
                    wgpu::Extent3d { width: sw, height: sh, depth_or_array_layers: 1 },
                );
            }
        }

        let count = self.instances.len().min(self.max_instances as usize);

        // Only re-encode + upload when the instance list changed.
        if self.instances_dirty {
            if let Some(buf) = &self.instance_buffer {
                if count > 0 {
                    // Reuse staging buffer — no alloc at steady state.
                    self.gpu_staging.clear();
                    self.gpu_staging.reserve(count.saturating_sub(self.gpu_staging.capacity()));
                    self.gpu_staging.extend(self.instances[..count].iter().map(|b| GpuBillboardInstance {
                        position: b.position,
                        _pad1: 0.0,
                        scale: b.scale,
                        screen_scale: b.screen_scale as u32,
                        _pad2: 0,
                        color: b.color,
                    }));
                    ctx.queue.write_buffer(buf, 0, bytemuck::cast_slice(&self.gpu_staging));
                    gpu_transfer::track_upload(
                        (self.gpu_staging.len() * std::mem::size_of::<GpuBillboardInstance>()) as u64,
                    );
                }
            }
            self.instance_count
                .store(count as u32, std::sync::atomic::Ordering::Relaxed);
            self.instances_dirty = false;
        }

        Ok(())
    }

    fn shader_defines(&self) -> HashMap<String, ShaderDefine> {
        HashMap::new()
    }

    fn is_enabled(&self) -> bool {
        self.enabled
    }

    fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }
}
