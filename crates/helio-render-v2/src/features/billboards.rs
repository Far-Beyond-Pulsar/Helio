//! Billboard rendering feature

use super::{FeatureContext, PrepareContext};
use crate::features::{Feature, ShaderDefine};
use crate::passes::BillboardPass;
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

/// Billboard rendering feature
pub struct BillboardsFeature {
    enabled: bool,
    instances: Vec<BillboardInstance>,
    max_instances: u32,
    /// Optional sprite image: raw RGBA8 bytes + dimensions
    sprite_rgba: Option<(Vec<u8>, u32, u32)>,
    // Shared with BillboardPass via Arc
    vertex_buffer: Option<Arc<wgpu::Buffer>>,
    index_buffer: Option<Arc<wgpu::Buffer>>,
    instance_buffer: Option<Arc<wgpu::Buffer>>,
    instance_count: Arc<std::sync::atomic::AtomicU32>,
}

impl BillboardsFeature {
    pub fn new() -> Self {
        Self {
            enabled: true,
            instances: Vec::new(),
            max_instances: 1024,
            sprite_rgba: None,
            vertex_buffer: None,
            index_buffer: None,
            instance_buffer: None,
            instance_count: Arc::new(std::sync::atomic::AtomicU32::new(0)),
        }
    }

    /// Set a sprite texture from raw RGBA8 pixel data
    pub fn with_sprite(mut self, rgba: Vec<u8>, width: u32, height: u32) -> Self {
        self.sprite_rgba = Some((rgba, width, height));
        self
    }

    pub fn add_billboard(&mut self, instance: BillboardInstance) {
        self.instances.push(instance);
    }

    pub fn set_billboards(&mut self, instances: Vec<BillboardInstance>) {
        self.instances = instances;
    }

    pub fn clear_billboards(&mut self) {
        self.instances.clear();
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
        let (sprite_tex, tex_w, tex_h) = if let Some((ref rgba, w, h)) = self.sprite_rgba {
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
                wgpu::ImageCopyTexture { texture: &tex, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
                rgba,
                wgpu::ImageDataLayout { offset: 0, bytes_per_row: Some(w * 4), rows_per_image: Some(h) },
                wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
            );
            (tex, w, h)
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
                wgpu::ImageCopyTexture { texture: &tex, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
                &[255u8, 255, 255, 255],
                wgpu::ImageDataLayout { offset: 0, bytes_per_row: Some(4), rows_per_image: Some(1) },
                wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
            );
            (tex, 1u32, 1u32)
        };
        let _ = (tex_w, tex_h);

        let sprite_view = sprite_tex.create_view(&wgpu::TextureViewDescriptor::default());
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

        log::info!("Billboards feature registered: max {} instances", self.max_instances);
        Ok(())
    }

    fn prepare(&mut self, ctx: &PrepareContext) -> Result<()> {
        let count = self.instances.len().min(self.max_instances as usize);

        if let Some(buf) = &self.instance_buffer {
            if count > 0 {
                let gpu: Vec<GpuBillboardInstance> = self.instances[..count]
                    .iter()
                    .map(|b| GpuBillboardInstance {
                        position: b.position,
                        _pad1: 0.0,
                        scale: b.scale,
                        screen_scale: b.screen_scale as u32,
                        _pad2: 0,
                        color: b.color,
                    })
                    .collect();
                ctx.queue.write_buffer(buf, 0, bytemuck::cast_slice(&gpu));
            }
        }

        self.instance_count
            .store(count as u32, std::sync::atomic::Ordering::Relaxed);
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
