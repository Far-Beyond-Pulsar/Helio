//! Dynamic-mode voxel ray march pass.
//!
//! Fullscreen compute shader that DDA marches through voxel volumes
//! reading from the shared brick pool. Outputs shaded color via a fullscreen
//! triangle pass into `pre_aa` for consumption by TAA.

use bytemuck::{Pod, Zeroable};
use helio_core::{
    graph::{ResourceBuilder, ResourceFormat, ResourceSize},
    PassContext, PrepareContext, RenderPass, Result as HelioResult,
};

// ── GPU uniforms ──────────────────────────────────────────────────────────────

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct RayMarchParams {
    width: f32,
    height: f32,
    time: f32,
    volume_count: u32,
    light_count: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum AttachmentMode {
    Standalone,
    Composited,
}

impl AttachmentMode {
    fn color_load(self) -> wgpu::LoadOp<wgpu::Color> {
        match self {
            Self::Standalone => wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
            Self::Composited => wgpu::LoadOp::Load,
        }
    }
}

fn shade_bind_group_matches_size(bound_size: Option<(u32, u32)>, width: u32, height: u32) -> bool {
    bound_size == Some((width, height))
}

// ── Pass ──────────────────────────────────────────────────────────────────────

pub struct VoxelRayMarchPass {
    // Compute pipeline
    ray_march_pipeline: wgpu::ComputePipeline,
    compute_bgl: wgpu::BindGroupLayout,
    compute_bg: Option<wgpu::BindGroup>,
    compute_bg_key: Option<usize>,

    // Shade pipeline (fullscreen tri)
    shade_pipeline: wgpu::RenderPipeline,
    shade_bgl: wgpu::BindGroupLayout,
    shade_bg: Option<wgpu::BindGroup>,
    shade_bg_size: Option<(u32, u32)>,

    // Output textures
    color_tex: wgpu::Texture,
    color_view: wgpu::TextureView,
    normal_tex: wgpu::Texture,
    normal_view: wgpu::TextureView,

    // Params
    params_buf: wgpu::Buffer,
    width: u32,
    height: u32,
    surface_format: wgpu::TextureFormat,

    last_volume_count: u32,
    params_frame: u64,
    attachment_mode: AttachmentMode,
}

impl VoxelRayMarchPass {
    /// Creates a standalone pass that clears pixels missed by the raymarch.
    pub fn new(device: &wgpu::Device, surface_format: wgpu::TextureFormat) -> Self {
        Self::new_with_attachment_mode(device, surface_format, AttachmentMode::Standalone)
    }

    /// Creates a pass that preserves existing color where no voxel is hit.
    pub fn new_composited(device: &wgpu::Device, surface_format: wgpu::TextureFormat) -> Self {
        Self::new_with_attachment_mode(device, surface_format, AttachmentMode::Composited)
    }

    fn new_with_attachment_mode(
        device: &wgpu::Device,
        surface_format: wgpu::TextureFormat,
        attachment_mode: AttachmentMode,
    ) -> Self {
        let (color_tex, color_view) = Self::create_tex(device, 1, 1, "VoxelRayMarch Color");
        let (normal_tex, normal_view) = Self::create_tex(device, 1, 1, "VoxelRayMarch Normal");

        let params_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("VoxelRayMarch Params"),
            size: std::mem::size_of::<RayMarchParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // ── Compute BGL ─────────────────────────────────────────────────────
        let compute_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("VoxelRayMarch Compute BGL"),
            entries: &[
                wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 1, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 2, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 3, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 4, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 5, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::StorageTexture { access: wgpu::StorageTextureAccess::WriteOnly, format: wgpu::TextureFormat::Rgba8Unorm, view_dimension: wgpu::TextureViewDimension::D2 }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 6, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::StorageTexture { access: wgpu::StorageTextureAccess::WriteOnly, format: wgpu::TextureFormat::Rgba8Unorm, view_dimension: wgpu::TextureViewDimension::D2 }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 7, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            ],
        });

        let compute_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("VoxelRayMarch Compute"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/voxel_raymarch.wgsl").into()),
        });

        let compute_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("VoxelRayMarch Compute PL"),
            bind_group_layouts: &[Some(&compute_bgl)],
            immediate_size: 0,
        });

        let ray_march_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("VoxelRayMarch"),
            layout: Some(&compute_pl),
            module: &compute_shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        // ── Shade BGL ───────────────────────────────────────────────────────
        let shade_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("VoxelRayMarch Shade BGL"),
            entries: &[
                wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::FRAGMENT, ty: wgpu::BindingType::Texture { sample_type: wgpu::TextureSampleType::Float { filterable: false }, view_dimension: wgpu::TextureViewDimension::D2, multisampled: false }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 1, visibility: wgpu::ShaderStages::FRAGMENT, ty: wgpu::BindingType::Texture { sample_type: wgpu::TextureSampleType::Float { filterable: false }, view_dimension: wgpu::TextureViewDimension::D2, multisampled: false }, count: None },
            ],
        });

        let shade_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("VoxelRayMarch Shade"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/voxel_raymarch_shade.wgsl").into()),
        });

        let shade_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("VoxelRayMarch Shade PL"),
            bind_group_layouts: &[Some(&shade_bgl)],
            immediate_size: 0,
        });

        let shade_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("VoxelRayMarch Shade"),
            layout: Some(&shade_pl),
            vertex: wgpu::VertexState {
                module: &shade_shader,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shade_shader,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            cache: None,
            multiview_mask: None,
        });

        Self {
            ray_march_pipeline,
            compute_bgl,
            compute_bg: None,
            compute_bg_key: None,
            shade_pipeline,
            shade_bgl,
            shade_bg: None,
            shade_bg_size: None,
            color_tex,
            color_view,
            normal_tex,
            normal_view,
            params_buf,
            width: 1,
            height: 1,
            surface_format,
            last_volume_count: 0,
            params_frame: u64::MAX,
            attachment_mode,
        }
    }

    fn create_tex(
        device: &wgpu::Device,
        w: u32,
        h: u32,
        label: &str,
    ) -> (wgpu::Texture, wgpu::TextureView) {
        let tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some(label),
            size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let view = tex.create_view(&Default::default());
        (tex, view)
    }

    fn rebuild_compute_bg(&mut self, ctx: &PassContext) {
        let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("VoxelRayMarch Compute BG"),
            layout: &self.compute_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: ctx.scene.camera.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: self.params_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: ctx.scene.voxel_volumes.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: ctx.scene.voxel_brick_pool.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: ctx.scene.voxel_data_pool.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: wgpu::BindingResource::TextureView(&self.color_view) },
                wgpu::BindGroupEntry { binding: 6, resource: wgpu::BindingResource::TextureView(&self.normal_view) },
                wgpu::BindGroupEntry { binding: 7, resource: ctx.scene.lights.as_entire_binding() },
            ],
        });
        self.compute_bg = Some(bg);
        self.compute_bg_key = Some(ctx.scene.camera_generation as usize);
    }

    fn rebuild_shade_bg(&mut self, ctx: &PassContext) {
        let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("VoxelRayMarch Shade BG"),
            layout: &self.shade_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&self.color_view) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&self.normal_view) },
            ],
        });
        self.shade_bg = Some(bg);
        self.shade_bg_size = Some((self.width, self.height));
    }

    pub fn set_params(&mut self, _width: u32, _height: u32, _volume_count: u32) {
        // Will be picked up in prepare()
    }
}

impl RenderPass for VoxelRayMarchPass {
    fn name(&self) -> &'static str {
        "VoxelRayMarch"
    }

    fn reads(&self) -> &'static [&'static str] {
        &[]
    }

    fn writes(&self) -> &'static [&'static str] {
        &["pre_aa"]
    }

    fn declare_resources(&self, builder: &mut ResourceBuilder) {
        builder.write_color(
            "pre_aa",
            ResourceFormat::from(self.surface_format),
            ResourceSize::MatchSurface,
        );
    }

    fn prepare(&mut self, ctx: &PrepareContext) -> HelioResult<()> {
        if ctx.scene.voxel_volume_count != self.last_volume_count
            || ctx.frame_num != self.params_frame
        {
            self.last_volume_count = ctx.scene.voxel_volume_count;

            let params = RayMarchParams {
                width: self.width as f32,
                height: self.height as f32,
                time: ctx.frame_num as f32 * 0.016,
                volume_count: ctx.scene.voxel_volume_count,
                light_count: ctx.scene.lights.len() as u32,
                _pad0: 0,
                _pad1: 0,
                _pad2: 0,
            };
            ctx.write_buffer(&self.params_buf, 0, bytemuck::bytes_of(&params));
            self.params_frame = ctx.frame_num;
        }
        Ok(())
    }

    fn execute(&mut self, ctx: &mut PassContext) -> HelioResult<()> {
        // Skip when no voxel volumes are present (composited into default graph).
        if ctx.scene.voxel_volume_count == 0 {
            return Ok(());
        }

        let gen = ctx.scene.camera_generation as usize;
        if self.compute_bg_key != Some(gen) || self.compute_bg.is_none() {
            self.rebuild_compute_bg(ctx);
        }
        if !shade_bind_group_matches_size(self.shade_bg_size, self.width, self.height)
            || self.shade_bg.is_none()
        {
            self.rebuild_shade_bg(ctx);
        }

        // Step 1: Compute — DDA ray march
        {
            let mut cpass = ctx.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("VoxelRayMarch"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.ray_march_pipeline);
            if let Some(ref bg) = self.compute_bg {
                cpass.set_bind_group(0, bg, &[]);
            }
            let wg_x = self.width.div_ceil(8);
            let wg_y = self.height.div_ceil(8);
            cpass.dispatch_workgroups(wg_x, wg_y, 1);
        }

        // Step 2: Render — fullscreen tri to output `pre_aa`
        {
            let rp = unsafe { &mut *ctx.active_render_pass_ptr().unwrap() };
            rp.set_pipeline(&self.shade_pipeline);
            if let Some(ref bg) = self.shade_bg {
                rp.set_bind_group(0, bg, &[]);
            }
            rp.draw(0..3, 0..1);
        }

        Ok(())
    }

    fn render_pass_descriptor<'a>(
        &'a self,
        _target: &'a wgpu::TextureView,
        _depth: &'a wgpu::TextureView,
        resources: &'a libhelio::FrameResources<'a>,
    ) -> Option<wgpu::RenderPassDescriptor<'a>> {
        let pre_aa_view = resources.pre_aa.read("VoxelRayMarch")?;
        let color_attachments: &'a [Option<wgpu::RenderPassColorAttachment<'a>>] = Box::leak(Box::new([
            Some(wgpu::RenderPassColorAttachment {
                view: pre_aa_view,
                resolve_target: None,
                depth_slice: None,
                ops: wgpu::Operations {
                    load: self.attachment_mode.color_load(),
                    store: wgpu::StoreOp::Store,
                },
            }),
        ]));
        Some(wgpu::RenderPassDescriptor {
            label: Some("VoxelRayMarch"),
            color_attachments,
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        })
    }

    fn on_resize(&mut self, device: &wgpu::Device, width: u32, height: u32) {
        if width == self.width && height == self.height {
            return;
        }
        self.width = width;
        self.height = height;

        let (ct, cv) = Self::create_tex(device, width, height, "VoxelRayMarch Color");
        let (nt, nv) = Self::create_tex(device, width, height, "VoxelRayMarch Normal");
        self.color_tex = ct;
        self.color_view = cv;
        self.normal_tex = nt;
        self.normal_view = nv;
        self.compute_bg_key = None;
        // The shading bind group owns views of the old-sized textures. Drop it
        // now so the next frame cannot stretch stale raymarch output.
        self.shade_bg = None;
        self.shade_bg_size = None;
    }
}

#[cfg(test)]
mod tests {
    use super::{shade_bind_group_matches_size, AttachmentMode};

    #[test]
    fn resized_output_invalidates_the_shading_bind_group() {
        assert!(shade_bind_group_matches_size(Some((1280, 720)), 1280, 720));
        assert!(!shade_bind_group_matches_size(
            Some((1280, 720)),
            1920,
            1080
        ));
        assert!(!shade_bind_group_matches_size(None, 1280, 720));
    }

    #[test]
    fn standalone_mode_clears_pixels_discarded_by_the_shade_pass() {
        assert!(matches!(
            AttachmentMode::Standalone.color_load(),
            wgpu::LoadOp::Clear(color) if color == wgpu::Color::TRANSPARENT
        ));
    }

    #[test]
    fn composited_mode_preserves_prior_color_on_ray_misses() {
        assert!(matches!(
            AttachmentMode::Composited.color_load(),
            wgpu::LoadOp::Load
        ));
    }
}
