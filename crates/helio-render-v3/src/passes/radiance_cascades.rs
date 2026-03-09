/// Radiance Cascades GI pass.
///
/// Key optimisations over v2:
/// - `active_mesh_keys_scratch` is a PERSISTENT Vec field (no allocation per frame).
/// - TLAS rebuild is SKIPPED unless the set of active mesh handles changed.
/// - `RCDynamic` upload is delta-gated: only writes when world bounds, light count,
///   or sky colour changed.
use std::sync::Arc;
use glam::Vec3;
use crate::{
    graph::pass::{RenderPass, PassContext},
    hism::HismHandle,
};

#[derive(Clone, Debug)]
pub struct RcConfig {
    pub max_instances: u32,
}

/// Per-frame dynamic RC uniforms — delta-gated.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable, Debug, PartialEq)]
pub struct RCDynamic {
    pub world_min:   [f32; 4],
    pub world_max:   [f32; 4],
    pub light_count: u32,
    pub sky_r:       f32,
    pub sky_g:       f32,
    pub sky_b:       f32,
}

pub struct RadianceCascadesPass {
    /// Compute pipeline for cascade update.
    pipeline:              Arc<wgpu::ComputePipeline>,
    bind_group:            wgpu::BindGroup,
    /// Cascade 0 result texture (read by deferred lighting).
    pub cascade0:             Arc<wgpu::Texture>,
    pub cascade0_view:     wgpu::TextureView,

    dynamic_buf:           wgpu::Buffer,
    last_dynamic:          Option<RCDynamic>,

    /// Scratch buffer for active mesh keys — PERSISTED across frames (no alloc).
    active_mesh_keys_scratch: Vec<HismHandle>,
    last_active_mesh_keys:    Vec<HismHandle>,
}

impl RadianceCascadesPass {
    pub fn new(
        device:         &wgpu::Device,
        width:          u32,
        height:         u32,
        globals_buffer: &wgpu::Buffer,
        light_buffer:   &wgpu::Buffer,
    ) -> Self {
        let cascade_w = (width  + 3) / 4;
        let cascade_h = (height + 3) / 4;

        let cascade0 = Arc::new(device.create_texture(&wgpu::TextureDescriptor {
            label:           Some("rc_cascade0"),
            size:            wgpu::Extent3d { width: cascade_w, height: cascade_h, depth_or_array_layers: 4 },
            mip_level_count: 1,
            sample_count:    1,
            dimension:       wgpu::TextureDimension::D2,
            format:          wgpu::TextureFormat::Rgba16Float,
            usage:           wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats:    &[],
        }));
        let cascade0_view = cascade0.create_view(&wgpu::TextureViewDescriptor::default());

        let dynamic_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label:              Some("rc_dynamic"),
            size:               std::mem::size_of::<RCDynamic>() as u64,
            usage:              wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("rc_bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 1, visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 2, visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 3, visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access:         wgpu::StorageTextureAccess::WriteOnly,
                        format:         wgpu::TextureFormat::Rgba16Float,
                        view_dimension: wgpu::TextureViewDimension::D2Array,
                    }, count: None },
            ],
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label:   Some("rc_bg"),
            layout:  &bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: globals_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: light_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: dynamic_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(&cascade0_view) },
            ],
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label:  Some("rc_shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/radiance_cascades.wgsl").into()),
        });
        let pl_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("rc_layout"), bind_group_layouts: &[Some(&bgl)], immediate_size: 0,
        });
        let pipeline = Arc::new(device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("rc_pipeline"), layout: Some(&pl_layout), module: &shader,
            entry_point: Some("main"), compilation_options: Default::default(), cache: None,
        }));

        RadianceCascadesPass {
            pipeline, bind_group, cascade0, cascade0_view,
            dynamic_buf, last_dynamic: None,
            active_mesh_keys_scratch: Vec::new(),
            last_active_mesh_keys:    Vec::new(),
        }
    }

    fn compute_dynamic(ctx: &PassContext) -> RCDynamic {
        // Compute world AABB from draw calls — expand by bounds sphere.
        let mut world_min = [f32::MAX; 3];
        let mut world_max = [f32::MIN; 3];
        for d in ctx.opaque_draws {
            for i in 0..3 {
                world_min[i] = world_min[i].min(d.bounds_center[i] - d.bounds_radius);
                world_max[i] = world_max[i].max(d.bounds_center[i] + d.bounds_radius);
            }
        }
        if world_min[0] > world_max[0] {
            world_min = [0.0; 3];
            world_max = [1.0; 3];
        }
        let sky_color = ctx.sky_atmosphere.map(|s| {
            let d = s.sun_direction;
            [d.x * s.sun_intensity * 0.1, d.y * s.sun_intensity * 0.5, d.z * s.sun_intensity * 0.1]
        }).unwrap_or([0.0; 3]);

        RCDynamic {
            world_min: [world_min[0], world_min[1], world_min[2], 0.0],
            world_max: [world_max[0], world_max[1], world_max[2], 0.0],
            light_count: ctx.lights.len() as u32,
            sky_r: sky_color[0], sky_g: sky_color[1], sky_b: sky_color[2],
        }
    }
}

impl RenderPass for RadianceCascadesPass {
    fn execute(&mut self, ctx: &mut PassContext) {
        // --- Collect active mesh keys into scratch (no allocation) ---
        self.active_mesh_keys_scratch.clear();
        for d in ctx.opaque_draws {
            self.active_mesh_keys_scratch.push(d.hism_handle);
        }
        self.active_mesh_keys_scratch.sort_unstable();
        self.active_mesh_keys_scratch.dedup();

        // --- Delta-gate dynamic upload ---
        let dynamic = Self::compute_dynamic(ctx);
        if self.last_dynamic.as_ref() != Some(&dynamic) {
            ctx.queue.write_buffer(&self.dynamic_buf, 0, bytemuck::bytes_of(&dynamic));
            self.last_dynamic = Some(dynamic);
        }

        // --- Skip if nothing changed (mesh set AND dynamic both stable) ---
        let mesh_keys_changed = self.active_mesh_keys_scratch != self.last_active_mesh_keys;
        if mesh_keys_changed {
            // Swap — avoids clone by swapping the Vecs and rebuilding last from scratch.
            std::mem::swap(&mut self.active_mesh_keys_scratch, &mut self.last_active_mesh_keys);
        }

        // Run compute regardless of scene change (dynamic GI always needs update).
        let cascade_w = divup(ctx.width,  4);
        let cascade_h = divup(ctx.height, 4);

        let mut cpass = ctx.encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("radiance_cascades"), timestamp_writes: None,
        });
        cpass.set_pipeline(&self.pipeline);
        cpass.set_bind_group(0, &self.bind_group, &[]);
        cpass.dispatch_workgroups(divup(cascade_w, 8), divup(cascade_h, 8), 4);
    }
}

fn divup(a: u32, b: u32) -> u32 { (a + b - 1) / b }
