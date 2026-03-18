//! Radiance Cascades GI pass.
//!
//! Traces screen-space radiance cascades for real-time global illumination using
//! hardware ray queries (wgpu `EXPERIMENTAL_RAY_QUERY` feature).
//!
//! O(1) CPU: single compute dispatch per frame.
//!
//! # Bindings (from rc_trace.wgsl group 0)
//! | b | Name                  | Type                                      |
//! |---|-----------------------|-------------------------------------------|
//! | 0 | cascade_out           | texture_storage_2d<rgba16float, write>    |
//! | 1 | cascade_parent        | texture_2d<f32>                           |
//! | 2 | rc_dyn                | uniform RCDynamic                         |
//! | 3 | rc_stat               | uniform CascadeStatic                     |
//! | 4 | acc_struct            | acceleration_structure (TLAS)             |
//! | 5 | lights                | storage array<GpuLight>                   |
//! | 6 | cascade_history       | texture_2d<f32>                           |
//! | 7 | cascade_history_write | texture_storage_2d<rgba16float, write>    |
//!
//! Requires `wgpu::Features::EXPERIMENTAL_RAY_QUERY` on the wgpu device.

use helio_v3::{RenderPass, PassContext, PrepareContext, Result as HelioResult};
use bytemuck::{Pod, Zeroable};

/// Probe grid dimension (one axis). Probes are probe_dim³.
const PROBE_DIM: u32 = 8;
/// Direction bins per atlas axis.
const DIR_DIM: u32 = 4;
/// Atlas width  = PROBE_DIM * DIR_DIM.
const ATLAS_W: u32 = PROBE_DIM * DIR_DIM;       // 32
/// Atlas height = PROBE_DIM² * DIR_DIM.
const ATLAS_H: u32 = PROBE_DIM * PROBE_DIM * DIR_DIM; // 256

const WORKGROUP_SIZE_X: u32 = 8;
const WORKGROUP_SIZE_Y: u32 = 8;

/// Dynamic (per-frame) RC uniforms — must match `RCDynamic` in rc_trace.wgsl.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct RCDynamic {
    world_min:   [f32; 4],
    world_max:   [f32; 4],
    frame:       u32,
    light_count: u32,
    _pad0:       u32,
    _pad1:       u32,
    sky_color:   [f32; 4],
}

/// Static per-cascade uniforms — must match `CascadeStatic` in rc_trace.wgsl.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct CascadeStatic {
    cascade_index:    u32,
    probe_dim:        u32,
    dir_dim:          u32,
    t_max_bits:       u32,
    parent_probe_dim: u32,
    parent_dir_dim:   u32,
    _pad0:            u32,
    _pad1:            u32,
}

pub struct RadianceCascadesPass {
    pipeline:         wgpu::ComputePipeline,
    #[allow(dead_code)]
    bind_group_layout: wgpu::BindGroupLayout,
    bind_group:       wgpu::BindGroup,
    uniform_buf:      wgpu::Buffer,   // RCDynamic — updated each frame
    #[allow(dead_code)]
    static_buf:       wgpu::Buffer,   // CascadeStatic — constant after construction
    /// Main cascade output texture (readable by later passes for GI).
    pub cascade_texture: wgpu::Texture,
    pub cascade_view:    wgpu::TextureView,
    #[allow(dead_code)]
    history_texture: wgpu::Texture,
    #[allow(dead_code)]
    history_write_view: wgpu::TextureView,
    #[allow(dead_code)]
    dummy_parent_texture: wgpu::Texture,
    // TLAS owned by this pass; caller must build it via queue.build_acceleration_structures.
    pub tlas:         wgpu::Tlas,
}

impl RadianceCascadesPass {
    /// Create the radiance cascades pass.
    ///
    /// Requires `wgpu::Features::EXPERIMENTAL_RAY_QUERY` on the device.
    ///
    /// - `lights_buf` — GPU light storage buffer (must match `GpuLight` in rc_trace.wgsl)
    pub fn new(
        device:     &wgpu::Device,
        lights_buf: &wgpu::Buffer,
    ) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label:  Some("RC Trace Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/rc_trace.wgsl").into()),
        });

        // Pipeline with auto-reflected layout (layout: None).
        // This infers all 8 binding types from the WGSL shader at runtime.
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label:               Some("RC Trace Pipeline"),
            layout:              None,
            module:              &shader,
            entry_point:         Some("cs_trace"),
            compilation_options: Default::default(),
            cache:               None,
        });
        let bind_group_layout = pipeline.get_bind_group_layout(0);

        // ── Uniform buffers ───────────────────────────────────────────────────
        let uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label:              Some("RC Dynamic Uniform"),
            size:               std::mem::size_of::<RCDynamic>() as u64,
            usage:              wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let t_max: f32 = 2.0; // metres — cascade 0 range
        let static_data = CascadeStatic {
            cascade_index:    0,
            probe_dim:        PROBE_DIM,
            dir_dim:          DIR_DIM,
            t_max_bits:       t_max.to_bits(),
            parent_probe_dim: 0, // no coarser parent for single cascade
            parent_dir_dim:   0,
            _pad0:            0,
            _pad1:            0,
        };
        let static_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label:              Some("RC Static Uniform"),
            size:               std::mem::size_of::<CascadeStatic>() as u64,
            usage:              wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        // Static data never changes after construction; written immediately.
        device.queue_write_buffer_static(&static_buf, 0, bytemuck::bytes_of(&static_data));

        // ── Cascade textures ──────────────────────────────────────────────────
        let cascade_desc = wgpu::TextureDescriptor {
            label:               Some("RC Cascade"),
            size:                wgpu::Extent3d { width: ATLAS_W, height: ATLAS_H, depth_or_array_layers: 1 },
            mip_level_count:     1,
            sample_count:        1,
            dimension:           wgpu::TextureDimension::D2,
            format:              wgpu::TextureFormat::Rgba16Float,
            usage:               wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats:        &[],
        };
        let cascade_texture = device.create_texture(&cascade_desc);
        let cascade_view    = cascade_texture.create_view(&wgpu::TextureViewDescriptor::default());

        // History texture: ping-pong buffer for temporal accumulation.
        let history_texture    = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("RC History"), ..cascade_desc
        });
        let history_read_view  = history_texture.create_view(&wgpu::TextureViewDescriptor::default());
        let history_write_view = history_texture.create_view(&wgpu::TextureViewDescriptor::default());

        // Dummy parent texture (1×1) used when there is no coarser cascade level.
        let dummy_parent_texture = device.create_texture(&wgpu::TextureDescriptor {
            label:               Some("RC Dummy Parent"),
            size:                wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
            mip_level_count:     1,
            sample_count:        1,
            dimension:           wgpu::TextureDimension::D2,
            format:              wgpu::TextureFormat::Rgba16Float,
            usage:               wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats:        &[],
        });
        let dummy_parent_view = dummy_parent_texture.create_view(&wgpu::TextureViewDescriptor::default());

        // ── TLAS — empty initially; caller must build via queue.build_acceleration_structures ──
        let tlas = device.create_tlas(&wgpu::CreateTlasDescriptor {
            label:       Some("RC TLAS"),
            flags:       wgpu::AccelerationStructureFlags::PREFER_FAST_TRACE,
            update_mode: wgpu::AccelerationStructureUpdateMode::Build,
            max_instances: 2048,
        });

        // ── Bind group ────────────────────────────────────────────────────────
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label:  Some("RC Bind Group"),
            layout: &bind_group_layout,
            entries: &[
                // b0: cascade_out (storage write)
                wgpu::BindGroupEntry {
                    binding:  0,
                    resource: wgpu::BindingResource::TextureView(&cascade_view),
                },
                // b1: cascade_parent (texture read — dummy for single cascade)
                wgpu::BindGroupEntry {
                    binding:  1,
                    resource: wgpu::BindingResource::TextureView(&dummy_parent_view),
                },
                // b2: rc_dyn uniform
                wgpu::BindGroupEntry { binding: 2, resource: uniform_buf.as_entire_binding() },
                // b3: rc_stat uniform
                wgpu::BindGroupEntry { binding: 3, resource: static_buf.as_entire_binding() },
                // b4: acceleration structure (TLAS)
                wgpu::BindGroupEntry {
                    binding:  4,
                    resource: wgpu::BindingResource::AccelerationStructure(&tlas),
                },
                // b5: lights storage buffer
                wgpu::BindGroupEntry { binding: 5, resource: lights_buf.as_entire_binding() },
                // b6: cascade_history (texture read)
                wgpu::BindGroupEntry {
                    binding:  6,
                    resource: wgpu::BindingResource::TextureView(&history_read_view),
                },
                // b7: cascade_history_write (storage write)
                wgpu::BindGroupEntry {
                    binding:  7,
                    resource: wgpu::BindingResource::TextureView(&history_write_view),
                },
            ],
        });

        Self {
            pipeline,
            bind_group_layout,
            bind_group,
            uniform_buf,
            static_buf,
            cascade_texture,
            cascade_view,
            history_texture,
            history_write_view,
            dummy_parent_texture,
            tlas,
        }
    }
}

impl RenderPass for RadianceCascadesPass {
    fn name(&self) -> &'static str { "RadianceCascades" }

    fn prepare(&mut self, ctx: &PrepareContext) -> HelioResult<()> {
        let light_count = ctx.scene.lights.len() as u32;
        let dyn_data = RCDynamic {
            world_min:   [-10.0, -1.0, -10.0, 0.0],
            world_max:   [ 10.0, 10.0,  10.0, 0.0],
            frame:       ctx.frame as u32,
            light_count,
            _pad0:       0,
            _pad1:       0,
            sky_color:   [0.0; 4],
        };
        ctx.queue.write_buffer(&self.uniform_buf, 0, bytemuck::bytes_of(&dyn_data));
        Ok(())
    }

    fn execute(&mut self, ctx: &mut PassContext) -> HelioResult<()> {
        // O(1): single compute dispatch — constant workgroup count for fixed atlas.
        let wg_x = ATLAS_W.div_ceil(WORKGROUP_SIZE_X);  // 32 / 8 = 4
        let wg_y = ATLAS_H.div_ceil(WORKGROUP_SIZE_Y);  // 256 / 8 = 32

        let desc = wgpu::ComputePassDescriptor {
            label:            Some("RadianceCascades"),
            timestamp_writes: None,
        };
        let mut pass = ctx.encoder.begin_compute_pass(&desc);
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.bind_group, &[]);
        pass.dispatch_workgroups(wg_x, wg_y, 1);
        Ok(())
    }
}
