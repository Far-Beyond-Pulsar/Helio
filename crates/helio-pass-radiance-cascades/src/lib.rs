//! Radiance Cascades GI pass.
//!
//! Traces screen-space radiance cascades for real-time global illumination.
//!
//! # wgpu 23 compatibility note
//!
//! The full `rc_trace.wgsl` shader requires `wgpu::Features::EXPERIMENTAL_RAY_QUERY`
//! (hardware ray tracing / TLAS), which was added to wgpu after 23.0.1.
//! This crate ships the verbatim `rc_trace.wgsl` shader for when the upgrade lands, but
//! the current Rust implementation uses a lightweight fallback compute shader that writes
//! a black cascade atlas so downstream passes have a valid texture to sample.
//!
//! The `cascade_texture` / `cascade_view` public API is fully functional; swapping in
//! the real shader requires upgrading the `wgpu` dependency and rebuilding the bind group
//! with a `wgpu::Tlas`.
//!
//! O(1) CPU — single `dispatch_workgroups` call.

// rc_trace.wgsl is bundled verbatim for inspection and future use.
// It requires `enable wgpu_ray_query` which is not available in wgpu 23.0.1.
const _RC_TRACE_WGSL: &str = include_str!("../shaders/rc_trace.wgsl");

use bytemuck::{Pod, Zeroable};
use helio_v3::{PassContext, PrepareContext, RenderPass, Result as HelioResult};

/// Probe grid dimension (one axis). Probes are PROBE_DIM³.
const PROBE_DIM: u32 = 8;
/// Direction bins per atlas axis.
const DIR_DIM: u32 = 4;
/// Atlas width  = PROBE_DIM * DIR_DIM = 32.
const ATLAS_W: u32 = PROBE_DIM * DIR_DIM;
/// Atlas height = PROBE_DIM² * DIR_DIM = 256.
const ATLAS_H: u32 = PROBE_DIM * PROBE_DIM * DIR_DIM;

const WORKGROUP_SIZE_X: u32 = 8;
const WORKGROUP_SIZE_Y: u32 = 8;

/// Per-frame dynamic RC uniforms.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct RCDynamic {
    world_min: [f32; 4],
    world_max: [f32; 4],
    frame: u32,
    light_count: u32,
    _pad0: u32,
    _pad1: u32,
    sky_color: [f32; 4],
}

pub struct RadianceCascadesPass {
    pipeline: wgpu::ComputePipeline,
    bind_group: wgpu::BindGroup,
    uniform_buf: wgpu::Buffer,
    /// Main cascade atlas texture (Rgba16Float). Downstream passes sample this for GI.
    pub cascade_texture: wgpu::Texture,
    pub cascade_view: wgpu::TextureView,
}

/// Minimal fallback WGSL shader — clears the cascade atlas to black.
///
/// Used in place of `rc_trace.wgsl` until the wgpu dependency is upgraded to a version
/// that exposes `wgpu::Tlas` / `EXPERIMENTAL_RAY_QUERY`.
const FALLBACK_WGSL: &str = r#"
struct RCDynamic {
    world_min:   vec4<f32>,
    world_max:   vec4<f32>,
    frame:       u32,
    light_count: u32,
    _pad0:       u32,
    _pad1:       u32,
    sky_color:   vec4<f32>,
}
@group(0) @binding(0) var cascade_out: texture_storage_2d<rgba16float, write>;
@group(0) @binding(1) var<uniform>  rc_dyn: RCDynamic;

@compute @workgroup_size(8, 8)
fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(cascade_out);
    if gid.x >= dims.x || gid.y >= dims.y { return; }
    // Write sky colour as ambient fallback (black until real RT is wired up).
    textureStore(cascade_out, vec2<i32>(i32(gid.x), i32(gid.y)),
        vec4<f32>(rc_dyn.sky_color.rgb * 0.05, 1.0));
}
"#;

impl RadianceCascadesPass {
    /// Create the radiance cascades pass.
    ///
    /// - `lights_buf` — kept for API compatibility with the full rc_trace.wgsl signature.
    ///   The fallback shader does not use it; it will be bound once the real RT shader is active.
    pub fn new(device: &wgpu::Device, lights_buf: &wgpu::Buffer) -> Self {
        let _ = lights_buf; // reserved for the full RT implementation

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("RC Fallback Shader"),
            source: wgpu::ShaderSource::Wgsl(FALLBACK_WGSL.into()),
        });

        // ── Cascade atlas texture ─────────────────────────────────────────────
        let cascade_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("RC Cascade"),
            size: wgpu::Extent3d {
                width: ATLAS_W,
                height: ATLAS_H,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let cascade_view = cascade_texture.create_view(&wgpu::TextureViewDescriptor::default());

        // ── Dynamic uniform buffer ────────────────────────────────────────────
        let uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("RC Dynamic Uniform"),
            size: std::mem::size_of::<RCDynamic>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // ── Bind group layout ─────────────────────────────────────────────────
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("RC Fallback BGL"),
            entries: &[
                // b0: cascade_out (storage texture write)
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rgba16Float,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
                // b1: rc_dyn uniform
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("RC Fallback BG"),
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&cascade_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: uniform_buf.as_entire_binding(),
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("RC Fallback PL"),
            bind_group_layouts: &[Some(&bgl)],
            immediate_size: 0,
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("RC Fallback Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("cs_main"),
            compilation_options: Default::default(),
            cache: None,
        });

        Self {
            pipeline,
            bind_group,
            uniform_buf,
            cascade_texture,
            cascade_view,
        }
    }
}

impl RenderPass for RadianceCascadesPass {
    fn name(&self) -> &'static str {
        "RadianceCascades"
    }

    fn prepare(&mut self, ctx: &PrepareContext) -> HelioResult<()> {
        let light_count = ctx.scene.lights.len() as u32;
        let sky = ctx.frame_resources.sky.sky_color;
        let dyn_data = RCDynamic {
            world_min: [-10.0, -1.0, -10.0, 0.0],
            world_max: [10.0, 10.0, 10.0, 0.0],
            frame: ctx.frame_num as u32,
            light_count,
            _pad0: 0,
            _pad1: 0,
            sky_color: [sky[0], sky[1], sky[2], 0.0],
        };
        ctx.write_buffer(&self.uniform_buf, 0, bytemuck::bytes_of(&dyn_data));
        Ok(())
    }

    fn execute(&mut self, ctx: &mut PassContext) -> HelioResult<()> {
        // O(1): single compute dispatch — constant workgroup count for fixed atlas size.
        let wg_x = ATLAS_W.div_ceil(WORKGROUP_SIZE_X); // 32 / 8 = 4
        let wg_y = ATLAS_H.div_ceil(WORKGROUP_SIZE_Y); // 256 / 8 = 32

        let desc = wgpu::ComputePassDescriptor {
            label: Some("RadianceCascades"),
            timestamp_writes: None,
        };
        let mut pass = ctx.encoder.begin_compute_pass(&desc);
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.bind_group, &[]);
        pass.dispatch_workgroups(wg_x, wg_y, 1);
        Ok(())
    }
    fn as_any(&self) -> &dyn std::any::Any { self }
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any { self }
}

