//! Water caustics generation pass (compute shader).
//!
//! Generates caustic patterns by simulating light refraction through
//! animated water surfaces. Uses a compute shader to ray-trace light
//! through Gerstner wave displacement.
//!
//! # Algorithm
//! 1. For each texel in the 512×512 caustics texture:
//!    - Sample water surface height at corresponding world XZ position
//!    - Calculate surface normal from Gerstner waves
//!    - Refract sunlight direction through the surface
//!    - Trace to bottom plane and measure light convergence
//! 2. Output caustic intensity to R16Float texture
//! 3. Result is tiled across the scene in subsequent passes
//!
//! # Performance
//! - Resolution: 512×512 (262,144 threads)
//! - Workgroup size: 32×32 (1,024 threads per group)
//! - Dispatch: 16×16 workgroups (256 total)
//! - Estimated cost: ~0.2ms @ 4K
//!
//! # Integration
//! - Runs after deferred lighting, before water surface pass
//! - Publishes `frame.water_caustics` for downstream passes
//! - Can be run every N frames (e.g., every 2 frames) to save cost

use bytemuck::{Pod, Zeroable};
use helio_v3::{PassContext, PrepareContext, RenderPass, Result as HelioResult};

/// Caustics generation parameters (uniform buffer).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct CausticsParams {
    /// Current time in seconds (for wave animation)
    time: f32,
    /// World-space scale for caustics tiling
    world_scale: f32,
    /// World-space offset for caustics origin
    world_offset: [f32; 2],
}

/// Water caustics generation pass.
///
/// Generates realistic caustic patterns using GPU ray-tracing through
/// animated water surfaces. Outputs to a 512×512 R16Float texture that
/// is tiled across the scene.
///
/// # Example
/// ```ignore
/// let caustics_pass = WaterCausticsPass::new(device);
/// graph.add_pass(Box::new(caustics_pass));
/// ```
pub struct WaterCausticsPass {
    pipeline: wgpu::ComputePipeline,
    bgl: wgpu::BindGroupLayout,
    params_buf: wgpu::Buffer,

    /// Caustics texture (512×512 R16Float)
    pub caustics_texture: wgpu::Texture,
    /// Caustics texture view for shader binding
    pub caustics_view: wgpu::TextureView,

    bind_group: Option<wgpu::BindGroup>,
}

impl WaterCausticsPass {
    /// Create a new water caustics pass.
    ///
    /// # Parameters
    /// - `device`: GPU device for resource creation
    ///
    /// # Returns
    /// A new `WaterCausticsPass` ready to be added to the render graph.
    pub fn new(device: &wgpu::Device) -> Self {
        // Load and compile compute shader
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Water Caustics Compute"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/caustics.wgsl").into()),
        });

        // Bind group layout
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Caustics BGL"),
            entries: &[
                // @binding(0) params uniform
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // @binding(1) water volumes storage buffer
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // @binding(2) output texture (storage, write-only)
                // Use RGBA16Float for broad compatibility on both desktop and web adapters.
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rgba16Float,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
            ],
        });

        // Pipeline layout
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Caustics Pipeline Layout"),
            bind_group_layouts: &[Some(&bgl)],
            immediate_size: 0,
        });

        // Compute pipeline
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Caustics Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        // Create caustics texture (512×512 RGBA16Float for wide storage-texture compatibility)
        let caustics_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Caustics Texture"),
            size: wgpu::Extent3d {
                width: 512,
                height: 512,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });

        let caustics_view = caustics_texture.create_view(&wgpu::TextureViewDescriptor::default());

        // Uniform buffer for parameters
        let params_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Caustics Params"),
            size: std::mem::size_of::<CausticsParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            pipeline,
            bgl,
            params_buf,
            caustics_texture,
            caustics_view,
            bind_group: None,
        }
    }
}

impl RenderPass for WaterCausticsPass {
    fn name(&self) -> &'static str {
        "WaterCaustics"
    }

    fn publish<'a>(&'a self, frame: &mut libhelio::FrameResources<'a>) {
        frame.water_caustics = Some(&self.caustics_view);
    }

    fn prepare(&mut self, ctx: &PrepareContext) -> HelioResult<()> {
        // Update uniform parameters
        let params = CausticsParams {
            time: ctx.frame as f32 * 0.016, // Assuming 60 FPS
            world_scale: 100.0,              // Tile caustics across 100m
            world_offset: [0.0, 0.0],        // Origin at world center
        };
        ctx.queue
            .write_buffer(&self.params_buf, 0, bytemuck::bytes_of(&params));
        Ok(())
    }

    fn execute(&mut self, ctx: &mut PassContext) -> HelioResult<()> {
        // Skip if no water volumes
        if let Some(volumes_buf) = ctx.frame.water_volumes {
            if ctx.frame.water_volume_count == 0 {
                return Ok(());
            }

            // Rebuild bind group if needed (only once per render)
            if self.bind_group.is_none() {
                self.bind_group = Some(ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("Caustics BG"),
                    layout: &self.bgl,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: self.params_buf.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: volumes_buf.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: wgpu::BindingResource::TextureView(&self.caustics_view),
                        },
                    ],
                }));
            }

            // Dispatch compute shader
            // 512×512 texture / 32×32 workgroup = 16×16 workgroups
            let mut pass = ctx.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("WaterCaustics"),
                timestamp_writes: None,
            });

            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, self.bind_group.as_ref().unwrap(), &[]);
            pass.dispatch_workgroups(16, 16, 1);
        }

        Ok(())
    }
}
