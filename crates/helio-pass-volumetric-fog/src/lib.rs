//! Volumetric fog accumulation.
//!
//! Ray-marches the camera rays against the depth buffer, accumulating in-scattered
//! light and Beer-Lambert extinction into a `fog_accum` target that the post-process
//! uber shader composites. Full internal resolution by default — see
//! [`VolumetricFogPass::with_resolution_divisor`].
//!
//! # Placement
//!
//! This pass runs **before TAA**, at internal resolution. That is deliberate:
//! `depth` lives at internal resolution, so accumulating there keeps the fog in the
//! same space as its input, and TAA then resolves the fog along with everything
//! else — which is why the pass needs no jitter handling of its own. Running it
//! after TAA would mean reconciling an output-resolution target against
//! internal-resolution depth and de-syncing the jitter by hand.
//!
//! # Config
//!
//! Fog settings live in `GpuPostProcessUniforms` (so they blend through
//! post-process volumes like every other effect). This pass copies the 64-byte fog
//! block out of that buffer instead of mirroring all 368 bytes in WGSL — see
//! `libhelio::GpuFogUniforms`.

use bytemuck::{Pod, Zeroable};
use helio_core::graph::{ResourceBuilder, ResourceFormat, ResourceSize};
use helio_core::{PassContext, PrepareContext, RenderPass, Result as HelioResult};

/// Ray-march steps per pixel.
///
/// Steps are the dominant cost: each one does a shadow tap per in-scattering light,
/// and the pass now runs at full resolution, so this is the first dial to turn down
/// if the pass gets expensive.
const DEFAULT_STEPS: u32 = 64;

const WG_SIZE: u32 = 8;

/// Default divisor applied to the internal resolution for the fog target.
///
/// 1 = full internal resolution. The spec called for a quarter, but the composite
/// multiplies the scene by the fog transmittance — so a reduced-resolution fog
/// buffer does not merely make the *fog* coarse, it drags the geometry behind it
/// down to the fog's resolution, which reads as pixelated edges wherever fog is
/// present. Fixing that properly at reduced resolution needs a depth-aware
/// (bilateral) upsample; until then, full resolution is the honest default and
/// `with_resolution_divisor` is there when the cost matters.
const DEFAULT_FOG_DIVISOR: u32 = 1;

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct FogGlobals {
    csm_splits: [f32; 4],
    light_count: u32,
    frame: u32,
    steps: u32,
    _pad: u32,
}

pub struct VolumetricFogPass {
    pipeline: wgpu::ComputePipeline,
    bgl: wgpu::BindGroupLayout,
    /// The fog block, copied out of the post-process uniform buffer each frame.
    fog_uniform_buf: wgpu::Buffer,
    globals_buf: wgpu::Buffer,
    shadow_sampler: wgpu::Sampler,
    bind_group: Option<wgpu::BindGroup>,
    bind_group_key: Option<[usize; 4]>,
    steps: u32,
    frame: u32,
    resolution_divisor: u32,
}

impl VolumetricFogPass {
    pub fn new(device: &wgpu::Device) -> Self {
        let shader = helio_core::shader::module(
            device,
            "Volumetric Fog Shader",
            include_str!("../shaders/volumetric_fog.wgsl"),
        );

        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Volumetric Fog BGL"),
            entries: &[
                // 0: camera
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
                // 1: fog uniforms
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
                // 2: fog globals
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // 3: lights
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // 4: shadow matrices
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // 5: depth
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Depth,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // 6: shadow atlas
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Depth,
                        view_dimension: wgpu::TextureViewDimension::D2Array,
                        multisampled: false,
                    },
                    count: None,
                },
                // 7: shadow comparison sampler
                wgpu::BindGroupLayoutEntry {
                    binding: 7,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Comparison),
                    count: None,
                },
                // 8: fog_accum storage output
                wgpu::BindGroupLayoutEntry {
                    binding: 8,
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

        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Volumetric Fog PL"),
            bind_group_layouts: &[Some(&bgl)],
            immediate_size: 0,
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Volumetric Fog Pipeline"),
            layout: Some(&layout),
            module: &shader,
            entry_point: Some("cs_fog"),
            compilation_options: Default::default(),
            cache: None,
        });

        let fog_uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Volumetric Fog Uniforms"),
            size: libhelio::GpuPostProcessUniforms::FOG_BLOCK_SIZE,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let globals_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Volumetric Fog Globals"),
            size: std::mem::size_of::<FogGlobals>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let shadow_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Volumetric Fog Shadow Sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            compare: Some(wgpu::CompareFunction::LessEqual),
            ..Default::default()
        });

        Self {
            pipeline,
            bgl,
            fog_uniform_buf,
            globals_buf,
            shadow_sampler,
            bind_group: None,
            bind_group_key: None,
            steps: DEFAULT_STEPS,
            frame: 0,
            resolution_divisor: DEFAULT_FOG_DIVISOR,
        }
    }

    /// Render fog at `1/divisor` of the internal resolution. 1 = full res (default).
    ///
    /// Cost scales with the pixel count, so 2 is a 4x saving. Be aware that the
    /// composite multiplies scene colour by fog transmittance, so anything above 1
    /// coarsens the geometry seen *through* the fog, not just the fog itself.
    /// Must be called before the pass is added to the graph — `declare_resources`
    /// reads it to size the target.
    pub fn with_resolution_divisor(mut self, divisor: u32) -> Self {
        self.resolution_divisor = divisor.max(1);
        self
    }

    /// Override the ray-march step count. Cost scales linearly.
    pub fn set_steps(&mut self, steps: u32) {
        self.steps = steps.max(1);
    }
}

impl RenderPass for VolumetricFogPass {
    fn name(&self) -> &'static str {
        "VolumetricFogPass"
    }

    fn reads(&self) -> &'static [&'static str] {
        &["depth"]
    }

    fn writes(&self) -> &'static [&'static str] {
        &["fog_accum"]
    }

    fn declare_resources(&self, builder: &mut ResourceBuilder) {
        builder.read("depth");
        // ScaledInternal, not Scaled: this pairs with internal-resolution depth,
        // and Scaled divides the *output* resolution, which differs whenever
        // render_scale != 1. At divisor 1 this is the internal resolution exactly,
        // so the fog texel grid lines up 1:1 with depth and there is no upsample.
        builder.write_color(
            "fog_accum",
            ResourceFormat::Rgba16Float,
            ResourceSize::ScaledInternal {
                divisor: self.resolution_divisor,
            },
        );
        // The graph creates this texture; a compute shader writes it, so it needs
        // STORAGE_BINDING on top of the default render-attachment usage.
        builder.with_extra_usage(wgpu::TextureUsages::STORAGE_BINDING);
    }

    fn render_pass_descriptor<'a>(
        &'a self,
        _target: &'a wgpu::TextureView,
        _depth: &'a wgpu::TextureView,
        _resources: &'a libhelio::FrameResources<'a>,
    ) -> Option<wgpu::RenderPassDescriptor<'a>> {
        None
    }

    fn chain_transparent(&self) -> bool {
        // execute() only touches ctx.compute_encoder_ptr.
        true
    }

    fn prepare(&mut self, ctx: &PrepareContext) -> HelioResult<()> {
        self.frame = ctx.frame_num as u32;

        let globals = FogGlobals {
            csm_splits: libhelio::CSM_SPLITS,
            light_count: ctx.scene.lights.len() as u32,
            frame: self.frame,
            steps: self.steps,
            _pad: 0,
        };
        ctx.queue
            .write_buffer(&self.globals_buf, 0, bytemuck::bytes_of(&globals));
        Ok(())
    }

    fn execute(&mut self, ctx: &mut PassContext) -> HelioResult<()> {
        let Some(fog_view) = ctx.resources.fog_accum.get() else {
            return Ok(());
        };
        let Some(postprocess_buf) = ctx.resources.postprocess_uniforms.get() else {
            return Ok(());
        };
        let Some(shadow_atlas) = ctx.resources.shadow_atlas.get() else {
            return Ok(());
        };

        let camera_buf = ctx.scene.camera;
        let lights_buf = ctx.scene.lights;
        let shadow_matrices = ctx.scene.shadow_matrices;

        let key = [
            fog_view as *const _ as usize,
            ctx.depth as *const _ as usize,
            shadow_atlas as *const _ as usize,
            lights_buf as *const _ as usize,
        ];
        if self.bind_group_key != Some(key) {
            self.bind_group = Some(ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Volumetric Fog BG"),
                layout: &self.bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: camera_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: self.fog_uniform_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: self.globals_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: lights_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: shadow_matrices.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: wgpu::BindingResource::TextureView(ctx.depth),
                    },
                    wgpu::BindGroupEntry {
                        binding: 6,
                        resource: wgpu::BindingResource::TextureView(shadow_atlas),
                    },
                    wgpu::BindGroupEntry {
                        binding: 7,
                        resource: wgpu::BindingResource::Sampler(&self.shadow_sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 8,
                        resource: wgpu::BindingResource::TextureView(fog_view),
                    },
                ],
            }));
            self.bind_group_key = Some(key);
        }

        let Some(bind_group) = self.bind_group.as_ref() else {
            return Ok(());
        };

        let ce = ctx.compute_encoder_ptr;

        // Pull this frame's fog config out of the post-process uniform buffer.
        // Copying the block keeps PostProcessSettings the single source of truth
        // instead of introducing a parallel fog upload path.
        unsafe { &mut *ce }.copy_buffer_to_buffer(
            postprocess_buf,
            libhelio::GpuPostProcessUniforms::FOG_BLOCK_OFFSET,
            &self.fog_uniform_buf,
            0,
            libhelio::GpuPostProcessUniforms::FOG_BLOCK_SIZE,
        );

        let (w, h) = (
            ctx.width / self.resolution_divisor,
            ctx.height / self.resolution_divisor,
        );
        let groups_x = (w.max(1) + WG_SIZE - 1) / WG_SIZE;
        let groups_y = (h.max(1) + WG_SIZE - 1) / WG_SIZE;

        let mut cpass = unsafe { &mut *ce }.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Volumetric Fog"),
            timestamp_writes: None,
        });
        cpass.set_pipeline(&self.pipeline);
        cpass.set_bind_group(0, bind_group, &[]);
        cpass.dispatch_workgroups(groups_x, groups_y, 1);

        Ok(())
    }
}
