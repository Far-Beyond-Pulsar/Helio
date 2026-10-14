//! Physically based screen-space lens: a focus field over the sensor and the
//! thin-lens defocus it implies, in scene-linear radiance. See README.md for
//! the optical model and the `lens_color` / `lens_resolved` graph contract.

use helio_core::graph::ResourceBuilder;
use helio_core::{
    CommandRecorder, PassContext, PrepareContext, RenderPass, ResourceKey, Result as HelioResult,
};

mod components;
pub use components::*;
mod gpu_types;
pub use gpu_types::*;

/// Scene-linear image after the lens, at the size of the pass's input.
pub const LENS_COLOR: &str = "lens_color";
pub const OUTPUT_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;

/// Tile edge of the circle-of-confusion extent grid; equals `TILE` in the WGSL.
const TILE: u32 = 16;

const COMPUTE: wgpu::ShaderStages = wgpu::ShaderStages::COMPUTE;

fn image(device: &wgpu::Device, width: u32, height: u32, label: &str) -> wgpu::TextureView {
    device
        .create_texture(&wgpu::TextureDescriptor {
            label: Some(label),
            size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: OUTPUT_FORMAT,
            usage: wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        })
        .create_view(&Default::default())
}

/// Everything sized by the input image.
struct Targets {
    size: (u32, u32),
    /// Per pixel: circle of confusion a and b, seam weight, transmission.
    coc: wgpu::TextureView,
    half_color: wgpu::TextureView,
    half_coc: wgpu::TextureView,
    far: wgpu::TextureView,
    near: wgpu::TextureView,
    output: wgpu::TextureView,
    tile_raw: wgpu::Buffer,
    tile_dilated: wgpu::Buffer,
}

impl Targets {
    fn new(device: &wgpu::Device, width: u32, height: u32) -> Self {
        let (width, height) = (width.max(1), height.max(1));
        let (half_w, half_h) = (width.div_ceil(2), height.div_ceil(2));
        let tiles = u64::from(width.div_ceil(TILE)) * u64::from(height.div_ceil(TILE));
        let tile_buffer = |label| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: tiles * 16,
                usage: wgpu::BufferUsages::STORAGE,
                mapped_at_creation: false,
            })
        };
        Self {
            size: (width, height),
            coc: image(device, width, height, "Lens circles of confusion"),
            half_color: image(device, half_w, half_h, "Lens half-resolution colour"),
            half_coc: image(device, half_w, half_h, "Lens half-resolution circles"),
            far: image(device, half_w, half_h, "Lens far layer"),
            near: image(device, half_w, half_h, "Lens near layer"),
            output: image(device, width, height, "Lens colour"),
            tile_raw: tile_buffer("Lens tile extents"),
            tile_dilated: tile_buffer("Lens dilated tile extents"),
        }
    }
}

/// Scene inputs. Every field is optional: missing producers fall back to
/// neutral stand-ins (no lens row, no elements).
#[derive(Default, Clone, Copy)]
pub struct LensInputs<'a> {
    pub camera: Option<&'a wgpu::Buffer>,
    pub depth: Option<&'a wgpu::TextureView>,
    pub lenses: Option<&'a wgpu::Buffer>,
    pub elements: Option<&'a wgpu::Buffer>,
}

type BindingKey = ([Option<wgpu::TextureView>; 2], [Option<wgpu::Buffer>; 3]);

// Stages in recording order; `args` holds one indirect triple per stage after
// the resolve.
const RESOLVE: usize = 0;
const COC: usize = 1;
const TILES: usize = 2;
const DILATE: usize = 3;
const PREFILTER: usize = 4;
const GATHER: usize = 5;
const COMPOSITE: usize = 6;
const STAGES: usize = 7;
const ARGS_SIZE: u64 = 18 * 4;

struct Stage {
    entry: &'static str,
    label: &'static str,
    layout: wgpu::BindGroupLayout,
    pipeline: wgpu::ComputePipeline,
}

fn storage(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn uniform(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn texture(binding: u32, filterable: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: COMPUTE,
        ty: wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable },
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled: false,
        },
        count: None,
    }
}

fn storage_texture(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: COMPUTE,
        ty: wgpu::BindingType::StorageTexture {
            access: wgpu::StorageTextureAccess::WriteOnly,
            format: OUTPUT_FORMAT,
            view_dimension: wgpu::TextureViewDimension::D2,
        },
        count: None,
    }
}

fn sampler(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: COMPUTE,
        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
        count: None,
    }
}

fn entry(binding: u32, resource: wgpu::BindingResource<'_>) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry { binding, resource }
}

/// All persistent state is derived GPU machinery: the autofocus position and
/// the resolved lens. No CPU copy of rows or settings.
pub struct LensPass {
    input_key: &'static str,
    stages: [Stage; STAGES],
    sampler: wgpu::Sampler,
    /// Indirect dispatch triples, sized by `cs_resolve` from the lens row.
    args: wgpu::Buffer,
    /// Autofocus position, persistent between frames.
    state: wgpu::Buffer,
    /// This frame's [`ResolvedLens`], published as [`RESOLVED_KEY`].
    resolved: wgpu::Buffer,
    /// Frame time in seconds, written in prepare().
    frame: wgpu::Buffer,
    fallback_camera: wgpu::Buffer,
    fallback_lenses: wgpu::Buffer,
    fallback_elements: wgpu::Buffer,
    fallback_depth: wgpu::TextureView,
    targets: Targets,
    bindings: Option<(BindingKey, Vec<wgpu::BindGroup>)>,
}

impl LensPass {
    pub fn new(device: &wgpu::Device, width: u32, height: u32) -> Self {
        let shader = helio_core::shader::module(
            device,
            "Screen-space lens",
            helio_core::include_wgsl!("../shaders/lens.wgsl"),
        );
        let stage = |entry_point: &'static str, label: &'static str, entries: &[wgpu::BindGroupLayoutEntry]| {
            let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some(label),
                entries,
            });
            let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some(label),
                bind_group_layouts: &[Some(&layout)],
                immediate_size: 0,
            });
            let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(label),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some(entry_point),
                compilation_options: Default::default(),
                cache: None,
            });
            Stage { entry: entry_point, label, layout, pipeline }
        };
        // The bindings of each stage are the globals its entry point uses
        // (lens.wgsl); one texture is never read and written in one stage.
        let stages = [
            stage("cs_resolve", "Lens resolve", &[
                storage(0, true), storage(2, true), texture(3, false), texture(4, false),
                storage(5, false), storage(6, false), storage(7, false), uniform(8),
            ]),
            stage("cs_coc", "Lens circles of confusion", &[
                storage(1, true), storage(2, true), texture(3, false), storage(6, false),
                storage_texture(9),
            ]),
            stage("cs_tiles", "Lens tile extents", &[texture(10, false), storage(11, false)]),
            stage("cs_dilate", "Lens tile dilation", &[
                storage(6, false), texture(10, false), storage(11, false), storage(12, false),
            ]),
            stage("cs_prefilter", "Lens half-resolution prefilter", &[
                texture(4, false), texture(10, false), storage_texture(13), storage_texture(14),
            ]),
            stage("cs_gather", "Lens gather", &[
                storage(6, false), storage(12, false), texture(15, false), texture(16, false),
                storage_texture(17), storage_texture(18),
            ]),
            stage("cs_composite", "Lens composite", &[
                texture(4, false), storage(6, false), texture(10, false), texture(19, true),
                texture(20, true), storage_texture(21), sampler(22),
            ]),
        ];
        debug_assert_eq!(stages[RESOLVE].entry, "cs_resolve");
        debug_assert_eq!(stages[COMPOSITE].entry, "cs_composite");
        let buffer = |label, size, usage| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size,
                usage,
                mapped_at_creation: false,
            })
        };
        Self {
            input_key: "fogged_hdr",
            stages,
            sampler: device.create_sampler(&wgpu::SamplerDescriptor {
                label: Some("Lens sampler"),
                mag_filter: wgpu::FilterMode::Linear,
                min_filter: wgpu::FilterMode::Linear,
                ..Default::default()
            }),
            args: buffer(
                "Lens indirect dispatch",
                ARGS_SIZE,
                wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::INDIRECT | wgpu::BufferUsages::COPY_SRC,
            ),
            // Zero: no history, the first autofocus snaps.
            state: buffer("Lens state", 16, wgpu::BufferUsages::STORAGE),
            resolved: buffer(
                "Lens resolved",
                std::mem::size_of::<ResolvedLens>() as u64,
                wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            ),
            frame: buffer("Lens frame", 16, wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST),
            // Zero rows: no enabled lens, no elements.
            fallback_camera: buffer(
                "Lens neutral camera",
                2 * std::mem::size_of::<helio_core::GpuCameraUniforms>() as u64,
                wgpu::BufferUsages::STORAGE,
            ),
            fallback_lenses: buffer(
                "Lens no rows",
                std::mem::size_of::<LensComponent>() as u64,
                wgpu::BufferUsages::STORAGE,
            ),
            fallback_elements: buffer(
                "Lens no elements",
                std::mem::size_of::<LensElementComponent>() as u64,
                wgpu::BufferUsages::STORAGE,
            ),
            fallback_depth: device
                .create_texture(&wgpu::TextureDescriptor {
                    label: Some("Lens neutral depth"),
                    size: wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: wgpu::TextureFormat::R32Float,
                    usage: wgpu::TextureUsages::TEXTURE_BINDING,
                    view_formats: &[],
                })
                .create_view(&Default::default()),
            targets: Targets::new(device, width, height),
            bindings: None,
        }
    }

    /// Select a resolved HDR texture, e.g. the TSR output. Configure before graph
    /// construction. Only the default fogged_hdr path falls back to pre_aa.
    pub fn with_color_input(mut self, key: &'static str) -> Self {
        self.input_key = key;
        self
    }

    pub fn output_view(&self) -> &wgpu::TextureView { &self.targets.output }

    /// The buffer holding this frame's [`ResolvedLens`].
    pub fn resolved_buffer(&self) -> &wgpu::Buffer { &self.resolved }

    fn resize(&mut self, device: &wgpu::Device, width: u32, height: u32) {
        if self.targets.size != (width.max(1), height.max(1)) {
            self.targets = Targets::new(device, width, height);
            self.bindings = None;
        }
    }

    fn clear(&self, cmds: &mut CommandRecorder<'_>) {
        let attachments = [Some(wgpu::RenderPassColorAttachment {
            view: &self.targets.output,
            resolve_target: None,
            depth_slice: None,
            ops: wgpu::Operations {
                load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                store: wgpu::StoreOp::Store,
            },
        })];
        // Fast clear: no stale image when the input vanishes.
        let _pass = cmds.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Lens clear"),
            color_attachments: &attachments,
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
    }

    fn record(
        &mut self,
        device: &wgpu::Device,
        cmds: &mut CommandRecorder<'_>,
        input: Option<&wgpu::TextureView>,
        scene: LensInputs<'_>,
    ) {
        let Some(input) = input else {
            self.clear(cmds);
            return;
        };
        self.resize(device, input.texture().width(), input.texture().height());
        let camera = scene.camera.unwrap_or(&self.fallback_camera);
        let depth = scene.depth.unwrap_or(&self.fallback_depth);
        let lenses = scene.lenses.unwrap_or(&self.fallback_lenses);
        let elements = scene.elements.unwrap_or(&self.fallback_elements);
        // Bind groups are cached per GPU handle; comparing handles, not
        // pointers, is safe across TSR's rotating outputs and resize.
        let key: BindingKey = (
            [Some(input.clone()), Some(depth.clone())],
            [Some(lenses.clone()), Some(elements.clone()), Some(camera.clone())],
        );
        if self.bindings.as_ref().map(|(k, _)| k) != Some(&key) {
            let t = &self.targets;
            let groups = [
                (RESOLVE, vec![
                    entry(0, lenses.as_entire_binding()),
                    entry(2, camera.as_entire_binding()),
                    entry(3, wgpu::BindingResource::TextureView(depth)),
                    entry(4, wgpu::BindingResource::TextureView(input)),
                    entry(5, self.state.as_entire_binding()),
                    entry(6, self.resolved.as_entire_binding()),
                    entry(7, self.args.as_entire_binding()),
                    entry(8, self.frame.as_entire_binding()),
                ]),
                (COC, vec![
                    entry(1, elements.as_entire_binding()),
                    entry(2, camera.as_entire_binding()),
                    entry(3, wgpu::BindingResource::TextureView(depth)),
                    entry(6, self.resolved.as_entire_binding()),
                    entry(9, wgpu::BindingResource::TextureView(&t.coc)),
                ]),
                (TILES, vec![
                    entry(10, wgpu::BindingResource::TextureView(&t.coc)),
                    entry(11, t.tile_raw.as_entire_binding()),
                ]),
                (DILATE, vec![
                    entry(6, self.resolved.as_entire_binding()),
                    entry(10, wgpu::BindingResource::TextureView(&t.coc)),
                    entry(11, t.tile_raw.as_entire_binding()),
                    entry(12, t.tile_dilated.as_entire_binding()),
                ]),
                (PREFILTER, vec![
                    entry(4, wgpu::BindingResource::TextureView(input)),
                    entry(10, wgpu::BindingResource::TextureView(&t.coc)),
                    entry(13, wgpu::BindingResource::TextureView(&t.half_color)),
                    entry(14, wgpu::BindingResource::TextureView(&t.half_coc)),
                ]),
                (GATHER, vec![
                    entry(6, self.resolved.as_entire_binding()),
                    entry(12, t.tile_dilated.as_entire_binding()),
                    entry(15, wgpu::BindingResource::TextureView(&t.half_color)),
                    entry(16, wgpu::BindingResource::TextureView(&t.half_coc)),
                    entry(17, wgpu::BindingResource::TextureView(&t.far)),
                    entry(18, wgpu::BindingResource::TextureView(&t.near)),
                ]),
                (COMPOSITE, vec![
                    entry(4, wgpu::BindingResource::TextureView(input)),
                    entry(6, self.resolved.as_entire_binding()),
                    entry(10, wgpu::BindingResource::TextureView(&t.coc)),
                    entry(19, wgpu::BindingResource::TextureView(&t.far)),
                    entry(20, wgpu::BindingResource::TextureView(&t.near)),
                    entry(21, wgpu::BindingResource::TextureView(&t.output)),
                    entry(22, wgpu::BindingResource::Sampler(&self.sampler)),
                ]),
            ];
            let groups = groups
                .into_iter()
                .map(|(stage, entries)| {
                    device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some(self.stages[stage].label),
                        layout: &self.stages[stage].layout,
                        entries: &entries,
                    })
                })
                .collect();
            self.bindings = Some((key, groups));
        }
        let groups = &self.bindings.as_ref().unwrap().1;
        {
            let mut pass = cmds.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some(self.stages[RESOLVE].label),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.stages[RESOLVE].pipeline);
            pass.set_bind_group(0, &groups[RESOLVE], &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
        // Every later dispatch is GPU-sized: zero groups while the view has no
        // enabled lens, except the composite, which copies the image through.
        for stage in COC..STAGES {
            let mut pass = cmds.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some(self.stages[stage].label),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.stages[stage].pipeline);
            pass.set_bind_group(0, &groups[stage], &[]);
            pass.dispatch_workgroups_indirect(&self.args, 12 * (stage as u64 - 1));
        }
    }
}

impl RenderPass for LensPass {
    fn name(&self) -> &'static str { "Lens" }
    fn writes(&self) -> &'static [&'static str] { &[LENS_COLOR, RESOLVED_KEY] }
    fn declare_resources(&self, builder: &mut ResourceBuilder) {
        builder.read(self.input_key);
        if self.input_key == "fogged_hdr" { builder.read("pre_aa"); }
        // Manually owned resources: these declarations track the dependency,
        // while publish routes the actual views without a second texture.
        builder.write_buffer(LENS_COLOR);
        builder.write_buffer(RESOLVED_KEY);
    }
    fn prepare(&mut self, ctx: &PrepareContext) -> HelioResult<()> {
        let frame = [ctx.delta_time.max(0.0), 0.0, 0.0, 0.0];
        ctx.queue.write_buffer(&self.frame, 0, bytemuck::cast_slice(&frame));
        Ok(())
    }
    fn on_resize(&mut self, device: &wgpu::Device, width: u32, height: u32) {
        self.resize(device, width, height);
    }
    fn publish<'a>(&self, frame: &mut helio_core::ResourceRegistry<'a>) {
        frame.route_named_texture(LENS_COLOR, &self.targets.output, self.name());
        // SAFETY: the buffer is owned by this pass, which the graph keeps
        // alive for longer than the per-frame registry (as PostProcess does
        // for `postprocess_uniforms`).
        let resolved: &'a wgpu::Buffer = unsafe { std::mem::transmute(&self.resolved) };
        frame.write(ResourceKey::new(RESOLVED_KEY), resolved, self.name());
    }
    fn execute(&mut self, ctx: &mut PassContext) -> HelioResult<()> {
        let input = ctx.registry.get::<&wgpu::TextureView>(ResourceKey::new(self.input_key)).or_else(|| {
            (self.input_key == "fogged_hdr")
                .then(|| ctx.registry.get::<&wgpu::TextureView>(ResourceKey::new("pre_aa")))
                .flatten()
        });
        let scene = LensInputs {
            camera: Some(ctx.camera),
            depth: Some(ctx.depth),
            lenses: ctx
                .scene_buffers
                .get(helio_core::BufferKey::of("camera_lenses"))
                .map(|handle| &handle.buffer),
            elements: ctx
                .scene_buffers
                .get(helio_core::BufferKey::of("lens_elements"))
                .map(|handle| &handle.buffer),
        };
        // Must follow fog/TSR on the graphics encoder. The separate compute
        // encoder is submitted BEFORE graphics and would sample stale HDR.
        self.record(ctx.device, &mut ctx.graphics_cmds(), input, scene);
        Ok(())
    }
}

#[cfg(test)]
mod tests;
