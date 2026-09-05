//! Hierarchical Light-Field Sampling (HLFS) Pass
//!
//! Implements visibility-guided stochastic direct lighting with bounded
//! per-pixel sampling and shadow evaluation.
//!
//! Architecture:
//! 1. Conservative coarse-to-fine light culling
//! 2. Visible and hidden light reservoirs with visibility feedback
//! 3. Temporal filtering of diffuse and specular lighting
//! 4. Sparse spatial filtering and full-resolution composition
//!
//! CPU work is constant in scene size. GPU culling scales with light count;
//! per-pixel candidate and shadow work stays within the configured budget.

use bytemuck::{Pod, Zeroable};
use helio_core::graph::{ResourceBuilder, ResourceSize};
use helio_core::{PassContext, PrepareContext, RenderPass, Result};
mod bindings;
mod pipelines;
mod resources;
use bindings::{ExternalBindings, Inputs, InternalBindings};
use pipelines::Pipelines;
use resources::{Fallbacks, Targets, COARSE_TILE_SIZE, TILE_SIZE};

/// Lighting output used by capture/benchmark tools. Reference deliberately
/// evaluates every light and disables denoising; never use it for gameplay.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(u32)]
pub enum HlfsDebugMode {
    #[default]
    Final = 0,
    Reference = 1,
    Unfiltered = 2,
    Confidence = 3,
}

/// Bounded quality controls. Changes invalidate all history.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HlfsConfig {
    /// Shadowed light samples per shading pixel, clamped to 1..=4.
    pub samples_per_pixel: u32,
    /// Steady-state candidates per sample, clamped to 1..=16. Disocclusion
    /// doubles discovery candidates up to the hard limit of 16.
    pub candidates_per_sample: u32,
    /// 1 = full resolution, 2 = quarter as many shading pixels.
    pub sample_scale: u32,
    pub max_history_frames: u32,
    /// Fraction of sampling weight budget reserved for discovering hidden lights.
    pub discovery_fraction: f32,
    /// Pre-exposure of intermediate lighting; undone during composition.
    pub pre_exposure: f32,
    /// Maximum contact-shadow trace distance in world units; 0 disables it.
    pub screen_trace_distance: f32,
    pub debug_mode: HlfsDebugMode,
}
impl Default for HlfsConfig {
    fn default() -> Self {
        Self {
            samples_per_pixel: 2,
            candidates_per_sample: 8,
            sample_scale: 1,
            max_history_frames: 16,
            discovery_fraction: 0.2,
            pre_exposure: 1.0,
            screen_trace_distance: 0.5,
            debug_mode: HlfsDebugMode::Final,
        }
    }
}
impl HlfsConfig {
    /// Four samples over each 2x2 shading block: one sample per output pixel.
    pub fn performance() -> Self {
        Self {
            samples_per_pixel: 4,
            sample_scale: 2,
            ..Self::default()
        }
    }
    fn normalized(mut self) -> Self {
        self.samples_per_pixel = self.samples_per_pixel.clamp(1, 4);
        self.candidates_per_sample = self.candidates_per_sample.clamp(1, 16);
        self.sample_scale = self.sample_scale.clamp(1, 2);
        self.max_history_frames = self.max_history_frames.clamp(1, 32);
        self.discovery_fraction = if self.discovery_fraction.is_finite() {
            self.discovery_fraction.clamp(0.05, 1.0)
        } else {
            0.2
        };
        self.pre_exposure = if self.pre_exposure.is_finite() {
            self.pre_exposure.clamp(1e-6, 1e6)
        } else {
            1.0
        };
        self.screen_trace_distance = if self.screen_trace_distance.is_finite() {
            self.screen_trace_distance.clamp(0.0, 10.0)
        } else {
            0.5
        };
        if self.debug_mode == HlfsDebugMode::Reference {
            self.sample_scale = 1;
        }
        self
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Globals {
    frame: u32,
    sample_count: u32,
    light_count: u32,
    history_valid: u32,
    screen_size: [u32; 2],
    sample_size: [u32; 2],
    sample_scale: u32,
    candidate_count: u32,
    has_velocity: u32,
    surface_flags: u32,
    max_history: f32,
    discovery_fraction: f32,
    exposure: f32,
    debug_mode: u32,
    ambient: [f32; 4],
    csm_splits: [f32; 4],
    previous_view: [f32; 16],
}

pub struct HlfsPass {
    pipelines: Pipelines,
    targets: Targets,
    fallbacks: Fallbacks,
    internal: InternalBindings,
    external: ExternalBindings,
    globals: wgpu::Buffer,
    shadows: wgpu::Buffer,
    config: HlfsConfig,
    shadow_quality: libhelio::ShadowQuality,
    output_format: wgpu::TextureFormat,
    write_history: usize,
    history_valid: bool,
    previous_camera: Option<libhelio::GpuCameraUniforms>,
    previous_light_count: Option<u32>,
    previous_light_generation: Option<u64>,
    previous_frame: Option<u64>,
    timing_query: Option<wgpu::QuerySet>,
}
impl HlfsPass {
    /// Compact linear HDR when renderable on this device, with a portable fallback.
    pub fn preferred_output_format(device: &wgpu::Device) -> wgpu::TextureFormat {
        if device
            .features()
            .contains(wgpu::Features::RG11B10UFLOAT_RENDERABLE)
        {
            wgpu::TextureFormat::Rg11b10Ufloat
        } else {
            wgpu::TextureFormat::Rgba16Float
        }
    }

    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        width: u32,
        height: u32,
        output_format: wgpu::TextureFormat,
    ) -> Self {
        Self::with_config(
            device,
            queue,
            width,
            height,
            output_format,
            HlfsConfig::default(),
        )
    }
    pub fn with_config(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        width: u32,
        height: u32,
        output_format: wgpu::TextureFormat,
        config: HlfsConfig,
    ) -> Self {
        let config = config.normalized();
        let pipelines = Pipelines::new(device, output_format);
        let targets = Targets::new(device, width, height, output_format, config, None);
        let internal = InternalBindings::new(device, &pipelines, &targets);
        let uniform = |label, size| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        };
        let globals = uniform("HLFS globals", std::mem::size_of::<Globals>() as u64);
        let shadows = uniform(
            "HLFS shadow settings",
            std::mem::size_of::<libhelio::ShadowConfig>() as u64,
        );
        let shadow_quality = libhelio::ShadowQuality::High;
        queue.write_buffer(
            &shadows,
            0,
            bytemuck::bytes_of(&libhelio::ShadowConfig::from_quality(shadow_quality)),
        );
        Self {
            pipelines,
            targets,
            fallbacks: Fallbacks::new(device, queue),
            internal,
            external: ExternalBindings::default(),
            globals,
            shadows,
            config,
            shadow_quality,
            output_format,
            write_history: 0,
            history_valid: false,
            previous_camera: None,
            previous_light_count: None,
            previous_light_generation: None,
            previous_frame: None,
            timing_query: None,
        }
    }
    pub fn config(&self) -> HlfsConfig {
        self.config
    }
    pub fn set_config(&mut self, device: &wgpu::Device, config: HlfsConfig) {
        let config = config.normalized();
        if config == self.config {
            return;
        }
        let resize = config.sample_scale != self.config.sample_scale;
        self.config = config;
        if resize {
            self.recreate_targets(device, self.targets.width, self.targets.height);
        }
        self.invalidate_history();
    }
    pub fn set_shadow_quality(&mut self, quality: libhelio::ShadowQuality, queue: &wgpu::Queue) {
        self.shadow_quality = quality;
        queue.write_buffer(
            &self.shadows,
            0,
            bytemuck::bytes_of(&libhelio::ShadowConfig::from_quality(quality)),
        );
        self.invalidate_history();
    }
    pub fn invalidate_history(&mut self) {
        self.history_valid = false;
        self.previous_camera = None;
        self.previous_frame = None;
    }
    pub fn resize(&mut self, device: &wgpu::Device, width: u32, height: u32) {
        if (width.max(1), height.max(1)) == (self.targets.width, self.targets.height) {
            return;
        }
        self.recreate_targets(device, width, height);
    }
    fn recreate_targets(&mut self, device: &wgpu::Device, width: u32, height: u32) {
        // Internal shading changes retain the published output handle so
        // downstream passes can keep their existing bindings.
        let output = ((width.max(1), height.max(1)) == (self.targets.width, self.targets.height))
            .then(|| resources::Image {
                texture: self.targets.output.texture.clone(),
                view: self.targets.output.view.clone(),
            });
        self.targets = Targets::new(
            device,
            width,
            height,
            self.output_format,
            self.config,
            output,
        );
        self.internal = InternalBindings::new(device, &self.pipelines, &self.targets);
        self.external = ExternalBindings::default();
        self.write_history = 0;
        self.invalidate_history();
    }
    /// Stable capture source. Reacquire after resize; includes COPY_SRC usage.
    pub fn output_texture(&self) -> &wgpu::Texture {
        &self.targets.output.texture
    }
    /// Enable seven frame-boundary timestamps: coarse, fine, sampling, temporal,
    /// spatial, composite and completion. Resolve after the frame submission completes.
    pub fn enable_timing(&mut self, device: &wgpu::Device) -> bool {
        if !device.features().contains(
            wgpu::Features::TIMESTAMP_QUERY | wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS,
        ) {
            return false;
        }
        self.timing_query = Some(device.create_query_set(&wgpu::QuerySetDescriptor {
            label: Some("HLFS stage timings"),
            ty: wgpu::QueryType::Timestamp,
            count: 7,
        }));
        true
    }
    pub fn timing_query(&self) -> Option<&wgpu::QuerySet> {
        self.timing_query.as_ref()
    }
    /// Exact requested allocation bytes, excluding backend alignment/driver overhead.
    pub fn allocation_bytes(&self) -> u64 {
        let t = &self.targets;
        let pixels = u64::from(t.sample_width) * u64::from(t.sample_height);
        let output = u64::from(t.width)
            * u64::from(t.height)
            * u64::from(self.output_format.block_copy_size(None).unwrap_or(4));
        pixels * 40
            + (0..t.depth_bounds.texture.mip_level_count())
                .map(|mip| {
                    u64::from((t.depth_bounds.texture.width() >> mip).max(1))
                        * u64::from((t.depth_bounds.texture.height() >> mip).max(1))
                        * 4
                })
                .sum::<u64>()
            + output
            + t.coarse.size()
            + t.grid.size()
            + t.history.iter().map(|h| h.visible.size()).sum::<u64>()
            + self.globals.size()
            + self.shadows.size()
            + 32 * 32 * 32
            + 8
            + 16
            + 4
    }
    fn record(&mut self, encoder: &mut wgpu::CommandEncoder) {
        let p = &self.pipelines;
        let t = &self.targets;
        let parity = self.write_history;
        let common = self.external.common.as_ref().expect("HLFS inputs bound");
        let gbuffer = self.external.gbuffer.as_ref().expect("HLFS GBuffer bound");
        let timestamp = |encoder: &mut wgpu::CommandEncoder, index| {
            if let Some(query) = &self.timing_query {
                encoder.write_timestamp(query, index);
            }
        };
        let dispatch = |encoder: &mut wgpu::CommandEncoder,
                        name,
                        pipeline: &wgpu::ComputePipeline,
                        resources: &wgpu::BindGroup,
                        x,
                        y,
                        rt: Option<&wgpu::BindGroup>| {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some(name),
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, common, &[]);
            pass.set_bind_group(1, gbuffer, &[]);
            pass.set_bind_group(2, resources, &[]);
            if let Some(rt) = rt {
                pass.set_bind_group(3, rt, &[]);
            }
            pass.dispatch_workgroups(x, y, 1);
        };
        timestamp(encoder, 0);
        dispatch(
            encoder,
            "HLFS coarse light cull",
            &p.coarse,
            &self.internal.grid,
            t.width.div_ceil(COARSE_TILE_SIZE),
            t.height.div_ceil(COARSE_TILE_SIZE),
            None,
        );
        timestamp(encoder, 1);
        dispatch(
            encoder,
            "HLFS fine light cull",
            &p.fine,
            &self.internal.grid,
            t.width.div_ceil(TILE_SIZE),
            t.height.div_ceil(TILE_SIZE),
            None,
        );
        for (index, resources) in self.internal.depth_reduce.iter().enumerate() {
            let mip = index as u32 + 1;
            dispatch(
                encoder,
                "HLFS depth pyramid",
                &p.depth_reduce,
                resources,
                (t.depth_bounds.texture.width() >> mip).max(1).div_ceil(8),
                (t.depth_bounds.texture.height() >> mip).max(1).div_ceil(8),
                None,
            );
        }
        let small = self
            .previous_light_count
            .is_some_and(|count| count <= self.config.samples_per_pixel);
        let sample = if small { &p.sample_small } else { &p.sample };
        let ray = (if small {
            &p.sample_small_rt
        } else {
            &p.sample_rt
        })
        .as_ref()
        .zip(self.external.rt.as_ref());
        timestamp(encoder, 2);
        dispatch(
            encoder,
            "HLFS sample and visibility",
            ray.map_or(sample, |(pipe, _)| pipe),
            &self.internal.sample[parity],
            t.sample_width.div_ceil(8),
            t.sample_height.div_ceil(8),
            ray.map(|(_, bg)| bg),
        );
        timestamp(encoder, 3);
        dispatch(
            encoder,
            "HLFS temporal filter",
            &p.temporal,
            &self.internal.temporal[parity],
            t.sample_width.div_ceil(8),
            t.sample_height.div_ceil(8),
            None,
        );
        timestamp(encoder, 4);
        dispatch(
            encoder,
            "HLFS spatial filter",
            &p.spatial,
            &self.internal.spatial[parity],
            t.sample_width.div_ceil(8),
            t.sample_height.div_ceil(8),
            None,
        );
        timestamp(encoder, 5);
        {
            let attachments = [Some(wgpu::RenderPassColorAttachment {
                view: &t.output.view,
                resolve_target: None,
                depth_slice: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
            })];
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("HLFS spatial filter and composite"),
                color_attachments: &attachments,
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&p.composite);
            pass.set_bind_group(0, common, &[]);
            pass.set_bind_group(1, gbuffer, &[]);
            pass.set_bind_group(2, &self.internal.composite[parity], &[]);
            pass.draw(0..3, 0..1);
        }
        timestamp(encoder, 6);
        self.write_history = 1 - parity;
        self.history_valid = true;
    }
}
impl RenderPass for HlfsPass {
    fn on_resize(&mut self, device: &wgpu::Device, width: u32, height: u32) {
        self.resize(device, width, height);
    }
    fn name(&self) -> &'static str {
        "HLFS"
    }
    fn reads(&self) -> &'static [&'static str] {
        &["gbuffer", "pre_aa"]
    }
    fn writes(&self) -> &'static [&'static str] {
        &["pre_aa"]
    }
    fn publish<'a>(&'a self, frame: &mut libhelio::FrameResources<'a>) {
        frame.pre_aa.write(&self.targets.output.view, "HLFS");
    }
    fn render_pass_descriptor<'a>(
        &'a self,
        _target: &'a wgpu::TextureView,
        _depth: &'a wgpu::TextureView,
        _resources: &'a libhelio::FrameResources<'a>,
    ) -> Option<wgpu::RenderPassDescriptor<'a>> {
        None
    }
    fn declare_resources(&self, builder: &mut ResourceBuilder) {
        builder.write_color_raw("pre_aa", self.output_format, ResourceSize::MatchSurface);
    }
    fn prepare(&mut self, ctx: &PrepareContext) -> Result<()> {
        let camera = *ctx.scene.camera.data();
        let light_count = ctx.scene.movable_light_count;
        let continuity = self
            .previous_frame
            .is_some_and(|f| f.wrapping_add(1) == ctx.frame_num);
        let camera_continuity = self.previous_camera.as_ref().is_some_and(|old| {
            old.view_proj
                .iter()
                .zip(camera.prev_view_proj)
                .all(|(a, b)| (a - b).abs() < 1e-3)
        });
        // Guided IDs are reweighted against current light data every frame.
        // Smooth light animation retains history; photometric clamping handles
        // changed illumination and a changed population invalidates all IDs.
        self.history_valid &= continuity
            && camera_continuity
            && self.previous_light_count == Some(light_count)
            && !ctx.resize;
        let mut ambient = [0.03, 0.03, 0.03, self.config.screen_trace_distance];
        if let Some(scene) = ctx.frame_resources.main_scene.get() {
            for (i, v) in ambient[..3].iter_mut().enumerate() {
                *v = scene.ambient_color[i] * scene.ambient_intensity;
            }
        }
        let g = Globals {
            frame: ctx.frame_num as u32,
            sample_count: self.config.samples_per_pixel,
            light_count,
            history_valid: self.history_valid as u32,
            screen_size: [self.targets.width, self.targets.height],
            sample_size: [self.targets.sample_width, self.targets.sample_height],
            sample_scale: self.config.sample_scale,
            candidate_count: self.config.candidates_per_sample,
            has_velocity: ctx.frame_resources.gbuffer_velocity.get().is_some() as u32,
            surface_flags: (ctx.frame_resources.baked_lightmap.get().is_some()
                && ctx.frame_resources.gbuffer_lightmap_uv.get().is_some())
                as u32
                | (u32::from(
                    self.previous_light_generation == Some(ctx.scene.movable_lights_generation),
                ) << 1),
            max_history: self.config.max_history_frames as f32,
            discovery_fraction: self.config.discovery_fraction,
            exposure: self.config.pre_exposure,
            debug_mode: self.config.debug_mode as u32,
            ambient,
            csm_splits: libhelio::CSM_SPLITS,
            previous_view: self.previous_camera.map_or(camera.view, |c| c.view),
        };
        ctx.write_buffer(&self.globals, 0, bytemuck::bytes_of(&g));
        self.previous_camera = Some(camera);
        self.previous_frame = Some(ctx.frame_num);
        self.previous_light_count = Some(light_count);
        self.previous_light_generation = Some(ctx.scene.movable_lights_generation);
        Ok(())
    }
    fn execute(&mut self, ctx: &mut PassContext) -> Result<()> {
        let gbuffer = ctx.resources.gbuffer.read("HLFS").ok_or_else(|| {
            helio_core::Error::InvalidPassConfig("HLFS requires a GBuffer".into())
        })?;
        let pre_aa =
            ctx.resources.pre_aa.get().ok_or_else(|| {
                helio_core::Error::InvalidPassConfig("HLFS requires pre_aa".into())
            })?;
        let f = &self.fallbacks;
        let inputs = Inputs {
            camera: ctx.scene.camera,
            lights: ctx.scene.lights,
            shadow_matrices: ctx.scene.shadow_matrices,
            shadow_atlas: ctx.resources.shadow_atlas.get().unwrap_or(&f.shadow_view),
            shadow_sampler: ctx
                .resources
                .shadow_sampler
                .get()
                .unwrap_or(&f.shadow_sampler),
            textures: [
                gbuffer.albedo,
                gbuffer.normal,
                gbuffer.orm,
                gbuffer.emissive,
                ctx.depth,
                ctx.resources
                    .gbuffer_lightmap_uv
                    .get()
                    .unwrap_or(&f.lightmap_uv.view),
                ctx.resources.baked_lightmap.get().unwrap_or(&f.black.view),
                pre_aa,
                ctx.resources
                    .gbuffer_velocity
                    .get()
                    .unwrap_or(&f.black.view),
            ],
            lightmap_sampler: ctx
                .resources
                .baked_lightmap_sampler
                .get()
                .unwrap_or(&f.linear_sampler),
            tlas: ctx.resources.main_scene.get().and_then(|s| s.tlas),
        };
        self.external.update(
            ctx.device,
            &self.pipelines,
            &self.fallbacks,
            &self.globals,
            &self.shadows,
            &inputs,
        );
        // These dispatches read the GBuffer just rendered, so they belong on the
        // render encoder rather than the early compute encoder.
        // Returning no render descriptor guarantees there is no active render pass.
        self.record(unsafe { &mut *ctx.encoder_ptr });
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn all_wgsl_stages_validate() {
        for stage in [
            "grid",
            "depth",
            "sample",
            "temporal",
            "spatial",
            "composite",
            "sample_rt",
        ] {
            let source = super::pipelines::shader_source(stage);
            let module = naga::front::wgsl::parse_str(&source)
                .unwrap_or_else(|e| panic!("{stage}: {}", e.emit_to_string(&source)));
            naga::valid::Validator::new(
                naga::valid::ValidationFlags::all(),
                naga::valid::Capabilities::all(),
            )
            .validate(&module)
            .unwrap_or_else(|e| panic!("{stage}: {e:?}"));
        }
    }
}
