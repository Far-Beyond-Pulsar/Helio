//! Physically based atmosphere (Hillaire 2020, "A Scalable and Production
//! Ready Sky and Atmosphere Rendering Technique"), for any scene: a flat
//! world (the ground at the world origin) or a planet (its centre anywhere).
//!
//! An [`AtmosphereComponent`] row describes the air: planet and atmosphere
//! radii, Rayleigh and Mie scattering, Mie and ozone absorption, ground
//! albedo. The scene's first directional light is its sun. Without a row
//! there is no atmosphere (space, a moon).
//!
//! [`AtmospherePass`] resolves the row and the sun on the GPU each frame and
//! builds the lookup tables: transmittance, multiple scattering, the sky
//! seen from the camera, the aerial-perspective volume (light scattered
//! between the camera and each pixel) and the sky's diffuse irradiance as
//! spherical harmonics. It publishes:
//!
//! - `atmosphere_frame` (buffer, `AtmosphereFrame` in
//!   `shaders/atmosphere_common.wgsl`): planet, sun, irradiance. Deferred
//!   lighting attenuates the sun through the air at each pixel and lights
//!   ambient with the sky's irradiance.
//! - `atmosphere_transmittance`, `atmosphere_sky_view`,
//!   `atmosphere_aerial` (textures).
//!
//! [`AtmosphereCompositePass`] runs after opaque lighting: the sky and the
//! sun's disk where no geometry is, aerial perspective where it is.
//!
//! Positions stay exact in camera-relative frames: the planet centre is
//! rebased by the frame's world origin, split into an f32 and its
//! remainder.

use bytemuck::{Pod, Zeroable};
use helio_core::graph::ResourceBuilder;
use helio_core::{PassContext, PrepareContext, RenderPass, ResourceKey, Result as HelioResult};
use pulsar_scenedb::gpu::BufferKey;
use pulsar_scenedb_derive::SceneStore;

/// Graph key of the per-frame atmosphere buffer (`AtmosphereFrame`).
pub const ATMOSPHERE_FRAME: &str = "atmosphere_frame";
pub const ATMOSPHERE_TRANSMITTANCE: &str = "atmosphere_transmittance";
pub const ATMOSPHERE_MULTI_SCATTERING: &str = "atmosphere_multi_scattering";
pub const ATMOSPHERE_SKY_VIEW: &str = "atmosphere_sky_view";
pub const ATMOSPHERE_AERIAL: &str = "atmosphere_aerial";
/// Bytes of `AtmosphereFrame`. The buffer is usable as a uniform as well as
/// storage: consumers already at the per-stage storage-buffer limit (the
/// deferred lighting pass) bind it as a uniform.
pub const ATMOSPHERE_FRAME_BYTES: u64 = 320;

/// Marker opting a shader into the shared atmosphere WGSL.
pub const ATMOSPHERE_MARKER: &str = "//!use atmosphere";

/// The atmosphere's shared WGSL (`AtmosphereFrame`, the medium, the LUT
/// parameterisations) as a snippet: any pass reading the published
/// `atmosphere_frame` opts in with [`ATMOSPHERE_MARKER`].
pub const ATMOSPHERE_SNIPPET: helio_core::shader::ShaderSnippet =
    helio_core::wgsl_snippet!(ATMOSPHERE_MARKER, "shaders/atmosphere_common.wgsl");

/// Where an atmosphere's planet is.
pub mod placement {
    /// The ground is at the world origin; the planet's centre is straight
    /// below it. For flat worlds.
    pub const GROUND_AT_ORIGIN: u32 = 0;
    /// The planet's centre is at `AtmosphereComponent::center` (world
    /// metres). For planets.
    pub const CENTER: u32 = 1;
}

/// The air around a planet (or above a flat world). Coefficients are per
/// kilometre, heights and radii in kilometres. The first enabled row is
/// used.
#[derive(SceneStore, Pod, Zeroable, Clone, Copy, Debug, PartialEq)]
#[repr(C)]
#[gpu(layout = packed, buffer = "atmospheres")]
pub struct AtmosphereComponent {
    /// Planet centre in world metres (with [`placement::CENTER`]).
    #[gpu]
    pub center: [f32; 3],
    /// [`placement`].
    #[gpu]
    pub placement: u32,
    #[gpu]
    pub rayleigh_scattering: [f32; 3],
    #[gpu]
    pub rayleigh_scale_height: f32,
    #[gpu]
    pub mie_scattering: [f32; 3],
    #[gpu]
    pub mie_scale_height: f32,
    #[gpu]
    pub mie_absorption: [f32; 3],
    /// Mie phase asymmetry (forward scattering, the haze around the sun).
    #[gpu]
    pub mie_g: f32,
    #[gpu]
    pub ozone_absorption: [f32; 3],
    /// Height of the ozone layer's peak.
    #[gpu]
    pub ozone_center: f32,
    /// Diffuse reflectance of the ground, for light it bounces into the air.
    #[gpu]
    pub ground_albedo: [f32; 3],
    /// Thickness of the ozone layer (a tent around its peak).
    #[gpu]
    pub ozone_width: f32,
    /// Planet radius.
    #[gpu]
    pub bottom_radius: f32,
    /// Radius of the top of the atmosphere.
    #[gpu]
    pub top_radius: f32,
    /// Angular radius of the sun's disk (radians).
    #[gpu]
    pub sun_angular_radius: f32,
    /// Non-zero to use this atmosphere.
    #[gpu]
    pub enabled: u32,
}

const _: () = assert!(std::mem::size_of::<AtmosphereComponent>() == 112);

impl Default for AtmosphereComponent {
    fn default() -> Self {
        Self::earth()
    }
}

impl AtmosphereComponent {
    /// Earth's atmosphere (Hillaire 2020's reference values), the ground at
    /// the world origin.
    pub fn earth() -> Self {
        Self {
            center: [0.0; 3],
            placement: placement::GROUND_AT_ORIGIN,
            rayleigh_scattering: [5.802e-3, 13.558e-3, 33.1e-3],
            rayleigh_scale_height: 8.0,
            mie_scattering: [3.996e-3; 3],
            mie_scale_height: 1.2,
            mie_absorption: [4.4e-3; 3],
            mie_g: 0.8,
            ozone_absorption: [0.65e-3, 1.881e-3, 0.085e-3],
            ozone_center: 25.0,
            ozone_width: 30.0,
            ground_albedo: [0.3; 3],
            bottom_radius: 6360.0,
            top_radius: 6460.0,
            sun_angular_radius: 0.004675,
            enabled: 1,
        }
    }

    /// A thin, dusty Martian atmosphere: butterscotch by day, blue around
    /// the setting sun.
    pub fn mars() -> Self {
        Self {
            rayleigh_scattering: [0.19e-3, 1.0e-3, 2.0e-3],
            rayleigh_scale_height: 11.1,
            mie_scattering: [11.0e-3, 8.0e-3, 5.0e-3],
            mie_scale_height: 11.1,
            mie_absorption: [4.0e-3, 6.5e-3, 9.0e-3],
            mie_g: 0.75,
            ozone_absorption: [0.0; 3],
            ground_albedo: [0.35, 0.22, 0.12],
            bottom_radius: 3389.5,
            top_radius: 3489.5,
            sun_angular_radius: 0.0031,
            ..Self::earth()
        }
    }

    /// This atmosphere around a planet of `radius_m` centred at `center_m`
    /// (world metres); the atmosphere keeps its thickness.
    pub fn around_planet(mut self, center_m: [f32; 3], radius_m: f64) -> Self {
        let thickness = self.top_radius - self.bottom_radius;
        self.placement = placement::CENTER;
        self.center = center_m;
        self.bottom_radius = (radius_m * 0.001) as f32;
        self.top_radius = self.bottom_radius + thickness;
        self
    }
}

const TRANSMITTANCE: [u32; 2] = [256, 64];
const MULTI_SCATTERING: u32 = 32;
const SKY_VIEW: [u32; 2] = [192, 108];
const AERIAL: u32 = 32;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ResolveUniform {
    origin_hi: [f32; 4],
    origin_lo: [f32; 4],
}

fn lut(device: &wgpu::Device, label: &str, size: [u32; 3], dimension: wgpu::TextureDimension) -> wgpu::TextureView {
    device
        .create_texture(&wgpu::TextureDescriptor {
            label: Some(label),
            size: wgpu::Extent3d { width: size[0], height: size[1], depth_or_array_layers: size[2] },
            mip_level_count: 1,
            sample_count: 1,
            dimension,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::STORAGE_BINDING,
            view_formats: &[],
        })
        .create_view(&Default::default())
}

fn storage_texture(binding: u32, dimension: wgpu::TextureViewDimension) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::StorageTexture {
            access: wgpu::StorageTextureAccess::WriteOnly,
            format: wgpu::TextureFormat::Rgba16Float,
            view_dimension: dimension,
        },
        count: None,
    }
}

fn sampled_texture(binding: u32, visibility: wgpu::ShaderStages, dimension: wgpu::TextureViewDimension) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility,
        ty: wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable: true },
            view_dimension: dimension,
            multisampled: false,
        },
        count: None,
    }
}

fn buffer_entry(binding: u32, visibility: wgpu::ShaderStages, ty: wgpu::BufferBindingType) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility,
        ty: wgpu::BindingType::Buffer { ty, has_dynamic_offset: false, min_binding_size: None },
        count: None,
    }
}

struct Kernel {
    pipeline: wgpu::ComputePipeline,
    group: Option<wgpu::BindGroup>,
    groups: [u32; 3],
}

/// Resolves the scene's atmosphere and builds its lookup tables (see the
/// module docs).
pub struct AtmospherePass {
    shared_layout: wgpu::BindGroupLayout,
    shared_group: Option<wgpu::BindGroup>,
    shared_key: Option<(wgpu::Buffer, wgpu::Buffer, wgpu::Buffer)>,
    resolve: wgpu::ComputePipeline,
    kernels: Vec<Kernel>,
    frame: wgpu::Buffer,
    uniform: wgpu::Buffer,
    no_rows: wgpu::Buffer,
    sampler: wgpu::Sampler,
    transmittance: wgpu::TextureView,
    multi_scattering: wgpu::TextureView,
    sky_view: wgpu::TextureView,
    aerial: wgpu::TextureView,
    rows: helio_core::SceneBufferLiveness,
    active: bool,
    /// `HELIO_ATMOSPHERE_TRACE=1`: the resolved frame's head (planet, sun,
    /// illuminance, eye) read back and printed each frame (diagnostics).
    trace: Option<Vec<(u64, wgpu::Buffer)>>,
    frame_index: u64,
}

impl AtmospherePass {
    pub fn new(device: &wgpu::Device) -> Self {
        let module = helio_core::shader::module_with(
            device,
            "Atmosphere LUTs",
            helio_core::include_wgsl!("shaders/atmosphere_luts.wgsl"),
            &[ATMOSPHERE_SNIPPET],
        );
        let compute = wgpu::ShaderStages::COMPUTE;
        let storage = |read_only| wgpu::BufferBindingType::Storage { read_only };
        let shared_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Atmosphere shared"),
            entries: &[
                buffer_entry(0, compute, wgpu::BufferBindingType::Uniform),
                buffer_entry(1, compute, storage(true)),
                buffer_entry(2, compute, storage(true)),
                buffer_entry(3, compute, storage(true)),
                buffer_entry(4, compute, storage(false)),
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: compute,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });
        let d2 = wgpu::TextureViewDimension::D2;
        let transmittance = lut(device, "Atmosphere transmittance", [TRANSMITTANCE[0], TRANSMITTANCE[1], 1], wgpu::TextureDimension::D2);
        let multi = lut(device, "Atmosphere multiple scattering", [MULTI_SCATTERING, MULTI_SCATTERING, 1], wgpu::TextureDimension::D2);
        let multi_scattering = multi.clone();
        let sky_view = lut(device, "Atmosphere sky view", [SKY_VIEW[0], SKY_VIEW[1], 1], wgpu::TextureDimension::D2);
        let aerial = lut(device, "Atmosphere aerial perspective", [AERIAL; 3], wgpu::TextureDimension::D3);
        let pipeline = |layout: Option<&wgpu::BindGroupLayout>, entry: &str| {
            let groups: Vec<Option<&wgpu::BindGroupLayout>> = match layout {
                Some(layout) => vec![Some(&shared_layout), Some(layout)],
                None => vec![Some(&shared_layout)],
            };
            let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some(entry),
                bind_group_layouts: &groups,
                immediate_size: 0,
            });
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: Some(&pipeline_layout),
                module: &module,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        // Each kernel binds only its own outputs and inputs: a texture may
        // not be written and sampled in one dispatch.
        let kernel = |entry: &str, entries: &[(wgpu::BindGroupLayoutEntry, wgpu::BindingResource)], groups: [u32; 3]| {
            let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some(entry),
                entries: &entries.iter().map(|(e, _)| *e).collect::<Vec<_>>(),
            });
            let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(entry),
                layout: &layout,
                entries: &entries
                    .iter()
                    .map(|(e, r)| wgpu::BindGroupEntry { binding: e.binding, resource: r.clone() })
                    .collect::<Vec<_>>(),
            });
            Kernel { pipeline: pipeline(Some(&layout), entry), group: Some(group), groups }
        };
        let view = wgpu::BindingResource::TextureView;
        let kernels = vec![
            kernel(
                "transmittance_lut_kernel",
                &[(storage_texture(0, d2), view(&transmittance))],
                [TRANSMITTANCE[0].div_ceil(8), TRANSMITTANCE[1].div_ceil(8), 1],
            ),
            kernel(
                "multi_scattering_kernel",
                &[(sampled_texture(1, compute, d2), view(&transmittance)), (storage_texture(2, d2), view(&multi))],
                [MULTI_SCATTERING.div_ceil(8), MULTI_SCATTERING.div_ceil(8), 1],
            ),
            kernel(
                "sky_view_kernel",
                &[
                    (sampled_texture(1, compute, d2), view(&transmittance)),
                    (sampled_texture(3, compute, d2), view(&multi)),
                    (storage_texture(4, d2), view(&sky_view)),
                ],
                [SKY_VIEW[0].div_ceil(8), SKY_VIEW[1].div_ceil(8), 1],
            ),
            kernel(
                "aerial_kernel",
                &[
                    (sampled_texture(1, compute, d2), view(&transmittance)),
                    (sampled_texture(3, compute, d2), view(&multi)),
                    (storage_texture(6, wgpu::TextureViewDimension::D3), view(&aerial)),
                ],
                [AERIAL.div_ceil(8), AERIAL.div_ceil(8), 1],
            ),
            kernel("irradiance_kernel", &[(sampled_texture(5, compute, d2), view(&sky_view))], [1, 1, 1]),
        ];
        let resolve = pipeline(None, "resolve");
        let frame = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Atmosphere frame"),
            size: ATMOSPHERE_FRAME_BYTES,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Atmosphere resolve"),
            size: std::mem::size_of::<ResolveUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        // Bound while a scene has no atmosphere or light rows yet.
        let no_rows = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Atmosphere no rows"),
            size: 128,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Atmosphere LUT"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            ..Default::default()
        });
        Self {
            shared_layout,
            shared_group: None,
            shared_key: None,
            resolve,
            kernels,
            frame,
            uniform,
            no_rows,
            sampler,
            transmittance,
            multi_scattering,
            sky_view,
            aerial,
            rows: helio_core::SceneBufferLiveness::with_row_predicate(|row| {
                // `enabled` is the last word.
                row.len() >= 112 && row[108..112].iter().any(|&b| b != 0)
            }),
            active: false,
            trace: std::env::var_os("HELIO_ATMOSPHERE_TRACE").map(|_| Vec::new()),
            frame_index: 0,
        }
    }

    /// The pass that composites this atmosphere over the lit scene, in
    /// `format` (the lighting target's).
    pub fn composite(device: &wgpu::Device, format: wgpu::TextureFormat) -> AtmosphereCompositePass {
        AtmosphereCompositePass::new(device, format)
    }
}

impl RenderPass for AtmospherePass {
    fn name(&self) -> &'static str {
        "Atmosphere"
    }

    fn writes(&self) -> &'static [&'static str] {
        &[ATMOSPHERE_FRAME, ATMOSPHERE_TRANSMITTANCE, ATMOSPHERE_MULTI_SCATTERING, ATMOSPHERE_SKY_VIEW, ATMOSPHERE_AERIAL]
    }

    fn declare_resources(&self, _builder: &mut ResourceBuilder) {}

    fn render_pass_descriptor<'a>(
        &'a self,
        _target: &'a wgpu::TextureView,
        _depth: &'a wgpu::TextureView,
        _resources: &'a helio_core::ResourceRegistry<'a>,
    ) -> Option<wgpu::RenderPassDescriptor<'a>> {
        None
    }

    fn prepare(&mut self, ctx: &PrepareContext) -> HelioResult<()> {
        let rows = ctx.scene_buffers.get(BufferKey::of("atmospheres"));
        self.rows.update(ctx.device, ctx.queue, rows);
        let active = rows.is_some_and(|handle| self.rows.maybe_live(handle));
        if !active && self.active {
            // Consumers read "no atmosphere" from the frame.
            ctx.write_buffer(&self.frame, 0, &[0u8; 16]);
        }
        self.active = active;
        let origin = ctx.world_origin.unwrap_or_default();
        let hi = origin.as_vec3();
        let lo = (origin - hi.as_dvec3()).as_vec3();
        let uniform = ResolveUniform { origin_hi: hi.extend(0.0).to_array(), origin_lo: lo.extend(0.0).to_array() };
        ctx.write_buffer(&self.uniform, 0, bytemuck::bytes_of(&uniform));
        Ok(())
    }

    fn execute(&mut self, ctx: &mut PassContext) -> HelioResult<()> {
        if !self.active {
            return Ok(());
        }
        let atmospheres = ctx.scene_buffers.get(BufferKey::of("atmospheres")).map_or(&self.no_rows, |h| &h.buffer);
        let lights = ctx.scene_buffers.get(BufferKey::of("scene_lights")).map_or(&self.no_rows, |h| &h.buffer);
        let key = (atmospheres.clone(), lights.clone(), ctx.camera.clone());
        if self.shared_key.as_ref() != Some(&key) {
            self.shared_group = Some(ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Atmosphere shared"),
                layout: &self.shared_layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: self.uniform.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: atmospheres.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: lights.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 3, resource: ctx.camera.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 4, resource: self.frame.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 5, resource: wgpu::BindingResource::Sampler(&self.sampler) },
                ],
            }));
            self.shared_key = Some(key);
        }
        let mut cmds = ctx.graphics_cmds();
        let mut pass = cmds.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("Atmosphere"), timestamp_writes: None });
        pass.set_bind_group(0, self.shared_group.as_ref(), &[]);
        pass.set_pipeline(&self.resolve);
        pass.dispatch_workgroups(1, 1, 1);
        for kernel in &self.kernels {
            pass.set_pipeline(&kernel.pipeline);
            pass.set_bind_group(1, kernel.group.as_ref(), &[]);
            pass.dispatch_workgroups(kernel.groups[0], kernel.groups[1], kernel.groups[2]);
        }
        drop(pass);
        self.frame_index += 1;
        if let Some(trace) = &mut self.trace {
            // Map last frame's copy (submitted by now), copy this frame's.
            for (index, buffer) in trace.drain(..) {
                let probe = buffer.clone();
                buffer.slice(..).map_async(wgpu::MapMode::Read, move |result| {
                    if result.is_err() { return; }
                    let bytes = probe.slice(..).get_mapped_range().unwrap().to_vec();
                    let f: &[f32] = bytemuck::cast_slice(&bytes);
                    eprintln!(
                        "ATMOSPHERE_TRACE frame {index} planet {:?} sun {:?} illuminance {:?} eye {:?} r {:.3}",
                        &f[0..4], &f[4..8], &f[8..12], &f[12..16],
                        (f[12] * f[12] + f[13] * f[13] + f[14] * f[14]).sqrt(),
                    );
                });
            }
            let staging = ctx.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Atmosphere trace"),
                size: 64,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });
            cmds.copy_buffer_to_buffer(&self.frame, 0, &staging, 0, 64);
            trace.push((self.frame_index, staging));
        }
        Ok(())
    }

    fn publish<'a>(&self, frame: &mut helio_core::ResourceRegistry<'a>) {
        // Pass-owned resources, handed over for the frame.
        let buffer: &'a wgpu::Buffer = unsafe { std::mem::transmute(&self.frame) };
        frame.write_buffer(ResourceKey::new(ATMOSPHERE_FRAME), buffer, "Atmosphere");
        for (key, view) in [
            (ATMOSPHERE_TRANSMITTANCE, &self.transmittance),
            (ATMOSPHERE_MULTI_SCATTERING, &self.multi_scattering),
            (ATMOSPHERE_SKY_VIEW, &self.sky_view),
            (ATMOSPHERE_AERIAL, &self.aerial),
        ] {
            let view: &'a wgpu::TextureView = unsafe { std::mem::transmute(view) };
            frame.write_texture_view(ResourceKey::new(key), view, "Atmosphere");
        }
        let active: bool = self.active;
        frame.write(ResourceKey::new("atmosphere_active"), active, "Atmosphere");
    }
}

/// Composites the atmosphere over the lit scene in place (`pre_aa`): the
/// sky and the sun where no geometry is, aerial perspective where it is.
/// Draws nothing without an atmosphere.
pub struct AtmosphereCompositePass {
    pipeline: wgpu::RenderPipeline,
    layout: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
    group: Option<wgpu::BindGroup>,
    key: Option<(wgpu::Buffer, wgpu::Buffer, [wgpu::TextureView; 5])>,
    active: bool,
}

impl AtmosphereCompositePass {
    pub fn new(device: &wgpu::Device, format: wgpu::TextureFormat) -> Self {
        let module = helio_core::shader::module_with(
            device,
            "Atmosphere composite",
            helio_core::include_wgsl!("shaders/atmosphere_composite.wgsl"),
            &[ATMOSPHERE_SNIPPET],
        );
        let fragment = wgpu::ShaderStages::FRAGMENT;
        let read_only = wgpu::BufferBindingType::Storage { read_only: true };
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Atmosphere composite"),
            entries: &[
                buffer_entry(0, fragment, read_only),
                buffer_entry(1, fragment, read_only),
                sampled_texture(2, fragment, wgpu::TextureViewDimension::D2),
                sampled_texture(3, fragment, wgpu::TextureViewDimension::D2),
                sampled_texture(4, fragment, wgpu::TextureViewDimension::D3),
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: fragment,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Depth,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
                    visibility: fragment,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                sampled_texture(7, fragment, wgpu::TextureViewDimension::D2),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Atmosphere composite"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        // dst * alpha + src: alpha 0 replaces (sky), alpha T attenuates.
        let blend = wgpu::BlendComponent {
            src_factor: wgpu::BlendFactor::One,
            dst_factor: wgpu::BlendFactor::SrcAlpha,
            operation: wgpu::BlendOperation::Add,
        };
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Atmosphere composite"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &module,
                entry_point: Some("vs_fullscreen"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &module,
                entry_point: Some("fs_composite"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: Some(wgpu::BlendState {
                        color: blend,
                        alpha: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::Zero,
                            dst_factor: wgpu::BlendFactor::One,
                            operation: wgpu::BlendOperation::Add,
                        },
                    }),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Atmosphere composite"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        Self { pipeline, layout, sampler, group: None, key: None, active: false }
    }
}

impl RenderPass for AtmosphereCompositePass {
    fn name(&self) -> &'static str {
        "AtmosphereComposite"
    }

    fn reads(&self) -> &'static [&'static str] {
        &["pre_aa", "depth", ATMOSPHERE_FRAME, ATMOSPHERE_TRANSMITTANCE, ATMOSPHERE_MULTI_SCATTERING, ATMOSPHERE_SKY_VIEW, ATMOSPHERE_AERIAL]
    }

    fn writes(&self) -> &'static [&'static str] {
        &["pre_aa"]
    }

    fn declare_resources(&self, builder: &mut ResourceBuilder) {
        builder.read("pre_aa");
    }

    fn prepare(&mut self, ctx: &PrepareContext) -> HelioResult<()> {
        self.active = ctx.registry.get::<bool>(ResourceKey::new("atmosphere_active")).unwrap_or(false);
        Ok(())
    }

    fn render_pass_descriptor_with_storage<'a>(
        &'a self,
        _target: &'a wgpu::TextureView,
        _depth: &'a wgpu::TextureView,
        resources: &'a helio_core::ResourceRegistry<'a>,
        storage: &'a mut helio_core::RenderFrameStorage,
    ) -> Option<wgpu::RenderPassDescriptor<'a>> {
        if !self.active {
            return None;
        }
        let target = resources.texture_view(ResourceKey::new("pre_aa"))?;
        let color_attachments: &'a [Option<wgpu::RenderPassColorAttachment<'a>>] =
            storage.retain_boxed_slice(Box::new([Some(wgpu::RenderPassColorAttachment {
                view: target,
                resolve_target: None,
                depth_slice: None,
                ops: wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store },
            })]));
        Some(wgpu::RenderPassDescriptor {
            label: Some("Atmosphere composite"),
            color_attachments,
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        })
    }

    fn execute(&mut self, ctx: &mut PassContext) -> HelioResult<()> {
        if !self.active {
            return Ok(());
        }
        let Some(mut pass) = ctx.render_cmds() else { return Ok(()) };
        let registry = ctx.registry;
        let (Some(frame), Some(transmittance), Some(multi_scattering), Some(sky_view), Some(aerial)) = (
            registry.get::<&wgpu::Buffer>(ResourceKey::new(ATMOSPHERE_FRAME)),
            registry.get::<&wgpu::TextureView>(ResourceKey::new(ATMOSPHERE_TRANSMITTANCE)),
            registry.get::<&wgpu::TextureView>(ResourceKey::new(ATMOSPHERE_MULTI_SCATTERING)),
            registry.get::<&wgpu::TextureView>(ResourceKey::new(ATMOSPHERE_SKY_VIEW)),
            registry.get::<&wgpu::TextureView>(ResourceKey::new(ATMOSPHERE_AERIAL)),
        ) else {
            return Ok(());
        };
        let key = (ctx.camera.clone(), frame.clone(), [ctx.depth.clone(), transmittance.clone(), multi_scattering.clone(), sky_view.clone(), aerial.clone()]);
        if self.key.as_ref() != Some(&key) {
            self.group = Some(ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Atmosphere composite"),
                layout: &self.layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: ctx.camera.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: frame.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(transmittance) },
                    wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(sky_view) },
                    wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::TextureView(aerial) },
                    wgpu::BindGroupEntry { binding: 5, resource: wgpu::BindingResource::TextureView(ctx.depth) },
                    wgpu::BindGroupEntry { binding: 6, resource: wgpu::BindingResource::Sampler(&self.sampler) },
                    wgpu::BindGroupEntry { binding: 7, resource: wgpu::BindingResource::TextureView(multi_scattering) },
                ],
            }));
            self.key = Some(key);
        }
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, self.group.as_ref(), &[]);
        pass.draw(0..3, 0..1);
        Ok(())
    }

    fn on_resize(&mut self, _device: &wgpu::Device, _width: u32, _height: u32) {
        self.group = None;
        self.key = None;
    }
}
