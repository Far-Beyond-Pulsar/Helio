//! Ultra-Water integrated with Helio's voxel terrain, meadow foliage, sky and haze.
//!
//! Run from the Helio workspace root:
//! `cargo run -p helio-default-passes --release --example water_world -- coast`
//! A deterministic one-shot capture (after voxel streaming warms up) can be
//! rendered and closed with:
//! `cargo run -p helio-default-passes --example water_world -- shoreline --capture-and-exit`
//!
//! Presets: `coast` (wide island-and-ocean view), `shoreline` (low camera at
//! the breakers), and `underwater` (looking through the
//! surface toward the island). Keys 1/2/3 switch views, Space triggers a
//! shoreline splash, and C saves a PNG to `target/water-world/`. The terrain,
//! foliage, clouds and shoreline are seeded so the views stay comparable.
//! Optional CLI: pass `--capture` to save the startup preset after 120 frames;
//! add `--capture-and-exit` to close after that capture. Without either flag,
//! this runs as a live window and never writes a file automatically.

#[path = "voxel_flight/worlds.rs"]
#[allow(dead_code)]
mod worlds;

use glam::{DVec3, Vec3};
use helio::{Camera, Renderer, RendererBuilder, RendererConfig};
use helio_default_passes::{build_default_graph_external_with_voxel_passes, VoxelPassFactory};
use helio_pass_voxel_planet::chunks::pass::ChunkPass;
use helio_pass_voxel_planet::engine::{PlanetFrame, SharedPlanetFrame};
use helio_pass_voxel_planet::{Brush, BrushOp, BrushShape, Planet, PlanetRecipe, VolumeTransform};
use helio_pass_water_sim::{
    GpuWaterVolume, OceanSpectrum, WaterSimPass, WaterVolumeComponent, WaterSpraySettings,
};
use pulsar_scenedb::gpu::{EngineGpuContext, GpuMirrorHandle, SceneGpuConfig, SceneGpuStore};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use winit::{
    application::ApplicationHandler,
    event::{ElementState, KeyEvent, WindowEvent},
    event_loop::{ActiveEventLoop, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    window::{Window, WindowId},
};

const SEED: u64 = 34_120_261;
const REST_Y: f64 = 0.0;
const EXTENT: f32 = 2048.0;

struct Showcase {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    renderer: Renderer,
    source: SharedPlanetFrame,
    planet: Arc<Planet>,
    target: wgpu::Texture,
    target_view: wgpu::TextureView,
    size: [u32; 2],
    present_pipeline: wgpu::RenderPipeline,
    present_layout: wgpu::BindGroupLayout,
    present_sampler: wgpu::Sampler,
    format: wgpu::TextureFormat,
}

fn terrain(seed: u64) -> Planet {
    let mut planet = Planet::new(PlanetRecipe {
        shape: helio_pass_voxel_planet::grid::Shape::Plane,
        plane_size_m: EXTENT as f64 * 2.0,
        voxel_size_m: 0.5,
        terrain: {
            let mut seabed = helio_pass_voxel_planet::layers::TerrainLayers::flat_at(-42.0)
                .heightfield();
            seabed.surface = "Sand".into();
            seabed.source(seed)
        },
        features: worlds::Preset::Earth.features().grow(0.5, seed),
        ..Default::default()
    })
    .expect("create deterministic island terrain");

    // Broad island profile: a ~330 m waterline radius, gentle beach slope and
    // a ~60 m peak, matching the scale of the Ultra-Water reference island.
    // The bake below samples this exact edited terrain via `from_voxel_planet`.
    planet
        .apply(Brush {
            center: [0.0, -878.0, 0.0],
            radius: 938.0,
            shape: BrushShape::Sphere,
            op: BrushOp::Add,
            material: helio_pass_voxel_planet::terrain::material::GRASS,
            height: 0.0,
        })
        .expect("sculpt seeded island");
    planet
}

impl Showcase {
    fn new(
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
        format: wgpu::TextureFormat,
    ) -> Self {
        let size = [1600, 900];
        let planet = Arc::new(terrain(SEED));
        let context = EngineGpuContext::new(device.clone(), queue.clone());
        let mut store = SceneGpuStore::new(
            &context,
            SceneGpuConfig {
                classes: vec![],
                tombstone_headroom: 0,
                max_cells_metadata: 0,
            },
        );
        helio_pass_sky::AtmosphereComponent::register_gpu_columns_growable(&mut store, 4, &device);
        helio_pass_gbuffer::MeshComponent::register_gpu_columns_growable(&mut store, 16, &device);
        helio_pass_gbuffer::MaterialComponent::register_gpu_columns_growable(
            &mut store, 16, &device,
        );
        helio_pass_gbuffer::StaticObjectComponent::register_gpu_columns_growable(
            &mut store, 16, &device,
        );
        helio_pass_forward_lit::LightComponent::register_gpu_columns_growable(
            &mut store, 16, &device,
        );
        helio_pass_volumetric_fog::GlobalFogComponent::register_gpu_columns_growable(
            &mut store, 4, &device,
        );
        helio_pass_water_sim::WaterVolumeComponent::register_gpu_columns_growable(
            &mut store, 8, &device,
        );
        let mirror = GpuMirrorHandle::new(Arc::new(store), queue.clone());
        let mut scene = pulsar_scenedb::SceneDb::new();
        scene.world.attach_gpu_mirror(mirror.clone());

        let sun = Vec3::new(-0.42, 0.78, 0.46).normalize();
        let light = scene.world.spawn();
        scene.world.insert(
            light,
            helio_pass_forward_lit::LightComponent::from(helio::GpuLight {
                position_range: [0.0, 0.0, 0.0, f32::MAX],
                direction_outer: [-sun.x, -sun.y, -sun.z, 0.0],
                color_intensity: [1.0, 0.94, 0.82, 3.0],
                shadow_index: u32::MAX,
                light_type: helio::LightType::Directional as u32,
                god_rays_enabled: 1,
                god_rays_density: 1.0,
                god_rays_weight: 1.0,
                god_rays_exposure: 1.0,
                god_rays_decay: 1.0,
                ..Default::default()
            }),
        );

        // A near-ground atmosphere centred so the plane's y=0 surface lies at
        // its ground radius; Earth's scattering and the same authored haze
        // recipe used by voxel_flight provide the air and horizon depth.
        let world_radius = 10_000.0;
        let atmosphere = scene.world.spawn();
        scene.world.insert(
            atmosphere,
            worlds::Preset::Earth
                .sky(world_radius)
                .expect("Earth showcase atmosphere")
                .around_planet([0.0, -world_radius as f32, 0.0], world_radius),
        );
        let haze = scene.world.spawn();
        scene.world.insert(
            haze,
            worlds::Preset::Earth
                .haze(world_radius)
                .expect("Earth showcase haze")
                .around_planet([0.0, -world_radius as f32, 0.0], world_radius),
        );

        let meadow = worlds::Preset::Earth
            .meadow()
            .expect("Earth meadow recipe")
            .with_tall_stands()
            .with_wind([1.0, 0.35], 4.5, 0.8);
        for row in &meadow.types {
            let entity = scene.world.spawn();
            scene.world.insert(entity, *row);
        }
        let wind = scene.world.spawn();
        scene.world.insert(wind, meadow.wind);

        let mut water = GpuWaterVolume::default_ocean();
        water.bounds_min = [-EXTENT, -500.0, -EXTENT, 0.0];
        water.bounds_max = [EXTENT, 300.0, EXTENT, REST_Y as f32];
        water.wave_params = [0.55, 30.0, 1.0, 1.0];
        // Ultra-Water OceanSurface defaults. Helio's row maps medium scatter
        // to water_color.rgb, absorption to extinction.rgb, foam bias/scale
        // to water_color.w/extinction.w, refraction to reflection_refraction.y,
        // foam brightness to reflection_refraction.w, and exponential depth
        // fog to fog_params.x. wave_params carries the closest amplitude / 
        // wavelength / speed / steepness values for uShoreWave; the current
        // Helio shoreline train already uses the same 0.55 / 30 / 1 / 1 defaults.
        water.water_color = [0.0, 0.05, 0.08, 0.05];
        water.extinction = [0.42, 0.075, 0.048, 1.3];
        water.reflection_refraction[1] = 0.06;
        water.reflection_refraction[3] = 0.85;
        water.fog_params[0] = 1.0 / 18_000.0;
        // No direct WaterVolume equivalents exist for uSSS, uOptics roughness
        // or underwater-window distortion, the foam tint, or the uFoam shore
        // band (whose default .35 is already the shoreline shader's floor).
        // Keep Helio's independent caustics, refraction and Fresnel defaults.
        water.wind_params = [0.939_692_6, 0.342_020_15, 9.0, 1.0];
        let water_entity = scene.world.spawn();
        scene.world.insert(water_entity, WaterVolumeComponent::from(water));

        let source: SharedPlanetFrame = Arc::new(Mutex::new(None));
        let pass_source = source.clone();
        let config = helio_pass_voxel_planet::chunks::ResidencyConfig::default();
        let appearance = worlds::Preset::Earth.appearance(planet.field().appearance());
        let factory: VoxelPassFactory = Arc::new(move |_, _, _, _| {
            let mut pass = ChunkPass::with_config(pass_source.clone(), config);
            pass.set_appearance(Some(appearance));
            Box::new(pass)
        });
        let mut renderer_config = RendererConfig::new(
            size[0],
            size[1],
            format,
        );
        renderer_config.enable_foliage = true;
        renderer_config.foliage_tufts_per_m2 = Some(meadow.tufts_per_m2 * 1.4);
        renderer_config.foliage_blades_per_tuft = Some(meadow.blades_per_tuft);
        renderer_config.foliage_range = Some(meadow.range * 1.5);
        let mut renderer = RendererBuilder::new(renderer_config, mirror)
            .with_external_device()
            .with_pass_build_context(Box::new(move |ctx| {
                build_default_graph_external_with_voxel_passes(ctx, vec![factory.clone()])
            }))
            .build(
                device.clone(),
                queue.clone(),
                size[0],
                size[1],
                renderer_config.surface_format,
            );
        renderer.set_ambient([0.45, 0.58, 0.8], 0.35);
        let mut post = helio_pass_postprocess::PostProcessSettings::default();
        post.tonemap_operator = helio::TonemapOperator::AgxPunchy;
        post.tonemap_exposure = 2.0;
        post.color_contrast = [1.06; 3];
        let camera = scene.world.spawn();
        scene.world.insert(
            camera,
            helio_pass_postprocess::CameraPostProcessComponent::new(0, &post),
        );
        scene.world.flush_gpu_mirror(&queue);
        std::mem::forget(scene);

        if let Some(sim) = renderer.find_pass_mut::<WaterSimPass>() {
            sim.set_shore_map_from_voxel_planet(
                [0.0, 0.0],
                [EXTENT * 2.0, EXTENT * 2.0],
                REST_Y,
                &planet,
                VolumeTransform::default(),
            )
            .expect("bake shoreline from the edited voxel terrain");
            sim.set_ocean_rest_height(REST_Y)
                .expect("match the ocean volume rest plane");
            // Match `defaultOceanParams()` and the JS Spray defaults exactly.
            sim.set_wind([0.939_692_6, 0.342_020_15], 9.0);
            sim.set_ocean_spectrum(OceanSpectrum::default());
            sim.set_spray_settings(WaterSpraySettings::default());
        } else {
            panic!("the default render graph did not build WaterSimPass");
        }

        let target = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("water world showcase output"),
            size: wgpu::Extent3d {
                width: size[0],
                height: size[1],
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::COPY_SRC
                | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let target_view = target.create_view(&Default::default());
        let present_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("water world present layout"),
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
        });
        let present_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("water world present sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        let present_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("water world present shader"),
            source: wgpu::ShaderSource::Wgsl(
                r#"
@group(0) @binding(0) var image: texture_2d<f32>;
@group(0) @binding(1) var image_sampler: sampler;
struct Out { @builtin(position) position: vec4<f32>, @location(0) uv: vec2<f32> }
@vertex fn vs(@builtin(vertex_index) index: u32) -> Out {
    let x = f32((index << 1u) & 2u);
    let y = f32(index & 2u);
    var out: Out;
    out.position = vec4<f32>(x * 2.0 - 1.0, 1.0 - y * 2.0, 0.0, 1.0);
    out.uv = vec2<f32>(x, y);
    return out;
}
@fragment fn fs(input: Out) -> @location(0) vec4<f32> {
    return textureSample(image, image_sampler, input.uv);
}
"#
                .into(),
            ),
        });
        let present_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("water world present pipeline layout"),
            bind_group_layouts: &[Some(&present_layout)],
            immediate_size: 0,
        });
        let present_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("water world present pipeline"),
            layout: Some(&present_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &present_shader,
                entry_point: Some("vs"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &present_shader,
                entry_point: Some("fs"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: Default::default(),
            multiview_mask: None,
            cache: None,
        });
        Self {
            device,
            queue,
            renderer,
            source,
            planet,
            target,
            target_view,
            size,
            present_pipeline,
            present_layout,
            present_sampler,
            format,
        }
    }

    fn draw(&mut self, eye: DVec3, forward: Vec3) {
        let sun = Vec3::new(-0.42, 0.78, 0.46).normalize();
        *self.source.lock().expect("planet frame lock") =
            Some(PlanetFrame::single(eye, self.planet.clone(), sun));
        let forward = forward.normalize();
        let up = if forward.y.abs() > 0.999 {
            Vec3::X
        } else {
            Vec3::Y
        };
        let aspect = self.size[0] as f32 / self.size[1] as f32;
        let camera = Camera::perspective_look_to(
            eye,
            forward,
            up,
            48_f32.to_radians(),
            aspect,
            0.05,
            20_000.0,
        );
        self.renderer
            .render(&camera, &self.target_view)
            .expect("render water world showcase");
        let _ = self.device.poll(wgpu::PollType::Poll);
    }

    fn present(&self, view: &wgpu::TextureView) {
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("water world present bind group"),
            layout: &self.present_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&self.target_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.present_sampler),
                },
            ],
        });
        let mut encoder = self.device.create_command_encoder(&Default::default());
        let attachments = [Some(wgpu::RenderPassColorAttachment {
            view,
            resolve_target: None,
            depth_slice: None,
            ops: wgpu::Operations {
                load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                store: wgpu::StoreOp::Store,
            },
        })];
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("water world present"),
            color_attachments: &attachments,
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_pipeline(&self.present_pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.draw(0..3, 0..1);
        drop(pass);
        self.queue.submit([encoder.finish()]);
    }

    fn save(&self, output: &Path, preset: &str) {
        let row = (self.size[0] * 4).div_ceil(256) * 256;
        let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("water world readback"),
            size: u64::from(row) * u64::from(self.size[1]),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = self.device.create_command_encoder(&Default::default());
        encoder.copy_texture_to_buffer(
            self.target.as_image_copy(),
            wgpu::TexelCopyBufferInfo {
                buffer: &buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(row),
                    rows_per_image: Some(self.size[1]),
                },
            },
            self.target.size(),
        );
        self.queue.submit([encoder.finish()]);
        buffer.slice(..).map_async(wgpu::MapMode::Read, |result| result.unwrap());
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("wait for showcase capture");
        let data = buffer
            .slice(..)
            .get_mapped_range()
            .expect("mapped showcase readback");
        let mut pixels: Vec<u8> = data
            .chunks(row as usize)
            .flat_map(|line| line[..self.size[0] as usize * 4].to_vec())
            .collect();
        if matches!(self.format, wgpu::TextureFormat::Bgra8UnormSrgb | wgpu::TextureFormat::Bgra8Unorm)
        {
            for pixel in pixels.chunks_exact_mut(4) {
                pixel.swap(0, 2);
            }
        }
        let path = output.join(format!("water_world_{preset}.png"));
        image::save_buffer(
            &path,
            &pixels,
            self.size[0],
            self.size[1],
            image::ColorType::Rgba8,
        )
        .expect("write showcase PNG");
        eprintln!("WATER_WORLD_CAPTURE {}", path.display());
    }
}

struct App {
    state: Option<AppState>,
    initial_preset: String,
    output: PathBuf,
    capture_on_start: bool,
    exit_after_capture: bool,
}

struct AppState {
    window: Arc<Window>,
    surface: wgpu::Surface<'static>,
    surface_config: wgpu::SurfaceConfiguration,
    device: Arc<wgpu::Device>,
    showcase: Showcase,
    preset: String,
    capture_requested: bool,
    capture_countdown: Option<u32>,
}

impl AppState {
    fn set_preset(&mut self, preset: &str) {
        self.preset = preset.to_owned();
        self.window.set_title(&format!(
            "Helio Water World — {}  |  1 Coast · 2 Shoreline · 3 Underwater · Space Splash · C Capture",
            preset
        ));
    }

    fn draw(&mut self, output: &Path) -> bool {
        let (eye, forward) = preset_view(&self.preset);
        self.showcase.draw(eye, forward);
        let mut capture_now = self.capture_requested;
        if let Some(frames) = self.capture_countdown.as_mut() {
            if *frames <= 1 {
                self.capture_countdown = None;
                capture_now = true;
            } else {
                *frames -= 1;
            }
        }
        let frame = match self.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(frame)
            | wgpu::CurrentSurfaceTexture::Suboptimal(frame) => frame,
            wgpu::CurrentSurfaceTexture::Lost | wgpu::CurrentSurfaceTexture::Outdated => {
                self.surface.configure(&self.device, &self.surface_config);
                self.capture_requested |= capture_now;
                return false;
            }
            _ => {
                self.capture_requested |= capture_now;
                return false;
            }
        };
        let view = frame.texture.create_view(&Default::default());
        self.showcase.present(&view);
        self.showcase.queue.present(frame);
        if capture_now {
            std::fs::create_dir_all(output).expect("create capture output directory");
            self.showcase.save(output, &self.preset);
            self.capture_requested = false;
            return true;
        }
        false
    }
}

fn preset_view(preset: &str) -> (DVec3, Vec3) {
    match preset {
        "coast" => (
            DVec3::new(-980.0, 115.0, -1_240.0),
            Vec3::new(0.62, -0.025, 0.78),
        ),
        "shoreline" => (
            DVec3::new(-210.0, 36.0, -220.0),
            Vec3::new(-0.68, -0.12, 0.72),
        ),
        "underwater" => (
            DVec3::new(-1_120.0, -28.0, -1_060.0),
            Vec3::new(0.68, 0.035, 0.73),
        ),
        other => panic!("unknown preset {other}; use coast, shoreline, or underwater"),
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.state.is_some() {
            return;
        }
        let window = Arc::new(
            event_loop
                .create_window(
                    Window::default_attributes()
                        .with_title("Helio Water World")
                        .with_inner_size(winit::dpi::LogicalSize::new(1600u32, 900u32)),
                )
                .expect("create water-world window"),
        );
        let size = window.inner_size();
        let graphics = pollster::block_on(helio::HelioGraphicsContext::new(
            window.clone(),
            size.width.max(1),
            size.height.max(1),
            helio::HelioGraphicsConfig {
                application_name: "Water World Showcase".into(),
                ..Default::default()
            },
        ))
        .expect("create Helio graphics context");
        let device = graphics.device.clone();
        let queue = graphics.queue.clone();
        let surface_config = graphics.surface_config.clone();
        let surface = graphics.surface;
        let showcase = Showcase::new(device.clone(), queue, surface_config.format);
        let mut state = AppState {
            window,
            surface,
            surface_config,
            device,
            showcase,
            preset: "coast".to_owned(),
            capture_requested: false,
            capture_countdown: self.capture_on_start.then_some(120),
        };
        state.set_preset(&self.initial_preset);
        self.state = Some(state);
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        let Some(state) = &mut self.state else { return };
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) if size.width > 0 && size.height > 0 => {
                state.surface_config.width = size.width;
                state.surface_config.height = size.height;
                state.surface.configure(&state.device, &state.surface_config);
            }
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        state: ElementState::Pressed,
                        physical_key: PhysicalKey::Code(key),
                        repeat: false,
                        ..
                    },
                ..
            } => match key {
                KeyCode::Escape => event_loop.exit(),
                KeyCode::Digit1 => state.set_preset("coast"),
                KeyCode::Digit2 => state.set_preset("shoreline"),
                KeyCode::Digit3 => state.set_preset("underwater"),
                KeyCode::KeyC => state.capture_requested = true,
                KeyCode::Space => {
                    if let Some(sim) = state.showcase.renderer.find_pass_mut::<WaterSimPass>() {
                let _ = sim.add_spray_impact(
                            DVec3::new(-350.0, REST_Y, -500.0),
                            0.9,
                            1.7,
                        );
                    }
                }
                _ => {}
            },
            WindowEvent::RedrawRequested => {
                let captured = state.draw(&self.output);
                if captured && self.exit_after_capture {
                    event_loop.exit();
                }
                state.window.request_redraw();
            }
            _ => {}
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(state) = &self.state {
            state.window.request_redraw();
        }
    }
}

fn main() {
    let mut initial_preset = None;
    let mut output = None;
    let mut capture_on_start = false;
    let mut exit_after_capture = false;
    for arg in std::env::args().skip(1) {
        match arg.as_str() {
            "--capture" => capture_on_start = true,
            "--capture-and-exit" => {
                capture_on_start = true;
                exit_after_capture = true;
            }
            _ if initial_preset.is_none() => initial_preset = Some(arg),
            _ if output.is_none() => output = Some(PathBuf::from(arg)),
            _ => panic!("unexpected argument {arg}"),
        }
    }
    let initial_preset = initial_preset.unwrap_or_else(|| "coast".to_owned());
    let output = output.unwrap_or_else(|| PathBuf::from("target/water-world"));
    preset_view(&initial_preset);
    let event_loop = EventLoop::new().expect("create event loop");
    event_loop
        .run_app(&mut App {
            state: None,
            initial_preset,
            output,
            capture_on_start,
            exit_after_capture,
        })
        .expect("run water-world showcase");
}
