/// Helio v3 renderer — top-level entry point.
///
/// ## Frame CPU path (persistent scene, delta-only GPU uploads)
///
/// 1. `flush_dirty_handles()` — re-uploads instance buffers only for handles
///    whose instances changed since the last frame.  Empty at steady state.
/// 2. Camera + globals upload — always, 144+48 bytes.
/// 3. Light upload — only when `lights_dirty`.
/// 4. Debug batch build.
/// 5. `graph.execute()` — all passes run in topological order.
use std::{collections::HashMap, sync::Arc};

use crate::{
    camera::{Camera, GlobalsUniform},
    debug_draw::{DebugDrawBatch, DebugShape},
    graph::{RenderGraph, pass::PassContext},
    hism::{HismRegistry, HismHandle},
    mesh::{DrawCall, INSTANCE_STRIDE},
    material::GpuMaterial,
    passes::{
        sky_lut::SkyLutPass,
        sky::SkyPass,
        shadow::{ShadowPass, ShadowConfig, GpuShadowMatrix},
        depth_prepass::DepthPrepass,
        gbuffer::GBufferPass,
        deferred_lighting::DeferredLightingPass,
        transparent::TransparentPass,
        billboard::{BillboardPass, BillboardConfig},
        radiance_cascades::{RadianceCascadesPass, RcConfig},
        debug_draw_pass::DebugDrawPass,
        fxaa::FxaaPass,
        taa::TaaPass,
    },
    profiler::{GpuProfiler, CpuFrameStats},
    resources::{FrameTextures, StubTextures, linear_sampler, shadow_comparison_sampler},
    scene::{InstanceId, LightId, SceneLight, SkyAtmosphere, Skylight, GpuLight},
};

// ── Config types ─────────────────────────────────────────────────────────────

#[derive(Clone, Debug)]
pub struct RendererConfig {
    pub width:          u32,
    pub height:         u32,
    pub surface_format: wgpu::TextureFormat,
    pub anti_aliasing:  AntiAliasingMode,
    pub shadows:        Option<ShadowConfig>,
    pub radiance_cascades: Option<RcConfig>,
    pub billboards:     Option<BillboardConfig>,
    pub bloom:          Option<BloomConfig>,
    pub ssao:           Option<SsaoConfig>,
    pub gpu_driven:     bool,
    pub debug_printout: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AntiAliasingMode { None, Fxaa, Taa }

#[derive(Clone, Debug)]
pub struct BloomConfig  { pub threshold: f32, pub intensity: f32 }
#[derive(Clone, Debug)]
pub struct SsaoConfig   { pub radius: f32, pub bias: f32, pub power: f32, pub samples: u32 }

// ── Error ─────────────────────────────────────────────────────────────────────

#[derive(thiserror::Error, Debug)]
pub enum RendererError {
    #[error("wgpu surface error: {0}")]
    Surface(#[from] wgpu::SurfaceError),
    #[error("no output surface texture")]
    NoSurface,
}

// ── Instance buffer pool ───────────────────────────────────────────────────────

struct InstancePool {
    /// Persistent per-handle buffers. Resized (not replaced) when capacity is exceeded.
    /// DrawCalls hold `Arc` clones of these same buffers — zero copy overhead at steady state.
    buffers: HashMap<HismHandle, (Arc<wgpu::Buffer>, u32)>,  // (buf, capacity)
}

impl InstancePool {
    fn new() -> Self { InstancePool { buffers: HashMap::new() } }

    /// Return (or resize) the persistent instance buffer for `handle`.
    /// Returns an `Arc` the caller can clone into a `DrawCall`.
    fn get_or_resize(
        &mut self,
        device: &wgpu::Device,
        handle: HismHandle,
        count:  u32,
    ) -> Arc<wgpu::Buffer> {
        let needs_resize = match self.buffers.get(&handle) {
            None           => true,
            Some((_, cap)) => *cap < count,
        };
        if needs_resize {
            let buf = Arc::new(device.create_buffer(&wgpu::BufferDescriptor {
                label:              None,
                size:               count.max(1) as u64 * INSTANCE_STRIDE,
                usage:              wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
            self.buffers.insert(handle, (buf, count));
        }
        self.buffers[&handle].0.clone()
    }
}

// ── Persistent CPU-side scene state ──────────────────────────────────────────

/// CPU-resident slot for one instance. Index == InstanceId.0.
struct InstanceSlot {
    handle:    HismHandle,
    transform: glam::Mat4,
}

/// All mutable scene state owned by the renderer.
///
/// Instances and lights are stored in slot vecs for O(1) insertion/removal with
/// stable IDs. Dirty flags drive the minimal GPU upload path each frame.
struct SceneState {
    camera:         Camera,

    // ── Instances ────────────────────────────────────────────────────────
    instance_slots: Vec<Option<InstanceSlot>>,
    instance_free:  Vec<u32>,       // recycled slot indices
    instance_next:  u32,            // high-water mark
    /// Handles whose instance buffers need a GPU re-upload this frame.
    dirty_handles:  std::collections::HashSet<HismHandle>,
    /// True when the draw-call list structure changed (add/remove a handle entirely).
    needs_sort:     bool,

    // ── Lights ───────────────────────────────────────────────────────────
    light_slots:    Vec<Option<SceneLight>>,
    light_free:     Vec<u32>,
    light_next:     u32,
    lights_dirty:   bool,

    // ── Sky ──────────────────────────────────────────────────────────────
    sky:            Option<SkyAtmosphere>,
    skylight:       Option<Skylight>,
}

impl SceneState {
    fn new() -> Self {
        SceneState {
            camera:         bytemuck::Zeroable::zeroed(),
            instance_slots: Vec::new(),
            instance_free:  Vec::new(),
            instance_next:  0,
            dirty_handles:  std::collections::HashSet::new(),
            needs_sort:     false,
            light_slots:    Vec::new(),
            light_free:     Vec::new(),
            light_next:     0,
            lights_dirty:   false,
            sky:            None,
            skylight:       None,
        }
    }
}

// ── Renderer ──────────────────────────────────────────────────────────────────

pub struct Renderer {
    config:          RendererConfig,

    // Per-frame GPU buffers — written every frame.
    camera_buffer:   wgpu::Buffer,
    globals_buffer:  wgpu::Buffer,
    light_buffer:    wgpu::Buffer,
    light_buf_cap:   usize,

    frame_textures:  FrameTextures,
    stubs:           StubTextures,
    linear_sampler:  wgpu::Sampler,
    shadow_sampler:  wgpu::Sampler,

    // Bind group layouts shared across passes.
    camera_bgl:     wgpu::BindGroupLayout,
    material_bgl:   wgpu::BindGroupLayout,
    camera_bg:      wgpu::BindGroup,

    instance_pool:   InstancePool,
    graph:           RenderGraph,
    profiler:        GpuProfiler,
    profiler_scopes: HashMap<String, u32>,

    // Accumulated debug shapes from the current frame.
    debug_shapes:    Vec<DebugShape>,

    // Persistent scene state (mutated by add/remove/update methods).
    scene:       SceneState,
    hism:        HismRegistry,

    // Draw lists — rebuilt incrementally by flush_dirty_handles.
    opaque_draws:      Vec<DrawCall>,
    transparent_draws: Vec<DrawCall>,

    // GPU-side light mirror (updated only when lights_dirty).
    last_lights:   Vec<GpuLight>,

    frame_index:   u64,
}

impl Renderer {
    /// Create the renderer. Allocates all GPU resources upfront.
    ///
    /// `hism` is moved into the renderer. Build and register all HISM entries
    /// before calling `new`; you may add more later via `register_hism`.
    pub fn new(
        device:  &wgpu::Device,
        queue:   &wgpu::Queue,
        config:  RendererConfig,
        hism:    HismRegistry,
    ) -> Self {
        let camera_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label:              Some("camera"),
            size:               std::mem::size_of::<Camera>() as u64,
            usage:              wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let globals_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label:              Some("globals"),
            size:               std::mem::size_of::<GlobalsUniform>() as u64,
            usage:              wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let initial_light_cap = 64usize;
        let light_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label:              Some("lights"),
            size:               (initial_light_cap * std::mem::size_of::<GpuLight>()).max(16) as u64,
            usage:              wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let stubs        = StubTextures::new(device, queue);
        let lin_sampler  = linear_sampler(device);
        let shad_sampler = shadow_comparison_sampler(device);
        let taa          = config.anti_aliasing == AntiAliasingMode::Taa;
        let frame_textures = FrameTextures::new(device, config.width, config.height, config.surface_format, taa);

        // Camera bind group layout (group 0)
        let camera_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("camera_bgl"),
            entries: &[
                wgpu_uniform_entry(0, wgpu::ShaderStages::VERTEX_FRAGMENT),
                wgpu_uniform_entry(1, wgpu::ShaderStages::VERTEX_FRAGMENT),
            ],
        });

        let camera_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label:   Some("camera_bg"),
            layout:  &camera_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: camera_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: globals_buffer.as_entire_binding() },
            ],
        });

        // Material bind group layout (group 1)
        let material_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("material_bgl"),
            entries: &[
                wgpu_uniform_entry(0, wgpu::ShaderStages::FRAGMENT),
                wgpu_tex2d_entry(1, wgpu::ShaderStages::FRAGMENT),
                wgpu_tex2d_entry(2, wgpu::ShaderStages::FRAGMENT),
                wgpu::BindGroupLayoutEntry { binding: 3, visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering), count: None },
                wgpu_tex2d_entry(4, wgpu::ShaderStages::FRAGMENT),
                wgpu_tex2d_entry(5, wgpu::ShaderStages::FRAGMENT),
            ],
        });

        let mut profiler        = GpuProfiler::new(device, queue, config.debug_printout, 32);
        let mut profiler_scopes = HashMap::new();

        let pass_labels = [
            "sky_lut", "sky", "shadow", "depth_prepass",
            "gbuffer", "radiance_cascades", "deferred_lighting",
            "transparent", "billboard", "debug_draw", "fxaa", "taa",
        ];
        for label in &pass_labels {
            if config.debug_printout {
                let idx = profiler.register_scope(label);
                profiler_scopes.insert(label.to_string(), idx);
            }
        }

        // ── Build render graph ─────────────────────────────────────────────
        let mut graph = RenderGraph::new();

        // Sky LUT
        let sky_lut = SkyLutPass::new(device, &globals_buffer);
        let lut_view_ref = sky_lut.lut.create_view(&wgpu::TextureViewDescriptor::default());
        let sky_lut_idx = graph.add_pass("sky_lut", sky_lut);

        // Sky
        let sky = SkyPass::new(device, config.surface_format, &camera_buffer, &globals_buffer, &lut_view_ref, &lin_sampler);
        let sky_idx = graph.add_pass("sky", sky);
        graph.add_dependency(sky_lut_idx, sky_idx);

        // Shadow (optional)
        let shadow_atlas_view = stubs.shadow_stub.create_view(&wgpu::TextureViewDescriptor {
            dimension: Some(wgpu::TextureViewDimension::D2Array), ..Default::default()
        });
        // Stub must be at least one full GpuShadowMatrix (96 bytes) so the deferred shader
        // passes wgpu's minimum-binding-size validation even when shadows are disabled.
        let shadow_matrix_buf_stub = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("shadow_matrix_stub"),
            size: std::mem::size_of::<GpuShadowMatrix>() as u64,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        // When shadows are configured, clone the Arc handles *before* moving the pass
        // into the graph so we can bind the real buffers into DeferredLightingPass.
        let (real_shadow_atlas_view, real_shadow_matrix_arc) =
            if let Some(shadow_cfg) = config.shadows.as_ref() {
                let pass = ShadowPass::new(device, ShadowConfig {
                    atlas_size: shadow_cfg.atlas_size,
                    max_shadow_lights: shadow_cfg.max_shadow_lights,
                }, &camera_bgl);
                // Clone arcs while pass is still in scope.
                let atlas_arc = pass.atlas.clone();
                let mat_arc   = pass.shadow_matrix_buf_arc();
                graph.add_pass("shadow", pass);
                let atlas_view = atlas_arc.create_view(&wgpu::TextureViewDescriptor {
                    dimension: Some(wgpu::TextureViewDimension::D2Array), ..Default::default()
                });
                (Some(atlas_view), Some(mat_arc))
            } else {
                (None, None)
            };

        let eff_shadow_atlas  = real_shadow_atlas_view.as_ref().unwrap_or(&shadow_atlas_view);
        let eff_shadow_matrix: &wgpu::Buffer = real_shadow_matrix_arc
            .as_deref()
            .unwrap_or(&shadow_matrix_buf_stub);

        // Depth prepass
        let depth = DepthPrepass::new(device, &camera_bgl);
        let depth_idx = graph.add_pass("depth_prepass", depth);

        // GBuffer
        let gbuf = GBufferPass::new(device, &camera_bgl, &material_bgl);
        let gbuf_idx = graph.add_pass("gbuffer", gbuf);
        graph.add_dependency(depth_idx, gbuf_idx);

        // RC GI (optional)
        let rc_cascade0_view = if config.radiance_cascades.is_some() {
            let rc = RadianceCascadesPass::new(device, config.width, config.height, &globals_buffer, &light_buffer);
            let v  = rc.cascade0.create_view(&wgpu::TextureViewDescriptor::default());
            let rc_idx = graph.add_pass("radiance_cascades", rc);
            graph.add_dependency(gbuf_idx, rc_idx);
            v
        } else {
            stubs.white_linear.create_view(&wgpu::TextureViewDescriptor::default())
        };

        let env_cube_view = stubs.cube_stub.create_view(&wgpu::TextureViewDescriptor {
            dimension: Some(wgpu::TextureViewDimension::Cube), ..Default::default()
        });

        // Deferred lighting
        let deferred = DeferredLightingPass::new(
            device, config.surface_format,
            &camera_buffer, &globals_buffer, &light_buffer,
            &frame_textures.gbuf_albedo_view, &frame_textures.gbuf_normal_view,
            &frame_textures.gbuf_orm_view,    &frame_textures.gbuf_emissive_view,
            &frame_textures.depth_view,
            eff_shadow_atlas,
            eff_shadow_matrix,
            &rc_cascade0_view,
            &env_cube_view,
            &lin_sampler, &shad_sampler,
        );
        let deferred_idx = graph.add_pass("deferred_lighting", deferred);
        graph.add_dependency(gbuf_idx, deferred_idx);

        // Transparent
        let transparent = TransparentPass::new(device, config.surface_format, &camera_bgl, &material_bgl);
        let trans_idx = graph.add_pass("transparent", transparent);
        graph.add_dependency(deferred_idx, trans_idx);

        // Billboards (optional)
        if let Some(bb_cfg) = config.billboards.as_ref() {
            let atlas_view = stubs.white_srgb.create_view(&wgpu::TextureViewDescriptor::default());
            let bb = BillboardPass::new(device, BillboardConfig { max_instances: bb_cfg.max_instances },
                config.surface_format, &camera_buffer, &atlas_view, &lin_sampler);
            let bb_idx = graph.add_pass("billboard", bb);
            graph.add_dependency(trans_idx, bb_idx);
        }

        // Debug draw
        let dbg = DebugDrawPass::new(device, config.surface_format, &camera_buffer, 65536);
        let dbg_idx = graph.add_pass("debug_draw", dbg);
        graph.add_dependency(trans_idx, dbg_idx);

        // AA resolve (FXAA or TAA)
        match config.anti_aliasing {
            AntiAliasingMode::Fxaa => {
                let fxaa = FxaaPass::new(device, config.surface_format, &frame_textures.pre_aa_view, &lin_sampler);
                let fxaa_idx = graph.add_pass("fxaa", fxaa);
                graph.add_dependency(dbg_idx, fxaa_idx);
            }
            AntiAliasingMode::Taa => {
                if let (Some(ha), Some(hb), Some(vel)) =
                    (&frame_textures.taa_history_a, &frame_textures.taa_history_b, &frame_textures.velocity)
                {
                    let ha_view = ha.create_view(&wgpu::TextureViewDescriptor::default());
                    let hb_view = hb.create_view(&wgpu::TextureViewDescriptor::default());
                    let vel_view= vel.create_view(&wgpu::TextureViewDescriptor::default());
                    let taa = TaaPass::new(device, config.surface_format, &frame_textures.pre_aa_view,
                        &ha_view, &hb_view, &vel_view, &lin_sampler);
                    let taa_idx = graph.add_pass("taa", taa);
                    graph.add_dependency(dbg_idx, taa_idx);
                }
            }
            AntiAliasingMode::None => {
                // No AA → FXAA with quality=0 (plain blit)
                let fxaa = FxaaPass::new(device, config.surface_format, &frame_textures.pre_aa_view, &lin_sampler);
                let fxaa_idx = graph.add_pass("fxaa", fxaa);
                graph.add_dependency(dbg_idx, fxaa_idx);
            }
        }

        Renderer {
            config,
            camera_buffer, globals_buffer, light_buffer, light_buf_cap: initial_light_cap,
            frame_textures, stubs, linear_sampler: lin_sampler, shadow_sampler: shad_sampler,
            camera_bgl, material_bgl, camera_bg,
            instance_pool: InstancePool::new(),
            graph, profiler, profiler_scopes,
            debug_shapes:     Vec::new(),
            scene:            SceneState::new(),
            hism,
            opaque_draws:     Vec::new(),
            transparent_draws: Vec::new(),
            last_lights:      Vec::new(),
            frame_index:      0,
        }
    }

    /// Upload a CPU `Material` and return a GPU-resident `GpuMaterial`.
    ///
    /// The returned `Arc<GpuMaterial>` can be passed directly to
    /// `HismRegistry::register`.  All missing textures are filled with stub
    /// fallbacks so callers do not need access to the internal sampler /
    /// texture layouts.
    pub fn create_material(
        &self,
        device:   &wgpu::Device,
        queue:    &wgpu::Queue,
        material: &crate::material::Material,
    ) -> std::sync::Arc<GpuMaterial> {
        GpuMaterial::upload(device, queue, material, &self.material_bgl, &self.stubs, &self.linear_sampler)
    }

    /// Queue a debug shape for this frame. Cleared after `render`.
    pub fn draw_debug(&mut self, shape: DebugShape) {
        self.debug_shapes.push(shape);
    }

    // ── Scene mutation ─────────────────────────────────────────────────────

    /// Replace the active camera. Takes effect on the next `render` call.
    pub fn set_camera(&mut self, camera: Camera) {
        self.scene.camera = camera;
    }

    /// Add an instance of a registered HISM entry to the persistent scene.
    ///
    /// Returns an `InstanceId` you can pass to `remove_instance` or
    /// `set_instance_transform` later.
    pub fn add_instance(&mut self, handle: HismHandle, transform: glam::Mat4) -> InstanceId {
        let slot = if let Some(free) = self.scene.instance_free.pop() {
            self.scene.instance_slots[free as usize] = Some(InstanceSlot { handle, transform });
            free
        } else {
            let idx = self.scene.instance_next;
            self.scene.instance_next += 1;
            if idx as usize >= self.scene.instance_slots.len() {
                self.scene.instance_slots.push(Some(InstanceSlot { handle, transform }));
            } else {
                self.scene.instance_slots[idx as usize] = Some(InstanceSlot { handle, transform });
            }
            idx
        };
        self.scene.dirty_handles.insert(handle);
        self.scene.needs_sort = true;
        InstanceId(slot)
    }

    /// Remove a previously added instance. The `InstanceId` becomes invalid.
    pub fn remove_instance(&mut self, id: InstanceId) {
        let idx = id.0 as usize;
        if let Some(slot) = self.scene.instance_slots.get_mut(idx).and_then(|s| s.take()) {
            self.scene.dirty_handles.insert(slot.handle);
            self.scene.instance_free.push(id.0);
            self.scene.needs_sort = true;
        }
    }

    /// Update the transform of an existing instance (no GPU work until `render`).
    pub fn set_instance_transform(&mut self, id: InstanceId, transform: glam::Mat4) {
        if let Some(Some(slot)) = self.scene.instance_slots.get_mut(id.0 as usize) {
            slot.transform = transform;
            self.scene.dirty_handles.insert(slot.handle);
        }
    }

    /// Add a light to the persistent scene. Returns a `LightId` for later mutation.
    pub fn add_light(&mut self, light: SceneLight) -> LightId {
        let slot = if let Some(free) = self.scene.light_free.pop() {
            self.scene.light_slots[free as usize] = Some(light);
            free
        } else {
            let idx = self.scene.light_next;
            self.scene.light_next += 1;
            if idx as usize >= self.scene.light_slots.len() {
                self.scene.light_slots.push(Some(light));
            } else {
                self.scene.light_slots[idx as usize] = Some(light);
            }
            idx
        };
        self.scene.lights_dirty = true;
        LightId(slot)
    }

    /// Remove a light from the persistent scene.
    pub fn remove_light(&mut self, id: LightId) {
        if let Some(slot) = self.scene.light_slots.get_mut(id.0 as usize) {
            if slot.take().is_some() {
                self.scene.light_free.push(id.0);
                self.scene.lights_dirty = true;
            }
        }
    }

    /// Replace the parameters of an existing light.
    pub fn update_light(&mut self, id: LightId, light: SceneLight) {
        if let Some(Some(slot)) = self.scene.light_slots.get_mut(id.0 as usize) {
            *slot = light;
            self.scene.lights_dirty = true;
        }
    }

    /// Register a mesh+material with the renderer's HISM registry after construction.
    ///
    /// Returns a `HismHandle` you can pass to `add_instance`. This lets examples
    /// create materials with `create_material` first and then register geometry.
    pub fn register_hism(
        &mut self,
        mesh:     std::sync::Arc<crate::mesh::GpuMesh>,
        material: std::sync::Arc<GpuMaterial>,
    ) -> HismHandle {
        self.hism.register(mesh, material)
    }

    /// Set (or clear) the sky atmosphere. `None` → use a plain colour gradient.
    pub fn set_sky(&mut self, sky: Option<SkyAtmosphere>) {
        self.scene.sky = sky;
    }

    /// Set (or clear) the image-based skylight contribution.
    pub fn set_skylight(&mut self, skylight: Option<Skylight>) {
        self.scene.skylight = skylight;
    }

    // ── Rendering ──────────────────────────────────────────────────────────

    /// Render one frame using the current persistent scene state.
    pub fn render(
        &mut self,
        device:        &wgpu::Device,
        queue:         &wgpu::Queue,
        surface_view:  &wgpu::TextureView,
        delta_time:    f32,
    ) -> Result<CpuFrameStats, RendererError> {
        let cpu_start = std::time::Instant::now();

        // ── Step 1: Flush dirty instance handles to GPU ────────────────────
        self.flush_dirty_handles(device, queue);

        // ── Step 2: Camera + globals upload (always small, always dirty) ───
        queue.write_buffer(&self.camera_buffer, 0, bytemuck::bytes_of(&self.scene.camera));

        let light_count = self.scene.light_slots.iter().filter(|s| s.is_some()).count();
        let globals = GlobalsUniform {
            frame:             self.frame_index as u32,
            delta_time,
            light_count:       light_count as u32,
            ambient_intensity: 0.1,
            ambient_color:     [0.2, 0.25, 0.3],
            csm_split_count:   4,
            rc_world_min:      [0.0; 3],
            _pad0:             0,
        };
        queue.write_buffer(&self.globals_buffer, 0, bytemuck::bytes_of(&globals));

        // ── Step 3: Light upload (only when dirty) ─────────────────────────
        if self.scene.lights_dirty {
            let gpu_lights: Vec<GpuLight> = self.scene.light_slots
                .iter()
                .filter_map(|s| s.as_ref())
                .map(GpuLight::from_scene_light)
                .collect();
            self.upload_lights(device, queue, &gpu_lights);
            self.last_lights = gpu_lights;
            self.scene.lights_dirty = false;
        }

        let prep_ms = cpu_start.elapsed().as_secs_f32() * 1000.0;

        // ── Step 4: Build debug batch ──────────────────────────────────────
        let debug_shapes = std::mem::take(&mut self.debug_shapes);
        let debug_batch  = if debug_shapes.is_empty() { None } else { Some(DebugDrawBatch::build(debug_shapes)) };
        let debug_ref    = debug_batch.as_ref();

        // ── Step 5: graph.execute() ───────────────────────────────────────
        let graph_start = std::time::Instant::now();

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("frame_encoder"),
        });

        let mut ctx = PassContext {
            device,
            queue,
            encoder:         &mut encoder,
            frame_tex:       &self.frame_textures,
            camera:          self.scene.camera,
            camera_buffer:   &self.camera_buffer,
            globals_buffer:  &self.globals_buffer,
            camera_bg:       &self.camera_bg,
            opaque_draws:    &self.opaque_draws,
            transparent_draws: &self.transparent_draws,
            lights:          &self.last_lights,
            light_buffer:    &self.light_buffer,
            sky_atmosphere:  self.scene.sky.as_ref(),
            skylight:        self.scene.skylight.as_ref(),
            debug_batch:     debug_ref,
            frame_index:     self.frame_index,
            width:           self.config.width,
            height:          self.config.height,
            surface_view,
            profiler_scopes: &self.profiler_scopes,
            profiler:        &self.profiler,
        };

        self.graph.execute(&mut ctx);

        // Resolve GPU timestamps (no-op when debug_printout=false)
        self.profiler.resolve(&mut encoder);

        let graph_ms = graph_start.elapsed().as_secs_f32() * 1000.0;

        let submit_start = std::time::Instant::now();
        queue.submit(std::iter::once(encoder.finish()));
        let submit_ms = submit_start.elapsed().as_secs_f32() * 1000.0;

        self.frame_index += 1;

        Ok(CpuFrameStats {
            prep_ms,
            graph_ms,
            submit_ms,
            present_ms: 0.0, // caller measures surface.present()
        })
    }

    // ── Private ────────────────────────────────────────────────────────────

    /// Re-upload only the instance buffers whose handles are in `dirty_handles`,
    /// then rebuild the draw lists for those handles.
    ///
    /// At steady state (nothing changed) this is a no-op: zero GPU uploads,
    /// zero allocations, just an early-out on an empty HashSet.
    fn flush_dirty_handles(&mut self, device: &wgpu::Device, queue: &wgpu::Queue) {
        if self.scene.dirty_handles.is_empty() && !self.scene.needs_sort {
            return;
        }

        let cam_pos = self.scene.camera.position;

        // Collect transforms per dirty handle in one pass over instance_slots.
        let mut groups: HashMap<HismHandle, Vec<glam::Mat4>> = self.scene.dirty_handles
            .iter()
            .map(|&h| (h, Vec::new()))
            .collect();

        for slot in &self.scene.instance_slots {
            if let Some(inst) = slot {
                if let Some(list) = groups.get_mut(&inst.handle) {
                    list.push(inst.transform);
                }
            }
        }

        for (handle, transforms) in groups {
            if transforms.is_empty() {
                // All instances for this handle were removed → drop its draw calls.
                self.opaque_draws.retain(|dc| dc.hism_handle != handle);
                self.transparent_draws.retain(|dc| dc.hism_handle != handle);
                continue;
            }

            let count  = transforms.len() as u32;
            let ibuf   = self.instance_pool.get_or_resize(device, handle, count);
            let flat: Vec<f32> = transforms.iter()
                .flat_map(|m| m.to_cols_array().into_iter())
                .collect();
            queue.write_buffer(&ibuf, 0, bytemuck::cast_slice(&flat));

            // Update an existing draw call in-place, or create a new one.
            let existing = self.opaque_draws.iter_mut()
                .chain(self.transparent_draws.iter_mut())
                .find(|dc| dc.hism_handle == handle);

            if let Some(dc) = existing {
                dc.instance_buffer = ibuf;
                dc.instance_count  = count;
            } else {
                let entry = self.hism.get(handle);
                let mesh  = &entry.mesh;
                let mat   = &entry.material;

                let _depth_sq = {
                    let c = mesh.bounds_center;
                    let dx = c[0] - cam_pos[0];
                    let dy = c[1] - cam_pos[1];
                    let dz = c[2] - cam_pos[2];
                    dx*dx + dy*dy + dz*dz
                };

                let draw = DrawCall {
                    hism_handle:            handle,
                    vertex_buffer:          mesh.vertex_buffer.clone(),
                    index_buffer:           mesh.index_buffer.clone(),
                    vertex_count:           mesh.vertex_count,
                    index_count:            mesh.index_count,
                    material_bind_group:    mat.bind_group.clone(),
                    transparent_blend:      mat.transparent_blend,
                    bounds_center:          mesh.bounds_center,
                    bounds_radius:          mesh.bounds_radius,
                    material_id:            handle.0,
                    instance_buffer:        ibuf,
                    instance_count:         count,
                    instance_buffer_offset: 0,
                };

                if mat.transparent_blend {
                    self.transparent_draws.push(draw);
                } else {
                    self.opaque_draws.push(draw);
                }
                self.scene.needs_sort = true;
            }
        }

        self.scene.dirty_handles.clear();

        if self.scene.needs_sort {
            // Opaque: front-to-back (minimise overdraw).
            self.opaque_draws.sort_unstable_by(|a, b| {
                let da = a.depth_sq(cam_pos.into());
                let db = b.depth_sq(cam_pos.into());
                da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
                    .then(a.hism_handle.0.cmp(&b.hism_handle.0))
            });
            // Transparent: back-to-front (correct blending order).
            self.transparent_draws.sort_unstable_by(|a, b| {
                let da = a.depth_sq(cam_pos.into());
                let db = b.depth_sq(cam_pos.into());
                db.partial_cmp(&da).unwrap_or(std::cmp::Ordering::Equal)
            });
            self.scene.needs_sort = false;
        }
    }

    fn upload_lights(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, lights: &[GpuLight]) {
        let needed = (lights.len() * std::mem::size_of::<GpuLight>()).max(16);
        if needed > self.light_buf_cap * std::mem::size_of::<GpuLight>() {
            self.light_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label:              Some("lights"),
                size:               needed as u64,
                usage:              wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            self.light_buf_cap = lights.len().max(64);
        }
        if !lights.is_empty() {
            queue.write_buffer(&self.light_buffer, 0, bytemuck::cast_slice(lights));
        }
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

fn wgpu_uniform_entry(binding: u32, visibility: wgpu::ShaderStages) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry { binding, visibility,
        ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None },
        count: None }
}

fn wgpu_tex2d_entry(binding: u32, visibility: wgpu::ShaderStages) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry { binding, visibility,
        ty: wgpu::BindingType::Texture {
            sample_type:    wgpu::TextureSampleType::Float { filterable: true },
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled:   false,
        }, count: None }
}
