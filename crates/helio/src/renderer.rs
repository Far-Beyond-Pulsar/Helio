use std::sync::{Arc, Mutex};

use arrayvec::ArrayVec;
use helio_pass_billboard::BillboardPass;
use helio_pass_deferred_light::DeferredLightPass;
use helio_pass_depth_prepass::DepthPrepassPass;
use helio_pass_hiz::HiZBuildPass;
use helio_pass_light_cull::LightCullPass;
use helio_pass_occlusion_cull::OcclusionCullPass;
use helio_pass_gbuffer::GBufferPass;
use helio_pass_shadow::ShadowPass;
use helio_pass_shadow_matrix::ShadowMatrixPass;
use helio_pass_simple_cube::SimpleCubePass;
use helio_pass_sky_lut::SkyLutPass;
use helio_pass_sky::SkyPass;
use helio_pass_taa::TaaPass;
use helio_pass_fxaa::FxaaPass;
use helio_pass_transparent::TransparentPass;
use helio_pass_virtual_geometry::VirtualGeometryPass;
use helio_pass_debug::{DebugPass, DebugVertex, DebugCameraUniform};
use helio_v3::{ RenderGraph, RenderPass, Result as HelioResult };

// TODO: Add these passes once cross-reference issues are resolved:
// - SkyPass (needs sky_lut_view from SkyLutPass)
// - SsaoPass (needs gbuffer views + depth view)
// - SmaaPass, TaaPass (for higher-quality AA)
use std::collections::HashMap;

/// Halton (base-2, base-3) jitter table — matches `helio-pass-taa` so geometry
/// and the TAA resolve shader index the same sub-pixel offset each frame.
const HALTON_JITTER: [[f32; 2]; 16] = [
    [0.5,     0.333333],
    [0.25,    0.666667],
    [0.75,    0.111111],
    [0.125,   0.444444],
    [0.625,   0.777778],
    [0.375,   0.222222],
    [0.875,   0.555556],
    [0.0625,  0.888889],
    [0.5625,  0.037037],
    [0.3125,  0.37037 ],
    [0.8125,  0.703704],
    [0.1875,  0.148148],
    [0.6875,  0.481481],
    [0.4375,  0.814815],
    [0.9375,  0.259259],
    [0.03125, 0.592593],
];

use crate::groups::{ GroupId, GroupMask };
use crate::handles::{ LightId, MaterialId, MeshId, ObjectId, VirtualObjectId };
use crate::material::{ MaterialAsset, TextureUpload, MAX_TEXTURES };
use crate::mesh::{ MeshBuffers, MeshUpload };
use crate::scene::{ Camera, ObjectDescriptor, Result as SceneResult, Scene };
use crate::vg::{ VirtualMeshId, VirtualMeshUpload, VirtualObjectDescriptor };

/// Spotlight icon embedded at compile time — used as the editor billboard sprite.
static SPOTLIGHT_PNG: &[u8] = include_bytes!("../../../spotlight.png");

pub fn required_wgpu_features(adapter_features: wgpu::Features) -> wgpu::Features {
    #[cfg(not(target_arch = "wasm32"))]
    let required =
        wgpu::Features::TEXTURE_BINDING_ARRAY |
        wgpu::Features::SAMPLED_TEXTURE_AND_STORAGE_BUFFER_ARRAY_NON_UNIFORM_INDEXING;
    #[cfg(target_arch = "wasm32")]
    let required = wgpu::Features::empty();
    let optional =
        wgpu::Features::MULTI_DRAW_INDIRECT_COUNT | // compacted indirect count buffer
        wgpu::Features::SHADER_PRIMITIVE_INDEX; // @builtin(primitive_index) in fs
    required | (adapter_features & optional)
}

pub fn required_wgpu_limits(adapter_limits: wgpu::Limits) -> wgpu::Limits {
    wgpu::Limits {
        max_sampled_textures_per_shader_stage: MAX_TEXTURES as u32,
        max_samplers_per_shader_stage: MAX_TEXTURES as u32,
        ..adapter_limits
    }
}

/// Global Illumination configuration (AAA dual-tier: RC near, ambient far).
#[derive(Debug, Clone, Copy)]
pub struct GiConfig {
    /// Radiance Cascades volume radius around camera (world units).
    /// GI within this radius uses RC, outside uses cheap ambient fallback.
    /// Default: 80.0 (AAA near-field quality like Unreal Lumen).
    pub rc_radius: f32,
    /// Fade margin for smooth RC→ambient transition (world units).
    /// Default: 20.0 (soft blend zone).
    pub rc_fade_margin: f32,
}

impl Default for GiConfig {
    fn default() -> Self {
        Self {
            rc_radius: 80.0, // AAA near-field GI
            rc_fade_margin: 20.0, // Smooth transition
        }
    }
}

impl GiConfig {
    /// Disable RC entirely (use only ambient/hemisphere for all distances).
    pub fn ambient_only() -> Self {
        Self {
            rc_radius: 0.0,
            rc_fade_margin: 0.0,
        }
    }

    /// High-quality large-radius RC (for open worlds).
    pub fn large_radius(radius: f32) -> Self {
        Self {
            rc_radius: radius,
            rc_fade_margin: radius * 0.25,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct RendererConfig {
    pub width: u32,
    pub height: u32,
    pub surface_format: wgpu::TextureFormat,
    /// Global illumination configuration (RC near, ambient far).
    pub gi_config: GiConfig,
    /// Shadow quality (Low, Medium, High, Ultra).
    pub shadow_quality: libhelio::ShadowQuality,
    /// Debug visualisation mode (0=normal, 10=shadow heatmap, 11=light-space depth).
    pub debug_mode: u32,
    /// Render scale factor (0.25..=1.0). Geometry (depth, GBuffer, lighting) is
    /// rendered at `ceil(width*scale) × ceil(height*scale)` and temporally upscaled
    /// to native output resolution by TaaPass. Default: 1.0 (native resolution).
    pub render_scale: f32,
}

impl RendererConfig {
    pub fn new(width: u32, height: u32, surface_format: wgpu::TextureFormat) -> Self {
        Self {
            width,
            height,
            surface_format,
            gi_config: GiConfig::default(), // AAA default: RC close, ambient far
            shadow_quality: libhelio::ShadowQuality::Medium, // Default quality
            debug_mode: 0,
            render_scale: 0.75, // 75% res (~1.78x fewer shaded pixels), upscaled via TAA
        }
    }

    /// Builder: set custom GI configuration.
    pub fn with_gi_config(mut self, gi_config: GiConfig) -> Self {
        self.gi_config = gi_config;
        self
    }

    /// Builder: set shadow quality.
    pub fn with_shadow_quality(mut self, quality: libhelio::ShadowQuality) -> Self {
        self.shadow_quality = quality;
        self
    }

    /// Builder: set render scale factor (0.25..=1.0).
    /// 0.5 renders geometry at half resolution and upscales 2× via TAA temporal accumulation.
    pub fn with_render_scale(mut self, scale: f32) -> Self {
        self.render_scale = scale.clamp(0.25, 1.0);
        self
    }

    /// Internal (geometry-pass) render width = ceil(width × render_scale).
    pub fn internal_width(&self) -> u32 {
        (((self.width as f32) * self.render_scale).ceil() as u32).max(1)
    }

    /// Internal (geometry-pass) render height = ceil(height × render_scale).
    pub fn internal_height(&self) -> u32 {
        (((self.height as f32) * self.render_scale).ceil() as u32).max(1)
    }
}

pub struct Renderer {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    graph: RenderGraph,
    graph_kind: GraphKind,
    scene: Scene,
    depth_texture: wgpu::Texture,
    depth_view: wgpu::TextureView,
    /// Native output dimensions (separate from depth_texture which may be at internal res).
    output_width: u32,
    output_height: u32,
    /// Current render scale (stored for graph rebuilds).
    render_scale: f32,
    /// Camera position used for editor grid and debug world-gizmo (camera-follow).
    editor_camera_pos: glam::Vec3,
    /// Full-resolution depth texture — only present when render_scale < 1.0.
    /// BillboardPass (and other post-upscale passes) use this instead of ctx.depth
    /// so their render-pass dimensions match the native-resolution colour target.
    full_res_depth_texture: Option<wgpu::Texture>,
    full_res_depth_view: Option<wgpu::TextureView>,
    surface_format: wgpu::TextureFormat,
    debug_camera_buffer: wgpu::Buffer,
    ambient_color: [f32; 3],
    ambient_intensity: f32,
    clear_color: [f32; 4],
    gi_config: GiConfig,
    shadow_quality: libhelio::ShadowQuality,
    debug_mode: u32,
    /// Depth-test mode for debug drawable lines
    debug_depth_test: bool,
    /// Editor mode root flag (scenes with editor helpers like grid and icons).
    editor_mode: bool,
    /// Shared debug drawing state (lines, grid toggles).
    debug_state: Arc<Mutex<DebugDrawState>>,
    /// User-supplied billboard instances set via `set_billboard_instances()`.
    billboard_instances: Vec<helio_pass_billboard::BillboardInstance>,
    /// Per-frame scratch buffer: user billboards + auto editor-light icons.
    billboard_scratch: Vec<helio_pass_billboard::BillboardInstance>,
    /// Tracks the last-known GpuLight data for every light currently in the scene,
    /// used to auto-generate editor billboard icons in the GroupId::EDITOR group.
    editor_lights: HashMap<LightId, crate::GpuLight>,
}

/// Which graph is currently active — used by `set_render_size` to rebuild correctly.
enum GraphKind {
    /// Full deferred pipeline (default).
    Default,
    /// Minimal single-cube debug graph.
    Simple,
    /// User-provided graph; never rebuilt automatically.
    Custom,
}

struct DebugDrawState {
    /// Whether editor-mode grid should be rendered.
    editor_enabled: bool,
    /// Current camera position used to build camera-following grid in editor mode.
    camera_position: glam::Vec3,
    /// User lines to render this frame.
    user_lines: Vec<DebugVertex>,
}

impl Default for DebugDrawState {
    fn default() -> Self {
        Self {
            editor_enabled: false,
            camera_position: glam::Vec3::ZERO,
            user_lines: Vec::new(),
        }
    }
}

struct DebugDrawPass {
    pass: DebugPass,
    state: Arc<Mutex<DebugDrawState>>,
    editor_mode: bool,
}

impl DebugDrawPass {
    fn new(
        device: &wgpu::Device,
        camera_buf: &wgpu::Buffer,
        surface_format: wgpu::TextureFormat,
        state: Arc<Mutex<DebugDrawState>>,
        depth_test: bool,
        editor_mode: bool,
    ) -> Self {
        Self {
            pass: DebugPass::new(device, camera_buf, surface_format, depth_test),
            state,
            editor_mode,
        }
    }

    fn build_frame_vertices(&self) -> Vec<DebugVertex> {
        let state = self.state.lock().unwrap();

        let mut output = if self.editor_mode {
            Vec::new()
        } else {
            state.user_lines.clone()
        };

        let cam = state.camera_position;
        let cam_dist = cam.length();

        if self.editor_mode && state.editor_enabled {
            let grid_step = if cam_dist < 20.0_f32 {
                1.0_f32
            } else if cam_dist < 60.0_f32 {
                2.0_f32
            } else if cam_dist < 150.0_f32 {
                5.0_f32
            } else if cam_dist < 300.0_f32 {
                10.0_f32
            } else {
                20.0_f32
            };

            let minor_color: [f32; 4] = [0.25, 0.25, 0.25, 1.0];
            let major_color: [f32; 4] = [0.5, 0.5, 0.5, 1.0];
            let axis_color_x: [f32; 4] = [1.0, 0.2, 0.2, 1.0];
            let axis_color_z: [f32; 4] = [0.2, 1.0, 0.2, 1.0];

            let range: f32 = 40.0;
            let count = (range / grid_step).ceil() as i32;
            let center_x = (cam.x / grid_step).round() * grid_step;
            let center_z = (cam.z / grid_step).round() * grid_step;

            for i in -count..=count {
                let x = center_x + i as f32 * grid_step;
                let z = center_z + i as f32 * grid_step;
                let x_color = if i.rem_euclid(5) == 0 { major_color } else { minor_color };
                let z_color = if i.rem_euclid(5) == 0 { major_color } else { minor_color };

                // Lines parallel to Z at varying X
                output.push(DebugVertex { position: [x, 0.0, center_z - range], _pad: 0.0, color: if x.abs() < 0.01 { axis_color_x } else { x_color } });
                output.push(DebugVertex { position: [x, 0.0, center_z + range], _pad: 0.0, color: if x.abs() < 0.01 { axis_color_x } else { x_color } });

                // Lines parallel to X at varying Z
                output.push(DebugVertex { position: [center_x - range, 0.0, z], _pad: 0.0, color: if z.abs() < 0.01 { axis_color_z } else { z_color } });
                output.push(DebugVertex { position: [center_x + range, 0.0, z], _pad: 0.0, color: if z.abs() < 0.01 { axis_color_z } else { z_color } });
            }

            // World origin marker (fixed in place)
            let origin_color = [1.0, 1.0, 0.0, 1.0];
            output.push(DebugVertex { position: [-3.0, 0.0, 0.0], _pad: 0.0, color: origin_color });
            output.push(DebugVertex { position: [3.0, 0.0, 0.0], _pad: 0.0, color: origin_color });
            output.push(DebugVertex { position: [0.0, 0.0, -3.0], _pad: 0.0, color: origin_color });
            output.push(DebugVertex { position: [0.0, 0.0, 3.0], _pad: 0.0, color: origin_color });

            // Camera-relative origin marker
            let camera_marker_color = [0.0, 1.0, 1.0, 1.0];
            let mark = cam;
            output.push(DebugVertex { position: [mark.x - 0.3, mark.y, mark.z], _pad: 0.0, color: camera_marker_color });
            output.push(DebugVertex { position: [mark.x + 0.3, mark.y, mark.z], _pad: 0.0, color: camera_marker_color });
            output.push(DebugVertex { position: [mark.x, mark.y - 0.3, mark.z], _pad: 0.0, color: camera_marker_color });
            output.push(DebugVertex { position: [mark.x, mark.y + 0.3, mark.z], _pad: 0.0, color: camera_marker_color });
            output.push(DebugVertex { position: [mark.x, mark.y, mark.z - 0.3], _pad: 0.0, color: camera_marker_color });
            output.push(DebugVertex { position: [mark.x, mark.y, mark.z + 0.3], _pad: 0.0, color: camera_marker_color });
        }

        output
    }
}

impl RenderPass for DebugDrawPass {
    fn name(&self) -> &'static str {
        "DebugDraw"
    }

    fn prepare(&mut self, ctx: &helio_v3::PrepareContext) -> HelioResult<()> {
        let lines = self.build_frame_vertices();
        log::debug!("DebugDraw: {} verts", lines.len());
        self.pass.update_lines(ctx.queue, &lines);
        Ok(())
    }

    fn execute(&mut self, ctx: &mut helio_v3::PassContext) -> HelioResult<()> {
        log::debug!("DebugDrawPass execute (editor_mode={}, vertex_count={})", self.editor_mode, self.pass.vertex_count);

        // Editor grid draws into pre_aa buffer as a background layer before geometry resolves.
        // World-space debug lines draw into final target after all scene passes.
        let target_view = if self.editor_mode {
            ctx.frame.pre_aa.unwrap_or(ctx.target)
        } else {
            ctx.target
        };

        // Temporarily swap `ctx.target` for internal pass rendering.
        let previous_target = ctx.target;
        ctx.target = target_view;

        let count_before = self.pass.vertex_count;
        let res = self.pass.execute(ctx);

        // restore target (ctx is borrowed so reassign after draw)
        ctx.target = previous_target;

        log::debug!("DebugDrawPass executed, count={} err={:?}", count_before, res);
        res
    }
}

impl Renderer {
    pub fn new(device: Arc<wgpu::Device>, queue: Arc<wgpu::Queue>, config: RendererConfig) -> Self {
        let mut scene = Scene::new(device.clone(), queue.clone());
        scene.set_render_size(config.width, config.height);

        let debug_state = Arc::new(Mutex::new(DebugDrawState::default()));

        let debug_camera_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Debug Camera Buffer"),
            size: std::mem::size_of::<helio_pass_debug::DebugCameraUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let graph = build_default_graph(&device, &queue, &scene, config, debug_state.clone(), &debug_camera_buffer, true);

        // Depth buffer at INTERNAL resolution (all geometry passes render here).
        let (depth_texture, depth_view) = create_depth_resources(
            &device,
            config.internal_width(),
            config.internal_height()
        );
        // Full-res depth only needed when render_scale < 1.0 so post-upscale passes
        // (BillboardPass) have a depth attachment matching the native-res colour target.
        let (full_res_depth_texture, full_res_depth_view) = if config.render_scale < 1.0 {
            let (t, v) = create_depth_resources(&device, config.width, config.height);
            (Some(t), Some(v))
        } else {
            (None, None)
        };
        Self {
            device,
            queue,
            graph,
            graph_kind: GraphKind::Default,
            scene,
            depth_texture,
            depth_view,
            output_width: config.width,
            output_height: config.height,
            render_scale: config.render_scale,
            editor_camera_pos: glam::Vec3::ZERO,
            full_res_depth_texture,
            full_res_depth_view,
            surface_format: config.surface_format,
            debug_camera_buffer,
            ambient_color: [0.05, 0.05, 0.08],
            ambient_intensity: 1.0,
            clear_color: [0.02, 0.02, 0.03, 1.0],
            gi_config: config.gi_config,
            shadow_quality: config.shadow_quality,
            debug_mode: 0,
            debug_depth_test: true,
            editor_mode: false,
            debug_state,
            billboard_instances: Vec::new(),
            billboard_scratch: Vec::new(),
            editor_lights: HashMap::new(),
        }
    }

    /// Set GI configuration (RC radius, fade margin).
    pub fn set_gi_config(&mut self, gi_config: GiConfig) {
        self.gi_config = gi_config;
    }

    /// Get current GI configuration.
    pub fn gi_config(&self) -> GiConfig {
        self.gi_config
    }

    /// Set shadow quality (Low, Medium, High, Ultra). Rebuilds the render graph.
    pub fn set_shadow_quality(&mut self, quality: libhelio::ShadowQuality) {
        self.shadow_quality = quality;
        // Rebuild the default graph to apply the new shadow quality
        if matches!(self.graph_kind, GraphKind::Default) {
            let config = RendererConfig {
                width: self.output_width,
                height: self.output_height,
                surface_format: self.surface_format,
                gi_config: self.gi_config,
                shadow_quality: self.shadow_quality,
                debug_mode: self.debug_mode,
                render_scale: self.render_scale,
            };
            self.graph = build_default_graph(
                &self.device,
                &self.queue,
                &self.scene,
                config,
                self.debug_state.clone(),
                &self.debug_camera_buffer,
                self.debug_depth_test,
            );
        }
    }

    /// Set the debug visualisation mode (0=normal, 10=shadow heatmap, 11=light-space depth).
    pub fn set_debug_mode(&mut self, mode: u32) {
        self.debug_mode = mode;
        if matches!(self.graph_kind, GraphKind::Default) {
            let config = RendererConfig {
                width: self.output_width,
                height: self.output_height,
                surface_format: self.surface_format,
                gi_config: self.gi_config,
                shadow_quality: self.shadow_quality,
                debug_mode: self.debug_mode,
                render_scale: self.render_scale,
            };
            self.graph = build_default_graph(
                &self.device,
                &self.queue,
                &self.scene,
                config,
                self.debug_state.clone(),
                &self.debug_camera_buffer,
                self.debug_depth_test,
            );
        }
    }

    /// Enable or disable debug depth testing (geometry occlusion of debug lines).
    pub fn set_debug_depth_test(&mut self, enabled: bool) {
        self.debug_depth_test = enabled;
        if matches!(self.graph_kind, GraphKind::Default) {
            let config = RendererConfig {
                width: self.output_width,
                height: self.output_height,
                surface_format: self.surface_format,
                gi_config: self.gi_config,
                shadow_quality: self.shadow_quality,
                debug_mode: self.debug_mode,
                render_scale: self.render_scale,
            };
            self.graph = build_default_graph(
                &self.device,
                &self.queue,
                &self.scene,
                config,
                self.debug_state.clone(),
                &self.debug_camera_buffer,
                self.debug_depth_test,
            );
        }
    }

    /// Enable or disable editor mode.
    pub fn set_editor_mode(&mut self, enabled: bool) {
        self.editor_mode = enabled;
        if enabled {
            self.scene.show_group(GroupId::EDITOR);
        } else {
            self.scene.hide_group(GroupId::EDITOR);
        }
        if let Ok(mut s) = self.debug_state.lock() {
            s.editor_enabled = enabled;
        }
    }

    /// Get the current editor-mode state.
    pub fn is_editor_mode(&self) -> bool {
        self.editor_mode
    }

    /// Clear all per-frame debug geometry.
    pub fn debug_clear(&mut self) {
        if let Ok(mut s) = self.debug_state.lock() {
            s.user_lines.clear();
        }
    }

    /// Add a one-frame debug line to the debug renderer.
    pub fn debug_line(&mut self, from: [f32; 3], to: [f32; 3], color: [f32; 4]) {
        if let Ok(mut s) = self.debug_state.lock() {
            s.user_lines.push(DebugVertex { position: from, _pad: 0.0, color });
            s.user_lines.push(DebugVertex { position: to, _pad: 0.0, color });
        }
    }

    /// Add a one-frame circle (approx) to the debug renderer.
    pub fn debug_circle(&mut self, center: [f32; 3], radius: f32, color: [f32; 4], segments: u32) {
        if segments < 3 { return; }
        let (cx, cy, cz) = (center[0], center[1], center[2]);
        let step = std::f32::consts::TAU / segments as f32;
        let mut last = (cx + radius, cy, cz);
        for i in 1..=segments {
            let theta = i as f32 * step;
            let next = (cx + radius * theta.cos(), cy, cz + radius * theta.sin());
            self.debug_line([last.0, last.1, last.2], [next.0, next.1, next.2], color);
            last = next;
        }
    }

    /// Add a one-frame sphere approximation to the debug renderer.
    pub fn debug_sphere(&mut self, center: [f32; 3], radius: f32, color: [f32; 4], segments: u32) {
        if segments < 4 { return; }
        for plane in 0..3 {
            let mut prev = glam::Vec3::ZERO;
            for i in 0..=segments {
                let theta = i as f32 / segments as f32 * std::f32::consts::TAU;
                let pos = match plane {
                    0 => glam::Vec3::new(radius * theta.cos(), radius * theta.sin(), 0.0),
                    1 => glam::Vec3::new(radius * theta.cos(), 0.0, radius * theta.sin()),
                    _ => glam::Vec3::new(0.0, radius * theta.cos(), radius * theta.sin()),
                } + glam::Vec3::from(center);
                if i > 0 {
                    self.debug_line(prev.to_array(), pos.to_array(), color);
                }
                prev = pos;
            }
        }
    }

    /// Add a one-frame torus approximation.
    pub fn debug_torus(&mut self, center: [f32; 3], normal: [f32; 3], major_radius: f32, minor_radius: f32, color: [f32; 4], major_segments: u32, minor_segments: u32) {
        if major_segments < 3 || minor_segments < 3 { return; }
        let c = glam::Vec3::from(center);
        let n = glam::Vec3::from(normal).normalize_or_zero();
        let up = if n.abs_diff_eq(glam::Vec3::Y, 1e-6) { glam::Vec3::X } else { glam::Vec3::Y };
        let tangent = n.cross(up).normalize_or_zero();
        let bitangent = n.cross(tangent).normalize_or_zero();

        for j in 0..major_segments {
            let theta0 = 2.0 * std::f32::consts::TAU * (j as f32) / (major_segments as f32);
            let theta1 = 2.0 * std::f32::consts::TAU * ((j + 1) as f32) / (major_segments as f32);
            let center0 = c + (tangent * theta0.cos() + bitangent * theta0.sin()) * major_radius;
            let center1 = c + (tangent * theta1.cos() + bitangent * theta1.sin()) * major_radius;

            let mut pprev0 = center0 + (n * minor_radius);
            let mut pprev1 = center1 + (n * minor_radius);
            for i in 1..=minor_segments {
                let phi = 2.0 * std::f32::consts::TAU * (i as f32) / (minor_segments as f32);
                let offset = (n * phi.cos() + (tangent * theta0.cos() + bitangent * theta0.sin()) * phi.sin()).normalize_or_zero() * minor_radius;
                let cur0 = center0 + offset;
                let offset1 = (n * phi.cos() + (tangent * theta1.cos() + bitangent * theta1.sin()) * phi.sin()).normalize_or_zero() * minor_radius;
                let cur1 = center1 + offset1;

                self.debug_line(pprev0.to_array(), cur0.to_array(), color);
                self.debug_line(pprev1.to_array(), cur1.to_array(), color);
                self.debug_line(pprev0.to_array(), pprev1.to_array(), color);

                pprev0 = cur0;
                pprev1 = cur1;
            }
        }
    }

    /// Add a one-frame cylinder outline.
    pub fn debug_cylinder(&mut self, base_center: [f32; 3], axis: [f32; 3], height: f32, radius: f32, color: [f32; 4], segments: u32) {
        if segments < 3 { return; }
        let base = glam::Vec3::from(base_center);
        let dir = glam::Vec3::from(axis).normalize_or_zero();
        let top = base + dir * height;
        let up = if dir.abs_diff_eq(glam::Vec3::Y, 1e-5) { glam::Vec3::X } else { glam::Vec3::Y };
        let tangent = dir.cross(up).normalize_or_zero();
        let bitangent = dir.cross(tangent).normalize_or_zero();
        let mut prev_base = base + tangent * radius;
        let mut prev_top = top + tangent * radius;
        for i in 1..=segments {
            let theta = i as f32 / segments as f32 * std::f32::consts::TAU;
            let dir_circle = tangent * theta.cos() + bitangent * theta.sin();
            let cur_base = base + dir_circle * radius;
            let cur_top = top + dir_circle * radius;
            self.debug_line(prev_base.to_array(), cur_base.to_array(), color);
            self.debug_line(prev_top.to_array(), cur_top.to_array(), color);
            self.debug_line(prev_base.to_array(), prev_top.to_array(), color);
            prev_base = cur_base;
            prev_top = cur_top;
        }
    }

    /// Add a one-frame cone outline.
    pub fn debug_cone(&mut self, apex: [f32; 3], axis: [f32; 3], height: f32, base_radius: f32, color: [f32; 4], segments: u32) {
        if segments < 3 { return; }
        let apex_v = glam::Vec3::from(apex);
        let dir = glam::Vec3::from(axis).normalize_or_zero();
        let base = apex_v + dir * height;
        let up = if dir.abs_diff_eq(glam::Vec3::Y, 1e-5) { glam::Vec3::X } else { glam::Vec3::Y };
        let tangent = dir.cross(up).normalize_or_zero();
        let bitangent = dir.cross(tangent).normalize_or_zero();
        let mut prev = base + tangent * base_radius;
        for i in 1..=segments {
            let theta = i as f32 / segments as f32 * std::f32::consts::TAU;
            let cur = base + (tangent * theta.cos() + bitangent * theta.sin()) * base_radius;
            self.debug_line(prev.to_array(), cur.to_array(), color);
            self.debug_line(cur.to_array(), apex_v.to_array(), color);
            prev = cur;
        }
    }

    /// Add a one-frame frustum outline.
    pub fn debug_frustum(&mut self, origin: [f32; 3], forward: [f32; 3], up: [f32; 3], fov_y: f32, aspect: f32, near: f32, far: f32, color: [f32; 4]) {
        let o = glam::Vec3::from(origin);
        let fwd = glam::Vec3::from(forward).normalize_or_zero();
        let upv = glam::Vec3::from(up).normalize_or_zero();
        let rightv = fwd.cross(upv).normalize_or_zero();
        let n_center = o + fwd * near;
        let f_center = o + fwd * far;
        let nh = (fov_y*0.5).tan() * near;
        let nw = nh * aspect;
        let fh = (fov_y*0.5).tan() * far;
        let fw = fh * aspect;

        let n = [
            n_center + upv*nh - rightv*nw,
            n_center + upv*nh + rightv*nw,
            n_center - upv*nh + rightv*nw,
            n_center - upv*nh - rightv*nw,
        ];
        let f = [
            f_center + upv*fh - rightv*fw,
            f_center + upv*fh + rightv*fw,
            f_center - upv*fh + rightv*fw,
            f_center - upv*fh - rightv*fw,
        ];

        for i in 0..4 {
            self.debug_line(n[i].to_array(), n[(i+1)%4].to_array(), color);
            self.debug_line(f[i].to_array(), f[(i+1)%4].to_array(), color);
            self.debug_line(n[i].to_array(), f[i].to_array(), color);
        }
    }

    /// Get current shadow quality.
    pub fn shadow_quality(&self) -> libhelio::ShadowQuality {
        self.shadow_quality
    }

    pub fn scene(&self) -> &Scene {
        &self.scene
    }

    pub fn scene_mut(&mut self) -> &mut Scene {
        &mut self.scene
    }

    /// Returns a reference to the GPU camera uniform buffer.
    /// Useful for creating custom render passes that need to read camera data (e.g. SDF ray march).
    pub fn camera_buffer(&self) -> &wgpu::Buffer {
        self.scene.gpu_scene().camera.buffer()
    }

    pub fn mesh_buffers(&self) -> MeshBuffers<'_> {
        self.scene.mesh_buffers()
    }

    pub fn add_pass(&mut self, pass: Box<dyn RenderPass>) {
        self.graph.add_pass(pass);
    }

    pub fn set_render_size(&mut self, width: u32, height: u32) {
        self.output_width = width;
        self.output_height = height;
        self.scene.set_render_size(width, height);
        let config = RendererConfig {
            width,
            height,
            surface_format: self.surface_format,
            gi_config: self.gi_config,
            shadow_quality: self.shadow_quality,
            debug_mode: self.debug_mode,
            render_scale: self.render_scale,
        };
        let (depth_texture, depth_view) = create_depth_resources(
            &self.device,
            config.internal_width(),
            config.internal_height()
        );
        self.depth_texture = depth_texture;
        self.depth_view = depth_view;
        (self.full_res_depth_texture, self.full_res_depth_view) = if self.render_scale < 1.0 {
            let (t, v) = create_depth_resources(&self.device, width, height);
            (Some(t), Some(v))
        } else {
            (None, None)
        };
        match self.graph_kind {
            GraphKind::Default => {
                self.graph = build_default_graph(
                    &self.device,
                    &self.queue,
                    &self.scene,
                    config,
                    self.debug_state.clone(),
                    &self.debug_camera_buffer,
                    self.debug_depth_test,
                );
            }
            GraphKind::Simple => {
                self.graph = build_simple_graph(&self.device, &self.queue, self.surface_format);
            }
            GraphKind::Custom => {
                // User-provided graph: do not replace it.
            }
        }
    }

    /// Set render scale factor (0.25..=1.0) at runtime. Rebuilds the render graph.
    ///
    /// - `1.0` = native resolution (default)
    /// - `0.5` = half-resolution (4× fewer shaded pixels), upscaled 2× via TAA
    /// - `0.25` = quarter-resolution, upscaled 4× via TAA
    ///
    /// Lower values give a large GPU performance gain while TAA temporal accumulation
    /// reconstructs high-quality detail over multiple frames, similar to Unreal TSR.
    pub fn set_render_scale(&mut self, scale: f32) {
        self.render_scale = scale.clamp(0.25, 1.0);
        self.set_render_size(self.output_width, self.output_height);
    }

    /// Returns the current render scale factor.
    pub fn render_scale(&self) -> f32 {
        self.render_scale
    }

    pub fn set_clear_color(&mut self, color: [f32; 4]) {
        self.clear_color = color;
    }

    pub fn set_ambient(&mut self, color: [f32; 3], intensity: f32) {
        self.ambient_color = color;
        self.ambient_intensity = intensity;
    }

    /// Replace the active render graph. Use [`build_simple_graph`] or
    /// [`build_default_graph`](fn.build_default_graph.html) to construct one.
    /// The graph will NOT be automatically rebuilt on window resize.
    pub fn set_graph(&mut self, graph: RenderGraph) {
        self.graph = graph;
        self.graph_kind = GraphKind::Custom;
    }

    /// Convenience helper: switch to the simple single-cube graph.
    pub fn use_simple_graph(&mut self) {
        self.graph = build_simple_graph(&self.device, &self.queue, self.surface_format);
        self.graph_kind = GraphKind::Simple;
    }

    /// Convenience helper: switch back to the full deferred graph.
    pub fn use_default_graph(&mut self) {
        let config = RendererConfig {
            width: self.output_width,
            height: self.output_height,
            surface_format: self.surface_format,
            gi_config: self.gi_config,
            shadow_quality: self.shadow_quality,
            debug_mode: self.debug_mode,
            render_scale: self.render_scale,
        };
        self.graph = build_default_graph(
            &self.device,
            &self.queue,
            &self.scene,
            config,
            self.debug_state.clone(),
            &self.debug_camera_buffer,
            self.debug_depth_test,
        );
        self.graph_kind = GraphKind::Default;
    }

    pub fn insert_mesh(&mut self, mesh: MeshUpload) -> MeshId {
        self.scene.insert_mesh(mesh)
    }

    pub fn insert_texture(&mut self, texture: TextureUpload) -> SceneResult<crate::TextureId> {
        self.scene.insert_texture(texture)
    }

    pub fn insert_material(&mut self, material: crate::GpuMaterial) -> MaterialId {
        self.scene.insert_material(material)
    }

    pub fn insert_material_asset(&mut self, material: MaterialAsset) -> SceneResult<MaterialId> {
        self.scene.insert_material_asset(material)
    }

    pub fn update_material(
        &mut self,
        id: MaterialId,
        material: crate::GpuMaterial
    ) -> SceneResult<()> {
        self.scene.update_material(id, material)
    }

    pub fn update_material_asset(
        &mut self,
        id: MaterialId,
        material: MaterialAsset
    ) -> SceneResult<()> {
        self.scene.update_material_asset(id, material)
    }

    pub fn insert_light(&mut self, light: crate::GpuLight) -> LightId {
        let id = self.scene.insert_light(light);
        self.editor_lights.insert(id, light);
        id
    }

    pub fn update_light(&mut self, id: LightId, light: crate::GpuLight) -> SceneResult<()> {
        self.editor_lights.insert(id, light);
        self.scene.update_light(id, light)
    }

    pub fn remove_light(&mut self, id: LightId) -> SceneResult<()> {
        self.editor_lights.remove(&id);
        self.scene.remove_light(id)
    }

    // ── Group management ─────────────────────────────────────────────────────

    /// Hide all scene objects (and editor light icons) belonging to `group`.
    pub fn hide_group(&mut self, group: GroupId) {
        self.scene.hide_group(group);
    }

    /// Show all scene objects (and editor light icons) belonging to `group`.
    pub fn show_group(&mut self, group: GroupId) {
        self.scene.show_group(group);
    }

    /// Returns `true` if `group` is currently hidden.
    pub fn is_group_hidden(&self, group: GroupId) -> bool {
        self.scene.is_group_hidden(group)
    }

    /// Batch hide/show multiple groups at once.
    pub fn set_group_visibility(&mut self, mask: GroupMask, visible: bool) {
        self.scene.set_group_visibility(mask, visible);
    }

    /// Replace an object's group membership mask.
    pub fn set_object_groups(&mut self, id: ObjectId, mask: GroupMask) -> SceneResult<()> {
        self.scene.set_object_groups(id, mask)
    }

    /// Add one group to an object's membership mask.
    pub fn add_object_to_group(&mut self, id: ObjectId, group: GroupId) -> SceneResult<()> {
        self.scene.add_object_to_group(id, group)
    }

    /// Remove one group from an object's membership mask.
    pub fn remove_object_from_group(&mut self, id: ObjectId, group: GroupId) -> SceneResult<()> {
        self.scene.remove_object_from_group(id, group)
    }

    /// Apply a transform delta to every object in `group`.
    pub fn move_group(&mut self, group: GroupId, delta: glam::Mat4) {
        self.scene.move_group(group, delta);
    }

    /// Translate every object in `group` by `delta`.
    pub fn translate_group(&mut self, group: GroupId, delta: glam::Vec3) {
        self.scene.translate_group(group, delta);
    }

    pub fn insert_object(&mut self, desc: ObjectDescriptor) -> SceneResult<ObjectId> {
        self.scene.insert_object(desc)
    }

    /// Optimizes the scene layout for cache coherency and GPU instancing.
    ///
    /// See [`Scene::optimize_scene_layout`] for details.
    pub fn optimize_scene_layout(&mut self) {
        self.scene.optimize_scene_layout();
    }

    // ── Virtual geometry ──────────────────────────────────────────────────────

    /// Meshletise a high-resolution mesh and register it for GPU-driven rendering.
    /// Returns a `VirtualMeshId` to pass to `insert_virtual_object`.
    pub fn insert_virtual_mesh(&mut self, upload: VirtualMeshUpload) -> VirtualMeshId {
        self.scene.insert_virtual_mesh(upload)
    }

    pub fn remove_virtual_mesh(&mut self, id: VirtualMeshId) -> SceneResult<()> {
        self.scene.remove_virtual_mesh(id)
    }

    /// Place an instance of a virtual mesh into the scene.
    pub fn insert_virtual_object(
        &mut self,
        desc: VirtualObjectDescriptor
    ) -> SceneResult<VirtualObjectId> {
        self.scene.insert_virtual_object(desc)
    }

    pub fn update_virtual_object_transform(
        &mut self,
        id: VirtualObjectId,
        transform: glam::Mat4
    ) -> SceneResult<()> {
        self.scene.update_virtual_object_transform(id, transform)
    }

    pub fn remove_virtual_object(&mut self, id: VirtualObjectId) -> SceneResult<()> {
        self.scene.remove_virtual_object(id)
    }

    pub fn update_object_transform(
        &mut self,
        id: ObjectId,
        transform: glam::Mat4
    ) -> SceneResult<()> {
        self.scene.update_object_transform(id, transform)
    }

    pub fn update_object_bounds(&mut self, id: ObjectId, bounds: [f32; 4]) -> SceneResult<()> {
        self.scene.update_object_bounds(id, bounds)
    }

    /// Replace the billboard instance list for the next frame.
    ///
    /// Call once per frame (or whenever the billboard set changes).
    /// Billboards are camera-facing quads rendered with alpha blending on top of
    /// the opaque scene.  Pass an empty slice to disable billboards entirely.
    pub fn set_billboard_instances(
        &mut self,
        instances: &[helio_pass_billboard::BillboardInstance]
    ) {
        self.billboard_instances.clear();
        self.billboard_instances.extend_from_slice(instances);
    }

    pub fn render(&mut self, camera: &Camera, target: &wgpu::TextureView) -> HelioResult<()> {
        // Apply TAA sub-pixel jitter to the projection matrix so geometry shifts
        // by a different fraction of a pixel each frame.  TaaPass accumulates
        // these 16 Halton positions into a temporally anti-aliased image.
        let frame_idx = self.scene.gpu_scene().frame_count;
        let raw = HALTON_JITTER[(frame_idx % 16) as usize];
        let internal_w = (((self.output_width as f32) * self.render_scale).ceil() as u32).max(1);
        let internal_h = (((self.output_height as f32) * self.render_scale).ceil() as u32).max(1);
        let jx = ((raw[0] - 0.5) * 2.0) / (internal_w as f32);
        let jy = ((raw[1] - 0.5) * 2.0) / (internal_h as f32);
        let jitter_mat = glam::Mat4::from_translation(glam::Vec3::new(jx, jy, 0.0));
        let jittered_m = jitter_mat * camera.proj * camera.view;
        let col = jittered_m.to_cols_array();
        // Use jittered view_proj at the same time as geometry depth buffer in this frame,
        // so debug lines can be occluded by geometry.
        let debug_camera_uniform = DebugCameraUniform {
            view_proj: [
                [col[0], col[1], col[2], col[3]],
                [col[4], col[5], col[6], col[7]],
                [col[8], col[9], col[10], col[11]],
                [col[12], col[13], col[14], col[15]],
            ],
        };
        self.queue.write_buffer(
            &self.debug_camera_buffer,
            0,
            bytemuck::bytes_of(&debug_camera_uniform),
        );

        let mut jittered_camera = *camera;
        jittered_camera.proj = jitter_mat * camera.proj;
        jittered_camera.jitter = [jx, jy];
        self.scene.update_camera(jittered_camera);
        self.scene.flush();

        // Compose final billboard list for this frame:
        //   1. User-supplied instances (set via set_billboard_instances)
        //   2. Auto editor-light icons when GroupId::EDITOR is visible
        self.billboard_scratch.clear();
        self.billboard_scratch.extend_from_slice(&self.billboard_instances);
        if !self.scene.is_group_hidden(GroupId::EDITOR) {
            for (_, light) in &self.editor_lights {
                let [x, y, z, _] = light.position_range;
                let [r, g, b, _] = light.color_intensity;
                self.billboard_scratch.push(helio_pass_billboard::BillboardInstance {
                    world_pos: [x, y, z, 0.0],
                    // scale_flags.z = 0.0 → world-space size: the icon is scale.xy metres
                    // wide/tall regardless of camera distance (normal perspective applies).
                    scale_flags: [0.25, 0.25, 0.0, 0.0],
                    color: [r, g, b, 1.0],
                });
            }
        }

        let mut texture_views = ArrayVec::<&wgpu::TextureView, MAX_TEXTURES>::new();
        let mut samplers = ArrayVec::<&wgpu::Sampler, MAX_TEXTURES>::new();
        for slot in 0..MAX_TEXTURES {
            texture_views.push(self.scene.texture_view_for_slot(slot));
            samplers.push(self.scene.texture_sampler_for_slot(slot));
        }

        let mesh_buffers = self.scene.mesh_buffers();
        // Compute RC bounds (AAA dual-tier GI: RC near, ambient far)
        let cam_pos = camera.position; // Camera.position is a field (Vec3), not a method
        if let Ok(mut state) = self.debug_state.lock() {
            state.camera_position = cam_pos;
        }
        let rc_radius = self.gi_config.rc_radius;
        let rc_min = [cam_pos.x - rc_radius, cam_pos.y - rc_radius, cam_pos.z - rc_radius];
        let rc_max = [cam_pos.x + rc_radius, cam_pos.y + rc_radius, cam_pos.z + rc_radius];

        let frame_resources = libhelio::FrameResources {
            gbuffer: None,
            shadow_atlas: None,
            shadow_sampler: None,
            hiz: None,
            hiz_sampler: None,
            sky_lut: None,
            sky_lut_sampler: None,
            ssao: None,
            pre_aa: None,
            main_scene: Some(libhelio::MainSceneResources {
                mesh_buffers: libhelio::MeshBuffers {
                    vertices: mesh_buffers.vertices,
                    indices: mesh_buffers.indices,
                },
                material_textures: libhelio::MaterialTextureBindings {
                    material_textures: self.scene.material_texture_buffer(),
                    texture_views: texture_views.as_slice(),
                    samplers: samplers.as_slice(),
                    version: self.scene.texture_binding_version(),
                },
                clear_color: self.clear_color,
                ambient_color: self.ambient_color,
                ambient_intensity: self.ambient_intensity,
                rc_world_min: rc_min,
                rc_world_max: rc_max,
            }),
            tile_light_lists: None,
            tile_light_counts: None,
            full_res_depth: self.full_res_depth_view.as_ref().map(|v| v as &wgpu::TextureView),
            sky: libhelio::SkyContext::default(),
            billboards: if self.billboard_scratch.is_empty() {
                None
            } else {
                Some(libhelio::BillboardFrameData {
                    instances: bytemuck::cast_slice(&self.billboard_scratch),
                    count: self.billboard_scratch.len() as u32,
                })
            },
            vg: self.scene.vg_frame_data(),
        };

        self.graph.execute_with_frame_resources(
            self.scene.gpu_scene(),
            target,
            &self.depth_view,
            &frame_resources,
        )?;
        // frame_resources is Copy, no need to drop
        drop(texture_views);
        drop(samplers);
        self.scene.advance_frame();
        Ok(())
    }
}

fn build_default_graph(
    device: &Arc<wgpu::Device>,
    queue: &Arc<wgpu::Queue>,
    scene: &Scene,
    config: RendererConfig,
    debug_state: Arc<Mutex<DebugDrawState>>,
    debug_camera_buf: &wgpu::Buffer,
    debug_depth_test: bool,
) -> RenderGraph {
    let gpu_scene = scene.gpu_scene();
    let mut graph = RenderGraph::new(device, queue);

    let camera_buf = gpu_scene.camera.buffer();

    // Build the Hi-Z pass first so we can clone its Arc views for OcclusionCullPass.
    // HiZBuildPass is added to the graph AFTER DepthPrepass (see below).
    let hiz_pass = HiZBuildPass::new(device, config.internal_width(), config.internal_height());
    let hiz_view = Arc::clone(&hiz_pass.hiz_view);
    let hiz_sampler = Arc::clone(&hiz_pass.hiz_sampler);

    // 1. ShadowMatrixPass — GPU compute: writes correct view-projection matrices for
    //    every shadow face into the shadow_matrices storage buffer.
    //    Scene::flush() pre-sizes that buffer to N_lights*6 entries so shadow_count
    //    is nonzero and this binding covers the full range.
    let shadow_dirty_buf = device.create_buffer(
        &(wgpu::BufferDescriptor {
            label: Some("Shadow Dirty Flags"),
            size: 64,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        })
    );
    let shadow_hashes_buf = device.create_buffer(
        &(wgpu::BufferDescriptor {
            label: Some("Shadow Hashes"),
            size: 64,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        })
    );
    graph.add_pass(
        Box::new(
            ShadowMatrixPass::new(
                device,
                gpu_scene.lights.buffer(),
                gpu_scene.shadow_matrices.buffer(),
                camera_buf,
                &shadow_dirty_buf,
                &shadow_hashes_buf
            )
        )
    );

    // 2. ShadowPass — renders geometry into the shadow atlas (one face per entry in shadow_matrices)
    // publish()es shadow_atlas + shadow_sampler into FrameResources for DeferredLightPass
    graph.add_pass(Box::new(ShadowPass::new(device)));

    // 3. SkyLutPass — generates atmospheric sky lookup texture
    // Publishes "sky_lut" resource for SkyPass to consume
    let sky_lut_pass = SkyLutPass::new(device, camera_buf);
    let sky_lut_view = sky_lut_pass.sky_lut_view.clone();
    graph.add_pass(Box::new(sky_lut_pass));

    // 3a. SkyPass — full-screen sky dome into pre_aa before DeferredLight.
    graph.add_pass(Box::new(SkyPass::new(
        device,
        camera_buf,
        &sky_lut_view,
        config.internal_width(),
        config.internal_height(),
        config.surface_format,
    )));

    // 3a.5. Editor-grid debug draw pass — render world grid lines before geometry.
    graph.add_pass(Box::new(DebugDrawPass::new(
        device,
        debug_camera_buf,
        config.surface_format,
        debug_state.clone(),
        false, // no depth test for editor grid (non-occluded)
        true,  // editor mode (grid only)
    )));

    // 3b. OcclusionCullPass — temporal Hi-Z culling (reads PREVIOUS frame's pyramid).
    //     Must run BEFORE DepthPrepass so depth isn't overwritten yet.
    //     Frame 0 is a no-op (the pass guards internally with ctx.frame_num == 0).
    graph.add_pass(Box::new(OcclusionCullPass::new(device, hiz_view, hiz_sampler)));

    // 3. DepthPrepassPass — early depth pass for better GPU culling
    graph.add_pass(Box::new(DepthPrepassPass::new(device, wgpu::TextureFormat::Depth32Float)));

    // 3c. HiZBuildPass — builds the Hi-Z pyramid from this frame's depth.
    //     Runs AFTER DepthPrepass so the pyramid is ready for NEXT frame's
    //     OcclusionCullPass (temporal).
    graph.add_pass(Box::new(hiz_pass));

    // 3d. LightCullPass — GPU compute: builds per-tile light lists for
    //     tiled Forward+ shading in DeferredLightPass.
    graph.add_pass(
        Box::new(LightCullPass::new(device, config.internal_width(), config.internal_height()))
    );

    // 4. GBufferPass — fills G-buffer (albedo, normal, ORM, emissive)
    graph.add_pass(
        Box::new(GBufferPass::new(device, config.internal_width(), config.internal_height()))
    );

    // 4b. VirtualGeometryPass — GPU-driven meshlet cull + draw into the same GBuffer
    let mut vg_pass = VirtualGeometryPass::new(device, camera_buf);
    vg_pass.debug_mode = config.debug_mode;
    graph.add_pass(Box::new(vg_pass));

    // TODO: Add SsaoPass — needs resource declaration support

    // 5. DeferredLightPass — lighting pass (reads G-buffer, shadow maps)
    // With automatic resource management, this will write to "pre_aa" if declared,
    // or directly to surface if no post-processing passes are present
    let mut deferred_light_pass = DeferredLightPass::new(
        device,
        queue,
        camera_buf,
        config.internal_width(),
        config.internal_height(),
        config.surface_format
    );
    deferred_light_pass.set_shadow_quality(config.shadow_quality, queue);
    deferred_light_pass.debug_mode = config.debug_mode;
    graph.add_pass(Box::new(deferred_light_pass));

    // 5b. TaaPass — temporal anti-aliasing: resolves pre_aa into history-blended
    //     output, then blits the result to ctx.target.
    //
    // 6. BillboardPass — camera-facing instanced quads (rendered into pre_aa so that
    //     they are depth-tested against geometry and participate in the TAA pipeline).
    let spotlight = image
        ::load_from_memory(SPOTLIGHT_PNG)
        .unwrap_or_else(|_| image::DynamicImage::new_rgba8(1, 1))
        .into_rgba8();
    let (sw, sh) = spotlight.dimensions();
    let mut billboard_pass = BillboardPass::new_with_sprite_rgba(
        device,
        queue,
        camera_buf,
        config.surface_format,
        spotlight.as_raw(),
        sw,
        sh,
    );
    billboard_pass.set_occluded_by_geometry(true);
    graph.add_pass(Box::new(billboard_pass));

    graph.add_pass(
        Box::new(
            TaaPass::new(
                device,
                config.internal_width(),
                config.internal_height(),
                config.width,
                config.height,
                config.surface_format,
            )
        )
    );

    // TODO: Enable these passes once they declare resources properly:
    // - SkyPass (reads "sky_lut", writes to scene color)
    // - TransparentPass (reads scene depth, writes to scene color with blending)
    // - FxaaPass (reads "pre_aa", writes to final surface)

    // 7. World-space debug draw pass (after main geometry and billboards).
    //  World-space lines are rendered without depth test by default (no occlusion).
    graph.add_pass(Box::new(DebugDrawPass::new(
        device,
        debug_camera_buf,
        config.surface_format,
        debug_state.clone(),
        false, // worldspace debug lines with no occlusion
        false, // worldspace debug mode
    )));

    // Initialize transient textures from pass declarations
    graph.set_render_size(config.width, config.height);

    graph
}

/// A minimal graph with a single geometry-only pass that always renders one
/// hardcoded cube at full brightness. Useful as a sanity-check baseline.
pub fn build_simple_graph(
    device: &Arc<wgpu::Device>,
    queue: &Arc<wgpu::Queue>,
    surface_format: wgpu::TextureFormat
) -> RenderGraph {
    let mut graph = RenderGraph::new(device, queue);
    graph.add_pass(Box::new(SimpleCubePass::new(device, surface_format)));
    graph
}

fn create_depth_resources(
    device: &wgpu::Device,
    width: u32,
    height: u32
) -> (wgpu::Texture, wgpu::TextureView) {
    let texture = device.create_texture(
        &(wgpu::TextureDescriptor {
            label: Some("Helio Depth Texture"),
            size: wgpu::Extent3d {
                width: width.max(1),
                height: height.max(1),
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        })
    );
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    (texture, view)
}
