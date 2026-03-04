//! Main renderer implementation

use crate::resources::ResourceManager;
use crate::features::{FeatureRegistry, FeatureContext, PrepareContext};
use crate::pipeline::{PipelineCache, PipelineVariant};
use crate::graph::{RenderGraph, GraphContext};
use crate::passes::{GeometryPass, SkyPass, SkyLutPass, SKY_LUT_W, SKY_LUT_H, SKY_LUT_FORMAT, ShadowCullLight, GBufferPass, GBufferTargets, DeferredLightingPass};
use crate::mesh::{GpuMesh, DrawCall};
use crate::camera::Camera;
use crate::scene::Scene;
use crate::features::lighting::{GpuLight, MAX_LIGHTS};
use crate::features::BillboardsFeature;
use crate::material::{Material, GpuMaterial, MaterialUniform, DefaultMaterialViews, build_gpu_material};
use crate::profiler::{GpuProfiler, PassTiming};
use crate::Result;
use std::sync::{Arc, Mutex, atomic::{AtomicU32, Ordering}};
use wgpu::util::DeviceExt;
use bytemuck::Zeroable;
use glam::{Mat4, Vec3, Vec4Swizzles};

/// Main renderer configuration
pub struct RendererConfig {
    pub width: u32,
    pub height: u32,
    pub surface_format: wgpu::TextureFormat,
    pub features: FeatureRegistry,
}

/// Globals uniform data
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct GlobalsUniform {
    frame: u32,
    delta_time: f32,
    light_count: u32,
    ambient_intensity: f32,
    ambient_color: [f32; 4],  // w unused, ensures alignment
    rc_world_min: [f32; 4],   // xyz = RC probe grid world AABB min, w unused
    rc_world_max: [f32; 4],   // xyz = RC probe grid world AABB max, w unused
    /// View-space distance at each CSM cascade boundary (4 cascades → 3 splits + sentinel).
    csm_splits: [f32; 4],
}

/// Sky uniform data – 112 bytes, must exactly match SkyUniforms in sky.wgsl
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct SkyUniform {
    sun_direction:      [f32; 3],  // offset   0 (12 bytes)
    sun_intensity:      f32,       // offset  12
    rayleigh_scatter:   [f32; 3],  // offset  16
    rayleigh_h_scale:   f32,       // offset  28
    mie_scatter:        f32,       // offset  32
    mie_h_scale:        f32,       // offset  36
    mie_g:              f32,       // offset  40
    sun_disk_cos:       f32,       // offset  44
    earth_radius:       f32,       // offset  48
    atm_radius:         f32,       // offset  52
    exposure:           f32,       // offset  56
    clouds_enabled:     u32,       // offset  60
    cloud_coverage:     f32,       // offset  64
    cloud_density:      f32,       // offset  68
    cloud_base:         f32,       // offset  72
    cloud_top:          f32,       // offset  76
    cloud_wind_x:       f32,       // offset  80
    cloud_wind_z:       f32,       // offset  84
    cloud_speed:        f32,       // offset  88
    time_sky:           f32,       // offset  92
    skylight_intensity: f32,       // offset  96
    _pad0:              f32,       // offset 100
    _pad1:              f32,       // offset 104
    _pad2:              f32,       // offset 108 → total 112 (multiple of 16)
}

/// GPU shadow light-space matrix (must match WGSL LightMatrix struct = 64 bytes)
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuShadowMatrix {
    mat: [f32; 16],
}

/// Compute 6 cube-face view-proj matrices for a point light (±X, ±Y, ±Z).
fn compute_point_light_matrices(position: [f32; 3], range: f32) -> [Mat4; 6] {
    let pos = Vec3::from(position);
    let proj = Mat4::perspective_rh(std::f32::consts::FRAC_PI_2, 1.0, 0.05, range.max(0.1));
    let views = [
        Mat4::look_at_rh(pos, pos + Vec3::X,  -Vec3::Y), // +X
        Mat4::look_at_rh(pos, pos - Vec3::X,  -Vec3::Y), // -X
        Mat4::look_at_rh(pos, pos + Vec3::Y,   Vec3::Z), // +Y
        Mat4::look_at_rh(pos, pos - Vec3::Y,  -Vec3::Z), // -Y
        Mat4::look_at_rh(pos, pos + Vec3::Z,  -Vec3::Y), // +Z
        Mat4::look_at_rh(pos, pos - Vec3::Z,  -Vec3::Y), // -Z
    ];
    views.map(|view| proj * view)
}

/// World-space distance thresholds for the 4 CSM cascade boundaries.
/// Cascade i covers [CSM_SPLITS[i-1], CSM_SPLITS[i]] (cascade 0 starts at 0).
const CSM_SPLITS: [f32; 4] = [16.0, 80.0, 300.0, 1400.0];

// Deferred clustered-lighting configuration.
const CLUSTER_TILE_SIZE_PX: u32 = 16;
const CLUSTER_Z_BINS: u32 = 16;
const CLUSTER_NEAR: f32 = 0.1;
const CLUSTER_FAR: f32 = 3000.0;
const MAX_CLUSTERS: u32 = 262_144;
const MAX_LIGHTS_PER_CLUSTER: u32 = 128;

/// Compute 4 cascaded ortho light-space matrices for a directional light.
///
/// Uses **sphere-fit + texel snapping** — the standard "stable CSM" algorithm
/// used by UE4, Unity HDRP, and id Tech 7:
///
/// 1. Fit a bounding **sphere** (not AABB) to each camera-frustum slice.
///    A sphere is rotation-invariant, so the ortho-box dimensions never change
///    as the camera turns → zero shadow swimming.
/// 2. **Snap** the light-view origin to shadow-map texel boundaries so the
///    projected shadow never crawls as the camera translates.
/// 3. Pull the light camera back by `SCENE_DEPTH` and extend the far plane by
///    the same amount so off-screen casters (ceilings, terrain…) always cast.
///
/// Slots 0-3 hold the four cascade matrices; slots 4-5 are identity (reserved
/// for point-light cube faces which are not used for directional lights).
fn compute_directional_cascades(
    cam_pos: Vec3,
    view_proj_inv: Mat4,
    direction: [f32; 3],
) -> [Mat4; 4] {
    /// Maximum world-space distance from any frustum centre that shadow
    /// casters are guaranteed to be pulled in from.
    const SCENE_DEPTH: f32 = 4000.0;
    /// Shadow-atlas resolution per cascade (must match ShadowsFeature atlas_size).
    const ATLAS_TEXELS: f32 = 2048.0;

    let dir = Vec3::from(direction).normalize();
    let up  = if dir.dot(Vec3::Y).abs() > 0.99 { Vec3::Z } else { Vec3::Y };

    // Un-project 8 NDC corners to world space (wgpu depth: 0 = near, 1 = far)
    let ndc: [[f32; 4]; 8] = [
        [-1.0,-1.0, 0.0, 1.0], [1.0,-1.0, 0.0, 1.0],
        [-1.0, 1.0, 0.0, 1.0], [1.0, 1.0, 0.0, 1.0],
        [-1.0,-1.0, 1.0, 1.0], [1.0,-1.0, 1.0, 1.0],
        [-1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0],
    ];
    let world: Vec<Vec3> = ndc.iter().map(|c| {
        let v = view_proj_inv * glam::Vec4::from(*c);
        v.xyz() / v.w
    }).collect();
    // world[0..4] = near plane corners, world[4..8] = far plane corners

    let near_dist: f32 = world[..4].iter().map(|c| (*c - cam_pos).length()).sum::<f32>() / 4.0;
    let far_dist:  f32 = world[4..].iter().map(|c| (*c - cam_pos).length()).sum::<f32>() / 4.0;
    let depth = (far_dist - near_dist).max(1.0);

    let prev_d = [0.0_f32, CSM_SPLITS[0], CSM_SPLITS[1], CSM_SPLITS[2]];
    let mut matrices = [Mat4::IDENTITY; 4];

    for i in 0..4 {
        let t0 = ((prev_d[i]     - near_dist) / depth).clamp(0.0, 1.0);
        let t1 = ((CSM_SPLITS[i] - near_dist) / depth).clamp(0.0, 1.0);

        // 8 world-space corners of this frustum slice
        let mut cc = [Vec3::ZERO; 8];
        for j in 0..4 {
            cc[j * 2]     = world[j].lerp(world[j + 4], t0);
            cc[j * 2 + 1] = world[j].lerp(world[j + 4], t1);
        }

        // ── Step 1: bounding sphere of the 8 corners ──────────────────────────
        // Centre = centroid; radius = max distance from centre to any corner.
        // A sphere is rotation-invariant → ortho extents never change with yaw/pitch.
        let centroid = cc.iter().copied().fold(Vec3::ZERO, |a, b| a + b) / 8.0;
        let radius   = cc.iter().map(|c| (*c - centroid).length()).fold(0.0_f32, f32::max);
        // Round radius up to the nearest texel to eliminate sub-texel size jitter
        let texel_size  = (2.0 * radius) / ATLAS_TEXELS;
        let radius_snap = (radius / texel_size).ceil() * texel_size;

        // ── Step 2: texel-snapped light-view origin ────────────────────────────
        // Project centroid onto the light's view plane (XY), then quantise to
        // integer texel offsets so the shadow grid never sub-texel-crawls.
        let light_view_raw = Mat4::look_at_rh(centroid - dir * SCENE_DEPTH, centroid, up);
        let centroid_ls    = light_view_raw.transform_point3(centroid);
        let snap           = texel_size;
        let snapped_x      = (centroid_ls.x / snap).round() * snap;
        let snapped_y      = (centroid_ls.y / snap).round() * snap;

        // Reconstruct the right/up axes of the light view to apply the snap in world space
        let right_ws = light_view_raw.row(0).truncate().normalize();
        let up_ws    = light_view_raw.row(1).truncate().normalize();
        let snap_offset = right_ws * (snapped_x - centroid_ls.x)
                        + up_ws   * (snapped_y - centroid_ls.y);
        let stable_centroid = centroid + snap_offset;

        // Final light view: look from far behind the stable centroid
        let light_view = Mat4::look_at_rh(
            stable_centroid - dir * SCENE_DEPTH,
            stable_centroid,
            up,
        );

        // ── Step 3: ortho projection from the sphere radius ────────────────────
        // Z: near = 0.1 (camera is SCENE_DEPTH behind centroid),
        //    far  = SCENE_DEPTH * 2 (covers SCENE_DEPTH in front of centroid).
        // Any caster within SCENE_DEPTH of the scene is guaranteed to cast.
        let proj = Mat4::orthographic_rh(
            -radius_snap, radius_snap,
            -radius_snap, radius_snap,
            0.1, SCENE_DEPTH * 2.0,
        );
        matrices[i] = proj * light_view;
    }

    matrices
}

/// Compute the light-space matrix for a spot light (perspective projection).
fn compute_spot_matrix(position: [f32; 3], direction: [f32; 3], range: f32, outer_angle: f32) -> Mat4 {
    let pos = Vec3::from(position);
    let dir = Vec3::from(direction).normalize();
    let up = if dir.dot(Vec3::Y).abs() > 0.99 { Vec3::Z } else { Vec3::Y };
    let view = Mat4::look_at_rh(pos, pos + dir, up);
    let fov = (outer_angle * 2.0).clamp(std::f32::consts::FRAC_PI_4, std::f32::consts::PI - 0.01);
    let proj = Mat4::perspective_rh(fov, 1.0, 0.05, range.max(0.1));
    proj * view
}

/// Create a Depth32Float texture + two views (write + depth-only-sample) at the given resolution.
/// Both RENDER_ATTACHMENT and TEXTURE_BINDING are set so the deferred lighting pass can read depth.
fn create_depth_texture(device: &wgpu::Device, width: u32, height: u32) -> (wgpu::Texture, wgpu::TextureView, wgpu::TextureView) {
    let tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("Depth Texture"),
        size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Depth32Float,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    // Full-aspect view for render pass attachment
    let write_view = tex.create_view(&wgpu::TextureViewDescriptor::default());
    // Depth-only view for texture binding in the deferred lighting bind group
    let sample_view = tex.create_view(&wgpu::TextureViewDescriptor {
        aspect: wgpu::TextureAspect::DepthOnly,
        ..Default::default()
    });
    (tex, write_view, sample_view)
}

/// Create all four G-buffer textures and return their views packaged as `GBufferTargets`.
fn create_gbuffer_textures(
    device: &wgpu::Device,
    width: u32,
    height: u32,
) -> (wgpu::Texture, wgpu::Texture, wgpu::Texture, wgpu::Texture, GBufferTargets) {
    let make = |label: &str, format: wgpu::TextureFormat| {
        device.create_texture(&wgpu::TextureDescriptor {
            label: Some(label),
            size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        })
    };
    let albedo_tex  = make("GBuf Albedo",   wgpu::TextureFormat::Rgba8Unorm);
    let normal_tex  = make("GBuf Normal",   wgpu::TextureFormat::Rgba16Float);
    let orm_tex     = make("GBuf ORM",      wgpu::TextureFormat::Rgba8Unorm);
    let emissive_tex = make("GBuf Emissive", wgpu::TextureFormat::Rgba16Float);
    let albedo_view   = albedo_tex.create_view(&Default::default());
    let normal_view   = normal_tex.create_view(&Default::default());
    let orm_view      = orm_tex.create_view(&Default::default());
    let emissive_view = emissive_tex.create_view(&Default::default());
    let targets = GBufferTargets { albedo_view, normal_view, orm_view, emissive_view };
    (albedo_tex, normal_tex, orm_tex, emissive_tex, targets)
}

/// Build the G-buffer read bind group used by the deferred lighting pass (group 1).
fn create_gbuffer_bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    targets: &GBufferTargets,
    depth_sample_view: &wgpu::TextureView,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("GBuffer Read Bind Group"),
        layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&targets.albedo_view) },
            wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&targets.normal_view) },
            wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(&targets.orm_view) },
            wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(&targets.emissive_view) },
            wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::TextureView(depth_sample_view) },
        ],
    })
}

/// Approximate sky zenith ambient colour from sun elevation (-1=night, 0=horizon, 1=zenith).
fn estimate_sky_ambient(sun_elev: f32, rayleigh: &[f32; 3]) -> [f32; 3] {
    if sun_elev < -0.05 {
        let t = ((-sun_elev - 0.05) / 0.95).clamp(0.0, 1.0);
        lerp3([0.04, 0.06, 0.15], [0.01, 0.01, 0.02], t)
    } else if sun_elev < 0.15 {
        let t = ((sun_elev + 0.05) / 0.2).clamp(0.0, 1.0);
        lerp3([0.04, 0.06, 0.15], [0.55, 0.38, 0.20], t)
    } else {
        let t = ((sun_elev - 0.15) / 0.85).clamp(0.0, 1.0);
        let day_blue = [rayleigh[0] * 8.0, rayleigh[1] * 5.5, rayleigh[2] * 3.5];
        let noon = [day_blue[0].min(0.7), day_blue[1].min(0.85), day_blue[2].min(1.0)];
        lerp3([0.55, 0.38, 0.20], noon, t)
    }
}

fn lerp3(a: [f32; 3], b: [f32; 3], t: f32) -> [f32; 3] {
    [a[0]+(b[0]-a[0])*t, a[1]+(b[1]-a[1])*t, a[2]+(b[2]-a[2])*t]
}

/// Main renderer
pub struct Renderer {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,

    resources: ResourceManager,
    graph: RenderGraph,
    pipelines: PipelineCache,
    features: FeatureRegistry,

    // Uniform buffers
    camera_buffer: wgpu::Buffer,
    globals_buffer: wgpu::Buffer,

    // Bind groups
    global_bind_group: wgpu::BindGroup,
    lighting_bind_group: Arc<wgpu::BindGroup>,
    lighting_layout: Arc<wgpu::BindGroupLayout>,
    default_material_bind_group: Arc<wgpu::BindGroup>,

    // Default 1×1 texture views + sampler shared by all materials
    default_material_views: DefaultMaterialViews,

    // Draw list (shared with GeometryPass)
    draw_list: Arc<Mutex<Vec<DrawCall>>>,

    // Light buffer for scene writes
    light_buffer: Arc<wgpu::Buffer>,
    light_buffer_capacity_lights: u32,
    // Deferred clustered-lighting indirection buffers
    cluster_light_offsets_buffer: Arc<wgpu::Buffer>,
    cluster_light_indices_buffer: Arc<wgpu::Buffer>,
    lighting_shadow_view: Arc<wgpu::TextureView>,
    lighting_shadow_sampler: Arc<wgpu::Sampler>,
    lighting_env_cube_view: Arc<wgpu::TextureView>,
    lighting_rc_view: Arc<wgpu::TextureView>,
    lighting_env_sampler: Arc<wgpu::Sampler>,
    // Shadow light-space matrix buffer (shared with ShadowPass)
    shadow_matrix_buffer: Arc<wgpu::Buffer>,
    // Shared light count for ShadowPass (updated each frame before graph exec)
    light_count_arc: Arc<AtomicU32>,
    // Per-light active face counts: 6=point, 4=directional, 1=spot
    light_face_counts: Arc<std::sync::Mutex<Vec<u8>>>,
    // Per-light position/range/type for ShadowPass draw-call culling
    shadow_cull_lights: Arc<std::sync::Mutex<Vec<ShadowCullLight>>>,
    // Current scene ambient (updated by render_scene)
    scene_ambient_color: [f32; 3],
    scene_ambient_intensity: f32,
    scene_light_count: u32,
    scene_sky_color: [f32; 3],
    scene_has_sky: bool,
    /// CSM cascade split distances uploaded each frame to GlobalsUniform.
    scene_csm_splits: [f32; 4],
    // RC world bounds (set from RadianceCascadesFeature, zeroed if disabled)
    rc_world_min: [f32; 3],
    rc_world_max: [f32; 3],

    // Sky pass resources
    sky_uniform_buffer: wgpu::Buffer,
    sky_bind_group: Arc<wgpu::BindGroup>,

    // Depth buffer (Depth32Float, recreated on resize)
    depth_texture:      wgpu::Texture,
    depth_view:         wgpu::TextureView,
    /// Depth-only view (DepthOnly aspect) bound into the G-buffer read bind group.
    depth_sample_view:  wgpu::TextureView,

    // ── Deferred G-buffer ──────────────────────────────────────────────────
    gbuf_albedo_texture:   wgpu::Texture,
    gbuf_normal_texture:   wgpu::Texture,
    gbuf_orm_texture:      wgpu::Texture,
    gbuf_emissive_texture: wgpu::Texture,
    /// Shared with GBufferPass.  Swapped on resize.
    gbuffer_targets: Arc<Mutex<GBufferTargets>>,
    /// Shared with DeferredLightingPass.  Swapped on resize.
    deferred_bg: Arc<Mutex<Arc<wgpu::BindGroup>>>,

    // Frame state
    frame_count: u64,
    width: u32,
    height: u32,
    /// Tracks the start time of the previous frame for frame-to-frame latency measurement.
    last_frame_start: Option<std::time::Instant>,
    /// Tracks the end time of the previous frame for frame-to-frame latency measurement.
    last_frame_end: Option<std::time::Instant>,

    /// GPU + CPU pass-level profiler.  `None` when TIMESTAMP_QUERY is unavailable.
    profiler: Option<GpuProfiler>,

    /// When true, GPU timing stats are printed to stderr every frame (toggled by `debug_key_pressed`).
    debug_printout: bool,
}

impl Renderer {
    /// Create a new renderer
    pub fn new(
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
        config: RendererConfig,
    ) -> Result<Self> {
        log::info!("Creating Helio Render V2");
        log::info!("  Surface format: {:?}", config.surface_format);
        log::info!("  Resolution: {}x{}", config.width, config.height);

        let mut resources = ResourceManager::new(device.clone());
        let lighting_layout = resources.bind_group_layouts.lighting.clone();
        let bind_group_layouts = Arc::new(resources.bind_group_layouts.clone());
        let mut pipelines = PipelineCache::new(device.clone(), bind_group_layouts.clone(), config.surface_format);
        let mut graph = RenderGraph::new();
        let mut features = config.features;

        // ── Uniform buffers ──────────────────────────────────────────────────
        let camera_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Camera Uniform Buffer"),
            size: std::mem::size_of::<Camera>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let globals_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Globals Uniform Buffer"),
            size: std::mem::size_of::<GlobalsUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let global_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Global Bind Group"),
            layout: &resources.bind_group_layouts.global,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: camera_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: globals_buffer.as_entire_binding() },
            ],
        });

        // ── Geometry pass — must be added to graph BEFORE features so it ────────
        // ── executes first (billboard/post passes render on top).           ────────
        let draw_list: Arc<Mutex<Vec<DrawCall>>> = Arc::new(Mutex::new(Vec::new()));

        // Shadow matrix buffer: 16 lights × 6 faces × mat4x4<f32> = 6144 bytes
        let shadow_matrix_buffer = Arc::new(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Shadow Matrix Buffer"),
            size: (MAX_LIGHTS as u64) * 6 * 64, // 16 × 6 × 64 bytes
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));

        // Shared light count — updated each frame; read by ShadowPass
        let light_count_arc = Arc::new(AtomicU32::new(0));
        // Per-light face counts — updated each frame; read by ShadowPass
        let light_face_counts: Arc<std::sync::Mutex<Vec<u8>>> =
            Arc::new(std::sync::Mutex::new(Vec::new()));
        // Per-light cull data (position, range, type) — updated each frame; read by ShadowPass
        let shadow_cull_lights: Arc<std::sync::Mutex<Vec<ShadowCullLight>>> =
            Arc::new(std::sync::Mutex::new(Vec::new()));

        // ── Sky pass (added FIRST so it runs before geometry) ─────────────────
        let sky_uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Sky Uniform Buffer"),
            size: std::mem::size_of::<SkyUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Sky-View LUT: small panoramic Rgba16Float texture the SkyLutPass writes
        // into every frame. The main SkyPass samples it instead of running the
        // full atmosphere ray-march per pixel (~46× cheaper at 1280×720).
        let sky_lut_tex = device.create_texture(&wgpu::TextureDescriptor {
            label:           Some("Sky View LUT"),
            size:            wgpu::Extent3d { width: SKY_LUT_W, height: SKY_LUT_H, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count:    1,
            dimension:       wgpu::TextureDimension::D2,
            format:          SKY_LUT_FORMAT,
            usage:           wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats:    &[],
        });
        let sky_lut_view = Arc::new(sky_lut_tex.create_view(&wgpu::TextureViewDescriptor::default()));
        let sky_lut_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label:     Some("Sky LUT Sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            address_mode_u: wgpu::AddressMode::Repeat, // wrap azimuth
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            ..Default::default()
        });

        // Uniform-only bind group for the LUT pass (no LUT texture – it produces it)
        let sky_uniform_bg = Arc::new(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label:   Some("Sky Uniform Bind Group"),
            layout:  &resources.bind_group_layouts.sky_uniform,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: sky_uniform_buffer.as_entire_binding() },
            ],
        }));

        // Full sky bind group (uniform + LUT texture + sampler) for the SkyPass
        let sky_bind_group = Arc::new(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label:   Some("Sky Bind Group"),
            layout:  &resources.bind_group_layouts.sky,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: sky_uniform_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&sky_lut_view) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(&sky_lut_sampler) },
            ],
        }));

        // SkyLutPass: renders atmosphere into the small LUT each frame
        let sky_lut_pass = SkyLutPass::new(
            &device,
            &resources.bind_group_layouts.sky_uniform,
            &resources.bind_group_layouts.global,
            sky_uniform_bg,
            sky_lut_view,
        );
        graph.add_pass(sky_lut_pass);

        let sky_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Sky Pipeline Layout"),
            bind_group_layouts: &[
                Some(resources.bind_group_layouts.global.as_ref()),
                Some(resources.bind_group_layouts.sky.as_ref()),
            ],
            immediate_size: 0,
        });
        let sky_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Sky Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/passes/sky.wgsl").into()),
        });
        let sky_pipeline = Arc::new(device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Sky Pipeline"),
            layout: Some(&sky_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &sky_shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &sky_shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.surface_format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        }));

        let sky_pass = SkyPass::new(sky_pipeline, sky_bind_group.clone());
        graph.add_pass(sky_pass);

        // ── Deferred: depth + G-buffer textures (created before graph build) ────
        let defines = features.collect_shader_defines();

        // Depth texture with TEXTURE_BINDING so the deferred lighting pass can sample it.
        let (depth_texture, depth_view, depth_sample_view) =
            create_depth_texture(&device, config.width, config.height);

        // Four G-buffer render targets.
        let (gbuf_albedo_texture, gbuf_normal_texture, gbuf_orm_texture, gbuf_emissive_texture, gbuf_targets) =
            create_gbuffer_textures(&device, config.width, config.height);
        let gbuffer_targets = Arc::new(Mutex::new(gbuf_targets));

        // G-buffer write pass (replaces forward GeometryPass)
        let gbuf_pipeline = pipelines.get_or_create(
            include_str!("../shaders/passes/gbuffer.wgsl"),
            "gbuffer".to_string(),
            &defines,
            PipelineVariant::GBufferWrite,
        )?;
        let gbuffer_pass = GBufferPass::new(
            gbuffer_targets.clone(),
            gbuf_pipeline,
            draw_list.clone(),
        );
        graph.add_pass(gbuffer_pass);

        // ── Register features (adds BillboardPass etc. after GeometryPass) ───────
        let (feat_light_buf, feat_shadow_view, feat_shadow_sampler, feat_rc_view, feat_rc_bounds) = {
            let mut ctx = FeatureContext::new(
                &device, &queue, &mut graph, &mut resources, config.surface_format,
                device.clone(),
                draw_list.clone(),
                shadow_matrix_buffer.clone(),
                light_count_arc.clone(),
                light_face_counts.clone(),
                shadow_cull_lights.clone(),
            );
            features.register_all(&mut ctx)?;
            (ctx.light_buffer, ctx.shadow_atlas_view, ctx.shadow_sampler,
             ctx.rc_cascade0_view, ctx.rc_world_bounds)
        };

        // ── Default material views (shared by create_material + default material) ──
        let default_material_views = DefaultMaterialViews::new(&device, &queue);

        let mat_uniform = MaterialUniform {
            base_color: [1.0, 1.0, 1.0, 1.0],
            metallic: 0.0, roughness: 0.5, emissive_factor: 0.0, ao: 1.0,
            emissive_color: [0.0; 3], _pad: 0.0,
        };
        let material_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Default Material Uniform"),
            contents: bytemuck::bytes_of(&mat_uniform),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let default_material_bind_group = Arc::new(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Default Material Bind Group"),
            layout: &resources.bind_group_layouts.material,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: material_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&default_material_views.white_srgb) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(&default_material_views.flat_normal) },
                wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::Sampler(&default_material_views.sampler) },
                wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::TextureView(&default_material_views.white_orm) },
                wgpu::BindGroupEntry { binding: 5, resource: wgpu::BindingResource::TextureView(&default_material_views.black_emissive) },
            ],
        }));

        // ── Lighting bind group (group 2) ────────────────────────────────────
        // Fallback null light buffer when LightingFeature is not registered
        let null_light_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Null Light Buffer"),
            size: 64,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        // Fallback 1-layer shadow atlas
        let default_shadow_atlas = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Default Shadow Atlas"),
            size: wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
            mip_level_count: 1, sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let default_shadow_atlas_view = default_shadow_atlas.create_view(&wgpu::TextureViewDescriptor {
            dimension: Some(wgpu::TextureViewDimension::D2Array),
            ..Default::default()
        });

        let default_comparison_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Default Comparison Sampler"),
            compare: Some(wgpu::CompareFunction::LessEqual),
            ..Default::default()
        });

        // Fallback black cube (env map)
        let cube_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Default Env Cube"),
            size: wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 6 },
            mip_level_count: 1, sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        for face in 0..6u32 {
            queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &cube_tex, mip_level: 0,
                    origin: wgpu::Origin3d { x: 0, y: 0, z: face },
                    aspect: wgpu::TextureAspect::All,
                },
                &[0u8, 0, 0, 255],
                wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(4), rows_per_image: Some(1) },
                wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
            );
        }
        let cube_view = cube_tex.create_view(&wgpu::TextureViewDescriptor {
            dimension: Some(wgpu::TextureViewDimension::Cube),
            array_layer_count: Some(6),
            ..Default::default()
        });

        // Pick the real resources if features provided them, otherwise use defaults
        let light_buf     = feat_light_buf.unwrap_or_else(|| Arc::new(null_light_buf));
        let light_buffer  = light_buf.clone();
        let shadow_view   = feat_shadow_view.unwrap_or_else(|| Arc::new(default_shadow_atlas_view));
        let shadow_samp   = feat_shadow_sampler.unwrap_or_else(|| Arc::new(default_comparison_sampler));

        // Fallback 1×1 black Rgba16Float texture for RC when feature not registered
        let default_rc_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Default RC Cascade 0"),
            size: wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
            mip_level_count: 1, sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let default_rc_view = default_rc_tex.create_view(&Default::default());
        let rc_view = feat_rc_view.unwrap_or_else(|| Arc::new(default_rc_view));
        let rc_world_min = feat_rc_bounds.map(|(mn, _)| mn).unwrap_or([0.0; 3]);
        let rc_world_max = feat_rc_bounds.map(|(_, mx)| mx).unwrap_or([0.0; 3]);

        // Linear sampler for env-IBL reads in the deferred lighting shader (group 2 binding 6).
        let env_linear_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Env Linear Sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Linear,
            ..Default::default()
        });

        // Deferred clustered-lighting indirection buffers:
        // - offsets: cluster_count + 1 prefix-sum table
        // - indices: flattened light index list referenced by offsets
        let cluster_light_offsets_buffer = Arc::new(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Cluster Light Offsets"),
            size: ((MAX_CLUSTERS + 1) as u64) * std::mem::size_of::<u32>() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));
        let cluster_light_indices_buffer = Arc::new(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Cluster Light Indices"),
            size: (MAX_CLUSTERS as u64)
                * (MAX_LIGHTS_PER_CLUSTER as u64)
                * std::mem::size_of::<u32>() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));

        let lighting_bind_group = Arc::new(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Lighting Bind Group"),
            layout: &resources.bind_group_layouts.lighting,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: light_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&shadow_view) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(&shadow_samp) },
                wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(&cube_view) },
                wgpu::BindGroupEntry { binding: 4, resource: shadow_matrix_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: wgpu::BindingResource::TextureView(&rc_view) },
                wgpu::BindGroupEntry { binding: 6, resource: wgpu::BindingResource::Sampler(&env_linear_sampler) },
                wgpu::BindGroupEntry { binding: 7, resource: cluster_light_offsets_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 8, resource: cluster_light_indices_buffer.as_entire_binding() },
            ],
        }));

        // ── Deferred lighting pass (fullscreen PBR over the G-buffer) ─────────
        let gbuf_bg_inner = create_gbuffer_bind_group(
            &device,
            &resources.bind_group_layouts.gbuffer_read,
            &*gbuffer_targets.lock().unwrap(),
            &depth_sample_view,
        );
        let deferred_bg = Arc::new(Mutex::new(Arc::new(gbuf_bg_inner)));

        let deferred_pipeline = pipelines.get_or_create(
            include_str!("../shaders/passes/deferred_lighting.wgsl"),
            "deferred_lighting".to_string(),
            &defines,
            PipelineVariant::DeferredLighting,
        )?;
        let deferred_pass = DeferredLightingPass::new(deferred_bg.clone(), deferred_pipeline);
        graph.add_pass(deferred_pass);

        // Build the render graph
        graph.build()?;

        // Create profiler if TIMESTAMP_QUERY was requested on this device
        let profiler = GpuProfiler::new(&device, &queue);

        log::info!("Helio Render V2 initialized successfully");

        Ok(Self {
            device,
            queue,
            resources,
            graph,
            pipelines,
            features,
            camera_buffer,
            globals_buffer,
            global_bind_group,
            lighting_bind_group,
            lighting_layout,
            default_material_bind_group,
            draw_list,
            light_buffer,
            light_buffer_capacity_lights: MAX_LIGHTS,
            cluster_light_offsets_buffer,
            cluster_light_indices_buffer,
            lighting_shadow_view: shadow_view,
            lighting_shadow_sampler: shadow_samp,
            lighting_env_cube_view: Arc::new(cube_view),
            lighting_rc_view: rc_view,
            lighting_env_sampler: Arc::new(env_linear_sampler),
            shadow_matrix_buffer,
            light_count_arc,
            light_face_counts,
            shadow_cull_lights,
            scene_ambient_color: [0.0, 0.0, 0.0],
            scene_ambient_intensity: 0.0,
            scene_light_count: 0,
            scene_sky_color: [0.0, 0.0, 0.0],
            scene_has_sky: false,
            scene_csm_splits: CSM_SPLITS,
            rc_world_min,
            rc_world_max,
            sky_uniform_buffer,
            sky_bind_group,
            depth_texture,
            depth_view,
            depth_sample_view,
            gbuf_albedo_texture,
            gbuf_normal_texture,
            gbuf_orm_texture,
            gbuf_emissive_texture,
            gbuffer_targets,
            deferred_bg,
            default_material_views,
            frame_count: 0,
            width: config.width,
            height: config.height,
            last_frame_start: None,
            last_frame_end: None,
            profiler,
            debug_printout: false,
        })
    }

    // ── Material creation ─────────────────────────────────────────────────────

    /// Upload a [`Material`] to the GPU and return a [`GpuMaterial`] ready for use in
    /// [`Scene::add_object_with_material`] or [`Renderer::draw_mesh_with_material`].
    pub fn create_material(&self, mat: &Material) -> GpuMaterial {
        build_gpu_material(
            &self.device,
            &self.queue,
            &self.resources.bind_group_layouts.material,
            mat,
            &self.default_material_views,
        )
    }

    // ── Draw submission ───────────────────────────────────────────────────────

    /// Queue a mesh to be drawn this frame using the default white material
    pub fn draw_mesh(&self, mesh: &GpuMesh) {
        self.draw_list.lock().unwrap().push(DrawCall::new(mesh, self.default_material_bind_group.clone()));
    }

    /// Queue a mesh with a custom material bind group
    pub fn draw_mesh_with_material(&self, mesh: &GpuMesh, material: Arc<wgpu::BindGroup>) {
        self.draw_list.lock().unwrap().push(DrawCall::new(mesh, material));
    }

    fn rebuild_lighting_bind_group(&mut self) {
        self.lighting_bind_group = Arc::new(self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Lighting Bind Group"),
            layout: &self.lighting_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: self.light_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&self.lighting_shadow_view) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(&self.lighting_shadow_sampler) },
                wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(&self.lighting_env_cube_view) },
                wgpu::BindGroupEntry { binding: 4, resource: self.shadow_matrix_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: wgpu::BindingResource::TextureView(&self.lighting_rc_view) },
                wgpu::BindGroupEntry { binding: 6, resource: wgpu::BindingResource::Sampler(&self.lighting_env_sampler) },
                wgpu::BindGroupEntry { binding: 7, resource: self.cluster_light_offsets_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 8, resource: self.cluster_light_indices_buffer.as_entire_binding() },
            ],
        }));
    }

    fn ensure_light_buffer_capacity(&mut self, required_lights: u32) {
        if required_lights <= self.light_buffer_capacity_lights {
            return;
        }

        let mut new_capacity = self.light_buffer_capacity_lights.max(1);
        while new_capacity < required_lights {
            new_capacity = new_capacity.saturating_mul(2);
            if new_capacity == u32::MAX {
                break;
            }
        }

        let new_size = (std::mem::size_of::<GpuLight>() as u64) * (new_capacity as u64);
        self.light_buffer = Arc::new(self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Light Storage Buffer"),
            size: new_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));
        self.light_buffer_capacity_lights = new_capacity;
        self.rebuild_lighting_bind_group();
        log::info!(
            "Grew light buffer to {} lights ({} bytes)",
            new_capacity,
            new_size,
        );
    }

    // ── Feature enable/disable ────────────────────────────────────────────────

    /// Enable a feature at runtime
    pub fn enable_feature(&mut self, name: &str) -> Result<()> {
        self.features.enable(name)?;
        let flags = self.features.active_flags();
        self.pipelines.set_active_features(flags);
        log::info!("Enabled feature: {}", name);
        Ok(())
    }

    /// Toggle the per-frame GPU timing printout.  Call this when the user presses the 4 key.
    /// When on, GPU pass timings are printed to stderr roughly once per second (every 60 frames).
    /// When off, no timing output is produced — no call-site boilerplate needed in examples.
    pub fn debug_key_pressed(&mut self) {
        self.debug_printout = !self.debug_printout;
        eprintln!("[Helio] GPU timing printout: {}",
            if self.debug_printout { "ON  (press 4 to hide)" } else { "OFF" });
    }

    /// Disable a feature at runtime
    pub fn disable_feature(&mut self, name: &str) -> Result<()> {
        self.features.disable(name)?;
        let flags = self.features.active_flags();
        self.pipelines.set_active_features(flags);
        log::info!("Disabled feature: {}", name);
        Ok(())
    }

    /// Get a mutable reference to a registered feature by name and type.
    pub fn get_feature_mut<T: crate::features::Feature + 'static>(&mut self, name: &str) -> Option<&mut T> {
        self.features.get_typed_mut::<T>(name)
    }

    // ── Frame rendering ───────────────────────────────────────────────────────

    /// Render a frame.  Call `draw_mesh()` BEFORE calling this.
    pub fn render(&mut self, camera: &Camera, target: &wgpu::TextureView, delta_time: f32) -> Result<()> {
        let frame_start = std::time::Instant::now();
        log::trace!("Rendering frame {}", self.frame_count);

        // Update global uniforms
        self.queue.write_buffer(&self.camera_buffer, 0, bytemuck::bytes_of(camera));
        let globals = GlobalsUniform {
            frame: self.frame_count as u32,
            delta_time,
            light_count: self.scene_light_count,
            ambient_intensity: self.scene_ambient_intensity,
            ambient_color: [self.scene_ambient_color[0], self.scene_ambient_color[1], self.scene_ambient_color[2], 0.0],
            rc_world_min: [self.rc_world_min[0], self.rc_world_min[1], self.rc_world_min[2], 0.0],
            rc_world_max: [self.rc_world_max[0], self.rc_world_max[1], self.rc_world_max[2], 0.0],
            csm_splits: self.scene_csm_splits,
        };
        self.queue.write_buffer(&self.globals_buffer, 0, bytemuck::bytes_of(&globals));

        // Prepare features (upload lights etc.)
        let prep_ctx = PrepareContext::new(
            &self.device, &self.queue, &self.resources,
            self.frame_count, delta_time, camera,
        );
        self.features.prepare_all(&prep_ctx)?;

        // Execute render graph
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Render Encoder"),
        });

        let mut graph_ctx = GraphContext {
            encoder: &mut encoder,
            resources: &self.resources,
            target,
            depth_view: &self.depth_view,
            frame: self.frame_count,
            global_bind_group: &self.global_bind_group,
            lighting_bind_group: &self.lighting_bind_group,
            sky_color: self.scene_sky_color,
            has_sky: self.scene_has_sky,
            sky_bind_group: None,
        };

        if let Some(p) = &mut self.profiler { p.begin_frame(); }

        self.graph.execute(&mut graph_ctx, self.profiler.as_mut())?;

        // Resolve GPU timestamp queries into staging buffers (before finish)
        if let Some(p) = &mut self.profiler { p.resolve(&mut encoder); }

        // Submit and clear the draw list for next frame
        let submit_start = std::time::Instant::now();
        self.queue.submit(Some(encoder.finish()));
        let submit_ms = submit_start.elapsed().as_secs_f32() * 1000.0;
        if submit_ms > 10.0 {
            eprintln!("⚠️  queue.submit() blocked for {:.2}ms", submit_ms);
        }
        
        self.draw_list.lock().unwrap().clear();

        // Schedule map_async AFTER submit so the buffer is no longer used by the encoder
        if let Some(p) = &mut self.profiler { p.begin_readback(); }

        // Non-blocking readback of the previous frame's timing results
        let poll_start = std::time::Instant::now();
        if let Some(p) = &mut self.profiler { p.poll_results(&self.device); }
        let poll_ms = poll_start.elapsed().as_secs_f32() * 1000.0;
        if poll_ms > 10.0 {
            eprintln!("⚠️  poll_results() blocked for {:.2}ms", poll_ms);
        }

        // Measure frame-to-frame latency (time from start of previous frame to start of this frame)
        let frame_to_frame_ms = if let Some(last_start) = self.last_frame_start {
            last_start.elapsed().as_secs_f32() * 1000.0
        } else {
            0.0 // First frame; no previous frame to measure from
        };

        // Measure render duration (time from start of this frame to end)
        let frame_time_ms = frame_start.elapsed().as_secs_f32() * 1000.0;
        
        // Track when this frame's render started for next frame's calculation
        self.last_frame_start = Some(frame_start);
        self.last_frame_end = Some(std::time::Instant::now());

        if self.debug_printout && self.frame_count % 60 == 0 {
            let total_elapsed_ms = frame_start.elapsed().as_secs_f32() * 1000.0;
            eprintln!("🎮 RENDERER: total elapsed from render() start: {:.2}ms (submit={:.2}ms, poll={:.2}ms)", 
                total_elapsed_ms, submit_ms, poll_ms);
            
            if let Some(p) = &mut self.profiler {
                p.set_frame_time_ms(frame_time_ms);
                p.set_frame_to_frame_ms(frame_to_frame_ms);
            }
            if let Some(p) = &self.profiler {
                if !p.last_timings.is_empty() {
                    let timings = p.last_timings.clone();
                    let total_gpu   = p.last_total_gpu_ms;
                    let total_cpu   = p.last_total_cpu_ms;
                    std::thread::spawn(move || {
                        crate::profiler::GpuProfiler::print_snapshot(timings, total_gpu, total_cpu, frame_time_ms, frame_to_frame_ms);
                    });
                }
            }
        }

        self.frame_count += 1;
        Ok(())
    }

    // ── Utilities ─────────────────────────────────────────────────────────────

    /// Render the full scene. Everything in the scene is drawn; nothing else.
    pub fn render_scene(&mut self, scene: &Scene, camera: &Camera, target: &wgpu::TextureView, delta_time: f32) -> Result<()> {
        // Queue draw calls for all objects
        for obj in &scene.objects {
            match &obj.material {
                Some(mat_bg) => self.draw_mesh_with_material(&obj.mesh, mat_bg.clone()),
                None => self.draw_mesh(&obj.mesh),
            }
        }

        // Upload scene lights to GPU buffer.
        // Lights are sorted by relevance so the limited shadow budget is spent
        // on lights that actually contribute near the current camera view.
        let count = scene.lights.len();
        let mut sorted_lights: Vec<&crate::scene::SceneLight> = scene.lights[..count].iter().collect();
        sorted_lights.sort_by(|a, b| {
            fn score(light: &crate::scene::SceneLight, camera_pos: glam::Vec3) -> f32 {
                match light.light_type {
                    crate::features::LightType::Directional => {
                        // Keep directional lights at highest priority (sun/moon).
                        f32::MAX
                    }
                    crate::features::LightType::Point | crate::features::LightType::Spot { .. } => {
                        let lp = glam::Vec3::from(light.position);
                        let d  = camera_pos.distance(lp);
                        let r  = light.range.max(0.001);
                        if d >= r {
                            0.0
                        } else {
                            // Smooth attenuation proxy in [0, 1], stronger near light center.
                            let x = d / r;
                            let attenuation = (1.0 - x * x).max(0.0);
                            light.intensity.max(0.0) * attenuation
                        }
                    }
                }
            }

            let sb = score(b, camera.position);
            let sa = score(a, camera.position);
            sb.partial_cmp(&sa).unwrap_or(std::cmp::Ordering::Equal)
        });

        self.ensure_light_buffer_capacity(count as u32);

        let gpu_lights: Vec<GpuLight> = sorted_lights.iter().map(|l| {
            let light_type = match l.light_type {
                crate::features::LightType::Directional => 0.0,
                crate::features::LightType::Point => 1.0,
                crate::features::LightType::Spot { .. } => 2.0,
            };
            let (cos_inner, cos_outer) = match l.light_type {
                crate::features::LightType::Spot { inner_angle, outer_angle } => {
                    (inner_angle.cos(), outer_angle.cos())
                }
                _ => (0.0, 0.0),
            };
            // Prenormalize direction on the CPU so fragment shaders never call normalize().
            let dir_len = (l.direction[0] * l.direction[0]
                + l.direction[1] * l.direction[1]
                + l.direction[2] * l.direction[2]).sqrt();
            let direction = if dir_len > 1e-6 {
                [l.direction[0] / dir_len, l.direction[1] / dir_len, l.direction[2] / dir_len]
            } else {
                [0.0, -1.0, 0.0]
            };
            GpuLight {
                position: l.position,
                light_type,
                direction,
                range: l.range,
                color: l.color,
                intensity: l.intensity,
                cos_inner,
                cos_outer,
                _pad: [0.0; 2],
            }
        }).collect();
        if !gpu_lights.is_empty() {
            self.queue.write_buffer(&self.light_buffer, 0, bytemuck::cast_slice(&gpu_lights));
        }

        // Build deferred clustered-lighting indirection tables (x/y tiles + z bins).
        // Each cluster stores only relevant lights, reducing shading cost from
        // O(total_lights) to O(cluster_lights).
        let cluster_count_x = ((self.width + CLUSTER_TILE_SIZE_PX - 1) / CLUSTER_TILE_SIZE_PX).max(1);
        let cluster_count_y = ((self.height + CLUSTER_TILE_SIZE_PX - 1) / CLUSTER_TILE_SIZE_PX).max(1);
        let cluster_count = (cluster_count_x * cluster_count_y * CLUSTER_Z_BINS)
            .min(MAX_CLUSTERS) as usize;
        let per_cluster_capacity = MAX_LIGHTS_PER_CLUSTER as usize;
        let mut cluster_counts = vec![0u32; cluster_count];
        let mut cluster_slots = vec![0u32; cluster_count * per_cluster_capacity];

        let project = |p: [f32; 3]| -> Option<(f32, f32)> {
            let clip = camera.view_proj * glam::Vec4::new(p[0], p[1], p[2], 1.0);
            if clip.w <= 0.0001 {
                return None;
            }
            let ndc = clip.xyz() / clip.w;
            let sx = (ndc.x * 0.5 + 0.5) * self.width as f32;
            let sy = (1.0 - (ndc.y * 0.5 + 0.5)) * self.height as f32;
            Some((sx, sy))
        };

        let cluster_z_from_distance = |distance: f32| -> u32 {
            let d = distance.clamp(CLUSTER_NEAR, CLUSTER_FAR);
            let t = (d / CLUSTER_NEAR).ln() / (CLUSTER_FAR / CLUSTER_NEAR).ln();
            ((t * CLUSTER_Z_BINS as f32).floor() as u32).min(CLUSTER_Z_BINS - 1)
        };

        let mut cluster_overflow_count: u32 = 0;
        for (light_idx, l) in sorted_lights.iter().enumerate() {
            let (x0, x1, y0, y1, z0, z1) = match l.light_type {
                crate::features::LightType::Directional => {
                    (0u32, cluster_count_x - 1, 0u32, cluster_count_y - 1, 0u32, CLUSTER_Z_BINS - 1)
                }
                crate::features::LightType::Point | crate::features::LightType::Spot { .. } => {
                    let r = l.range.max(0.001);
                    let d_center = camera.position.distance(glam::Vec3::from(l.position));
                    let min_d = (d_center - r).max(CLUSTER_NEAR);
                    let max_d = (d_center + r).min(CLUSTER_FAR);

                    // If the camera is inside (or very near) the light volume,
                    // use full XY coverage to avoid frustum-edge popping.
                    if d_center <= r * 1.2 {
                        (
                            0u32,
                            cluster_count_x - 1,
                            0u32,
                            cluster_count_y - 1,
                            cluster_z_from_distance(min_d),
                            cluster_z_from_distance(max_d),
                        )
                    } else {
                    let mut min_x = f32::INFINITY;
                    let mut max_x = f32::NEG_INFINITY;
                    let mut min_y = f32::INFINITY;
                    let mut max_y = f32::NEG_INFINITY;
                    let mut any = false;

                    let samples = [
                        [0.0, 0.0, 0.0],
                        [ r, 0.0, 0.0],
                        [-r, 0.0, 0.0],
                        [0.0,  r, 0.0],
                        [0.0, -r, 0.0],
                        [0.0, 0.0,  r],
                        [0.0, 0.0, -r],
                    ];

                    for s in samples {
                        let p = [l.position[0] + s[0], l.position[1] + s[1], l.position[2] + s[2]];
                        if let Some((sx, sy)) = project(p) {
                            any = true;
                            min_x = min_x.min(sx);
                            max_x = max_x.max(sx);
                            min_y = min_y.min(sy);
                            max_y = max_y.max(sy);
                        }
                    }

                    if !any {
                        (
                            0u32,
                            cluster_count_x - 1,
                            0u32,
                            cluster_count_y - 1,
                            cluster_z_from_distance(min_d),
                            cluster_z_from_distance(max_d),
                        )
                    } else {
                    if max_x < 0.0 || max_y < 0.0
                        || min_x > self.width as f32 || min_y > self.height as f32 {
                        continue;
                    }

                    // Conservative padding avoids tiny sub-pixel misses that create seams.
                    let pad_px = (CLUSTER_TILE_SIZE_PX as f32) * 2.0;
                    min_x -= pad_px;
                    max_x += pad_px;
                    min_y -= pad_px;
                    max_y += pad_px;

                    let clamped_min_x = min_x.clamp(0.0, (self.width.saturating_sub(1)) as f32);
                    let clamped_max_x = max_x.clamp(0.0, (self.width.saturating_sub(1)) as f32);
                    let clamped_min_y = min_y.clamp(0.0, (self.height.saturating_sub(1)) as f32);
                    let clamped_max_y = max_y.clamp(0.0, (self.height.saturating_sub(1)) as f32);

                    (
                        (clamped_min_x as u32 / CLUSTER_TILE_SIZE_PX).min(cluster_count_x - 1),
                        (clamped_max_x as u32 / CLUSTER_TILE_SIZE_PX).min(cluster_count_x - 1),
                        (clamped_min_y as u32 / CLUSTER_TILE_SIZE_PX).min(cluster_count_y - 1),
                        (clamped_max_y as u32 / CLUSTER_TILE_SIZE_PX).min(cluster_count_y - 1),
                        cluster_z_from_distance(min_d),
                        cluster_z_from_distance(max_d),
                    )
                    }
                    }
                }
            };

            for z in z0..=z1 {
                for ty in y0..=y1 {
                    for tx in x0..=x1 {
                        let cluster = ((z * cluster_count_y + ty) * cluster_count_x + tx) as usize;
                        if cluster >= cluster_count {
                            continue;
                        }
                        let count_in_cluster = cluster_counts[cluster] as usize;
                        if count_in_cluster >= per_cluster_capacity {
                            cluster_overflow_count += 1;
                            continue;
                        }
                        cluster_slots[cluster * per_cluster_capacity + count_in_cluster] = light_idx as u32;
                        cluster_counts[cluster] += 1;
                    }
                }
            }
        }

        let mut cluster_offsets = vec![0u32; cluster_count + 1];
        for i in 0..cluster_count {
            cluster_offsets[i + 1] = cluster_offsets[i] + cluster_counts[i];
        }
        let total_indices = cluster_offsets[cluster_count] as usize;
        let mut cluster_indices = Vec::with_capacity(total_indices);
        for i in 0..cluster_count {
            let base = i * per_cluster_capacity;
            let n = cluster_counts[i] as usize;
            cluster_indices.extend_from_slice(&cluster_slots[base..base + n]);
        }

        self.queue.write_buffer(&self.cluster_light_offsets_buffer, 0, bytemuck::cast_slice(&cluster_offsets));
        if !cluster_indices.is_empty() {
            self.queue.write_buffer(&self.cluster_light_indices_buffer, 0, bytemuck::cast_slice(&cluster_indices));
        }
        if cluster_overflow_count > 0 && self.frame_count % 120 == 0 {
            log::warn!(
                "Clustered lighting overflow: {} assignments dropped (capacity {} per cluster)",
                cluster_overflow_count,
                MAX_LIGHTS_PER_CLUSTER,
            );
        }

        // Compute and upload light-space matrices — 6 per light (cube faces).
        // Directional lights: slots 0-3 = 4 CSM cascades, slots 4-5 = identity.
        // Point lights: 6 real cube-face matrices.
        let identity = Mat4::IDENTITY;
        let mut shadow_mats: Vec<GpuShadowMatrix> = Vec::with_capacity(MAX_LIGHTS as usize * 6);
        for l in &sorted_lights {
            let six: [Mat4; 6] = match l.light_type {
                crate::features::LightType::Point => {
                    compute_point_light_matrices(l.position, l.range)
                }
                crate::features::LightType::Directional => {
                    let [c0, c1, c2, c3] = compute_directional_cascades(
                        camera.position,
                        camera.view_proj_inv,
                        l.direction,
                    );
                    [c0, c1, c2, c3, identity, identity]
                }
                crate::features::LightType::Spot { outer_angle, .. } => {
                    let m0 = compute_spot_matrix(l.position, l.direction, l.range, outer_angle);
                    [m0, identity, identity, identity, identity, identity]
                }
            };
            for m in &six {
                shadow_mats.push(GpuShadowMatrix { mat: m.to_cols_array() });
            }
        }
        shadow_mats.resize(MAX_LIGHTS as usize * 6, GpuShadowMatrix::zeroed());
        self.queue.write_buffer(&self.shadow_matrix_buffer, 0, bytemuck::cast_slice(&shadow_mats));

        // Update shared light count and per-light face counts (ShadowPass reads these)
        self.light_count_arc.store(count as u32, Ordering::Relaxed);
        {
            let mut fc = self.light_face_counts.lock().unwrap();
            fc.clear();
            for l in &sorted_lights {
                let faces: u8 = match l.light_type {
                    crate::features::LightType::Point       => 6,
                    crate::features::LightType::Directional => 4, // CSM cascades 0-3 only
                    crate::features::LightType::Spot { .. } => 1, // single projection
                };
                fc.push(faces);
            }
        }
        {
            let mut cull = self.shadow_cull_lights.lock().unwrap();
            cull.clear();
            for l in &sorted_lights {
                cull.push(ShadowCullLight {
                    position:       l.position,
                    range:          l.range,
                    is_directional: matches!(l.light_type, crate::features::LightType::Directional),
                    is_point:       matches!(l.light_type, crate::features::LightType::Point),
                });
            }
        }

        // Update billboard instances from scene
        if let Some(bb) = self.features.get_typed_mut::<BillboardsFeature>("billboards") {
            bb.set_billboards(scene.billboards.clone());
        }

        // Store scene ambient + sky for globals upload in render()
        // Start from scene's own settings, then optionally blend in sky
        self.scene_ambient_color    = scene.ambient_color;
        self.scene_ambient_intensity = scene.ambient_intensity;
        self.scene_light_count = count as u32;
        self.scene_sky_color = scene.sky_color;
        self.scene_has_sky = scene.sky_atmosphere.is_some();
        // CSM splits are static constants; stored so render() can pass them to globals
        self.scene_csm_splits = CSM_SPLITS;

        // Build and upload sky uniforms (even when sky is disabled the buffer exists)
        if let Some(atm) = &scene.sky_atmosphere {
            // Use the first directional light as the sun; fall back to default noon sun
            let sun_dir = scene.lights.iter()
                .find(|l| matches!(l.light_type, crate::features::LightType::Directional))
                .map(|l| {
                    let d = Vec3::from(l.direction).normalize();
                    // Convert from "direction of light ray" to "toward sun"
                    [-d.x, -d.y, -d.z]
                })
                .unwrap_or([0.0, 1.0, 0.0]);

            let clouds = atm.clouds.as_ref();
            let sky_uni = SkyUniform {
                sun_direction:    sun_dir,
                sun_intensity:    atm.sun_intensity,
                rayleigh_scatter: atm.rayleigh_scatter,
                rayleigh_h_scale: atm.rayleigh_h_scale,
                mie_scatter:      atm.mie_scatter,
                mie_h_scale:      atm.mie_h_scale,
                mie_g:            atm.mie_g,
                sun_disk_cos:     atm.sun_disk_angle.cos(),
                earth_radius:     atm.earth_radius,
                atm_radius:       atm.atm_radius,
                exposure:         atm.exposure,
                clouds_enabled:   clouds.is_some() as u32,
                cloud_coverage:   clouds.map(|c| c.coverage).unwrap_or(0.5),
                cloud_density:    clouds.map(|c| c.density).unwrap_or(0.8),
                cloud_base:       clouds.map(|c| c.base_height).unwrap_or(800.0),
                cloud_top:        clouds.map(|c| c.top_height).unwrap_or(1800.0),
                cloud_wind_x:     clouds.map(|c| c.wind_direction[0]).unwrap_or(1.0),
                cloud_wind_z:     clouds.map(|c| c.wind_direction[1]).unwrap_or(0.0),
                cloud_speed:      clouds.map(|c| c.wind_speed).unwrap_or(0.3),
                time_sky:         self.frame_count as f32 / 60.0,
                skylight_intensity: scene.skylight.as_ref().map(|sl| sl.intensity).unwrap_or(1.0),
                _pad0: 0.0, _pad1: 0.0, _pad2: 0.0,
            };
            self.queue.write_buffer(&self.sky_uniform_buffer, 0, bytemuck::bytes_of(&sky_uni));

            // Inject sky-based ambient when a Skylight is present.
            // RC cascades already handle most indirect diffuse; skylight should
            // act as a subtle fill/tint (0.06 scale keeps it from overpowering
            // direct lights and avoids double-counting with RC GI).
            if let Some(skylight) = &scene.skylight {
                let sun_elev = sun_dir[1].clamp(-1.0, 1.0);
                let sky_amb  = estimate_sky_ambient(sun_elev, &atm.rayleigh_scatter);
                let tint     = skylight.color_tint;
                // ~0.004 = very subtle sky tint, will not overpower direct lights or GI
                let si = skylight.intensity * 0.004;
                let base_i = scene.ambient_intensity;
                self.scene_ambient_color = [
                    scene.ambient_color[0] * base_i + sky_amb[0] * tint[0] * si,
                    scene.ambient_color[1] * base_i + sky_amb[1] * tint[1] * si,
                    scene.ambient_color[2] * base_i + sky_amb[2] * tint[2] * si,
                ];
                self.scene_ambient_intensity = 1.0;
            }
        }

        self.render(camera, target, delta_time)
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        log::info!("Resizing renderer to {}x{}", width, height);
        self.width = width;
        self.height = height;

        // Recreate depth texture (new size; keeps TEXTURE_BINDING for deferred read)
        let (tex, view, sample_view) = create_depth_texture(&self.device, width, height);
        self.depth_texture = tex;
        self.depth_view = view;
        self.depth_sample_view = sample_view;

        // Recreate G-buffer textures at new resolution
        let (albedo_tex, normal_tex, orm_tex, emissive_tex, new_targets) =
            create_gbuffer_textures(&self.device, width, height);
        self.gbuf_albedo_texture  = albedo_tex;
        self.gbuf_normal_texture  = normal_tex;
        self.gbuf_orm_texture     = orm_tex;
        self.gbuf_emissive_texture = emissive_tex;

        // Hot-swap the targets shared with GBufferPass
        *self.gbuffer_targets.lock().unwrap() = new_targets;

        // Rebuild the G-buffer read bind group and hot-swap into DeferredLightingPass
        let new_bg = Arc::new(create_gbuffer_bind_group(
            &self.device,
            &self.resources.bind_group_layouts.gbuffer_read,
            &*self.gbuffer_targets.lock().unwrap(),
            &self.depth_sample_view,
        ));
        *self.deferred_bg.lock().unwrap() = new_bg;
    }

    pub fn frame_count(&self) -> u64 { self.frame_count }

    // ── Profiling ─────────────────────────────────────────────────────────────

    /// Returns the per-pass timing results collected during the last completed frame.
    /// Each entry has `name`, `gpu_ms` (GPU execution time), and `cpu_ms` (driver overhead).
    /// Returns an empty slice before the 2nd frame or when TIMESTAMP_QUERY is unavailable.
    pub fn last_frame_timings(&self) -> &[PassTiming] {
        self.profiler.as_ref()
            .map(|p| p.last_timings.as_slice())
            .unwrap_or(&[])
    }

    /// Print a compact timing summary to stderr.  Pass the current frame number;
    /// output is emitted only when `frame % interval == 0` so you can call this
    /// every frame without flooding the console.
    ///
    /// Example (interval = 60):
    /// ```text
    /// [GPU TIMING]  sky_lut: 0.11ms(gpu) | sky: 0.43ms(gpu) | radiance_cascades: 8.71ms(gpu) | ...
    /// ```
    pub fn print_timings_every(&self, interval: u64) {
        if self.frame_count % interval != 0 { return; }
        if let Some(p) = &self.profiler {
            // Clone the snapshot so the print can happen on a detached thread
            // without blocking the render loop.
            let timings = p.last_timings.clone();
            let total_gpu   = p.last_total_gpu_ms;
            let total_cpu   = p.last_total_cpu_ms;
            let frame_time  = p.last_frame_time_ms;
            let frame_to_frame = p.last_frame_to_frame_ms;
            std::thread::spawn(move || {
                crate::profiler::GpuProfiler::print_snapshot(timings, total_gpu, total_cpu, frame_time, frame_to_frame);
            });
        } else {
            log::info!("[TIMING] TIMESTAMP_QUERY unavailable — add wgpu::Features::TIMESTAMP_QUERY to device descriptor");
        }
    }
    pub fn device(&self) -> &wgpu::Device { &self.device }
    pub fn queue(&self) -> &wgpu::Queue { &self.queue }
}
