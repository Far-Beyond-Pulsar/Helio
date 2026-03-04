//! Main renderer implementation

use crate::resources::ResourceManager;
use crate::features::{FeatureRegistry, FeatureContext, PrepareContext};
use crate::pipeline::{PipelineCache, PipelineVariant};
use crate::graph::{RenderGraph, GraphContext};
use crate::passes::{GeometryPass, SkyPass, SkyLutPass, SKY_LUT_W, SKY_LUT_H, SKY_LUT_FORMAT};
use crate::mesh::{GpuMesh, DrawCall};
use crate::camera::Camera;
use crate::scene::Scene;
use crate::features::lighting::{GpuLight, MAX_LIGHTS};
use crate::features::BillboardsFeature;
use crate::material::{Material, GpuMaterial, MaterialUniform, DefaultMaterialViews, build_gpu_material};
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
const CSM_SPLITS: [f32; 4] = [8.0, 30.0, 100.0, 500.0];

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
    const SCENE_DEPTH: f32 = 2000.0;
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

/// Create a Depth32Float texture + view at the given resolution
fn create_depth_texture(device: &wgpu::Device, width: u32, height: u32) -> (wgpu::Texture, wgpu::TextureView) {
    let tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("Depth Texture"),
        size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Depth32Float,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
    (tex, view)
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
    default_material_bind_group: Arc<wgpu::BindGroup>,

    // Default 1×1 texture views + sampler shared by all materials
    default_material_views: DefaultMaterialViews,

    // Draw list (shared with GeometryPass)
    draw_list: Arc<Mutex<Vec<DrawCall>>>,

    // Light buffer for scene writes
    light_buffer: Arc<wgpu::Buffer>,
    // Shadow light-space matrix buffer (shared with ShadowPass)
    shadow_matrix_buffer: Arc<wgpu::Buffer>,
    // Shared light count for ShadowPass (updated each frame before graph exec)
    light_count_arc: Arc<AtomicU32>,
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
    depth_texture: wgpu::Texture,
    depth_view: wgpu::TextureView,

    // Frame state
    frame_count: u64,
    width: u32,
    height: u32,
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

        // ── Geometry pass ─────────────────────────────────────────────────────
        let defines = features.collect_shader_defines();
        let geometry_pipeline = pipelines.get_or_create(
            include_str!("../shaders/passes/geometry.wgsl"),
            "geometry".to_string(),
            &defines,
            PipelineVariant::Forward,
        )?;

        let mut geometry_pass = GeometryPass::with_draw_list(draw_list.clone());
        geometry_pass.set_pipeline(geometry_pipeline);
        graph.add_pass(geometry_pass);

        // ── Register features (adds BillboardPass etc. after GeometryPass) ───────
        let (feat_light_buf, feat_shadow_view, feat_shadow_sampler, feat_rc_view, feat_rc_bounds) = {
            let mut ctx = FeatureContext::new(
                &device, &queue, &mut graph, &mut resources, config.surface_format,
                device.clone(),
                draw_list.clone(),
                shadow_matrix_buffer.clone(),
                light_count_arc.clone(),
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
            ],
        }));

        // Build the render graph
        graph.build()?;

        let (depth_texture, depth_view) = create_depth_texture(&device, config.width, config.height);

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
            default_material_bind_group,
            draw_list,
            light_buffer,
            shadow_matrix_buffer,
            light_count_arc,
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
            default_material_views,
            frame_count: 0,
            width: config.width,
            height: config.height,
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

    // ── Feature enable/disable ────────────────────────────────────────────────

    /// Enable a feature at runtime
    pub fn enable_feature(&mut self, name: &str) -> Result<()> {
        self.features.enable(name)?;
        let flags = self.features.active_flags();
        self.pipelines.set_active_features(flags);
        log::info!("Enabled feature: {}", name);
        Ok(())
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

        self.graph.execute(&mut graph_ctx)?;

        // Submit and clear the draw list for next frame
        self.queue.submit(Some(encoder.finish()));
        self.draw_list.lock().unwrap().clear();

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

        // Upload scene lights to GPU buffer
        let count = scene.lights.len().min(MAX_LIGHTS as usize);
        let mut gpu_lights: Vec<GpuLight> = scene.lights[..count].iter().map(|l| {
            let light_type = match l.light_type {
                crate::features::LightType::Directional => 0.0,
                crate::features::LightType::Point => 1.0,
                crate::features::LightType::Spot { .. } => 2.0,
            };
            let (inner_angle, outer_angle) = match l.light_type {
                crate::features::LightType::Spot { inner_angle, outer_angle } => (inner_angle, outer_angle),
                _ => (0.0, 0.0),
            };
            GpuLight {
                position: l.position,
                light_type,
                direction: l.direction,
                range: l.range,
                color: l.color,
                intensity: l.intensity,
                inner_angle,
                outer_angle,
                _pad: [0.0; 2],
            }
        }).collect();
        gpu_lights.resize(MAX_LIGHTS as usize, GpuLight::zeroed());
        self.queue.write_buffer(&self.light_buffer, 0, bytemuck::cast_slice(&gpu_lights));

        // Compute and upload light-space matrices — 6 per light (cube faces).
        // Directional lights: slots 0-3 = 4 CSM cascades, slots 4-5 = identity.
        // Point lights: 6 real cube-face matrices.
        let identity = Mat4::IDENTITY;
        let mut shadow_mats: Vec<GpuShadowMatrix> = Vec::with_capacity(MAX_LIGHTS as usize * 6);
        for l in &scene.lights[..count] {
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

        // Update shared light count (ShadowPass reads this before drawing)
        self.light_count_arc.store(count as u32, Ordering::Relaxed);

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
        let (tex, view) = create_depth_texture(&self.device, width, height);
        self.depth_texture = tex;
        self.depth_view = view;
    }

    pub fn frame_count(&self) -> u64 { self.frame_count }
    pub fn device(&self) -> &wgpu::Device { &self.device }
    pub fn queue(&self) -> &wgpu::Queue { &self.queue }
}
