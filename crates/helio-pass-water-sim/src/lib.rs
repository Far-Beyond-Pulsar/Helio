//! Water heightfield simulation + rendering pass (full WebGPU water port).
//!
//! This single pass bundles:
//!  - The 256x256 `Rgba16Float` shallow-water wave simulation (ping-pong)
//!  - Caustics projection onto a `Rgba16Float` accumulation texture
//!  - Water surface rendering (above-water Fresnel + below-water view)
//!
//! Pool walls/floor are NOT rendered as explicit geometry — the surface shaders
//! ray-march to the pool interior for reflections and refractions internally,
//! giving an identical visual result without separate pool draw calls.
//!
//! ## Pre-TAA integration
//!
//! Water runs **before TAA** and writes into its own `water_output_tex`
//! intermediary at internal resolution.  `publish()` overwrites `frame.pre_aa`
//! with this intermediary so TAA picks up the water-composited scene without
//! any changes to the TAA pass.
//!
//! Execute order each frame:
//!   1. (optional) AABB hitbox displacement
//!   2. (optional) Drop ripple
//!   3. 2x wave-propagation update steps
//!   4. Normal recomputation
//!   5. Caustics projection -> caustics texture
//!   6. Blit pre_aa -> water_output (scene baseline)
//!   7. Water surface render (above + below faces) -> water_output

use wgpu::util::DeviceExt;
use bytemuck::{Pod, Zeroable};
use helio_v3::{PassContext, PrepareContext, RenderPass, Result as HelioResult};

/// Simple fullscreen blit: copies a texture to the render target as-is.
const BLIT_WGSL: &str = "
@group(0) @binding(0) var blit_tex:  texture_2d<f32>;
@group(0) @binding(1) var blit_samp: sampler;
struct V { @builtin(position) pos: vec4<f32>, @location(0) uv: vec2<f32> }
@vertex fn vs(@builtin(vertex_index) vi: u32) -> V {
    let x = f32((vi << 1u) & 2u);
    let y = f32(vi & 2u);
    return V(vec4<f32>(x * 2.0 - 1.0, 1.0 - y * 2.0, 0.0, 1.0), vec2<f32>(x, y));
}
@fragment fn fs(in: V) -> @location(0) vec4<f32> {
    return textureSample(blit_tex, blit_samp, in.uv);
}
";

const SIM_SIZE: u32 = 256;
const CAUSTICS_SIZE: u32 = 256;
const MAX_DROPS_BUFFERED: usize = 16;

// ---- GPU uniform structs --------------------------------------------------------

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct DropUniform {
    center: [f32; 2],
    radius: f32,
    strength: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct DeltaUniform {
    delta: [f32; 2],
    _pad: [f32; 2],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct HitboxCountUniform {
    count: u32,
    _pad: [u32; 3],
}

// ---- Mesh helpers ----------------------------------------------------------------

fn make_surface_mesh(device: &wgpu::Device) -> (wgpu::Buffer, wgpu::Buffer, u32) {
    const DETAIL: u32 = 128;
    let n = DETAIL + 1;
    let mut verts: Vec<[f32; 3]> = Vec::with_capacity((n * n) as usize);
    for j in 0..n {
        for i in 0..n {
            let x = i as f32 / DETAIL as f32 * 2.0 - 1.0;
            let y = j as f32 / DETAIL as f32 * 2.0 - 1.0;
            verts.push([x, y, 0.0]);
        }
    }
    let mut indices: Vec<u32> = Vec::with_capacity((DETAIL * DETAIL * 6) as usize);
    for j in 0..DETAIL {
        for i in 0..DETAIL {
            let tl = j * n + i;
            let tr = j * n + (i + 1);
            let bl = (j + 1) * n + i;
            let br = (j + 1) * n + (i + 1);
            indices.extend_from_slice(&[tl, bl, tr, tr, bl, br]);
        }
    }
    let vbuf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Water Surface VB"),
        contents: bytemuck::cast_slice(&verts),
        usage: wgpu::BufferUsages::VERTEX,
    });
    let ibuf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Water Surface IB"),
        contents: bytemuck::cast_slice(&indices),
        usage: wgpu::BufferUsages::INDEX,
    });
    (vbuf, ibuf, indices.len() as u32)
}

// Create a box mesh for rendering water volume walls (4 sides + bottom)
fn make_volume_box_mesh(device: &wgpu::Device) -> (wgpu::Buffer, wgpu::Buffer, u32) {
    // Unit box from -1 to +1 in all axes
    // We'll render bottom and 4 sides (top is the displaced surface mesh)
    let verts: Vec<[f32; 3]> = vec![
        // Bottom face (Y = -1)
        [-1.0, -1.0, -1.0], [ 1.0, -1.0, -1.0], [ 1.0, -1.0,  1.0], [-1.0, -1.0,  1.0],
        // Front face (Z = -1)
        [-1.0, -1.0, -1.0], [-1.0,  1.0, -1.0], [ 1.0,  1.0, -1.0], [ 1.0, -1.0, -1.0],
        // Back face (Z = 1)
        [-1.0, -1.0,  1.0], [ 1.0, -1.0,  1.0], [ 1.0,  1.0,  1.0], [-1.0,  1.0,  1.0],
        // Left face (X = -1)
        [-1.0, -1.0, -1.0], [-1.0, -1.0,  1.0], [-1.0,  1.0,  1.0], [-1.0,  1.0, -1.0],
        // Right face (X = 1)
        [ 1.0, -1.0, -1.0], [ 1.0,  1.0, -1.0], [ 1.0,  1.0,  1.0], [ 1.0, -1.0,  1.0],
    ];

    let indices: Vec<u32> = vec![
        // Bottom
        0, 1, 2,  0, 2, 3,
        // Front
        4, 5, 6,  4, 6, 7,
        // Back
        8, 9, 10,  8, 10, 11,
        // Left
        12, 13, 14,  12, 14, 15,
        // Right
        16, 17, 18,  16, 18, 19,
    ];

    let vbuf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Water Volume Box VB"),
        contents: bytemuck::cast_slice(&verts),
        usage: wgpu::BufferUsages::VERTEX,
    });
    let ibuf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Water Volume Box IB"),
        contents: bytemuck::cast_slice(&indices),
        usage: wgpu::BufferUsages::INDEX,
    });
    (vbuf, ibuf, indices.len() as u32)
}

// ---- Pass struct ----------------------------------------------------------------

pub struct WaterSimPass {
    // Simulation BGLs + pipelines
    sim_bgl: wgpu::BindGroupLayout,
    hitbox_bgl: wgpu::BindGroupLayout,

    drop_pipeline: wgpu::RenderPipeline,
    update_pipeline: wgpu::RenderPipeline,
    normal_pipeline: wgpu::RenderPipeline,
    hitbox_pipeline: wgpu::RenderPipeline,

    // Sim ping-pong textures
    _tex_a: wgpu::Texture,
    _tex_b: wgpu::Texture,
    view_a: wgpu::TextureView,
    view_b: wgpu::TextureView,
    front: bool,

    sampler: wgpu::Sampler,
    output_sampler: wgpu::Sampler,
    depth_sampler: wgpu::Sampler,

    // Sim uniform buffers
    drop_buf: wgpu::Buffer,
    update_buf: wgpu::Buffer,
    normal_buf: wgpu::Buffer,
    hitbox_count_buf: wgpu::Buffer,

    pending_drops: std::collections::VecDeque<DropUniform>,
    drop_staged: bool,

    // ---- Rendering resources ----

    surface_vbuf: wgpu::Buffer,
    surface_ibuf: wgpu::Buffer,
    surface_index_count: u32,

    // Volume box mesh (sides + bottom of water volume)
    volume_vbuf: wgpu::Buffer,
    volume_ibuf: wgpu::Buffer,
    volume_index_count: u32,

    _caustics_tex: wgpu::Texture,
    caustics_view: wgpu::TextureView,
    caustics_sampler: wgpu::Sampler,

    // BGLs for rendering passes
    caustics_render_bgl: wgpu::BindGroupLayout,
    render_bgl: wgpu::BindGroupLayout,

    // Rendering pipelines
    caustics_pipeline: wgpu::RenderPipeline,
    surface_above_pipeline: wgpu::RenderPipeline,
    surface_under_pipeline: wgpu::RenderPipeline,
    volume_walls_pipeline: wgpu::RenderPipeline,

    // Fallback 1×1 black texture used when pre_aa is not yet available
    _pre_aa_fallback_tex: wgpu::Texture,
    pre_aa_fallback_view: wgpu::TextureView,

    // Fallback 1×1 black texture used when GBuffer is not available
    _gbuffer_fallback_tex: wgpu::Texture,
    gbuffer_fallback_view: wgpu::TextureView,

    // Depth copy for SSR (allows reading depth while using original as attachment)
    _depth_copy_tex: wgpu::Texture,
    depth_copy_view: wgpu::TextureView,

    // Pre-TAA water composite intermediary (internal resolution, surface_format)
    _water_output_tex: wgpu::Texture,
    water_output_view: wgpu::TextureView,
    // Internal render resolution — used to build the correct viewport uniform.
    // ctx.width/height = full output res (scene.width), NOT internal res.
    internal_width: u32,
    internal_height: u32,
    // Blit pipeline: copies pre_aa -> water_output as the scene baseline
    blit_bgl: wgpu::BindGroupLayout,
    blit_pipeline: wgpu::RenderPipeline,
    blit_bg: Option<wgpu::BindGroup>,
    blit_bg_key: Option<usize>,

    // Cached bind groups (invalidated when key pointer changes)
    caustics_bg_key: Option<(usize, usize)>,
    caustics_bg: Option<wgpu::BindGroup>,

    // Underwater fullscreen effect — reads water_output, writes to scratch,
    // then blits scratch back to water_output.  This allows the shader to
    // distort the image (you can't read and write the same texture).
    _tint_scratch_tex: wgpu::Texture,
    tint_scratch_view: wgpu::TextureView,
    underwater_tint_bgl: wgpu::BindGroupLayout,
    underwater_tint_pipeline: wgpu::RenderPipeline,
}

// Shared vertex buffer layout: packed [f32; 3] positions, location 0
fn vec3_vbl() -> wgpu::VertexBufferLayout<'static> {
    wgpu::VertexBufferLayout {
        array_stride: 12,
        step_mode: wgpu::VertexStepMode::Vertex,
        attributes: &[wgpu::VertexAttribute {
            format: wgpu::VertexFormat::Float32x3,
            offset: 0,
            shader_location: 0,
        }],
    }
}

impl WaterSimPass {
    pub fn new(
        device: &wgpu::Device,
        _camera_buf: &wgpu::Buffer,
        internal_width: u32,
        internal_height: u32,
        surface_format: wgpu::TextureFormat,
    ) -> Self {
        // ------------------------------------------------------------------ sim
        let vert = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("WaterSim VS"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../shaders/fullscreen.vert.wgsl").into(),
            ),
        });
        let drop_frag = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("WaterSim Drop FS"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/drop.frag.wgsl").into()),
        });
        let update_frag = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("WaterSim Update FS"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/update.frag.wgsl").into()),
        });
        let normal_frag = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("WaterSim Normal FS"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/normal.frag.wgsl").into()),
        });
        let hitbox_frag = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("WaterSim Hitbox FS"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/hitbox.frag.wgsl").into()),
        });

        let sim_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("WaterSim BGL"),
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
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let hitbox_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("WaterSim Hitbox BGL"),
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
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let sim_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("WaterSim PL"),
            bind_group_layouts: &[Some(&sim_bgl)],
            immediate_size: 0,
        });
        let hitbox_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("WaterSim Hitbox PL"),
            bind_group_layouts: &[Some(&hitbox_bgl)],
            immediate_size: 0,
        });

        let make_sim_pipeline = |label, layout: &wgpu::PipelineLayout, frag: &wgpu::ShaderModule| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(layout),
                vertex: wgpu::VertexState {
                    module: &vert,
                    entry_point: Some("vs_main"),
                    buffers: &[],
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: frag,
                    entry_point: Some("fs_main"),
                    compilation_options: Default::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: wgpu::TextureFormat::Rgba16Float,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    ..Default::default()
                },
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                multiview_mask: None,
                cache: None,
            })
        };

        let drop_pipeline   = make_sim_pipeline("WaterSim Drop",   &sim_pl,    &drop_frag);
        let update_pipeline = make_sim_pipeline("WaterSim Update", &sim_pl,    &update_frag);
        let normal_pipeline = make_sim_pipeline("WaterSim Normal", &sim_pl,    &normal_frag);
        let hitbox_pipeline = make_sim_pipeline("WaterSim Hitbox", &hitbox_pl, &hitbox_frag);

        let make_sim_tex = |label| {
            device.create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size: wgpu::Extent3d { width: SIM_SIZE, height: SIM_SIZE, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba16Float,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            })
        };

        let tex_a = make_sim_tex("WaterSim Tex A");
        let tex_b = make_sim_tex("WaterSim Tex B");
        let view_a = tex_a.create_view(&wgpu::TextureViewDescriptor::default());
        let view_b = tex_b.create_view(&wgpu::TextureViewDescriptor::default());

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("WaterSim Internal Sampler"),
            min_filter: wgpu::FilterMode::Linear,
            mag_filter: wgpu::FilterMode::Linear,
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            ..Default::default()
        });
        let output_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("WaterSim Output Sampler"),
            min_filter: wgpu::FilterMode::Linear,
            mag_filter: wgpu::FilterMode::Linear,
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            ..Default::default()
        });
        let depth_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Water Depth Sampler"),
            min_filter: wgpu::FilterMode::Linear,
            mag_filter: wgpu::FilterMode::Linear,
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            ..Default::default()
        });

        let make_ubuf = |label, size: usize| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: size as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        };

        let drop_buf         = make_ubuf("WaterSim Drop Uniform",  std::mem::size_of::<DropUniform>());
        let update_buf       = make_ubuf("WaterSim Update Uniform", std::mem::size_of::<DeltaUniform>());
        let normal_buf       = make_ubuf("WaterSim Normal Uniform", std::mem::size_of::<DeltaUniform>());
        let hitbox_count_buf = make_ubuf("WaterSim Hitbox Count",   std::mem::size_of::<HitboxCountUniform>());

        // ----------------------------------------------------------- caustics BGL
        let caustics_render_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("WaterCaustics Render BGL"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        // --------------------------------------------------------- render BGL (pool + surface)
        let render_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Water Render BGL"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                // scene_color: opaque scene rendered before this pass (for screen-space refraction)
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // viewport: vec4f(px_w, px_h, 1/px_w, 1/px_h)
                wgpu::BindGroupLayoutEntry {
                    binding: 7,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // depth_texture: scene depth for SSR ray marching
                wgpu::BindGroupLayoutEntry {
                    binding: 8,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Depth,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // depth_sampler: linear clamp sampler for depth
                wgpu::BindGroupLayoutEntry {
                    binding: 9,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                // gbuffer_normal: scene normals for SSR quality (optional)
                wgpu::BindGroupLayoutEntry {
                    binding: 10,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
            ],
        });

        // --------------------------------------------------------- caustics texture
        let caustics_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Water Caustics Tex"),
            size: wgpu::Extent3d { width: CAUSTICS_SIZE, height: CAUSTICS_SIZE, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let caustics_view = caustics_tex.create_view(&wgpu::TextureViewDescriptor::default());
        let caustics_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Water Caustics Sampler"),
            min_filter: wgpu::FilterMode::Linear,
            mag_filter: wgpu::FilterMode::Linear,
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            ..Default::default()
        });

        // --------------------------------------------------------- pre_aa fallback (1×1 black)
        // Used when the opaque scene hasn't been rendered yet (e.g. water is the first pass).
        let pre_aa_fallback_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Water PreAA Fallback"),
            size: wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let pre_aa_fallback_view = pre_aa_fallback_tex.create_view(&wgpu::TextureViewDescriptor::default());

        // --------------------------------------------------------- gbuffer fallback (1×1 black)
        // Used when GBuffer is not available (e.g. water renders before GBuffer pass).
        let gbuffer_fallback_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Water GBuffer Fallback"),
            size: wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let gbuffer_fallback_view = gbuffer_fallback_tex.create_view(&wgpu::TextureViewDescriptor::default());

        // --------------------------------------------------------- depth copy for SSR
        // Copy of depth texture so we can sample from it while rendering with depth testing.
        // Same size as internal resolution to match the render target.
        let depth_copy_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Water Depth Copy"),
            size: wgpu::Extent3d {
                width: internal_width.max(1),
                height: internal_height.max(1),
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let depth_copy_view = depth_copy_tex.create_view(&wgpu::TextureViewDescriptor::default());

        // --------------------------------------------------------- water output intermediary
        // Owned render target at internal resolution. Water composites onto a copy
        // of pre_aa here, then publish() overwrites frame.pre_aa so TAA picks it up.
        let water_output_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Water Output Tex"),
            size: wgpu::Extent3d {
                width: internal_width.max(1),
                height: internal_height.max(1),
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: surface_format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let water_output_view = water_output_tex.create_view(&wgpu::TextureViewDescriptor::default());

        // --------------------------------------------------------- blit BGL + pipeline
        let blit_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Water Blit BGL"),
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
        let blit_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Water Blit Shader"),
            source: wgpu::ShaderSource::Wgsl(BLIT_WGSL.into()),
        });
        let blit_pl_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Water Blit PL"),
            bind_group_layouts: &[Some(&blit_bgl)],
            immediate_size: 0,
        });
        let blit_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Water Blit Pipeline"),
            layout: Some(&blit_pl_layout),
            vertex: wgpu::VertexState {
                module: &blit_shader,
                entry_point: Some("vs"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &blit_shader,
                entry_point: Some("fs"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        // --------------------------------------------------------- underwater tint
        // Scratch texture at internal resolution used by the underwater effect.
        // The effect reads water_output (scene) and writes here, then we blit
        // scratch → water_output so the result is visible downstream.
        let tint_scratch_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Water Tint Scratch"),
            size: wgpu::Extent3d {
                width: internal_width.max(1),
                height: internal_height.max(1),
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: surface_format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let tint_scratch_view = tint_scratch_tex.create_view(&wgpu::TextureViewDescriptor::default());

        let underwater_tint_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Water Underwater Tint BGL"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // scene texture — water_output at this point contains the full scene
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });
        let underwater_tint_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Water Underwater Tint Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/underwater_fog.wgsl").into()),
        });
        let underwater_tint_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Water Underwater Tint PL"),
            bind_group_layouts: &[Some(&underwater_tint_bgl)],
            immediate_size: 0,
        });
        let underwater_tint_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Water Underwater Tint Pipeline"),
            layout: Some(&underwater_tint_pl),
            vertex: wgpu::VertexState {
                module: &underwater_tint_shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &underwater_tint_shader,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_format,
                    blend: None, // full replace — writes complete processed color
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        // --------------------------------------------------------- rendering pipelines
        let caustics_pl_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Water Caustics PL"),
            bind_group_layouts: &[Some(&caustics_render_bgl)],
            immediate_size: 0,
        });
        let render_pl_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Water Render PL"),
            bind_group_layouts: &[Some(&render_bgl)],
            immediate_size: 0,
        });

        let caustics_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Water Caustics Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/caustics.wgsl").into()),
        });
        let surface_above_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Water Surface Above Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/surface_above.wgsl").into()),
        });
        let surface_under_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Water Surface Under Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/surface_under.wgsl").into()),
        });
        let volume_walls_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Water Volume Walls Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/volume_walls.wgsl").into()),
        });

        let vbl = vec3_vbl();

        let caustics_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Water Caustics Pipeline"),
            layout: Some(&caustics_pl_layout),
            vertex: wgpu::VertexState {
                module: &caustics_shader,
                entry_point: Some("vs_main"),
                buffers: &[vbl.clone()],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &caustics_shader,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::Rgba16Float,
                    blend: Some(wgpu::BlendState {
                        color: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::One,
                            dst_factor: wgpu::BlendFactor::One,
                            operation: wgpu::BlendOperation::Add,
                        },
                        alpha: wgpu::BlendComponent::OVER,
                    }),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        let surface_above_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Water Surface Above Pipeline"),
            layout: Some(&render_pl_layout),
            vertex: wgpu::VertexState {
                module: &surface_above_shader,
                entry_point: Some("vs_main"),
                buffers: &[vbl.clone()],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &surface_above_shader,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                cull_mode: Some(wgpu::Face::Back),
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: Some(false),
                depth_compare: Some(wgpu::CompareFunction::LessEqual),
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        let surface_under_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Water Surface Under Pipeline"),
            layout: Some(&render_pl_layout),
            vertex: wgpu::VertexState {
                module: &surface_under_shader,
                entry_point: Some("vs_main"),
                buffers: &[vbl.clone()],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &surface_under_shader,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                cull_mode: Some(wgpu::Face::Front),
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: Some(false),
                depth_compare: Some(wgpu::CompareFunction::LessEqual),
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        let volume_walls_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Water Volume Walls Pipeline"),
            layout: Some(&render_pl_layout),
            vertex: wgpu::VertexState {
                module: &volume_walls_shader,
                entry_point: Some("vs_main"),
                buffers: &[vbl.clone()],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &volume_walls_shader,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_format,
                    blend: Some(wgpu::BlendState {
                        color: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::SrcAlpha,
                            dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                            operation: wgpu::BlendOperation::Add,
                        },
                        alpha: wgpu::BlendComponent::OVER,
                    }),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                cull_mode: Some(wgpu::Face::Back),  // Cull back faces
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: Some(true),  // Write depth
                depth_compare: Some(wgpu::CompareFunction::LessEqual),
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        // --------------------------------------------------------- meshes
        let (surface_vbuf, surface_ibuf, surface_index_count) = make_surface_mesh(device);
        let (volume_vbuf, volume_ibuf, volume_index_count) = make_volume_box_mesh(device);

        Self {
            sim_bgl,
            hitbox_bgl,
            drop_pipeline,
            update_pipeline,
            normal_pipeline,
            hitbox_pipeline,
            _tex_a: tex_a,
            _tex_b: tex_b,
            view_a,
            view_b,
            front: true,
            sampler,
            output_sampler,
            depth_sampler,
            drop_buf,
            update_buf,
            normal_buf,
            hitbox_count_buf,
            pending_drops: std::collections::VecDeque::new(),
            drop_staged: false,
            surface_vbuf,
            surface_ibuf,
            surface_index_count,
            volume_vbuf,
            volume_ibuf,
            volume_index_count,
            _caustics_tex: caustics_tex,
            caustics_view,
            caustics_sampler,
            caustics_render_bgl,
            render_bgl,
            caustics_pipeline,
            surface_above_pipeline,
            surface_under_pipeline,
            volume_walls_pipeline,
            _pre_aa_fallback_tex: pre_aa_fallback_tex,
            pre_aa_fallback_view,
            _gbuffer_fallback_tex: gbuffer_fallback_tex,
            gbuffer_fallback_view,
            _depth_copy_tex: depth_copy_tex,
            depth_copy_view,
            _water_output_tex: water_output_tex,
            water_output_view,
            internal_width,
            internal_height,
            blit_bgl,
            blit_pipeline,
            blit_bg: None,
            blit_bg_key: None,
            caustics_bg_key: None,
            caustics_bg: None,
            _tint_scratch_tex: tint_scratch_tex,
            tint_scratch_view,
            underwater_tint_bgl,
            underwater_tint_pipeline,
        }
    }

    /// Queue a water-drop ripple to be applied next frame.
    ///
    /// `center_x`, `center_z` are in [-1, 1] sim-texture space.
    /// `radius` is a fraction of texture space (e.g. 0.05).
    /// `strength` is the height increment at the drop centre.
    pub fn add_drop(&mut self, center_x: f32, center_z: f32, radius: f32, strength: f32) {
        if self.pending_drops.len() < MAX_DROPS_BUFFERED {
            self.pending_drops.push_back(DropUniform {
                center: [center_x, center_z],
                radius,
                strength,
            });
        }
    }
}

impl RenderPass for WaterSimPass {
    fn name(&self) -> &'static str {
        "WaterSim"
    }

    fn publish<'a>(&'a self, frame: &mut libhelio::FrameResources<'a>) {
        let view = if self.front { &self.view_a } else { &self.view_b };
        frame.water_sim_texture = Some(view);
        frame.water_sim_sampler = Some(&self.output_sampler);
        frame.water_caustics = Some(&self.caustics_view);
        // Overwrite pre_aa with the water-composited intermediary so TAA
        // accumulates the water surface without any changes to the TAA pass.
        frame.pre_aa = Some(&self.water_output_view);
    }

    fn prepare(&mut self, ctx: &PrepareContext) -> HelioResult<()> {
        let delta = DeltaUniform {
            delta: [1.0 / SIM_SIZE as f32, 1.0 / SIM_SIZE as f32],
            _pad: [0.0; 2],
        };
        ctx.write_buffer(&self.update_buf, 0, bytemuck::bytes_of(&delta));
        ctx.write_buffer(&self.normal_buf, 0, bytemuck::bytes_of(&delta));

        let count = ctx.frame_resources.water_hitbox_count;
        ctx.write_buffer(
            &self.hitbox_count_buf,
            0,
            bytemuck::bytes_of(&HitboxCountUniform { count, _pad: [0; 3] }),
        );

        self.drop_staged = false;
        if let Some(drop) = self.pending_drops.pop_front() {
            ctx.write_buffer(&self.drop_buf, 0, bytemuck::bytes_of(&drop));
            self.drop_staged = true;
        }

        Ok(())
    }

    fn execute(&mut self, ctx: &mut PassContext) -> HelioResult<()> {
        // ---- 1. Hitbox displacement ------------------------------------------
        if ctx.frame.water_hitbox_count > 0 {
            if let Some(hitboxes_buf) = ctx.frame.water_hitboxes {
                // SAFETY: view_a and view_b are separate, non-overlapping wgpu
                // TextureView allocations. We render FROM src INTO dst — never
                // the same texture for both roles simultaneously.
                let src: &wgpu::TextureView =
                    if self.front { &self.view_a } else { &self.view_b };
                let dst_ptr: *const wgpu::TextureView =
                    if self.front { &self.view_b } else { &self.view_a };

                let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("WaterSim Hitbox BG"),
                    layout: &self.hitbox_bgl,
                    entries: &[
                        wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(src) },
                        wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&self.sampler) },
                        wgpu::BindGroupEntry { binding: 2, resource: self.hitbox_count_buf.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 3, resource: hitboxes_buf.as_entire_binding() },
                    ],
                });

                let dst = unsafe { &*dst_ptr };
                let color_attachments = [Some(wgpu::RenderPassColorAttachment {
                    view: dst,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store },
                })];
                let desc = wgpu::RenderPassDescriptor {
                    label: Some("WaterSim Hitbox"),
                    color_attachments: &color_attachments,
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                    multiview_mask: None,
                };
                let mut pass = ctx.begin_render_pass(&desc);
                pass.set_pipeline(&self.hitbox_pipeline);
                pass.set_bind_group(0, &bg, &[]);
                pass.draw(0..6, 0..1);
                drop(pass);
                self.front = !self.front;
            }
        }

        // ---- 2. Drop ripple --------------------------------------------------
        if self.drop_staged {
            // SAFETY: same as hitbox block above.
            let src: &wgpu::TextureView =
                if self.front { &self.view_a } else { &self.view_b };
            let dst_ptr: *const wgpu::TextureView =
                if self.front { &self.view_b } else { &self.view_a };

            let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("WaterSim Drop BG"),
                layout: &self.sim_bgl,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(src) },
                    wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&self.sampler) },
                    wgpu::BindGroupEntry { binding: 2, resource: self.drop_buf.as_entire_binding() },
                ],
            });

            let dst = unsafe { &*dst_ptr };
            let color_attachments = [Some(wgpu::RenderPassColorAttachment {
                view: dst,
                resolve_target: None,
                depth_slice: None,
                ops: wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store },
            })];
            let desc = wgpu::RenderPassDescriptor {
                label: Some("WaterSim Drop"),
                color_attachments: &color_attachments,
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            };
            let mut pass = ctx.begin_render_pass(&desc);
            pass.set_pipeline(&self.drop_pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.draw(0..6, 0..1);
            drop(pass);
            self.front = !self.front;
        }

        // ---- 3. Wave propagation (2 steps per frame) ------------------------
        for i in 0..2u32 {
            // SAFETY: same as hitbox block above.
            let src: &wgpu::TextureView =
                if self.front { &self.view_a } else { &self.view_b };
            let dst_ptr: *const wgpu::TextureView =
                if self.front { &self.view_b } else { &self.view_a };

            let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("WaterSim Update BG"),
                layout: &self.sim_bgl,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(src) },
                    wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&self.sampler) },
                    wgpu::BindGroupEntry { binding: 2, resource: self.update_buf.as_entire_binding() },
                ],
            });

            let dst = unsafe { &*dst_ptr };
            let label = if i == 0 { "WaterSim Update 1" } else { "WaterSim Update 2" };
            let color_attachments = [Some(wgpu::RenderPassColorAttachment {
                view: dst,
                resolve_target: None,
                depth_slice: None,
                ops: wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store },
            })];
            let desc = wgpu::RenderPassDescriptor {
                label: Some(label),
                color_attachments: &color_attachments,
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            };
            let mut pass = ctx.begin_render_pass(&desc);
            pass.set_pipeline(&self.update_pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.draw(0..6, 0..1);
            drop(pass);
            self.front = !self.front;
        }

        // ---- 4. Normal recomputation ----------------------------------------
        {
            // SAFETY: same as hitbox block above.
            let src: &wgpu::TextureView =
                if self.front { &self.view_a } else { &self.view_b };
            let dst_ptr: *const wgpu::TextureView =
                if self.front { &self.view_b } else { &self.view_a };

            let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("WaterSim Normal BG"),
                layout: &self.sim_bgl,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(src) },
                    wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&self.sampler) },
                    wgpu::BindGroupEntry { binding: 2, resource: self.normal_buf.as_entire_binding() },
                ],
            });

            let dst = unsafe { &*dst_ptr };
            let color_attachments = [Some(wgpu::RenderPassColorAttachment {
                view: dst,
                resolve_target: None,
                depth_slice: None,
                ops: wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store },
            })];
            let desc = wgpu::RenderPassDescriptor {
                label: Some("WaterSim Normal"),
                color_attachments: &color_attachments,
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            };
            let mut pass = ctx.begin_render_pass(&desc);
            pass.set_pipeline(&self.normal_pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.draw(0..6, 0..1);
            drop(pass);
            self.front = !self.front;
        }

        // ---- 5. Caustics projection ------------------------------------------
        if ctx.frame.water_volume_count > 0 {
            if let Some(vols_buf) = ctx.frame.water_volumes {
                let sim_view = if self.front { &self.view_a } else { &self.view_b };

                let vols_key = vols_buf as *const wgpu::Buffer as usize;
                let sim_key = sim_view as *const wgpu::TextureView as usize;
                let new_key = (vols_key, sim_key);

                if self.caustics_bg_key != Some(new_key) {
                    self.caustics_bg = Some(ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("Water Caustics BG"),
                        layout: &self.caustics_render_bgl,
                        entries: &[
                            wgpu::BindGroupEntry { binding: 0, resource: vols_buf.as_entire_binding() },
                            wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(sim_view) },
                            wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(&self.output_sampler) },
                        ],
                    }));
                    self.caustics_bg_key = Some(new_key);
                }

                let cau_attachments = [Some(wgpu::RenderPassColorAttachment {
                    view: &self.caustics_view,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })];
                let desc = wgpu::RenderPassDescriptor {
                    label: Some("Water Caustics"),
                    color_attachments: &cau_attachments,
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                    multiview_mask: None,
                };
                let mut pass = ctx.begin_render_pass(&desc);
                pass.set_pipeline(&self.caustics_pipeline);
                pass.set_bind_group(0, self.caustics_bg.as_ref().unwrap(), &[]);
                pass.set_vertex_buffer(0, self.surface_vbuf.slice(..));
                pass.set_index_buffer(self.surface_ibuf.slice(..), wgpu::IndexFormat::Uint32);
                pass.draw_indexed(0..self.surface_index_count, 0, 0..1);
                drop(pass);
            }
        }

        // ---- 6. Blit pre_aa → water_output (scene baseline) ----------------------
        // Always runs so TAA always receives a valid intermediary as its pre_aa
        // input, even when there are no water volumes this frame.
        // NOTE: use self.caustics_view directly — it was filled in stage 5 this
        // same frame. ctx.frame.water_caustics is None during execute() because
        // publish() hasn't run yet, so we must NOT guard on it here.
        let scene_view: &wgpu::TextureView = ctx.frame.pre_aa
            .unwrap_or(&self.pre_aa_fallback_view);
        let blit_key = scene_view as *const _ as usize;
        if self.blit_bg_key != Some(blit_key) {
            self.blit_bg = Some(ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Water Blit BG"),
                layout: &self.blit_bgl,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(scene_view) },
                    wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&self.output_sampler) },
                ],
            }));
            self.blit_bg_key = Some(blit_key);
        }
        {
            let attachments = [Some(wgpu::RenderPassColorAttachment {
                view: &self.water_output_view,
                resolve_target: None,
                depth_slice: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
            })];
            let desc = wgpu::RenderPassDescriptor {
                label: Some("Water Blit"),
                color_attachments: &attachments,
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            };
            let mut pass = ctx.begin_render_pass(&desc);
            pass.set_pipeline(&self.blit_pipeline);
            pass.set_bind_group(0, self.blit_bg.as_ref().unwrap(), &[]);
            pass.draw(0..3, 0..1);
        }

        // ---- 7. Water surface render → water_output --------------------------
        if ctx.frame.water_volume_count > 0 {
            if let Some(vols_buf) = ctx.frame.water_volumes {
                let sim_view = if self.front { &self.view_a } else { &self.view_b };

                // Per-frame viewport uniform: (w, h, 1/w, 1/h)
                // IMPORTANT: water now renders at INTERNAL resolution (pre-TAA).
                // ctx.width/height = full output res (scene.width/height), which is WRONG here.
                // Use self.internal_width/height so depth_coord and screen_uv are correct.
                let vp = [
                    self.internal_width  as f32, self.internal_height as f32,
                    1.0 / self.internal_width  as f32, 1.0 / self.internal_height as f32,
                ];
                let viewport_buf = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Water Viewport"),
                    contents: bytemuck::cast_slice(&vp),
                    usage: wgpu::BufferUsages::UNIFORM,
                });

                // Copy depth texture for SSR sampling
                // This allows us to sample from the copy while using the original as a depth attachment
                let src_depth_tex = ctx.frame.depth_texture.expect("Depth texture required for water SSR");
                ctx.encoder.copy_texture_to_texture(
                    src_depth_tex.as_image_copy(),
                    self._depth_copy_tex.as_image_copy(),
                    wgpu::Extent3d {
                        width: self.internal_width,
                        height: self.internal_height,
                        depth_or_array_layers: 1,
                    },
                );

                // Get GBuffer normal texture (fallback to 1×1 black if not available)
                let gbuffer_normal_view = ctx.frame.gbuffer
                    .map(|gb| gb.normal)
                    .unwrap_or(&self.gbuffer_fallback_view);

                let render_bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("Water Render BG"),
                    layout: &self.render_bgl,
                    entries: &[
                        wgpu::BindGroupEntry { binding: 0, resource: ctx.scene.camera.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 1, resource: vols_buf.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(sim_view) },
                        wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::Sampler(&self.output_sampler) },
                        wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::TextureView(&self.caustics_view) },
                        wgpu::BindGroupEntry { binding: 5, resource: wgpu::BindingResource::Sampler(&self.caustics_sampler) },
                        wgpu::BindGroupEntry { binding: 6, resource: wgpu::BindingResource::TextureView(scene_view) },
                        wgpu::BindGroupEntry { binding: 7, resource: viewport_buf.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 8, resource: wgpu::BindingResource::TextureView(&self.depth_copy_view) },
                        wgpu::BindGroupEntry { binding: 9, resource: wgpu::BindingResource::Sampler(&self.depth_sampler) },
                        wgpu::BindGroupEntry { binding: 10, resource: wgpu::BindingResource::TextureView(gbuffer_normal_view) },
                    ],
                });

                // -- VOLUMETRIC WATER RENDERING ORDER --
                // Render in correct order: walls first (depth write), then surface (refraction/reflection)
                
                let depth_view = ctx.depth;
                let color_attachments = [Some(wgpu::RenderPassColorAttachment {
                    view: &self.water_output_view,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store },
                })];

                // 1. Water volume walls (sides + bottom) - Render FIRST to establish volume depth
                {
                    let mut pass = ctx.encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("Water Volume Walls"),
                        color_attachments: &color_attachments,
                        depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                            view: depth_view,
                            depth_ops: Some(wgpu::Operations {
                                load: wgpu::LoadOp::Load,
                                store: wgpu::StoreOp::Store,  // Write depth so surface renders on top
                            }),
                            stencil_ops: None,
                        }),
                        timestamp_writes: None,
                        occlusion_query_set: None,
                        multiview_mask: None,
                    });
                    pass.set_pipeline(&self.volume_walls_pipeline);
                    pass.set_bind_group(0, &render_bg, &[]);
                    pass.set_vertex_buffer(0, self.volume_vbuf.slice(..));
                    pass.set_index_buffer(self.volume_ibuf.slice(..), wgpu::IndexFormat::Uint32);
                    pass.draw_indexed(0..self.volume_index_count, 0, 0..1);
                }

                // 2. Water surface (top) - above-water view with refraction
                {
                    let mut pass = ctx.encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("Water Surface Above"),
                        color_attachments: &color_attachments,
                        depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                            view: depth_view,
                            depth_ops: Some(wgpu::Operations {
                                load: wgpu::LoadOp::Load,
                                store: wgpu::StoreOp::Store,
                            }),
                            stencil_ops: None,
                        }),
                        timestamp_writes: None,
                        occlusion_query_set: None,
                        multiview_mask: None,
                    });
                    pass.set_pipeline(&self.surface_above_pipeline);
                    pass.set_bind_group(0, &render_bg, &[]);
                    pass.set_vertex_buffer(0, self.surface_vbuf.slice(..));
                    pass.set_index_buffer(self.surface_ibuf.slice(..), wgpu::IndexFormat::Uint32);
                    pass.draw_indexed(0..self.surface_index_count, 0, 0..1);
                }

                // 3. Water surface (bottom) - underwater view (looking up through water)
                {
                    let mut pass = ctx.encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("Water Surface Under"),
                        color_attachments: &color_attachments,
                        depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                            view: depth_view,
                            depth_ops: Some(wgpu::Operations {
                                load: wgpu::LoadOp::Load,
                                store: wgpu::StoreOp::Store,
                            }),
                            stencil_ops: None,
                        }),
                        timestamp_writes: None,
                        occlusion_query_set: None,
                        multiview_mask: None,
                    });
                    pass.set_pipeline(&self.surface_under_pipeline);
                    pass.set_bind_group(0, &render_bg, &[]);
                    pass.set_vertex_buffer(0, self.surface_vbuf.slice(..));
                    pass.set_index_buffer(self.surface_ibuf.slice(..), wgpu::IndexFormat::Uint32);
                    pass.draw_indexed(0..self.surface_index_count, 0, 0..1);
                }

                // 4. Underwater effect — reads water_output (scene), writes to scratch.
                // The shader does distortion + chromatic aberration + color tint.
                // When the camera is above water the shader passes through unchanged.
                {
                    let tint_bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("Water Underwater Tint BG"),
                        layout: &self.underwater_tint_bgl,
                        entries: &[
                            wgpu::BindGroupEntry { binding: 0, resource: ctx.scene.camera.as_entire_binding() },
                            wgpu::BindGroupEntry { binding: 1, resource: vols_buf.as_entire_binding() },
                            wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(&self.water_output_view) },
                            wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::Sampler(&self.depth_sampler) },
                        ],
                    });
                    // Draw to scratch (can't read and write water_output simultaneously)
                    let tint_attachments = [Some(wgpu::RenderPassColorAttachment {
                        view: &self.tint_scratch_view,
                        resolve_target: None,
                        depth_slice: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                            store: wgpu::StoreOp::Store,
                        },
                    })];
                    let mut tint_pass = ctx.encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("Water Underwater Tint"),
                        color_attachments: &tint_attachments,
                        depth_stencil_attachment: None,
                        timestamp_writes: None,
                        occlusion_query_set: None,
                        multiview_mask: None,
                    });
                    tint_pass.set_pipeline(&self.underwater_tint_pipeline);
                    tint_pass.set_bind_group(0, &tint_bg, &[]);
                    tint_pass.draw(0..3, 0..1);
                    drop(tint_pass);

                    // Blit scratch → water_output to make the result visible
                    let scratch_blit_bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("Water Tint Blit BG"),
                        layout: &self.blit_bgl,
                        entries: &[
                            wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&self.tint_scratch_view) },
                            wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&self.output_sampler) },
                        ],
                    });
                    let blit_attachments = [Some(wgpu::RenderPassColorAttachment {
                        view: &self.water_output_view,
                        resolve_target: None,
                        depth_slice: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                            store: wgpu::StoreOp::Store,
                        },
                    })];
                    let mut blit_pass = ctx.encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("Water Tint Blit Back"),
                        color_attachments: &blit_attachments,
                        depth_stencil_attachment: None,
                        timestamp_writes: None,
                        occlusion_query_set: None,
                        multiview_mask: None,
                    });
                    blit_pass.set_pipeline(&self.blit_pipeline);
                    blit_pass.set_bind_group(0, &scratch_blit_bg, &[]);
                    blit_pass.draw(0..3, 0..1);
                }
            }
        }

        Ok(())
    }
}
