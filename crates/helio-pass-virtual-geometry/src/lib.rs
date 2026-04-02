//! Virtual Geometry pass — GPU-driven per-meshlet culling + GBuffer draw.
//!
//! # Architecture
//!
//! ```text
//! CPU each frame:  O(1) — hand VgFrameData byte-slices to the pass
//! GPU cull:        1 compute dispatch (64 threads/group × ceil(meshlets/64) groups)
//!                  → writes DrawIndexedIndirect per meshlet (instance_count = 0 or 1)
//! GPU draw:        multi_draw_indexed_indirect on the same indirect buffer
//!                  → skips commands with instance_count = 0 automatically
//! ```
//!
//! This gives true O(1) CPU even with tens of millions of triangles — the GPU
//! decides what to render each frame.

use std::num::NonZeroU32;

use bytemuck::{Pod, Zeroable};
use helio_v3::{PassContext, PrepareContext, RenderPass, Result as HelioResult};

/// Bindless texture array size per shader stage.
/// Capped at 16 on wasm32 (WebGPU baseline); 256 on native.
#[cfg(not(target_arch = "wasm32"))]
const MAX_TEXTURES: usize = 256;
#[cfg(target_arch = "wasm32")]
const MAX_TEXTURES: usize = 16;

// ── Uniform types ─────────────────────────────────────────────────────────────

/// Per-frame globals uploaded to the GPU (matches `Globals` in vg_gbuffer.wgsl).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct VgGlobals {
    frame: u32,
    delta_time: f32,
    light_count: u32,
    ambient_intensity: f32,
    ambient_color: [f32; 4],
    rc_world_min: [f32; 4],
    rc_world_max: [f32; 4],
    csm_splits: [f32; 4],
    debug_mode: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

/// Controls how aggressively distant objects are simplified.
///
/// Each level raises the screen-coverage thresholds at which LOD transitions
/// fire, so higher quality = full-detail geometry is visible at greater
/// distances.  All values are fraction of screen height covered by the
/// object's bounding sphere (`screen_radius = obj_radius * cot(fov/2) / dist`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum LodQuality {
    /// Aggressive simplification — good for low-end GPUs / perf testing.
    /// LOD0 only when the object covers ≥ 2 % of screen height.
    Low,
    /// Balanced default.  LOD0 ≥ 5 %, LOD1 ≥ 1 %.
    #[default]
    Medium,
    /// Sharper detail at a distance.  LOD0 ≥ 10 %, LOD1 ≥ 2 %.
    High,
    /// Near-cinematic — LOD transitions barely visible.  LOD0 ≥ 18 %, LOD1 ≥ 4 %.
    Ultra,
}

impl LodQuality {
    /// Returns `(lod_s0, lod_s1)` screen-radius thresholds for this quality level.
    ///
    /// `lod_s0` : transition from LOD 0 → 1 (full → medium)  
    /// `lod_s1` : transition from LOD 1 → 2 (medium → coarse)
    pub fn thresholds(self) -> (f32, f32) {
        match self {
            LodQuality::Low => (0.02, 0.004),
            LodQuality::Medium => (0.05, 0.010),
            LodQuality::High => (0.10, 0.020),
            LodQuality::Ultra => (0.18, 0.040),
        }
    }
}

/// Cull uniforms passed to the compute shader.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct CullUniforms {
    meshlet_count: u32,
    // Nanite-style screen-space LOD thresholds (0..1 fraction of screen height).
    // screen_radius = (obj_radius * cot(fov/2)) / dist — FOV/scale/resolution-invariant.
    lod_s0: f32, // LOD 0→1 when screen_radius drops below this (e.g. 0.05 = 5 % of screen)
    lod_s1: f32, // LOD 1→2 when screen_radius drops below this (e.g. 0.01 = 1 % of screen)
    _pad2: u32,
}

// ── Pass struct ───────────────────────────────────────────────────────────────

pub struct VirtualGeometryPass {
    // ── Compute (cull) ────────────────────────────────────────────────────────
    cull_pipeline: wgpu::ComputePipeline,
    cull_bgl: wgpu::BindGroupLayout,
    /// Rebuilt whenever owned buffers are reallocated (set to None then).
    cull_bind_group: Option<wgpu::BindGroup>,
    cull_buf: wgpu::Buffer, // CullUniforms

    // ── Render (VG GBuffer) ───────────────────────────────────────────────────
    draw_pipeline: wgpu::RenderPipeline,
    /// Separate pipeline using `fs_debug` (requires SHADER_PRIMITIVE_INDEX).
    /// Only bound when debug_mode == 20; normal pipeline has zero primitive_index overhead.
    debug_draw_pipeline: Option<wgpu::RenderPipeline>,
    draw_bgl_0: wgpu::BindGroupLayout,
    draw_bgl_1: wgpu::BindGroupLayout,
    draw_bg_0: Option<wgpu::BindGroup>,
    draw_bg_1: Option<wgpu::BindGroup>,
    bg1_version: Option<u64>,
    globals_buf: wgpu::Buffer,

    // ── VG data buffers (owned; rebuilt when buffer_version changes) ──────────
    meshlet_buf: wgpu::Buffer,
    instance_buf: wgpu::Buffer,
    /// Compacted visible-only draw commands (AAA approach: written by atomic appending).
    indirect_buf: wgpu::Buffer,
    /// Single u32 written by the cull shader via atomicAdd — the visible meshlet count.
    /// Passed as the count argument to multi_draw_indexed_indirect_count.
    draw_count_buf: wgpu::Buffer,
    /// True when the device supports MULTI_DRAW_INDIRECT_COUNT (Vulkan 1.2+, DX12 tier2).
    use_count_indirect: bool,
    /// Current debug mode inherited from the renderer — written into globals_buf each prepare().
    pub debug_mode: u32,
    /// LOD quality preset — controls the screen-size thresholds at which meshlets
    /// switch from LOD 0 → 1 → 2.  Defaults to `LodQuality::Medium`.
    pub lod_quality: LodQuality,
    last_version: u64,
    last_meshlet_count: u32,
}

// ─── initial buffer size constants ───────────────────────────────────────────

/// Starting capacity (in entries) for GPU VG buffers.  Grown on demand.
const INITIAL_MESHLETS: u64 = 1024;
const INITIAL_INSTANCES: u64 = 256;

impl VirtualGeometryPass {
    /// Create the virtual geometry pass.
    ///
    /// `camera_buf` must be the shared scene camera uniform buffer (stable for
    /// the lifetime of the renderer — camera is fixed-size and never reallocates).
    pub fn new(device: &wgpu::Device, camera_buf: &wgpu::Buffer) -> Self {
        // ── Shaders ───────────────────────────────────────────────────────────
        let cull_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("VG Cull Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/vg_cull.wgsl").into()),
        });
        let draw_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("VG GBuffer Shader"),
            source: wgpu::ShaderSource::Wgsl({
                let s = include_str!("../shaders/vg_gbuffer.wgsl")
                    .replace(
                        "binding_array<texture_2d<f32>, 256>",
                        &format!("binding_array<texture_2d<f32>, {MAX_TEXTURES}>"),
                    )
                    .replace(
                        "binding_array<sampler, 256>",
                        &format!("binding_array<sampler, {MAX_TEXTURES}>"),
                    );
                // WebGPU does not support binding arrays; collapse to single texture/sampler.
                #[cfg(target_arch = "wasm32")]
                let s = s
                    .replace(
                        &format!("binding_array<sampler, {MAX_TEXTURES}>"),
                        "sampler",
                    )
                    .replace("scene_samplers[slot.texture_index]", "scene_samplers")
                    .replace(
                        &format!("binding_array<texture_2d<f32>, {MAX_TEXTURES}>"),
                        "texture_2d<f32>",
                    )
                    .replace("scene_textures[slot.texture_index]", "scene_textures")
                    // textureSample requires uniform control flow on WebGPU; use textureSampleLevel (LOD 0) instead.
                    .replace(
                        "return textureSample(scene_textures, scene_samplers, uv);",
                        "return textureSampleLevel(scene_textures, scene_samplers, uv, 0.0);",
                    )
                    // @builtin(primitive_index) requires SHADER_PRIMITIVE_INDEX which WebGPU may not expose.
                    .replace(", @builtin(primitive_index) prim_id: u32", "")
                    .replace(
                        "    var h = prim_id *",
                        "    let prim_id: u32 = 0u;\n    var h = prim_id *",
                    );
                s.into()
            }),
        });

        // ── Initial GPU buffers (grown in prepare() when data arrives) ─────────
        let meshlet_buf = Self::make_meshlet_buf(device, INITIAL_MESHLETS);
        let instance_buf = Self::make_instance_buf(device, INITIAL_INSTANCES);
        let indirect_buf = Self::make_indirect_buf(device, INITIAL_MESHLETS);
        let draw_count_buf = Self::make_draw_count_buf(device);
        let use_count_indirect = device
            .features()
            .contains(wgpu::Features::MULTI_DRAW_INDIRECT_COUNT);

        // ── Cull uniform buffer ───────────────────────────────────────────────
        let cull_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("VG CullUniforms"),
            size: std::mem::size_of::<CullUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // ── Globals buffer (for draw pass) ────────────────────────────────────
        let globals_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("VG Globals"),
            size: std::mem::size_of::<VgGlobals>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // ── Cull BGL: camera(u) + cull_uni(u) + meshlets(s) + instances(s) + indirect(s rw) ──
        let cull_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("VG Cull BGL"),
            entries: &[
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
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
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
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // binding 5: atomic draw count — written by atomicAdd in cs_cull
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        // Initial cull bind group (built with starting empty buffers; rebuilt if they grow).
        let cull_bind_group = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("VG Cull BG"),
            layout: &cull_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: camera_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: cull_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: meshlet_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: instance_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: indirect_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: draw_count_buf.as_entire_binding(),
                },
            ],
        }));

        // ── Cull pipeline ─────────────────────────────────────────────────────
        let cull_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("VG Cull PL"),
            bind_group_layouts: &[Some(&cull_bgl)],
            immediate_size: 0,
        });
        let cull_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("VG Cull Pipeline"),
            layout: Some(&cull_pipeline_layout),
            module: &cull_shader,
            entry_point: Some("cs_cull"),
            compilation_options: Default::default(),
            cache: None,
        });

        // ── Draw BGL 0: camera(u VERT|FRAG) + globals(u FRAG) + vg_instances(s VERT) ──
        let draw_bgl_0 = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("VG Draw BGL0"),
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
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        // Initial draw bind group 0.
        let draw_bg_0 = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("VG Draw BG0"),
            layout: &draw_bgl_0,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: camera_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: globals_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: instance_buf.as_entire_binding(),
                },
            ],
        }));

        // ── Draw BGL 1: materials + material_textures + bindless arrays ───────
        let draw_bgl_1 = create_material_bgl(device);

        // ── Draw pipeline ─────────────────────────────────────────────────────
        let draw_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("VG Draw PL"),
            bind_group_layouts: &[Some(&draw_bgl_0), Some(&draw_bgl_1)],
            immediate_size: 0,
        });
        let vg_vertex_buffers = &[wgpu::VertexBufferLayout {
            array_stride: 40,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32x3,
                    offset: 0,
                    shader_location: 0,
                },
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32,
                    offset: 12,
                    shader_location: 1,
                },
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32x2,
                    offset: 16,
                    shader_location: 2,
                },
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Uint32,
                    offset: 32,
                    shader_location: 3,
                },
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Uint32,
                    offset: 36,
                    shader_location: 4,
                },
            ],
        }];
        let gbuffer_targets = &[
            Some(wgpu::ColorTargetState {
                format: wgpu::TextureFormat::Rgba8Unorm,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            }),
            Some(wgpu::ColorTargetState {
                format: wgpu::TextureFormat::Rgba16Float,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            }),
            Some(wgpu::ColorTargetState {
                format: wgpu::TextureFormat::Rgba8Unorm,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            }),
            Some(wgpu::ColorTargetState {
                format: wgpu::TextureFormat::Rgba16Float,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            }),
        ];
        let draw_primitive = wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            cull_mode: Some(wgpu::Face::Back),
            ..Default::default()
        };
        let draw_depth = Some(wgpu::DepthStencilState {
            format: wgpu::TextureFormat::Depth32Float,
            depth_write_enabled: Some(true),
            depth_compare: Some(wgpu::CompareFunction::LessEqual),
            stencil: wgpu::StencilState::default(),
            bias: wgpu::DepthBiasState::default(),
        });

        let draw_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("VG Draw Pipeline"),
            layout: Some(&draw_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &draw_shader,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: vg_vertex_buffers,
            },
            fragment: Some(wgpu::FragmentState {
                module: &draw_shader,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: gbuffer_targets,
            }),
            primitive: draw_primitive,
            depth_stencil: draw_depth.clone(),
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        // Debug pipeline: same layout / vertex state, but uses fs_debug which
        // has @builtin(primitive_index).  Only created when the GPU supports it.
        let debug_draw_pipeline = if device
            .features()
            .contains(wgpu::Features::SHADER_PRIMITIVE_INDEX)
        {
            Some(
                device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                    label: Some("VG Debug Pipeline"),
                    layout: Some(&draw_pipeline_layout),
                    vertex: wgpu::VertexState {
                        module: &draw_shader,
                        entry_point: Some("vs_main"),
                        compilation_options: Default::default(),
                        buffers: vg_vertex_buffers,
                    },
                    fragment: Some(wgpu::FragmentState {
                        module: &draw_shader,
                        entry_point: Some("fs_debug"),
                        compilation_options: Default::default(),
                        targets: gbuffer_targets,
                    }),
                    primitive: draw_primitive,
                    depth_stencil: draw_depth,
                    multisample: wgpu::MultisampleState::default(),
                    multiview_mask: None,
                    cache: None,
                }),
            )
        } else {
            None
        };

        Self {
            cull_pipeline,
            cull_bgl,
            cull_bind_group,
            cull_buf,
            draw_pipeline,
            debug_draw_pipeline,
            draw_bgl_0,
            draw_bgl_1,
            draw_bg_0,
            draw_bg_1: None,
            bg1_version: None,
            globals_buf,
            meshlet_buf,
            instance_buf,
            indirect_buf,
            draw_count_buf,
            use_count_indirect,
            debug_mode: 0,
            lod_quality: LodQuality::default(),
            last_version: u64::MAX, // force upload on first frame
            last_meshlet_count: 0,
        }
    }

    // ── Buffer helpers ────────────────────────────────────────────────────────

    fn make_meshlet_buf(device: &wgpu::Device, capacity: u64) -> wgpu::Buffer {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("VG Meshlet Buffer"),
            size: capacity * 64, // GpuMeshletEntry = 64 bytes
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }

    fn make_instance_buf(device: &wgpu::Device, capacity: u64) -> wgpu::Buffer {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("VG Instance Buffer"),
            size: capacity * 144, // GpuInstanceData = 144 bytes
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }

    fn make_indirect_buf(device: &wgpu::Device, capacity: u64) -> wgpu::Buffer {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("VG Indirect Buffer"),
            size: capacity * 20, // DrawIndexedIndirectArgs = 20 bytes
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::INDIRECT
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }

    fn make_draw_count_buf(device: &wgpu::Device) -> wgpu::Buffer {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("VG Draw Count"),
            size: 4, // single u32 atomic counter
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::INDIRECT
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }

    /// Rebuild cull + draw_bg_0 bind groups after buffer reallocation.
    fn rebuild_owned_bind_groups(&mut self, device: &wgpu::Device, camera_buf: &wgpu::Buffer) {
        self.cull_bind_group = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("VG Cull BG"),
            layout: &self.cull_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: camera_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.cull_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.meshlet_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.instance_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: self.indirect_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: self.draw_count_buf.as_entire_binding(),
                },
            ],
        }));
        self.draw_bg_0 = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("VG Draw BG0"),
            layout: &self.draw_bgl_0,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: camera_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.globals_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.instance_buf.as_entire_binding(),
                },
            ],
        }));
    }
}

impl RenderPass for VirtualGeometryPass {
    fn name(&self) -> &'static str {
        "VirtualGeometry"
    }

    fn prepare(&mut self, ctx: &PrepareContext) -> HelioResult<()> {
        let Some(vg) = ctx.frame_resources.vg else {
            return Ok(());
        };

        // ── Upload meshlet / instance data if version changed ─────────────────
        if vg.buffer_version != self.last_version {
            let camera_buf = ctx.scene.camera.buffer();
            let mut grew = false;

            let meshlet_capacity = self.meshlet_buf.size() / 64;
            if (vg.meshlet_count as u64) > meshlet_capacity {
                self.meshlet_buf = Self::make_meshlet_buf(ctx.device, vg.meshlet_count as u64 * 2);
                self.indirect_buf =
                    Self::make_indirect_buf(ctx.device, vg.meshlet_count as u64 * 2);
                grew = true;
            }
            let instance_capacity = self.instance_buf.size() / 144;
            if (vg.instance_count as u64) > instance_capacity {
                self.instance_buf =
                    Self::make_instance_buf(ctx.device, vg.instance_count as u64 * 2);
                grew = true;
            }

            if grew {
                self.rebuild_owned_bind_groups(ctx.device, camera_buf);
            }

            ctx.write_buffer(&self.meshlet_buf, 0, vg.meshlets);
            ctx.write_buffer(&self.instance_buf, 0, vg.instances);
            self.last_version = vg.buffer_version;
            self.last_meshlet_count = vg.meshlet_count;
        }

        if self.last_meshlet_count == 0 {
            return Ok(());
        }

        // ── Upload cull uniforms ──────────────────────────────────────────────
        let (lod_s0, lod_s1) = self.lod_quality.thresholds();
        let cull_uni = CullUniforms {
            meshlet_count: self.last_meshlet_count,
            lod_s0,
            lod_s1,
            _pad2: 0,
        };
        ctx.write_buffer(&self.cull_buf, 0, bytemuck::bytes_of(&cull_uni));

        // ── Material bind group (rebuild when texture version changes) ─────────
        let Some(main_scene) = ctx.frame_resources.main_scene else {
            return Ok(());
        };
        if self.draw_bg_1.is_none()
            || self.bg1_version != Some(main_scene.material_textures.version)
        {
            self.draw_bg_1 = Some(
                ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("VG Draw BG1"),
                    layout: &self.draw_bgl_1,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: ctx.scene.materials.buffer().as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: main_scene
                                .material_textures
                                .material_textures
                                .as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            #[cfg(not(target_arch = "wasm32"))]
                            resource: wgpu::BindingResource::TextureViewArray(
                                main_scene.material_textures.texture_views,
                            ),
                            #[cfg(target_arch = "wasm32")]
                            resource: wgpu::BindingResource::TextureView(
                                main_scene
                                    .material_textures
                                    .texture_views
                                    .first()
                                    .copied()
                                    .expect("scene must have at least one texture view"),
                            ),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            #[cfg(not(target_arch = "wasm32"))]
                            resource: wgpu::BindingResource::SamplerArray(
                                main_scene.material_textures.samplers,
                            ),
                            #[cfg(target_arch = "wasm32")]
                            resource: wgpu::BindingResource::Sampler(
                                main_scene
                                    .material_textures
                                    .samplers
                                    .first()
                                    .copied()
                                    .expect("scene must have at least one sampler"),
                            ),
                        },
                    ],
                }),
            );
            self.bg1_version = Some(main_scene.material_textures.version);
        }

        // ── Upload globals ────────────────────────────────────────────────────
        let globals = VgGlobals {
            frame: ctx.frame_num as u32,
            delta_time: 0.016,
            light_count: ctx.scene.lights.len() as u32,
            ambient_intensity: main_scene.ambient_intensity,
            ambient_color: [
                main_scene.ambient_color[0],
                main_scene.ambient_color[1],
                main_scene.ambient_color[2],
                0.0,
            ],
            rc_world_min: [
                main_scene.rc_world_min[0],
                main_scene.rc_world_min[1],
                main_scene.rc_world_min[2],
                0.0,
            ],
            rc_world_max: [
                main_scene.rc_world_max[0],
                main_scene.rc_world_max[1],
                main_scene.rc_world_max[2],
                0.0,
            ],
            csm_splits: [5.0, 20.0, 60.0, 200.0],
            debug_mode: self.debug_mode,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        };
        ctx.write_buffer(&self.globals_buf, 0, bytemuck::bytes_of(&globals));

        Ok(())
    }

    fn execute(&mut self, ctx: &mut PassContext) -> HelioResult<()> {
        if self.last_meshlet_count == 0 || ctx.resources.vg.is_none() {
            return Ok(());
        }

        let Some(cull_bg) = self.cull_bind_group.as_ref() else {
            return Ok(());
        };
        let Some(draw_bg0) = self.draw_bg_0.as_ref() else {
            return Ok(());
        };
        let Some(draw_bg1) = self.draw_bg_1.as_ref() else {
            return Ok(());
        };
        let Some(main_scene) = ctx.resources.main_scene else {
            return Ok(());
        };
        let Some(gbuffer) = ctx.resources.gbuffer else {
            return Ok(());
        };

        let meshlet_count = self.last_meshlet_count;

        // Reset the atomic draw count so the cull shader appends from slot 0 each frame.
        ctx.encoder.clear_buffer(&self.draw_count_buf, 0, None);
        if !self.use_count_indirect {
            // No count-indirect support: must zero the indirect buffer so stale
            // entries beyond the visible range have instance_count = 0.
            ctx.encoder.clear_buffer(&self.indirect_buf, 0, None);
        }

        // ── Cull compute pass ─────────────────────────────────────────────────
        {
            let mut cpass = ctx
                .encoder
                .begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("VG Cull"),
                    timestamp_writes: None,
                });
            cpass.set_pipeline(&self.cull_pipeline);
            cpass.set_bind_group(0, cull_bg, &[]);
            cpass.dispatch_workgroups((meshlet_count + 63) / 64, 1, 1);
        }

        // ── VG GBuffer draw pass (LoadOp::Load — appends to existing GBuffer) ─
        {
            let mut rpass = ctx.encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("VG GBuffer"),
                color_attachments: &[
                    Some(wgpu::RenderPassColorAttachment {
                        view: gbuffer.albedo,
                        resolve_target: None,
                        depth_slice: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Load,
                            store: wgpu::StoreOp::Store,
                        },
                    }),
                    Some(wgpu::RenderPassColorAttachment {
                        view: gbuffer.normal,
                        resolve_target: None,
                        depth_slice: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Load,
                            store: wgpu::StoreOp::Store,
                        },
                    }),
                    Some(wgpu::RenderPassColorAttachment {
                        view: gbuffer.orm,
                        resolve_target: None,
                        depth_slice: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Load,
                            store: wgpu::StoreOp::Store,
                        },
                    }),
                    Some(wgpu::RenderPassColorAttachment {
                        view: gbuffer.emissive,
                        resolve_target: None,
                        depth_slice: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Load,
                            store: wgpu::StoreOp::Store,
                        },
                    }),
                ],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: ctx.depth,
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

            let active_pipeline = if self.debug_mode == 20 {
                self.debug_draw_pipeline
                    .as_ref()
                    .unwrap_or(&self.draw_pipeline)
            } else {
                &self.draw_pipeline
            };
            rpass.set_pipeline(active_pipeline);
            rpass.set_bind_group(0, draw_bg0, &[]);
            rpass.set_bind_group(1, draw_bg1, &[]);
            rpass.set_vertex_buffer(0, main_scene.mesh_buffers.vertices.slice(..));
            rpass.set_index_buffer(
                main_scene.mesh_buffers.indices.slice(..),
                wgpu::IndexFormat::Uint32,
            );
            if self.use_count_indirect {
                // AAA compaction path: GPU-written count drives how many compact
                // draw commands the hardware reads — zero wasted command reads.
                rpass.multi_draw_indexed_indirect_count(
                    &self.indirect_buf,
                    0,
                    &self.draw_count_buf,
                    0,
                    meshlet_count,
                );
            } else {
                // Fallback: compacted writes fill indirect[0..N_visible]; the rest
                // were cleared above so instance_count = 0 for the tail.
                #[cfg(not(target_arch = "wasm32"))]
                rpass.multi_draw_indexed_indirect(&self.indirect_buf, 0, meshlet_count);
                #[cfg(target_arch = "wasm32")]
                for i in 0..meshlet_count {
                    rpass.draw_indexed_indirect(&self.indirect_buf, i as u64 * 20);
                }
            }
        }

        Ok(())
    }
    fn as_any(&self) -> &dyn std::any::Any { self }
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any { self }
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

fn create_material_bgl(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    let count = NonZeroU32::new(MAX_TEXTURES as u32).expect("non-zero");
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("VG Material BGL"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
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
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                #[cfg(not(target_arch = "wasm32"))]
                count: Some(count),
                #[cfg(target_arch = "wasm32")]
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 3,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                #[cfg(not(target_arch = "wasm32"))]
                count: Some(count),
                #[cfg(target_arch = "wasm32")]
                count: None,
            },
        ],
    })
}

