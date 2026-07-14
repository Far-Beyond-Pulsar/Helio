//! G-Buffer pass.
//!
//! Renders opaque geometry to 4 render targets (albedo, normal, ORM, emissive) + depth.
//! O(1) CPU: single `multi_draw_indexed_indirect` call regardless of scene size.
//!
//! # Render Targets (owned by this pass)
//!
//! | Slot | Name     | Format        | Contents                          |
//! |------|----------|---------------|-----------------------------------|
//! | 0    | albedo   | Rgba8Unorm    | albedo.rgb + alpha                |
//! | 1    | normal   | Rgba16Float   | world normal.xyz + F0.r           |
//! | 2    | orm      | Rgba8Unorm    | AO, roughness, metallic, F0.g     |
//! | 3    | emissive | Rgba16Float   | emissive.rgb + F0.b               |
//!
//! # Material Bind Group
//!
//! Group 1 provides bindless texture access:
//!  - binding 0: materials storage buffer
//!  - binding 1: material_textures storage buffer (MaterialTextureData array)
//!  - binding 2: scene_textures (256-slot texture array)
//!  - binding 3: scene_samplers (256-slot sampler array)
//!
//! # Vertex / Index Buffers
//!
//! This pass owns no mesh data.  The caller must bind the shared mesh vertex
//! buffer (slot 0) and index buffer before this pass executes.

use bytemuck::{Pod, Zeroable};
use helio::radiant::{RadiantShaderCache, RadiantShaderKey, RadiantTemplateRegistry};
use helio_core::graph::{ResourceBuilder, ResourceSize};
use helio_core::{DebugViewDescriptor, PassContext, PrepareContext, RenderPass, Result as HelioResult};
use std::collections::HashMap;
use std::num::NonZeroU32;

/// Bindless texture array size per shader stage.
/// Capped at 16 on wasm32 and Apple native Metal; 256 on other desktop backends.
#[cfg(not(any(target_arch = "wasm32", target_os = "macos", target_os = "ios")))]
const MAX_TEXTURES: usize = 256;
#[cfg(any(target_arch = "wasm32", target_os = "macos", target_os = "ios"))]
const MAX_TEXTURES: usize = 16;

// ── Uniform types ─────────────────────────────────────────────────────────────

/// Per-frame globals uploaded to the GPU each frame (matches `Globals` in gbuffer.wgsl).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct GBufferGlobals {
    pub frame: u32,
    pub delta_time: f32,
    pub light_count: u32,
    pub ambient_intensity: f32,
    pub ambient_color: [f32; 4],
    pub rc_world_min: [f32; 4],
    pub rc_world_max: [f32; 4],
    pub csm_splits: [f32; 4],
    pub debug_mode: u32,
    pub _pad0: u32,
    pub _pad1: u32,
    pub _pad2: u32,
}

// ── Pass struct ───────────────────────────────────────────────────────────────

pub struct GBufferPass {
    pipelines: HashMap<RadiantShaderKey, wgpu::RenderPipeline>,
    shader_cache: RadiantShaderCache,
    template_registry: RadiantTemplateRegistry,
    pipeline_layout: wgpu::PipelineLayout,
    bind_group_layout_0: wgpu::BindGroupLayout,
    bind_group_layout_1: wgpu::BindGroupLayout,
    /// Group 0: camera + globals + instance_data. Rebuilt when buffer pointers change.
    bind_group_0: Option<wgpu::BindGroup>,
    bind_group_0_key: Option<(usize, usize)>,
    /// Group 1: materials + material_textures + bindless texture arrays.
    bind_group_1: Option<wgpu::BindGroup>,
    bind_group_1_version: Option<u64>,
    /// Per-frame globals uploaded in `prepare()`.
    globals_buf: wgpu::Buffer,
    /// CSM cascade split distances. Must match the values used in shadow_matrices.wgsl
    /// so that cascade selection in any shader that reads `globals.csm_splits` is
    /// consistent with the shadow maps that were actually generated.
    pub csm_splits: [f32; 4],
    /// Debug visualisation mode forwarded to the GBuffer shader (0 = off).
    pub debug_mode: u32,
    /// Lightmap atlas regions buffer (empty until bake data is loaded)
    lightmap_atlas_regions_buf: wgpu::Buffer,
}

impl GBufferPass {
    /// Create the GBuffer pass.
    pub fn new(device: &wgpu::Device) -> Self {
        // ── Globals buffer ────────────────────────────────────────────────────
        let globals_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("GBufferGlobals"),
            size: std::mem::size_of::<GBufferGlobals>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // ── Bind Group Layout 0 ───────────────────────────────────────────────
        let bind_group_layout_0 =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("GBuffer BGL 0"),
                entries: &[
                    // binding 0: camera (uniform, VERTEX | FRAGMENT)
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
                    // binding 1: globals (uniform, FRAGMENT)
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
                    // binding 2: instance_data (storage read, VERTEX)
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
                    // binding 3: lightmap_atlas_regions (storage read, VERTEX)
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
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

        // ── Bind Group Layout 1: material + textures ──────────────────────────
        let bind_group_layout_1 = create_gbuffer_material_bgl(device);

        // ── Pipeline layout (shared by all pipeline variants) ─────────────────
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("GBuffer PL"),
            bind_group_layouts: &[Some(&bind_group_layout_0), Some(&bind_group_layout_1)],
            immediate_size: 0,
        });

        // Create empty lightmap atlas regions buffer (populated when bake data is loaded).
        // Each region is 32 bytes: uv_offset(8) + uv_scale(8) + uv_clamp_min(8) + uv_clamp_max(8).
        let lightmap_atlas_regions_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Lightmap Atlas Regions"),
            size: 32,  // One empty region sentinel; recreated when real data loads.
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            pipelines: HashMap::new(),
            shader_cache: RadiantShaderCache::new(),
            template_registry: RadiantTemplateRegistry::new(),
            pipeline_layout,
            bind_group_layout_0,
            bind_group_layout_1,
            bind_group_0: None,
            bind_group_0_key: None,
            bind_group_1: None,
            bind_group_1_version: None,
            globals_buf,
            // Default CSM splits — single source of truth is libhelio::CSM_SPLITS.
            csm_splits: libhelio::CSM_SPLITS,
            debug_mode: 0,
            lightmap_atlas_regions_buf,
        }
    }
}

impl RenderPass for GBufferPass {
    fn name(&self) -> &'static str {
        "GBuffer"
    }

    fn declare_resources(&self, builder: &mut ResourceBuilder) {
        builder.write_color_raw("gbuffer_albedo", wgpu::TextureFormat::Rgba8Unorm, ResourceSize::MatchSurface);
        builder.write_color_raw("gbuffer_normal", wgpu::TextureFormat::Rgba16Float, ResourceSize::MatchSurface);
        builder.write_color_raw("gbuffer_orm", wgpu::TextureFormat::Rgba8Unorm, ResourceSize::MatchSurface);
        builder.write_color_raw("gbuffer_emissive", wgpu::TextureFormat::Rgba16Float, ResourceSize::MatchSurface);
        builder.write_color_raw("gbuffer_lightmap_uv", wgpu::TextureFormat::Rg16Float, ResourceSize::MatchSurface);
    }

    fn publish<'a>(&'a self, _frame: &mut libhelio::FrameResources<'a>) {}

    fn render_pass_descriptor<'a>(
        &'a self,
        _target: &'a wgpu::TextureView,
        depth: &'a wgpu::TextureView,
        resources: &'a libhelio::FrameResources<'a>,
    ) -> Option<wgpu::RenderPassDescriptor<'a>> {
        let gbuffer = resources.gbuffer.read("GBuffer")?;
        let lightmap_uv = resources.gbuffer_lightmap_uv.read("GBuffer")?;
        let color_attachments: &'a [Option<wgpu::RenderPassColorAttachment<'a>>] = Box::leak(Box::new([
            Some(wgpu::RenderPassColorAttachment {
                view: gbuffer.albedo,
                resolve_target: None,
                depth_slice: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                    store: wgpu::StoreOp::Store,
                },
            }),
            Some(wgpu::RenderPassColorAttachment {
                view: gbuffer.normal,
                resolve_target: None,
                depth_slice: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                    store: wgpu::StoreOp::Store,
                },
            }),
            Some(wgpu::RenderPassColorAttachment {
                view: gbuffer.orm,
                resolve_target: None,
                depth_slice: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                    store: wgpu::StoreOp::Store,
                },
            }),
            Some(wgpu::RenderPassColorAttachment {
                view: gbuffer.emissive,
                resolve_target: None,
                depth_slice: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                    store: wgpu::StoreOp::Store,
                },
            }),
            Some(wgpu::RenderPassColorAttachment {
                view: lightmap_uv,
                resolve_target: None,
                depth_slice: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                    store: wgpu::StoreOp::Store,
                },
            }),
        ]));
        Some(wgpu::RenderPassDescriptor {
            label: Some("GBuffer"),
            color_attachments,
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: depth,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(1.0),
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        })
    }

    fn prepare(&mut self, ctx: &PrepareContext) -> HelioResult<()> {
        // Read per-scene values from frame_resources so the GBuffer globals match
        // what the renderer configured (ambient light, GI bounds, etc.).
        let (ambient_color, ambient_intensity, rc_world_min, rc_world_max) =
            if let Some(ref ms) = ctx.frame_resources.main_scene.get().as_ref() {
                (
                    [ms.ambient_color[0], ms.ambient_color[1], ms.ambient_color[2], 1.0],
                    ms.ambient_intensity,
                    [ms.rc_world_min[0], ms.rc_world_min[1], ms.rc_world_min[2], 0.0],
                    [ms.rc_world_max[0], ms.rc_world_max[1], ms.rc_world_max[2], 0.0],
                )
            } else {
                // Fallback for headless / test usage without a full renderer.
                ([0.1, 0.1, 0.15, 1.0], 0.1, [-100.0_f32; 4], [100.0_f32; 4])
            };

        // Upload per-frame globals (O(1) — fixed-size struct).
        let globals = GBufferGlobals {
            frame: ctx.frame_num as u32,
            delta_time: ctx.delta_time,
            light_count: ctx.scene.lights.len() as u32,
            ambient_intensity,
            ambient_color,
            rc_world_min,
            rc_world_max,
            csm_splits: self.csm_splits,
            debug_mode: self.debug_mode,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        };
        ctx.write_buffer(&self.globals_buf, 0, bytemuck::bytes_of(&globals));
        Ok(())
    }

    fn execute(&mut self, ctx: &mut PassContext) -> HelioResult<()> {
        let draw_count = ctx.scene.draw_count;
        let main_scene = ctx.resources.main_scene;

        if draw_count == 0 || main_scene.is_none() {
            return Ok(());
        }
        let main_scene = main_scene.read("GBuffer").unwrap();

        // Rebuild bind group 0 when camera or instances buffer pointers change (GrowableBuffer realloc).
        let camera_ptr = ctx.scene.camera as *const _ as usize;
        let instances_ptr = ctx.scene.instances as *const _ as usize;
        let key = (camera_ptr, instances_ptr);
        if self.bind_group_0_key != Some(key) {
            log::debug!("GBuffer: rebuilding bind group 0 (buffer pointers changed)");
            self.bind_group_0 = Some(ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("GBuffer BG 0"),
                layout: &self.bind_group_layout_0,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: ctx.scene.camera.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: self.globals_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: ctx.scene.instances.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: self.lightmap_atlas_regions_buf.as_entire_binding(),
                    },
                ],
            }));
            self.bind_group_0_key = Some(key);
        }

        // Rebuild bind group 1 when material textures version changes.
        let needs_rebuild = self.bind_group_1_version != Some(main_scene.material_textures.version)
            || self.bind_group_1.is_none();
        if needs_rebuild {
            log::debug!("GBuffer: rebuilding bind group 1 (material textures version changed)");
            self.bind_group_1 = Some(
                ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("GBuffer BG 1"),
                    layout: &self.bind_group_layout_1,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: ctx.scene.materials.as_entire_binding(),
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
            self.bind_group_1_version = Some(main_scene.material_textures.version);
        }

        let indirect = ctx.scene.indirect;

        let pass = unsafe { &mut *ctx.active_render_pass_ptr().unwrap() };
        pass.set_bind_group(0, self.bind_group_0.as_ref().unwrap(), &[]);
        pass.set_bind_group(1, self.bind_group_1.as_ref().unwrap(), &[]);
        pass.set_vertex_buffer(0, main_scene.mesh_buffers.vertices.slice(..));
        pass.set_index_buffer(
            main_scene.mesh_buffers.indices.slice(..),
            wgpu::IndexFormat::Uint32,
        );

        let ranges = ctx.scene.material_class_ranges;
        if ranges.is_empty() {
            // Fallback: no ranges (e.g. legacy mode without material_class data).
            // Use the default PBR pipeline and draw everything in one batch.
            let key = RadiantShaderKey {
                template_id: 0,
                graph_hash: 0,
                feature_flags: 0,
            };
            let pipeline = self.get_or_create_pipeline(&ctx.device, key, "");
            pass.set_pipeline(pipeline);
            #[cfg(not(target_arch = "wasm32"))]
            pass.multi_draw_indexed_indirect(indirect, 0, draw_count);
            #[cfg(target_arch = "wasm32")]
            for i in 0..draw_count {
                pass.draw_indexed_indirect(indirect, i as u64 * 20);
            }
        } else {
            for &(class, graph_hash, start, count) in ranges {
                if count == 0 {
                    continue;
                }
                let key = RadiantShaderKey {
                    template_id: class,
                    graph_hash,
                    feature_flags: 0,
                };
                let graph_wgsl = ctx
                    .scene
                    .graph_wgsl_snippets
                    .get(&graph_hash)
                    .map(|s| s.as_str())
                    .unwrap_or("");
                let pipeline = self.get_or_create_pipeline(&ctx.device, key, graph_wgsl);
                pass.set_pipeline(pipeline);
                // DrawIndexedIndirectArgs = 5 × u32 = 20 bytes per entry
                #[cfg(not(target_arch = "wasm32"))]
                pass.multi_draw_indexed_indirect(indirect, start as u64 * 20, count);
                #[cfg(target_arch = "wasm32")]
                for i in start..start + count {
                    pass.draw_indexed_indirect(indirect, i as u64 * 20);
                }
            }
        }
        Ok(())
    }

    fn set_debug_mode(&mut self, mode: u32) {
        self.debug_mode = mode;
    }

    fn debug_views(&self) -> &'static [DebugViewDescriptor] {
        static VIEWS: &[DebugViewDescriptor] = &[
            DebugViewDescriptor {
                name: "UV Visualisation",
                debug_mode: 1,
                description: "Show UV coordinates as R=U, G=V",
            },
            DebugViewDescriptor {
                name: "Raw Texture",
                debug_mode: 2,
                description: "Raw texture sample without material multiply",
            },
            DebugViewDescriptor {
                name: "Geometry Normals",
                debug_mode: 3,
                description: "Geometry normals only (skip normal mapping)",
            },
        ];
        VIEWS
    }

    fn reads(&self) -> &'static [&'static str] {
        &["main_scene"]
    }

    fn writes(&self) -> &'static [&'static str] {
        &["gbuffer", "gbuffer_lightmap_uv"]
    }
}

impl GBufferPass {
    /// Populate the lightmap atlas regions buffer from baked data.
    ///
    /// Called by the renderer after a successful bake to upload per-mesh UV atlas regions.
    /// Each region maps a mesh_id to its atlas location with precomputed clamp bounds that
    /// prevent bilinear filtering from bleeding across adjacent atlas region boundaries.
    ///
    /// # Arguments
    /// * `queue`   - GPU queue for buffer uploads
    /// * `regions` - `[uv_offset.x, uv_offset.y, uv_scale.x, uv_scale.y,
    ///                uv_clamp_min.x, uv_clamp_min.y, uv_clamp_max.x, uv_clamp_max.y]`
    ///              per mesh (32 bytes each, matching `LightmapAtlasRegion` in gbuffer.wgsl)
    pub fn upload_lightmap_atlas_regions(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        regions: &[[f32; 8]],
    ) {
        if regions.is_empty() {
            return;
        }

        // Each region is 32 bytes: uv_offset(8) + uv_scale(8) + uv_clamp_min(8) + uv_clamp_max(8).
        let buf_size = (regions.len() * 32) as u64;

        // Recreate buffer if size changed
        if self.lightmap_atlas_regions_buf.size() < buf_size {
            self.lightmap_atlas_regions_buf = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Lightmap Atlas Regions"),
                size: buf_size.max(32),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            
            // Invalidate bind group since buffer changed
            self.bind_group_0_key = None;
        }

        queue.write_buffer(&self.lightmap_atlas_regions_buf, 0, bytemuck::cast_slice(regions));

        log::info!(
            "[GBufferPass] Uploaded {} lightmap atlas regions ({} bytes)",
            regions.len(),
            buf_size
        );
    }

    /// Access the template registry for loading new surface archetypes at runtime.
    pub fn template_registry_mut(&mut self) -> &mut RadiantTemplateRegistry {
        &mut self.template_registry
    }

    /// Get or create a render pipeline for the given key.
    /// Compiles the shader lazily on first access, injecting `graph_wgsl` when
    /// `graph_hash != 0`.
    fn get_or_create_pipeline(
        &mut self,
        device: &wgpu::Device,
        key: RadiantShaderKey,
        graph_wgsl: &str,
    ) -> &wgpu::RenderPipeline {
        if !self.pipelines.contains_key(&key) {
            let template = match self.template_registry.get(key.template_id) {
                Some(t) => t,
                None => {
                    log::warn!(
                        "Radiant: template class {} not found, falling back to class 0",
                        key.template_id,
                    );
                    self.template_registry.get(0).expect("Default PBR template (class 0) missing")
                }
            };
            let module = self.shader_cache.get_or_compile(
                device, key, template, graph_wgsl, MAX_TEXTURES, "GBuffer Shader",
            );
            let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("GBuffer Pipeline"),
                layout: Some(&self.pipeline_layout),
                vertex: wgpu::VertexState {
                    module,
                    entry_point: Some("vs_main"),
                    compilation_options: Default::default(),
                    // Full vertex layout (stride = 40 bytes, matching shared mesh buffer).
                    //   offset  0 — position       Float32x3  location 0
                    //   offset 12 — bitangent_sign Float32    location 1
                    //   offset 16 — tex_coords0   Float32x2  location 2  (UV0: material/albedo)
                    //   offset 24 — tex_coords1   Float32x2  location 5  (UV1: lightmap)
                    //   offset 32 — normal        Uint32     location 3
                    //   offset 36 — tangent       Uint32     location 4
                    buffers: &[wgpu::VertexBufferLayout {
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
                                format: wgpu::VertexFormat::Float32x2,
                                offset: 24,
                                shader_location: 5,
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
                    }],
                },
                fragment: Some(wgpu::FragmentState {
                    module,
                    entry_point: Some("fs_main"),
                    compilation_options: Default::default(),
                    targets: &[
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
                        Some(wgpu::ColorTargetState {
                            format: wgpu::TextureFormat::Rg16Float,
                            blend: None,
                            write_mask: wgpu::ColorWrites::ALL,
                        }),
                    ],
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    cull_mode: Some(wgpu::Face::Back),
                    ..Default::default()
                },
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: wgpu::TextureFormat::Depth32Float,
                    depth_write_enabled: Some(true),
                    depth_compare: Some(wgpu::CompareFunction::LessEqual),
                    stencil: wgpu::StencilState::default(),
                    bias: wgpu::DepthBiasState::default(),
                }),
                multisample: wgpu::MultisampleState::default(),
                multiview_mask: None,
                cache: None,
            });
            self.pipelines.insert(key, pipeline);
        }
        self.pipelines.get(&key).unwrap()
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Build the BGL for group 1 (bindless materials + textures).
fn create_gbuffer_material_bgl(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    let texture_array_count =
        NonZeroU32::new(MAX_TEXTURES as u32).expect("non-zero texture table size");

    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("GBuffer BGL 1"),
        entries: &[
            // binding 0: materials storage buffer
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
            // binding 1: material_textures storage buffer
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
            // binding 2: scene_textures (texture array; collapsed to 1 on wasm32)
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                #[cfg(not(target_arch = "wasm32"))]
                count: Some(texture_array_count),
                #[cfg(target_arch = "wasm32")]
                count: None,
            },
            // binding 3: scene_samplers (256-slot sampler array)
            wgpu::BindGroupLayoutEntry {
                binding: 3,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                #[cfg(not(target_arch = "wasm32"))]
                count: Some(texture_array_count),
                #[cfg(target_arch = "wasm32")]
                count: None,
            },
        ],
    })
}

