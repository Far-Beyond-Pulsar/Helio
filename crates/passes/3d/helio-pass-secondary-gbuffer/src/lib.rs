//! Secondary G-buffer fill — off-screen 8-target G-buffer + depth for a camera slot.
//!
//! One pass, per-slot activation like `ShadowPass`'s shadow faces (self-managed
//! render passes opened directly against pool textures, `render_pass_descriptor`
//! returns `None`). Pooled textures: `MAX_SECONDARY_VIEWS` array layers × (8
//! colour + depth), matching the main G-buffer's formats byte-for-byte so
//! `ProxyCompositePass` can merge without conversion.
//!
//! Per active view: a compute cull (frustum + sublevel-membership test, mirrors
//! `helio-pass-indirect-dispatch`'s per-draw-group cooperative compaction) feeds
//! a G-buffer fill draw with full default-PBR bindless-texture material
//! sampling — group 1 (materials + material textures + `scene_textures`/
//! `scene_samplers`) mirrors `GBufferPass`'s BGL 1 exactly (`create_material_bgl`
//! below is the same ~30 lines `helio-pass-gbuffer` and `helio-pass-forward-lit`
//! each keep their own copy of — there is no shared helper for it anywhere in
//! the codebase yet, tracked by a long-standing `// TODO: Generalize material
//! BGL creation` in `helio-pass-gbuffer/src/lib.rs`). See
//! `shaders/secondary_gbuffer.wgsl` for exactly what's in and out of scope
//! (out: Radiant custom material graphs, baked lightmaps, alpha-blended /
//! forward-shaded materials — none of those reach the main `GBufferPass`
//! either, so this isn't a portal/sublevel-specific gap).

use bytemuck::{Pod, Zeroable};
use helio_core::graph::{ResourceBuilder, ResourceSize};
use helio_core::{PassContext, PrepareContext, RenderPass, Result as HelioResult};
use helio_secondary_core::{GpuSecondaryView, FIRST_SECONDARY_SLOT, MAX_PORTAL_VIEWS, MAX_SECONDARY_VIEWS};
use libhelio::MaterialBindingConfig;

/// The 8 secondary G-buffer colour targets, in the exact order the main
/// `GBufferPass` (and this pass's own fragment shader / `ProxyCompositePass`)
/// use: albedo, normal, orm, emissive, lightmap_uv, sss, extra, velocity.
pub const SECONDARY_GBUFFER_TARGETS: [(&str, wgpu::TextureFormat); 8] = [
    ("secondary_gbuffer_albedo", wgpu::TextureFormat::Rgba8Unorm),
    ("secondary_gbuffer_normal", wgpu::TextureFormat::Rgba16Float),
    ("secondary_gbuffer_orm", wgpu::TextureFormat::Rgba8Unorm),
    ("secondary_gbuffer_emissive", wgpu::TextureFormat::Rgba16Float),
    ("secondary_gbuffer_lightmap_uv", wgpu::TextureFormat::Rg16Float),
    ("secondary_gbuffer_sss", wgpu::TextureFormat::Rgba16Float),
    ("secondary_gbuffer_extra", wgpu::TextureFormat::Rgba16Float),
    ("secondary_gbuffer_velocity", wgpu::TextureFormat::Rg16Float),
];

/// Pooled depth array texture name.
pub const SECONDARY_GBUFFER_DEPTH: &str = "secondary_gbuffer_depth";

/// First camera slot reserved for sublevel cameras — mirrors
/// `helio::scene::secondary_views`'s private constant of the same value.
/// Used here only to classify a `GpuSecondaryView.camera_slot` as "portal" vs
/// "sublevel" for the cull's membership filter.
const FIRST_SUBLEVEL_SLOT: u32 = FIRST_SECONDARY_SLOT + MAX_PORTAL_VIEWS;

const CULL_SHADER: &str = include_str!("../shaders/secondary_cull.wgsl");
const GBUFFER_SHADER: &str = include_str!("../shaders/secondary_gbuffer.wgsl");

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ViewUniform {
    camera_slot: u32,
    /// `0` = portal (no filter), `1..=15` = required sublevel membership nibble.
    membership_filter: u32,
    draw_count: u32,
    _pad0: u32,
}

/// Per-secondary-view-slot GPU state. One entry per `0..MAX_SECONDARY_VIEWS`,
/// allocated once at construction (tiny uniform buffer) and grown lazily
/// (cull output buffers) as the scene's draw/instance count grows.
struct PerView {
    uniform_buf: wgpu::Buffer,
    indirect_buf: wgpu::Buffer,
    compacted_buf: wgpu::Buffer,
    draw_capacity: u32,
    instance_capacity: u32,
    cull_bind_group: Option<wgpu::BindGroup>,
    gbuffer_bind_group: Option<wgpu::BindGroup>,
    /// Set in `prepare()` when this slot is used this frame; read by `execute()`.
    active: bool,
    /// Draw-call-group count this frame (== `ctx.scene.draw_calls.len()`),
    /// cached so `execute()` doesn't need `PrepareContext`'s richer `scene`
    /// type again (`PassContext::scene` only exposes raw buffers + counts).
    draw_count: u32,
}

fn create_per_view(device: &wgpu::Device, index: usize) -> PerView {
    let draw_capacity = 64u32;
    let instance_capacity = 256u32;
    PerView {
        uniform_buf: device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Secondary View Uniform"),
            size: std::mem::size_of::<ViewUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }),
        indirect_buf: create_indirect_buffer(device, draw_capacity, index),
        compacted_buf: create_compacted_buffer(device, instance_capacity, index),
        draw_capacity,
        instance_capacity,
        cull_bind_group: None,
        gbuffer_bind_group: None,
        active: false,
        draw_count: 0,
    }
}

fn create_indirect_buffer(device: &wgpu::Device, capacity: u32, index: usize) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Secondary Cull Indirect"),
        size: (capacity as u64) * 20, // DrawIndexedIndirectArgs: 5 x u32/i32
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::INDIRECT
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
    .tap_label(index)
}

fn create_compacted_buffer(device: &wgpu::Device, capacity: u32, index: usize) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Secondary Compacted Indices"),
        size: (capacity as u64) * 4,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
    .tap_label(index)
}

/// No-op extension trait: `wgpu::Buffer` has no post-creation label setter, so
/// this exists purely to keep the `index` parameter from looking unused —
/// buffer labels are set at creation via `BufferDescriptor::label` and are
/// cosmetic (debugger/validation-layer names) only.
trait TapLabel {
    fn tap_label(self, _index: usize) -> Self;
}
impl TapLabel for wgpu::Buffer {
    fn tap_label(self, _index: usize) -> Self {
        self
    }
}

fn ensure_capacity(view: &mut PerView, device: &wgpu::Device, index: usize, draw_needed: u32, instance_needed: u32) {
    if view.draw_capacity < draw_needed {
        let cap = draw_needed.max(view.draw_capacity * 2).max(64);
        view.indirect_buf = create_indirect_buffer(device, cap, index);
        view.draw_capacity = cap;
        view.cull_bind_group = None;
        view.gbuffer_bind_group = None;
    }
    if view.instance_capacity < instance_needed {
        let cap = instance_needed.max(view.instance_capacity * 2).max(256);
        view.compacted_buf = create_compacted_buffer(device, cap, index);
        view.instance_capacity = cap;
        view.cull_bind_group = None;
        view.gbuffer_bind_group = None;
    }
}

/// Build per-layer `TextureView`s over a `MAX_SECONDARY_VIEWS`-layer pooled
/// array texture. Rebuilt every frame from whatever the pool currently holds
/// (cheap — a `TextureView` is a lightweight descriptor object) rather than
/// cached, so a graph resize (which reallocates the pooled texture) can never
/// leave this pass holding a view into a freed texture.
pub fn create_layer_views(texture: &wgpu::Texture, format: wgpu::TextureFormat, label: &str) -> Vec<wgpu::TextureView> {
    (0..MAX_SECONDARY_VIEWS)
        .map(|i| {
            texture.create_view(&wgpu::TextureViewDescriptor {
                label: Some(label),
                format: Some(format),
                dimension: Some(wgpu::TextureViewDimension::D2),
                base_array_layer: i,
                array_layer_count: Some(1),
                ..Default::default()
            })
        })
        .collect()
}

pub struct SecondaryGBufferPass {
    cull_pipeline: wgpu::ComputePipeline,
    cull_bgl: wgpu::BindGroupLayout,
    gbuffer_pipeline: wgpu::RenderPipeline,
    gbuffer_bgl: wgpu::BindGroupLayout,
    material_bgl: wgpu::BindGroupLayout,
    material_binding: MaterialBindingConfig,
    /// Cached so the material bind group is only rebuilt when the scene's
    /// texture table actually changes (`MaterialTextureBindings::version`),
    /// not every frame — unlike the tiny per-view bind groups, this one binds
    /// up to `max_textures` individual texture/sampler views on the
    /// `Expanded` tier and is worth not rebuilding gratuitously.
    material_bind_group: Option<wgpu::BindGroup>,
    material_bind_group_version: Option<u64>,
    views: Vec<PerView>,
    active_view_count: usize,
}

impl SecondaryGBufferPass {
    pub fn new(device: &wgpu::Device) -> Self {
        let material_binding = MaterialBindingConfig::for_device(device);

        let cull_module = helio_core::shader::module(device, "SecondaryCull", CULL_SHADER);
        let cull_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("SecondaryCull BGL"),
            entries: &[
                storage_entry(0, wgpu::ShaderStages::COMPUTE, true),
                storage_entry(1, wgpu::ShaderStages::COMPUTE, true),
                storage_entry(2, wgpu::ShaderStages::COMPUTE, true),
                uniform_entry(3, wgpu::ShaderStages::COMPUTE),
                storage_entry(4, wgpu::ShaderStages::COMPUTE, false),
                storage_entry(5, wgpu::ShaderStages::COMPUTE, false),
            ],
        });
        let cull_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("SecondaryCull Pipeline Layout"),
            bind_group_layouts: &[Some(&cull_bgl)],
            immediate_size: 0,
        });
        let cull_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("SecondaryCull Pipeline"),
            layout: Some(&cull_layout),
            module: &cull_module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        let gbuffer_source = build_gbuffer_shader_source(material_binding);
        let gbuffer_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("SecondaryGBuffer Shader"),
            source: wgpu::ShaderSource::Wgsl(gbuffer_source.into()),
        });
        let gbuffer_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("SecondaryGBuffer BGL 0"),
            entries: &[
                storage_entry(0, wgpu::ShaderStages::VERTEX, true),
                storage_entry(1, wgpu::ShaderStages::VERTEX, true),
                storage_entry(2, wgpu::ShaderStages::VERTEX, true),
                uniform_entry(3, wgpu::ShaderStages::VERTEX),
            ],
        });
        let material_bgl = create_material_bgl(device, material_binding);
        let gbuffer_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("SecondaryGBuffer Pipeline Layout"),
            bind_group_layouts: &[Some(&gbuffer_bgl), Some(&material_bgl)],
            immediate_size: 0,
        });
        let color_targets: Vec<Option<wgpu::ColorTargetState>> = SECONDARY_GBUFFER_TARGETS
            .iter()
            .map(|&(_, format)| {
                Some(wgpu::ColorTargetState {
                    format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })
            })
            .collect();
        let gbuffer_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("SecondaryGBuffer Pipeline"),
            layout: Some(&gbuffer_layout),
            vertex: wgpu::VertexState {
                module: &gbuffer_module,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                // Matches the shared mesh vertex layout (stride 40) — see
                // `helio-pass-gbuffer`'s pipeline for the authoritative comment.
                buffers: &[Some(wgpu::VertexBufferLayout {
                    array_stride: 40,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &[
                        wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x3, offset: 0, shader_location: 0 },
                        wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32, offset: 12, shader_location: 1 },
                        wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x2, offset: 16, shader_location: 2 },
                        wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x2, offset: 24, shader_location: 5 },
                        wgpu::VertexAttribute { format: wgpu::VertexFormat::Uint32, offset: 32, shader_location: 3 },
                        wgpu::VertexAttribute { format: wgpu::VertexFormat::Uint32, offset: 36, shader_location: 4 },
                    ],
                })],
            },
            fragment: Some(wgpu::FragmentState {
                module: &gbuffer_module,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &color_targets,
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

        let views = (0..MAX_SECONDARY_VIEWS as usize)
            .map(|i| create_per_view(device, i))
            .collect();

        Self {
            cull_pipeline,
            cull_bgl,
            gbuffer_pipeline,
            gbuffer_bgl,
            material_bgl,
            material_binding,
            material_bind_group: None,
            material_bind_group_version: None,
            views,
            active_view_count: 0,
        }
    }
}

/// Build the material bind group layout for group 1: materials storage,
/// material-texture-metadata storage, then `material_binding`'s bindless
/// texture/sampler entries starting at binding 2.
///
/// Byte-for-byte the same shape as `helio_pass_gbuffer::create_material_bgl`
/// (and `helio-pass-forward-lit`'s private copy of the same function) — kept
/// as its own copy rather than a cross-crate call, matching the existing
/// convention every pass wanting bindless materials follows today (see this
/// module's doc comment).
fn create_material_bgl(device: &wgpu::Device, material_binding: MaterialBindingConfig) -> wgpu::BindGroupLayout {
    let mut entries = vec![
        storage_entry(0, wgpu::ShaderStages::FRAGMENT, true),
        storage_entry(1, wgpu::ShaderStages::FRAGMENT, true),
    ];
    material_binding.append_layout_entries(&mut entries, 2, wgpu::ShaderStages::FRAGMENT);
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("SecondaryGBuffer Material BGL"),
        entries: &entries,
    })
}

/// Apply the same two device-tier-specific WGSL transforms
/// `RadiantTemplate::build_shader_source`/`apply_webgpu_fixups` apply to
/// `gbuffer.wgsl` (see `helio::radiant::template`), directly on
/// `secondary_gbuffer.wgsl`'s raw text: rewrite the binding-array capacity
/// literal to the device's actual `max_textures`, and — on the `Expanded`
/// (WebGPU/16-slot-fallback) tier — replace the native `binding_array`
/// bindings and `sample_texture`'s indexed lookup with individually-numbered
/// bindings and a `switch`. `libhelio::shader::apply_webgpu_material_bindings`
/// matches against `sample_texture`'s body text verbatim, which is why that
/// function in `secondary_gbuffer.wgsl` is byte-identical to `gbuffer.wgsl`'s.
fn build_gbuffer_shader_source(material_binding: MaterialBindingConfig) -> String {
    let max_tex = material_binding.max_textures.to_string();
    let mut source = GBUFFER_SHADER
        .replace("binding_array<texture_2d<f32>, 256>", &format!("binding_array<texture_2d<f32>, {max_tex}>"))
        .replace("binding_array<sampler, 256>", &format!("binding_array<sampler, {max_tex}>"));

    #[cfg(target_arch = "wasm32")]
    {
        source = source.replace("enable wgpu_binding_array;\n", "");
        source = source.replace("enable wgpu_binding_array;\r\n", "");
    }

    if !material_binding.uses_binding_arrays() {
        source = libhelio::shader::apply_webgpu_material_bindings(&source, material_binding.max_textures);
    }
    source
}

fn storage_entry(binding: u32, visibility: wgpu::ShaderStages, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn uniform_entry(binding: u32, visibility: wgpu::ShaderStages) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

impl RenderPass for SecondaryGBufferPass {
    fn name(&self) -> &'static str {
        "SecondaryGBuffer"
    }

    fn reads(&self) -> &'static [&'static str] {
        &["main_scene"]
    }

    fn writes(&self) -> &'static [&'static str] {
        &["secondary_gbuffers"]
    }

    fn declare_resources(&self, builder: &mut ResourceBuilder) {
        let size = ResourceSize::ScaledInternal { divisor: 2 };
        for &(name, format) in SECONDARY_GBUFFER_TARGETS.iter() {
            builder.write_color_raw(name, format, size);
            builder.with_layers(MAX_SECONDARY_VIEWS);
            // `ProxyCompositePass` samples these (not just attaches to them).
            builder.with_extra_usage(wgpu::TextureUsages::TEXTURE_BINDING);
        }
        builder.write_depth(SECONDARY_GBUFFER_DEPTH, size);
        builder.with_layers(MAX_SECONDARY_VIEWS);
        builder.with_extra_usage(wgpu::TextureUsages::TEXTURE_BINDING);
    }

    fn render_pass_descriptor<'a>(
        &'a self,
        _target: &'a wgpu::TextureView,
        _depth: &'a wgpu::TextureView,
        _resources: &'a libhelio::FrameResources<'a>,
    ) -> Option<wgpu::RenderPassDescriptor<'a>> {
        // Self-managed: this pass opens one render pass per active view
        // directly on `ctx.encoder_ptr` (ShadowPass's pattern), since the
        // executor's chain-fusion mechanism assumes one fixed set of
        // attachments per pass, not a variable per-frame count.
        None
    }

    fn prepare(&mut self, ctx: &PrepareContext) -> HelioResult<()> {
        let Some(secondary) = ctx.frame_resources.secondary.get() else {
            println!("SecondaryGBuffer: no secondary frame data");
            self.active_view_count = 0;
            for view in &mut self.views {
                view.active = false;
            }
            return Ok(());
        };

        let views: &[GpuSecondaryView] = bytemuck::cast_slice(secondary.view_bytes);
        let view_count = (secondary.view_count as usize).min(views.len()).min(self.views.len());
        self.active_view_count = view_count;
        println!("SecondaryGBuffer: {} active views", view_count);

        let draw_needed = ctx.scene.draw_calls.len() as u32;
        let instance_needed = ctx.scene.instances.len() as u32;
        let cameras_buf = ctx.scene.camera.buffer();
        let instances_buf = ctx.scene.instances.buffer();
        let draw_calls_buf = ctx.scene.draw_calls.buffer();
        let materials_buf = ctx.scene.materials.buffer();

        for i in 0..self.views.len() {
            if i >= view_count {
                self.views[i].active = false;
                continue;
            }
            let gpu_view = views[i];
            let membership_filter = if gpu_view.camera_slot >= FIRST_SUBLEVEL_SLOT {
                gpu_view.camera_slot - FIRST_SUBLEVEL_SLOT + 1
            } else {
                0
            };

            ensure_capacity(&mut self.views[i], ctx.device, i, draw_needed, instance_needed);

            let uniform = ViewUniform {
                camera_slot: gpu_view.camera_slot,
                membership_filter,
                draw_count: draw_needed,
                _pad0: 0,
            };
            ctx.queue.write_buffer(&self.views[i].uniform_buf, 0, bytemuck::bytes_of(&uniform));

            // Rebuilt every active frame rather than staleness-cached: bind
            // group creation here is a handful of storage-buffer descriptors
            // for at most `MAX_SECONDARY_VIEWS` (5) slots — correctness over
            // micro-optimisation, see this crate's module doc.
            let view = &mut self.views[i];
            view.cull_bind_group = Some(ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("SecondaryCull BG"),
                layout: &self.cull_bgl,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: cameras_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: instances_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: draw_calls_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 3, resource: view.uniform_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 4, resource: view.indirect_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 5, resource: view.compacted_buf.as_entire_binding() },
                ],
            }));
            view.gbuffer_bind_group = Some(ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("SecondaryGBuffer BG 0"),
                layout: &self.gbuffer_bgl,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: cameras_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: instances_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: view.compacted_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 3, resource: view.uniform_buf.as_entire_binding() },
                ],
            }));
            view.active = true;
            view.draw_count = draw_needed;
        }

        // Material bind group (group 1): rebuilt only when the scene's
        // texture table version changes, not per view / per frame — see the
        // field doc comment on `material_bind_group`.
        if let Some(main_scene) = ctx.frame_resources.main_scene.get() {
            let mt = main_scene.material_textures;
            if self.material_bind_group.is_none() || self.material_bind_group_version != Some(mt.version) {
                let mut entries = vec![
                    wgpu::BindGroupEntry { binding: 0, resource: materials_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: mt.material_textures.as_entire_binding() },
                ];
                self.material_binding.append_bind_group_entries(&mut entries, 2, mt.texture_views, mt.samplers);
                self.material_bind_group = Some(ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("SecondaryGBuffer Material BG"),
                    layout: &self.material_bgl,
                    entries: &entries,
                }));
                self.material_bind_group_version = Some(mt.version);
            }
        }

        Ok(())
    }

    fn execute(&mut self, ctx: &mut PassContext) -> HelioResult<()> {
        if self.active_view_count == 0 {
            return Ok(());
        }

        // ── Cull: one compute pass, one dispatch per active view ───────────
        {
            let mut pass = unsafe { &mut *ctx.encoder_ptr }.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("SecondaryCull"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.cull_pipeline);
            for view in self.views.iter().filter(|v| v.active) {
                let Some(bg) = view.cull_bind_group.as_ref() else { continue };
                pass.set_bind_group(0, bg, &[]);
                pass.dispatch_workgroups(view.draw_count.max(1), 1, 1);
            }
        }

        // ── Fill: one self-managed render pass per active view ─────────────
        let Some(pool_textures) = self.gather_pool_textures(ctx) else {
            // A resize/rebuild left a pooled resource unallocated this frame
            // (e.g. the graph was just (re)built and hasn't run `declare_resources`
            // through yet) — skip cleanly rather than panicking.
            return Ok(());
        };
        let Some(main_scene) = ctx.resources.main_scene.get() else {
            return Ok(());
        };
        let Some(material_bg) = self.material_bind_group.as_ref() else {
            // First frame after construction, before `prepare()` has had a
            // chance to see a scene with any materials registered — nothing
            // to bind yet, so nothing to draw.
            return Ok(());
        };

        for i in 0..self.views.len() {
            if !self.views[i].active {
                continue;
            }
            let Some(bg) = self.views[i].gbuffer_bind_group.as_ref() else { continue };

            let color_attachments: Vec<Option<wgpu::RenderPassColorAttachment>> = pool_textures
                .color
                .iter()
                .map(|views| {
                    Some(wgpu::RenderPassColorAttachment {
                        view: &views[i],
                        resolve_target: None,
                        depth_slice: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                            store: wgpu::StoreOp::Store,
                        },
                    })
                })
                .collect();

            let mut pass = unsafe { &mut *ctx.encoder_ptr }.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("SecondaryGBufferFill"),
                color_attachments: &color_attachments,
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &pool_textures.depth[i],
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });

            pass.set_pipeline(&self.gbuffer_pipeline);
            pass.set_bind_group(0, bg, &[]);
            pass.set_bind_group(1, material_bg, &[]);
            pass.set_vertex_buffer(0, main_scene.mesh_buffers.vertices.slice(..));
            pass.set_index_buffer(main_scene.mesh_buffers.indices.slice(..), wgpu::IndexFormat::Uint32);

            let draw_count = self.views[i].draw_count;
            #[cfg(not(target_arch = "wasm32"))]
            pass.multi_draw_indexed_indirect(&self.views[i].indirect_buf, 0, draw_count);
            #[cfg(target_arch = "wasm32")]
            for d in 0..draw_count {
                pass.draw_indexed_indirect(&self.views[i].indirect_buf, d as u64 * 20);
            }
        }

        Ok(())
    }
}

struct PoolTextures<'a> {
    color: [Vec<wgpu::TextureView>; 8],
    depth: Vec<wgpu::TextureView>,
    _marker: std::marker::PhantomData<&'a ()>,
}

impl SecondaryGBufferPass {
    fn gather_pool_textures(&self, ctx: &PassContext) -> Option<PoolTextures<'static>> {
        let mut color: [Vec<wgpu::TextureView>; 8] = Default::default();
        for (i, &(name, format)) in SECONDARY_GBUFFER_TARGETS.iter().enumerate() {
            let tex = ctx.resource_pool.get_texture(name)?;
            color[i] = create_layer_views(tex, format, name);
        }
        let depth_tex = ctx.resource_pool.get_texture(SECONDARY_GBUFFER_DEPTH)?;
        let depth = create_layer_views(depth_tex, wgpu::TextureFormat::Depth32Float, SECONDARY_GBUFFER_DEPTH);
        Some(PoolTextures { color, depth, _marker: std::marker::PhantomData })
    }
}

#[cfg(test)]
mod material_shader_tests {
    use super::*;
    use libhelio::MaterialBindingMode;
    use naga::valid::{Capabilities, ValidationFlags, Validator};

    fn validate(source: &str, capabilities: Capabilities) {
        let module = match naga::front::wgsl::parse_str(source) {
            Ok(m) => m,
            Err(e) => panic!("parse failed: {}", e.emit_to_string(source)),
        };
        if let Err(e) = Validator::new(ValidationFlags::all(), capabilities).validate(&module) {
            panic!("validation failed: {e:?}\n\nsource:\n{source}");
        }
    }

    /// The native `BindingArray` tier: `binding_array<...>` at the device's
    /// real `max_textures`, `enable wgpu_binding_array;` kept.
    #[test]
    fn binding_array_tier_shader_is_valid() {
        let cfg = MaterialBindingConfig { mode: MaterialBindingMode::BindingArray, max_textures: 128 };
        let source = build_gbuffer_shader_source(cfg);
        assert!(source.contains("binding_array<texture_2d<f32>, 128>"));
        assert!(source.contains("binding_array<sampler, 128>"));
        validate(&source, Capabilities::all());
    }

    /// The `Expanded` (WebGPU / native-16-slot-fallback) tier: individually
    /// numbered `scene_texture_N`/`scene_sampler_N` bindings and a `switch`
    /// in `sample_texture`, no native `binding_array` extension.
    #[test]
    fn expanded_tier_shader_is_valid_and_has_no_binding_array() {
        let cfg = MaterialBindingConfig { mode: MaterialBindingMode::Expanded, max_textures: 16 };
        let source = build_gbuffer_shader_source(cfg);
        assert!(!source.contains("binding_array"), "Expanded tier must not reference binding_array:\n{source}");
        assert!(!source.contains("enable wgpu_binding_array;"));
        assert!(source.contains("scene_texture_0:"));
        assert!(source.contains("scene_texture_15:"));
        assert!(source.contains("switch slot.texture_index"));
        // Expanded-tier WGSL (individual texture/sampler bindings, no
        // binding_array) needs no special naga capability.
        validate(&source, Capabilities::empty());
    }

    #[test]
    fn create_material_bgl_entry_count_matches_the_tier() {
        let device_free_check = |cfg: MaterialBindingConfig, expected_extra: usize| {
            let mut entries = vec![
                storage_entry(0, wgpu::ShaderStages::FRAGMENT, true),
                storage_entry(1, wgpu::ShaderStages::FRAGMENT, true),
            ];
            cfg.append_layout_entries(&mut entries, 2, wgpu::ShaderStages::FRAGMENT);
            assert_eq!(entries.len(), 2 + expected_extra);
        };
        device_free_check(MaterialBindingConfig { mode: MaterialBindingMode::BindingArray, max_textures: 64 }, 2);
        device_free_check(MaterialBindingConfig { mode: MaterialBindingMode::Expanded, max_textures: 16 }, 32);
    }
}
