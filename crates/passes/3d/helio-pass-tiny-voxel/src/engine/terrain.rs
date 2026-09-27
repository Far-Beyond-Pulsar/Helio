use super::{
    residency::{self, pipeline::Pipeline},
    DepthConvention, GBufferTargets, GBUFFER_FORMATS,
};
use crate::{GpuEdit, Params, World, SHADER};
use std::sync::Arc;
use wgpu::util::DeviceExt;

/// Logical allocations owned by the terrain, excluding driver padding, shared
/// graph attachments, optional profiler buffers, allocator overhead and CPU
/// caches. This is not process VRAM.
#[derive(Clone, Copy, Debug, serde::Serialize)]
pub struct TerrainMemoryStats {
    pub buffers_bytes: u64,
    pub textures_bytes: u64,
    pub material_capacity_bytes: u64,
    pub primary_hits_bytes: u64,
}

/// The sole engine terrain backend: budgeted brick production and stored rays.
pub struct StoredTerrain {
    #[cfg(feature = "surface-cache-experiment")]
    pub(super) patch: super::surface_patch::Patch,
    #[cfg(feature = "canonical-far-experiment")]
    canonical: super::canonical::Source,
    pub(crate) device: wgpu::Device,
    queue: wgpu::Queue,
    pub size: [u32; 2],
    world: Arc<World>,
    generation_world: Option<Arc<World>>,
    residency: Pipeline,
    nodes: wgpu::Buffer,
    materials: wgpu::Buffer,
    exact_occupied: wgpu::Buffer,
    jobs: wgpu::Buffer,
    edits: wgpu::Buffer,
    edit_references: wgpu::Buffer,
    field_settings: wgpu::Buffer,
    heights: wgpu::Buffer,
    pub(super) uniform: wgpu::Buffer,
    pub(super) hits: wgpu::Buffer,
    generate: wgpu::ComputePipeline,
    bounds: wgpu::ComputePipeline,
    trace: wgpu::ComputePipeline,
    #[cfg(feature = "surface-reference")]
    prepare_rays: wgpu::ComputePipeline,
    #[cfg(feature = "regional-publication-experiment")]
    regional_trace: wgpu::ComputePipeline,
    #[cfg(feature = "regional-publication-experiment")]
    regional_visibility: wgpu::ComputePipeline,
    trace_shader: wgpu::ShaderModule,
    diagnostic_trace: [std::sync::OnceLock<wgpu::ComputePipeline>; 8],
    visibility: wgpu::ComputePipeline,
    surface: wgpu::RenderPipeline,
    sun: wgpu::Texture,
    pub(crate) sun_view: wgpu::TextureView,
    direction: wgpu::Texture,
    pub(crate) profiler: Option<helio_core::profiling::GpuProfiler>,
}
pub(super) fn buffer(device: &wgpu::Device, label: &str, size: u64) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    })
}
fn image(
    device: &wgpu::Device,
    label: &str,
    size: [u32; 2],
    format: wgpu::TextureFormat,
) -> wgpu::Texture {
    device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d {
            width: size[0],
            height: size[1],
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::STORAGE_BINDING
            | wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    })
}
impl StoredTerrain {
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        width: u32,
        height: u32,
        depth: DepthConvention,
    ) -> Self {
        // Checked-out WGSL may use CRLF on Windows; splitting the generation
        // and trace modules must not depend on the checkout's line endings.
        let mut source = format!("{SHADER}\n{}", include_str!("stored.wgsl")).replace("\r\n", "\n");
        let generation_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("voxel brick generation"),
            source: wgpu::ShaderSource::Wgsl(source.clone().into()),
        });
        let begin = source
            .find("@compute @workgroup_size(64)\nfn generate_bricks")
            .unwrap();
        let end = source.find("struct StoredCamera").unwrap();
        source.replace_range(begin..end, "");
        source = source.replace(
            "var<storage,read_write> stored_materials",
            "var<storage,read> stored_materials",
        );
        source = source.replace(
            "var<storage,read_write> exact_occupied",
            "var<storage,read> exact_occupied",
        );
        #[cfg(feature = "surface-reference")]
        { source = source.replace("override STORED_REFERENCE_RAYS:bool=false;", "override STORED_REFERENCE_RAYS:bool=true;"); }
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("stored voxel terrain"),
            source: wgpu::ShaderSource::Wgsl({
                #[cfg(feature = "canonical-far-experiment")]
                {
                    source = source.replace("override STORED_CANONICAL:bool=false;", "override STORED_CANONICAL:bool=true;");
                    let begin = source.find("fn stored_far_hit(").unwrap();
                    let end = source[begin..].find("// Camera-relative origin").unwrap() + begin;
                    source.replace_range(begin..end, &format!(
                        "{}\n{}",
                        include_str!("canonical_position.wgsl"),
                        include_str!("canonical_far.wgsl"),
                    ));
                }
                #[cfg(feature = "surface-cache-experiment")]
                {
                    // The compact cache pass completes known hits first. The
                    // canonical shader handles unresolved rays without cache
                    // lookups or extra traversal state in its inner loops.
                    const HOOK: &str = "let index=id.x+id.y*u32(p.screen.x);\n    var rd=";
                    assert_eq!(source.matches(HOOK).count(), 1, "cache primary hook drifted");
                    source = source.replace(HOOK, "let index=id.x+id.y*u32(p.screen.x);\n    if (primary_hits[index].status&0x08000000u)!=0u {return;}\n    var rd=");
                }
                source.into()
            }),
        });
        let compute = |entry: &str| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: None,
                module: if entry == "generate_bricks" || entry == "bound_bricks" {
                    &generation_shader
                } else {
                    &shader
                },
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        #[cfg(feature = "regional-publication-experiment")]
        let regional_compute = |entry: &str| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: None,
                module: &shader,
                entry_point: Some(entry),
                compilation_options: wgpu::PipelineCompilationOptions {
                    constants: &[("STORED_REGIONAL", 1.0)],
                    ..Default::default()
                },
                cache: None,
            })
        };
        let surface = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("stored voxel GBuffer"),
            layout: None,
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("stored_fullscreen"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("stored_surface"),
                compilation_options: wgpu::PipelineCompilationOptions {
                    constants: &[(
                        "STORED_REVERSE_DEPTH",
                        if matches!(depth, DepthConvention::Reverse) {
                            1.0
                        } else {
                            0.0
                        },
                    )],
                    ..Default::default()
                },
                targets: &GBUFFER_FORMATS.map(|format| {
                    Some(wgpu::ColorTargetState {
                        format,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    })
                }),
            }),
            primitive: Default::default(),
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: Some(true),
                depth_compare: Some(if matches!(depth, DepthConvention::Reverse) {
                    wgpu::CompareFunction::Greater
                } else {
                    wgpu::CompareFunction::Less
                }),
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: Default::default(),
            multiview_mask: None,
            cache: None,
        });
        let uniform = |label| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: std::mem::size_of::<Params>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        };
        let field = crate::landforms::default_field();
        let field_settings = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("voxel recipe settings"),
            contents: bytemuck::cast_slice(&field.gpu_settings()),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let heights = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("voxel recipe landforms"),
            contents: bytemuck::cast_slice(field.landforms().snapshot().height_atlas()),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let size = [width, height];
        let sun = image(
            device,
            "stored voxel sunlight",
            size,
            wgpu::TextureFormat::Rgba16Float,
        );
        let sun_view = sun.create_view(&Default::default());
        let capacity = (device.limits().max_storage_buffer_binding_size as usize
            / (residency::BRICK_WORDS * 4))
            .min(residency::BRICK_CAPACITY);
        Self {
            #[cfg(feature = "surface-cache-experiment")]
            patch: super::surface_patch::Patch::new(device),
            device: device.clone(),
            #[cfg(feature = "canonical-far-experiment")]
            canonical: super::canonical::Source::new(device),
            queue: queue.clone(),
            size,
            world: Arc::new(World::default()),
            generation_world: None,
            exact_occupied: buffer(device, "exact voxel brick occupancy", capacity as u64 * 4),
            residency: Pipeline::new(capacity),
            nodes: buffer(
                device,
                "resident voxel tree",
                (residency::PUBLICATION_NODE_CAPACITY as u64 + 4) * 4,
            ),
            materials: buffer(
                device,
                "resident two-bit voxel materials",
                (capacity * residency::BRICK_WORDS * 4) as u64,
            ),
            jobs: buffer(
                device,
                "bounded voxel generation batch",
                residency::GENERATION_BATCH as u64 * 32,
            ),
            edits: buffer(
                device,
                "voxel generation edits",
                crate::world::MAX_EDITS as u64 * 32,
            ),
            edit_references: buffer(device, "voxel brick edit references", 262_144 * 4),
            field_settings,
            heights,
            uniform: uniform("stored voxel frame"),
            hits: buffer(
                device,
                "stored voxel primary hits",
                width as u64 * height as u64 * 32,
            ),
            generate: compute("generate_bricks"),
            bounds: compute("bound_bricks"),
            trace: compute("stored_primary"),
            #[cfg(feature = "surface-reference")]
            prepare_rays: compute("stored_prepare_rays"),
            #[cfg(feature = "regional-publication-experiment")]
            regional_trace: regional_compute("stored_primary"),
            #[cfg(feature = "regional-publication-experiment")]
            regional_visibility: regional_compute("stored_visibility"),
            trace_shader: shader.clone(),
            diagnostic_trace: std::array::from_fn(|_| std::sync::OnceLock::new()),
            visibility: compute("stored_visibility"),
            surface,
            sun,
            sun_view,
            direction: image(
                device,
                "stored voxel sun direction",
                [1, 1],
                wgpu::TextureFormat::Rgba32Float,
            ),
            profiler: None,
        }
    }
    fn upload_links(&self, links: &[u32]) {
        self.queue
            .write_buffer(&self.nodes, 0, bytemuck::cast_slice(links));
    }
    #[cfg(test)]
    fn upload_nodes(&self, nodes: &[residency::Node]) {
        self.upload_links(&residency::pipeline::pack(nodes));
    }
    pub fn upload_edits(&mut self, world: &World) {
        self.world = Arc::new(world.clone());
    }
    pub fn set_world(&mut self, world: Arc<World>) {
        self.world = world;
    }
    pub fn set_stage_profiling(&mut self, enabled: bool) {
        if enabled {
            self.profiler.get_or_insert_with(|| {
                helio_core::profiling::GpuProfiler::new(&self.device, &self.queue)
            });
        } else {
            self.profiler = None;
        }
    }
    pub fn stats(&self) -> residency::Stats {
        self.residency.stats
    }
    pub fn memory_stats(&self) -> TerrainMemoryStats {
        let buffers_bytes = [
            &self.nodes,
            &self.materials,
            &self.exact_occupied,
            &self.jobs,
            &self.edits,
            &self.edit_references,
            &self.field_settings,
            &self.heights,
            &self.uniform,
            &self.hits,
        ]
        .iter()
        .map(|buffer| buffer.size())
        .sum::<u64>();
        #[cfg(feature = "canonical-far-experiment")]
        let buffers_bytes =
            buffers_bytes + self.canonical.edits.size() + self.canonical.settings.size();
        #[cfg(feature = "surface-cache-experiment")]
        let buffers_bytes = buffers_bytes
            + self.patch.settings.size()
            + self.patch.directory.size()
            + self.patch.words.size() + self.patch.mesh_memory().0;
        let textures_bytes = u64::from(self.size[0]) * u64::from(self.size[1]) * 8 + 16;
        #[cfg(feature = "surface-cache-experiment")]
        let textures_bytes = textures_bytes + self.patch.mesh_memory().1;
        TerrainMemoryStats {
            buffers_bytes,
            textures_bytes,
            material_capacity_bytes: self.materials.size(),
            primary_hits_bytes: self.hits.size(),
        }
    }
    pub fn primary_hit_buffer(&self) -> Option<&wgpu::Buffer> {
        Some(&self.hits)
    }
    /// Re-run the current primary rays with work counters into a separate
    /// buffer. Each 64-byte record contains the complete original hit followed
    /// by a copy whose first three words are leaf/exact/far iteration counts.
    /// Disable acceleration to compare against voxel-by-voxel
    /// walking through the identical cut and camera, without streaming drift.
    /// This expensive diagnostic is outside normal rendering and timings.
    pub fn encode_trace_work(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        accelerated: bool,
    ) -> wgpu::Buffer {
        self.encode_ray_work(encoder, accelerated, false)
    }
    /// Replay rays from terrain hits, irrespective of later mesh coverage.
    /// Each record is a blocker hit followed by counters and the ray origin.
    /// This does not measure shadow fidelity or exact GPU execution offsets.
    pub fn encode_sun_trace_work(&self, encoder: &mut wgpu::CommandEncoder) -> wgpu::Buffer {
        self.encode_ray_work(encoder, false, true)
    }
    fn encode_ray_work(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        accelerated: bool,
        sunlight: bool,
    ) -> wgpu::Buffer {
        let regional = self.residency.stats.fallback_regions > 0;
        let index = usize::from(accelerated) + 2 * usize::from(sunlight) + 4 * usize::from(regional);
        let pipeline = self.diagnostic_trace[index].get_or_init(|| {
            self.device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("voxel traversal work diagnostic"),
                    layout: None,
                    module: &self.trace_shader,
                    entry_point: Some("stored_primary_work"),
                    compilation_options: wgpu::PipelineCompilationOptions {
                        constants: &[
                            ("STORED_TRACE_WORK", 1.0),
                            ("STORED_WORK_SUN", f64::from(sunlight)),
                            ("STORED_REGIONAL", f64::from(regional)),
                            ("STORED_SKIP_EMPTY", if accelerated { 1.0 } else { 0.0 }),
                        ],
                        ..Default::default()
                    },
                    cache: None,
                })
        });
        let output = buffer(
            &self.device,
            "voxel work diagnostic output",
            self.hits.size() * 2,
        );
        let inputs = self.trace_group(
            &pipeline.get_bind_group_layout(0),
            &[
                (0, &self.uniform),
                (9, &output),
                (24, &self.nodes),
                (25, &self.materials),
                (28, &self.exact_occupied),
                (29, &self.hits),
            ],
        );
        self.compute(
            encoder,
            pipeline,
            &[inputs],
            [self.size[0].div_ceil(8), self.size[1].div_ceil(8), 1],
        );
        output
    }
    pub fn sun(&self) -> &wgpu::Texture {
        &self.sun
    }
    pub fn direction(&self) -> &wgpu::Texture {
        &self.direction
    }
    pub fn resize(&mut self, width: u32, height: u32) {
        if self.size == [width, height] {
            return;
        }
        self.size = [width, height];
        self.hits = buffer(
            &self.device,
            "stored voxel primary hits",
            width as u64 * height as u64 * 32,
        );
        self.sun = image(
            &self.device,
            "stored voxel sunlight",
            self.size,
            wgpu::TextureFormat::Rgba16Float,
        );
        self.sun_view = self.sun.create_view(&Default::default());
    }
    fn group(
        &self,
        layout: &wgpu::BindGroupLayout,
        bindings: &[(u32, &wgpu::Buffer)],
    ) -> wgpu::BindGroup {
        let entries: Vec<_> = bindings
            .iter()
            .map(|(binding, b)| wgpu::BindGroupEntry {
                binding: *binding,
                resource: b.as_entire_binding(),
            })
            .collect();
        self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("stored voxel inputs"),
            layout,
            entries: &entries,
        })
    }
    fn trace_group(
        &self,
        layout: &wgpu::BindGroupLayout,
        bindings: &[(u32, &wgpu::Buffer)],
    ) -> wgpu::BindGroup {
        #[cfg(feature = "canonical-far-experiment")]
        {
            let mut bindings = bindings.to_vec();
            bindings.extend([
                (1, &self.canonical.edits),
                (20, &self.field_settings),
                (21, &self.heights),
                (30, &self.canonical.settings),
            ]);
            self.group(layout, &bindings)
        }
        #[cfg(not(feature = "canonical-far-experiment"))]
        self.group(layout, bindings)
    }
    fn compute(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        pipeline: &wgpu::ComputePipeline,
        groups: &[wgpu::BindGroup],
        dispatch: [u32; 3],
    ) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("stored voxel compute"),
            timestamp_writes: None,
        });
        pass.set_pipeline(pipeline);
        for (i, g) in groups.iter().enumerate() {
            pass.set_bind_group(i as u32, g, &[]);
        }
        pass.dispatch_workgroups(dispatch[0], dispatch[1], dispatch[2]);
    }
    pub fn encode(
        &mut self,
        params: &Params,
        camera: &wgpu::Buffer,
        encoder: &mut wgpu::CommandEncoder,
        targets: GBufferTargets<'_>,
        sunlight: bool,
    ) {
        self.resize(params.screen[0] as u32, params.screen[1] as u32);
        if let Some(mut prepared) = self.residency.prepare(&self.world, params) {
            if let Some((jobs, references, world)) = prepared.batch.take() {
                if !jobs.is_empty() {
                    if self
                        .generation_world
                        .as_ref()
                        .is_none_or(|w| !Arc::ptr_eq(w, &world))
                    {
                        let common = self.generation_world.as_ref().map_or(0, |old| {
                            old.edits
                                .iter()
                                .zip(&world.edits)
                                .take_while(|(a, b)| a == b)
                                .count()
                        });
                        let edits: Vec<_> = world.edits[common..]
                            .iter()
                            .map(|e| GpuEdit {
                                cell: e.cell,
                                material: e.material,
                                radius: e.radius,
                                radius_units: e.radius_units(),
                                pad: [0.0; 2],
                            })
                            .collect();
                        if !edits.is_empty() {
                            self.queue.write_buffer(
                                &self.edits,
                                common as u64 * 32,
                                bytemuck::cast_slice(&edits),
                            );
                        }
                        self.generation_world = Some(world.clone());
                    }
                    self.queue
                        .write_buffer(&self.jobs, 0, bytemuck::cast_slice(&jobs));
                    if !references.is_empty() {
                        self.queue.write_buffer(
                            &self.edit_references,
                            0,
                            bytemuck::cast_slice(&references),
                        );
                    }
                    let group = self.group(
                        &self.generate.get_bind_group_layout(0),
                        &[
                            (1, &self.edits),
                            (20, &self.field_settings),
                            (21, &self.heights),
                            (25, &self.materials),
                            (26, &self.jobs),
                            (27, &self.edit_references),
                        ],
                    );
                    if let Some(p) = &mut self.profiler {
                        p.begin_pass(encoder, "voxel_generation");
                    }
                    self.compute(
                        encoder,
                        &self.generate,
                        &[group],
                        [32, jobs.len() as u32, 1],
                    );
                    let bounds_group = self.group(
                        &self.bounds.get_bind_group_layout(0),
                        &[
                            (25, &self.materials),
                            (26, &self.jobs),
                            (28, &self.exact_occupied),
                        ],
                    );
                    self.compute(
                        encoder,
                        &self.bounds,
                        &[bounds_group],
                        [8, jobs.len() as u32, 1],
                    );
                    if let Some(p) = &mut self.profiler {
                        p.end_pass(encoder, "voxel_generation");
                    }
                }
            }
            if let Some(links) = prepared.publication.take() {
                self.upload_links(&links);
            }
            self.residency.accept_after_encode(prepared);
        }
        let mut p = *params;
        #[cfg(feature = "canonical-far-experiment")]
        if let Some(world) = self.residency.active_world() {
            self.canonical.publish(&self.queue, world);
            #[cfg(feature = "surface-cache-experiment")]
            self.patch.update(&self.queue, world, params);
        }
        p.settings[3] = self.residency.active_voxel_step() as f32;
        p.settings[2] = if self.residency.stats.ready { 1.0 } else { 0.0 };
        self.queue
            .write_buffer(&self.uniform, 0, bytemuck::bytes_of(&p));
        // Completed cuts use the original specialization. Decode ancestor
        // coordinates only when this particular cut actually references them.
        let trace = &self.trace;
        let visibility = &self.visibility;
        #[cfg(feature = "regional-publication-experiment")]
        let (trace, visibility) = if self.residency.stats.fallback_regions > 0 {
            (&self.regional_trace, &self.regional_visibility)
        } else {
            (trace, visibility)
        };
        let group = self.trace_group(
            &trace.get_bind_group_layout(0),
            &[
                (0, &self.uniform),
                (9, &self.hits),
                (24, &self.nodes),
                (25, &self.materials),
                (28, &self.exact_occupied),
            ],
        );
        let cameras = self.group(&trace.get_bind_group_layout(1), &[(0, camera)]);
        if let Some(p) = &mut self.profiler {
            p.begin_pass(encoder, "voxel_primary");
        }
        #[cfg(feature = "surface-reference")]
        {
            // Force one materialized ray value for every traversal return path.
            // This is an offline reference cost, not a production optimization.
            let group = self.group(&self.prepare_rays.get_bind_group_layout(0),
                &[(0, &self.uniform), (9, &self.hits)]);
            let cameras = self.group(&self.prepare_rays.get_bind_group_layout(1), &[(0, camera)]);
            self.compute(encoder, &self.prepare_rays, &[group, cameras],
                [self.size[0].div_ceil(8), self.size[1].div_ceil(8), 1]);
        }
        #[cfg(feature = "surface-cache-experiment")]
        self.patch.encode_primary(&self.device, &self.uniform, camera, &self.hits, encoder, self.size, self.profiler.as_mut());
        self.compute(
            encoder,
            trace,
            &[group, cameras],
            [self.size[0].div_ceil(8), self.size[1].div_ceil(8), 1],
        );
        if let Some(p) = &mut self.profiler {
            p.end_pass(encoder, "voxel_primary");
            p.begin_pass(encoder, "voxel_gbuffer");
        }
        let group = self.group(
            &self.surface.get_bind_group_layout(0),
            &[
                (0, &self.uniform),
                (9, &self.hits),
                (20, &self.field_settings),
                (21, &self.heights),
            ],
        );
        let cameras = self.group(&self.surface.get_bind_group_layout(1), &[(0, camera)]);
        let attachments = targets.colors.map(|view| {
            Some(wgpu::RenderPassColorAttachment {
                view,
                resolve_target: None,
                depth_slice: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
            })
        });
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("stored voxel GBuffer"),
                color_attachments: &attachments,
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: targets.depth,
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
            pass.set_pipeline(&self.surface);
            pass.set_bind_group(0, &group, &[]);
            pass.set_bind_group(1, &cameras, &[]);
            pass.draw(0..3, 0..1);
        }
        if let Some(p) = &mut self.profiler {
            p.end_pass(encoder, "voxel_gbuffer");
        }
        if sunlight {
            let group = self.trace_group(
                &visibility.get_bind_group_layout(0),
                &[
                    (0, &self.uniform),
                    (9, &self.hits),
                    (24, &self.nodes),
                    (25, &self.materials),
                    (28, &self.exact_occupied),
                ],
            );
            let direction = self.direction.create_view(&Default::default());
            let outputs = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("stored sunlight outputs"),
                layout: &visibility.get_bind_group_layout(1),
                entries: &[(1, targets.colors[4]), (2, &self.sun_view), (3, &direction)].map(
                    |(binding, v)| wgpu::BindGroupEntry {
                        binding,
                        resource: wgpu::BindingResource::TextureView(v),
                    },
                ),
            });
            if let Some(p) = &mut self.profiler {
                p.begin_pass(encoder, "voxel_sun");
            }
            self.compute(
                encoder,
                visibility,
                &[group, outputs],
                [self.size[0].div_ceil(8), self.size[1].div_ceil(8), 1],
            );
            if let Some(p) = &mut self.profiler {
                p.end_pass(encoder, "voxel_sun");
            }
        }
        // Prepared metadata may now advance, but all future GPU work stays in
        // a later frame on this same queue, after this frame's traversal.
        self.residency.finish_frame();
    }
}

#[cfg(all(test, feature = "regional-publication-experiment"))]
#[path = "regional_gpu_tests.rs"]
mod regional_gpu_tests;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn far_traversal_matches_voxels_and_exits_at_planetary_boundaries() {
        pollster::block_on(async {
            let instance =
                wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
            let adapter = instance.request_adapter(&Default::default()).await.unwrap();
            let mut limits = wgpu::Limits::default();
            limits.max_storage_buffers_per_shader_stage = 8;
            limits.max_color_attachments = 8;
            limits.max_color_attachment_bytes_per_sample = 64;
            let (device, queue) = adapter
                .request_device(&wgpu::DeviceDescriptor {
                    required_limits: limits,
                    ..Default::default()
                })
                .await
                .unwrap();
            let terrain = StoredTerrain::new(&device, &queue, 16, 8, DepthConvention::Forward);
            let low = [-67_108_864, 0, -67_108_864];
            let node = residency::Node {
                low,
                level: 21,
                child: 0x8000_0000,
            };
            let job = residency::Job {
                low,
                level: node.level,
                slot: 0,
                pad: [0; 3],
            };
            terrain.upload_nodes(&[node]);
            queue.write_buffer(&terrain.jobs, 0, bytemuck::bytes_of(&job));
            // A completely empty sampled field must terminate at the root,
            // including negative crossings where boundary-1 rounds up in f32.
            queue.write_buffer(
                &terrain.materials,
                0,
                bytemuck::cast_slice(&[(-1.0f32).to_bits() | 1; 729]),
            );
            let mut params: Params = bytemuck::Zeroable::zeroed();
            params.origin[..3].copy_from_slice(&low.map(|v| v + 33_554_432));
            params.fraction = [0.5; 4];
            queue.write_buffer(&terrain.uniform, 0, bytemuck::bytes_of(&params));
            let source = format!(
                "{SHADER}\n{}\n{}",
                include_str!("stored.wgsl"),
                r#"
@compute @workgroup_size(1)
fn test_empty_rays(@builtin(global_invocation_id) id:vec3<u32>) {
    let i=id.x%27u;
    let q=vec3<i32>(i32(i%3u),i32((i/3u)%3u),i32(i/9u))-vec3<i32>(1);
    if all(q==vec3<i32>(0)) {return;}
    let rd=normalize(vec3<f32>(q));
    if id.x<27u {
        primary_hits[id.x]=stored_trace(vec3<f32>(0.0),rd,30000000.0);
    } else {
        // Also exercise the interpolation-cell walk directly. Whole-brick
        // rejection must not hide its large-coordinate progress failures.
        let n=StoredNode(vec3<i32>(bitcast<i32>(stored_nodes[0]),bitcast<i32>(stored_nodes[1]),bitcast<i32>(stored_nodes[2])),stored_nodes[3],stored_nodes[4]);
        let low=(vec3<f32>(n.low-p.origin.xyz)-p.fraction.xyz)*0.1;
        let end=stored_box(low,f32(32u<<n.level)*0.1,rd).far;
        primary_hits[id.x]=stored_density_hit(n,vec3<f32>(0.0),rd,0.0,end);
    }
}
// Independent, deliberately unaccelerated voxel DDA through the same field.
fn reference_density(n:StoredNode,ro:vec3<f32>,rd:vec3<f32>)->Hit {
    let entry=p.fraction.xyz+ro*10.0;
    let anchor=p.origin.xyz+vec3<i32>(floor(entry));let fraction=fract(entry);
    var cell=anchor;let high=n.low+vec3<i32>(i32(32u<<n.level));
    let step=select(vec3<i32>(-1),vec3<i32>(1),rd>=vec3<f32>(0.0));
    let inverse=1.0/select(vec3<f32>(1e-30),rd,abs(rd)>vec3<f32>(1e-30));
    for(var i=0u;i<32768u;i++) {
        let q=(cell-n.low)>>vec3<u32>(n.level+2u);
        let f=clamp(vec3<f32>(cell-n.low)/f32(4u<<n.level)-vec3<f32>(q),vec3<f32>(0.0),vec3<f32>(1.0));
        if stored_density_value(stored_density_grid(n,q,rd),f)>=0.0 {
            return Hit(cell,1u,rd,0.0);
        }
        let boundary=cell+select(vec3<i32>(0),vec3<i32>(1),rd>=vec3<f32>(0.0));
        let next=(vec3<f32>(boundary-anchor)-fraction)*0.1*inverse;
        var axis=0u;if next.y<next.x {axis=1u;}if next.z<next[axis] {axis=2u;}
        cell[axis]+=step[axis];
        if any(cell<n.low) || any(cell>=high) {return Hit(vec3<i32>(0),0u,rd,0.0);}
    }
    return Hit(cell,2u,rd,0.0);
}
@compute @workgroup_size(1)
fn test_density_rays(@builtin(global_invocation_id) id:vec3<u32>) {
    let n=StoredNode(vec3<i32>(bitcast<i32>(stored_nodes[0]),bitcast<i32>(stored_nodes[1]),bitcast<i32>(stored_nodes[2])),stored_nodes[3],stored_nodes[4]);
    let side=i32(32u<<n.level);let i=id.x;
    if p.lighting.w>0.0 {
        // Enter from above the brick, then hit the known plane's exact voxel
        // boundary. Recentring a clamped entry by half a cell changes distance
        // even when the returned voxel identity is correct.
        let point=vec3<f32>(f32(side)*0.5+0.37,f32(side)+7.37+f32(i)*0.131,f32(side)*0.5+0.73);
        let ro=(vec3<f32>(n.low-p.origin.xyz)+point-p.fraction.xyz)*0.1;
        let rd=vec3<f32>(0.0,-1.0,0.0);
        let low=(vec3<f32>(n.low-p.origin.xyz)-p.fraction.xyz)*0.1-ro;
        let interval=stored_box(low,f32(side)*0.1,rd);
        let last_solid=i32(floor(3.173*f32(4u<<n.level)));
        let expected_cell=n.low+vec3<i32>(i32(floor(point.x)),last_solid,i32(floor(point.z)));
        let expected_distance=(point.y-f32(last_solid+1))*0.1;
        primary_hits[i*2u]=stored_density_hit(n,ro,rd,interval.near,interval.far);
        primary_hits[i*2u+1u]=Hit(expected_cell,1u,rd,expected_distance);
        return;
    }
    let start_cell=n.low+vec3<i32>(i32(i*37u+3u)%side,i32(i*53u+7u)%side,i32(i*71u+11u)%side);
    let ro=(vec3<f32>(start_cell-p.origin.xyz)+vec3<f32>(0.37,0.51,0.73)-p.fraction.xyz)*0.1;
    var direction=vec3<f32>(f32(i%3u)-1.0,f32((i/3u)%3u)-1.0,f32((i/9u)%3u)-1.0);
    if all(direction==vec3<f32>(0.0)) {direction.x=1.0;}
    if i>=27u {direction.y*=0.0001;}
    let rd=normalize(direction);
    let low=(vec3<f32>(n.low-p.origin.xyz)-p.fraction.xyz)*0.1-ro;
    let end=stored_box(low,f32(side)*0.1,rd).far;
    primary_hits[i*2u]=stored_density_hit(n,ro,rd,0.0,end);
    primary_hits[i*2u+1u]=reference_density(n,ro,rd);
}

@compute @workgroup_size(1)
fn test_corner_rays(@builtin(global_invocation_id) id:vec3<u32>) {
    // Sunlight ray recorded during walking frame 27. Neighbouring ulps cover
    // both ownership orders of the nearly simultaneous Y/Z leaf crossing.
    var ro=vec3<f32>(-32.1799,-1.8997903,-86.789925);
    if id.x>0u {
        let axis=(id.x-1u)%3u;
        ro[axis]=bitcast<f32>(bitcast<u32>(ro[axis])+id.x/3u-10u);
    }
    let rd=vec3<f32>(0.42399913,0.84799826,0.31799936);
    primary_hits[id.x]=stored_trace(ro,rd,100.0);
}

"#
            );
            let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("far empty-space termination check"),
                source: wgpu::ShaderSource::Wgsl(source.into()),
            });
            let trace = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: None,
                layout: None,
                module: &shader,
                entry_point: Some("test_empty_rays"),
                compilation_options: Default::default(),
                cache: None,
            });
            let bounds_group = terrain.group(
                &terrain.bounds.get_bind_group_layout(0),
                &[
                    (25, &terrain.materials),
                    (26, &terrain.jobs),
                    (28, &terrain.exact_occupied),
                ],
            );
            let trace_group = terrain.group(
                &trace.get_bind_group_layout(0),
                &[
                    (0, &terrain.uniform),
                    (9, &terrain.hits),
                    (24, &terrain.nodes),
                    (25, &terrain.materials),
                    (28, &terrain.exact_occupied),
                ],
            );
            let readback = device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size: 128 * 32,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let mut encoder = device.create_command_encoder(&Default::default());
            terrain.compute(&mut encoder, &terrain.bounds, &[bounds_group], [8, 1, 1]);
            terrain.compute(&mut encoder, &trace, &[trace_group], [54, 1, 1]);
            encoder.copy_buffer_to_buffer(&terrain.hits, 0, &readback, 0, 54 * 32);
            queue.submit([encoder.finish()]);
            let (tx, rx) = std::sync::mpsc::channel();
            readback
                .slice(..)
                .map_async(wgpu::MapMode::Read, move |r| tx.send(r).unwrap());
            device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
            rx.recv().unwrap().unwrap();
            let bytes = readback.slice(..).get_mapped_range().unwrap();
            let words: &[u32] = bytemuck::cast_slice(&bytes);
            for (i, hit) in words.chunks_exact(8).take(54).enumerate() {
                if i % 27 == 13 {
                    continue;
                }
                assert_eq!(hit[3] & 3, 0, "empty ray {i} must exit, not exhaust");
                let distance = f32::from_bits(hit[7]);
                assert!(
                    distance.is_finite() && distance > 3_000_000.0,
                    "ray {i}: {distance}"
                );
            }
            drop(bytes);
            readback.unmap();
            let trace = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("accelerated versus voxel-by-voxel traversal"),
                layout: None,
                module: &shader,
                entry_point: Some("test_density_rays"),
                compilation_options: Default::default(),
                cache: None,
            });
            for level in [1, 4, 8] {
                for field in 0..6 {
                    let samples: Vec<u32> = (0..729)
                        .map(|i| {
                            let x = (i % 9) as f32;
                            let y = (i / 9 % 9) as f32;
                            let z = (i / 81) as f32;
                            let density = match field {
                                0 => y - 0.031 * x - 3.173,
                                1 => (x - 3.73) * (y - 4.27) * 0.23 + (z - 4.1) * 0.41,
                                2 => {
                                    if (i * 73 + 19) % 17 == 0 {
                                        0.19
                                    } else {
                                        -1.31
                                    }
                                }
                                3 => -0.7,
                                4 => 0.7,
                                _ => 3.173 - y,
                            };
                            (density.to_bits() & 0xffff_fffc) | 1
                        })
                        .collect();
                    let node = residency::Node {
                        low,
                        level,
                        child: 0x8000_0000,
                    };
                    terrain.upload_nodes(&[node]);
                    let job = residency::Job {
                        low,
                        level,
                        slot: 0,
                        pad: [0; 3],
                    };
                    queue.write_buffer(&terrain.jobs, 0, bytemuck::bytes_of(&job));
                    queue.write_buffer(&terrain.materials, 0, bytemuck::cast_slice(&samples));
                    params.origin[..3].copy_from_slice(&low);
                    params.lighting[3] = f32::from(field == 5);
                    queue.write_buffer(&terrain.uniform, 0, bytemuck::bytes_of(&params));
                    let bounds_group = terrain.group(
                        &terrain.bounds.get_bind_group_layout(0),
                        &[
                            (25, &terrain.materials),
                            (26, &terrain.jobs),
                            (28, &terrain.exact_occupied),
                        ],
                    );
                    let trace_group = terrain.group(
                        &trace.get_bind_group_layout(0),
                        &[
                            (0, &terrain.uniform),
                            (9, &terrain.hits),
                            (24, &terrain.nodes),
                            (25, &terrain.materials),
                        ],
                    );
                    let mut encoder = device.create_command_encoder(&Default::default());
                    terrain.compute(&mut encoder, &terrain.bounds, &[bounds_group], [8, 1, 1]);
                    terrain.compute(&mut encoder, &trace, &[trace_group], [64, 1, 1]);
                    encoder.copy_buffer_to_buffer(&terrain.hits, 0, &readback, 0, 128 * 32);
                    queue.submit([encoder.finish()]);
                    let (tx, rx) = std::sync::mpsc::channel();
                    readback
                        .slice(..)
                        .map_async(wgpu::MapMode::Read, move |r| tx.send(r).unwrap());
                    device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
                    rx.recv().unwrap().unwrap();
                    let bytes = readback.slice(..).get_mapped_range().unwrap();
                    let words: &[u32] = bytemuck::cast_slice(&bytes);
                    for (ray, pair) in words[..128 * 8].chunks_exact(16).enumerate() {
                        let actual = &pair[..8];
                        let expected = &pair[8..];
                        assert!(
                            expected[3] & 3 < 2,
                            "reference exhausted: {level}/{field}/{ray}"
                        );
                        assert_eq!(
                            actual[3] & 3,
                            expected[3] & 3,
                            "hit state: {level}/{field}/{ray}"
                        );
                        if expected[3] & 3 == 1 {
                            assert_eq!(
                                &actual[..3],
                                &expected[..3],
                                "first voxel: {level}/{field}/{ray}"
                            );
                            if field == 5 {
                                let actual_distance = f32::from_bits(actual[7]);
                                let expected_distance = f32::from_bits(expected[7]);
                                let tolerance = 0.00001 * (1u32 << level) as f32;
                                assert!(
                                    (actual_distance - expected_distance).abs() <= tolerance,
                                    "entry changed ray distance: level={level} ray={ray} actual={actual_distance} expected={expected_distance}"
                                );
                            }
                        }
                    }
                    drop(bytes);
                    readback.unmap();
                }
            }
            // An entirely empty, explicitly subdivided region must terminate
            // regardless of whether its leaves contain empty stored bricks.
            let mut nodes = vec![residency::Node {
                low: [-384, 63_717_504, -896], level: 2, child: 0,
            }];
            for index in 0..9 {
                let n = nodes[index];
                nodes[index].child = nodes.len() as u32;
                for octant in 0..8 {
                    nodes.push(residency::Node {
                        low: std::array::from_fn(|a| n.low[a] + ((octant >> a) & 1) * (16 << n.level)),
                        level: n.level - 1,
                        child: residency::AIR,
                    });
                }
            }
            terrain.upload_nodes(&nodes);
            let eye = World::default().ground_spawn(0.0, 0.0, 3.0) + glam::DVec3::new(1.08, 0.0, -0.81);
            let origin = crate::world::render_origin(eye);
            params.origin[..3].copy_from_slice(&origin);
            params.fraction = std::array::from_fn(|a| if a < 3 { (eye[a] / 0.1 - f64::from(origin[a])) as f32 } else { 0.0 });
            queue.write_buffer(&terrain.uniform, 0, bytemuck::bytes_of(&params));
            let corner = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("near-coincident empty leaf crossings"),
                layout: None, module: &shader, entry_point: Some("test_corner_rays"),
                compilation_options: Default::default(), cache: None,
            });
            let group = terrain.group(&corner.get_bind_group_layout(0), &[
                (0, &terrain.uniform), (9, &terrain.hits), (24, &terrain.nodes),
                (25, &terrain.materials), (28, &terrain.exact_occupied),
            ]);
            let mut encoder = device.create_command_encoder(&Default::default());
            terrain.compute(&mut encoder, &corner, &[group], [64, 1, 1]);
            encoder.copy_buffer_to_buffer(&terrain.hits, 0, &readback, 0, 64 * 32);
            queue.submit([encoder.finish()]);
            let (tx, rx) = std::sync::mpsc::channel();
            readback.slice(..).map_async(wgpu::MapMode::Read, move |r| tx.send(r).unwrap());
            device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
            rx.recv().unwrap().unwrap();
            let bytes = readback.slice(..).get_mapped_range().unwrap();
            for (i, hit) in bytemuck::cast_slice::<u8, u32>(&bytes)[..64*8].chunks_exact(8).enumerate() {
                assert_eq!(hit[3] & 3, 0, "empty corner ray {i} must exit, not exhaust at {}", f32::from_bits(hit[7]));
                assert!(f32::from_bits(hit[7]) > 7.0);
            }
        });
    }

    #[test]
    fn generated_exact_brick_matches_cpu_after_overlapping_edits() {
        pollster::block_on(async {
            let instance =
                wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
            let adapter = instance
                .request_adapter(&Default::default())
                .await
                .expect("GPU required");
            let mut limits = wgpu::Limits::default();
            limits.max_storage_buffers_per_shader_stage = 8;
            limits.max_color_attachments = 8;
            limits.max_color_attachment_bytes_per_sample = 64;
            let (device, queue) = adapter
                .request_device(&wgpu::DeviceDescriptor {
                    required_limits: limits,
                    ..Default::default()
                })
                .await
                .unwrap();
            let terrain = StoredTerrain::new(&device, &queue, 8, 8, DepthConvention::Forward);
            let mut world = World::default();
            let surface = world.ground_spawn(0.03, -256.03, 0.0);
            let center = crate::world::cell_of(surface);
            let low = center.map(|v| (v - 16).div_euclid(32) * 32);
            world
                .apply_edit(crate::world::Edit {
                    cell: center,
                    radius: 1.5,
                    material: 0,
                })
                .unwrap();
            world
                .apply_edit(crate::world::Edit {
                    cell: center.map(|v| v - 7),
                    radius: 1.0,
                    material: 3,
                })
                .unwrap();
            let edits: Vec<_> = world
                .edits
                .iter()
                .map(|e| GpuEdit {
                    cell: e.cell,
                    material: e.material,
                    radius: e.radius,
                    radius_units: e.radius_units(),
                    pad: [0.0; 2],
                })
                .collect();
            queue.write_buffer(&terrain.edits, 0, bytemuck::cast_slice(&edits));
            queue.write_buffer(
                &terrain.edit_references,
                0,
                bytemuck::cast_slice(&[0u32, 1u32]),
            );
            for step in [1, 2, 3, 5, 10] {
                world.set_voxel_size(f64::from(step) * 0.1).unwrap();
                queue.write_buffer(
                    &terrain.jobs,
                    0,
                    bytemuck::bytes_of(&residency::Job {
                        low,
                        level: 0,
                        slot: 0,
                        pad: [0, 2, step],
                    }),
                );
                let group = terrain.group(
                    &terrain.generate.get_bind_group_layout(0),
                    &[
                        (1, &terrain.edits),
                        (20, &terrain.field_settings),
                        (21, &terrain.heights),
                        (25, &terrain.materials),
                        (26, &terrain.jobs),
                        (27, &terrain.edit_references),
                    ],
                );
                let readback = device.create_buffer(&wgpu::BufferDescriptor {
                    label: None,
                    size: 8196,
                    usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
                let mut encoder = device.create_command_encoder(&Default::default());
                terrain.compute(&mut encoder, &terrain.generate, &[group], [32, 1, 1]);
                let bounds = terrain.group(
                    &terrain.bounds.get_bind_group_layout(0),
                    &[
                        (25, &terrain.materials),
                        (26, &terrain.jobs),
                        (28, &terrain.exact_occupied),
                    ],
                );
                terrain.compute(&mut encoder, &terrain.bounds, &[bounds], [8, 1, 1]);
                encoder.copy_buffer_to_buffer(&terrain.materials, 0, &readback, 0, 8192);
                encoder.copy_buffer_to_buffer(&terrain.exact_occupied, 0, &readback, 8192, 4);
                queue.submit([encoder.finish()]);
                let (tx, rx) = std::sync::mpsc::channel();
                readback
                    .slice(..)
                    .map_async(wgpu::MapMode::Read, move |r| tx.send(r).unwrap());
                device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
                rx.recv().unwrap().unwrap();
                let bytes = readback.slice(..).get_mapped_range().unwrap();
                let words: &[u32] = bytemuck::cast_slice(&bytes);
                assert_eq!(words[2048], words[..2048].iter().fold(0, |a, b| a | b));
                let mut counts = [0usize; 4];
                for i in 0..32768usize {
                    let cell = [
                        low[0] + (i % 32) as i32,
                        low[1] + (i / 32 % 32) as i32,
                        low[2] + (i / 1024) as i32,
                    ];
                    let gpu = (words[i / 16] >> ((i % 16) * 2)) & 3;
                    let cpu = world.material(cell);
                    assert_eq!(gpu, cpu, "generated material at {cell:?}, step={step}");
                    counts[gpu as usize] += 1;
                }
                assert!(
                    counts[0] > 0 && counts[1] > 0 && counts[3] > 0,
                    "test must cover air, terrain and added material: {counts:?}"
                );
            }
            // A reused slot must lose its old certificate after destruction.
            // The last material in the final word also exercises all lanes of
            // the complete reduction, rather than just the first samples.
            terrain.upload_nodes(&[residency::Node {
                low,
                level: 0,
                child: 0x8000_0000,
            }]);
            let mut params: Params = bytemuck::Zeroable::zeroed();
            params.origin[..3].copy_from_slice(&low.map(|v| v + 16));
            params.fraction = [0.5; 4];
            params.settings[3] = 1.0;
            queue.write_buffer(&terrain.uniform, 0, bytemuck::bytes_of(&params));
            let source = format!(
                "{SHADER}\n{}\n{}",
                include_str!("stored.wgsl"),
                r#"
@compute @workgroup_size(1)
fn certificate_rays(@builtin(global_invocation_id) id:vec3<u32>) {
    let i=id.x;
    var rd=vec3<f32>(f32(i%3u)-1.0,f32((i/3u)%3u)-1.0,f32((i/9u)%3u)-1.0);
    if i==13u {rd.x=1.0;}
    primary_hits[i]=stored_trace(vec3<f32>(0.0),normalize(rd),10.0);
}
"#
            );
            let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("exact empty brick certificate rays"),
                source: wgpu::ShaderSource::Wgsl(source.into()),
            });
            let trace = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: None,
                layout: None,
                module: &shader,
                entry_point: Some("certificate_rays"),
                compilation_options: Default::default(),
                cache: None,
            });
            for tail in [0u32, 3 << 30, 0] {
                let mut payload = [0u32; 2048];
                payload[2047] = tail;
                queue.write_buffer(&terrain.materials, 0, bytemuck::cast_slice(&payload));
                let bounds = terrain.group(
                    &terrain.bounds.get_bind_group_layout(0),
                    &[
                        (25, &terrain.materials),
                        (26, &terrain.jobs),
                        (28, &terrain.exact_occupied),
                    ],
                );
                let readback = device.create_buffer(&wgpu::BufferDescriptor {
                    label: None,
                    size: 4 + 27 * 32,
                    usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
                let mut encoder = device.create_command_encoder(&Default::default());
                terrain.compute(&mut encoder, &terrain.bounds, &[bounds], [8, 1, 1]);
                let rays = terrain.group(
                    &trace.get_bind_group_layout(0),
                    &[
                        (0, &terrain.uniform),
                        (9, &terrain.hits),
                        (24, &terrain.nodes),
                        (25, &terrain.materials),
                        (28, &terrain.exact_occupied),
                    ],
                );
                terrain.compute(&mut encoder, &trace, &[rays], [27, 1, 1]);
                encoder.copy_buffer_to_buffer(&terrain.exact_occupied, 0, &readback, 0, 4);
                encoder.copy_buffer_to_buffer(&terrain.hits, 0, &readback, 4, 27 * 32);
                queue.submit([encoder.finish()]);
                let (tx, rx) = std::sync::mpsc::channel();
                readback
                    .slice(..)
                    .map_async(wgpu::MapMode::Read, move |r| tx.send(r).unwrap());
                device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
                rx.recv().unwrap().unwrap();
                let bytes = readback.slice(..).get_mapped_range().unwrap();
                assert_eq!(u32::from_le_bytes(bytes[..4].try_into().unwrap()), tail);
                let hits: &[u32] = bytemuck::cast_slice(&bytes[4..]);
                for (ray, hit) in hits.chunks_exact(8).enumerate() {
                    let expected = u32::from(tail != 0 && ray == 26);
                    assert_eq!(hit[3] & 3, expected, "certificate ray={ray}, tail={tail}");
                    assert!(f32::from_bits(hit[7]).is_finite());
                    if expected != 0 {
                        assert_eq!(&hit[..3], &low.map(|v| (v + 31) as u32));
                        assert_eq!((hit[3] >> 8) & 3, 3);
                    }
                }
            }
        });
    }
}
