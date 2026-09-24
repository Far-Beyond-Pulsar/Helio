use super::{
    residency::{self, Residency},
    DepthConvention, GBufferTargets, GBUFFER_FORMATS,
};
use crate::{GpuEdit, Params, World, SHADER};
use std::sync::Arc;
use wgpu::util::DeviceExt;

/// The sole engine terrain backend: budgeted brick production and stored rays.
pub struct StoredTerrain {
    pub(crate) device: wgpu::Device,
    queue: wgpu::Queue,
    pub size: [u32; 2],
    world: Arc<World>,
    generation_world: Option<Arc<World>>,
    residency: Residency,
    nodes: wgpu::Buffer,
    materials: wgpu::Buffer,
    jobs: wgpu::Buffer,
    edits: wgpu::Buffer,
    edit_references: wgpu::Buffer,
    field_settings: wgpu::Buffer,
    heights: wgpu::Buffer,
    uniform: wgpu::Buffer,
    hits: wgpu::Buffer,
    generate: wgpu::ComputePipeline,
    bounds: wgpu::ComputePipeline,
    trace: wgpu::ComputePipeline,
    visibility: wgpu::ComputePipeline,
    surface: wgpu::RenderPipeline,
    sun: wgpu::Texture,
    pub(crate) sun_view: wgpu::TextureView,
    direction: wgpu::Texture,
    pub(crate) profiler: Option<helio_core::profiling::GpuProfiler>,
}
fn buffer(device: &wgpu::Device, label: &str, size: u64) -> wgpu::Buffer {
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
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("stored voxel terrain"),
            source: wgpu::ShaderSource::Wgsl(source.into()),
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
            device: device.clone(),
            queue: queue.clone(),
            size,
            world: Arc::new(World::default()),
            generation_world: None,
            residency: Residency::new(capacity),
            nodes: buffer(
                device,
                "resident voxel tree",
                (residency::NODE_CAPACITY as u64 + 4) * 4,
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
    fn upload_nodes(&self, nodes: &[residency::Node]) {
        let root = &nodes[0];
        let mut links = Vec::with_capacity(nodes.len() + 4);
        links.extend(root.low.map(|v| v as u32));
        links.push(root.level);
        links.extend(nodes.iter().map(|n| n.child));
        self.queue
            .write_buffer(&self.nodes, 0, bytemuck::cast_slice(&links));
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
    pub fn primary_hit_buffer(&self) -> Option<&wgpu::Buffer> {
        Some(&self.hits)
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
        self.residency.update(&self.world, params);
        if let Some((jobs, references, world)) = self.residency.next_batch() {
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
                    &[(25, &self.materials), (26, &self.jobs)],
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
            if let Some(nodes) = self.residency.publish() {
                self.upload_nodes(&nodes);
            }
        }
        let mut p = *params;
        p.settings[2] = if self.residency.stats.ready { 1.0 } else { 0.0 };
        self.queue
            .write_buffer(&self.uniform, 0, bytemuck::bytes_of(&p));
        let group = self.group(
            &self.trace.get_bind_group_layout(0),
            &[
                (0, &self.uniform),
                (9, &self.hits),
                (24, &self.nodes),
                (25, &self.materials),
            ],
        );
        let cameras = self.group(&self.trace.get_bind_group_layout(1), &[(0, camera)]);
        if let Some(p) = &mut self.profiler {
            p.begin_pass(encoder, "voxel_primary");
        }
        self.compute(
            encoder,
            &self.trace,
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
            let group = self.group(
                &self.visibility.get_bind_group_layout(0),
                &[
                    (0, &self.uniform),
                    (9, &self.hits),
                    (24, &self.nodes),
                    (25, &self.materials),
                ],
            );
            let direction = self.direction.create_view(&Default::default());
            let outputs = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("stored sunlight outputs"),
                layout: &self.visibility.get_bind_group_layout(1),
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
                &self.visibility,
                &[group, outputs],
                [self.size[0].div_ceil(8), self.size[1].div_ceil(8), 1],
            );
            if let Some(p) = &mut self.profiler {
                p.end_pass(encoder, "voxel_sun");
            }
        }
    }
}

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
                &[(25, &terrain.materials), (26, &terrain.jobs)],
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
                for field in 0..5 {
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
                                _ => 0.7,
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
                    queue.write_buffer(&terrain.uniform, 0, bytemuck::bytes_of(&params));
                    let bounds_group = terrain.group(
                        &terrain.bounds.get_bind_group_layout(0),
                        &[(25, &terrain.materials), (26, &terrain.jobs)],
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
                        }
                    }
                    drop(bytes);
                    readback.unmap();
                }
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
            queue.write_buffer(
                &terrain.jobs,
                0,
                bytemuck::bytes_of(&residency::Job {
                    low,
                    level: 0,
                    slot: 0,
                    pad: [0, 2, 0],
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
                size: 8192,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let mut encoder = device.create_command_encoder(&Default::default());
            terrain.compute(&mut encoder, &terrain.generate, &[group], [32, 1, 1]);
            encoder.copy_buffer_to_buffer(&terrain.materials, 0, &readback, 0, 8192);
            queue.submit([encoder.finish()]);
            let (tx, rx) = std::sync::mpsc::channel();
            readback
                .slice(..)
                .map_async(wgpu::MapMode::Read, move |r| tx.send(r).unwrap());
            device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
            rx.recv().unwrap().unwrap();
            let bytes = readback.slice(..).get_mapped_range().unwrap();
            let words: &[u32] = bytemuck::cast_slice(&bytes);
            let mut counts = [0usize; 4];
            for i in 0..32768usize {
                let cell = [
                    low[0] + (i % 32) as i32,
                    low[1] + (i / 32 % 32) as i32,
                    low[2] + (i / 1024) as i32,
                ];
                let gpu = (words[i / 16] >> ((i % 16) * 2)) & 3;
                let cpu = world.material(cell);
                assert_eq!(gpu, cpu, "generated material at {cell:?}");
                counts[gpu as usize] += 1;
            }
            assert!(
                counts[0] > 0 && counts[1] > 0 && counts[3] > 0,
                "test must cover air, terrain and added material: {counts:?}"
            );
        });
    }
}
