//! Opt-in primary visibility accelerator. A bounded arena stores exact faces;
//! nonresident mesh pages render only an internal fallback marker.
use super::*;
use crate::surface_mesh::{Mesh, Quad};

const BUCKETS: usize = 7;
const CAPACITY: u32 = 65_536;
const MAX_TILE_QUADS: usize = 16_384;

pub(super) struct Prepared {
    buckets: [Vec<Quad>; BUCKETS],
    rejected: bool,
}
impl Prepared {
    pub fn new(mesh: Mesh) -> Self {
        let mut out = Self {
            buckets: Default::default(),
            rejected: mesh.quads.len() > MAX_TILE_QUADS,
        };
        if !out.rejected {
            for quad in mesh.quads {
                let bucket = quad.triangles().next_power_of_two().ilog2() as usize - 1;
                out.buckets[bucket].push(quad);
            }
        }
        out
    }
}

struct Target {
    size: [u32; 2],
    color: wgpu::TextureView,
    depth: wgpu::TextureView,
}
pub(super) struct RasterPatch {
    faces: wgpu::RenderPipeline,
    missing: wgpu::RenderPipeline,
    resolve: wgpu::ComputePipeline,
    quads: wgpu::Buffer,
    ready: wgpu::Buffer,
    counts: [u32; BUCKETS],
    target: Option<Target>,
    pub accepted: usize,
    pub rejected: usize,
    pub uploaded_bytes: u64,
}
fn source() -> String {
    format!(
        "{}\n{}\n{}\n{}\n{}\n{}",
        crate::SHADER,
        crate::surface_cache::GPU_SHADER,
        include_str!("../surface_patch.wgsl"),
        include_str!("../../surface_mesh/topology.wgsl"),
        include_str!("../surface_patch_predicates.wgsl"),
        include_str!("mesh.wgsl")
    )
}
impl RasterPatch {
    pub fn new(device: &wgpu::Device) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("exact surface mesh patch"),
            source: wgpu::ShaderSource::Wgsl(source().into()),
        });
        let render = |entry| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(entry),
                layout: None,
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some(entry),
                    compilation_options: Default::default(),
                    buffers: &[],
                },
                primitive: wgpu::PrimitiveState {
                    cull_mode: Some(wgpu::Face::Back),
                    conservative: true,
                    ..Default::default()
                },
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: wgpu::TextureFormat::Depth32Float,
                    depth_write_enabled: Some(true),
                    depth_compare: Some(wgpu::CompareFunction::Greater),
                    stencil: Default::default(),
                    bias: Default::default(),
                }),
                multisample: Default::default(),
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some("mesh_identity"),
                    compilation_options: Default::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: wgpu::TextureFormat::Rg32Uint,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                multiview_mask: None,
                cache: None,
            })
        };
        let faces = render("mesh_vertex");
        let missing = render("missing_vertex");
        let resolve = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("mesh exact hit resolve"),
            layout: None,
            module: &shader,
            entry_point: Some("mesh_resolve"),
            compilation_options: Default::default(),
            cache: None,
        });
        Self {
            faces,
            missing,
            resolve,
            quads: super::super::terrain::buffer(
                device,
                "bounded surface mesh quads",
                BUCKETS as u64 * u64::from(CAPACITY) * 8,
            ),
            ready: device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("mesh page readiness"),
                contents: bytemuck::cast_slice(&[0u32; TILES]),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            }),
            counts: [0; BUCKETS],
            target: None,
            accepted: 0,
            rejected: 0,
            uploaded_bytes: 0,
        }
    }
    pub fn reset(&mut self, queue: &wgpu::Queue) {
        self.counts.fill(0);
        self.accepted = 0;
        self.rejected = 0;
        self.uploaded_bytes = 0;
        queue.write_buffer(&self.ready, 0, bytemuck::cast_slice(&[0u32; TILES]));
    }
    pub fn upload(&mut self, queue: &wgpu::Queue, slot: usize, mut prepared: Prepared) {
        assert!(slot < TILES);
        if prepared.rejected
            || prepared
                .buckets
                .iter()
                .enumerate()
                .any(|(b, q)| self.counts[b] as usize + q.len() > CAPACITY as usize)
        {
            self.rejected += 1;
            return;
        }
        for (b, quads) in prepared.buckets.iter_mut().enumerate() {
            if quads.is_empty() {
                continue;
            }
            for quad in &mut *quads {
                quad.extent |= (slot as u32) << 12;
            }
            queue.write_buffer(
                &self.quads,
                (b as u64 * u64::from(CAPACITY) + u64::from(self.counts[b])) * 8,
                bytemuck::cast_slice(quads),
            );
            self.counts[b] += quads.len() as u32;
            self.uploaded_bytes += quads.len() as u64 * 8;
        }
        queue.write_buffer(&self.ready, slot as u64 * 4, bytemuck::bytes_of(&1u32));
        self.accepted += 1;
        self.uploaded_bytes += 4;
    }
    pub fn memory(&self) -> (u64, u64) {
        (
            self.quads.size() + self.ready.size(),
            self.target
                .as_ref()
                .map_or(0, |t| u64::from(t.size[0]) * u64::from(t.size[1]) * 12),
        )
    }
    pub fn encode(
        &mut self,
        device: &wgpu::Device,
        params: &wgpu::Buffer,
        camera: &wgpu::Buffer,
        hits: &wgpu::Buffer,
        settings: &wgpu::Buffer,
        directory: &wgpu::Buffer,
        words: &wgpu::Buffer,
        encoder: &mut wgpu::CommandEncoder,
        size: [u32; 2],
    ) {
        if self.target.as_ref().is_none_or(|t| t.size != size) {
            let texture = |format, usage| {
                device
                    .create_texture(&wgpu::TextureDescriptor {
                        label: Some("surface mesh visibility"),
                        size: wgpu::Extent3d {
                            width: size[0],
                            height: size[1],
                            depth_or_array_layers: 1,
                        },
                        mip_level_count: 1,
                        sample_count: 1,
                        dimension: wgpu::TextureDimension::D2,
                        format,
                        usage,
                        view_formats: &[],
                    })
                    .create_view(&Default::default())
            };
            self.target = Some(Target {
                size,
                color: texture(
                    wgpu::TextureFormat::Rg32Uint,
                    wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
                ),
                depth: texture(
                    wgpu::TextureFormat::Depth32Float,
                    wgpu::TextureUsages::RENDER_ATTACHMENT,
                ),
            });
        }
        let target = self.target.as_ref().unwrap();
        let group = |layout: &wgpu::BindGroupLayout, buffers: &[(u32, &wgpu::Buffer)]| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("mesh patch buffers"),
                layout,
                entries: &buffers
                    .iter()
                    .map(|(binding, buffer)| wgpu::BindGroupEntry {
                        binding: *binding,
                        resource: buffer.as_entire_binding(),
                    })
                    .collect::<Vec<_>>(),
            })
        };
        let faces = group(
            &self.faces.get_bind_group_layout(0),
            &[(0, params), (31, settings), (34, &self.quads)],
        );
        let missing = group(
            &self.missing.get_bind_group_layout(0),
            &[(0, params), (31, settings), (35, &self.ready)],
        );
        let face_camera = group(&self.faces.get_bind_group_layout(1), &[(0, camera)]);
        let missing_camera = group(&self.missing.get_bind_group_layout(1), &[(0, camera)]);
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("voxel exact faces and unavailable page markers"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &target.color,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &target.depth,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(0.0),
                        store: wgpu::StoreOp::Discard,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&self.faces);
            pass.set_bind_group(0, &faces, &[]);
            pass.set_bind_group(1, &face_camera, &[]);
            for (bucket, &count) in self.counts.iter().enumerate() {
                if count > 0 {
                    let start = bucket as u32 * CAPACITY;
                    pass.draw(0..(2u32 << bucket) * 3, start..start + count);
                }
            }
            pass.set_pipeline(&self.missing);
            pass.set_bind_group(0, &missing, &[]);
            pass.set_bind_group(1, &missing_camera, &[]);
            if self.accepted < TILES {
                pass.draw(0..36, 0..TILES as u32);
            }
        }
        let mut entries = [
            (0, params),
            (9, hits),
            (31, settings),
            (32, directory),
            (33, words),
            (35, &self.ready),
        ]
        .map(|(binding, buffer)| wgpu::BindGroupEntry {
            binding,
            resource: buffer.as_entire_binding(),
        })
        .to_vec();
        entries.push(wgpu::BindGroupEntry {
            binding: 36,
            resource: wgpu::BindingResource::TextureView(&target.color),
        });
        let inputs = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("mesh hit resolve"),
            layout: &self.resolve.get_bind_group_layout(0),
            entries: &entries,
        });
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&self.resolve);
        pass.set_bind_group(0, &inputs, &[]);
        pass.dispatch_workgroups(size[0].div_ceil(8), size[1].div_ceil(8), 1);
    }
}

#[cfg(test)]
mod tests;
