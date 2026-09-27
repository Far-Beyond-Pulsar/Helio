//! Local exact-face raster experiment. Coordinates are authored cells relative
//! to one tile. Planetary camera rebasing, residency and engine lighting are not
//! implemented here; this stage tests whether raster visibility is worth using.
use super::Mesh;
use glam::{DVec3, Vec3};
use wgpu::util::DeviceExt;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Camera {
    pub eye: [f32; 4],
    pub right: [f32; 4],
    pub up: [f32; 4],
    pub forward: [f32; 4],
    pub light: [f32; 4],
}
impl Camera {
    pub fn new(eye: DVec3, forward: DVec3, size: [u32; 2], light: Vec3) -> Self {
        assert!(size.iter().all(|&v| v > 0));
        let forward = forward.normalize().as_vec3();
        let reference = if forward.y.abs() > 0.95 {
            Vec3::Z
        } else {
            Vec3::Y
        };
        let right = forward.cross(reference).normalize();
        let up = right.cross(forward).normalize();
        Self {
            eye: eye.as_vec3().extend(0.55).to_array(),
            right: right.extend(size[0] as f32 / size[1] as f32).to_array(),
            up: up.extend(0.01).to_array(),
            forward: forward.extend(size[1] as f32).to_array(),
            light: light.normalize().extend(size[0] as f32).to_array(),
        }
    }
}

#[derive(Clone, Copy)]
pub enum Output {
    /// Packed local cell, material/entry face, ray distance, quad index.
    Identity,
    /// Synthetic directional Lambert lighting, for spatial filtering probes.
    /// This is deliberately not presented as the engine's lighting model.
    Lit,
}
impl Output {
    pub fn format(self) -> wgpu::TextureFormat {
        match self {
            Self::Identity => wgpu::TextureFormat::Rgba32Uint,
            Self::Lit => wgpu::TextureFormat::Rgba16Float,
        }
    }
}

pub struct Raster {
    pipeline: wgpu::RenderPipeline,
    group: wgpu::BindGroup,
    camera: wgpu::Buffer,
    groups: Vec<(u32, std::ops::Range<u32>)>,
}
impl Raster {
    pub fn new(device: &wgpu::Device, mesh: &Mesh, output: Output, samples: u32) -> Self {
        assert!(samples == 1 || samples == 4);
        assert!(!matches!(output, Output::Identity) || samples == 1);
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("exact voxel face raster experiment"),
            source: wgpu::ShaderSource::Wgsl(
                (include_str!("topology.wgsl").to_owned() + include_str!("raster.wgsl")).into(),
            ),
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("exact voxel face raster experiment"),
            layout: None,
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vertex"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            primitive: wgpu::PrimitiveState {
                cull_mode: Some(wgpu::Face::Back),
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: Some(true),
                depth_compare: Some(wgpu::CompareFunction::Greater),
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: wgpu::MultisampleState {
                count: samples,
                ..Default::default()
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some(match output {
                    Output::Identity => "identity",
                    Output::Lit => "lit",
                }),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: output.format(),
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview_mask: None,
            cache: None,
        });
        // Procedurally generate the exact conforming fan from each 8-byte
        // record. Power-of-two groups bound degenerate padding below 2x and
        // require at most seven draws, without a triangle-index allocation.
        let mut buckets: [Vec<super::Quad>; 8] = Default::default();
        for &quad in &mesh.quads {
            let bucket = quad.triangles().next_power_of_two().ilog2() as usize;
            buckets[bucket].push(quad);
        }
        let mut packed = Vec::with_capacity(mesh.quads.len());
        let mut groups = Vec::new();
        for (bucket, quads) in buckets.into_iter().enumerate() {
            if quads.is_empty() {
                continue;
            }
            let start = packed.len() as u32;
            packed.extend(quads);
            groups.push((1 << bucket, start..packed.len() as u32));
        }
        // Empty meshes still need a legal storage binding.
        let empty = [0u32; 2];
        let quads = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("exact integer surface quads"),
            contents: if packed.is_empty() {
                bytemuck::cast_slice(&empty)
            } else {
                bytemuck::cast_slice(&packed)
            },
            usage: wgpu::BufferUsages::STORAGE,
        });
        let camera = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("local surface mesh camera"),
            size: std::mem::size_of::<Camera>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: camera.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: quads.as_entire_binding(),
                },
            ],
        });
        Self {
            pipeline,
            group,
            camera,
            groups,
        }
    }
    pub fn camera(&self, queue: &wgpu::Queue, camera: &Camera) {
        queue.write_buffer(&self.camera, 0, bytemuck::bytes_of(camera));
    }
    pub fn encode(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        color: &wgpu::TextureView,
        depth: &wgpu::TextureView,
        resolve: Option<&wgpu::TextureView>,
        timestamps: Option<wgpu::RenderPassTimestampWrites<'_>>,
    ) {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("exact voxel face visibility"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: color,
                depth_slice: None,
                resolve_target: resolve,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: depth,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(0.0),
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: timestamps,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.group, &[]);
        for (triangles, instances) in &self.groups {
            pass.draw(0..triangles * 3, instances.clone());
        }
    }
}
