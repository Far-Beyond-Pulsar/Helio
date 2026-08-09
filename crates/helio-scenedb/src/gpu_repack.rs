//! GPU instance repack pass: reads SceneDB's HelioGpuInstance packed column +
//! cull output, writes sorted instance data to render-owned pipeline buffers.
//!
//! This is the GPU compute equivalent of the CPU-side
//! `rebuild_instance_buffers()`. It replaces the CPU sort with a GPU atomic
//! bucket gather, eliminating ALL per-frame CPU work for instance data.

use std::sync::Arc;

use crate::wgsl::{REPACK_WGSL, SCENE_BINDINGS_WGSL};

const WORKGROUP_SIZE: u32 = 64;
const BUCKET_COUNT: u32 = 1024;

/// Pipeline buffer targets for the repack pass output.
pub struct RepackOutputs {
    pub sorted_instances: Arc<wgpu::Buffer>,
    pub draw_calls: Arc<wgpu::Buffer>,
    pub indirect_args: Arc<wgpu::Buffer>,
    pub aabbs: Arc<wgpu::Buffer>,
    pub capacity: u32,
}

/// GPU compute pass that sorts and gathers instance data.
pub struct GpuInstanceRepackPass {
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    bucket_counts_buf: wgpu::Buffer,
    bucket_offsets_buf: wgpu::Buffer,
}

impl GpuInstanceRepackPass {
    #[must_use]
    pub fn new(device: &wgpu::Device, instances_layout: &wgpu::BindGroupLayout) -> Self {
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("GpuInstanceRepackPass output layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("GpuInstanceRepackPass shader"),
            source: wgpu::ShaderSource::Wgsl(
                format!("{SCENE_BINDINGS_WGSL}\n{REPACK_WGSL}").into(),
            ),
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("GpuInstanceRepackPass pipeline layout"),
            bind_group_layouts: &[Some(instances_layout), Some(&bind_group_layout)],
            immediate_size: 0,
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("GpuInstanceRepackPass"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("repack_main"),
            compilation_options: Default::default(),
            cache: None,
        });

        let bucket_counts_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("GpuInstanceRepackPass bucket counts"),
            size: BUCKET_COUNT as u64 * 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bucket_offsets_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("GpuInstanceRepackPass bucket offsets"),
            size: BUCKET_COUNT as u64 * 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self { pipeline, bind_group_layout, bucket_counts_buf, bucket_offsets_buf }
    }

    pub fn output_layout(&self) -> &wgpu::BindGroupLayout {
        &self.bind_group_layout
    }

    pub fn record(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        instances_bind_group: &wgpu::BindGroup,
        output_bind_group: &wgpu::BindGroup,
        instance_count: u32,
    ) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("RepackInstances Count"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, instances_bind_group, &[]);
        pass.set_bind_group(1, output_bind_group, &[]);
        pass.dispatch_workgroups(instance_count.div_ceil(WORKGROUP_SIZE), 1, 1);
    }

    #[must_use]
    pub fn compute_offsets(&self, device: &wgpu::Device, queue: &wgpu::Queue) -> Vec<u32> {
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("bucket counts readback"),
            size: BUCKET_COUNT as u64 * 4,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("bucket count readback"),
        });
        enc.copy_buffer_to_buffer(&self.bucket_counts_buf, 0, &staging, 0, BUCKET_COUNT as u64 * 4);
        queue.submit([enc.finish()]);
        let slice = staging.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        device.poll(wgpu::PollType::wait_indefinitely()).expect("poll");
        let data: Vec<u8> = slice.get_mapped_range().expect("mapped range").to_vec();
        staging.unmap();

        let counts: Vec<u32> = data.chunks_exact(4).map(|c| u32::from_ne_bytes(c.try_into().unwrap())).collect();
        let offsets: Vec<u32> = counts.iter().scan(0u32, |state, &c| {
            let off = *state;
            *state += c;
            Some(off)
        }).collect();

        let offset_bytes: Vec<u8> = offsets.iter().flat_map(|o| o.to_ne_bytes()).collect();
        queue.write_buffer(&self.bucket_offsets_buf, 0, &offset_bytes);

        let mut sorted: Vec<(u32, u32)> = counts.iter().copied().enumerate().filter(|(_, c)| *c > 0).map(|(i, c)| (i as u32, c)).collect();
        sorted.sort_by_key(|(_, c)| *c);
        sorted.into_iter().map(|(i, _)| i).collect()
    }

    pub fn reset_bucket_counts(&self, queue: &wgpu::Queue) {
        queue.write_buffer(&self.bucket_counts_buf, 0, &[0u8; BUCKET_COUNT as usize * 4]);
    }
}
