//! A stable GPU radix sort of `(u32 key, u32 index)` pairs, ascending by key:
//! eight passes of 4-bit digits, three dispatches a pass, all recorded into
//! the caller's compute pass (`shaders/radix_sort.wgsl`).
//!
//! The caller owns the pairs (two ping-pong pairs of buffers, the sorted
//! result back in the first after the eight passes) and a prepare step of
//! its own that writes the live count: a storage buffer holding
//! `(count, num_blocks)`, `num_blocks = max(ceil(count / 256), 1)`, and an
//! indirect dispatch of `(num_blocks, 1, 1)`. The sort's dispatches follow
//! that count, not the buffers' capacity.

use wgpu::util::DeviceExt;

/// Elements a block (the kernels' workgroup size).
pub const RADIX_SORT_BLOCK: u32 = 256;
const DIGITS: u32 = 16;
const PASSES: usize = 8;

/// The sort's pipelines, shared by every set of buffers it sorts.
pub struct RadixSort {
    layout: wgpu::BindGroupLayout,
    histogram: wgpu::ComputePipeline,
    scan: wgpu::ComputePipeline,
    scatter: wgpu::ComputePipeline,
}

/// One set of buffers bound for sorting: a bind group a pass, and the
/// per-block digit counts.
pub struct RadixSortBinds {
    passes: [wgpu::BindGroup; PASSES],
}

fn entry(binding: u32, buffer: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry { binding, resource: buffer.as_entire_binding() }
}

impl RadixSort {
    pub fn new(device: &wgpu::Device) -> Self {
        let storage = |binding: u32, read_only: bool| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only }, has_dynamic_offset: false, min_binding_size: None },
            count: None,
        };
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("radix sort"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None },
                    count: None,
                },
                storage(1, true),
                storage(2, true),
                storage(3, true),
                storage(4, false),
                storage(5, false),
                storage(6, false),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("radix sort"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let module = crate::shader::module(device, "radix sort", crate::include_wgsl!("../shaders/radix_sort.wgsl"));
        let pipeline = |entry: &str| {
            crate::shader::create_compute_pipeline(&device, wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: Some(&pipeline_layout),
                module: &module,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        Self { histogram: pipeline("cs_histogram"), scan: pipeline("cs_scan"), scatter: pipeline("cs_scatter"), layout }
    }

    /// Binds pairs `keys[0]`/`indices[0]` (input, then the sorted result)
    /// and `keys[1]`/`indices[1]` (scratch), each `capacity` elements, with
    /// the caller's `frame` (`(count, num_blocks)` as two `u32`s).
    pub fn bind(&self, device: &wgpu::Device, frame: &wgpu::Buffer, keys: [&wgpu::Buffer; 2], indices: [&wgpu::Buffer; 2], capacity: u32) -> RadixSortBinds {
        let blocks = capacity.div_ceil(RADIX_SORT_BLOCK).max(1);
        let hist = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("radix sort digit counts"),
            size: u64::from(blocks * DIGITS) * 4,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let passes = std::array::from_fn(|i| {
            let shift = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("radix sort pass"),
                contents: bytemuck::bytes_of(&[(i as u32) * 4, 0, 0, 0]),
                usage: wgpu::BufferUsages::UNIFORM,
            });
            let (from, to) = if i % 2 == 0 { (0, 1) } else { (1, 0) };
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("radix sort pass"),
                layout: &self.layout,
                entries: &[
                    entry(0, &shift),
                    entry(1, frame),
                    entry(2, keys[from]),
                    entry(3, indices[from]),
                    entry(4, keys[to]),
                    entry(5, indices[to]),
                    entry(6, &hist),
                ],
            })
        });
        RadixSortBinds { passes }
    }

    /// The sort's 24 dispatches into `pass`; `dispatch` holds the caller's
    /// `(num_blocks, 1, 1)` at `offset`.
    pub fn record(&self, pass: &mut crate::cmd::ComputeCmds<'_>, binds: &RadixSortBinds, dispatch: &wgpu::Buffer, offset: u64) {
        for group in &binds.passes {
            pass.set_bind_group(0, group, &[]);
            pass.set_pipeline(&self.histogram);
            pass.dispatch_workgroups_indirect(dispatch, offset);
            pass.set_pipeline(&self.scan);
            pass.dispatch_workgroups(1, 1, 1);
            pass.set_pipeline(&self.scatter);
            pass.dispatch_workgroups_indirect(dispatch, offset);
        }
    }
}
