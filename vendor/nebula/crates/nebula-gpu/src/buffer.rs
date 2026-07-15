use std::marker::PhantomData;
use std::sync::Arc;
use bytemuck::Pod;

// ── Uniform buffer ─────────────────────────────────────────────────────────────

/// A host-mapped uniform buffer that holds exactly one value of type `T`.
pub struct UniformBuffer<T: Pod> {
    pub buf:    wgpu::Buffer,
    pub layout: wgpu::BindGroupLayout,
    pub group:  wgpu::BindGroup,
    _pd: PhantomData<T>,
}

impl<T: Pod> UniformBuffer<T> {
    pub fn new(device: &wgpu::Device, label: &str, value: &T) -> Self {
        use wgpu::util::DeviceExt;
        let buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label:    Some(label),
            contents: bytemuck::bytes_of(value),
            usage:    wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label:   Some(&format!("{label}_layout")),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding:    0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty:                 wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size:   None,
                },
                count: None,
            }],
        });

        let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label:   Some(&format!("{label}_bind_group")),
            layout:  &layout,
            entries: &[wgpu::BindGroupEntry {
                binding:  0,
                resource: buf.as_entire_binding(),
            }],
        });

        Self { buf, layout, group, _pd: PhantomData }
    }

    pub fn write(&self, queue: &wgpu::Queue, value: &T) {
        queue.write_buffer(&self.buf, 0, bytemuck::bytes_of(value));
    }
}

// ── Storage buffer ─────────────────────────────────────────────────────────────

/// A GPU storage buffer — supports read/write from compute shaders and, when
/// `COPY_SRC` is set, CPU readback.
pub struct StorageBuffer<T: Pod> {
    pub buf:        wgpu::Buffer,
    pub len:        usize,
    pub read_back:  bool,
    _pd: PhantomData<T>,
}

impl<T: Pod> StorageBuffer<T> {
    pub fn zeroed(device: &wgpu::Device, label: &str, len: usize, read_back: bool) -> Self {
        let size = (std::mem::size_of::<T>() * len) as wgpu::BufferAddress;
        let mut usage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST;
        if read_back { usage |= wgpu::BufferUsages::COPY_SRC; }
        let buf = device.create_buffer(&wgpu::BufferDescriptor {
            label:              Some(label),
            size,
            usage,
            mapped_at_creation: false,
        });
        Self { buf, len, read_back, _pd: PhantomData }
    }

    pub fn from_slice(device: &wgpu::Device, label: &str, data: &[T], read_back: bool) -> Self {
        use wgpu::util::DeviceExt;
        let mut usage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST;
        if read_back { usage |= wgpu::BufferUsages::COPY_SRC; }
        let buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(label),
            contents: bytemuck::cast_slice(data),
            usage,
        });
        Self { buf, len: data.len(), read_back, _pd: PhantomData }
    }

    pub fn binding_resource(&self) -> wgpu::BindingResource<'_> {
        self.buf.as_entire_binding()
    }

    /// Size in bytes.
    pub fn byte_size(&self) -> u64 {
        (std::mem::size_of::<T>() * self.len) as u64
    }
}
