use blade_graphics as gpu;
use bytemuck::Pod;

pub struct Buffer<T> {
    pub gpu_buffer: gpu::Buffer,
    pub size: u64,
    pub count: usize,
    _phantom: std::marker::PhantomData<T>,
}

impl<T: Pod> Buffer<T> {
    pub fn new(context: &gpu::Context, count: usize, name: &str) -> Self {
        let size = (count * std::mem::size_of::<T>()) as u64;
        let gpu_buffer = context.create_buffer(gpu::BufferDesc {
            name,
            size,
            memory: gpu::Memory::Device,
        });
        
        Self {
            gpu_buffer,
            size,
            count,
            _phantom: std::marker::PhantomData,
        }
    }
    
    pub fn element_count(&self) -> usize {
        self.count
    }
}

pub struct UniformBuffer<T> {
    buffer: Buffer<T>,
}

impl<T: Pod> UniformBuffer<T> {
    pub fn new(context: &gpu::Context, name: &str) -> Self {
        Self {
            buffer: Buffer::new(context, 1, name),
        }
    }
}

pub struct StorageBuffer<T> {
    buffer: Buffer<T>,
}

impl<T: Pod> StorageBuffer<T> {
    pub fn new(context: &gpu::Context, count: usize, name: &str) -> Self {
        Self {
            buffer: Buffer::new(context, count, name),
        }
    }
}
