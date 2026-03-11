//! GPU-driven preprocessing pass: light culling, sorting, and indirect buffer building
//!
//! Runs a compute shader that:
//! 1. Culls lights by frustum + distance from camera
//! 2. Sorts visible lights by screen-space impact
//! 3. Builds indirect draw buffers for opaque/transparent passes
//! 4. Reorders light data for cache coherence
//!
//! This is the Unreal Engine approach: move CPU-side light sorting onto GPU
//! compute, feeding results back to indirect rendering passes.

use bytemuck::{Pod, Zeroable};
use std::sync::Arc;
use crate::Result;

/// Per-light data written by compute shader (sorted result)
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct GpuPreprocessLight {
    pub position: [f32; 3],
    pub range: f32,
    pub direction: [f32; 3],
    pub light_type: u32,  // 0=directional, 1=point, 2=spot
    pub intensity: f32,
    pub cos_inner: f32,
    pub cos_outer: f32,
    pub _pad: u32,
    pub color: [f32; 3],
    pub _pad2: u32,
}

/// Input to the preprocessing compute: scene state this frame
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct PreprocessInput {
    pub camera_pos: [f32; 3],
    pub _pad0: u32,
    pub camera_forward: [f32; 3],
    pub _pad1: u32,
    pub camera_right: [f32; 3],
    pub _pad2: u32,
    pub camera_up: [f32; 3],
    pub camera_fov_y: f32,
    pub total_lights: u32,
    pub viewport_width: u32,
    pub viewport_height: u32,
    pub frame: u32,
    /// Six frustum planes extracted from view_proj (Gribb-Hartmann).
    /// Each plane is (nx, ny, nz, d) where the inside satisfies dot(n,p)+d >= 0.
    /// Planes are NOT normalized; sphere test divides by length(n).
    pub frustum_planes: [[f32; 4]; 6],
}

/// Output from preprocessing compute
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct PreprocessOutput {
    pub visible_light_count: u32,
    pub opaque_draw_count: u32,
    pub transparent_draw_count: u32,
    pub _pad: u32,
}

/// GPU preprocessing feature: compute-side light culling and sorting.
/// TODO: Wire into the frame graph — pipeline/bind_group/enabled are placeholders
/// for GPU skinning, morphing, and indirect draw list generation.
#[allow(dead_code)]
pub struct GpuPreprocessingFeature {
    enabled: bool,
    pipeline: Option<Arc<wgpu::ComputePipeline>>,
    
    // Input/output buffers
    input_buffer: Option<wgpu::Buffer>,
    output_buffer: Option<wgpu::Buffer>,
    visible_lights_buffer: Option<Arc<wgpu::Buffer>>,  // Reordered by compute
    indirect_opaque_buffer: Option<Arc<wgpu::Buffer>>,   // Indirect commands for opaque
    indirect_transparent_buffer: Option<Arc<wgpu::Buffer>>,  // Indirect commands for transparent
    
    bind_group: Option<wgpu::BindGroup>,
    max_lights: u32,
}

impl GpuPreprocessingFeature {
    pub fn new() -> Self {
        Self {
            enabled: true,
            pipeline: None,
            input_buffer: None,
            output_buffer: None,
            visible_lights_buffer: None,
            indirect_opaque_buffer: None,
            indirect_transparent_buffer: None,
            bind_group: None,
            max_lights: 128,
        }
    }

    pub fn with_max_lights(mut self, count: u32) -> Self {
        self.max_lights = count;
        self
    }

    /// Initialize GPU resources (call during renderer setup)
    pub fn initialize(&mut self, device: &wgpu::Device) -> Result<()> {
        // Input uniform buffer
        self.input_buffer = Some(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("GPU Preprocess Input"),
            size: std::mem::size_of::<PreprocessInput>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));

        // Output counter buffer (written by compute)
        self.output_buffer = Some(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("GPU Preprocess Output"),
            size: std::mem::size_of::<PreprocessOutput>() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));

        // Visible lights buffer (output of compute; read by opaque/transparent)
        self.visible_lights_buffer = Some(Arc::new(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Visible Lights Buffer"),
            size: (self.max_lights as u64) * std::mem::size_of::<GpuPreprocessLight>() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        })));

        // Indirect draw buffers (built by compute for opaque/transparent passes)
        // Each indirect draw is wgpu::util::DrawIndexedIndirectArgs (20 bytes)
        // Max draws: assume 1000 per pass (rough upper bound)
        let max_indirect_draws = 1000u64;
        self.indirect_opaque_buffer = Some(Arc::new(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Indirect Opaque Draws"),
            size: max_indirect_draws * 20,  // DrawIndexedIndirectArgs = 5×u32
            usage: wgpu::BufferUsages::INDIRECT | wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })));

        self.indirect_transparent_buffer = Some(Arc::new(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Indirect Transparent Draws"),
            size: max_indirect_draws * 20,
            usage: wgpu::BufferUsages::INDIRECT | wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })));

        Ok(())
    }

    pub fn visible_lights_buffer(&self) -> Option<Arc<wgpu::Buffer>> {
        self.visible_lights_buffer.clone()
    }

    pub fn indirect_opaque_buffer(&self) -> Option<Arc<wgpu::Buffer>> {
        self.indirect_opaque_buffer.clone()
    }

    pub fn indirect_transparent_buffer(&self) -> Option<Arc<wgpu::Buffer>> {
        self.indirect_transparent_buffer.clone()
    }

    /// Upload per-frame camera + frustum data to the `input_buffer`.
    ///
    /// Call this once per frame BEFORE dispatching the preprocessing compute.
    /// `frustum_planes` comes from `camera.frustum().as_raw()`.
    pub fn prepare_input(
        &self,
        queue:          &wgpu::Queue,
        camera_pos:     [f32; 3],
        camera_forward: [f32; 3],
        camera_right:   [f32; 3],
        camera_up:      [f32; 3],
        camera_fov_y:   f32,
        total_lights:   u32,
        viewport_width: u32,
        viewport_height: u32,
        frame:          u32,
        frustum_planes: [[f32; 4]; 6],
    ) {
        let Some(buf) = &self.input_buffer else { return; };
        let input = PreprocessInput {
            camera_pos,
            _pad0: 0,
            camera_forward,
            _pad1: 0,
            camera_right,
            _pad2: 0,
            camera_up,
            camera_fov_y,
            total_lights,
            viewport_width,
            viewport_height,
            frame,
            frustum_planes,
        };
        queue.write_buffer(buf, 0, bytemuck::bytes_of(&input));
    }
}
