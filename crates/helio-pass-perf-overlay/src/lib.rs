//! GPU-based performance overlay pass for Helio renderer.
//!
//! Provides real-time visualization of rendering performance metrics:
//! - **Pass-to-pass overdraw**: Tracks when passes overwrite each other's pixels
//! - **Shader complexity**: Heatmap based on GBuffer ORM values
//! - **Tile light count**: Forward+ culling visualization
//! - **Pass output inspection**: Debug viewer for render targets
//!
//! Zero cost when disabled. Works universally with all passes without shader modifications.

use std::sync::{Arc, Mutex};
use bytemuck::{Pod, Zeroable};
use helio_v3::{PassContext, PrepareContext, RenderPass, Result as HelioResult};

pub const TILE_SIZE: u32 = 16;

// ─────────────────────────────────────────────────────────────────────────────
// Visualization Modes
// ─────────────────────────────────────────────────────────────────────────────

/// Performance overlay visualization modes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(u32)]
pub enum PerfOverlayMode {
    /// Disabled: zero GPU cost.
    #[default]
    Disabled = 0,
    /// Pass-to-pass overdraw tracking (warm = high pass overlap).
    PassOverdraw = 1,
    /// Shader complexity based on GBuffer ORM heuristic.
    ShaderComplexity = 2,
    /// Tile light count from forward+ culling (warm = many lights).
    TileLightCount = 3,
    /// Pass output inspector (debug viewer for render targets).
    PassOutput = 4,
}

// ─────────────────────────────────────────────────────────────────────────────
// GPU-side uniforms
// ─────────────────────────────────────────────────────────────────────────────

/// Depth comparison compute shader parameters.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ColorCompareParams {
    screen_width: u32,
    screen_height: u32,
    _pad0: u32,
    _pad1: u32,
}

/// Shader cost computation parameters.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ComputeCostParams {
    screen_width: u32,
    screen_height: u32,
    num_tiles_x: u32,
    num_timing_entries: u32,
}

/// Tile aggregation compute shader parameters.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct AggregateParams {
    num_tiles_x: u32,
    num_tiles_y: u32,
    num_tiles: u32,
    screen_width: u32,
    screen_height: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

/// Tile metrics (aggregated per 16×16 tile).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct TileMetrics {
    pass_overdraw_max: u32,  // Max pass overwrites in tile
    light_count: u32,        // From LightCullPass
    complexity_avg: u32,     // GBuffer ORM heuristic
    _pad: u32,
}

/// Visualization shader parameters.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct VisualizeParams {
    mode: u32,              // PerfOverlayMode as u32
    num_tiles_x: u32,
    num_tiles_y: u32,
    internal_width: u32,    // Buffer dimensions (internal resolution)
    internal_height: u32,
    display_width: u32,     // Target dimensions (display resolution)
    display_height: u32,
    heatmap_scale: f32,     // Max value for normalization
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

/// Material profiling shader parameters.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct MaterialProfileParams {
    roughness: f32,
    metallic: f32,
    num_lights: u32,
    num_shadow_lights: u32,
}

/// Material timing lookup table entry.
///
/// Stores measured GPU execution time for a specific material configuration.
#[derive(Clone, Copy, Debug)]
struct MaterialTimingEntry {
    roughness: f32,
    metallic: f32,
    num_lights: u32,
    gpu_time_ns: u64,
}

struct PerfOverlayRuntime {
    frame_num: u64,
    snapshot_valid: bool,
}

pub struct PerfOverlayShared {
    // Internal (render) resolution - buffers are sized to this
    internal_width: u32,
    internal_height: u32,
    // Display (output) resolution - rendering target size
    display_width: u32,
    display_height: u32,
    num_tiles_x: u32,
    num_tiles_y: u32,

    color_snapshot_prev: wgpu::Texture,
    color_snapshot_prev_view: wgpu::TextureView,
    pass_overdraw_buf: wgpu::Buffer,
    shader_cost_buf: wgpu::Buffer,
    material_timing_buf: wgpu::Buffer,

    color_compare_pipeline: wgpu::ComputePipeline,
    color_compare_bgl: wgpu::BindGroupLayout,
    color_compare_params_buf: wgpu::Buffer,

    blit_pipeline: wgpu::ComputePipeline,
    blit_bgl: wgpu::BindGroupLayout,

    cost_compute_pipeline: wgpu::ComputePipeline,
    cost_compute_bgl: wgpu::BindGroupLayout,
    cost_compute_params_buf: wgpu::Buffer,

    material_profiler: Option<MaterialProfiler>,

    mode: Mutex<PerfOverlayMode>,
    runtime: Mutex<PerfOverlayRuntime>,
}

// ─────────────────────────────────────────────────────────────────────────────
// Pass struct
// ─────────────────────────────────────────────────────────────────────────────

pub struct PerfOverlayPass {
    shared: Arc<Mutex<PerfOverlayShared>>,
    aggregate_pipeline: wgpu::ComputePipeline,
    aggregate_bgl: wgpu::BindGroupLayout,
    aggregate_params_buf: wgpu::Buffer,
    aggregate_bind_group: Option<wgpu::BindGroup>,
    tile_metrics_buf: wgpu::Buffer,
    visualize_pipeline: wgpu::RenderPipeline,
    visualize_bgl: wgpu::BindGroupLayout,
    visualize_params_buf: wgpu::Buffer,
    visualize_bind_group: Option<wgpu::BindGroup>,
    bind_group_key: Option<(usize, usize, usize)>,
}

/// Analyzer pass that runs after render passes and compares the current depth
/// buffer against the previous snapshot to count pass overwrites.
pub struct PerfOverlayAnalyzerPass {
    shared: Arc<Mutex<PerfOverlayShared>>,
}

/// Cost analyzer pass that computes per-pixel shader cost based on
/// light counts, material properties, and shadow complexity.
pub struct PerfOverlayCostAnalyzerPass {
    shared: Arc<Mutex<PerfOverlayShared>>,
}

/// Material profiler for measuring actual GPU execution time.
///
/// Samples different material configurations by rendering test patches
/// and measuring GPU time with timestamp queries.
/// Number of samples to take per material configuration for averaging
const SAMPLES_PER_CONFIG: usize = 3;

pub struct MaterialProfiler {
    profile_pipeline: wgpu::RenderPipeline,
    profile_bgl: wgpu::BindGroupLayout,
    profile_params_bufs: Vec<wgpu::Buffer>, // One buffer per configuration
    test_texture: wgpu::Texture,
    test_texture_view: wgpu::TextureView,
    query_set: wgpu::QuerySet,
    query_buffer: wgpu::Buffer,
    resolve_buffer: wgpu::Buffer,
    timing_samples: Vec<Vec<u64>>, // Accumulated timing samples per config (in GPU ticks)
    timing_table: Vec<MaterialTimingEntry>,
    profiling_complete: bool,
    timings_uploaded: bool,
    current_config_index: usize,
    current_sample_index: usize,
    timestamp_period: f32,
}

impl MaterialProfiler {
    /// Create a new material profiler.
    ///
    /// Initializes GPU resources for profiling and defines material configurations to test.
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue) -> Self {
        // Create small test texture (32×32 pixels)
        let test_size = 32;
        let test_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Material Profile Test Texture"),
            size: wgpu::Extent3d {
                width: test_size,
                height: test_size,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        
        let test_texture_view = test_texture.create_view(&wgpu::TextureViewDescriptor::default());
        
        // Create timestamp query resources
        // We only need 2 queries at a time since we process one sample per frame
        let query_set = device.create_query_set(&wgpu::QuerySetDescriptor {
            label: Some("Material Profile QuerySet"),
            ty: wgpu::QueryType::Timestamp,
            count: 2, // Start/end timestamps for current sample
        });
        
        let query_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Material Profile Query Buffer"),
            size: 2 * 8,
            usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        
        let resolve_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Material Profile Resolve Buffer"),
            size: 2 * 8,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        
        // Create profiling shader and pipeline
        let profile_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Material Profile Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/profile_material.wgsl").into()),
        });
        
        let profile_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Material Profile BGL"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        
        let profile_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Material Profile PL"),
            bind_group_layouts: &[Some(&profile_bgl)],
            immediate_size: 0,
        });
        
        let profile_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Material Profile Pipeline"),
            layout: Some(&profile_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &profile_shader,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &profile_shader,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::Rgba16Float,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });
        
        // Pre-create parameter buffers for all configurations
        let configs = Self::get_material_configs();
        let mut profile_params_bufs = Vec::with_capacity(configs.len());
        for (roughness, metallic, num_lights) in &configs {
            let params = MaterialProfileParams {
                roughness: *roughness,
                metallic: *metallic,
                num_lights: *num_lights,
                num_shadow_lights: 0,
            };
            let buf = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Material Profile Params"),
                size: std::mem::size_of::<MaterialProfileParams>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: true,
            });
            {
                let mut view = buf.slice(..).get_mapped_range_mut();
                view.copy_from_slice(bytemuck::bytes_of(&params));
            }
            buf.unmap();
            profile_params_bufs.push(buf);
        }
        
        let timestamp_period = queue.get_timestamp_period();
        
        // Initialize timing samples storage - one vec per config
        let timing_samples = vec![Vec::with_capacity(SAMPLES_PER_CONFIG); configs.len()];
        
        Self {
            profile_pipeline,
            profile_bgl,
            profile_params_bufs,
            test_texture,
            test_texture_view,
            query_set,
            query_buffer,
            resolve_buffer,
            timing_samples,
            timing_table: Vec::new(),
            profiling_complete: false,
            timings_uploaded: false,
            current_config_index: 0,
            current_sample_index: 0,
            timestamp_period,
        }
    }
    
    /// Returns true if all material configurations have been profiled.
    pub fn is_complete(&self) -> bool {
        self.profiling_complete
    }
    
    /// Get material configurations to profile.
    ///
    /// Returns a grid of (roughness, metallic, num_lights) combinations.
    fn get_material_configs() -> Vec<(f32, f32, u32)> {
        let mut configs = Vec::new();
        
        // Sample roughness: 0.1 (smooth), 0.4, 0.7, 1.0 (rough)
        // Sample metallic: 0.0 (dielectric), 0.5, 1.0 (metal)
        // Sample light counts: 4, 16, 32
        for &roughness in &[0.1, 0.4, 0.7, 1.0] {
            for &metallic in &[0.0, 0.5, 1.0] {
                for &num_lights in &[4, 16, 32] {
                    configs.push((roughness, metallic, num_lights));
                }
            }
        }
        
        configs
    }
    
    /// Profile the next material configuration.
    ///
    /// Renders a test patch and measures GPU time. Call this from a render pass
    /// until is_complete() returns true.
    pub fn profile_next(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        light_buf: &wgpu::Buffer,
    ) -> bool {
        if self.profiling_complete {
            return true;
        }
        
        let configs = Self::get_material_configs();
        if self.current_config_index >= configs.len() {
            self.profiling_complete = true;
            return true;
        }
        
        // Use pre-created parameter buffer for this configuration
        let params_buf = &self.profile_params_bufs[self.current_config_index];
        
        // Create bind group
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Material Profile BG"),
            layout: &self.profile_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: light_buf.as_entire_binding(),
                },
            ],
        });
        
        // Write start timestamp (always use query 0 and 1 for current sample)
        encoder.write_timestamp(&self.query_set, 0);
        
        // Render test patch
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Material Profile Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.test_texture_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            
            pass.set_pipeline(&self.profile_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.draw(0..3, 0..1); // Fullscreen triangle
        }
        
        // Write end timestamp
        encoder.write_timestamp(&self.query_set, 1);
        
        // Resolve queries to buffer
        encoder.resolve_query_set(&self.query_set, 0..2, &self.query_buffer, 0);
        
        // Copy to resolve buffer for readback
        encoder.copy_buffer_to_buffer(&self.query_buffer, 0, &self.resolve_buffer, 0, 2 * 8);
        
        // Advance to next sample or config
        self.current_sample_index += 1;
        if self.current_sample_index >= SAMPLES_PER_CONFIG {
            self.current_sample_index = 0;
            self.current_config_index += 1;
        }
        
        false // More configs/samples to profile
    }
    
    /// Read back current sample timing from GPU and accumulate it.
    ///
    /// This blocks until GPU results are available. Call after each profile_next().
    pub fn read_current_sample_blocking(&mut self, device: &wgpu::Device) {
        // Map buffer and read timestamps
        let buffer_slice = self.resolve_buffer.slice(..);
        buffer_slice.map_async(wgpu::MapMode::Read, |_| {});
        device.poll(wgpu::PollType::wait_indefinitely());
        
        let data = buffer_slice.get_mapped_range();
        let timestamps: &[u64] = bytemuck::cast_slice(&data);
        
        // Store the timing sample (in GPU ticks, not nanoseconds yet)
        let start_ts = timestamps[0];
        let end_ts = timestamps[1];
        let gpu_ticks = end_ts.saturating_sub(start_ts);
        
        // Track which config we're sampling (accounting for the fact we advanced indices already)
        let config_idx = if self.current_sample_index == 0 {
            self.current_config_index.saturating_sub(1)
        } else {
            self.current_config_index
        };
        
        if config_idx < self.timing_samples.len() {
            self.timing_samples[config_idx].push(gpu_ticks);
        }
        
        drop(data);
        self.resolve_buffer.unmap();
    }
    
    /// Compute final averaged timing table from all samples.
    ///
    /// Call this once after all profiling is complete.
    pub fn compute_final_timings(&mut self) {
        let configs = Self::get_material_configs();
        self.timing_table.clear();
        
        for (i, (roughness, metallic, num_lights)) in configs.iter().enumerate() {
            let samples = &self.timing_samples[i];
            if samples.is_empty() {
                continue;
            }
            
            // Average the samples
            let avg_ticks = samples.iter().sum::<u64>() / samples.len() as u64;
            let gpu_time_ns = (avg_ticks as f32 * self.timestamp_period) as u64;
            
            self.timing_table.push(MaterialTimingEntry {
                roughness: *roughness,
                metallic: *metallic,
                num_lights: *num_lights,
                gpu_time_ns,
            });
        }
    }
    
    /// Get the completed timing table.
    pub fn get_timing_table(&self) -> &[MaterialTimingEntry] {
        &self.timing_table
    }
}

impl PerfOverlayShared {
    pub fn new(device: &wgpu::Device, width: u32, height: u32) -> Arc<Mutex<Self>> {
        let num_tiles_x = width.div_ceil(TILE_SIZE);
        let num_tiles_y = height.div_ceil(TILE_SIZE);

        let color_snapshot_prev = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("PerfOverlay Color Snapshot"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::STORAGE_BINDING,
            view_formats: &[],
        });
        let color_snapshot_prev_view =
            color_snapshot_prev.create_view(&wgpu::TextureViewDescriptor::default());

        let pixel_count = (width as u64) * (height as u64);
        let pass_overdraw_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("PerfOverlay Pass Overdraw Counters"),
            size: pixel_count * 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let color_compare_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("PerfOverlay Color Compare Shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../shaders/analyze_color_overdraw.wgsl").into(),
            ),
        });

        let color_compare_bgl =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("PerfOverlay Color Compare BGL"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: false },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: false },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
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
                ],
            });

        let color_compare_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("PerfOverlay Color Compare PL"),
                bind_group_layouts: &[Some(&color_compare_bgl)],
                immediate_size: 0,
            });

        let color_compare_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("PerfOverlay Color Compare Pipeline"),
                layout: Some(&color_compare_pipeline_layout),
                module: &color_compare_shader,
                entry_point: Some("analyze_color_overdraw"),
                compilation_options: Default::default(),
                cache: None,
            });

        let color_compare_params_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("PerfOverlay Color Compare Params"),
            size: std::mem::size_of::<ColorCompareParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Blit shader for copying color textures
        let blit_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("PerfOverlay Blit Shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../shaders/blit_color.wgsl").into(),
            ),
        });

        let blit_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("PerfOverlay Blit BGL"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rgba16Float,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
            ],
        });

        let blit_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("PerfOverlay Blit PL"),
            bind_group_layouts: &[Some(&blit_bgl)],
            immediate_size: 0,
        });

        let blit_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("PerfOverlay Blit Pipeline"),
            layout: Some(&blit_pipeline_layout),
            module: &blit_shader,
            entry_point: Some("blit_color"),
            compilation_options: Default::default(),
            cache: None,
        });

        // Shader cost computation buffer
        let shader_cost_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("PerfOverlay Shader Cost"),
            size: pixel_count * 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Material timing data buffer (16 bytes per entry: roughness, metallic, num_lights, gpu_time_ns)
        let material_timing_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("PerfOverlay Material Timing Data"),
            size: 128 * 16, // 128 entries max, 16 bytes each
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Shader cost compute shader
        let cost_compute_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("PerfOverlay Cost Compute Shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../shaders/compute_shader_cost.wgsl").into(),
            ),
        });

        let cost_compute_bgl =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("PerfOverlay Cost Compute BGL"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: false },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: false },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 4,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 5,
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

        let cost_compute_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("PerfOverlay Cost Compute PL"),
                bind_group_layouts: &[Some(&cost_compute_bgl)],
                immediate_size: 0,
            });

        let cost_compute_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("PerfOverlay Cost Compute Pipeline"),
                layout: Some(&cost_compute_pipeline_layout),
                module: &cost_compute_shader,
                entry_point: Some("cs_compute_shader_cost"),
                compilation_options: Default::default(),
                cache: None,
            });

        let cost_compute_params_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("PerfOverlay Cost Compute Params"),
            size: std::mem::size_of::<ComputeCostParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Arc::new(Mutex::new(Self {
            internal_width: width,
            internal_height: height,
            display_width: width, // Initially same as internal
            display_height: height,
            num_tiles_x,
            num_tiles_y,
            color_snapshot_prev,
            color_snapshot_prev_view,
            pass_overdraw_buf,
            shader_cost_buf,
            material_timing_buf,
            color_compare_pipeline,
            color_compare_bgl,
            color_compare_params_buf,
            blit_pipeline,
            blit_bgl,
            cost_compute_pipeline,
            cost_compute_bgl,
            cost_compute_params_buf,
            material_profiler: None, // Initialized on first use
            mode: Mutex::new(PerfOverlayMode::Disabled),
            runtime: Mutex::new(PerfOverlayRuntime {
                frame_num: 0,
                snapshot_valid: false,
            }),
        }))
    }

    pub fn on_resize(&mut self, _device: &wgpu::Device, width: u32, height: u32) {
        // Update display resolution (rendering target size)
        // Buffers remain at internal resolution to match pre_aa buffer
        if width == self.display_width && height == self.display_height {
            return;
        }

        self.display_width = width;
        self.display_height = height;
        // Note: internal buffers are NOT resized here - they stay at internal resolution
        // to match the pre_aa color buffer they're tracking
    }

    pub fn get_mode(&self) -> PerfOverlayMode {
        *self.mode.lock().unwrap()
    }

    pub fn set_mode(&self, mode: PerfOverlayMode) {
        *self.mode.lock().unwrap() = mode;
    }

    /// Set overlay opacity (no-op: overlay now outputs raw data).
    pub fn set_opacity(&self, _opacity: f32) {
        // No-op: opacity removed, kept for API compatibility
    }

    /// Initialize material profiler for measuring actual GPU execution times.
    pub fn init_material_profiler(&mut self, device: &wgpu::Device, queue: &wgpu::Queue) {
        if self.material_profiler.is_none() {
            self.material_profiler = Some(MaterialProfiler::new(device, queue));
        }
    }

    /// Check if material profiling is complete.
    pub fn is_profiling_complete(&self) -> bool {
        self.material_profiler
            .as_ref()
            .map(|p| p.is_complete())
            .unwrap_or(false)
    }

    /// Get material timing table (returns empty slice if not complete).
    pub fn get_material_timings(&self) -> &[MaterialTimingEntry] {
        self.material_profiler
            .as_ref()
            .map(|p| p.get_timing_table())
            .unwrap_or(&[])
    }
}

impl PerfOverlayPass {
    pub fn new(
        device: &wgpu::Device,
        shared: Arc<Mutex<PerfOverlayShared>>,
        target_format: wgpu::TextureFormat,
    ) -> Self {
        let shared_guard = shared.lock().unwrap();
        let num_tiles = shared_guard
            .num_tiles_x
            .checked_mul(shared_guard.num_tiles_y)
            .expect("tile grid overflow: viewport dimensions too large");
        drop(shared_guard);

        let aggregate_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("PerfOverlay Aggregate Shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../shaders/aggregate_tiles.wgsl").into(),
            ),
        });

        let aggregate_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("PerfOverlay Aggregate BGL"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let aggregate_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("PerfOverlay Aggregate PL"),
                bind_group_layouts: &[Some(&aggregate_bgl)],
                immediate_size: 0,
            });

        let aggregate_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("PerfOverlay Aggregate Pipeline"),
            layout: Some(&aggregate_pipeline_layout),
            module: &aggregate_shader,
            entry_point: Some("cs_aggregate_tiles"),
            compilation_options: Default::default(),
            cache: None,
        });

        let aggregate_params_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("PerfOverlay Aggregate Params"),
            size: std::mem::size_of::<AggregateParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let tile_metrics_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("PerfOverlay Tile Metrics"),
            size: (num_tiles as u64 * std::mem::size_of::<TileMetrics>() as u64).max(4),
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        let visualize_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("PerfOverlay Visualize Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/visualize.wgsl").into()),
        });

        let visualize_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("PerfOverlay Visualize BGL"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let visualize_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("PerfOverlay Visualize PL"),
                bind_group_layouts: &[Some(&visualize_bgl)],
                immediate_size: 0,
            });

        let visualize_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("PerfOverlay Visualize Pipeline"),
            layout: Some(&visualize_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &visualize_shader,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &visualize_shader,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: target_format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,  
        });

        let visualize_params_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("PerfOverlay Visualize Params"),
            size: std::mem::size_of::<VisualizeParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            shared,
            aggregate_pipeline,
            aggregate_bgl,
            aggregate_params_buf,
            aggregate_bind_group: None,
            tile_metrics_buf,
            visualize_pipeline,
            visualize_bgl,
            visualize_params_buf,
            visualize_bind_group: None,
            bind_group_key: None,
        }
    }

    pub fn set_mode(&mut self, mode: PerfOverlayMode) {
        self.shared.lock().unwrap().set_mode(mode);
    }

    /// Set overlay opacity (no-op: overlay now outputs raw data).
    pub fn set_opacity(&mut self, _opacity: f32) {
        // No-op: opacity removed, kept for API compatibility
    }
}

impl PerfOverlayAnalyzerPass {
    pub fn new(shared: Arc<Mutex<PerfOverlayShared>>) -> Self {
        Self { shared }
    }
}

impl RenderPass for PerfOverlayPass {
    fn name(&self) -> &'static str {
        "PerfOverlay"
    }

    fn prepare(&mut self, ctx: &PrepareContext) -> HelioResult<()> {
        let shared = self.shared.lock().unwrap();
        let aggregate_params = AggregateParams {
            num_tiles_x: shared.num_tiles_x,
            num_tiles_y: shared.num_tiles_y,
            num_tiles: shared.num_tiles_x * shared.num_tiles_y,
            screen_width: shared.internal_width,
            screen_height: shared.internal_height,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        };
        ctx.write_buffer(
            &self.aggregate_params_buf,
            0,
            bytemuck::bytes_of(&aggregate_params),
        );

        let visualize_params = VisualizeParams {
            mode: shared.mode.lock().unwrap().clone() as u32,
            num_tiles_x: shared.num_tiles_x,
            num_tiles_y: shared.num_tiles_y,
            internal_width: shared.internal_width,
            internal_height: shared.internal_height,
            display_width: shared.display_width,
            display_height: shared.display_height,
            heatmap_scale: 5.0,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        };
        ctx.write_buffer(
            &self.visualize_params_buf,
            0,
            bytemuck::bytes_of(&visualize_params),
        );

        Ok(())
    }

    fn execute(&mut self, ctx: &mut PassContext) -> HelioResult<()> {
        let shared = self.shared.lock().unwrap();
        if *shared.mode.lock().unwrap() == PerfOverlayMode::Disabled {
            return Ok(());
        }

        if let (Some(gbuffer), Some(tile_light_counts)) = (ctx.resources.gbuffer, ctx.resources.tile_light_counts) {
            let gbuffer_orm_ptr = gbuffer.orm as *const _ as usize;
            let tile_light_counts_ptr = tile_light_counts as *const _ as usize;
            let key = (gbuffer_orm_ptr, tile_light_counts_ptr, 0);

            if self.bind_group_key != Some(key) {
                self.aggregate_bind_group = Some(ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("PerfOverlay Aggregate BG"),
                    layout: &self.aggregate_bgl,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: self.aggregate_params_buf.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: shared.pass_overdraw_buf.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: wgpu::BindingResource::TextureView(gbuffer.orm),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: tile_light_counts.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 4,
                            resource: self.tile_metrics_buf.as_entire_binding(),
                        },
                    ],
                }));
                self.bind_group_key = Some(key);
            }

            let mut pass = ctx.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("PerfOverlay Aggregate"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.aggregate_pipeline);
            pass.set_bind_group(0, self.aggregate_bind_group.as_ref().unwrap(), &[]);
            let num_tiles = shared.num_tiles_x * shared.num_tiles_y;
            pass.dispatch_workgroups(num_tiles.div_ceil(256), 1, 1);
        }

        if let (Some(_pre_aa), Some(gbuffer)) = (ctx.resources.pre_aa, ctx.resources.gbuffer) {
            let gbuffer_orm_ptr = gbuffer.orm as *const _ as usize;
            let key = (gbuffer_orm_ptr, 0, 0);

            if self.bind_group_key != Some(key) || self.visualize_bind_group.is_none() {
                self.visualize_bind_group = Some(ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("PerfOverlay Visualize BG"),
                    layout: &self.visualize_bgl,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: self.visualize_params_buf.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: shared.pass_overdraw_buf.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: self.tile_metrics_buf.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: wgpu::BindingResource::TextureView(gbuffer.orm),
                        },
                        wgpu::BindGroupEntry {
                            binding: 4,
                            resource: shared.shader_cost_buf.as_entire_binding(),
                        },
                    ],
                }));
            }

            let color_attachments = [Some(wgpu::RenderPassColorAttachment {
                view: ctx.target,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
                depth_slice: None,
            })];
            let render_pass_desc = wgpu::RenderPassDescriptor {
                label: Some("PerfOverlay Visualize"),
                color_attachments: &color_attachments,
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            };

            let mut pass = ctx.begin_render_pass(&render_pass_desc);
            pass.set_pipeline(&self.visualize_pipeline);
            pass.set_bind_group(0, self.visualize_bind_group.as_ref().unwrap(), &[]);
            pass.draw(0..3, 0..1);
        }

        Ok(())
    }

    fn on_resize(&mut self, device: &wgpu::Device, width: u32, height: u32) {
        let mut shared = self.shared.lock().unwrap();
        shared.on_resize(device, width, height);

        let num_tiles = shared.num_tiles_x * shared.num_tiles_y;
        self.tile_metrics_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("PerfOverlay Tile Metrics"),
            size: (num_tiles as u64 * std::mem::size_of::<TileMetrics>() as u64).max(4),
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        self.aggregate_bind_group = None;
        self.visualize_bind_group = None;
        self.bind_group_key = None;
    }
}

impl RenderPass for PerfOverlayAnalyzerPass {
    fn name(&self) -> &'static str {
        "PerfOverlay Color Analyzer"
    }

    fn prepare(&mut self, ctx: &PrepareContext) -> HelioResult<()> {
        let shared = self.shared.lock().unwrap();
        let color_compare_params = ColorCompareParams {
            screen_width: shared.internal_width,
            screen_height: shared.internal_height,
            _pad0: 0,
            _pad1: 0,
        };
        ctx.write_buffer(
            &shared.color_compare_params_buf,
            0,
            bytemuck::bytes_of(&color_compare_params),
        );
        Ok(())
    }

    fn execute(&mut self, ctx: &mut PassContext) -> HelioResult<()> {
        let shared = self.shared.lock().unwrap();
        if *shared.mode.lock().unwrap() != PerfOverlayMode::PassOverdraw {
            return Ok(());
        }

        // Get the color render target (pre-AA buffer)
        let color_texture = if let Some(pre_aa) = ctx.resources.pre_aa {
            pre_aa
        } else {
            // If no pre_aa, use the main target (though this is less ideal)
            ctx.target
        };

        if shared.runtime.lock().unwrap().frame_num != ctx.frame_num {
            ctx.encoder.clear_buffer(&shared.pass_overdraw_buf, 0, None);
            let mut runtime = shared.runtime.lock().unwrap();
            runtime.frame_num = ctx.frame_num;
            runtime.snapshot_valid = false;
        }

        let mut runtime = shared.runtime.lock().unwrap();
        if runtime.snapshot_valid {
            let color_compare_bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("PerfOverlay Color Compare BG"),
                layout: &shared.color_compare_bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: shared.color_compare_params_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&shared.color_snapshot_prev_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::TextureView(color_texture),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: shared.pass_overdraw_buf.as_entire_binding(),
                    },
                ],
            });

            let mut pass = ctx.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("PerfOverlay Color Compare"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&shared.color_compare_pipeline);
            pass.set_bind_group(0, &color_compare_bg, &[]);
            let dispatch_x = shared.internal_width.div_ceil(16);
            let dispatch_y = shared.internal_height.div_ceil(16);
            pass.dispatch_workgroups(dispatch_x, dispatch_y, 1);
        } else {
            runtime.snapshot_valid = true;
        }
        drop(runtime);

        // Copy current color to snapshot for next pass comparison using blit shader
        let blit_bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("PerfOverlay Blit BG"),
            layout: &shared.blit_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(color_texture),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&shared.color_snapshot_prev_view),
                },
            ],
        });

        let mut pass = ctx.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("PerfOverlay Blit Color"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&shared.blit_pipeline);
        pass.set_bind_group(0, &blit_bg, &[]);
        let dispatch_x = shared.internal_width.div_ceil(16);
        let dispatch_y = shared.internal_height.div_ceil(16);
        pass.dispatch_workgroups(dispatch_x, dispatch_y, 1);

        Ok(())
    }

    fn on_resize(&mut self, device: &wgpu::Device, width: u32, height: u32) {
        self.shared.lock().unwrap().on_resize(device, width, height);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Cost Analyzer Pass
// ─────────────────────────────────────────────────────────────────────────────

impl PerfOverlayCostAnalyzerPass {
    pub fn new(shared: Arc<Mutex<PerfOverlayShared>>) -> Self {
        Self { shared }
    }
}

impl RenderPass for PerfOverlayCostAnalyzerPass {
    fn name(&self) -> &'static str {
        "PerfOverlay Cost Analyzer"
    }

    fn prepare(&mut self, ctx: &PrepareContext) -> HelioResult<()> {
        let mut shared = self.shared.lock().unwrap();
        
        // Initialize material profiler if Mode 2 is active and profiler doesn't exist
        if *shared.mode.lock().unwrap() == PerfOverlayMode::ShaderComplexity && shared.material_profiler.is_none() {
            shared.init_material_profiler(ctx.device, ctx.queue);
        }
        
        // Upload timing data if profiling is complete and not yet uploaded
        let timing_data_to_upload = if let Some(profiler) = &mut shared.material_profiler {
            if profiler.profiling_complete && !profiler.timings_uploaded {
                // Compute final averaged timings
                profiler.compute_final_timings();
                
                // Convert to GPU format (16 bytes per entry: f32, f32, u32, u32)
                let mut timing_data: Vec<u8> = Vec::new();
                for entry in &profiler.timing_table {
                    timing_data.extend_from_slice(bytemuck::bytes_of(&entry.roughness));
                    timing_data.extend_from_slice(bytemuck::bytes_of(&entry.metallic));
                    timing_data.extend_from_slice(bytemuck::bytes_of(&entry.num_lights));
                    timing_data.extend_from_slice(bytemuck::bytes_of(&entry.gpu_time_ns));
                }
                
                profiler.timings_uploaded = true;
                Some(timing_data)
            } else {
                None
            }
        } else {
            None
        };
        
        // Upload timing data if we have any
        if let Some(timing_data) = timing_data_to_upload {
            ctx.queue.write_buffer(&shared.material_timing_buf, 0, &timing_data);
        }
        
        let num_timing_entries = if let Some(profiler) = &shared.material_profiler {
            if profiler.timings_uploaded {
                profiler.timing_table.len() as u32
            } else {
                0
            }
        } else {
            0
        };
        
        let cost_params = ComputeCostParams {
            screen_width: shared.internal_width,
            screen_height: shared.internal_height,
            num_tiles_x: shared.num_tiles_x,
            num_timing_entries,
        };
        ctx.write_buffer(
            &shared.cost_compute_params_buf,
            0,
            bytemuck::bytes_of(&cost_params),
        );
        Ok(())
    }

    fn execute(&mut self, ctx: &mut PassContext) -> HelioResult<()> {
        let mut shared = self.shared.lock().unwrap();
        if *shared.mode.lock().unwrap() != PerfOverlayMode::ShaderComplexity {
            return Ok(());
        }

        // Run material profiling incrementally if not complete
        if let Some(profiler) = &mut shared.material_profiler {
            if !profiler.profiling_complete {
                // Profile one sample per frame
                profiler.profile_next(
                    ctx.device,
                    ctx.encoder,
                    ctx.scene.lights,
                );
                
                // Read back the sample we just profiled
                profiler.read_current_sample_blocking(ctx.device);
            }
        }

        // Get GBuffer and tile light counts
        if let (Some(gbuffer), Some(tile_light_counts)) = 
            (ctx.resources.gbuffer, ctx.resources.tile_light_counts) 
        {
            // Create bind group for cost computation
            let cost_compute_bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("PerfOverlay Cost Compute BG"),
                layout: &shared.cost_compute_bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: shared.cost_compute_params_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(gbuffer.orm),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::TextureView(ctx.depth),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: tile_light_counts.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: shared.shader_cost_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: shared.material_timing_buf.as_entire_binding(),
                    },
                ],
            });

            // Clear cost buffer at start of frame
            if shared.runtime.lock().unwrap().frame_num != ctx.frame_num {
                ctx.encoder.clear_buffer(&shared.shader_cost_buf, 0, None);
                shared.runtime.lock().unwrap().frame_num = ctx.frame_num;
            }

            // Dispatch cost computation
            let mut pass = ctx.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("PerfOverlay Cost Compute"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&shared.cost_compute_pipeline);
            pass.set_bind_group(0, &cost_compute_bg, &[]);
            let dispatch_x = shared.internal_width.div_ceil(16);
            let dispatch_y = shared.internal_height.div_ceil(16);
            pass.dispatch_workgroups(dispatch_x, dispatch_y, 1);
        }

        Ok(())
    }

    fn on_resize(&mut self, device: &wgpu::Device, width: u32, height: u32) {
        self.shared.lock().unwrap().on_resize(device, width, height);
    }
}

