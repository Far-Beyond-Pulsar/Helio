//! GPU tiled light culling pass (Forward+).
//!
//! Divides the screen into 16×16-pixel tiles, then runs a compute shader that
//! tests every light sphere against each tile's view-space frustum.  The result
//! is two storage buffers:
//!
//! * `tile_light_counts[tile_idx]`  — number of lights that hit this tile
//! * `tile_light_lists[tile_idx * MAX_LIGHTS_PER_TILE + i]` — light index i
//!
//! These buffers are published into `FrameResources` so `DeferredLightPass` can
//! skip every light that doesn't touch the current pixel's tile.

use bytemuck::{Pod, Zeroable};
use helio_v3::{PassContext, PrepareContext, RenderPass, Result as HelioResult};

pub const TILE_SIZE: u32 = 16;
pub const MAX_LIGHTS_PER_TILE: u32 = 64;

// ─────────────────────────────────────────────────────────────────────────────
// GPU-side uniform mirroring LightCullParams in the WGSL shader.
// ─────────────────────────────────────────────────────────────────────────────

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct LightCullParams {
    num_tiles_x: u32,
    num_tiles_y: u32,
    num_lights: u32,
    screen_width: u32,
    screen_height: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

// ─────────────────────────────────────────────────────────────────────────────
// Pass struct
// ─────────────────────────────────────────────────────────────────────────────

pub struct LightCullPass {
    pipeline: wgpu::ComputePipeline,
    bgl: wgpu::BindGroupLayout,
    params_buf: wgpu::Buffer,
    /// Storage buffer: u32 per light-slot per tile.
    /// Size: num_tiles * MAX_LIGHTS_PER_TILE * 4 bytes.
    pub tile_light_lists: wgpu::Buffer,
    /// Storage buffer: one u32 count per tile.
    /// Size: num_tiles * 4 bytes.
    pub tile_light_counts: wgpu::Buffer,
    /// Cached bind group, rebuilt when camera or lights buffer pointer changes.
    bind_group: Option<wgpu::BindGroup>,
    /// Key: (camera_ptr, lights_ptr) — used to skip needless bind-group rebuilds.
    bind_group_key: Option<(usize, usize)>,
    /// Light culling cache key: (camera_generation, lights_generation, light_count) — used to skip culling compute when scene static.
    cull_cache_key: Option<(u64, u64, u32)>,
    num_tiles_x: u32,
    num_tiles_y: u32,
    width: u32,
    height: u32,
}

impl LightCullPass {
    pub fn new(device: &wgpu::Device, width: u32, height: u32) -> Self {
        let num_tiles_x = width.div_ceil(TILE_SIZE);
        let num_tiles_y = height.div_ceil(TILE_SIZE);
        let num_tiles = num_tiles_x
            .checked_mul(num_tiles_y)
            .expect("tile grid overflow: viewport dimensions too large");

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("LightCull Shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../shaders/light_cull.wgsl").into(),
            ),
        });

        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("LightCull BGL"),
            entries: &[
                // 0: camera uniform
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
                // 1: LightCullParams uniform
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // 2: lights storage read
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // 3: tile_light_lists read_write
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
                // 4: tile_light_counts read_write
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

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("LightCull PL"),
            bind_group_layouts: &[Some(&bgl)],
            immediate_size: 0,
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("LightCull Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        let params_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("LightCull Params"),
            size: std::mem::size_of::<LightCullParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let list_buf_size = (num_tiles * MAX_LIGHTS_PER_TILE * 4) as u64;
        let tile_light_lists = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("TileLightLists"),
            size: list_buf_size.max(4),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let count_buf_size = (num_tiles * 4) as u64;
        let tile_light_counts = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("TileLightCounts"),
            size: count_buf_size.max(4),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            pipeline,
            bgl,
            params_buf,
            tile_light_lists,
            tile_light_counts,
            bind_group: None,
            bind_group_key: None,
            cull_cache_key: None,
            num_tiles_x,
            num_tiles_y,
            width,
            height,
        }
    }
}

impl RenderPass for LightCullPass {
    fn name(&self) -> &'static str {
        "LightCull"
    }

    fn publish<'a>(&'a self, frame: &mut libhelio::FrameResources<'a>) {
        frame.tile_light_lists = Some(&self.tile_light_lists);
        frame.tile_light_counts = Some(&self.tile_light_counts);
    }

    fn prepare(&mut self, ctx: &PrepareContext) -> HelioResult<()> {
        let params = LightCullParams {
            num_tiles_x: self.num_tiles_x,
            num_tiles_y: self.num_tiles_y,
            num_lights: ctx.scene.movable_light_count, // Only process movable lights (static lights are baked)
            screen_width: self.width,
            screen_height: self.height,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        };
        ctx.queue
            .write_buffer(&self.params_buf, 0, bytemuck::bytes_of(&params));
        Ok(())
    }

    fn execute(&mut self, ctx: &mut PassContext) -> HelioResult<()> {
        if ctx.scene.movable_light_count == 0 {
            // No active movable lights: clear light lists/counts to avoid stale data usage.
            // Static/stationary lights are baked and don't need runtime culling.
            ctx.encoder.clear_buffer(&self.tile_light_lists, 0, None);
            ctx.encoder.clear_buffer(&self.tile_light_counts, 0, None);
            self.cull_cache_key = None; // Invalidate cache
            return Ok(());
        }

        // ── Light culling cache: skip compute if scene static ─────────────────
        // Use generation counters to detect actual data changes (not pointer addresses).
        let camera_gen = ctx.scene.camera_generation;
        let lights_gen = ctx.scene.movable_lights_generation;

        let cache_key = (camera_gen, lights_gen, ctx.scene.movable_light_count);

        // Check if resolution changed (window resize invalidates tile grid)
        let resolution_changed = ctx.width != self.width || ctx.height != self.height;

        // Check if we can reuse previous frame's culling results
        if self.cull_cache_key == Some(cache_key) && !resolution_changed {
            // Camera, lights, and resolution unchanged - reuse cached tile culling results
            return Ok(());
        }

        // Update cache key
        self.cull_cache_key = Some(cache_key);

        let camera_ptr = ctx.scene.camera as *const _ as usize;
        let lights_ptr = ctx.scene.lights as *const _ as usize;
        let key = (camera_ptr, lights_ptr);

        if self.bind_group_key != Some(key) {
            self.bind_group = Some(ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("LightCull BG"),
                layout: &self.bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: ctx.scene.camera.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: self.params_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: ctx.scene.lights.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: self.tile_light_lists.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: self.tile_light_counts.as_entire_binding(),
                    },
                ],
            }));
            self.bind_group_key = Some(key);
        }

        let total_tiles = self.num_tiles_x * self.num_tiles_y;
        // Each workgroup has 256 threads, each thread handles one tile.
        let workgroups = total_tiles.div_ceil(256);

        let mut pass = ctx
            .encoder
            .begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("LightCull"),
                timestamp_writes: None,
            });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, self.bind_group.as_ref().unwrap(), &[]);
        pass.dispatch_workgroups(workgroups, 1, 1);
        Ok(())
    }
}
