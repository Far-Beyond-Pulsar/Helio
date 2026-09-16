//! GPU tiled light culling pass (Forward+).
//!
//! Divides the screen into 16×16-pixel tiles, then runs a compute shader that
//! tests every light sphere against each tile's view-space frustum.  The result
//! is two storage buffers:
//!
//! * `tile_light_counts[tile_idx]`  — number of lights that hit this tile
//! * `tile_light_lists[tile_idx * MAX_LIGHTS_PER_TILE + i]` — light index i
//!
//! These buffers are published into `ResourceRegistry` so `DeferredLightPass` can
//! skip every light that doesn't touch the current pixel's tile.

use bytemuck::{Pod, Zeroable};
use helio_core::graph::ResourceBuilder;
use helio_core::{PassContext, PrepareContext, RenderPass, Result as HelioResult};

pub mod gpu_types;
pub use gpu_types::*;
use pulsar_scenedb::gpu::{world_mirror::DEFAULT_AUTO_REGISTER_CAPACITY, BufferKey};

pub const TILE_SIZE: u32 = 16;
pub const MAX_LIGHTS_PER_TILE: u32 = 64;
/// Fixed capacity for the `"scene_lights"` SceneDB buffer this pass culls --
/// kept equal to `helio_pass_forward_lit::MAX_LIGHTS` by construction (both
/// are literally `DEFAULT_AUTO_REGISTER_CAPACITY`), so culling always covers
/// every light `ForwardLitPass` can possibly shade.
pub const MAX_LIGHTS: u32 = DEFAULT_AUTO_REGISTER_CAPACITY;

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
    light_mode_direct_index: u32,
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
    /// Key: (camera_ptr, lights_ptr, light_entity_indices_ptr, transforms_ptr)
    /// — used to skip needless bind-group rebuilds.
    bind_group_key: Option<(usize, usize, usize, usize)>,
    /// Light culling cache key: (camera_generation, lights generation --
    /// either the SceneDB buffer's epoch or `movable_lights_generation`
    /// depending on which source is active, movable_light_count,
    /// use_direct_index) — used to skip culling compute when nothing the
    /// shader reads has changed.
    cull_cache_key: Option<(u64, u64, u32, bool)>,
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
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/light_cull.wgsl").into()),
        });

        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("LightCull BGL"),
            entries: &[
                // 0: camera storage
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
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
                // 3: SceneDB `Transform` storage read -- entity-indexed the same
                // way `lights` is, see `light_cull.wgsl`'s binding doc.
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
                // 4: light_entity_indices storage read -- parallel to `lights`
                // only in CPU-resolved mode, see `light_cull.wgsl`'s binding doc.
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
                // 5: tile_light_lists read_write
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // 6: tile_light_counts read_write
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
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

    fn writes(&self) -> &'static [&'static str] {
        &["tile_light_lists", "tile_light_counts"]
    }

    fn declare_resources(&self, builder: &mut ResourceBuilder) {
        builder.write_buffer("tile_light_lists");
        builder.write_buffer("tile_light_counts");
    }

    fn on_resize(&mut self, device: &wgpu::Device, width: u32, height: u32) {
        let num_tiles_x = width.div_ceil(TILE_SIZE);
        let num_tiles_y = height.div_ceil(TILE_SIZE);
        let num_tiles = num_tiles_x
            .checked_mul(num_tiles_y)
            .expect("tile grid overflow: viewport dimensions too large");

        let list_buf_size = (num_tiles * MAX_LIGHTS_PER_TILE * 4) as u64;
        self.tile_light_lists = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("TileLightLists"),
            size: list_buf_size.max(4),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let count_buf_size = (num_tiles * 4) as u64;
        self.tile_light_counts = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("TileLightCounts"),
            size: count_buf_size.max(4),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        self.num_tiles_x = num_tiles_x;
        self.num_tiles_y = num_tiles_y;
        self.width = width;
        self.height = height;
        // Invalidate cached bind group so it gets rebuilt with the new buffers.
        self.bind_group = None;
        self.bind_group_key = None;
        self.cull_cache_key = None;
    }

    fn publish<'a>(&self, frame: &mut helio_core::ResourceRegistry<'a>) {
        // SAFETY: extended out of `self`'s own owned buffers, which are
        // pass-lifetime (owned by the RenderGraph across many frames), never
        // frame-scoped -- see `helio-pass-object-batch::publish`'s identical
        // comment for the full reasoning.
        let tile_light_lists: &'a wgpu::Buffer =
            unsafe { std::mem::transmute(&self.tile_light_lists) };
        let tile_light_counts: &'a wgpu::Buffer =
            unsafe { std::mem::transmute(&self.tile_light_counts) };
        frame.write(helio_core::ResourceKey::new("tile_light_lists"), tile_light_lists, "LightCull");
        frame.write(helio_core::ResourceKey::new("tile_light_counts"), tile_light_counts, "LightCull");
        frame.write(helio_core::ResourceKey::new("cluster_light_grid"),
            crate::ClusterLightGrid {
                tile_light_lists,
                tile_light_counts,
                num_tiles_x: self.num_tiles_x,
                num_tiles_y: self.num_tiles_y,
            },
            "LightCull",
        );
    }

    fn render_pass_descriptor<'a>(
        &'a self,
        _target: &'a wgpu::TextureView,
        _depth: &'a wgpu::TextureView,
        _resources: &'a helio_core::ResourceRegistry<'a>,
    ) -> Option<wgpu::RenderPassDescriptor<'a>> {
        None
    }

    fn prepare(&mut self, ctx: &PrepareContext) -> HelioResult<()> {
        // Tile grid must match the internal (render-target) resolution, not output.
        // ctx.width/height are internal_w/h from the graph.
        self.num_tiles_x = ctx.width.div_ceil(TILE_SIZE);
        self.num_tiles_y = ctx.height.div_ceil(TILE_SIZE);
        // Prefer the SceneDB-direct `"scene_lights"` buffer (fixed capacity
        // `MAX_LIGHTS`, no per-frame CPU query) when populated; else
        // `ctx.scene.movable_light_count`, production's actual light count
        // today (`Renderer::submit_light_frame`, driven by `engine_backend`'s
        // own SceneDB resolve) -- see `light_mode_direct_index`'s doc.
        let use_direct_index = ctx.scene_buffers.contains(BufferKey::of("scene_lights"));
        let num_lights = if use_direct_index { MAX_LIGHTS } else { 0 };
        let params = LightCullParams {
            num_tiles_x: self.num_tiles_x,
            num_tiles_y: ctx.height.div_ceil(TILE_SIZE),
            num_lights,
            screen_width: ctx.width,
            screen_height: ctx.height,
            light_mode_direct_index: use_direct_index as u32,
            _pad1: 0,
            _pad2: 0,
        };
        ctx.queue
            .write_buffer(&self.params_buf, 0, bytemuck::bytes_of(&params));
        Ok(())
    }

    fn execute(&mut self, ctx: &mut PassContext) -> HelioResult<()> {
        let scene_lights_handle = ctx.scene_buffers.get(BufferKey::of("scene_lights"));
        let use_direct_index = scene_lights_handle.is_some();
        let lights_buf = scene_lights_handle
            .map(|handle| &handle.buffer)
            .unwrap_or(ctx.camera);
        let light_entity_indices_buf = ctx.camera;
        let movable_light_count = if use_direct_index { MAX_LIGHTS } else { 0 };

        if !use_direct_index && movable_light_count == 0 {
            // No active movable lights via either source: clear light
            // lists/counts to avoid stale data usage. Static/stationary
            // lights are baked and don't need runtime culling.
            unsafe { &mut *ctx.encoder_ptr }.clear_buffer(&self.tile_light_lists, 0, None);
            unsafe { &mut *ctx.encoder_ptr }.clear_buffer(&self.tile_light_counts, 0, None);
            self.cull_cache_key = None; // Invalidate cache
            return Ok(());
        }

        // Fallback mirrors `ForwardLitPass`'s: before any entity has a
        // `Transform` yet, bind *some* valid buffer so bind-group creation
        // can't fail -- `params.num_lights` is 0 whenever `transforms` would
        // actually be dereferenced at a live light's index, so this is never
        // read in practice.
        let transforms_buf = ctx.camera;

        // ── Light culling cache: skip compute if scene static ─────────────────
        // Use generation counters to detect actual data changes (not pointer
        // addresses) for the CPU-resolved path; the SceneDB-direct path uses
        // its buffer's own epoch instead, since nothing else identifies "did
        // the row data change" for it.
        let camera_gen = ctx.camera_generation;
        let lights_gen = scene_lights_handle.map(|h| h.epoch).unwrap_or(0);

        let cache_key = (
            camera_gen,
            lights_gen,
            movable_light_count,
            use_direct_index,
        );

        // `self.width/height` are internal-resolution values maintained by
        // on_resize. ctx.width/height are full output resolution, so do not
        // use them as a resize signal here.
        let resolution_changed = false;

        // Check if we can reuse previous frame's culling results
        if self.cull_cache_key == Some(cache_key) && !resolution_changed {
            // Camera, lights, and resolution unchanged - reuse cached tile culling results
            return Ok(());
        }

        // Update cache key
        self.cull_cache_key = Some(cache_key);

        let camera_ptr = ctx.camera as *const _ as usize;
        let lights_ptr = lights_buf as *const _ as usize;
        let light_entity_indices_ptr = light_entity_indices_buf as *const _ as usize;
        let transforms_ptr = transforms_buf as *const _ as usize;
        let key = (
            camera_ptr,
            lights_ptr,
            light_entity_indices_ptr,
            transforms_ptr,
        );

        if self.bind_group_key != Some(key) {
            self.bind_group = Some(ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("LightCull BG"),
                layout: &self.bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: ctx.camera.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: self.params_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: lights_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: transforms_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: light_entity_indices_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: self.tile_light_lists.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 6,
                        resource: self.tile_light_counts.as_entire_binding(),
                    },
                ],
            }));
            self.bind_group_key = Some(key);
        }

        let total_tiles = self.num_tiles_x * self.num_tiles_y;
        // Each workgroup has 256 threads, each thread handles one tile.
        let workgroups = total_tiles.div_ceil(256);

        let mut pass =
            unsafe { &mut *ctx.encoder_ptr }.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("LightCull"),
                timestamp_writes: None,
            });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, self.bind_group.as_ref().unwrap(), &[]);
        pass.dispatch_workgroups(workgroups, 1, 1);
        Ok(())
    }
}
