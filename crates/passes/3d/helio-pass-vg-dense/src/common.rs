//! Types and buffers shared by the 3-NonRT and 3-RT backends.

use crate::config::DenseConfig;
use bytemuck::{Pod, Zeroable};

/// Byte size of the `state` counters header (16 u32). The dense tile list
/// follows it.
pub const STATE_HEADER_WORDS: u32 = 16;

/// `DenseParams` in `common.wgsl`. 192 bytes.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct DenseParamsGpu {
    pub view_proj: [[f32; 4]; 4],
    pub inv_view_proj: [[f32; 4]; 4],
    pub camera_pos: [f32; 4],
    pub viewport: [u32; 4],
    pub limits: [u32; 4],
    pub misc: [f32; 4],
}

/// Per-frame inputs both backends need.
#[derive(Clone, Copy, Debug)]
pub struct DenseFrame {
    /// Column-major view-projection (reversed-Z, `m[col][row]`).
    pub view_proj: [[f32; 4]; 4],
    /// Its inverse (3-RT only).
    pub inv_view_proj: [[f32; 4]; 4],
    pub camera_pos: [f32; 3],
    pub viewport: [u32; 2],
    /// Capacity of T5's visible-cluster list (3-NonRT).
    pub visible_capacity: u32,
    /// Ray interval (3-RT). `tmin` should be the camera near distance.
    pub ray_t: [f32; 2],
}

impl DenseFrame {
    pub fn tiles(&self, cfg: &DenseConfig) -> [u32; 2] {
        [self.viewport[0].div_ceil(cfg.tile), self.viewport[1].div_ceil(cfg.tile)]
    }
}

/// Counters the kernels maintain (`state[0..8]`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DenseStats {
    /// Dense tiles requested (may exceed the slot cap).
    pub dense_tiles: u32,
    /// Sum of peel layers over all tiles (3-NonRT).
    pub layers: u32,
    /// Tiles that hit `max_layers` and returned a partial peel.
    pub clamped_tiles: u32,
    /// Triangles left unprocessed in clamped tiles.
    pub clamped_tris: u32,
    /// Clusters dropped from per-tile lists (cap `cluster_cap`).
    pub bin_overflow: u32,
    /// Dense tiles dropped because `max_dense_slots` was exceeded.
    pub slot_overflow: u32,
    /// Layers beyond `nominal_layers` (the extra peel iterations).
    pub extra_layers: u32,
    /// Tiles actually processed.
    pub tiles_processed: u32,
}

impl DenseStats {
    pub fn from_words(w: &[u32]) -> Self {
        Self {
            dense_tiles: w[0],
            layers: w[1],
            clamped_tiles: w[2],
            clamped_tris: w[3],
            bin_overflow: w[4],
            slot_overflow: w[5],
            extra_layers: w[6],
            tiles_processed: w[7],
        }
    }

    /// True when any result was degraded (dropped or clamped work). Surface
    /// this in the VG debug UI ("overflow" counter of plan §5.4).
    pub fn overflowed(&self) -> bool {
        self.clamped_tiles + self.bin_overflow + self.slot_overflow > 0
    }
}

/// Buffers owned by a Tier-3 backend, sized for one viewport.
pub struct DenseBuffers {
    pub viewport: [u32; 2],
    pub tiles: [u32; 2],
    pub max_slots: u32,
    pub cfg: DenseConfig,
    pub params: wgpu::Buffer,
    /// `u32` per tile: slot or `u32::MAX`.
    pub tile_slot: wgpu::Buffer,
    /// Counters header + dense tile list.
    pub state: wgpu::Buffer,
    /// Per-slot cluster lists (3-NonRT): `[count, ids...]` x slots.
    pub slot_lists: wgpu::Buffer,
    /// Indirect args: `[peel x,y,z, bin x,y,z]`.
    pub args: wgpu::Buffer,
    /// Output: `vec2<u32>(depth bits, vis id)` per pixel. Only dense-tile
    /// pixels are written; consumers read it through the dense tile mask.
    pub dense_key: wgpu::Buffer,
    pub stats_staging: wgpu::Buffer,
}

impl DenseBuffers {
    pub fn new(device: &wgpu::Device, cfg: &DenseConfig, viewport: [u32; 2], with_lists: bool) -> Self {
        let tiles = [viewport[0].div_ceil(cfg.tile), viewport[1].div_ceil(cfg.tile)];
        let tile_count = tiles[0] * tiles[1];
        let max_slots = cfg.max_dense_slots.min(tile_count).max(1);
        let storage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC;
        let make = |label: &str, size: u64, usage: wgpu::BufferUsages| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: size.max(16),
                usage,
                mapped_at_creation: false,
            })
        };
        Self {
            viewport,
            tiles,
            max_slots,
            cfg: *cfg,
            params: make(
                "vg_dense params",
                std::mem::size_of::<DenseParamsGpu>() as u64,
                wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            ),
            tile_slot: make("vg_dense tile_slot", tile_count as u64 * 4, storage),
            state: make(
                "vg_dense state",
                (STATE_HEADER_WORDS + max_slots) as u64 * 4,
                storage,
            ),
            slot_lists: make(
                "vg_dense slot_lists",
                if with_lists { max_slots as u64 * (cfg.cluster_cap as u64 + 1) * 4 } else { 16 },
                storage,
            ),
            args: make("vg_dense args", (8 + 12 + 13) * 4, storage | wgpu::BufferUsages::INDIRECT),
            dense_key: make(
                "vg_dense key",
                viewport[0] as u64 * viewport[1] as u64 * 8,
                storage,
            ),
            stats_staging: make(
                "vg_dense stats staging",
                STATE_HEADER_WORDS as u64 * 4,
                wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            ),
        }
    }

    pub fn matches(&self, cfg: &DenseConfig, viewport: [u32; 2]) -> bool {
        self.cfg == *cfg && self.viewport == viewport
    }

    pub fn tile_count(&self) -> u32 {
        self.tiles[0] * self.tiles[1]
    }

    pub(crate) fn write_params(&self, queue: &wgpu::Queue, frame: &DenseFrame) {
        let p = DenseParamsGpu {
            view_proj: frame.view_proj,
            inv_view_proj: frame.inv_view_proj,
            camera_pos: [frame.camera_pos[0], frame.camera_pos[1], frame.camera_pos[2], 0.0],
            viewport: [frame.viewport[0], frame.viewport[1], self.tiles[0], self.tiles[1]],
            limits: [self.tile_count(), self.max_slots, frame.visible_capacity, 0],
            misc: [frame.ray_t[0], frame.ray_t[1], 0.0, 0.0],
        };
        queue.write_buffer(&self.params, 0, bytemuck::bytes_of(&p));
    }

    /// Record a copy of the counters header for readback with
    /// [`DenseBuffers::read_stats`] after submission.
    pub fn copy_stats(&self, encoder: &mut wgpu::CommandEncoder) {
        encoder.copy_buffer_to_buffer(&self.state, 0, &self.stats_staging, 0, STATE_HEADER_WORDS as u64 * 4);
    }

    /// Blocking read of the counters copied by [`DenseBuffers::copy_stats`]
    /// (call after the submission completes). For tests and debug UI.
    pub fn read_stats(&self, device: &wgpu::Device) -> DenseStats {
        let slice = self.stats_staging.slice(..);
        slice.map_async(wgpu::MapMode::Read, |r| r.expect("map stats"));
        let _ = device.poll(wgpu::PollType::wait_indefinitely());
        let words: Vec<u32> = bytemuck::cast_slice(&slice.get_mapped_range().unwrap()).to_vec();
        self.stats_staging.unmap();
        DenseStats::from_words(&words)
    }
}

/// Blocking buffer readback helper (tests, tools).
pub fn read_buffer(device: &wgpu::Device, queue: &wgpu::Queue, src: &wgpu::Buffer, size: u64) -> Vec<u8> {
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("vg_dense readback"),
        size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("vg_dense readback") });
    enc.copy_buffer_to_buffer(src, 0, &staging, 0, size);
    queue.submit(Some(enc.finish()));
    let slice = staging.slice(..);
    slice.map_async(wgpu::MapMode::Read, |r| r.expect("map readback"));
    let _ = device.poll(wgpu::PollType::wait_indefinitely());
    let data = slice.get_mapped_range().unwrap().to_vec();
    staging.unmap();
    data
}

pub(crate) fn bgl_entry(binding: u32, ty: wgpu::BindingType) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry { binding, visibility: wgpu::ShaderStages::COMPUTE, ty, count: None }
}

pub(crate) fn storage(read_only: bool) -> wgpu::BindingType {
    wgpu::BindingType::Buffer {
        ty: wgpu::BufferBindingType::Storage { read_only },
        has_dynamic_offset: false,
        min_binding_size: None,
    }
}

pub(crate) fn uniform() -> wgpu::BindingType {
    wgpu::BindingType::Buffer {
        ty: wgpu::BufferBindingType::Uniform,
        has_dynamic_offset: false,
        min_binding_size: None,
    }
}

pub(crate) fn bgl(device: &wgpu::Device, label: &str, entries: &[wgpu::BindGroupLayoutEntry]) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: Some(label), entries })
}

pub(crate) fn pipeline(
    device: &wgpu::Device,
    label: &str,
    source: &str,
    entry: &str,
    layout: &wgpu::BindGroupLayout,
) -> wgpu::ComputePipeline {
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(label),
        source: wgpu::ShaderSource::Wgsl(source.into()),
    });
    let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some(label),
        bind_group_layouts: &[Some(layout)],
        immediate_size: 0,
    });
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(label),
        layout: Some(&pl),
        module: &module,
        entry_point: Some(entry),
        compilation_options: Default::default(),
        cache: None,
    })
}

pub(crate) fn bg_entry(binding: u32, buffer: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry { binding, resource: buffer.as_entire_binding() }
}
