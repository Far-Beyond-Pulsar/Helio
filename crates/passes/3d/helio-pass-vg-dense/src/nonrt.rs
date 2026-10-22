//! Tier 3-NonRT: tiled depth-peel compute raster (runs everywhere, no int64).

use crate::common::*;
use crate::config::{self, DenseConfig};

/// Why a [`DenseNonRtPass`] could not be created.
#[derive(Debug, PartialEq, Eq)]
pub enum DenseError {
    InvalidConfig(String),
    /// The device offers less workgroup storage than the peel kernel declares.
    InsufficientWorkgroupStorage { needed: u32, available: u32 },
}

impl std::fmt::Display for DenseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidConfig(m) => write!(f, "invalid dense config: {m}"),
            Self::InsufficientWorkgroupStorage { needed, available } => {
                write!(f, "peel kernel needs {needed} B workgroup storage, device has {available} B")
            }
        }
    }
}
impl std::error::Error for DenseError {}

/// Buffers the NonRT kernels read (see `shaders/inputs.wgsl`).
pub struct DenseNonRtInputs<'a> {
    /// T7 `tile_classify`: one `u32` per tile, nonzero = dense.
    pub tile_mask: &'a wgpu::Buffer,
    /// The cull's `state` buffer; its per-phase/bin `{first, count}` ranges
    /// (word `CULL_STATE_RANGES + phase * 12 + bin * 2`) say which entries of
    /// `visible` are live. Both phases (the late one is the delta) are binned.
    pub cull_state: &'a wgpu::Buffer,
    /// The cull's `visible_clusters`: `GpuVisibleCluster { instance, cluster_slot }`
    /// rows indexed by the unified visible index.
    pub visible: &'a wgpu::Buffer,
    /// `helio-vg-stream` page pool.
    pub page_pool: &'a wgpu::Buffer,
    /// `vg_asset_quant` by asset slot.
    pub asset_quant: &'a wgpu::Buffer,
    /// `GpuVgInstance` rows.
    pub vg_instances: &'a wgpu::Buffer,
    /// The scene join's `static_objects` rows (252 B).
    pub objects: &'a wgpu::Buffer,
}

pub struct DenseNonRtPass {
    cfg: DenseConfig,
    compact: wgpu::ComputePipeline,
    args: wgpu::ComputePipeline,
    bin: wgpu::ComputePipeline,
    peel: wgpu::ComputePipeline,
    compact_bgl: wgpu::BindGroupLayout,
    args_bgl: wgpu::BindGroupLayout,
    bin_bgl: wgpu::BindGroupLayout,
    peel_bgl: wgpu::BindGroupLayout,
    buffers: Option<DenseBuffers>,
}

impl DenseNonRtPass {
    pub fn new(device: &wgpu::Device, cfg: DenseConfig) -> Result<Self, DenseError> {
        cfg.validate().map_err(DenseError::InvalidConfig)?;
        let available = device.limits().max_compute_workgroup_storage_size;
        if cfg.shared_memory_bytes() > available {
            return Err(DenseError::InsufficientWorkgroupStorage {
                needed: cfg.shared_memory_bytes(),
                available,
            });
        }
        let compact_bgl = bgl(
            device,
            "vg_dense compact",
            &[
                bgl_entry(0, uniform()),
                bgl_entry(1, storage(true)),
                bgl_entry(2, storage(false)),
                bgl_entry(3, storage(false)),
                bgl_entry(4, storage(false)),
            ],
        );
        let args_bgl = bgl(
            device,
            "vg_dense args",
            &[
                bgl_entry(0, uniform()),
                bgl_entry(1, storage(false)),
                bgl_entry(2, storage(true)),
                bgl_entry(3, storage(false)),
            ],
        );
        let bin_bgl = bgl(
            device,
            "vg_dense bin",
            &[
                bgl_entry(0, uniform()),
                bgl_entry(2, storage(true)),
                bgl_entry(3, storage(true)),
                bgl_entry(5, storage(true)),
                bgl_entry(6, storage(true)),
                bgl_entry(7, storage(false)),
                bgl_entry(8, storage(false)),
                bgl_entry(10, storage(true)),
                bgl_entry(11, storage(true)),
            ],
        );
        let peel_bgl = bgl(
            device,
            "vg_dense peel",
            &[
                bgl_entry(0, uniform()),
                bgl_entry(2, storage(true)),
                bgl_entry(3, storage(true)),
                bgl_entry(4, storage(true)),
                bgl_entry(5, storage(true)),
                bgl_entry(7, storage(false)),
                bgl_entry(8, storage(false)),
                bgl_entry(9, storage(false)),
                bgl_entry(10, storage(true)),
            ],
        );
        let tiles = config::tiles_source(&cfg);
        let args = config::args_source(&cfg);
        let bin = config::bin_source(&cfg);
        let peel = config::peel_source(&cfg);
        Ok(Self {
            compact: pipeline(device, "vg_dense compact", &tiles, "compact", &compact_bgl),
            args: pipeline(device, "vg_dense make_args", &args, "make_args", &args_bgl),
            bin: pipeline(device, "vg_dense bin", &bin, "bin", &bin_bgl),
            peel: pipeline(device, "vg_dense peel", &peel, "peel", &peel_bgl),
            compact_bgl,
            args_bgl,
            bin_bgl,
            peel_bgl,
            cfg,
            buffers: None,
        })
    }

    pub fn config(&self) -> &DenseConfig {
        &self.cfg
    }

    /// (Re)allocate the owned buffers for `viewport`. Done lazily by
    /// [`DenseNonRtPass::record`]; exposed so callers can pre-size.
    pub fn ensure_buffers(&mut self, device: &wgpu::Device, viewport: [u32; 2]) -> &DenseBuffers {
        if !self.buffers.as_ref().is_some_and(|b| b.matches(&self.cfg, viewport)) {
            self.buffers = Some(DenseBuffers::new(device, &self.cfg, viewport, true));
        }
        self.buffers.as_ref().unwrap()
    }

    pub fn buffers(&self) -> Option<&DenseBuffers> {
        self.buffers.as_ref()
    }

    /// Record the whole NonRT dense pass: compaction, indirect args, cluster
    /// binning and the per-tile peel. Output lands in `buffers().dense_key`.
    pub fn record(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        frame: &DenseFrame,
        inputs: &DenseNonRtInputs<'_>,
    ) {
        let mut rec = helio_core::CommandRecorder::from_encoder(encoder);
        self.record_rec(device, queue, &mut rec, frame, inputs);
    }

    /// As [`Self::record`], into a render-graph command stream.
    pub fn record_rec(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut helio_core::CommandRecorder<'_>,
        frame: &DenseFrame,
        inputs: &DenseNonRtInputs<'_>,
    ) {
        self.ensure_buffers(device, frame.viewport);
        let b = self.buffers.as_ref().unwrap();
        b.write_params(queue, frame);
        // Counters only; the tile list tail is rewritten by `compact`.
        encoder.clear_buffer(&b.state, 0, Some(STATE_HEADER_WORDS as u64 * 4));

        let compact_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("vg_dense compact"),
            layout: &self.compact_bgl,
            entries: &[
                bg_entry(0, &b.params),
                bg_entry(1, inputs.tile_mask),
                bg_entry(2, &b.tile_slot),
                bg_entry(3, &b.state),
                bg_entry(4, &b.slot_lists),
            ],
        });
        let args_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("vg_dense args"),
            layout: &self.args_bgl,
            entries: &[
                bg_entry(0, &b.params),
                bg_entry(1, &b.state),
                bg_entry(2, inputs.cull_state),
                bg_entry(3, &b.args),
            ],
        });
        let bin_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("vg_dense bin"),
            layout: &self.bin_bgl,
            entries: &[
                bg_entry(0, &b.params),
                bg_entry(2, inputs.visible),
                bg_entry(3, inputs.page_pool),
                bg_entry(5, inputs.vg_instances),
                bg_entry(6, &b.tile_slot),
                bg_entry(7, &b.slot_lists),
                bg_entry(8, &b.state),
                bg_entry(10, inputs.objects),
                bg_entry(11, &b.args),
            ],
        });
        let peel_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("vg_dense peel"),
            layout: &self.peel_bgl,
            entries: &[
                bg_entry(0, &b.params),
                bg_entry(2, inputs.visible),
                bg_entry(3, inputs.page_pool),
                bg_entry(4, inputs.asset_quant),
                bg_entry(5, inputs.vg_instances),
                bg_entry(7, &b.slot_lists),
                bg_entry(8, &b.state),
                bg_entry(9, &b.dense_key),
                bg_entry(10, inputs.objects),
            ],
        });

        let mut run = |label: &str, f: &dyn Fn(&mut helio_core::ComputeCmds<'_>)| {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some(label),
                timestamp_writes: None,
            });
            f(&mut pass);
        };
        run("vg_dense compact", &|p| {
            p.set_pipeline(&self.compact);
            p.set_bind_group(0, &compact_bg, &[]);
            p.dispatch_workgroups(b.tile_count().div_ceil(64), 1, 1);
        });
        run("vg_dense make_args", &|p| {
            p.set_pipeline(&self.args);
            p.set_bind_group(0, &args_bg, &[]);
            p.dispatch_workgroups(1, 1, 1);
        });
        run("vg_dense bin", &|p| {
            p.set_pipeline(&self.bin);
            p.set_bind_group(0, &bin_bg, &[]);
            p.dispatch_workgroups_indirect(&b.args, 12);
        });
        run("vg_dense peel", &|p| {
            p.set_pipeline(&self.peel);
            p.set_bind_group(0, &peel_bg, &[]);
            p.dispatch_workgroups_indirect(&b.args, 0);
        });
    }
}
