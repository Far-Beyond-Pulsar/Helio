//! Render-graph wrapper of Tier 3: [`VgDensePass`] re-resolves the dense tiles
//! the raster classified, with either backend:
//!
//! * 3-NonRT depth peel (always available), or
//! * 3-RT, primary rays over a TLAS of proxy-LOD BLASes
//!   ([`crate::RtTier`]), when the device supports ray queries, the mode is
//!   `Auto`, every dense instance has a resident proxy and the dense-tile
//!   count justifies it ([`Tier3Selector`], with hysteresis).
//!
//! Both fill a `(depth bits, vis id)` key per pixel that `VgShade`'s
//! `vis_resolve` consumes identically; RT ids (cluster index
//! `>= DENSE_RT_VISIBLE_BASE`) resolve through [`VgDenseFrameData::rt_visible`].
//!
//! Policy per frame (decided in `prepare`, executed in `execute`):
//!
//! | mode | backend |
//! |---|---|
//! | `Off`, `ForceHwRaster` | none (Tier 1/2 only) |
//! | `ForceNonRtFallback` | NonRT; the RT machinery is not touched at all |
//! | `Auto`, no ray-query hardware / no runtime | NonRT |
//! | `Auto`, RT ready and enough dense tiles | RT |
//! | `Auto` otherwise | NonRT (the candidate collection keeps running so RT can become ready) |

use helio_core::graph::ResourceBuilder;
use helio_core::{PassContext, PrepareContext, RenderPass, ResourceKey, Result as HelioResult};
use helio_pass_vg_cull::{VgCullFrameData, VgCullInputs, BIN_COUNT, KEY_CULL, KEY_INPUTS};
use helio_pass_vg_raster::{VgRasterFrameData, KEY_RASTER};
use helio_vg_format::VgMode;
use helio_vg_stream::VgRuntime;

use crate::mask_union::MaskUnion;
use crate::stats_readback::StatsReadback;
use crate::{
    pixels_per_unit_from_proj, rt_available, AutoPolicy, DenseConfig, DenseError, DenseFrame, DenseNonRtInputs,
    DenseNonRtPass, RtEnv, RtTier, RtTierConfig, Tier3Backend, Tier3Selector,
};

/// [`VgDenseFrameData`] after Tier 3.
pub const KEY_DENSE: &str = "vg_dense";

/// The Tier 3 result for `vis_resolve`.
#[derive(Clone, Copy)]
pub struct VgDenseFrameData<'a> {
    /// `width * height` of `vec2<u32>{depth bits, vis id}` (see
    /// [`crate::DenseBuffers::dense_key`]); only dense-tile pixels are valid.
    pub dense_key: &'a wgpu::Buffer,
    /// The dense-tile mask `dense_key` is valid under.
    pub tile_mask: &'a wgpu::Buffer,
    /// Hardware barycentrics of the ray-traced backend, when it ran.
    pub rt_bary: Option<&'a wgpu::Buffer>,
    /// The ray-traced backend's visible rows (ids `>= DENSE_RT_VISIBLE_BASE`),
    /// when it ran.
    pub rt_visible: Option<&'a wgpu::Buffer>,
}

struct Published {
    key: wgpu::Buffer,
    mask: wgpu::Buffer,
    bary: Option<wgpu::Buffer>,
    visible: Option<wgpu::Buffer>,
}

/// Tier 3 as a graph pass.
pub struct VgDensePass {
    cfg: DenseConfig,
    nonrt: DenseNonRtPass,
    mask_union: MaskUnion,
    stats_rb: StatsReadback,
    selector: Tier3Selector,
    mode: Box<dyn Fn() -> VgMode + Send + Sync>,
    runtime: Option<VgRuntime>,
    tier_cfg: RtTierConfig,
    tier: Option<RtTier>,
    tier_failed: bool,
    /// Decided in `prepare`.
    backend: Tier3Backend,
    tier_active: bool,
    prep_dense_tiles: u32,
    published: Option<Published>,
}

impl VgDensePass {
    /// `mode` is read every frame (the project setting).
    pub fn new(
        device: &wgpu::Device,
        mode: Box<dyn Fn() -> VgMode + Send + Sync>,
    ) -> Result<Self, DenseError> {
        let cfg = DenseConfig::for_workgroup_storage(device.limits().max_compute_workgroup_storage_size);
        Ok(Self {
            cfg,
            nonrt: DenseNonRtPass::new(device, cfg)?,
            mask_union: MaskUnion::new(device, &cfg),
            stats_rb: StatsReadback::new(device),
            selector: Tier3Selector::new(AutoPolicy::dense_regions()),
            mode,
            runtime: None,
            tier_cfg: RtTierConfig::default(),
            tier: None,
            tier_failed: false,
            backend: Tier3Backend::Off,
            tier_active: false,
            prep_dense_tiles: 0,
            published: None,
        })
    }

    /// Enable the ray-traced backend: the runtime gives the pass the stream
    /// store (proxy sources, page residency), the settings (tau) and the
    /// counters it reports to. Without it the pass is NonRT-only.
    pub fn with_runtime(mut self, runtime: VgRuntime) -> Self {
        self.runtime = Some(runtime);
        self
    }

    pub fn with_rt_config(mut self, cfg: RtTierConfig) -> Self {
        self.tier_cfg = cfg;
        self
    }

    pub fn with_auto_policy(mut self, policy: AutoPolicy) -> Self {
        self.selector.policy = policy;
        self
    }
}

fn column_major(m: &[f32; 16]) -> [[f32; 4]; 4] {
    let mut out = [[0.0f32; 4]; 4];
    for (c, col) in out.iter_mut().enumerate() {
        col.copy_from_slice(&m[c * 4..c * 4 + 4]);
    }
    out
}

impl RenderPass for VgDensePass {
    fn name(&self) -> &'static str {
        "VgDense"
    }

    fn declare_resources(&self, builder: &mut ResourceBuilder) {
        builder.read(KEY_INPUTS);
        builder.read(KEY_CULL);
        builder.read(KEY_RASTER);
        builder.write_buffer(KEY_DENSE);
    }

    fn prepare(&mut self, ctx: &PrepareContext) -> HelioResult<()> {
        let mode = (self.mode)();
        self.tier_active = false;
        self.stats_rb.begin_maps();
        if let (Some(s), Some(runtime)) = (self.stats_rb.poll(ctx.device), self.runtime.as_ref()) {
            runtime.update_tier3_stats(|t| {
                t.dense_tiles_processed = s.tiles_processed;
                t.bin_overflow = s.bin_overflow;
                t.slot_overflow = s.slot_overflow;
                t.clamped_tiles = s.clamped_tiles;
                t.clamped_tris = s.clamped_tris;
            });
        }
        self.prep_dense_tiles = 0;
        let tiles_total = ctx.width.div_ceil(self.cfg.tile) * ctx.height.div_ceil(self.cfg.tile);
        let rt_hw = rt_available(ctx.device);
        if let Some(runtime) = self.runtime.clone() {
            runtime.update_tier3_stats(|s| s.rt_hardware = rt_hw);
        }
        // `ForceNonRtFallback` (and Off / ForceHwRaster) never touch the RT
        // machinery: no tier, no readbacks, no proxy builds, no BLAS/TLAS.
        let want_tier = mode == VgMode::Auto
            && rt_hw
            && !self.tier_failed
            && self.runtime.as_ref().is_some_and(|r| r.any_ready());
        if !want_tier {
            self.backend = self.selector.select(mode, false, false, 0, tiles_total);
            return Ok(());
        }
        let runtime = self.runtime.clone().expect("checked above");
        if self.tier.is_none() {
            match RtTier::new(ctx.device, self.cfg, self.tier_cfg) {
                Ok(t) => self.tier = Some(t),
                Err(e) => {
                    log::warn!("virtual geometry: ray-traced Tier 3 unavailable ({e}); dense tiles use the depth peel");
                    self.tier_failed = true;
                    self.backend = self.selector.select(mode, false, false, 0, tiles_total);
                    return Ok(());
                }
            }
        }
        let cam = ctx.camera_data;
        let frame = DenseFrame {
            view_proj: column_major(&cam.view_proj),
            inv_view_proj: column_major(&cam.inv_view_proj),
            camera_pos: [cam.position_near[0], cam.position_near[1], cam.position_near[2]],
            viewport: [ctx.width, ctx.height],
            visible_capacity: 0,
            ray_t: [cam.position_near[3], 1.0e30],
        };
        let env = RtEnv {
            device: ctx.device,
            queue: ctx.queue,
            runtime: &runtime,
            frame: &frame,
            tau_px: runtime.settings().tau_px,
            px_per_unit: pixels_per_unit_from_proj(&cam.proj, ctx.height),
        };
        let tier = self.tier.as_mut().expect("created above");
        let prep = tier.prepare(&env);
        self.tier_active = true;
        self.prep_dense_tiles = prep.dense_tiles;
        // A depth peel that overflowed its per-tile cluster lists is losing
        // geometry: prefer RT (the mask size alone may undercount the need).
        let overflowing = self.runtime.as_ref().is_some_and(|r| r.tier3_stats().bin_overflow > 0);
        let dense_for_policy = if overflowing { prep.dense_tiles.max(tiles_total) } else { prep.dense_tiles };
        self.backend = self.selector.select(mode, true, prep.rt_ready, dense_for_policy, tiles_total);
        if self.backend != Tier3Backend::Rt && prep.dense_tiles > 0 && !prep.rt_ready {
            tier.note_incomplete(&runtime);
        }
        Ok(())
    }

    fn execute(&mut self, ctx: &mut PassContext) -> HelioResult<()> {
        self.published = None;
        let Some(inputs) = ctx.registry.get::<VgCullInputs<'_>>(ResourceKey::new(KEY_INPUTS)) else {
            return Ok(());
        };
        let Some(cull) = ctx.registry.get::<VgCullFrameData<'_>>(ResourceKey::new(KEY_CULL)) else {
            return Ok(());
        };
        let Some(raster) = ctx.registry.get::<VgRasterFrameData<'_>>(ResourceKey::new(KEY_RASTER))
        else {
            return Ok(());
        };
        let mut backend = self.backend;
        if backend == Tier3Backend::Off {
            return Ok(());
        }
        let cam = ctx.camera_data;
        let frame = DenseFrame {
            view_proj: column_major(&cam.view_proj),
            inv_view_proj: column_major(&cam.inv_view_proj),
            camera_pos: [cam.position_near[0], cam.position_near[1], cam.position_near[2]],
            viewport: [raster.width, raster.height],
            visible_capacity: cull.bin_capacity * BIN_COUNT,
            ray_t: [cam.position_near[3], 1.0e30],
        };
        let mut nonrt_inputs = DenseNonRtInputs {
            tile_mask: raster.dense_mask,
            cull_state: cull.state,
            visible: cull.visible_clusters,
            page_pool: inputs.stream.page_pool,
            asset_quant: inputs.stream.asset_quant,
            vg_instances: inputs.instances,
            objects: inputs.objects,
        };
        // Tiles to resolve: this frame's classification, the mask the cull routed
        // clusters by, and the footprint of the dense-bin clusters (see
        // `mask_union.rs`: `raster.dense_mask` alone alternates ON/OFF and misses
        // the pixels of diverted clusters outside the dense tiles).
        let union_mask: wgpu::Buffer = {
            let mut urec = ctx.graphics_cmds();
            let last = inputs.dense_mask.unwrap_or(raster.dense_mask);
            self.mask_union
                .record(ctx.device, ctx.queue, &mut urec, &frame, &nonrt_inputs, raster.dense_mask, last)
                .clone()
        };
        nonrt_inputs.tile_mask = &union_mask;
        // Alpha-masked clusters (cull bins 6 and 7) need the material tables: the
        // masked peel evaluates the alpha test per fragment.
        let masked_tables = match (
            ctx.scene_buffers.get(helio_core::BufferKey::of("materials")),
            ctx.registry.get::<helio_mats::MaterialTextureBindings<'_>>(helio_core::resource_keys::material_textures()),
        ) {
            (Some(materials), Some(textures)) => Some((materials, textures)),
            _ => None,
        };
        self.nonrt.set_masked(masked_tables.as_ref().map(|(m, t)| (&m.buffer, *t)));
        let mut rec = ctx.graphics_cmds();
        let tier_on = self.tier_active && self.tier.is_some();
        if tier_on {
            // Cheap front half first: compaction + args give `tile_slot` and
            // the bin ranges, which the candidate collection needs whichever
            // backend draws.
            self.nonrt.record_setup_rec(ctx.device, ctx.queue, &mut rec, &frame, &nonrt_inputs);
            if let (Some(tier), Some(dense)) = (self.tier.as_mut(), self.nonrt.buffers()) {
                tier.record_collect(ctx.device, ctx.queue, &mut rec, dense, &nonrt_inputs, inputs.instance_count);
            }
            if backend == Tier3Backend::Rt {
                let ok = self
                    .tier
                    .as_mut()
                    .map(|t| t.record_rt(ctx.device, ctx.queue, &mut rec, &frame, &union_mask))
                    .is_some_and(|r| match r {
                        Ok(()) => true,
                        Err(e) => {
                            log::warn!("virtual geometry: ray-traced Tier 3 failed this frame ({e}); using the depth peel");
                            false
                        }
                    });
                if !ok {
                    backend = Tier3Backend::NonRt;
                }
            }
            if backend == Tier3Backend::NonRt {
                self.nonrt.record_work_rec(ctx.device, ctx.queue, &mut rec, &frame, &nonrt_inputs);
            }
        } else {
            self.nonrt.record_rec(ctx.device, ctx.queue, &mut rec, &frame, &nonrt_inputs);
        }
        drop(rec);
        {
            let counters = match backend {
                Tier3Backend::Rt => self.tier.as_ref().and_then(|t| t.rt().buffers()),
                _ => self.nonrt.buffers(),
            };
            if let Some(counters) = counters {
                let mut srec = ctx.graphics_cmds();
                self.stats_rb.record(&mut srec, counters);
            }
        }

        let published = match backend {
            Tier3Backend::Rt => self.tier.as_ref().and_then(|t| t.outputs()).map(|(key, bary, visible)| Published {
                key,
                mask: union_mask.clone(),
                bary: Some(bary),
                visible: Some(visible),
            }),
            _ => self.nonrt.buffers().map(|b| Published {
                key: b.dense_key.clone(),
                mask: union_mask.clone(),
                bary: None,
                visible: None,
            }),
        };
        if let Some(runtime) = &self.runtime {
            runtime.update_tier3_stats(|s| match backend {
                Tier3Backend::Rt => s.frames_rt += 1,
                _ => s.frames_nonrt += 1,
            });
        }
        self.published = published;
        Ok(())
    }

    fn publish<'a>(&self, frame: &mut helio_core::ResourceRegistry<'a>) {
        if let Some(p) = &self.published {
            // SAFETY: the clones live in `self`, which outlives the frame registry
            // and are only replaced in `execute`.
            let (dense_key, tile_mask, rt_bary, rt_visible): (
                &'a wgpu::Buffer,
                &'a wgpu::Buffer,
                Option<&'a wgpu::Buffer>,
                Option<&'a wgpu::Buffer>,
            ) = unsafe {
                (
                    std::mem::transmute(&p.key),
                    std::mem::transmute(&p.mask),
                    p.bary.as_ref().map(|b| std::mem::transmute(b)),
                    p.visible.as_ref().map(|b| std::mem::transmute(b)),
                )
            };
            frame.write(
                ResourceKey::new(KEY_DENSE),
                VgDenseFrameData { dense_key, tile_mask, rt_bary, rt_visible },
                "VgDense",
            );
        }
    }
}
