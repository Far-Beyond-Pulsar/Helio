//! Render-graph wrapper of Tier 3: [`VgDensePass`] re-resolves the dense tiles
//! the raster classified, with the 3-NonRT depth peel.
//!
//! The 3-RT backend ([`crate::DenseRtPass`]) is not driven from the graph: it
//! needs a CPU list of instances with proxy-LOD assets
//! ([`crate::RtInstanceRequest`], [`crate::ProxyAsset`]), and the graph only
//! has GPU-resident instance rows. [`Tier3Selector`] is still consulted, with
//! the RT inputs reported as unavailable, so the policy for `ForceHwRaster`
//! (Tier 3 off) and `ForceNonRtFallback` is exactly the documented one.

use helio_core::graph::ResourceBuilder;
use helio_core::{PassContext, RenderPass, ResourceKey, Result as HelioResult};
use helio_pass_vg_cull::{
    VgCullFrameData, VgCullInputs, BIN_COUNT, KEY_CULL, KEY_INPUTS,
};
use helio_pass_vg_raster::{VgRasterFrameData, KEY_RASTER};
use helio_vg_format::VgMode;

use crate::{
    AutoPolicy, DenseConfig, DenseError, DenseFrame, DenseNonRtInputs, DenseNonRtPass, Tier3Backend,
    Tier3Selector,
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

/// Tier 3 as a graph pass.
pub struct VgDensePass {
    nonrt: DenseNonRtPass,
    selector: Tier3Selector,
    mode: Box<dyn Fn() -> VgMode + Send + Sync>,
    published: Option<(wgpu::Buffer, wgpu::Buffer)>,
}

impl VgDensePass {
    /// `mode` is read every frame (the project setting).
    pub fn new(
        device: &wgpu::Device,
        mode: Box<dyn Fn() -> VgMode + Send + Sync>,
    ) -> Result<Self, DenseError> {
        let cfg = DenseConfig::for_workgroup_storage(device.limits().max_compute_workgroup_storage_size);
        Ok(Self {
            nonrt: DenseNonRtPass::new(device, cfg)?,
            selector: Tier3Selector::new(AutoPolicy::default()),
            mode,
            published: None,
        })
    }
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
        let tiles_total = raster.tiles[0] * raster.tiles[1];
        let backend = self.selector.select((self.mode)(), false, false, 0, tiles_total);
        if backend != Tier3Backend::NonRt {
            return Ok(());
        }
        let cam = ctx.camera_data;
        let column_major = |m: &[f32; 16]| {
            let mut out = [[0.0f32; 4]; 4];
            for (c, col) in out.iter_mut().enumerate() {
                col.copy_from_slice(&m[c * 4..c * 4 + 4]);
            }
            out
        };
        let frame = DenseFrame {
            view_proj: column_major(&cam.view_proj),
            inv_view_proj: column_major(&cam.inv_view_proj),
            camera_pos: [cam.position_near[0], cam.position_near[1], cam.position_near[2]],
            viewport: [raster.width, raster.height],
            visible_capacity: cull.bin_capacity * BIN_COUNT,
            ray_t: [cam.position_near[3], 1.0e30],
        };
        let nonrt_inputs = DenseNonRtInputs {
            tile_mask: raster.dense_mask,
            cull_state: cull.state,
            visible: cull.visible_clusters,
            page_pool: inputs.stream.page_pool,
            asset_quant: inputs.stream.asset_quant,
            vg_instances: inputs.instances,
            objects: inputs.objects,
        };
        let mut rec = ctx.graphics_cmds();
        self.nonrt.record_rec(ctx.device, ctx.queue, &mut rec, &frame, &nonrt_inputs);
        if let Some(buffers) = self.nonrt.buffers() {
            self.published = Some((buffers.dense_key.clone(), raster.dense_mask.clone()));
        }
        Ok(())
    }

    fn publish<'a>(&self, frame: &mut helio_core::ResourceRegistry<'a>) {
        if let Some((key, mask)) = &self.published {
            // SAFETY: the clones live in `self`, which outlives the frame registry
            // and are only replaced in `execute`.
            let (dense_key, tile_mask): (&'a wgpu::Buffer, &'a wgpu::Buffer) =
                unsafe { (std::mem::transmute(key), std::mem::transmute(mask)) };
            frame.write(
                ResourceKey::new(KEY_DENSE),
                VgDenseFrameData { dense_key, tile_mask, rt_bary: None, rt_visible: None },
                "VgDense",
            );
        }
    }
}
