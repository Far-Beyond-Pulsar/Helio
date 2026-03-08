//! Forward transparent pass for alpha-blended materials.
//!
//! Draws objects flagged with `transparent_blend` after deferred lighting,
//! using the same PBR shader as forward geometry (`geometry.wgsl`) so lights,
//! shadows, environment and GI all affect transparent surfaces.

use std::sync::{Arc, Mutex};

use crate::graph::{PassContext, PassResourceBuilder, RenderPass, ResourceHandle};
use crate::mesh::{DrawCall, INSTANCE_STRIDE};
use crate::Result;

pub struct TransparentPass {
    device: Arc<wgpu::Device>,
    pipeline: Arc<wgpu::RenderPipeline>,
    draw_list: Arc<Mutex<Vec<DrawCall>>>,
    /// Surface format needed to create the RenderBundleEncoder with the correct
    /// color attachment format (must match the actual render pass target).
    surface_format: wgpu::TextureFormat,
    sorted_transparent_indices: Vec<usize>,
    /// Camera position and forward at the time of the last sort.
    last_sort_cam_pos: glam::Vec3,
    last_sort_cam_fwd: glam::Vec3,
    /// draw_list_generation value when sorted_transparent_indices was last computed.
    last_sort_generation: u64,
    /// Cached RenderBundle containing all back-to-front transparent draw commands.
    ///
    /// Recording 989 per-draw commands (set_bind_group, set_vertex_buffer×2,
    /// set_index_buffer, draw_indexed) directly into the main CommandEncoder makes
    /// `encoder.finish()` O(N draws) — measured at 27 ms for ~1000 transparent draws.
    /// Pre-compiling them into a RenderBundle reduces the main encoder to a single
    /// `execute_bundles` command, making `encoder.finish()` O(1) for this pass.
    ///
    /// The bundle is rebuilt only when the sort order changes (camera moved or draw
    /// list topology changed) or when the captured bind group objects are replaced.
    bundle_cache: Option<wgpu::RenderBundle>,
    cached_bundle_generation: u64,
    cached_bundle_cam_pos: glam::Vec3,
    cached_bundle_cam_fwd: glam::Vec3,
    /// Raw pointer identity guards — if the renderer replaces `global_bind_group`
    /// or `lighting_bind_group` (e.g. when light-buffer capacity grows) the stale
    /// references captured inside the bundle must be refreshed.
    cached_global_bg_ptr: usize,
    cached_lighting_bg_ptr: usize,
}

impl TransparentPass {
    pub fn new(
        device: Arc<wgpu::Device>,
        pipeline: Arc<wgpu::RenderPipeline>,
        draw_list: Arc<Mutex<Vec<DrawCall>>>,
        surface_format: wgpu::TextureFormat,
    ) -> Self {
        Self {
            device,
            pipeline,
            draw_list,
            surface_format,
            sorted_transparent_indices: Vec::new(),
            last_sort_cam_pos: glam::Vec3::ZERO,
            last_sort_cam_fwd: glam::Vec3::NEG_Z,
            last_sort_generation: u64::MAX,
            bundle_cache: None,
            cached_bundle_generation: u64::MAX,
            cached_bundle_cam_pos: glam::Vec3::ZERO,
            cached_bundle_cam_fwd: glam::Vec3::NEG_Z,
            cached_global_bg_ptr: 0,
            cached_lighting_bg_ptr: 0,
        }
    }
}

impl RenderPass for TransparentPass {
    fn name(&self) -> &str {
        "transparent"
    }

    fn declare_resources(&self, builder: &mut PassResourceBuilder) {
        builder.read(ResourceHandle::named("color_target"));
        builder.write(ResourceHandle::named("color_target"));
        builder.read(ResourceHandle::named("gbuffer"));
        // Ordering token consumed by overlay passes (billboards/debug overlays)
        // to ensure they execute after transparent composition.
        builder.write(ResourceHandle::named("transparent_done"));
    }

    fn execute(&mut self, ctx: &mut PassContext) -> Result<()> {
        let draw_calls = self.draw_list.lock().unwrap();

        // Skip re-sort when neither the draw list nor the camera has changed.
        // Back-to-front sort on 1000+ indices runs ~0.6 ms; skipping it on
        // static frames cuts per-frame CPU overhead significantly.
        let cam = ctx.camera_position;
        let fwd = ctx.camera_forward;
        let cam_moved = (cam - self.last_sort_cam_pos).length_squared() > 0.01
            || (fwd - self.last_sort_cam_fwd).length_squared() > 0.0001;
        let need_sort = ctx.draw_list_generation != self.last_sort_generation || cam_moved;

        if need_sort {
            self.sorted_transparent_indices.clear();
            self.sorted_transparent_indices.reserve(draw_calls.len());
            for (idx, dc) in draw_calls.iter().enumerate() {
                if dc.transparent_blend {
                    self.sorted_transparent_indices.push(idx);
                }
            }

            if self.sorted_transparent_indices.is_empty() {
                self.last_sort_cam_pos = cam;
                self.last_sort_cam_fwd = fwd;
                self.last_sort_generation = ctx.draw_list_generation;
                return Ok(());
            }

            // Back-to-front for standard alpha blending using view depth.
            self.sorted_transparent_indices.sort_by(|&ia, &ib| {
                let a = &draw_calls[ia];
                let b = &draw_calls[ib];
                let oa = glam::Vec3::from(a.bounds_center) - cam;
                let ob = glam::Vec3::from(b.bounds_center) - cam;

                let da = oa.dot(fwd);
                let db = ob.dot(fwd);

                db.partial_cmp(&da)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    // Deterministic fallback when depths match closely.
                    .then_with(|| {
                        let pa = Arc::as_ptr(&a.vertex_buffer) as usize;
                        let pb = Arc::as_ptr(&b.vertex_buffer) as usize;
                        pa.cmp(&pb)
                    })
            });

            self.last_sort_cam_pos = cam;
            self.last_sort_cam_fwd = fwd;
            self.last_sort_generation = ctx.draw_list_generation;
        }

        if self.sorted_transparent_indices.is_empty() {
            return Ok(());
        }

        // Rebuild the RenderBundle when the sort order changed or when the bind
        // group objects themselves were replaced (e.g. light buffer grew).
        // Recording draws into a bundle instead of the main CommandEncoder means
        // encoder.finish() sees a single execute_bundles command instead of
        // N×5 individual state-change commands, dropping finish() from O(N) to O(1).
        let global_bg_ptr  = ctx.global_bind_group  as *const _ as usize;
        let lighting_bg_ptr = ctx.lighting_bind_group as *const _ as usize;
        let need_bundle = need_sort
            || self.bundle_cache.is_none()
            || global_bg_ptr  != self.cached_global_bg_ptr
            || lighting_bg_ptr != self.cached_lighting_bg_ptr;

        if need_bundle {
            let mut benc = self.device.create_render_bundle_encoder(
                &wgpu::RenderBundleEncoderDescriptor {
                    label: Some("transparent_bundle"),
                    // Must match the render pass color attachment format (surface_format
                    // is used for both the pre-AA intermediate texture and the direct
                    // swapchain target, so one format covers all AA modes).
                    color_formats: &[Some(self.surface_format)],
                    depth_stencil: Some(wgpu::RenderBundleDepthStencil {
                        format: wgpu::TextureFormat::Depth32Float,
                        // Transparent objects depth-test but do not write depth
                        // (pipeline has depth_write_enabled=false), however the outer
                        // render pass stores depth (StoreOp::Store), so the render pass
                        // itself is not read-only — keep depth_read_only=false to match.
                        depth_read_only: false,
                        stencil_read_only: true,
                    }),
                    sample_count: 1,
                    multiview: None,
                },
            );

            benc.set_pipeline(&self.pipeline);
            // All three bind groups must be set inside the bundle; groups set on
            // the outer RenderPass are not inherited by execute_bundles.
            benc.set_bind_group(0, ctx.global_bind_group, &[]);
            benc.set_bind_group(2, ctx.lighting_bind_group, &[]);

            let mut last_material: Option<usize> = None;
            for &idx in &self.sorted_transparent_indices {
                let dc = &draw_calls[idx];
                let mat_ptr = Arc::as_ptr(&dc.material_bind_group) as usize;
                if last_material != Some(mat_ptr) {
                    benc.set_bind_group(1, Some(dc.material_bind_group.as_ref()), &[]);
                    last_material = Some(mat_ptr);
                }
                benc.set_vertex_buffer(0, dc.vertex_buffer.slice(..));
                let inst_start = dc.instance_buffer_offset;
                let inst_end   = inst_start + dc.instance_count as u64 * INSTANCE_STRIDE;
                benc.set_vertex_buffer(1, dc.instance_buffer.as_ref().unwrap().slice(inst_start..inst_end));
                benc.set_index_buffer(dc.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
                benc.draw_indexed(0..dc.index_count, 0, 0..dc.instance_count);
            }

            self.bundle_cache = Some(benc.finish(&wgpu::RenderBundleDescriptor { label: None }));
            self.cached_bundle_generation = ctx.draw_list_generation;
            self.cached_bundle_cam_pos  = cam;
            self.cached_bundle_cam_fwd  = fwd;
            self.cached_global_bg_ptr   = global_bg_ptr;
            self.cached_lighting_bg_ptr = lighting_bg_ptr;
        }

        drop(draw_calls);

        let mut pass = ctx.encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Transparent Pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: ctx.target,
                resolve_target: None,
                depth_slice: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: ctx.depth_view,
                // Transparent pass must still depth-test against opaque depth.
                // We load existing depth and keep it unchanged (pipeline has
                // depth_write_enabled = false).
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });

        if let Some(bundle) = &self.bundle_cache {
            pass.execute_bundles(std::iter::once(bundle));
        }

        Ok(())
    }
}
