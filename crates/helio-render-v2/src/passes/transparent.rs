//! Forward transparent pass for alpha-blended materials.
//!
//! Draws objects flagged with `transparent_blend` after deferred lighting,
//! using the same PBR shader as forward geometry (`geometry.wgsl`) so lights,
//! shadows, environment and GI all affect transparent surfaces.
//!
//! ## Bundling strategy
//!
//! Transparent draws are compiled into a `RenderBundle` keyed on
//! `draw_list_generation` and the bound bind-group identities.  The bundle is
//! rebuilt once when the scene changes (chunk added/removed) using the camera
//! position at that moment for back-to-front sorting.
//!
//! **Camera movement** reuses the cached bundle at zero CPU cost — no re-sort,
//! no re-record.  The sort order can be one scene-change behind the camera,
//! which is imperceptible for large voxel water/glass faces.
//!
//! ## Why not per-frame resort
//!
//! At 3 000+ transparent draws, per-frame back-to-front sorting + inline
//! recording of ~16 K primary-encoder commands was the dominant contributor
//! to `finish_ms`.  The bundle amortises that cost to the scene-change frame
//! only.

use std::sync::{Arc, Mutex};

use crate::graph::{PassContext, PassResourceBuilder, RenderPass, ResourceHandle};
use crate::mesh::{DrawCall, INSTANCE_STRIDE};
use crate::Result;

pub struct TransparentPass {
    pipeline:  Arc<wgpu::RenderPipeline>,
    device:    Arc<wgpu::Device>,
    draw_list: Arc<Mutex<Vec<DrawCall>>>,
    /// Pre-compiled bundle.  Replayed each frame until the scene changes.
    cached_bundle: Option<wgpu::RenderBundle>,
    /// Generation and bind-group identities embedded in `cached_bundle`.
    bundle_gen:          u64,
    bundle_global_bg_ptr:   usize,
    bundle_lighting_bg_ptr: usize,
    /// Camera position used for the sort baked into `cached_bundle`.
    bundle_cam_pos: glam::Vec3,
    bundle_cam_fwd: glam::Vec3,
}

impl TransparentPass {
    pub fn new(
        pipeline:  Arc<wgpu::RenderPipeline>,
        draw_list: Arc<Mutex<Vec<DrawCall>>>,
        device:    Arc<wgpu::Device>,
    ) -> Self {
        Self {
            pipeline,
            device,
            draw_list,
            cached_bundle: None,
            bundle_gen: u64::MAX,
            bundle_global_bg_ptr: 0,
            bundle_lighting_bg_ptr: 0,
            bundle_cam_pos: glam::Vec3::ZERO,
            bundle_cam_fwd: glam::Vec3::NEG_Z,
        }
    }

    fn compile_bundle(
        &mut self,
        draw_calls: &[DrawCall],
        transparent_start: usize,
        global_bg: &wgpu::BindGroup,
        lighting_bg: &wgpu::BindGroup,
        cam: glam::Vec3,
        fwd: glam::Vec3,
        generation: u64,
    ) {
        let t = std::time::Instant::now();

        // Collect transparent indices.
        let ts = transparent_start.min(draw_calls.len());
        let mut indices: Vec<usize> = (ts..draw_calls.len()).collect();

        // Back-to-front sort from the current camera position.
        indices.sort_unstable_by(|&ia, &ib| {
            let da = (glam::Vec3::from(draw_calls[ia].bounds_center) - cam).dot(fwd);
            let db = (glam::Vec3::from(draw_calls[ib].bounds_center) - cam).dot(fwd);
            db.partial_cmp(&da)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| {
                    let pa = Arc::as_ptr(&draw_calls[ia].vertex_buffer) as usize;
                    let pb = Arc::as_ptr(&draw_calls[ib].vertex_buffer) as usize;
                    pa.cmp(&pb)
                })
        });

        if indices.is_empty() {
            self.cached_bundle         = None;
            self.bundle_gen            = generation;
            self.bundle_global_bg_ptr  = global_bg   as *const _ as usize;
            self.bundle_lighting_bg_ptr = lighting_bg as *const _ as usize;
            self.bundle_cam_pos = cam;
            self.bundle_cam_fwd = fwd;
            return;
        }

        // Encode into a RenderBundle (format-agnostic — embed commands, not attachments).
        // The actual color format is specified at replay time via begin_render_pass.
        let mut enc = self.device.create_render_bundle_encoder(
            &wgpu::RenderBundleEncoderDescriptor {
                label: Some("transparent_bundle"),
                // Use Rgba16Float as the intermediate format — matches the pre-AA target.
                // If this renderer uses the surface format directly (no AA), the bundle
                // still works because RenderBundle color formats must match the pass they
                // execute in.  We set a broad superset here; mismatches would panic at
                // runtime on the first compile so they'd be caught immediately.
                color_formats: &[Some(wgpu::TextureFormat::Bgra8UnormSrgb)],
                depth_stencil: Some(wgpu::RenderBundleDepthStencil {
                    format: wgpu::TextureFormat::Depth32Float,
                    depth_read_only: false,
                    stencil_read_only: true,
                }),
                sample_count: 1,
                multiview: None,
            },
        );

        enc.set_pipeline(&self.pipeline);
        enc.set_bind_group(0, global_bg, &[]);
        enc.set_bind_group(2, lighting_bg, &[]);

        let mut last_material: Option<usize> = None;
        for &idx in &indices {
            let dc = &draw_calls[idx];
            let mat_ptr = Arc::as_ptr(&dc.material_bind_group) as usize;
            if last_material != Some(mat_ptr) {
                enc.set_bind_group(1, Some(dc.material_bind_group.as_ref()), &[]);
                last_material = Some(mat_ptr);
            }
            enc.set_vertex_buffer(0, dc.vertex_buffer.slice(..));
            let inst_start = dc.instance_buffer_offset;
            let inst_end   = inst_start + dc.instance_count as u64 * INSTANCE_STRIDE;
            enc.set_vertex_buffer(1, dc.instance_buffer.as_ref().unwrap().slice(inst_start..inst_end));
            enc.set_index_buffer(dc.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
            enc.draw_indexed(0..dc.index_count, 0, 0..dc.instance_count);
        }

        self.cached_bundle = Some(enc.finish(&wgpu::RenderBundleDescriptor {
            label: Some("transparent_bundle"),
        }));
        self.bundle_gen            = generation;
        self.bundle_global_bg_ptr  = global_bg   as *const _ as usize;
        self.bundle_lighting_bg_ptr = lighting_bg as *const _ as usize;
        self.bundle_cam_pos = cam;
        self.bundle_cam_fwd = fwd;

        let compile_ms = t.elapsed().as_secs_f32() * 1000.0;
        if compile_ms > 0.5 {
            eprintln!(
                "⚠️ [Transparent] Bundle compiled: {} draws — {:.2}ms (gen {})",
                indices.len(), compile_ms, generation,
            );
        }
    }
}

impl RenderPass for TransparentPass {
    fn name(&self) -> &str { "transparent" }

    fn declare_resources(&self, builder: &mut PassResourceBuilder) {
        builder.read(ResourceHandle::named("color_target"));
        builder.write(ResourceHandle::named("color_target"));
        builder.read(ResourceHandle::named("gbuffer"));
        builder.write(ResourceHandle::named("transparent_done"));
    }

    fn execute(&mut self, ctx: &mut PassContext) -> Result<()> {
        let global_bg_ptr   = ctx.global_bind_group   as *const _ as usize;
        let lighting_bg_ptr = ctx.lighting_bind_group as *const _ as usize;

        // Recompile only when scene composition or embedded bind groups change.
        // Camera movement reuses the cached bundle — zero overhead while flying.
        let need_rebuild = ctx.draw_list_generation != self.bundle_gen
            || global_bg_ptr   != self.bundle_global_bg_ptr
            || lighting_bg_ptr != self.bundle_lighting_bg_ptr;

        if need_rebuild {
            let draw_calls: Vec<DrawCall> = self.draw_list.lock().unwrap().clone();
            self.compile_bundle(
                &draw_calls,
                ctx.transparent_start,
                ctx.global_bind_group,
                ctx.lighting_bind_group,
                ctx.camera_position,
                ctx.camera_forward,
                ctx.draw_list_generation,
            );
        }

        let Some(bundle) = &self.cached_bundle else { return Ok(()); };

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

        pass.execute_bundles(std::slice::from_ref(bundle));

        Ok(())
    }
}
