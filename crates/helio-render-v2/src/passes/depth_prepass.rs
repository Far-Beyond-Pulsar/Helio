//! Depth prepass - renders depth-only before GBuffer for early fragment rejection.
//!
//! This pass renders all opaque geometry to write depth without color output.
//! GBuffer later loads this depth and skips fragments already in shadow/occluded,
//! saving expensive shading work on invisible pixels (typically 50-80%).

use std::sync::{Arc, Mutex};
use crate::graph::{RenderPass, PassContext, PassResourceBuilder, ResourceHandle};
use crate::mesh::{DrawCall, INSTANCE_STRIDE};
use crate::Result;

/// Depth-only pass that writes depth for all opaque draw calls
pub struct DepthPrepassPass {
    device: Arc<wgpu::Device>,
    pipeline: Arc<wgpu::RenderPipeline>,
    draw_list: Arc<Mutex<Vec<DrawCall>>>,
    sorted_opaque_indices: Vec<usize>,
    /// Sorted opaque index list shared with GBufferPass so the front-to-back
    /// ordering is computed once here and read directly by GBuffer.
    pub shared_sorted: Arc<Mutex<Vec<usize>>>,
    /// Cached RenderBundle.  A bundle pre-compiles all draw commands so the
    /// main encoder only records a single execute_bundles command, reducing
    /// encoder.finish() cost from O(N draws) to O(1).
    bundle_cache: Option<wgpu::RenderBundle>,
    /// Generation counter when the cached bundle was built.  When it differs
    /// from PassContext::draw_list_generation the bundle is rebuilt.
    cached_generation: u64,
}

impl DepthPrepassPass {
    pub fn new(
        device: Arc<wgpu::Device>,
        pipeline: Arc<wgpu::RenderPipeline>,
        draw_list: Arc<Mutex<Vec<DrawCall>>>,
    ) -> (Self, Arc<Mutex<Vec<usize>>>) {
        let shared_sorted = Arc::new(Mutex::new(Vec::new()));
        let pass = Self {
            device,
            pipeline,
            draw_list,
            sorted_opaque_indices: Vec::new(),
            shared_sorted: shared_sorted.clone(),
            bundle_cache: None,
            cached_generation: u64::MAX,
        };
        (pass, shared_sorted)
    }
}

impl RenderPass for DepthPrepassPass {
    fn name(&self) -> &str { "depth_prepass" }

    fn declare_resources(&self, builder: &mut PassResourceBuilder) {
        // Depth prepass writes to the shared depth buffer; GBuffer will load it
        builder.write(ResourceHandle::named("depth"));
    }

    fn execute(&mut self, ctx: &mut PassContext) -> Result<()> {
        let draw_calls = self.draw_list.lock().unwrap();

        // Rebuild index list and bundle when the draw list content changes.
        // Sort by material bind-group pointer — camera-independent, so this
        // sort and the resulting bundle remain valid across all camera moves.
        // (Front-to-back depth sorting is unnecessary here: the depth prepass
        // has no fragment cost and modern GPU hardware early-Z handles rejection
        // automatically. The sort only needs to change when objects are
        // added/removed/swapped, which is captured by draw_list_generation.)
        if ctx.draw_list_generation != self.cached_generation {
            self.sorted_opaque_indices.clear();
            self.sorted_opaque_indices.reserve(draw_calls.len());
            for (idx, dc) in draw_calls.iter().enumerate() {
                if !dc.transparent_blend {
                    self.sorted_opaque_indices.push(idx);
                }
            }
            // Sort by material pointer to minimise bind-group state changes;
            // this ordering is stable with respect to the camera.
            self.sorted_opaque_indices.sort_unstable_by_key(|&i| {
                Arc::as_ptr(&draw_calls[i].material_bind_group) as usize
            });

            // Publish the stable order to GBufferPass.
            {
                let mut shared = self.shared_sorted.lock().unwrap();
                shared.clear();
                shared.extend_from_slice(&self.sorted_opaque_indices);
            }
        }

        // Rebuild RenderBundle when the draw list content changes.
        //
        // A RenderBundle pre-compiles all draw commands into a replayable object.
        // The main encoder records a single execute_bundles command per pass,
        // so encoder.finish() is O(passes) rather than O(N draws × cmds/draw).
        // Bind group data is still sourced from the live uniform buffers each
        // frame via write_buffer — the bundle captures the BindGroup *reference*,
        // not the buffer payload, so camera/light updates are reflected correctly.
        if ctx.draw_list_generation != self.cached_generation {
            let _bundle_t = std::time::Instant::now();
            let mut benc = self.device.create_render_bundle_encoder(
                &wgpu::RenderBundleEncoderDescriptor {
                    label: Some("depth_prepass_bundle"),
                    color_formats: &[],
                    depth_stencil: Some(wgpu::RenderBundleDepthStencil {
                        format: wgpu::TextureFormat::Depth32Float,
                        depth_read_only: false,
                        stencil_read_only: true,
                    }),
                    sample_count: 1,
                    multiview: None,
                },
            );
            benc.set_pipeline(&self.pipeline);
            benc.set_bind_group(0, ctx.global_bind_group, &[]);
            benc.set_bind_group(2, ctx.lighting_bind_group, &[]);
            let mut last_material: Option<usize> = None;
            for &idx in &self.sorted_opaque_indices {
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
            self.cached_generation = ctx.draw_list_generation;
            eprintln!(
                "⚠️ [DepthPrepass] Bundle rebuild: {} draw calls — {:.2}ms",
                self.sorted_opaque_indices.len(),
                _bundle_t.elapsed().as_secs_f32() * 1000.0,
            );
        }
        drop(draw_calls);

        let mut pass = ctx.encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Depth Prepass"),
            color_attachments: &[],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: ctx.depth_view,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(1.0),
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
