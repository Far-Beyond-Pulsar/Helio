//! Radiance Cascades pass – builds BLAS/TLAS and dispatches the rc_trace shader
//!
//! Runs once per frame, cascades 4→0 (coarse to fine so each cascade can
//! read from its already-computed parent).

use crate::features::radiance_cascades::{CASCADE_COUNT, PROBE_DIMS, DIR_DIMS, ATLAS_HEIGHTS};
use crate::graph::{RenderPass, PassContext, PassResourceBuilder, ResourceHandle};
use crate::mesh::DrawCall;
use crate::Result;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// Cached BLAS entry for a single mesh
struct BlasEntry {
    blas: wgpu::Blas,
    size_desc: wgpu::BlasTriangleGeometrySizeDescriptor,
    vertex_buffer: Arc<wgpu::Buffer>,
    index_buffer: Arc<wgpu::Buffer>,
}

/// Row-major identity transform for TlasInstance  ([f32; 12] = 3×4 matrix)
const IDENTITY_TRANSFORM: [f32; 12] = [
    1.0, 0.0, 0.0, 0.0,
    0.0, 1.0, 0.0, 0.0,
    0.0, 0.0, 1.0, 0.0,
];

pub struct RadianceCascadesPass {
    device: Arc<wgpu::Device>,
    draw_list: Arc<Mutex<Vec<DrawCall>>>,
    pipeline: Arc<wgpu::ComputePipeline>,
    /// One bind group per cascade (index == cascade level)
    bind_groups: Vec<wgpu::BindGroup>,
    rc_dynamic_buf: Arc<wgpu::Buffer>,
    tlas: wgpu::Tlas,
    blas_cache: HashMap<usize, BlasEntry>,
    world_min: [f32; 3],
    world_max: [f32; 3],
}

impl RadianceCascadesPass {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        device: Arc<wgpu::Device>,
        draw_list: Arc<Mutex<Vec<DrawCall>>>,
        pipeline: Arc<wgpu::ComputePipeline>,
        bind_groups: Vec<wgpu::BindGroup>,
        rc_dynamic_buf: Arc<wgpu::Buffer>,
        tlas: wgpu::Tlas,
        world_min: [f32; 3],
        world_max: [f32; 3],
    ) -> Self {
        Self {
            device,
            draw_list,
            pipeline,
            bind_groups,
            rc_dynamic_buf,
            tlas,
            blas_cache: HashMap::new(),
            world_min,
            world_max,
        }
    }
}

impl RenderPass for RadianceCascadesPass {
    fn name(&self) -> &str { "radiance_cascades" }

    fn declare_resources(&self, builder: &mut PassResourceBuilder) {
        // Writes cascade 0 texture → geometry pass must read it → ordering enforced
        builder.write(ResourceHandle::named("rc_cascade0"));
    }

    fn execute(&mut self, ctx: &mut PassContext) -> Result<()> {
        let draw_calls: Vec<DrawCall> = self.draw_list.lock().unwrap().clone();
        if draw_calls.is_empty() {
            // No geometry → nothing to trace; textures stay at their initial state (zeros)
            return Ok(());
        }

        // ── 1. Create new BLASes for meshes not yet in the cache ──────────────
        let mut new_keys: Vec<usize> = Vec::new();
        for dc in &draw_calls {
            let key = Arc::as_ptr(&dc.vertex_buffer) as usize;
            if !self.blas_cache.contains_key(&key) {
                let size_desc = wgpu::BlasTriangleGeometrySizeDescriptor {
                    vertex_format: wgpu::VertexFormat::Float32x3,
                    vertex_count: dc.vertex_count,
                    index_format: Some(wgpu::IndexFormat::Uint32),
                    index_count: Some(dc.index_count),
                    flags: wgpu::AccelerationStructureGeometryFlags::OPAQUE,
                };
                let blas = self.device.create_blas(
                    &wgpu::CreateBlasDescriptor {
                        label: None,
                        flags: wgpu::AccelerationStructureFlags::PREFER_FAST_TRACE,
                        update_mode: wgpu::AccelerationStructureUpdateMode::Build,
                    },
                    wgpu::BlasGeometrySizeDescriptors::Triangles {
                        descriptors: vec![size_desc.clone()],
                    },
                );
                self.blas_cache.insert(key, BlasEntry {
                    blas,
                    size_desc,
                    vertex_buffer: dc.vertex_buffer.clone(),
                    index_buffer: dc.index_buffer.clone(),
                });
                new_keys.push(key);
            }
        }

        // ── 2. Set TLAS instances (all active draw calls) ─────────────────────
        for (i, dc) in draw_calls.iter().enumerate() {
            let key = Arc::as_ptr(&dc.vertex_buffer) as usize;
            let blas_ref = &self.blas_cache[&key].blas;
            self.tlas[i] = Some(wgpu::TlasInstance::new(blas_ref, IDENTITY_TRANSFORM, 0, 0xFF));
        }
        for i in draw_calls.len()..256 {
            self.tlas[i] = None;
        }

        // ── 3. Build all BLASes + TLAS ─────────────────────────────────────────
        // Collect build entries (borrows self.blas_cache + draw_calls until the call).
        // Build ALL cached BLASes each frame (simple; optimise to build-once later).
        let mut blas_entries: Vec<wgpu::BlasBuildEntry> = Vec::with_capacity(draw_calls.len());
        for dc in &draw_calls {
            let key = Arc::as_ptr(&dc.vertex_buffer) as usize;
            let entry = &self.blas_cache[&key];
            blas_entries.push(wgpu::BlasBuildEntry {
                blas: &entry.blas,
                geometry: wgpu::BlasGeometries::TriangleGeometries(vec![
                    wgpu::BlasTriangleGeometry {
                        size: &entry.size_desc,
                        vertex_buffer: &entry.vertex_buffer,
                        first_vertex: 0,
                        vertex_stride: 32, // PackedVertex is 32 bytes
                        index_buffer: Some(&entry.index_buffer),
                        first_index: Some(0),
                        transform_buffer: None,
                        transform_buffer_offset: None,
                    },
                ]),
            });
        }
        ctx.encoder.build_acceleration_structures(
            blas_entries.iter(),
            std::iter::once(&self.tlas),
        );

        // ── 4. Dispatch cascades coarse → fine (4 → 0) ───────────────────────
        let mut cpass = ctx.begin_compute_pass("RC Trace");
        cpass.set_pipeline(&self.pipeline);
        for c in (0..CASCADE_COUNT).rev() {
            let atlas_w = PROBE_DIMS[c] * DIR_DIMS[c]; // always 32
            let atlas_h = ATLAS_HEIGHTS[c];
            let dispatch_x = (atlas_w + 7) / 8;
            let dispatch_y = (atlas_h + 7) / 8;
            cpass.set_bind_group(0, &self.bind_groups[c], &[]);
            cpass.dispatch_workgroups(dispatch_x, dispatch_y, 1);
        }
        drop(cpass);

        Ok(())
    }
}
