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
    built: bool,  // Track whether this BLAS has been built
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
    /// Bind groups for even frames: read hist_a, write hist_b
    bind_groups_a: Vec<wgpu::BindGroup>,
    /// Bind groups for odd frames: read hist_b, write hist_a
    bind_groups_b: Vec<wgpu::BindGroup>,
    /// Output textures (what RC writes; geometry reads these)
    output_textures: Vec<Arc<wgpu::Texture>>,
    /// hist_b textures — kept alive here (hist_a lives in the feature struct)
    _hist_b_textures: Vec<Arc<wgpu::Texture>>,
    rc_dynamic_buf: Arc<wgpu::Buffer>,
    tlas: wgpu::Tlas,
    blas_cache: HashMap<usize, BlasEntry>,
    world_min: [f32; 3],
    world_max: [f32; 3],
    frame_idx: u64,
}

impl RadianceCascadesPass {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        device: Arc<wgpu::Device>,
        draw_list: Arc<Mutex<Vec<DrawCall>>>,
        pipeline: Arc<wgpu::ComputePipeline>,
        bind_groups_a: Vec<wgpu::BindGroup>,
        bind_groups_b: Vec<wgpu::BindGroup>,
        output_textures: Vec<Arc<wgpu::Texture>>,
        hist_b_textures: Vec<Arc<wgpu::Texture>>,
        rc_dynamic_buf: Arc<wgpu::Buffer>,
        tlas: wgpu::Tlas,
        world_min: [f32; 3],
        world_max: [f32; 3],
    ) -> Self {
        Self {
            device,
            draw_list,
            pipeline,
            bind_groups_a,
            bind_groups_b,
            output_textures,
            _hist_b_textures: hist_b_textures,
            rc_dynamic_buf,
            tlas,
            blas_cache: HashMap::new(),
            world_min,
            world_max,
            frame_idx: 0,
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
            log::warn!("RC: No draw calls! RC will output zeros");
            return Ok(());
        }
        log::trace!("RC: Executing with {} draw calls, bounds [{:?} .. {:?}]", 
            draw_calls.len(), self.world_min, self.world_max);
        
        // Note: light_count is uploaded via rc_dyn buffer in feature prepare,
        // so we can't easily log it here. The compute shader will use whatever
        // light count was set in the last prepare() call.

        // ── 1. Create new BLASes for meshes not yet in the cache ──────────────
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
                    built: false,  // Mark as not yet built
                });
            }
        }

        // ── 2. Set TLAS instances (all active draw calls, capped at TLAS capacity) ──
        const TLAS_MAX: usize = 2048;
        let active = draw_calls.len().min(TLAS_MAX);
        for (i, dc) in draw_calls[..active].iter().enumerate() {
            let key = Arc::as_ptr(&dc.vertex_buffer) as usize;
            let blas_ref = &self.blas_cache[&key].blas;
            self.tlas[i] = Some(wgpu::TlasInstance::new(blas_ref, IDENTITY_TRANSFORM, 0, 0xFF));
        }
        for i in active..TLAS_MAX {
            self.tlas[i] = None;
        }

        // ── 3. Build only unbuilt BLASes + TLAS ───────────────────────────────
        // Collect build entries only for BLASes that haven't been built yet.
        // Static geometry BLASes are built once and reused every frame.
        let mut blas_entries: Vec<wgpu::BlasBuildEntry> = Vec::new();
        let mut keys_to_mark_built: Vec<usize> = Vec::new();

        for dc in &draw_calls {
            let key = Arc::as_ptr(&dc.vertex_buffer) as usize;
            let entry = &self.blas_cache[&key];

            // Only build if not already built
            if !entry.built {
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
                keys_to_mark_built.push(key);
            }
        }

        // Build new BLASes and always rebuild TLAS (TLAS rebuild is cheap)
        let t_accel = ctx.scope_begin("rc/accel_build");
        ctx.encoder.build_acceleration_structures(
            blas_entries.iter(),
            std::iter::once(&self.tlas),
        );
        ctx.scope_end(t_accel);

        // Mark newly built BLASes as built
        for key in keys_to_mark_built {
            if let Some(entry) = self.blas_cache.get_mut(&key) {
                entry.built = true;
            }
        }

        // ── 4. Dispatch cascades coarse → fine (4 → 0) ───────────────────────
        // Select ping-pong bind group set: even frames use A (read hist_a, write hist_b),
        // odd frames use B (read hist_b, write hist_a). The shader writes history
        // directly via binding 7, so no copy_texture_to_texture is ever needed.
        let bind_groups = if self.frame_idx % 2 == 0 {
            &self.bind_groups_a
        } else {
            &self.bind_groups_b
        };
        self.frame_idx += 1;

        // One compute pass per cascade so each can be timed independently.
        for c in (0..CASCADE_COUNT).rev() {
            let atlas_w = PROBE_DIMS[c] * DIR_DIMS[c]; // always 32
            let atlas_h = ATLAS_HEIGHTS[c];
            let dispatch_x = (atlas_w + 7) / 8;
            let dispatch_y = (atlas_h + 7) / 8;

            let t_cas = ctx.scope_begin(&format!("rc/cascade_{c}"));
            let mut cpass = ctx.begin_compute_pass(&format!("RC Cascade {c}"));
            cpass.set_pipeline(&self.pipeline);
            cpass.set_bind_group(0, &bind_groups[c], &[]);
            cpass.dispatch_workgroups(dispatch_x, dispatch_y, 1);
            drop(cpass);
            ctx.scope_end(t_cas);
        }

        Ok(())
    }
}
