//! Derived opaque-triangle acceleration cache for the experimental RT path.
use super::Scene;
use crate::{handles::MeshId, mesh::MeshKind};
use helio_core::{BlasGeometry, Error, Result, TlasInstanceInput};
use std::collections::HashSet;

impl Scene {
    /// Mark an adopted mesh changed after an external writer edits its GPU pool
    /// range in place. Dynamic mesh updates through Scene do this automatically.
    pub fn invalidate_ray_tracing_mesh(&mut self, mesh: MeshId) -> super::errors::Result<()> {
        let record = self
            .mesh_pool
            .get_mut(mesh)
            .ok_or_else(|| super::errors::invalid("mesh"))?;
        record.revision = record.revision.wrapping_add(1);
        Ok(())
    }

    /// Build the complete opaque caster set, including off-screen objects.
    /// This initial control rebuilds the TLAS each frame and scans objects on the
    /// CPU. Builds are submitted without a host wait; their cost is additional
    /// to graph pass timings. Unsupported geometry fails explicitly.
    pub fn prepare_ray_tracing(&mut self) -> Result<()> {
        let error = |message: &str| Error::InvalidPassConfig(format!("Scene RT: {message}"));
        let mut encoder =
            self.gpu_scene
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("scene ray geometry"),
                });
        let result = (|| {
            if self.vg_objects.len() != 0
                || self.voxel_volumes.len() != 0
                || self.foliage_layers.len() != 0
            {
                return Err(error("virtual geometry, voxels and foliage are not supported by the opaque RT control"));
            }
            let static_buffers = self.mesh_pool.buffers();
            let dynamic_buffers = self.mesh_pool.dynamic_buffers();
            let mut live = HashSet::new();
            let mut instances = Vec::new();
            for (_, object) in self.objects.iter_with_handles() {
                if !super::helpers::object_is_visible(object.groups, self.group_hidden)
                    || object.instance.flags & libhelio::INSTANCE_FLAG_CASTS_SHADOW == 0
                {
                    continue;
                }
                if libhelio::coordinate_space(object.instance.flags) != 0 {
                    return Err(error(
                        "non-world coordinate-space casters are not supported yet",
                    ));
                }
                let material = self
                    .materials
                    .get(object.material)
                    .ok_or_else(|| error("missing caster material"))?;
                if material.gpu.flags
                    & (libhelio::FLAG_ALPHA_BLEND
                        | libhelio::FLAG_ALPHA_TEST
                        | libhelio::FLAG_HAS_CUSTOM_SHADER)
                    != 0
                {
                    return Err(error("masked, transparent and custom-shader casters require a separate RT material path"));
                }
                let mesh = self
                    .mesh_pool
                    .get(object.mesh)
                    .ok_or_else(|| error("missing caster mesh"))?;
                let id = ((object.mesh.generation() as u64) << 32) | object.mesh.slot() as u64;
                if live.insert(id) {
                    let buffers = match mesh.kind {
                        MeshKind::Static => &static_buffers,
                        MeshKind::Dynamic => &dynamic_buffers,
                    };
                    self.gpu_scene
                        .blas_manager
                        .build_from_buffers(
                            id,
                            &mut encoder,
                            BlasGeometry {
                                revision: mesh.revision,
                                vertices: &buffers.vertices,
                                first_vertex: mesh.slice.first_vertex,
                                vertex_count: mesh.slice.vertex_count,
                                vertex_stride: std::mem::size_of::<crate::PackedVertex>() as u64,
                                indices: Some(&buffers.indices),
                                first_index: mesh.slice.first_index,
                                index_count: mesh.slice.index_count,
                            },
                        )
                        .map_err(|e| error(&e.to_string()))?;
                }
                let m = object.instance.model;
                instances.push(TlasInstanceInput {
                    mesh_id: id,
                    transform: [
                        m[0], m[4], m[8], m[12], m[1], m[5], m[9], m[13], m[2], m[6], m[10], m[14],
                    ],
                });
            }
            self.gpu_scene.blas_manager.retain(|id| live.contains(&id));
            self.gpu_scene
                .tlas_manager
                .build(&mut encoder, &instances, &self.gpu_scene.blas_manager)
                .map_err(|e| error(&e.to_string()))
        })();
        if result.is_err() {
            // An abandoned encoder must not leave an apparently built cache.
            self.gpu_scene.blas_manager.clear();
            self.gpu_scene.tlas_manager.invalidate();
            return result;
        }
        self.gpu_scene.queue.submit([encoder.finish()]);
        Ok(())
    }
}
