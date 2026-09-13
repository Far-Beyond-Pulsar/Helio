use std::collections::HashMap;
use std::sync::Arc;

/// Failures detected before recording an acceleration-structure build.
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum AccelerationError {
    #[error(
        "ray queries and nonzero acceleration-structure limits must be enabled on this device"
    )]
    Unsupported,
    #[error("invalid BLAS geometry: {0}")]
    InvalidGeometry(&'static str),
    #[error("TLAS references missing BLAS for mesh {0}")]
    MissingBlas(u64),
    #[error("TLAS instance count exceeds the device or host index limit")]
    TooManyInstances,
    #[error("TLAS instance transform contains a non-finite value")]
    InvalidTransform,
}

/// An opaque triangle mesh in an existing GPU allocation. Positions are Float32x3
/// at the start of each vertex; optional indices are mesh-local Uint32 values.
/// Increment `revision` after every in-place geometry write. Buffer replacement,
/// offsets and counts are also part of the cache key, independently of revision.
pub struct BlasGeometry<'a> {
    pub revision: u64,
    pub vertices: &'a wgpu::Buffer,
    pub first_vertex: u32,
    pub vertex_count: u32,
    pub vertex_stride: u64,
    pub indices: Option<&'a wgpu::Buffer>,
    pub first_index: u32,
    pub index_count: u32,
}

#[derive(PartialEq, Eq)]
struct GeometryKey {
    revision: u64,
    vertices: wgpu::Buffer,
    first_vertex: u32,
    vertex_count: u32,
    vertex_stride: u64,
    indices: Option<wgpu::Buffer>,
    first_index: u32,
    index_count: u32,
}

impl BlasGeometry<'_> {
    fn validate(&self) -> Result<(), AccelerationError> {
        use AccelerationError::InvalidGeometry as Invalid;
        if self.vertex_count == 0 || self.vertex_stride < 12 || self.vertex_stride % 4 != 0 {
            return Err(Invalid(
                "positions require vertices and a stride >= 12 divisible by 4",
            ));
        }
        if !self
            .vertices
            .usage()
            .intersects(wgpu::BufferUsages::BLAS_INPUT | wgpu::BufferUsages::COPY_SRC)
        {
            return Err(Invalid(
                "vertex buffer requires BLAS_INPUT or COPY_SRC usage",
            ));
        }
        let count = u64::from(self.first_vertex) + u64::from(self.vertex_count);
        let end = count.checked_mul(self.vertex_stride);
        if end.is_none_or(|end| end > self.vertices.size()) {
            return Err(Invalid("vertex range exceeds its buffer"));
        }
        if let Some(indices) = self.indices {
            if !indices
                .usage()
                .intersects(wgpu::BufferUsages::BLAS_INPUT | wgpu::BufferUsages::COPY_SRC)
            {
                return Err(Invalid(
                    "index buffer requires BLAS_INPUT or COPY_SRC usage",
                ));
            }
            if self.index_count == 0 || self.index_count % 3 != 0 {
                return Err(Invalid(
                    "indexed triangles require a positive multiple of three indices",
                ));
            }
            if (u64::from(self.first_index) + u64::from(self.index_count)) * 4 > indices.size() {
                return Err(Invalid("index range exceeds its buffer"));
            }
        } else if self.first_index != 0 || self.index_count != 0 || self.vertex_count % 3 != 0 {
            return Err(Invalid(
                "non-indexed triangles require a multiple of three vertices and no index range",
            ));
        }
        Ok(())
    }

    fn key(&self) -> GeometryKey {
        GeometryKey {
            revision: self.revision,
            vertices: self.vertices.clone(),
            first_vertex: self.first_vertex,
            vertex_count: self.vertex_count,
            vertex_stride: self.vertex_stride,
            indices: self.indices.cloned(),
            first_index: self.first_index,
            index_count: self.index_count,
        }
    }
}

/// Manages Bottom-Level Acceleration Structures (BLAS) for scene meshes.
pub struct BlasManager {
    blas_map: HashMap<u64, wgpu::Blas>,
    geometry_keys: HashMap<u64, GeometryKey>,
    device: Arc<wgpu::Device>,
    rt_available: bool,
}

impl BlasManager {
    pub fn new(device: Arc<wgpu::Device>) -> Self {
        let rt_available = device
            .features()
            .contains(wgpu::Features::EXPERIMENTAL_RAY_QUERY)
            && device.limits().max_blas_geometry_count > 0
            && device.limits().max_blas_primitive_count > 0;
        Self {
            blas_map: HashMap::new(),
            geometry_keys: HashMap::new(),
            device,
            rt_available,
        }
    }

    pub fn is_rt_available(&self) -> bool {
        self.rt_available
    }

    /// Record a versioned opaque BLAS build without CPU copies, submission or
    /// host waits. BLAS_INPUT allocations are used directly; COPY_SRC-only pools
    /// use a GPU copy into temporary build input. Returns whether a build was recorded.
    ///
    /// Submit this encoder before any TLAS/query consuming the result. If the
    /// encoder is abandoned, remove the affected mesh or clear the manager before
    /// attempting to reuse it. GPU-written indices must be within vertex_count;
    /// their contents cannot be validated here without a readback.
    pub fn build_from_buffers(
        &mut self,
        mesh_id: u64,
        encoder: &mut wgpu::CommandEncoder,
        geometry: BlasGeometry<'_>,
    ) -> Result<bool, AccelerationError> {
        if !self.rt_available {
            return Err(AccelerationError::Unsupported);
        }
        if let Err(error) = geometry.validate() {
            // An invalid replacement must not leave the old shape queryable.
            self.remove_blas(mesh_id);
            return Err(error);
        }
        let primitives = if geometry.indices.is_some() {
            geometry.index_count
        } else {
            geometry.vertex_count
        } / 3;
        if primitives > self.device.limits().max_blas_primitive_count {
            self.remove_blas(mesh_id);
            return Err(AccelerationError::InvalidGeometry(
                "triangle count exceeds the device limit",
            ));
        }
        let key = geometry.key();
        if self.geometry_keys.get(&mesh_id) == Some(&key) {
            return Ok(false);
        }
        let mut copy_input = |source: &wgpu::Buffer, offset: u64, size: u64| {
            if source.usage().contains(wgpu::BufferUsages::BLAS_INPUT) {
                return None;
            }
            let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("BLAS pooled geometry build input"),
                size,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::BLAS_INPUT,
                mapped_at_creation: false,
            });
            encoder.copy_buffer_to_buffer(source, offset, &buffer, 0, size);
            Some(buffer)
        };
        let vertex_copy = copy_input(
            geometry.vertices,
            u64::from(geometry.first_vertex) * geometry.vertex_stride,
            u64::from(geometry.vertex_count) * geometry.vertex_stride,
        );
        let index_copy = geometry.indices.and_then(|indices| {
            copy_input(
                indices,
                u64::from(geometry.first_index) * 4,
                u64::from(geometry.index_count) * 4,
            )
        });
        let size = wgpu::BlasTriangleGeometrySizeDescriptor {
            vertex_format: wgpu::VertexFormat::Float32x3,
            vertex_count: geometry.vertex_count,
            index_format: geometry.indices.map(|_| wgpu::IndexFormat::Uint32),
            index_count: geometry.indices.map(|_| geometry.index_count),
            flags: wgpu::AccelerationStructureGeometryFlags::OPAQUE,
        };
        let blas = self.device.create_blas(
            &wgpu::CreateBlasDescriptor {
                label: Some("versioned_mesh_blas"),
                flags: wgpu::AccelerationStructureFlags::PREFER_FAST_TRACE,
                update_mode: wgpu::AccelerationStructureUpdateMode::Build,
            },
            wgpu::BlasGeometrySizeDescriptors::Triangles {
                descriptors: vec![size.clone()],
            },
        );
        let entry = wgpu::BlasBuildEntry {
            blas: &blas,
            geometry: wgpu::BlasGeometries::TriangleGeometries(vec![wgpu::BlasTriangleGeometry {
                size: &size,
                vertex_buffer: vertex_copy.as_ref().unwrap_or(geometry.vertices),
                first_vertex: if vertex_copy.is_some() {
                    0
                } else {
                    geometry.first_vertex
                },
                vertex_stride: geometry.vertex_stride,
                index_buffer: index_copy.as_ref().or(geometry.indices),
                first_index: geometry.indices.map(|_| {
                    if index_copy.is_some() {
                        0
                    } else {
                        geometry.first_index
                    }
                }),
                transform_buffer: None,
                transform_buffer_offset: None,
            }]),
        };
        encoder.build_acceleration_structures(
            std::iter::once(&entry),
            std::iter::empty::<&wgpu::Tlas>(),
        );
        self.blas_map.insert(mesh_id, blas);
        self.geometry_keys.insert(mesh_id, key);
        Ok(true)
    }

    /// Build a BLAS from vertex/index data.
    pub fn build_blas(
        &mut self,
        mesh_id: u64,
        queue: &wgpu::Queue,
        vertex_data: &[u8],
        vertex_count: u32,
        vertex_stride: u64,
        index_data: Option<&[u8]>,
        index_count: u32,
    ) -> Option<&wgpu::Blas> {
        if !self.rt_available || vertex_count == 0 || vertex_data.is_empty() {
            return None;
        }

        if self.blas_map.contains_key(&mesh_id) {
            return self.blas_map.get(&mesh_id);
        }

        let blas = self.build_blas_inner(
            queue,
            vertex_data,
            vertex_count,
            vertex_stride,
            index_data,
            index_count,
        )?;
        self.blas_map.insert(mesh_id, blas);
        self.blas_map.get(&mesh_id)
    }

    fn build_blas_inner(
        &self,
        queue: &wgpu::Queue,
        vertex_data: &[u8],
        vertex_count: u32,
        vertex_stride: u64,
        index_data: Option<&[u8]>,
        index_count: u32,
    ) -> Option<wgpu::Blas> {
        let device = &self.device;

        let size_desc_f = || wgpu::BlasTriangleGeometrySizeDescriptor {
            vertex_format: wgpu::VertexFormat::Float32x3,
            vertex_count,
            index_format: index_data.map(|_| wgpu::IndexFormat::Uint32),
            index_count: (index_count > 0).then_some(index_count),
            flags: wgpu::AccelerationStructureGeometryFlags::OPAQUE,
        };

        let sizes = wgpu::BlasGeometrySizeDescriptors::Triangles {
            descriptors: vec![size_desc_f()],
        };

        let blas = device.create_blas(
            &wgpu::CreateBlasDescriptor {
                label: Some("mesh_blas"),
                flags: wgpu::AccelerationStructureFlags::PREFER_FAST_TRACE,
                update_mode: wgpu::AccelerationStructureUpdateMode::Build,
            },
            sizes,
        );

        let vertex_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("blas_vertex"),
            size: vertex_data.len() as u64,
            usage: wgpu::BufferUsages::VERTEX
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::BLAS_INPUT,
            mapped_at_creation: false,
        });
        queue.write_buffer(&vertex_buf, 0, vertex_data);

        let index_buf = index_data.map(|data| {
            let buf = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("blas_index"),
                size: data.len() as u64,
                usage: wgpu::BufferUsages::INDEX
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::BLAS_INPUT,
                mapped_at_creation: false,
            });
            queue.write_buffer(&buf, 0, data);
            buf
        });

        let sd = size_desc_f();
        let geometry = wgpu::BlasTriangleGeometry {
            size: &sd,
            vertex_buffer: &vertex_buf,
            first_vertex: 0,
            vertex_stride,
            index_buffer: index_buf.as_ref(),
            first_index: index_buf.as_ref().map(|_| 0),
            transform_buffer: None,
            transform_buffer_offset: None,
        };

        let build_entry = wgpu::BlasBuildEntry {
            blas: &blas,
            geometry: wgpu::BlasGeometries::TriangleGeometries(vec![geometry]),
        };

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("blas_build"),
        });

        encoder.build_acceleration_structures(
            std::iter::once(&build_entry),
            std::iter::empty::<&wgpu::Tlas>(),
        );

        queue.submit(std::iter::once(encoder.finish()));
        // Later submissions on this queue are ordered after the build. A host
        // wait here unnecessarily serialized every mesh upload.

        Some(blas)
    }

    pub fn get_blas(&self, mesh_id: u64) -> Option<&wgpu::Blas> {
        self.blas_map.get(&mesh_id)
    }

    pub fn remove_blas(&mut self, mesh_id: u64) {
        self.blas_map.remove(&mesh_id);
        self.geometry_keys.remove(&mesh_id);
    }

    pub fn clear(&mut self) {
        self.blas_map.clear();
        self.geometry_keys.clear();
    }
    /// Evict geometry no longer referenced by the authoritative caster set.
    pub fn retain(&mut self, mut live: impl FnMut(u64) -> bool) {
        self.blas_map.retain(|id, _| live(*id));
        self.geometry_keys
            .retain(|id, _| self.blas_map.contains_key(id));
    }
}

/// Per-frame Top-Level Acceleration Structure (TLAS) for ray tracing.
pub struct TlasManager {
    tlas: Option<wgpu::Tlas>,
    device: Arc<wgpu::Device>,
    max_instances: u32,
    populated_slots: usize,
    rt_available: bool,
}

impl TlasManager {
    pub fn new(device: Arc<wgpu::Device>, max_instances: u32) -> Self {
        let rt_available = device
            .features()
            .contains(wgpu::Features::EXPERIMENTAL_RAY_QUERY)
            && device.limits().max_tlas_instance_count > 0;
        let max_instances = max_instances
            .max(1)
            .min(device.limits().max_tlas_instance_count.max(1));
        Self {
            tlas: None,
            device,
            max_instances: max_instances.max(1),
            populated_slots: 0,
            rt_available,
        }
    }

    pub fn is_rt_available(&self) -> bool {
        self.rt_available
    }

    pub fn tlas(&self) -> Option<&wgpu::Tlas> {
        self.tlas.as_ref()
    }

    pub fn as_binding(&self) -> Option<wgpu::BindingResource<'_>> {
        self.tlas.as_ref().map(|t| t.as_binding())
    }
    pub fn invalidate(&mut self) {
        self.tlas = None;
        self.populated_slots = 0;
    }

    /// Build the TLAS from a complete list of BLAS + transform pairs. Capacity
    /// grows instead of dropping casters. Invalid input clears readiness so a
    /// caller cannot accidentally obtain the previous frame's TLAS via `tlas()`.
    /// The caller must record/submit geometry builds before this build and queries.
    pub fn build(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        instances: &[TlasInstanceInput],
        blas_manager: &BlasManager,
    ) -> Result<(), AccelerationError> {
        if !self.rt_available {
            return Err(AccelerationError::Unsupported);
        }

        let validation = (|| {
            let count =
                u32::try_from(instances.len()).map_err(|_| AccelerationError::TooManyInstances)?;
            if count > self.device.limits().max_tlas_instance_count {
                return Err(AccelerationError::TooManyInstances);
            }
            for input in instances {
                if blas_manager.get_blas(input.mesh_id).is_none() {
                    return Err(AccelerationError::MissingBlas(input.mesh_id));
                }
                if input.transform.iter().any(|value| !value.is_finite()) {
                    return Err(AccelerationError::InvalidTransform);
                }
            }
            Ok(count)
        })();
        let count = match validation {
            Ok(count) => count,
            Err(error) => {
                self.tlas = None;
                self.populated_slots = 0;
                return Err(error);
            }
        };
        if count > self.max_instances {
            self.max_instances = count
                .checked_next_power_of_two()
                .unwrap_or(count)
                .min(self.device.limits().max_tlas_instance_count);
            self.tlas = None;
            self.populated_slots = 0;
        }

        let tlas = self.tlas.get_or_insert_with(|| {
            self.device.create_tlas(&wgpu::CreateTlasDescriptor {
                label: Some("frame_tlas"),
                max_instances: self.max_instances,
                flags: wgpu::AccelerationStructureFlags::PREFER_FAST_TRACE,
                update_mode: wgpu::AccelerationStructureUpdateMode::Build,
            })
        });

        for i in 0..count as usize {
            let input = &instances[i];
            if let Some(blas) = blas_manager.get_blas(input.mesh_id) {
                tlas[i] = Some(wgpu::TlasInstance::new(blas, input.transform, 0, 0xFF));
            }
        }
        for i in count as usize..self.populated_slots {
            tlas[i] = None;
        }
        self.populated_slots = count as usize;

        let tlas_ref: &wgpu::Tlas = &*tlas;
        encoder.build_acceleration_structures(
            std::iter::empty::<&wgpu::BlasBuildEntry<'_>>(),
            std::iter::once(tlas_ref),
        );
        Ok(())
    }
}

/// Input for one TLAS instance.
pub struct TlasInstanceInput {
    pub mesh_id: u64,
    pub transform: [f32; 12],
}
