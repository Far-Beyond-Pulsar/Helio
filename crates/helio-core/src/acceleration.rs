use std::collections::HashMap;
use std::sync::Arc;

/// Manages Bottom-Level Acceleration Structures (BLAS) for scene meshes.
pub struct BlasManager {
    blas_map: HashMap<u64, wgpu::Blas>,
    device: Arc<wgpu::Device>,
    rt_available: bool,
}

impl BlasManager {
    pub fn new(device: Arc<wgpu::Device>) -> Self {
        let rt_available = device
            .features()
            .contains(wgpu::Features::EXPERIMENTAL_RAY_QUERY);
        Self {
            blas_map: HashMap::new(),
            device,
            rt_available,
        }
    }

    pub fn is_rt_available(&self) -> bool {
        self.rt_available
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
            first_index: None,
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
        device.poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: None,
        }).ok();

        Some(blas)
    }

    pub fn get_blas(&self, mesh_id: u64) -> Option<&wgpu::Blas> {
        self.blas_map.get(&mesh_id)
    }

    pub fn remove_blas(&mut self, mesh_id: u64) {
        self.blas_map.remove(&mesh_id);
    }

    pub fn clear(&mut self) {
        self.blas_map.clear();
    }
}

/// Per-frame Top-Level Acceleration Structure (TLAS) for ray tracing.
pub struct TlasManager {
    tlas: Option<wgpu::Tlas>,
    device: Arc<wgpu::Device>,
    max_instances: u32,
    rt_available: bool,
}

impl TlasManager {
    pub fn new(device: Arc<wgpu::Device>, max_instances: u32) -> Self {
        let rt_available = device
            .features()
            .contains(wgpu::Features::EXPERIMENTAL_RAY_QUERY);
        Self {
            tlas: None,
            device,
            max_instances,
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

    /// Build the TLAS from a list of BLAS + transform pairs.
    pub fn build(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        instances: &[TlasInstanceInput],
        blas_manager: &BlasManager,
    ) {
        if !self.rt_available {
            return;
        }

        let tlas = self.tlas.get_or_insert_with(|| {
            self.device.create_tlas(&wgpu::CreateTlasDescriptor {
                label: Some("frame_tlas"),
                max_instances: self.max_instances,
                flags: wgpu::AccelerationStructureFlags::PREFER_FAST_TRACE,
                update_mode: wgpu::AccelerationStructureUpdateMode::Build,
            })
        });

        let count = instances.len().min(self.max_instances as usize);
        for i in 0..count {
            let input = &instances[i];
            if let Some(blas) = blas_manager.get_blas(input.mesh_id) {
                tlas[i] = Some(wgpu::TlasInstance::new(blas, input.transform, 0, 0xFF));
            }
        }
        for i in count..self.max_instances as usize {
            tlas[i] = None;
        }

        let tlas_ref: &wgpu::Tlas = &*tlas;
        encoder.build_acceleration_structures(
            std::iter::empty::<&wgpu::BlasBuildEntry<'_>>(),
            std::iter::once(tlas_ref),
        );
    }
}

/// Input for one TLAS instance.
pub struct TlasInstanceInput {
    pub mesh_id: u64,
    pub transform: [f32; 12],
}
