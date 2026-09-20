//! Frontend-owned acceleration projection of SceneDB's static shadow casters.
//! No renderer-owned scene table or GPU readback is involved.
use helio_core::{BlasGeometry, BlasManager, TlasInstanceInput, TlasManager};
use helio_pass_gbuffer::{
    MaterialComponent, MeshComponent, RenderGroupComponent, StaticObjectComponent,
};
use pulsar_scenedb::World;
use std::{
    collections::{HashMap, HashSet},
    hash::{DefaultHasher, Hasher},
    sync::Arc,
};

/// Explicit linear-RGB transmission for each crossed triangle of a thin sheet.
/// Attach to the material entity. This is independent of display alpha and does
/// not model refraction, thickness, or caustics. Closed meshes attenuate at both
/// entrance and exit surfaces; use a single surface for a single pane.
#[derive(Clone, Copy, Debug)]
pub struct RayTransmission(pub [f32; 3]);

pub struct SceneDbRayTracing {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    blas: BlasManager,
    tlas: TlasManager,
    transmission: Option<wgpu::Buffer>,
    has_transmission: bool,
    transmission_rows: Vec<[f32; 4]>,
    geometry_ids: HashMap<(u64, bool), u64>,
    next_geometry_id: u64,
}
impl SceneDbRayTracing {
    pub fn new(device: Arc<wgpu::Device>, queue: Arc<wgpu::Queue>) -> Self {
        Self {
            blas: BlasManager::new(device.clone()),
            tlas: TlasManager::new(device.clone(), 1),
            device,
            queue,
            transmission: None,
            has_transmission: false,
            transmission_rows: Vec::new(),
            geometry_ids: HashMap::new(),
            next_geometry_id: 0,
        }
    }

    /// Call after flushing the World GPU mirror and before rendering. The World
    /// remains authoritative. This control scans object rows on the CPU and hashes
    /// each referenced mesh/opacity variant once to detect in-place edits without consuming other
    /// clients' SceneDB change events. Hashing is a known CPU cost, not GPU timing.
    pub fn prepare(&mut self, world: &World) -> helio_core::Result<&wgpu::Tlas> {
        let result = self.build(world);
        if let Err(error) = result {
            self.blas.clear();
            self.geometry_ids.clear();
            self.tlas.invalidate();
            return Err(error);
        }
        Ok(self.tlas.tlas().expect("successful TLAS build"))
    }

    /// Valid only after successful preparation; rows follow TLAS instance order.
    pub fn transmission(&self) -> Option<&wgpu::Buffer> {
        self.has_transmission
            .then_some(self.transmission.as_ref())
            .flatten()
    }

    pub fn tlas(&self) -> Option<&wgpu::Tlas> {
        self.tlas.tlas()
    }

    fn build(&mut self, world: &World) -> helio_core::Result<()> {
        self.has_transmission = false;
        let error =
            |message: &str| helio_core::Error::InvalidPassConfig(format!("SceneDB RT: {message}"));
        let mirror = world
            .gpu_mirror()
            .ok_or_else(|| error("GPU mirror is required"))?;
        let store = mirror.store();
        // These geometry families need their own explicit projection. Never
        // silently omit known non-triangle or multi-material scene content.
        for name in [
            "sectioned_objects",
            "foliage_layers",
            "voxel_volumes",
            "vg_instances",
            "sublevels",
        ] {
            if store
                .resolve_buffer_handle(helio_core::BufferKey::of(name))
                .is_some()
            {
                return Err(error(&format!("unsupported geometry buffer {name}")));
            }
        }
        let meshes: HashMap<_, _> = world
            .query::<(&MeshComponent,)>()
            .map(|(entity, (mesh,))| (entity.index(), (entity, mesh)))
            .collect();
        let materials: HashMap<_, _> = world
            .query::<(&MaterialComponent,)>()
            .map(|(entity, (material,))| (entity.index(), (entity, material)))
            .collect();
        let vertices =
            store.resolve_buffer_handle(helio_core::BufferKey::of("builtin_mesh_vertex"));
        let indices = store.resolve_buffer_handle(helio_core::BufferKey::of("builtin_mesh_index"));
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("SceneDB RT acceleration"),
            });
        let mut live = HashSet::new();
        let mut instances = Vec::new();
        let mut transmission_rows = Vec::<[f32; 4]>::new();
        let mut has_transmission = false;
        for (object_entity, (object,)) in world.query::<(&StaticObjectComponent,)>() {
            if object.flags & helio_pass_object_batch::INSTANCE_FLAG_CASTS_SHADOW == 0 {
                continue;
            }
            if world
                .get::<RenderGroupComponent>(object_entity)
                .is_some_and(|group| group.group_mask != 0)
            {
                return Err(error("group visibility needs an explicit RT projection"));
            }
            if helio_pass_object_batch::coordinate_space(object.flags) != 0 {
                return Err(error("non-world coordinate-space casters are unsupported"));
            }
            let (material_entity, material) = materials
                .get(&object.material_slot)
                .ok_or_else(|| error("missing caster material"))?;
            if material_entity.generation().wrapping_add(1) != object.material_generation {
                return Err(error("stale caster material identity"));
            }
            let transmission = world.get::<RayTransmission>(*material_entity);
            if material.flags & (helio_mats::FLAG_ALPHA_TEST | helio_mats::FLAG_HAS_CUSTOM_SHADER)
                != 0
                || object.graph_hash() != 0
                || (material.flags & helio_mats::FLAG_ALPHA_BLEND != 0 && transmission.is_none())
            {
                return Err(error("masked, custom-shader, and transparent casters without explicit RayTransmission are unsupported"));
            }
            let rgb = transmission.map_or([0.0; 3], |value| value.0);
            if !rgb.iter().all(|v| v.is_finite() && (0.0..=1.0).contains(v)) {
                return Err(error("transmission must be finite linear RGB in [0, 1]"));
            }
            has_transmission |= transmission.is_some();
            transmission_rows.push([rgb[0], rgb[1], rgb[2], 0.0]);
            let (entity, mesh) = meshes
                .get(&object.mesh_slot)
                .ok_or_else(|| error("missing caster mesh"))?;
            if entity.generation().wrapping_add(1) != object.mesh_generation {
                return Err(error("stale caster mesh identity"));
            }
            let mesh_identity = ((entity.generation() as u64) << 32) | entity.index() as u64;
            let opaque = rgb.iter().all(|v| *v == 0.0);
            // A mesh can be instanced with both opaque and transmitting materials.
            // Allocate separate BLAS identities without truncating entity generations.
            let id = if let Some(id) = self.geometry_ids.get(&(mesh_identity, opaque)) {
                *id
            } else {
                let id = self.next_geometry_id;
                self.next_geometry_id = id
                    .checked_add(1)
                    .ok_or_else(|| error("BLAS identity exhausted"))?;
                self.geometry_ids.insert((mesh_identity, opaque), id);
                id
            };
            let vertex_range = MeshComponent::vertices_gpu_handle(store, entity.index())
                .ok_or_else(|| error("missing mirrored vertices"))?;
            let index_range = MeshComponent::indices_gpu_handle(store, entity.index())
                .ok_or_else(|| error("missing mirrored indices"))?;
            if object.index_count != index_range.count
                || object.first_index != index_range.offset
                || object.vertex_offset != vertex_range.offset as i32
            {
                return Err(error("caster draw range differs from current mesh; refresh the StaticObjectComponent"));
            }
            if live.insert(id) {
                if vertex_range.count as usize != mesh.vertices.len()
                    || index_range.count as usize != mesh.indices.len()
                {
                    return Err(error(
                        "mesh mirror is not current; flush SceneDB before RT preparation",
                    ));
                }
                let mut hash = DefaultHasher::new();
                hash.write(bytemuck::cast_slice(&mesh.vertices));
                hash.write(bytemuck::cast_slice(&mesh.indices));
                self.blas
                    .build_from_buffers_with_opacity(
                        id,
                        &mut encoder,
                        BlasGeometry {
                            revision: hash.finish(),
                            vertices: &vertices
                                .as_ref()
                                .ok_or_else(|| error("missing vertex pool"))?
                                .buffer,
                            first_vertex: vertex_range.offset,
                            vertex_count: vertex_range.count,
                            vertex_stride: std::mem::size_of::<helio_core::PackedVertex>() as u64,
                            indices: Some(
                                &indices
                                    .as_ref()
                                    .ok_or_else(|| error("missing index pool"))?
                                    .buffer,
                            ),
                            first_index: index_range.offset,
                            index_count: index_range.count,
                        },
                        opaque,
                    )
                    .map_err(|e| error(&e.to_string()))?;
            }
            let m = object.transform().to_cols_array();
            instances.push(TlasInstanceInput {
                mesh_id: id,
                transform: [
                    m[0], m[4], m[8], m[12], m[1], m[5], m[9], m[13], m[2], m[6], m[10], m[14],
                ],
            });
        }
        self.blas.retain(|id| live.contains(&id));
        self.geometry_ids.retain(|_, id| live.contains(id));
        self.tlas
            .build(&mut encoder, &instances, &self.blas)
            .map_err(|e| error(&e.to_string()))?;
        if has_transmission {
            let size = ((transmission_rows.len() + 1) * 16).max(32) as u64;
            let mut buffer_needs_upload = false;
            if self
                .transmission
                .as_ref()
                .is_none_or(|buffer| buffer.size() < size)
            {
                self.transmission = Some(self.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("SceneDB RT thin-sheet transmission"),
                    size: size.next_power_of_two(),
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                }));
                buffer_needs_upload = true;
            }
            let buffer = self.transmission.as_ref().unwrap();
            if buffer_needs_upload || self.transmission_rows != transmission_rows {
                let header = [1u32, transmission_rows.len() as u32, 0, 0];
                self.queue
                    .write_buffer(buffer, 0, bytemuck::cast_slice(&header));
                self.queue
                    .write_buffer(buffer, 16, bytemuck::cast_slice(&transmission_rows));
                self.transmission_rows = transmission_rows;
            }
        } else {
            self.transmission_rows.clear();
        }
        self.has_transmission = has_transmission;
        self.queue.submit([encoder.finish()]);
        Ok(())
    }
}
