//! Frontend-owned acceleration projection of SceneDB's static shadow casters.
//! No renderer-owned scene table or GPU readback is involved.
use helio_core::{BlasGeometry, BlasManager, Movability, TlasInstanceInput, TlasManager};
use helio_pass_gbuffer::{
    MaterialComponent, MeshComponent, RenderGroupComponent, StaticObjectComponent,
};
use pulsar_scenedb::{ChangeCursor, ChangeRead, ComponentChange, ComponentChangeKind, Entity, World};
use std::{
    collections::{HashMap, HashSet},
    hash::Hasher,
    sync::Arc,
};

/// Explicit linear-RGB transmission for each crossed triangle of a thin sheet.
/// Attach to the material entity. This is independent of display alpha and does
/// not model refraction, thickness, or caustics. Closed meshes attenuate at both
/// entrance and exit surfaces; use a single surface for a single pane.
#[derive(Clone, Copy, Debug)]
pub struct RayTransmission(pub [f32; 3]);

/// Change cursors for every component the projection reads. Objects are
/// applied incrementally; a change to any of the others forces a rebuild.
struct Cursors {
    objects: ChangeCursor,
    others: [ChangeCursor; 5],
}

impl Cursors {
    fn open(world: &World) -> Self {
        Self {
            objects: world.open_change_cursor::<StaticObjectComponent>(),
            others: [
                world.open_change_cursor::<MeshComponent>(),
                world.open_change_cursor::<MaterialComponent>(),
                world.open_change_cursor::<RayTransmission>(),
                world.open_change_cursor::<RenderGroupComponent>(),
                world.open_change_cursor::<Movability>(),
            ],
        }
    }
}

/// Every `StaticObjectComponent` field the projection depends on except the
/// transform. A change that leaves this equal only moves the TLAS instance.
#[derive(Clone, Copy, PartialEq, Eq)]
struct CasterKey {
    mesh: (u32, u32),
    material: (u32, u32),
    draw: (u32, u32, i32),
    graph_hash: u64,
    flags: u32,
}

impl CasterKey {
    fn of(object: &StaticObjectComponent) -> Self {
        Self {
            mesh: (object.mesh_slot, object.mesh_generation),
            material: (object.material_slot, object.material_generation),
            draw: (object.index_count, object.first_index, object.vertex_offset),
            graph_hash: object.graph_hash(),
            flags: object.flags,
        }
    }
}

fn casts_shadow(object: &StaticObjectComponent) -> bool {
    object.flags & helio_pass_object_batch::INSTANCE_FLAG_CASTS_SHADOW != 0
}

/// Row-major 3x4 TLAS transform of an object's column-major world matrix.
fn tlas_transform(object: &StaticObjectComponent) -> [f32; 12] {
    let m = object.transform().to_cols_array();
    [m[0], m[4], m[8], m[12], m[1], m[5], m[9], m[13], m[2], m[6], m[10], m[14]]
}

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
    /// `None` until the first full build, and after any error or journal
    /// overflow: the next `prepare` then rebuilds from a full scan.
    cursors: Option<Cursors>,
    /// TLAS instances and the caster that owns each, in the same order.
    instances: Vec<TlasInstanceInput>,
    casters: Vec<(Entity, CasterKey)>,
    caster_slots: HashMap<Entity, usize>,
    changes: Vec<ComponentChange>,
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
            cursors: None,
            instances: Vec::new(),
            casters: Vec::new(),
            caster_slots: HashMap::new(),
            changes: Vec::new(),
        }
    }

    /// Call after flushing the World GPU mirror and before rendering. The World
    /// remains authoritative.
    ///
    /// The first call scans every object row and builds each referenced
    /// mesh/opacity variant's BLAS. Later calls read SceneDB change journals
    /// (their own cursors; no other client's events are consumed): a frame
    /// with no changes does no per-object work, and a transform-only change
    /// to existing casters rewrites just those TLAS instances. Anything else
    /// (a caster added or removed, a mesh, material, transmission, render
    /// group or `Movability` change) rescans, as does a journal overflow.
    ///
    /// A rescan hashes each referenced mesh to detect in-place vertex edits.
    /// Insert a non-deforming [`Movability`] (anything but `Dynamic`) on a
    /// mesh entity to skip that hash: its BLAS is then built once and reused
    /// until SceneDB reallocates or replaces the mesh's geometry range.
    pub fn prepare(&mut self, world: &World) -> helio_core::Result<&wgpu::Tlas> {
        let result = match self.update(world) {
            Ok(true) => Ok(()),
            Ok(false) => self.build(world),
            Err(error) => Err(error),
        };
        if let Err(error) = result {
            self.blas.clear();
            self.geometry_ids.clear();
            self.tlas.invalidate();
            self.cursors = None;
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

    /// Applies journaled changes to the previous build. `Ok(false)` means
    /// the changes need a full rebuild (or there is nothing to update yet).
    fn update(&mut self, world: &World) -> helio_core::Result<bool> {
        let Some(cursors) = self.cursors.as_mut() else {
            return Ok(false);
        };
        if self.tlas.tlas().is_none() {
            return Ok(false);
        }
        reject_unsupported_geometry(world)?;
        self.changes.clear();
        for cursor in &mut cursors.others {
            if world.read_changes(cursor, &mut self.changes) != ChangeRead::Complete
                || !self.changes.is_empty()
            {
                return Ok(false);
            }
        }
        if world.read_changes(&mut cursors.objects, &mut self.changes) != ChangeRead::Complete {
            return Ok(false);
        }
        let mut moved = false;
        for change in &self.changes {
            let slot = self.caster_slots.get(&change.entity).copied();
            let object = match change.kind {
                ComponentChangeKind::Removed => None,
                ComponentChangeKind::Inserted | ComponentChangeKind::Mutated => {
                    world.get::<StaticObjectComponent>(change.entity)
                }
            };
            match (slot, object) {
                // A non-caster changed and still is not one.
                (None, None) => {}
                (None, Some(object)) if !casts_shadow(object) => {}
                (Some(slot), Some(object))
                    if casts_shadow(object) && CasterKey::of(object) == self.casters[slot].1 =>
                {
                    let transform = tlas_transform(object);
                    if self.instances[slot].transform != transform {
                        self.instances[slot].transform = transform;
                        moved = true;
                    }
                }
                // Caster added, removed or re-keyed.
                _ => return Ok(false),
            }
        }
        if moved {
            let mut encoder = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("SceneDB RT acceleration update"),
                });
            self.tlas
                .build(&mut encoder, &self.instances, &self.blas)
                .map_err(|e| rt_error(&e.to_string()))?;
            self.queue.submit([encoder.finish()]);
        }
        Ok(true)
    }

    fn build(&mut self, world: &World) -> helio_core::Result<()> {
        self.has_transmission = false;
        // Open before scanning: a change racing the scan is re-applied from
        // the journal next frame, never lost.
        self.cursors = Some(Cursors::open(world));
        self.casters.clear();
        self.caster_slots.clear();
        let error = rt_error;
        reject_unsupported_geometry(world)?;
        let store = world
            .gpu_mirror()
            .ok_or_else(|| error("GPU mirror is required"))?
            .store();
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
        let mut mesh_ranges = HashMap::new();
        let mut material_rows = HashMap::new();
        let mut instances = Vec::new();
        let mut transmission_rows = Vec::<[f32; 4]>::new();
        let mut has_transmission = false;
        for (object_entity, (object,)) in world.query::<(&StaticObjectComponent,)>() {
            if !casts_shadow(object) {
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
            // Resolve each material once per frame; many objects share one.
            let (material_generation, rgb, transmits) = match material_rows.get(&object.material_slot) {
                Some(row) => *row,
                None => {
                    let (material_entity, material) = materials
                        .get(&object.material_slot)
                        .ok_or_else(|| error("missing caster material"))?;
                    let transmission = world.get::<RayTransmission>(*material_entity);
                    if material.flags & (helio_mats::FLAG_ALPHA_TEST | helio_mats::FLAG_HAS_CUSTOM_SHADER)
                        != 0
                        || (material.flags & helio_mats::FLAG_ALPHA_BLEND != 0 && transmission.is_none())
                    {
                        return Err(error("masked, custom-shader, and transparent casters without explicit RayTransmission are unsupported"));
                    }
                    let rgb = transmission.map_or([0.0; 3], |value| value.0);
                    if !rgb.iter().all(|v| v.is_finite() && (0.0..=1.0).contains(v)) {
                        return Err(error("transmission must be finite linear RGB in [0, 1]"));
                    }
                    let row = (
                        material_entity.generation().wrapping_add(1),
                        rgb,
                        transmission.is_some(),
                    );
                    material_rows.insert(object.material_slot, row);
                    row
                }
            };
            if material_generation != object.material_generation {
                return Err(error("stale caster material identity"));
            }
            if object.graph_hash() != 0 {
                return Err(error("masked, custom-shader, and transparent casters without explicit RayTransmission are unsupported"));
            }
            has_transmission |= transmits;
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
            // Resolve each geometry once per frame; many objects share a mesh.
            let (vertex_range, index_range) = match mesh_ranges.get(&id) {
                Some(ranges) => *ranges,
                None => (
                    MeshComponent::vertices_gpu_handle(store, entity.index())
                        .ok_or_else(|| error("missing mirrored vertices"))?,
                    MeshComponent::indices_gpu_handle(store, entity.index())
                        .ok_or_else(|| error("missing mirrored indices"))?,
                ),
            };
            if object.index_count != index_range.count
                || object.first_index != index_range.offset
                || object.vertex_offset != vertex_range.offset as i32
            {
                return Err(error("caster draw range differs from current mesh; refresh the StaticObjectComponent"));
            }
            if live.insert(id) {
                mesh_ranges.insert(id, (vertex_range, index_range));
                if vertex_range.count as usize != mesh.vertices.len()
                    || index_range.count as usize != mesh.indices.len()
                {
                    return Err(error(
                        "mesh mirror is not current; flush SceneDB before RT preparation",
                    ));
                }
                // A mesh without `Movability` keeps content-based invalidation
                // for in-place edits. A mesh that promises not to deform is
                // never re-read: its BLAS key (buffer, ranges, opacity) still
                // rebuilds it when SceneDB moves or replaces the allocation.
                let revision = if world
                    .get::<Movability>(*entity)
                    .is_none_or(|movability| movability.can_deform())
                {
                    // Streaming hash's vectorized bulk path for large meshes.
                    let mut hash = twox_hash::XxHash3_64::default();
                    hash.write(bytemuck::cast_slice(&mesh.vertices));
                    hash.write(bytemuck::cast_slice(&mesh.indices));
                    hash.finish()
                } else {
                    0
                };
                self.blas
                    .build_from_buffers_with_opacity(
                        id,
                        &mut encoder,
                        BlasGeometry {
                            revision,
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
            self.caster_slots.insert(object_entity, self.casters.len());
            self.casters.push((object_entity, CasterKey::of(object)));
            instances.push(TlasInstanceInput {
                mesh_id: id,
                transform: tlas_transform(object),
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
        self.instances = instances;
        self.queue.submit([encoder.finish()]);
        Ok(())
    }
}

fn rt_error(message: &str) -> helio_core::Error {
    helio_core::Error::InvalidPassConfig(format!("SceneDB RT: {message}"))
}

/// These geometry families need their own explicit projection. Never
/// silently omit known non-triangle or multi-material scene content.
fn reject_unsupported_geometry(world: &World) -> helio_core::Result<()> {
    let store = world
        .gpu_mirror()
        .ok_or_else(|| rt_error("GPU mirror is required"))?
        .store();
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
            return Err(rt_error(&format!("unsupported geometry buffer {name}")));
        }
    }
    Ok(())
}
