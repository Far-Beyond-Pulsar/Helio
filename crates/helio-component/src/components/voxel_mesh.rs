//! A free-standing voxel volume as an ordinary mesh draw
//! (Pulsar-Native#1056).
//!
//! [`greedy_mesh_chunk`] turns one canonical 8³ material chunk into quads:
//! only faces between a solid sample and air are kept (the six face
//! neighbour chunks decide the chunk's own border), and coplanar faces of
//! one material slot merge into rectangles. [`VoxelMeshGeometry::assemble`]
//! joins the chunk meshes of one volume into one vertex/index range with one
//! section per material slot, in the volume's local space (voxel `(0,0,0)`'s
//! corner at the origin, `voxel_size` per sample).
//!
//! [`write_voxel_mesh_rows`] uploads that geometry as the voxel instance's
//! mesh rows, the ones a `StaticMeshComponent` at that row would write: the
//! content-interned `builtin_mesh_vertex`/`builtin_mesh_index` pools (under
//! the geometry's content id, [`MeshAssetPath::generated`]) and its derived
//! `StaticMeshDraw`. The scene join then draws it like any mesh instance,
//! placed by the owner's full transform (rotation and scale included), into
//! the G-buffer, shadows and picking. Removing the component clears the rows
//! and releases the geometry (a clear registration on [`VoxelComponent`]).
//!
//! Materials: the component's palette names SceneDB material records, which
//! a mesh section cannot resolve (it carries its material inline). Each
//! palette slot therefore draws as its own section with a default surface
//! coloured from its material ID ([`voxel_material_surface`]).

use pulsar_scenedb::gpu::GpuMirrorHandle;

use super::{
    MeshAssetPath, ObjectMovability, StaticMeshComponent, StaticMeshMaterialSlot,
    StaticMeshMaterialSlots, VoxelComponent,
};
use crate::mesh_cache::{ImportedSurfaceMaterial, MeshSection};
use helio::PackedVertex;
use helio_voxel_data::{VoxelChunkKey, VOXEL_CHUNK_EDGE, VOXEL_CHUNK_SAMPLES};

pub use helio_voxel_data::VOXEL_MESH_RENDERER;

const EDGE: usize = VOXEL_CHUNK_EDGE;
/// High word of every voxel geometry's content id, so it cannot meet a
/// mesh asset's id in the shared pools.
const CONTENT_TAG: u128 = 0x564f_5845_4c4d_4553 << 64; // "VOXELMES"

/// One merged face of a chunk, in chunk-local sample units.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct VoxelQuad {
    /// `axis * 2` for the positive direction, `axis * 2 + 1` for the
    /// negative one (axis 0 = x, 1 = y, 2 = z).
    pub face: u8,
    /// One-based palette slot of the solid sample the face belongs to.
    pub slot: u8,
    /// Plane of the face along `axis`, 0..=8.
    pub layer: u8,
    /// Extent on the face's two other axes (`(axis + 1) % 3`, then
    /// `(axis + 2) % 3`), half-open.
    pub u: [u8; 2],
    pub v: [u8; 2],
}

impl VoxelQuad {
    pub fn axis(self) -> usize {
        usize::from(self.face / 2)
    }

    pub fn positive(self) -> bool {
        self.face % 2 == 0
    }
}

/// The quads of one chunk and their geometry id: equal quads, equal id, so
/// a chunk re-meshed to the same surface keeps its id.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VoxelChunkMesh {
    pub quads: Vec<VoxelQuad>,
    pub id: u64,
}

/// The chunk's six face neighbours, `[+x, -x, +y, -y, +z, -z]`; a missing
/// neighbour is air.
pub type VoxelChunkNeighbours<'a> = [Option<&'a [u8; VOXEL_CHUNK_SAMPLES]>; 6];

fn fnv(hash: &mut u64, bytes: &[u8]) {
    for byte in bytes {
        *hash ^= u64::from(*byte);
        *hash = hash.wrapping_mul(0x0100_0000_01b3);
    }
}

const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;

/// Greedy-mesh one 8³ chunk (`x` fastest, zero is air). Faces on the
/// chunk's border are culled against `neighbours`.
pub fn greedy_mesh_chunk(
    samples: &[u8; VOXEL_CHUNK_SAMPLES],
    neighbours: VoxelChunkNeighbours<'_>,
) -> VoxelChunkMesh {
    let sample = |p: [i32; 3]| -> u8 {
        let outside = (0..3).find(|&axis| !(0..EDGE as i32).contains(&p[axis]));
        let Some(axis) = outside else {
            return samples[p[2] as usize * EDGE * EDGE + p[1] as usize * EDGE + p[0] as usize];
        };
        let positive = p[axis] >= EDGE as i32;
        let Some(neighbour) = neighbours[axis * 2 + usize::from(!positive)] else {
            return 0;
        };
        let mut q = p;
        q[axis] = if positive { 0 } else { EDGE as i32 - 1 };
        // A diagonal step never happens: faces only look along one axis.
        neighbour[q[2] as usize * EDGE * EDGE + q[1] as usize * EDGE + q[0] as usize]
    };
    let mut quads = Vec::new();
    let mut mask = [[0u8; EDGE]; EDGE];
    for axis in 0..3 {
        let (ua, va) = ((axis + 1) % 3, (axis + 2) % 3);
        for positive in [true, false] {
            let step = if positive { 1 } else { -1 };
            for layer in 0..EDGE {
                for (u, column) in mask.iter_mut().enumerate() {
                    for (v, cell) in column.iter_mut().enumerate() {
                        let mut p = [0i32; 3];
                        p[axis] = layer as i32;
                        p[ua] = u as i32;
                        p[va] = v as i32;
                        let slot = sample(p);
                        p[axis] += step;
                        *cell = if slot != 0 && sample(p) == 0 { slot } else { 0 };
                    }
                }
                for v in 0..EDGE {
                    let mut u = 0;
                    while u < EDGE {
                        let slot = mask[u][v];
                        if slot == 0 {
                            u += 1;
                            continue;
                        }
                        let mut width = 1;
                        while u + width < EDGE && mask[u + width][v] == slot {
                            width += 1;
                        }
                        let mut height = 1;
                        while v + height < EDGE
                            && (u..u + width).all(|uu| mask[uu][v + height] == slot)
                        {
                            height += 1;
                        }
                        for column in &mut mask[u..u + width] {
                            for cell in &mut column[v..v + height] {
                                *cell = 0;
                            }
                        }
                        quads.push(VoxelQuad {
                            face: (axis * 2 + usize::from(!positive)) as u8,
                            slot,
                            layer: (layer + usize::from(positive)) as u8,
                            u: [u as u8, (u + width) as u8],
                            v: [v as u8, (v + height) as u8],
                        });
                        u += width;
                    }
                }
            }
        }
    }
    let mut id = FNV_OFFSET;
    for quad in &quads {
        fnv(
            &mut id,
            &[quad.face, quad.slot, quad.layer, quad.u[0], quad.u[1], quad.v[0], quad.v[1]],
        );
    }
    VoxelChunkMesh { quads, id }
}

/// One volume's mesh, ready for [`write_voxel_mesh_rows`].
#[derive(Clone, Debug, Default)]
pub struct VoxelMeshGeometry {
    pub vertices: Vec<PackedVertex>,
    pub indices: Vec<u32>,
    /// `(palette slot, first index, index count)`, one per slot drawn.
    pub sections: Vec<(u8, u32, u32)>,
    /// Local bounding sphere `[center.xyz, radius]`.
    pub bounds_local: [f32; 4],
    /// Content id of the vertex and index data (never zero).
    pub id: u128,
}

impl VoxelMeshGeometry {
    /// Join `chunks` (LOD-zero keys and their meshes) into one mesh with
    /// `voxel_size`-sized samples. The result does not depend on the order
    /// of `chunks`.
    pub fn assemble<'a>(
        chunks: impl IntoIterator<Item = (VoxelChunkKey, &'a VoxelChunkMesh)>,
        voxel_size: f32,
    ) -> Self {
        let mut chunks: Vec<_> = chunks.into_iter().collect();
        chunks.sort_by_key(|(key, _)| *key);
        let mut id = FNV_OFFSET;
        fnv(&mut id, &voxel_size.to_bits().to_le_bytes());
        for (key, mesh) in &chunks {
            for coordinate in [key.x, key.y, key.z] {
                fnv(&mut id, &coordinate.to_le_bytes());
            }
            fnv(&mut id, &mesh.id.to_le_bytes());
        }
        let mut slots: Vec<u8> = chunks
            .iter()
            .flat_map(|(_, mesh)| mesh.quads.iter().map(|quad| quad.slot))
            .collect();
        slots.sort_unstable();
        slots.dedup();
        let mut geometry = Self {
            id: CONTENT_TAG | u128::from(id),
            ..Self::default()
        };
        let (mut min, mut max) = ([f32::MAX; 3], [f32::MIN; 3]);
        for slot in slots {
            let first = geometry.indices.len() as u32;
            for (key, mesh) in &chunks {
                let base = [key.x, key.y, key.z].map(|c| c as f32 * EDGE as f32);
                for quad in mesh.quads.iter().filter(|quad| quad.slot == slot) {
                    let axis = quad.axis();
                    let (ua, va) = ((axis + 1) % 3, (axis + 2) % 3);
                    let corner = |u: u8, v: u8| {
                        let mut p = [0.0f32; 3];
                        p[axis] = base[axis] + f32::from(quad.layer);
                        p[ua] = base[ua] + f32::from(u);
                        p[va] = base[va] + f32::from(v);
                        p.map(|c| c * voxel_size)
                    };
                    let sign = if quad.positive() { 1.0 } else { -1.0 };
                    let mut normal = [0.0f32; 3];
                    normal[axis] = sign;
                    let mut tangent = [0.0f32; 3];
                    tangent[ua] = 1.0;
                    // Counter-clockwise seen from the side the face looks to.
                    let corners = [
                        (quad.u[0], quad.v[0]),
                        (quad.u[1], quad.v[0]),
                        (quad.u[1], quad.v[1]),
                        (quad.u[0], quad.v[1]),
                    ];
                    let start = geometry.vertices.len() as u32;
                    for (u, v) in corners {
                        let position = corner(u, v);
                        for c in 0..3 {
                            min[c] = min[c].min(position[c]);
                            max[c] = max[c].max(position[c]);
                        }
                        geometry.vertices.push(PackedVertex::from_components(
                            position,
                            normal,
                            [
                                f32::from(u) + base[ua],
                                f32::from(v) + base[va],
                            ],
                            tangent,
                            sign,
                        ));
                    }
                    let order: [u32; 6] = if quad.positive() {
                        [0, 1, 2, 0, 2, 3]
                    } else {
                        [0, 2, 1, 0, 3, 2]
                    };
                    geometry
                        .indices
                        .extend(order.iter().map(|offset| start + offset));
                }
            }
            let count = geometry.indices.len() as u32 - first;
            geometry.sections.push((slot, first, count));
        }
        if !geometry.vertices.is_empty() {
            let center: [f32; 3] = std::array::from_fn(|c| (min[c] + max[c]) * 0.5);
            let radius = (0..3)
                .map(|c| (max[c] - min[c]) * 0.5)
                .map(|h| h * h)
                .sum::<f32>()
                .sqrt();
            geometry.bounds_local = [center[0], center[1], center[2], radius];
        }
        geometry
    }

    pub fn quad_count(&self) -> usize {
        self.indices.len() / 6
    }
}

/// The surface a voxel material ID draws with: ID 0 (the component's
/// default palette) is a neutral grey; other IDs get distinct, stable hues.
pub fn voxel_material_surface(material_id: u32) -> ImportedSurfaceMaterial {
    let base_color = if material_id == 0 {
        [0.6, 0.6, 0.6, 1.0]
    } else {
        // Golden-ratio hue steps keep neighbouring IDs apart.
        let hue = (material_id as f32 * 0.618_034).fract() * 6.0;
        let (saturation, value) = (0.55, 0.8);
        let chroma = value * saturation;
        let x = chroma * (1.0 - (hue % 2.0 - 1.0).abs());
        let (r, g, b) = match hue as u32 {
            0 => (chroma, x, 0.0),
            1 => (x, chroma, 0.0),
            2 => (0.0, chroma, x),
            3 => (0.0, x, chroma),
            4 => (x, 0.0, chroma),
            _ => (chroma, 0.0, x),
        };
        let m = value - chroma;
        [r + m, g + m, b + m, 1.0]
    };
    ImportedSurfaceMaterial {
        base_color,
        roughness: 0.85,
        metallic: 0.0,
        emissive: [0.0; 3],
        emissive_intensity: 0.0,
        alpha: 1.0,
    }
}

/// Upload `mesh` as the mesh rows of the voxel instance at `row` (its
/// entity index): exactly the rows a [`StaticMeshComponent`] at that row
/// writes, from a generated (never inserted) one. The geometry interns into
/// the shared pools under its content id (an equal volume elsewhere shares
/// it; the row's previous geometry is released), and the derived
/// [`StaticMeshDraw`](super::StaticMeshDraw) gets one section per palette
/// slot. The draw is movable: its geometry changes with every edit, so it
/// must not sit in the static shadow cache. An empty mesh clears the rows.
pub fn write_voxel_mesh_rows(
    mirror: &GpuMirrorHandle,
    row: u32,
    mesh: &VoxelMeshGeometry,
    material_ids: &[u32],
) {
    if mesh.indices.is_empty() {
        clear_voxel_mesh_rows(mirror, row);
        return;
    }
    let slots = mesh
        .sections
        .iter()
        .map(|&(slot, _, _)| {
            let material_id = material_ids
                .get(usize::from(slot).saturating_sub(1))
                .copied()
                .unwrap_or(0);
            StaticMeshMaterialSlot {
                name: format!("Voxel material {slot}"),
                surface_override: Some(voxel_material_surface(material_id)),
                ..Default::default()
            }
        })
        .collect();
    let mesh_sections = mesh
        .sections
        .iter()
        .enumerate()
        .map(|(index, &(_, first_index, index_count))| MeshSection {
            first_index,
            index_count,
            material_slot: index as u32,
        })
        .collect();
    let generated = StaticMeshComponent {
        mesh_asset: MeshAssetPath::generated(mesh.id),
        material_slots: StaticMeshMaterialSlots { slots },
        movability: ObjectMovability::Movable,
        vertices: mesh.vertices.clone(),
        indices: mesh.indices.clone(),
        mesh_sections,
        bounds_local: mesh.bounds_local,
        ..Default::default()
    };
    // `true`: the geometry columns are written once per insert, and this is
    // new geometry for the row.
    pulsar_scenedb::gpu::write_derived_row(mirror, row, &generated, true);
}

/// Clear the voxel instance's mesh rows at `row` and release its geometry:
/// exactly what removing a mesh instance at that row does.
pub fn clear_voxel_mesh_rows(mirror: &GpuMirrorHandle, row: u32) {
    pulsar_scenedb::gpu::clear_derived_row::<StaticMeshComponent>(mirror, row);
}

fn voxel_component_clear(mirror: &GpuMirrorHandle, row: u32) {
    clear_voxel_mesh_rows(mirror, row);
}

// The rows a renderer wrote for a voxel instance belong to the component:
// SceneDB clears them when it is removed or its entity despawns, before the
// row can be reused.
pulsar_scenedb::pulsar_reflection::inventory::submit! {
    pulsar_scenedb::gpu::GpuClearRegistration {
        component_id: pulsar_scenedb::component_id::<VoxelComponent>,
        clear: voxel_component_clear,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn chunk(solid: impl Fn(usize, usize, usize) -> u8) -> [u8; VOXEL_CHUNK_SAMPLES] {
        let mut samples = [0; VOXEL_CHUNK_SAMPLES];
        for z in 0..EDGE {
            for y in 0..EDGE {
                for x in 0..EDGE {
                    samples[z * 64 + y * 8 + x] = solid(x, y, z);
                }
            }
        }
        samples
    }

    #[test]
    fn a_cube_is_six_quads() {
        let cube = chunk(|x, y, z| u8::from(x < 4 && y < 4 && z < 4));
        let mesh = greedy_mesh_chunk(&cube, [None; 6]);
        assert_eq!(mesh.quads.len(), 6);
        let geometry = VoxelMeshGeometry::assemble([(VoxelChunkKey::new(0, 0, 0, 0), &mesh)], 0.5);
        assert_eq!(geometry.quad_count(), 6);
        assert_eq!(geometry.vertices.len(), 24);
        assert_eq!(geometry.sections, vec![(1, 0, 36)]);
        // Two units across, centred at (1, 1, 1).
        assert_eq!(&geometry.bounds_local[..3], &[1.0, 1.0, 1.0]);
        assert!((geometry.bounds_local[3] - 3.0f32.sqrt()).abs() < 1e-6);
        // Every face looks away from the cube's centre and winds
        // counter-clockwise seen from there.
        for triangle in geometry.indices.chunks_exact(3) {
            let p: [glam::Vec3; 3] = std::array::from_fn(|k| {
                glam::Vec3::from(geometry.vertices[triangle[k] as usize].position)
            });
            let normal = (p[1] - p[0]).cross(p[2] - p[0]);
            let outward = (p[0] + p[1] + p[2]) / 3.0 - glam::Vec3::ONE;
            assert!(normal.dot(outward) > 0.0, "{p:?}");
        }
    }

    #[test]
    fn an_l_shape_is_ten_quads() {
        // (0,0,0), (1,0,0) and (0,1,0): each L face merges into two
        // rectangles, the straight sides into one.
        let l = chunk(|x, y, z| u8::from(z == 0 && (x, y) != (1, 1) && x < 2 && y < 2));
        assert_eq!(greedy_mesh_chunk(&l, [None; 6]).quads.len(), 10);
    }

    #[test]
    fn an_empty_chunk_has_no_quads() {
        let empty = [0; VOXEL_CHUNK_SAMPLES];
        let mesh = greedy_mesh_chunk(&empty, [None; 6]);
        assert!(mesh.quads.is_empty());
        let geometry = VoxelMeshGeometry::assemble([(VoxelChunkKey::new(0, 0, 0, 0), &mesh)], 1.0);
        assert!(geometry.indices.is_empty() && geometry.sections.is_empty());
    }

    #[test]
    fn faces_between_chunks_and_between_slots_follow_the_samples() {
        let full = chunk(|_, _, _| 1);
        assert_eq!(greedy_mesh_chunk(&full, [None; 6]).quads.len(), 6);
        // A solid neighbour on +x hides that face.
        let hidden = greedy_mesh_chunk(&full, [Some(&full), None, None, None, None, None]);
        assert_eq!(hidden.quads.len(), 5);
        assert!(hidden.quads.iter().all(|quad| quad.face != 0));
        // Two materials side by side: solid against solid is no face, but
        // each material's outer faces stay separate.
        let halves = chunk(|x, _, _| if x < 4 { 1 } else { 2 });
        let mesh = greedy_mesh_chunk(&halves, [None; 6]);
        assert_eq!(mesh.quads.len(), 10);
        let geometry = VoxelMeshGeometry::assemble([(VoxelChunkKey::new(0, 0, 0, 0), &mesh)], 1.0);
        assert_eq!(geometry.sections, vec![(1, 0, 30), (2, 30, 30)]);
    }

    #[test]
    fn geometry_ids_follow_the_content() {
        let a = greedy_mesh_chunk(&chunk(|x, _, _| u8::from(x == 0)), [None; 6]);
        let again = greedy_mesh_chunk(&chunk(|x, _, _| u8::from(x == 0)), [None; 6]);
        let b = greedy_mesh_chunk(&chunk(|x, _, _| u8::from(x == 1)), [None; 6]);
        assert_eq!(a.id, again.id);
        assert_ne!(a.id, b.id);
        let key = |x| VoxelChunkKey::new(x, 0, 0, 0);
        let one = VoxelMeshGeometry::assemble([(key(0), &a), (key(1), &b)], 1.0);
        let swapped = VoxelMeshGeometry::assemble([(key(1), &b), (key(0), &a)], 1.0);
        assert_eq!(one.id, swapped.id);
        assert_ne!(one.id, VoxelMeshGeometry::assemble([(key(0), &a), (key(1), &a)], 1.0).id);
        assert_ne!(one.id, VoxelMeshGeometry::assemble([(key(0), &a), (key(1), &b)], 2.0).id);
        assert_eq!(one.id >> 64, CONTENT_TAG >> 64);
        // The rows name it by that id: it interns like an asset's geometry.
        use pulsar_scenedb::handle_ledger::{ContentAddressed, HandleId};
        let path = MeshAssetPath::generated(one.id);
        assert_eq!(path.generated_content_id(), Some(one.id));
        assert_eq!(path.content_id(), HandleId(one.id));
        assert_eq!(MeshAssetPath::new("meshes/a.mesh").generated_content_id(), None);
    }

    #[test]
    fn material_ids_get_distinct_surfaces() {
        let colors: Vec<_> = (0..8).map(|id| voxel_material_surface(id).base_color).collect();
        for (i, a) in colors.iter().enumerate() {
            for b in &colors[i + 1..] {
                assert_ne!(a, b);
            }
        }
    }
}
