//! CPU reference for feature-preserving manifold dual contouring.
//!
//! The reference follows [Dual Contouring of Hermite Data] and Schaefer, Ju,
//! and Warren's [Manifold Dual Contouring] construction: the
//! triangles of a manifold Marching Cubes case define distinct surface
//! components inside a cell, each component receives its own Hermite/QEF
//! vertex, and every sign-changing grid edge emits one dual polygon through
//! the four incident component vertices. This is intentionally separate from
//! the production GPU prototype so executable topology tests can reject an
//! invalid table or ownership convention first.
//!
//! [Dual Contouring of Hermite Data]: https://people.engr.tamu.edu/schaefer/research/dualcontour.pdf
//! [Manifold Dual Contouring]: https://people.engr.tamu.edu/schaefer/research/dualsimp_tvcg.pdf

use crate::{
    regular_case_from_fixture, GpuTerrainVertex, EXTRACTION_HALO, EXTRACTION_SAMPLE_COUNT,
    EXTRACTION_SAMPLE_EDGE, TRANSVOXEL_REGULAR_CORNERS,
};
use helio_planet_voxel_core::{CellWord, PAGE_EDGE};
use std::collections::{BTreeMap, BTreeSet};

pub const MANIFOLD_DC_CUBE_EDGE_COUNT: usize = 12;
pub const MANIFOLD_DC_MAX_COMPONENTS_PER_CELL: usize = 4;
pub const MANIFOLD_DC_CELL_HALO_MIN: i32 = -1;
pub const MANIFOLD_DC_CELL_EDGE: usize = PAGE_EDGE + 1;
pub const MANIFOLD_DC_CELL_COUNT: usize =
    MANIFOLD_DC_CELL_EDGE * MANIFOLD_DC_CELL_EDGE * MANIFOLD_DC_CELL_EDGE;
pub const MANIFOLD_DC_OWNED_EDGE_COUNT: usize = PAGE_EDGE * PAGE_EDGE * PAGE_EDGE * 3;
pub const MANIFOLD_DC_MAX_QEF_VERTICES: usize =
    MANIFOLD_DC_CELL_COUNT * MANIFOLD_DC_MAX_COMPONENTS_PER_CELL;
pub const MANIFOLD_DC_MAX_QUADS: usize = MANIFOLD_DC_OWNED_EDGE_COUNT;
pub const MANIFOLD_DC_INDICES_PER_QUAD: usize = 24;
pub const MANIFOLD_DC_MAX_VERTICES: usize =
    MANIFOLD_DC_MAX_QEF_VERTICES + MANIFOLD_DC_MAX_QUADS * 9;
pub const MANIFOLD_DC_MAX_INDICES: usize = MANIFOLD_DC_MAX_QUADS * MANIFOLD_DC_INDICES_PER_QUAD;
pub const MANIFOLD_DC_MAX_GPU_VERTICES: usize = MANIFOLD_DC_MAX_QUADS * 9;
/// Cell-local QEF coordinates use an open dyadic fixed-point grid. This makes
/// a halo-cell vertex and its owning-page copy bit-identical after integer
/// page translation, and prevents distinct cell vertices from geometrically
/// collapsing when the zero surface passes exactly through a grid sample.
/// The maximum added displacement is 1/65536 cell per coordinate.
pub const MANIFOLD_DC_POSITION_QUANTIZATION_STEPS: f64 = 65_536.0;

const INVALID_COMPONENT: u8 = u8::MAX;

/// Cube edges in the binary corner order used by the Transvoxel regular
/// tables: corner index = x + 2*y + 4*z.
pub const MANIFOLD_DC_CUBE_EDGES: [[u8; 2]; MANIFOLD_DC_CUBE_EDGE_COUNT] = [
    [0, 1],
    [2, 3],
    [4, 5],
    [6, 7],
    [0, 2],
    [1, 3],
    [4, 6],
    [5, 7],
    [0, 4],
    [1, 5],
    [2, 6],
    [3, 7],
];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ManifoldDcCellTopology {
    fixture_case: u8,
    component_count: u8,
    edge_components: [u8; MANIFOLD_DC_CUBE_EDGE_COUNT],
}

impl ManifoldDcCellTopology {
    pub const fn fixture_case(self) -> u8 {
        self.fixture_case
    }

    pub const fn component_count(self) -> u8 {
        self.component_count
    }

    pub fn component_for_edge(self, edge: usize) -> Option<u8> {
        self.edge_components
            .get(edge)
            .copied()
            .filter(|component| *component != INVALID_COMPONENT)
    }

    pub const fn edge_components(self) -> [u8; MANIFOLD_DC_CUBE_EDGE_COUNT] {
        self.edge_components
    }
}

pub fn manifold_dc_cell_topology(fixture_case: u8) -> ManifoldDcCellTopology {
    let topology = regular_case_from_fixture(fixture_case);
    let vertex_count = topology.vertex_count();
    let mut parents = [0_u8; MANIFOLD_DC_CUBE_EDGE_COUNT];
    let mut vertex_edges = [u8::MAX; MANIFOLD_DC_CUBE_EDGE_COUNT];
    for (vertex, edge_slot) in vertex_edges.iter_mut().enumerate().take(vertex_count) {
        parents[vertex] = vertex as u8;
        let endpoints = topology
            .vertex(vertex)
            .expect("vertex index is bounded by the table count")
            .endpoints();
        *edge_slot = cube_edge_index(endpoints)
            .expect("the audited regular table only references cube edges")
            as u8;
    }
    // Dual Marching Cubes vertices correspond to closed surface cycles on the
    // cube boundary, not merely to triangle-connected components in the
    // interior lookup. The latter can merge two distinct boundary cycles and
    // create parallel dual edges which collapse into non-manifold indexed
    // edges. Connect crossing edges only through the contour segment emitted
    // on each cube face; every resulting component is one boundary cycle.
    for axis in 0..3 {
        for positive_face in [false, true] {
            for triangle in 0..topology.triangle_count() {
                let triangle = topology
                    .triangle(triangle)
                    .expect("triangle index is bounded by the table count");
                let mut face_vertices = [usize::MAX; 3];
                let mut count = 0;
                for vertex in triangle {
                    let vertex = usize::from(vertex);
                    if cube_edge_lies_on_face(
                        usize::from(vertex_edges[vertex]),
                        axis,
                        positive_face,
                    ) {
                        face_vertices[count] = vertex;
                        count += 1;
                    }
                }
                if count == 2 {
                    union(&mut parents, face_vertices[0], face_vertices[1]);
                }
            }
        }
    }

    let mut root_components = [INVALID_COMPONENT; MANIFOLD_DC_CUBE_EDGE_COUNT];
    let mut edge_components = [INVALID_COMPONENT; MANIFOLD_DC_CUBE_EDGE_COUNT];
    let mut component_count = 0_u8;
    for (vertex, &vertex_edge) in vertex_edges.iter().enumerate().take(vertex_count) {
        let root = find(&mut parents, vertex);
        let component = if root_components[root] == INVALID_COMPONENT {
            let component = component_count;
            root_components[root] = component;
            component_count += 1;
            component
        } else {
            root_components[root]
        };
        let edge = usize::from(vertex_edge);
        assert_eq!(
            edge_components[edge], INVALID_COMPONENT,
            "a regular case must reference each crossing edge once"
        );
        edge_components[edge] = component;
    }
    debug_assert!(usize::from(component_count) <= MANIFOLD_DC_MAX_COMPONENTS_PER_CELL);
    ManifoldDcCellTopology {
        fixture_case,
        component_count,
        edge_components,
    }
}

fn cube_edge_lies_on_face(edge: usize, axis: usize, positive_face: bool) -> bool {
    let endpoints = MANIFOLD_DC_CUBE_EDGES[edge];
    let face_coordinate = u8::from(positive_face);
    TRANSVOXEL_REGULAR_CORNERS[usize::from(endpoints[0])][axis] == face_coordinate
        && TRANSVOXEL_REGULAR_CORNERS[usize::from(endpoints[1])][axis] == face_coordinate
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ManifoldDcTableAudit {
    pub cases: u16,
    pub active_cases: u16,
    pub components: u16,
    pub max_components: u8,
    pub face_pairing_mismatches: u32,
}

pub fn audit_manifold_dc_table() -> ManifoldDcTableAudit {
    let mut audit = ManifoldDcTableAudit::default();
    for fixture_case in 0..=u8::MAX {
        let topology = manifold_dc_cell_topology(fixture_case);
        audit.cases += 1;
        if topology.component_count != 0 {
            audit.active_cases += 1;
        }
        audit.components += u16::from(topology.component_count);
        audit.max_components = audit.max_components.max(topology.component_count);
        for (edge, endpoints) in MANIFOLD_DC_CUBE_EDGES.into_iter().enumerate() {
            let crosses = corner_is_solid(fixture_case, endpoints[0])
                != corner_is_solid(fixture_case, endpoints[1]);
            assert_eq!(topology.component_for_edge(edge).is_some(), crosses);
        }
    }
    audit.face_pairing_mismatches = count_face_pairing_mismatches();
    audit
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ManifoldDcVertex {
    /// Geometric vertex data. `material` is the deterministic dominant
    /// material for diagnostics; [`ManifoldDcMesh::gpu_mesh`] replaces it
    /// with each quad's authoritative solid-side material.
    pub gpu: GpuTerrainVertex,
    pub qef_error: f32,
    pub cell_min: [i32; 3],
    pub component: u8,
    pub hermite_count: u8,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ManifoldDcQuad {
    pub vertices: [u32; 4],
    pub material: u8,
    pub axis: u8,
    pub edge_min: [u8; 3],
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct ManifoldDcMesh {
    pub vertices: Vec<ManifoldDcVertex>,
    pub quads: Vec<ManifoldDcQuad>,
    pub indices: Vec<u32>,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct ManifoldDcGpuMesh {
    pub vertices: Vec<GpuTerrainVertex>,
    pub indices: Vec<u32>,
}

impl ManifoldDcMesh {
    /// Produces Helio's indexed render contract while preserving attribute
    /// seams. Vertices are welded by source QEF vertex and material, never by
    /// position alone.
    pub fn gpu_mesh(&self) -> Result<ManifoldDcGpuMesh, ManifoldDcError> {
        let expected_indices = self
            .quads
            .len()
            .saturating_mul(MANIFOLD_DC_INDICES_PER_QUAD);
        if self.indices.len() != expected_indices {
            return Err(ManifoldDcError::MeshIndexCount {
                actual: self.indices.len(),
                expected: expected_indices,
            });
        }
        if let Some(index) = self
            .indices
            .iter()
            .copied()
            .find(|index| *index as usize >= self.vertices.len())
        {
            return Err(ManifoldDcError::MeshIndex {
                index,
                vertices: self.vertices.len(),
            });
        }
        let mut output = ManifoldDcGpuMesh {
            vertices: Vec::with_capacity(self.vertices.len().min(MANIFOLD_DC_MAX_GPU_VERTICES)),
            indices: Vec::with_capacity(self.indices.len()),
        };
        let mut materials_by_source = vec![BTreeSet::new(); self.vertices.len()];
        for (quad_index, quad) in self.quads.iter().enumerate() {
            let first_index = quad_index * MANIFOLD_DC_INDICES_PER_QUAD;
            for &source_index in
                &self.indices[first_index..first_index + MANIFOLD_DC_INDICES_PER_QUAD]
            {
                materials_by_source[source_index as usize].insert(quad.material);
            }
        }
        let mut remap = BTreeMap::new();
        for (source_index, materials) in materials_by_source.into_iter().enumerate() {
            for material in materials {
                let mut vertex = self.vertices[source_index].gpu;
                vertex.material = u32::from(material);
                let output_index = output.vertices.len() as u32;
                output.vertices.push(vertex);
                remap.insert((source_index as u32, material), output_index);
            }
        }
        for (quad_index, quad) in self.quads.iter().enumerate() {
            let first_index = quad_index * MANIFOLD_DC_INDICES_PER_QUAD;
            for source_index in
                &self.indices[first_index..first_index + MANIFOLD_DC_INDICES_PER_QUAD]
            {
                let key = (*source_index, quad.material);
                let output_index = remap[&key];
                output.indices.push(output_index);
            }
        }
        Ok(output)
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ManifoldDcMeshAudit {
    pub vertices: u32,
    pub quads: u32,
    pub triangles: u32,
    pub invalid_indices: u32,
    pub degenerate_triangles: u32,
    pub boundary_edges: u32,
    pub nonmanifold_edges: u32,
    pub nonmanifold_vertices: u32,
}

impl ManifoldDcMeshAudit {
    pub const fn is_two_manifold(self) -> bool {
        self.invalid_indices == 0
            && self.degenerate_triangles == 0
            && self.nonmanifold_edges == 0
            && self.nonmanifold_vertices == 0
    }

    pub const fn is_closed_two_manifold(self) -> bool {
        self.is_two_manifold() && self.boundary_edges == 0
    }
}

/// Audits triangle-edge incidence and every used vertex link. A valid
/// 2-manifold vertex has one connected link that is either a cycle (interior)
/// or a path (mesh boundary).
pub fn audit_manifold_dc_mesh(mesh: &ManifoldDcMesh) -> ManifoldDcMeshAudit {
    let mut audit = ManifoldDcMeshAudit {
        vertices: mesh.vertices.len().try_into().unwrap_or(u32::MAX),
        quads: mesh.quads.len().try_into().unwrap_or(u32::MAX),
        triangles: (mesh.indices.len() / 3).try_into().unwrap_or(u32::MAX),
        ..Default::default()
    };
    if !mesh.indices.len().is_multiple_of(3) {
        audit.invalid_indices += 1;
    }
    let mut edge_faces: BTreeMap<[u32; 2], u32> = BTreeMap::new();
    let mut links: Vec<BTreeMap<u32, BTreeSet<u32>>> = vec![BTreeMap::new(); mesh.vertices.len()];
    for triangle in mesh.indices.chunks_exact(3) {
        let [a, b, c] = [triangle[0], triangle[1], triangle[2]];
        if [a, b, c]
            .into_iter()
            .any(|vertex| vertex as usize >= mesh.vertices.len())
        {
            audit.invalid_indices += 1;
            continue;
        }
        if a == b || b == c || c == a {
            audit.degenerate_triangles += 1;
            continue;
        }
        let first = mesh.vertices[a as usize].gpu.position;
        let second = mesh.vertices[b as usize].gpu.position;
        let third = mesh.vertices[c as usize].gpu.position;
        if triangle_area_squared(first, second, third) <= 1.0e-20 {
            audit.degenerate_triangles += 1;
            continue;
        }
        for mut edge in [[a, b], [b, c], [c, a]] {
            edge.sort_unstable();
            *edge_faces.entry(edge).or_default() += 1;
        }
        for (center, left, right) in [(a, b, c), (b, c, a), (c, a, b)] {
            links[center as usize]
                .entry(left)
                .or_default()
                .insert(right);
            links[center as usize]
                .entry(right)
                .or_default()
                .insert(left);
        }
    }
    for count in edge_faces.values() {
        match count {
            1 => audit.boundary_edges += 1,
            2 => {}
            _ => audit.nonmanifold_edges += 1,
        }
    }
    for link in links {
        if link.is_empty() {
            continue;
        }
        let degree_one = link
            .values()
            .filter(|neighbors| neighbors.len() == 1)
            .count();
        let degree_valid = link
            .values()
            .all(|neighbors| matches!(neighbors.len(), 1 | 2));
        let boundary_shape = degree_one == 2;
        let interior_shape = degree_one == 0 && link.values().all(|neighbors| neighbors.len() == 2);
        let start = *link.keys().next().expect("a nonempty link has a vertex");
        let mut visited = BTreeSet::new();
        let mut stack = vec![start];
        while let Some(current) = stack.pop() {
            if visited.insert(current) {
                stack.extend(link[&current].iter().copied());
            }
        }
        if !degree_valid || !(boundary_shape || interior_shape) || visited.len() != link.len() {
            audit.nonmanifold_vertices += 1;
        }
    }
    audit
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
pub enum ManifoldDcError {
    #[error("manifold dual contouring received {actual} scalar samples; expected {expected}")]
    SampleCount { actual: usize, expected: usize },
    #[error("manifold dual contouring table requires more than {max} components in one cell")]
    ComponentCapacity { max: usize },
    #[error(
        "manifold dual contouring mesh has {actual} indices; expected {expected} for its quads"
    )]
    MeshIndexCount { actual: usize, expected: usize },
    #[error("manifold dual contouring mesh index {index} exceeds its {vertices} vertices")]
    MeshIndex { index: u32, vertices: usize },
    #[error("manifold dual contouring could not resolve a component for owned edge {edge_min:?} axis {axis}")]
    MissingEdgeComponent { edge_min: [u8; 3], axis: u8 },
    #[error("manifold dual contouring could not resolve the face contour paired with cube edge {edge} on axis {axis} ({positive_face})")]
    MissingFacePairing {
        edge: usize,
        axis: usize,
        positive_face: bool,
    },
    #[error("manifold dual contouring paired one face segment with inconsistent QEF endpoints {first:?} and {second:?}")]
    FaceSegmentEndpointMismatch { first: [u32; 2], second: [u32; 2] },
    #[error(
        "manifold dual contouring produced an irreducible non-manifold link at vertex {vertex}"
    )]
    NonManifoldVertexLink { vertex: u32 },
}

pub fn extract_manifold_dc_page(samples: &[CellWord]) -> Result<ManifoldDcMesh, ManifoldDcError> {
    if samples.len() != EXTRACTION_SAMPLE_COUNT {
        return Err(ManifoldDcError::SampleCount {
            actual: samples.len(),
            expected: EXTRACTION_SAMPLE_COUNT,
        });
    }
    let mut mesh = ManifoldDcMesh {
        vertices: Vec::with_capacity(MANIFOLD_DC_MAX_VERTICES),
        quads: Vec::with_capacity(MANIFOLD_DC_MAX_QUADS),
        indices: Vec::with_capacity(MANIFOLD_DC_MAX_INDICES),
    };
    let mut cells = vec![CellRecord::default(); MANIFOLD_DC_CELL_COUNT];
    for z in MANIFOLD_DC_CELL_HALO_MIN..PAGE_EDGE as i32 {
        for y in MANIFOLD_DC_CELL_HALO_MIN..PAGE_EDGE as i32 {
            for x in MANIFOLD_DC_CELL_HALO_MIN..PAGE_EDGE as i32 {
                let cell_min = [x, y, z];
                let words = cell_words(samples, cell_min);
                let fixture_case = fixture_case(words);
                let topology = manifold_dc_cell_topology(fixture_case);
                if usize::from(topology.component_count()) > MANIFOLD_DC_MAX_COMPONENTS_PER_CELL {
                    return Err(ManifoldDcError::ComponentCapacity {
                        max: MANIFOLD_DC_MAX_COMPONENTS_PER_CELL,
                    });
                }
                let first_vertex = mesh.vertices.len() as u32;
                for component in 0..topology.component_count() {
                    mesh.vertices
                        .push(solve_component_vertex(cell_min, words, topology, component));
                }
                cells[cell_index(cell_min)] = CellRecord {
                    topology,
                    first_vertex,
                };
            }
        }
    }

    for z in 0..PAGE_EDGE as u8 {
        for y in 0..PAGE_EDGE as u8 {
            for x in 0..PAGE_EDGE as u8 {
                let edge_min = [x, y, z];
                for axis in 0..3_u8 {
                    emit_owned_edge(samples, &cells, edge_min, axis, &mut mesh)?;
                }
            }
        }
    }
    triangulate_manifold_cell_complex(&cells, &mut mesh)?;
    split_disconnected_vertex_links(&mut mesh)?;
    compact_mesh_vertices(&mut mesh);
    Ok(mesh)
}

fn compact_mesh_vertices(mesh: &mut ManifoldDcMesh) {
    let mut used = vec![false; mesh.vertices.len()];
    for quad in &mesh.quads {
        for vertex in quad.vertices {
            used[vertex as usize] = true;
        }
    }
    for &vertex in &mesh.indices {
        used[vertex as usize] = true;
    }
    let mut remap = vec![u32::MAX; mesh.vertices.len()];
    let mut compacted = Vec::with_capacity(used.iter().filter(|is_used| **is_used).count());
    for (old, vertex) in core::mem::take(&mut mesh.vertices).into_iter().enumerate() {
        if used[old] {
            remap[old] = compacted.len() as u32;
            compacted.push(vertex);
        }
    }
    for quad in &mut mesh.quads {
        for vertex in &mut quad.vertices {
            *vertex = remap[*vertex as usize];
        }
    }
    for index in &mut mesh.indices {
        *index = remap[*index as usize];
    }
    mesh.vertices = compacted;
}

#[derive(Clone, Copy, Debug)]
struct CellRecord {
    topology: ManifoldDcCellTopology,
    first_vertex: u32,
}

impl Default for CellRecord {
    fn default() -> Self {
        Self {
            topology: manifold_dc_cell_topology(0),
            first_vertex: 0,
        }
    }
}

fn emit_owned_edge(
    samples: &[CellWord],
    cells: &[CellRecord],
    edge_min: [u8; 3],
    axis: u8,
    mesh: &mut ManifoldDcMesh,
) -> Result<(), ManifoldDcError> {
    let mut edge_max = edge_min.map(i32::from);
    edge_max[usize::from(axis)] += 1;
    let edge_min_i32 = edge_min.map(i32::from);
    let first = sample(samples, edge_min_i32);
    let second = sample(samples, edge_max);
    if first.is_solid() == second.is_solid() {
        return Ok(());
    }
    let incident = incident_cells(edge_min_i32, axis);
    let mut vertices = [0_u32; 4];
    for (corner, cell_min) in incident.into_iter().enumerate() {
        let record = cells[cell_index(cell_min)];
        let local_first = subtract(edge_min_i32, cell_min);
        let local_second = subtract(edge_max, cell_min);
        let first_corner =
            regular_corner_index(local_first).expect("incident cell edge begins on a cube corner");
        let second_corner =
            regular_corner_index(local_second).expect("incident cell edge ends on a cube corner");
        let edge = cube_edge_index([first_corner, second_corner])
            .expect("incident cell endpoints form a cube edge");
        let component = record
            .topology
            .component_for_edge(edge)
            .ok_or(ManifoldDcError::MissingEdgeComponent { edge_min, axis })?;
        vertices[corner] = record.first_vertex + u32::from(component);
    }
    if !first.is_solid() {
        vertices.reverse();
    }
    let material = if first.is_solid() {
        first.material()
    } else {
        second.material()
    };
    let quad = ManifoldDcQuad {
        vertices,
        material,
        axis,
        edge_min,
    };
    mesh.quads.push(quad);
    Ok(())
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct GridEdgeKey {
    min: [i32; 3],
    axis: u8,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct DualEdgeKey {
    first: GridEdgeKey,
    second: GridEdgeKey,
}

/// Converts the manifold DMC polygonal cell complex into a simplicial mesh.
/// A DMC surface can legitimately contain multiple topological edges between
/// the same two QEF vertices. Directly triangulating its quads would collapse
/// those parallel edges into one indexed edge with more than two incident
/// triangles. Barycentric subdivision gives every face-contour segment one
/// canonical midpoint and every dual quad one center, preserving each edge's
/// identity in a conventional indexed triangle mesh.
fn triangulate_manifold_cell_complex(
    cells: &[CellRecord],
    mesh: &mut ManifoldDcMesh,
) -> Result<(), ManifoldDcError> {
    let mut midpoint_by_edge = BTreeMap::<DualEdgeKey, (u32, [u32; 2])>::new();
    let quads = mesh.quads.clone();
    for quad in &quads {
        let mut midpoints = [0_u32; 4];
        for (side, midpoint) in midpoints.iter_mut().enumerate() {
            let key = dual_edge_key(cells, &mesh.vertices, quad, side)?;
            let mut endpoints = [quad.vertices[side], quad.vertices[(side + 1) % 4]];
            endpoints.sort_unstable();
            *midpoint = if let Some((index, existing)) = midpoint_by_edge.get(&key).copied() {
                if existing != endpoints {
                    return Err(ManifoldDcError::FaceSegmentEndpointMismatch {
                        first: existing,
                        second: endpoints,
                    });
                }
                index
            } else {
                let first = quad.vertices[side];
                let second = quad.vertices[(side + 1) % 4];
                let index = mesh.vertices.len() as u32;
                mesh.vertices
                    .push(averaged_vertex([first, second].into_iter(), &mesh.vertices));
                midpoint_by_edge.insert(key, (index, endpoints));
                index
            };
        }
        let center = mesh.vertices.len() as u32;
        mesh.vertices
            .push(averaged_vertex(quad.vertices.into_iter(), &mesh.vertices));
        for (side, &midpoint) in midpoints.iter().enumerate() {
            let first = quad.vertices[side];
            let second = quad.vertices[(side + 1) % 4];
            mesh.indices
                .extend([first, midpoint, center, midpoint, second, center]);
        }
    }
    Ok(())
}

fn dual_edge_key(
    cells: &[CellRecord],
    vertices: &[ManifoldDcVertex],
    quad: &ManifoldDcQuad,
    side: usize,
) -> Result<DualEdgeKey, ManifoldDcError> {
    let edge_min = quad.edge_min.map(i32::from);
    let cell_min = vertices[quad.vertices[side] as usize].cell_min;
    let neighbor = vertices[quad.vertices[(side + 1) % 4] as usize].cell_min;
    let face_axis = (0..3)
        .find(|axis| cell_min[*axis] != neighbor[*axis])
        .expect("consecutive incident cells share one face");
    let positive_face = neighbor[face_axis] > cell_min[face_axis];
    let mut edge_max = edge_min;
    edge_max[usize::from(quad.axis)] += 1;
    let first_corner = regular_corner_index(subtract(edge_min, cell_min))
        .expect("an incident cell contains the owned edge's first corner");
    let second_corner = regular_corner_index(subtract(edge_max, cell_min))
        .expect("an incident cell contains the owned edge's second corner");
    let current_edge = cube_edge_index([first_corner, second_corner])
        .expect("the owned edge is a cube edge in every incident cell");
    let paired_edge = paired_cube_edge_on_face(
        cells[cell_index(cell_min)].topology,
        current_edge,
        face_axis,
        positive_face,
    )
    .ok_or(ManifoldDcError::MissingFacePairing {
        edge: current_edge,
        axis: face_axis,
        positive_face,
    })?;
    let mut pair = [
        grid_edge_key(cell_min, current_edge),
        grid_edge_key(cell_min, paired_edge),
    ];
    pair.sort_unstable();
    Ok(DualEdgeKey {
        first: pair[0],
        second: pair[1],
    })
}

fn paired_cube_edge_on_face(
    topology: ManifoldDcCellTopology,
    current_edge: usize,
    axis: usize,
    positive_face: bool,
) -> Option<usize> {
    let regular = regular_case_from_fixture(topology.fixture_case());
    for triangle_index in 0..regular.triangle_count() {
        let triangle = regular.triangle(triangle_index)?;
        let mut face_edges = [usize::MAX; 3];
        let mut count = 0;
        for vertex in triangle {
            let endpoints = regular.vertex(usize::from(vertex))?.endpoints();
            let edge = cube_edge_index(endpoints)?;
            if cube_edge_lies_on_face(edge, axis, positive_face) {
                face_edges[count] = edge;
                count += 1;
            }
        }
        if count == 2 {
            if face_edges[0] == current_edge {
                return Some(face_edges[1]);
            }
            if face_edges[1] == current_edge {
                return Some(face_edges[0]);
            }
        }
    }
    None
}

fn grid_edge_key(cell_min: [i32; 3], cube_edge: usize) -> GridEdgeKey {
    let endpoints = MANIFOLD_DC_CUBE_EDGES[cube_edge];
    let first = TRANSVOXEL_REGULAR_CORNERS[usize::from(endpoints[0])];
    let second = TRANSVOXEL_REGULAR_CORNERS[usize::from(endpoints[1])];
    let mut min = [0_i32; 3];
    let mut axis = 0_u8;
    for coordinate in 0..3 {
        min[coordinate] =
            cell_min[coordinate] + i32::from(first[coordinate].min(second[coordinate]));
        if first[coordinate] != second[coordinate] {
            axis = coordinate as u8;
        }
    }
    GridEdgeKey { min, axis }
}

fn averaged_vertex(
    indices: impl Iterator<Item = u32> + Clone,
    vertices: &[ManifoldDcVertex],
) -> ManifoldDcVertex {
    let count = indices.clone().count() as f64;
    let first = vertices[indices
        .clone()
        .next()
        .expect("an average has at least one vertex") as usize];
    let mut position = [0.0_f64; 3];
    let mut normal = [0.0_f64; 3];
    let mut qef_error = 0.0_f64;
    for index in indices {
        let vertex = vertices[index as usize];
        for axis in 0..3 {
            position[axis] += f64::from(vertex.gpu.position[axis]);
            normal[axis] += f64::from(vertex.gpu.normal[axis]);
        }
        qef_error += f64::from(vertex.qef_error);
    }
    for coordinate in &mut position {
        *coordinate /= count;
    }
    let normal = normalize_or(normal, [0.0, 1.0, 0.0]);
    ManifoldDcVertex {
        gpu: GpuTerrainVertex {
            position: position.map(|coordinate| coordinate as f32),
            material: first.gpu.material,
            normal: normal.map(|coordinate| coordinate as f32),
            flags: first.gpu.flags,
        },
        qef_error: (qef_error / count) as f32,
        cell_min: first.cell_min,
        component: u8::MAX,
        hermite_count: 0,
    }
}

fn split_disconnected_vertex_links(mesh: &mut ManifoldDcMesh) -> Result<(), ManifoldDcError> {
    let original_vertex_count = mesh.vertices.len();
    let mut links: Vec<BTreeMap<u32, BTreeSet<u32>>> = vec![BTreeMap::new(); original_vertex_count];
    for triangle in mesh.indices.chunks_exact(3) {
        let [a, b, c] = [triangle[0], triangle[1], triangle[2]];
        for (center, left, right) in [(a, b, c), (b, c, a), (c, a, b)] {
            if center as usize >= original_vertex_count {
                continue;
            }
            links[center as usize]
                .entry(left)
                .or_default()
                .insert(right);
            links[center as usize]
                .entry(right)
                .or_default()
                .insert(left);
        }
    }

    let mut component_vertices = vec![Vec::<u32>::new(); original_vertex_count];
    let mut component_by_neighbor = vec![BTreeMap::<u32, usize>::new(); original_vertex_count];
    for (vertex, link) in links.iter().enumerate() {
        let mut remaining: BTreeSet<u32> = link.keys().copied().collect();
        while let Some(start) = remaining.pop_first() {
            let mut component = BTreeSet::new();
            let mut stack = vec![start];
            while let Some(current) = stack.pop() {
                if component.insert(current) {
                    remaining.remove(&current);
                    stack.extend(link[&current].iter().copied());
                }
            }
            let degree_one = component
                .iter()
                .filter(|node| {
                    link[node]
                        .iter()
                        .filter(|next| component.contains(next))
                        .count()
                        == 1
                })
                .count();
            let degrees_valid = component.iter().all(|node| {
                matches!(
                    link[node]
                        .iter()
                        .filter(|next| component.contains(next))
                        .count(),
                    1 | 2
                )
            });
            let is_path = degree_one == 2;
            let is_cycle = degree_one == 0
                && component.iter().all(|node| {
                    link[node]
                        .iter()
                        .filter(|next| component.contains(next))
                        .count()
                        == 2
                });
            if !degrees_valid || !(is_path || is_cycle) {
                return Err(ManifoldDcError::NonManifoldVertexLink {
                    vertex: vertex as u32,
                });
            }
            let component_index = component_vertices[vertex].len();
            let output_vertex = if component_index == 0 {
                vertex as u32
            } else {
                let output = mesh.vertices.len() as u32;
                mesh.vertices.push(mesh.vertices[vertex]);
                output
            };
            component_vertices[vertex].push(output_vertex);
            for neighbor in component {
                component_by_neighbor[vertex].insert(neighbor, component_index);
            }
        }
    }

    for triangle in mesh.indices.chunks_exact_mut(3) {
        let original = [triangle[0], triangle[1], triangle[2]];
        for corner in 0..3 {
            let vertex = original[corner] as usize;
            if vertex >= original_vertex_count || component_vertices[vertex].len() <= 1 {
                continue;
            }
            let neighbor = original[(corner + 1) % 3];
            let component = component_by_neighbor[vertex][&neighbor];
            triangle[corner] = component_vertices[vertex][component];
        }
    }
    Ok(())
}

fn solve_component_vertex(
    cell_min: [i32; 3],
    words: [CellWord; 8],
    topology: ManifoldDcCellTopology,
    component: u8,
) -> ManifoldDcVertex {
    let densities = words.map(|word| f64::from(word.density()));
    let mut hermite = Vec::with_capacity(MANIFOLD_DC_CUBE_EDGE_COUNT);
    for (edge, endpoints) in MANIFOLD_DC_CUBE_EDGES.into_iter().enumerate() {
        if topology.component_for_edge(edge) != Some(component) {
            continue;
        }
        let first_corner = TRANSVOXEL_REGULAR_CORNERS[usize::from(endpoints[0])];
        let second_corner = TRANSVOXEL_REGULAR_CORNERS[usize::from(endpoints[1])];
        let first_density = densities[usize::from(endpoints[0])];
        let second_density = densities[usize::from(endpoints[1])];
        let denominator = first_density - second_density;
        let interpolation = if denominator.abs() > f64::EPSILON {
            (first_density / denominator).clamp(0.0, 1.0)
        } else {
            0.5
        };
        let mut point = [0.0_f64; 3];
        for coordinate in 0..3 {
            point[coordinate] = f64::from(first_corner[coordinate])
                + (f64::from(second_corner[coordinate]) - f64::from(first_corner[coordinate]))
                    * interpolation;
        }
        let normal = trilinear_gradient(densities, point);
        hermite.push(HermiteSample { point, normal });
    }
    let solution = solve_bounded_qef(&hermite).map(quantize_unit_coordinate);
    let mut position = [0.0_f32; 3];
    let mut normal_sum = [0.0_f64; 3];
    for axis in 0..3 {
        position[axis] = (f64::from(cell_min[axis]) + solution[axis]) as f32;
        for sample in &hermite {
            normal_sum[axis] += sample.normal[axis];
        }
    }
    let normal = normalize_or(normal_sum, [0.0, 1.0, 0.0]).map(|value| value as f32);
    let material = dominant_material(words, topology, component);
    ManifoldDcVertex {
        gpu: GpuTerrainVertex {
            position,
            material: u32::from(material),
            normal,
            flags: 0,
        },
        qef_error: qef_error(&hermite, solution) as f32,
        cell_min,
        component,
        hermite_count: hermite.len() as u8,
    }
}

fn dominant_material(words: [CellWord; 8], topology: ManifoldDcCellTopology, component: u8) -> u8 {
    let mut counts = [0_u8; 256];
    for (edge, endpoints) in MANIFOLD_DC_CUBE_EDGES.into_iter().enumerate() {
        if topology.component_for_edge(edge) != Some(component) {
            continue;
        }
        let first = words[usize::from(endpoints[0])];
        let second = words[usize::from(endpoints[1])];
        let material = if first.is_solid() {
            first.material()
        } else {
            second.material()
        };
        counts[usize::from(material)] = counts[usize::from(material)].saturating_add(1);
    }
    counts
        .into_iter()
        .enumerate()
        .max_by_key(|(material, count)| (*count, core::cmp::Reverse(*material)))
        .map(|(material, _)| material as u8)
        .unwrap_or(0)
}

#[derive(Clone, Copy, Debug)]
struct HermiteSample {
    point: [f64; 3],
    normal: [f64; 3],
}

fn solve_bounded_qef(samples: &[HermiteSample]) -> [f64; 3] {
    let mut mass = [0.0_f64; 3];
    for sample in samples {
        for (axis, coordinate) in mass.iter_mut().enumerate() {
            *coordinate += sample.point[axis];
        }
    }
    for coordinate in &mut mass {
        *coordinate /= samples.len().max(1) as f64;
    }
    let mut best = mass;
    let mut best_error = qef_error(samples, best);
    for mask in 0_u8..27 {
        let mut code = mask;
        let mut fixed = [None; 3];
        for coordinate in &mut fixed {
            *coordinate = match code % 3 {
                0 => None,
                1 => Some(0.0),
                _ => Some(1.0),
            };
            code /= 3;
        }
        let candidate = solve_qef_with_fixed(samples, mass, fixed);
        if candidate
            .iter()
            .any(|value| !(-1.0e-9..=1.0 + 1.0e-9).contains(value))
        {
            continue;
        }
        let error = qef_error(samples, candidate);
        if error + 1.0e-12 < best_error {
            best = candidate;
            best_error = error;
        }
    }
    best.map(|value| value.clamp(0.0, 1.0))
}

fn solve_qef_with_fixed(
    samples: &[HermiteSample],
    mass: [f64; 3],
    fixed: [Option<f64>; 3],
) -> [f64; 3] {
    let free: Vec<usize> = (0..3).filter(|axis| fixed[*axis].is_none()).collect();
    let mut result = mass;
    for axis in 0..3 {
        if let Some(value) = fixed[axis] {
            result[axis] = value;
        }
    }
    if free.is_empty() {
        return result;
    }
    let mut matrix = [[0.0_f64; 3]; 3];
    let mut rhs = [0.0_f64; 3];
    for sample in samples {
        let plane = dot(sample.normal, sample.point);
        let fixed_plane: f64 = (0..3)
            .filter_map(|axis| fixed[axis].map(|value| sample.normal[axis] * value))
            .sum();
        for (row, axis) in free.iter().copied().enumerate() {
            rhs[row] += sample.normal[axis] * (plane - fixed_plane);
            for (column, other_axis) in free.iter().copied().enumerate() {
                matrix[row][column] += sample.normal[axis] * sample.normal[other_axis];
            }
        }
    }
    let mut centered_rhs = rhs;
    for (row, _) in free.iter().enumerate() {
        for (column, axis) in free.iter().copied().enumerate() {
            centered_rhs[row] -= matrix[row][column] * mass[axis];
        }
    }
    let offset = pseudo_inverse_symmetric(matrix, centered_rhs, free.len());
    for (index, axis) in free.into_iter().enumerate() {
        result[axis] = mass[axis] + offset[index];
    }
    result
}

fn pseudo_inverse_symmetric(
    mut matrix: [[f64; 3]; 3],
    rhs: [f64; 3],
    dimension: usize,
) -> [f64; 3] {
    let mut eigenvectors = [[0.0_f64; 3]; 3];
    for (axis, row) in eigenvectors.iter_mut().enumerate().take(dimension) {
        row[axis] = 1.0;
    }
    for _ in 0..10 {
        for (p, q) in [(0, 1), (0, 2), (1, 2)] {
            if p >= dimension || q >= dimension {
                continue;
            }
            let off_diagonal = matrix[p][q];
            if off_diagonal.abs() <= 1.0e-15 {
                continue;
            }
            let tau = (matrix[q][q] - matrix[p][p]) / (2.0 * off_diagonal);
            let tangent = if tau >= 0.0 {
                1.0 / (tau + (1.0 + tau * tau).sqrt())
            } else {
                -1.0 / (-tau + (1.0 + tau * tau).sqrt())
            };
            let cosine = 1.0 / (1.0 + tangent * tangent).sqrt();
            let sine = tangent * cosine;
            let app = matrix[p][p];
            let aqq = matrix[q][q];
            matrix[p][p] = app - tangent * off_diagonal;
            matrix[q][q] = aqq + tangent * off_diagonal;
            matrix[p][q] = 0.0;
            matrix[q][p] = 0.0;
            let mut updated_p = [0.0_f64; 3];
            let mut updated_q = [0.0_f64; 3];
            for (r, row) in matrix.iter().enumerate().take(dimension) {
                if r == p || r == q {
                    continue;
                }
                updated_p[r] = cosine * row[p] - sine * row[q];
                updated_q[r] = sine * row[p] + cosine * row[q];
            }
            for (r, (&value_p, &value_q)) in
                updated_p.iter().zip(&updated_q).enumerate().take(dimension)
            {
                if r == p || r == q {
                    continue;
                }
                matrix[r][p] = value_p;
                matrix[p][r] = value_p;
                matrix[r][q] = value_q;
                matrix[q][r] = value_q;
            }
            for row in eigenvectors.iter_mut().take(dimension) {
                let vrp = row[p];
                let vrq = row[q];
                row[p] = cosine * vrp - sine * vrq;
                row[q] = sine * vrp + cosine * vrq;
            }
        }
    }
    let largest = (0..dimension)
        .map(|axis| matrix[axis][axis].abs())
        .fold(0.0_f64, f64::max);
    let threshold = largest.max(1.0) * 1.0e-6;
    let mut result = [0.0_f64; 3];
    for eigen in 0..dimension {
        let value = matrix[eigen][eigen];
        if value.abs() <= threshold {
            continue;
        }
        let projection: f64 = (0..dimension)
            .map(|row| eigenvectors[row][eigen] * rhs[row])
            .sum();
        for (row, result_value) in result.iter_mut().enumerate().take(dimension) {
            *result_value += eigenvectors[row][eigen] * projection / value;
        }
    }
    result
}

fn qef_error(samples: &[HermiteSample], point: [f64; 3]) -> f64 {
    samples
        .iter()
        .map(|sample| {
            let distance = dot(sample.normal, subtract_f64(point, sample.point));
            distance * distance
        })
        .sum()
}

fn quantize_unit_coordinate(value: f64) -> f64 {
    let step = (value.clamp(0.0, 1.0) * MANIFOLD_DC_POSITION_QUANTIZATION_STEPS)
        .round()
        .clamp(1.0, MANIFOLD_DC_POSITION_QUANTIZATION_STEPS - 1.0);
    step / MANIFOLD_DC_POSITION_QUANTIZATION_STEPS
}

fn trilinear_gradient(density: [f64; 8], point: [f64; 3]) -> [f64; 3] {
    let [x, y, z] = point;
    let gx = bilerp(
        density[1] - density[0],
        density[3] - density[2],
        density[5] - density[4],
        density[7] - density[6],
        y,
        z,
    );
    let gy = bilerp(
        density[2] - density[0],
        density[3] - density[1],
        density[6] - density[4],
        density[7] - density[5],
        x,
        z,
    );
    let gz = bilerp(
        density[4] - density[0],
        density[5] - density[1],
        density[6] - density[2],
        density[7] - density[3],
        x,
        y,
    );
    normalize_or([gx, gy, gz], [1.0, 0.0, 0.0])
}

fn bilerp(a: f64, b: f64, c: f64, d: f64, u: f64, v: f64) -> f64 {
    let first = a + (b - a) * u;
    let second = c + (d - c) * u;
    first + (second - first) * v
}

fn normalize_or(value: [f64; 3], fallback: [f64; 3]) -> [f64; 3] {
    let length_squared = dot(value, value);
    if length_squared <= 1.0e-24 {
        fallback
    } else {
        let inverse_length = length_squared.sqrt().recip();
        value.map(|coordinate| coordinate * inverse_length)
    }
}

fn count_face_pairing_mismatches() -> u32 {
    let mut mismatches = 0_u32;
    for axis in 0..3 {
        for shared in 0..16_u8 {
            for negative_other in 0..16_u8 {
                let negative_case = face_case(axis, true, shared, negative_other);
                let negative_pairing = face_pairing(negative_case, axis, true);
                for positive_other in 0..16_u8 {
                    let positive_case = face_case(axis, false, shared, positive_other);
                    let positive_pairing = face_pairing(positive_case, axis, false);
                    if negative_pairing != positive_pairing {
                        mismatches += 1;
                    }
                }
            }
        }
    }
    mismatches
}

fn face_case(axis: usize, positive_face: bool, shared: u8, other: u8) -> u8 {
    let mut case_index = 0_u8;
    let mut shared_bit = 0;
    let mut other_bit = 0;
    for corner in 0..8_u8 {
        let coordinate = TRANSVOXEL_REGULAR_CORNERS[usize::from(corner)][axis] != 0;
        let is_shared = coordinate == positive_face;
        let solid = if is_shared {
            let solid = shared & (1 << shared_bit) != 0;
            shared_bit += 1;
            solid
        } else {
            let solid = other & (1 << other_bit) != 0;
            other_bit += 1;
            solid
        };
        if solid {
            case_index |= 1 << fixture_corner_from_regular(corner);
        }
    }
    case_index
}

fn face_pairing(fixture_case: u8, axis: usize, positive_face: bool) -> [[u8; 2]; 2] {
    let topology = regular_case_from_fixture(fixture_case);
    let mut pairs = [[u8::MAX; 2]; 2];
    let mut pair_count = 0;
    for triangle in 0..topology.triangle_count() {
        let triangle = topology.triangle(triangle).unwrap();
        let mut face_edges = [u8::MAX; 3];
        let mut count = 0;
        for vertex in triangle {
            let endpoints = topology.vertex(usize::from(vertex)).unwrap().endpoints();
            if let Some(label) = projected_face_edge(endpoints, axis, positive_face) {
                face_edges[count] = label;
                count += 1;
            }
        }
        if count == 2 {
            let mut pair = [face_edges[0], face_edges[1]];
            pair.sort_unstable();
            if !pairs[..pair_count].contains(&pair) {
                pairs[pair_count] = pair;
                pair_count += 1;
            }
        }
    }
    pairs[..pair_count].sort_unstable();
    pairs
}

fn projected_face_edge(endpoints: [u8; 2], axis: usize, positive_face: bool) -> Option<u8> {
    let first = TRANSVOXEL_REGULAR_CORNERS[usize::from(endpoints[0])];
    let second = TRANSVOXEL_REGULAR_CORNERS[usize::from(endpoints[1])];
    let face_coordinate = u8::from(positive_face);
    if first[axis] != face_coordinate || second[axis] != face_coordinate {
        return None;
    }
    let projected_axes: [usize; 2] = match axis {
        0 => [1, 2],
        1 => [0, 2],
        _ => [0, 1],
    };
    let point = |corner: [u8; 3]| corner[projected_axes[0]] + 2 * corner[projected_axes[1]];
    let mut projected = [point(first), point(second)];
    projected.sort_unstable();
    match projected {
        [0, 1] => Some(0),
        [1, 3] => Some(1),
        [2, 3] => Some(2),
        [0, 2] => Some(3),
        _ => None,
    }
}

fn incident_cells(edge_min: [i32; 3], axis: u8) -> [[i32; 3]; 4] {
    let [x, y, z] = edge_min;
    match axis {
        0 => [[x, y - 1, z - 1], [x, y, z - 1], [x, y, z], [x, y - 1, z]],
        1 => [[x - 1, y, z - 1], [x - 1, y, z], [x, y, z], [x, y, z - 1]],
        2 => [[x - 1, y - 1, z], [x, y - 1, z], [x, y, z], [x - 1, y, z]],
        _ => unreachable!("an owned grid edge has one of three axes"),
    }
}

fn cell_words(samples: &[CellWord], cell_min: [i32; 3]) -> [CellWord; 8] {
    TRANSVOXEL_REGULAR_CORNERS.map(|corner| {
        sample(
            samples,
            [
                cell_min[0] + i32::from(corner[0]),
                cell_min[1] + i32::from(corner[1]),
                cell_min[2] + i32::from(corner[2]),
            ],
        )
    })
}

fn fixture_case(words: [CellWord; 8]) -> u8 {
    let mut case_index = 0_u8;
    for (regular_corner, word) in words.into_iter().enumerate() {
        if word.is_solid() {
            case_index |= 1 << fixture_corner_from_regular(regular_corner as u8);
        }
    }
    case_index
}

const fn fixture_corner_from_regular(regular: u8) -> u8 {
    [0, 1, 3, 2, 4, 5, 7, 6][regular as usize]
}

fn sample(samples: &[CellWord], local: [i32; 3]) -> CellWord {
    let shifted = local.map(|coordinate| coordinate + EXTRACTION_HALO);
    debug_assert!(shifted
        .iter()
        .all(|coordinate| (0..EXTRACTION_SAMPLE_EDGE as i32).contains(coordinate)));
    let [x, y, z] = shifted.map(|coordinate| coordinate as usize);
    samples[x + y * EXTRACTION_SAMPLE_EDGE + z * EXTRACTION_SAMPLE_EDGE * EXTRACTION_SAMPLE_EDGE]
}

fn cell_index(cell_min: [i32; 3]) -> usize {
    let [x, y, z] = cell_min.map(|coordinate| (coordinate - MANIFOLD_DC_CELL_HALO_MIN) as usize);
    x + y * MANIFOLD_DC_CELL_EDGE + z * MANIFOLD_DC_CELL_EDGE * MANIFOLD_DC_CELL_EDGE
}

fn cube_edge_index(mut endpoints: [u8; 2]) -> Option<usize> {
    endpoints.sort_unstable();
    MANIFOLD_DC_CUBE_EDGES
        .iter()
        .position(|candidate| *candidate == endpoints)
}

fn regular_corner_index(position: [i32; 3]) -> Option<u8> {
    TRANSVOXEL_REGULAR_CORNERS
        .iter()
        .position(|corner| corner.map(i32::from) == position)
        .map(|corner| corner as u8)
}

const fn corner_is_solid(fixture_case: u8, regular_corner: u8) -> bool {
    fixture_case & (1 << fixture_corner_from_regular(regular_corner)) != 0
}

fn find(parents: &mut [u8; MANIFOLD_DC_CUBE_EDGE_COUNT], mut vertex: usize) -> usize {
    while usize::from(parents[vertex]) != vertex {
        let parent = usize::from(parents[vertex]);
        parents[vertex] = parents[parent];
        vertex = usize::from(parents[vertex]);
    }
    vertex
}

fn union(parents: &mut [u8; MANIFOLD_DC_CUBE_EDGE_COUNT], a: usize, b: usize) {
    let a = find(parents, a);
    let b = find(parents, b);
    if a != b {
        let (root, child) = if a < b { (a, b) } else { (b, a) };
        parents[child] = root as u8;
    }
}

fn subtract(a: [i32; 3], b: [i32; 3]) -> [i32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn subtract_f64(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn triangle_area_squared(a: [f32; 3], b: [f32; 3], c: [f32; 3]) -> f32 {
    let cross = triangle_cross(a, b, c);
    cross[0] * cross[0] + cross[1] * cross[1] + cross[2] * cross[2]
}

fn triangle_cross(a: [f32; 3], b: [f32; 3], c: [f32; 3]) -> [f32; 3] {
    let ab = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
    let ac = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
    [
        ab[1] * ac[2] - ab[2] * ac[1],
        ab[2] * ac[0] - ab[0] * ac[2],
        ab[0] * ac[1] - ab[1] * ac[0],
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ExtractionFixture, ExtractionFixtureKind};
    use helio_planet_voxel_core::PageKey;
    use std::collections::{BTreeMap, BTreeSet};

    #[test]
    fn regular_table_defines_manifold_components_and_face_pairings() {
        let audit = audit_manifold_dc_table();
        assert_eq!(audit.cases, 256);
        assert_eq!(audit.active_cases, 254);
        assert_eq!(audit.components, 358);
        assert_eq!(
            audit.max_components as usize,
            MANIFOLD_DC_MAX_COMPONENTS_PER_CELL
        );
        assert_eq!(audit.face_pairing_mismatches, 0);
        assert_eq!(MANIFOLD_DC_CELL_COUNT, 35_937);
        assert_eq!(MANIFOLD_DC_MAX_QEF_VERTICES, 143_748);
        assert_eq!(MANIFOLD_DC_MAX_VERTICES, 1_028_484);
        assert_eq!(MANIFOLD_DC_MAX_QUADS, 98_304);
        assert_eq!(MANIFOLD_DC_MAX_INDICES, 2_359_296);
        assert_eq!(MANIFOLD_DC_MAX_GPU_VERTICES, 884_736);
    }

    #[test]
    fn qef_solution_is_bounded_and_preserves_a_sharp_corner() {
        let samples = [
            HermiteSample {
                point: [0.25, 0.0, 0.0],
                normal: [1.0, 0.0, 0.0],
            },
            HermiteSample {
                point: [0.0, 0.5, 0.0],
                normal: [0.0, 1.0, 0.0],
            },
            HermiteSample {
                point: [0.0, 0.0, 0.75],
                normal: [0.0, 0.0, 1.0],
            },
        ];
        let solution = solve_bounded_qef(&samples);
        for axis in 0..3 {
            assert!((solution[axis] - [0.25, 0.5, 0.75][axis]).abs() < 1.0e-10);
        }
        let outside = [
            HermiteSample {
                point: [2.0, 0.0, 0.0],
                normal: [1.0, 0.0, 0.0],
            },
            HermiteSample {
                point: [0.0, -1.0, 0.0],
                normal: [0.0, 1.0, 0.0],
            },
        ];
        let bounded = solve_bounded_qef(&outside);
        assert!(bounded.iter().all(|value| (0.0..=1.0).contains(value)));
        assert_eq!(bounded[0], 1.0);
        assert_eq!(bounded[1], 0.0);
    }

    #[test]
    fn page_extraction_is_deterministic_bounded_and_uses_owned_edges_once() {
        for kind in ExtractionFixtureKind::ALL {
            let page_xyz = match kind {
                ExtractionFixtureKind::Plane
                | ExtractionFixtureKind::ThinSlab
                | ExtractionFixtureKind::MaterialSeam => [0, -1, 0],
                _ => [0, 0, 0],
            };
            let fixture = ExtractionFixture::new(kind, PageKey::new(0, page_xyz)).unwrap();
            let first = extract_manifold_dc_page(fixture.samples()).unwrap();
            let second = extract_manifold_dc_page(fixture.samples()).unwrap();
            assert_eq!(first, second, "{} determinism", kind.name());
            assert!(!first.quads.is_empty(), "{} surface", kind.name());
            assert_eq!(
                first.indices.len(),
                first.quads.len() * MANIFOLD_DC_INDICES_PER_QUAD
            );
            assert!(first.quads.len() <= MANIFOLD_DC_OWNED_EDGE_COUNT);
            let audit = audit_manifold_dc_mesh(&first);
            assert!(audit.is_two_manifold(), "{} {audit:?}", kind.name());
            let gpu = first.gpu_mesh().unwrap();
            assert_eq!(gpu.indices.len(), first.indices.len());
            assert!(gpu.vertices.len() <= MANIFOLD_DC_MAX_GPU_VERTICES);
            assert!(gpu
                .indices
                .iter()
                .all(|index| (*index as usize) < gpu.vertices.len()));
            let mut owned = BTreeSet::new();
            for quad in &first.quads {
                assert!(owned.insert((quad.edge_min, quad.axis)));
                for vertex in quad.vertices {
                    assert!((vertex as usize) < first.vertices.len());
                }
            }
            for vertex in &first.vertices {
                assert!(vertex.gpu.position.iter().all(|value| value.is_finite()));
                assert!(vertex.gpu.normal.iter().all(|value| value.is_finite()));
                assert!(vertex.qef_error.is_finite() && vertex.qef_error >= 0.0);
                if vertex.component == u8::MAX {
                    continue;
                }
                for axis in 0..3 {
                    let local = vertex.gpu.position[axis] - vertex.cell_min[axis] as f32;
                    assert!((-1.0e-5..=1.0 + 1.0e-5).contains(&local));
                }
            }
        }
    }

    #[test]
    fn closed_sphere_is_a_two_manifold_with_disk_vertex_links() {
        let samples = local_sphere_samples([15.5, 15.5, 15.5], 10.0);
        let mesh = extract_manifold_dc_page(&samples).unwrap();
        assert!(!mesh.indices.is_empty());
        let audit = audit_manifold_dc_mesh(&mesh);
        assert!(audit.is_closed_two_manifold(), "{audit:?}");

        let mut duplicate_triangle = mesh.clone();
        duplicate_triangle
            .indices
            .extend_from_slice(&mesh.indices[..3]);
        let duplicate_audit = audit_manifold_dc_mesh(&duplicate_triangle);
        assert!(duplicate_audit.nonmanifold_edges > 0);
        assert!(matches!(
            duplicate_triangle.gpu_mesh(),
            Err(ManifoldDcError::MeshIndexCount { .. })
        ));

        let mut invalid_index = mesh.clone();
        invalid_index.indices[0] = u32::MAX;
        assert!(matches!(
            invalid_index.gpu_mesh(),
            Err(ManifoldDcError::MeshIndex { .. })
        ));
        let mut invalid_index = mesh;
        invalid_index.indices.extend([u32::MAX, 0, 1]);
        assert_eq!(audit_manifold_dc_mesh(&invalid_index).invalid_indices, 1);
    }

    #[test]
    fn dense_ambiguous_field_remains_a_two_manifold() {
        let samples = adversarial_samples(0xd1b5_4a32_d192_ed03);
        let mesh = extract_manifold_dc_page(&samples).unwrap();
        let audit = audit_manifold_dc_mesh(&mesh);
        assert!(audit.is_two_manifold(), "{audit:?}");
        assert!(audit.nonmanifold_edges == 0 && audit.nonmanifold_vertices == 0);
        assert_eq!(
            mesh.indices.len(),
            mesh.quads.len() * MANIFOLD_DC_INDICES_PER_QUAD
        );
    }

    #[test]
    fn planar_winding_materials_and_sharp_qef_feature_are_preserved() {
        let plane =
            ExtractionFixture::new(ExtractionFixtureKind::Plane, PageKey::new(0, [0, -1, 0]))
                .unwrap();
        let plane_mesh = extract_manifold_dc_page(plane.samples()).unwrap();
        for triangle in plane_mesh.indices.chunks_exact(3) {
            let a = plane_mesh.vertices[triangle[0] as usize].gpu.position;
            let b = plane_mesh.vertices[triangle[1] as usize].gpu.position;
            let c = plane_mesh.vertices[triangle[2] as usize].gpu.position;
            assert!(triangle_cross(a, b, c)[1] > 0.0);
        }

        let mut materials = BTreeSet::new();
        for page_x in [-1, 0] {
            let material = ExtractionFixture::new(
                ExtractionFixtureKind::MaterialSeam,
                PageKey::new(0, [page_x, -1, 0]),
            )
            .unwrap();
            let material_mesh = extract_manifold_dc_page(material.samples()).unwrap();
            materials.extend(material_mesh.quads.iter().map(|quad| quad.material));
            let gpu_mesh = material_mesh.gpu_mesh().unwrap();
            assert!(gpu_mesh
                .vertices
                .iter()
                .all(|vertex| matches!(vertex.material, 1 | 2)));
        }
        assert_eq!(materials, BTreeSet::from([1, 2]));

        let corner = ExtractionFixture::new(
            ExtractionFixtureKind::SharpCorner,
            PageKey::new(0, [0, 0, 0]),
        )
        .unwrap();
        let corner_mesh = extract_manifold_dc_page(corner.samples()).unwrap();
        let closest = corner_mesh
            .vertices
            .iter()
            .map(|vertex| {
                vertex
                    .gpu
                    .position
                    .iter()
                    .map(|coordinate| coordinate.abs())
                    .fold(0.0_f32, f32::max)
            })
            .fold(f32::INFINITY, f32::min);
        assert!(
            closest <= (1.0 / MANIFOLD_DC_POSITION_QUANTIZATION_STEPS) as f32,
            "sharp corner error {closest}"
        );
    }

    #[test]
    fn adjacent_pages_reproduce_identical_canonical_halo_vertices() {
        for axis in 0..3 {
            let mut negative_page = [0, 0, 0];
            negative_page[axis] = -1;
            let mut positive_page = negative_page;
            positive_page[axis] += 1;
            let negative_fixture = ExtractionFixture::new(
                ExtractionFixtureKind::Sphere,
                PageKey::new(0, negative_page),
            )
            .unwrap();
            let positive_fixture = ExtractionFixture::new(
                ExtractionFixtureKind::Sphere,
                PageKey::new(0, positive_page),
            )
            .unwrap();
            let negative = extract_manifold_dc_page(negative_fixture.samples()).unwrap();
            let positive = extract_manifold_dc_page(positive_fixture.samples()).unwrap();
            let negative_vertices = canonical_vertices(&negative, negative_fixture.page());
            let positive_vertices = canonical_vertices(&positive, positive_fixture.page());
            let shared_coordinate = negative_fixture.page().lod0_cell_min().unwrap()[axis] + 31;
            let shared: Vec<_> = negative_vertices
                .iter()
                .filter(|((cell, _), _)| cell[axis] == shared_coordinate)
                .collect();
            assert!(!shared.is_empty(), "axis {axis}");
            let mut compared = 0;
            for (key, negative_vertex) in shared {
                let Some(positive_vertex) = positive_vertices.get(key) else {
                    continue;
                };
                compared += 1;
                assert_eq!(
                    negative_vertex.gpu.position, positive_vertex.gpu.position,
                    "axis {axis} {key:?}"
                );
                assert_eq!(
                    negative_vertex.gpu.normal, positive_vertex.gpu.normal,
                    "axis {axis} {key:?}"
                );
                assert_eq!(
                    negative_vertex.qef_error, positive_vertex.qef_error,
                    "axis {axis} {key:?}"
                );
                assert_eq!(
                    negative_vertex.hermite_count, positive_vertex.hermite_count,
                    "axis {axis} {key:?}"
                );
            }
            assert!(compared > 0, "axis {axis} has shared published vertices");
        }
    }

    #[test]
    fn invalid_sample_count_is_rejected() {
        assert!(matches!(
            extract_manifold_dc_page(&[CellWord::AIR; 3]),
            Err(ManifoldDcError::SampleCount { .. })
        ));
    }

    fn local_sphere_samples(center: [f64; 3], radius: f64) -> Vec<CellWord> {
        let mut samples = Vec::with_capacity(EXTRACTION_SAMPLE_COUNT);
        for z in -EXTRACTION_HALO..PAGE_EDGE as i32 + EXTRACTION_HALO {
            for y in -EXTRACTION_HALO..PAGE_EDGE as i32 + EXTRACTION_HALO {
                for x in -EXTRACTION_HALO..PAGE_EDGE as i32 + EXTRACTION_HALO {
                    let delta = [
                        x as f64 - center[0],
                        y as f64 - center[1],
                        z as f64 - center[2],
                    ];
                    let distance = (dot(delta, delta).sqrt() - radius) * 256.0;
                    samples.push(CellWord::new(
                        distance.clamp(f64::from(i16::MIN), f64::from(i16::MAX)) as i16,
                        u8::from(distance <= 0.0),
                        0,
                    ));
                }
            }
        }
        samples
    }

    fn adversarial_samples(seed: u64) -> Vec<CellWord> {
        (0..EXTRACTION_SAMPLE_COUNT)
            .map(|linear| {
                let mut value = seed ^ linear as u64;
                value ^= value >> 30;
                value = value.wrapping_mul(0xbf58_476d_1ce4_e5b9);
                value ^= value >> 27;
                value = value.wrapping_mul(0x94d0_49bb_1331_11eb);
                value ^= value >> 31;
                let density = value as u16 as i16;
                let density = if density == 0 { 1 } else { density };
                let material = if density <= 0 {
                    ((value >> 16) as u8 % 7) + 1
                } else {
                    0
                };
                CellWord::new(density, material, 0)
            })
            .collect()
    }

    fn canonical_vertices(
        mesh: &ManifoldDcMesh,
        page: PageKey,
    ) -> BTreeMap<([i64; 3], u8), ManifoldDcVertex> {
        let page_min = page.lod0_cell_min().unwrap();
        mesh.vertices
            .iter()
            .copied()
            .filter(|vertex| vertex.component != u8::MAX)
            .map(|mut vertex| {
                let canonical_cell = [
                    page_min[0] + i64::from(vertex.cell_min[0]),
                    page_min[1] + i64::from(vertex.cell_min[1]),
                    page_min[2] + i64::from(vertex.cell_min[2]),
                ];
                for (coordinate, origin) in vertex.gpu.position.iter_mut().zip(page_min) {
                    *coordinate += origin as f32;
                }
                ((canonical_cell, vertex.component), vertex)
            })
            .collect()
    }
}
