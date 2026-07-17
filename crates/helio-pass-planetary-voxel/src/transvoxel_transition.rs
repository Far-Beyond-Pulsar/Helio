//! CPU reference contract for Transvoxel transition cells.
//!
//! A transition mesh belongs to the coarser page. One coarse page face has
//! 32x32 transition cells and consumes a 65x65 grid sampled at the adjacent
//! finer LOD. The four half-resolution samples duplicate the four corners of
//! each 3x3 full-resolution patch exactly.

use crate::{
    transition_case, transition_case_from_solidity, ExtractionFixtureKind, GpuTerrainVertex,
    TransvoxelCaseTopology, TRANSVOXEL_TRANSITION_DUPLICATE_CORNERS,
};
use helio_planet_voxel_core::{AddressError, CellWord, PageKey, PAGE_EDGE};

pub use helio_planet_voxel_core::TransitionFace;

pub const TRANSITION_FACE_CELL_EDGE: usize = PAGE_EDGE;
pub const TRANSITION_FACE_SAMPLE_EDGE: usize = PAGE_EDGE * 2 + 1;
pub const TRANSITION_FACE_SAMPLE_COUNT: usize =
    TRANSITION_FACE_SAMPLE_EDGE * TRANSITION_FACE_SAMPLE_EDGE;
pub const TRANSITION_CELL_WIDTH_COARSE_CELLS: f32 = 0.25;

/// Row-major sample positions from Figure 4.16, in half-coarse-cell units.
pub const TRANSITION_FULL_SAMPLE_UV: [[u8; 2]; 9] = [
    [0, 0],
    [1, 0],
    [2, 0],
    [0, 1],
    [1, 1],
    [2, 1],
    [0, 2],
    [1, 2],
    [2, 2],
];

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TransitionFaceBasis {
    pub origin: [f32; 3],
    pub u_axis: [f32; 3],
    pub v_axis: [f32; 3],
    pub outward: [f32; 3],
}

impl TransitionFaceBasis {
    /// Maps face coordinates into page-local coarse-cell coordinates. Depth is
    /// positive toward the interior of the coarse page.
    pub fn map(self, u: f32, v: f32, depth: f32) -> [f32; 3] {
        let mut position = self.origin;
        for (axis, coordinate) in position.iter_mut().enumerate() {
            *coordinate +=
                self.u_axis[axis] * u + self.v_axis[axis] * v - self.outward[axis] * depth;
        }
        position
    }
}

/// Each basis is right-handed (`u x v = outward`). Origins are selected so
/// increasing u/v stays inside the [0, PAGE_EDGE] page cube.
pub const fn transition_face_basis(face: TransitionFace) -> TransitionFaceBasis {
    let edge = PAGE_EDGE as f32;
    match face {
        TransitionFace::NegativeX => TransitionFaceBasis {
            origin: [0.0, 0.0, edge],
            u_axis: [0.0, 1.0, 0.0],
            v_axis: [0.0, 0.0, -1.0],
            outward: [-1.0, 0.0, 0.0],
        },
        TransitionFace::PositiveX => TransitionFaceBasis {
            origin: [edge, 0.0, 0.0],
            u_axis: [0.0, 1.0, 0.0],
            v_axis: [0.0, 0.0, 1.0],
            outward: [1.0, 0.0, 0.0],
        },
        TransitionFace::NegativeY => TransitionFaceBasis {
            origin: [edge, 0.0, 0.0],
            u_axis: [0.0, 0.0, 1.0],
            v_axis: [-1.0, 0.0, 0.0],
            outward: [0.0, -1.0, 0.0],
        },
        TransitionFace::PositiveY => TransitionFaceBasis {
            origin: [0.0, edge, 0.0],
            u_axis: [0.0, 0.0, 1.0],
            v_axis: [1.0, 0.0, 0.0],
            outward: [0.0, 1.0, 0.0],
        },
        TransitionFace::NegativeZ => TransitionFaceBasis {
            origin: [0.0, edge, 0.0],
            u_axis: [1.0, 0.0, 0.0],
            v_axis: [0.0, -1.0, 0.0],
            outward: [0.0, 0.0, -1.0],
        },
        TransitionFace::PositiveZ => TransitionFaceBasis {
            origin: [0.0, 0.0, edge],
            u_axis: [1.0, 0.0, 0.0],
            v_axis: [0.0, 1.0, 0.0],
            outward: [0.0, 0.0, 1.0],
        },
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct TransvoxelTransitionSample {
    pub cell: CellWord,
    pub gradient: [f32; 3],
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TransvoxelTransitionCell {
    samples: [TransvoxelTransitionSample; 13],
}

impl TransvoxelTransitionCell {
    pub fn from_full_resolution(full: [TransvoxelTransitionSample; 9]) -> Self {
        let mut samples = [TransvoxelTransitionSample::default(); 13];
        samples[..9].copy_from_slice(&full);
        for (half, full) in TRANSVOXEL_TRANSITION_DUPLICATE_CORNERS
            .into_iter()
            .enumerate()
        {
            samples[9 + half] = samples[usize::from(full)];
        }
        Self { samples }
    }

    pub fn new(
        full: [TransvoxelTransitionSample; 9],
        half: [TransvoxelTransitionSample; 4],
    ) -> Result<Self, TransvoxelTransitionError> {
        for (half_index, full_index) in TRANSVOXEL_TRANSITION_DUPLICATE_CORNERS
            .into_iter()
            .enumerate()
        {
            if half[half_index].cell != full[usize::from(full_index)].cell {
                return Err(TransvoxelTransitionError::MismatchedDuplicate {
                    full_corner: full_index,
                    half_corner: 9 + half_index as u8,
                });
            }
        }
        Ok(Self::from_full_resolution(full))
    }

    pub const fn samples(&self) -> &[TransvoxelTransitionSample; 13] {
        &self.samples
    }

    pub const fn sample(&self, index: u8) -> Option<TransvoxelTransitionSample> {
        if index < 13 {
            Some(self.samples[index as usize])
        } else {
            None
        }
    }

    pub fn case_index(&self) -> u16 {
        let mut solid = [false; 9];
        for (index, sample) in self.samples[..9].iter().enumerate() {
            solid[index] = sample.cell.is_solid();
        }
        transition_case_from_solidity(solid)
    }

    pub fn topology(&self) -> TransvoxelCaseTopology {
        transition_case(self.case_index()).expect("a 9-bit case is always in the official table")
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct TransvoxelTransitionVertex {
    pub vertex: GpuTerrainVertex,
    /// Primary position on the unshifted face in page-local u/v coordinates.
    pub face_uv: [f32; 2],
    /// Zero on the fine face, one on the coarse face, and interpolated for a
    /// vertex lying on an edge between them.
    pub depth_fraction: f32,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct TransvoxelTransitionMesh {
    pub case_index: u16,
    pub vertices: Vec<TransvoxelTransitionVertex>,
    pub indices: Vec<u32>,
}

pub fn extract_transvoxel_transition_cell(
    face: TransitionFace,
    cell_uv: [u8; 2],
    cell: &TransvoxelTransitionCell,
) -> TransvoxelTransitionMesh {
    let topology = cell.topology();
    let basis = transition_face_basis(face);
    let mut mesh = TransvoxelTransitionMesh {
        case_index: cell.case_index(),
        vertices: Vec::with_capacity(topology.vertex_count()),
        indices: Vec::with_capacity(topology.triangle_count() * 3),
    };

    for vertex_index in 0..topology.vertex_count() {
        let [first_corner, second_corner] = topology
            .vertex(vertex_index)
            .expect("vertex count came from the same topology")
            .endpoints();
        let first = cell
            .sample(first_corner)
            .expect("official table endpoints are audited");
        let second = cell
            .sample(second_corner)
            .expect("official table endpoints are audited");
        let first_density = f32::from(first.cell.density());
        let second_density = f32::from(second.cell.density());
        let denominator = first_density - second_density;
        let interpolation = if denominator.abs() > 1.0e-12 {
            (first_density / denominator).clamp(0.0, 1.0)
        } else {
            0.5
        };
        let first_uv = transition_corner_uv(first_corner);
        let second_uv = transition_corner_uv(second_corner);
        let face_uv = [
            f32::from(cell_uv[0]) + mix(first_uv[0], second_uv[0], interpolation),
            f32::from(cell_uv[1]) + mix(first_uv[1], second_uv[1], interpolation),
        ];
        let depth_fraction = mix(
            transition_corner_depth(first_corner),
            transition_corner_depth(second_corner),
            interpolation,
        );
        let normal = normalize_or_outward(
            mix3(first.gradient, second.gradient, interpolation),
            basis.outward,
        );
        let primary = basis.map(face_uv[0], face_uv[1], 0.0);
        let inward_offset = basis
            .outward
            .map(|axis| -axis * TRANSITION_CELL_WIDTH_COARSE_CELLS * depth_fraction);
        let projected_offset = project_onto_tangent(inward_offset, normal);
        let position = add3(primary, projected_offset);
        let material = if first_density <= 0.0 {
            first.cell.material()
        } else {
            second.cell.material()
        };
        mesh.vertices.push(TransvoxelTransitionVertex {
            vertex: GpuTerrainVertex {
                position,
                material: u32::from(material),
                normal,
                flags: 0,
            },
            face_uv,
            depth_fraction,
        });
    }
    for triangle in 0..topology.triangle_count() {
        let [a, b, c] = topology
            .triangle(triangle)
            .expect("triangle count came from the same topology");
        // The official transition table is wound toward the full-resolution
        // face. Our right-handed face bases point outward from the coarse page,
        // so one global flip matches the regular extractor's outward-gradient
        // convention. The per-case inverse flag was already applied above.
        mesh.indices
            .extend([u32::from(a), u32::from(c), u32::from(b)]);
    }
    mesh
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TransvoxelTransitionFaceFixture {
    kind: ExtractionFixtureKind,
    page: PageKey,
    face: TransitionFace,
    page_min: [i64; 3],
    coarse_scale: i64,
    fine_scale: i64,
}

impl TransvoxelTransitionFaceFixture {
    pub fn new(
        kind: ExtractionFixtureKind,
        page: PageKey,
        face: TransitionFace,
    ) -> Result<Self, TransvoxelTransitionError> {
        page.validate()?;
        if page.lod == 0 {
            return Err(TransvoxelTransitionError::FinestLodHasNoFinerNeighbor);
        }
        let coarse_scale = 1_i64
            .checked_shl(u32::from(page.lod))
            .ok_or(AddressError::CoordinateOverflow)?;
        Ok(Self {
            kind,
            page,
            face,
            page_min: page.lod0_cell_min()?,
            coarse_scale,
            fine_scale: coarse_scale / 2,
        })
    }

    pub const fn page(&self) -> PageKey {
        self.page
    }

    pub const fn face(&self) -> TransitionFace {
        self.face
    }

    pub fn full_resolution_sample(
        &self,
        face_sample: [u8; 2],
    ) -> Option<TransvoxelTransitionSample> {
        if face_sample
            .iter()
            .any(|coordinate| usize::from(*coordinate) >= TRANSITION_FACE_SAMPLE_EDGE)
        {
            return None;
        }
        let position = self.canonical_position(face_sample);
        Some(TransvoxelTransitionSample {
            cell: self.kind.sample_canonical(position),
            gradient: self.gradient(position),
        })
    }

    pub fn cell(&self, cell_uv: [u8; 2]) -> Option<TransvoxelTransitionCell> {
        if cell_uv
            .iter()
            .any(|coordinate| usize::from(*coordinate) >= TRANSITION_FACE_CELL_EDGE)
        {
            return None;
        }
        let mut full = [TransvoxelTransitionSample::default(); 9];
        for (sample, offset) in TRANSITION_FULL_SAMPLE_UV.into_iter().enumerate() {
            full[sample] = self
                .full_resolution_sample([cell_uv[0] * 2 + offset[0], cell_uv[1] * 2 + offset[1]])?;
        }
        Some(TransvoxelTransitionCell::from_full_resolution(full))
    }

    pub fn extract(&self) -> TransvoxelTransitionFaceMesh {
        let mut mesh = TransvoxelTransitionFaceMesh {
            face: self.face,
            cell_ranges: vec![None; TRANSITION_FACE_CELL_EDGE * TRANSITION_FACE_CELL_EDGE],
            ..TransvoxelTransitionFaceMesh::default()
        };
        for v in 0..TRANSITION_FACE_CELL_EDGE as u8 {
            for u in 0..TRANSITION_FACE_CELL_EDGE as u8 {
                let cell = self
                    .cell([u, v])
                    .expect("loop coordinates are inside the transition face");
                let cell_mesh = extract_transvoxel_transition_cell(self.face, [u, v], &cell);
                let first_vertex = mesh.vertices.len() as u32;
                let first_index = mesh.indices.len() as u32;
                let linear = usize::from(u) + usize::from(v) * TRANSITION_FACE_CELL_EDGE;
                mesh.cell_ranges[linear] = Some(TransitionCellRange {
                    first_vertex,
                    vertex_count: cell_mesh.vertices.len() as u32,
                    first_index,
                    index_count: cell_mesh.indices.len() as u32,
                });
                mesh.vertices.extend(cell_mesh.vertices);
                mesh.indices.extend(
                    cell_mesh
                        .indices
                        .into_iter()
                        .map(|index| first_vertex + index),
                );
            }
        }
        mesh
    }

    fn canonical_position(&self, face_sample: [u8; 2]) -> [i64; 3] {
        let basis = transition_face_integer_basis(self.face);
        let page_span = self.coarse_scale * PAGE_EDGE as i64;
        let mut position = self.page_min;
        for (axis, coordinate) in position.iter_mut().enumerate() {
            *coordinate += i64::from(basis.origin[axis]) * page_span
                + i64::from(basis.u_axis[axis]) * i64::from(face_sample[0]) * self.fine_scale
                + i64::from(basis.v_axis[axis]) * i64::from(face_sample[1]) * self.fine_scale;
        }
        position
    }

    fn gradient(&self, position: [i64; 3]) -> [f32; 3] {
        let mut gradient = [0.0; 3];
        for axis in 0..3 {
            let mut lower = position;
            let mut upper = position;
            lower[axis] -= self.fine_scale;
            upper[axis] += self.fine_scale;
            gradient[axis] = (f32::from(self.kind.sample_canonical(upper).density())
                - f32::from(self.kind.sample_canonical(lower).density()))
                * 0.5;
        }
        gradient
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct TransitionCellRange {
    pub first_vertex: u32,
    pub vertex_count: u32,
    pub first_index: u32,
    pub index_count: u32,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct TransvoxelTransitionFaceMesh {
    pub face: TransitionFace,
    pub vertices: Vec<TransvoxelTransitionVertex>,
    pub indices: Vec<u32>,
    pub cell_ranges: Vec<Option<TransitionCellRange>>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
pub enum TransvoxelTransitionError {
    #[error(transparent)]
    Address(#[from] AddressError),
    #[error("LOD0 has no finer neighbor and cannot own a transition mesh")]
    FinestLodHasNoFinerNeighbor,
    #[error(
        "transition half-resolution corner {half_corner:X} does not equal full-resolution corner {full_corner:X}"
    )]
    MismatchedDuplicate { full_corner: u8, half_corner: u8 },
}

#[derive(Clone, Copy)]
struct IntegerFaceBasis {
    origin: [i8; 3],
    u_axis: [i8; 3],
    v_axis: [i8; 3],
}

const fn transition_face_integer_basis(face: TransitionFace) -> IntegerFaceBasis {
    match face {
        TransitionFace::NegativeX => IntegerFaceBasis {
            origin: [0, 0, 1],
            u_axis: [0, 1, 0],
            v_axis: [0, 0, -1],
        },
        TransitionFace::PositiveX => IntegerFaceBasis {
            origin: [1, 0, 0],
            u_axis: [0, 1, 0],
            v_axis: [0, 0, 1],
        },
        TransitionFace::NegativeY => IntegerFaceBasis {
            origin: [1, 0, 0],
            u_axis: [0, 0, 1],
            v_axis: [-1, 0, 0],
        },
        TransitionFace::PositiveY => IntegerFaceBasis {
            origin: [0, 1, 0],
            u_axis: [0, 0, 1],
            v_axis: [1, 0, 0],
        },
        TransitionFace::NegativeZ => IntegerFaceBasis {
            origin: [0, 1, 0],
            u_axis: [1, 0, 0],
            v_axis: [0, -1, 0],
        },
        TransitionFace::PositiveZ => IntegerFaceBasis {
            origin: [0, 0, 1],
            u_axis: [1, 0, 0],
            v_axis: [0, 1, 0],
        },
    }
}

fn transition_corner_uv(corner: u8) -> [f32; 2] {
    let full_corner = if corner < 9 {
        corner
    } else {
        TRANSVOXEL_TRANSITION_DUPLICATE_CORNERS[usize::from(corner - 9)]
    };
    let uv = TRANSITION_FULL_SAMPLE_UV[usize::from(full_corner)];
    [f32::from(uv[0]) * 0.5, f32::from(uv[1]) * 0.5]
}

const fn transition_corner_depth(corner: u8) -> f32 {
    if corner < 9 {
        0.0
    } else {
        1.0
    }
}

fn project_onto_tangent(offset: [f32; 3], normal: [f32; 3]) -> [f32; 3] {
    let normal_component = dot(offset, normal);
    [
        offset[0] - normal[0] * normal_component,
        offset[1] - normal[1] * normal_component,
        offset[2] - normal[2] * normal_component,
    ]
}

fn normalize_or_outward(value: [f32; 3], outward: [f32; 3]) -> [f32; 3] {
    let squared = dot(value, value);
    if squared > 1.0e-12 {
        let inverse = squared.sqrt().recip();
        value.map(|axis| axis * inverse)
    } else {
        outward
    }
}

fn dot(left: [f32; 3], right: [f32; 3]) -> f32 {
    left.into_iter().zip(right).map(|(a, b)| a * b).sum()
}

fn add3(left: [f32; 3], right: [f32; 3]) -> [f32; 3] {
    [left[0] + right[0], left[1] + right[1], left[2] + right[2]]
}

fn mix(first: f32, second: f32, amount: f32) -> f32 {
    first + (second - first) * amount
}

fn mix3(first: [f32; 3], second: [f32; 3], amount: f32) -> [f32; 3] {
    [
        mix(first[0], second[0], amount),
        mix(first[1], second[1], amount),
        mix(first[2], second[2], amount),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_face_bases_are_right_handed_and_stay_inside_the_page() {
        for face in TransitionFace::ALL {
            let basis = transition_face_basis(face);
            assert_eq!(cross(basis.u_axis, basis.v_axis), basis.outward, "{face:?}");
            for [u, v] in [[0.0, 0.0], [32.0, 0.0], [0.0, 32.0], [32.0, 32.0]] {
                let full = basis.map(u, v, 0.0);
                let half = basis.map(u, v, TRANSITION_CELL_WIDTH_COARSE_CELLS);
                for axis in 0..3 {
                    assert!((0.0..=32.0).contains(&full[axis]), "{face:?} {full:?}");
                    assert!((0.0..=32.0).contains(&half[axis]), "{face:?} {half:?}");
                }
                let axis = face.axis();
                assert_eq!(full[axis], if face.is_positive() { 32.0 } else { 0.0 });
                assert_eq!(half[axis], if face.is_positive() { 31.75 } else { 0.25 });
            }
        }
    }

    #[test]
    fn transition_cell_rejects_nonidentical_duplicate_corners() {
        let full = [sample(-1, [1.0, 0.0, 0.0]); 9];
        let mut half = [full[0], full[2], full[6], full[8]];
        assert!(TransvoxelTransitionCell::new(full, half).is_ok());
        half[2].cell = CellWord::AIR;
        assert_eq!(
            TransvoxelTransitionCell::new(full, half),
            Err(TransvoxelTransitionError::MismatchedDuplicate {
                full_corner: 6,
                half_corner: 0xB,
            })
        );
    }

    #[test]
    fn every_case_emits_finite_index_safe_crossing_geometry() {
        for case_index in 0..512_u16 {
            let mut full = [sample(1, [1.0, 0.0, 0.0]); 9];
            for (sample_index, sample) in full.iter_mut().enumerate() {
                if case_index & crate::TRANSVOXEL_TRANSITION_CASE_WEIGHTS[sample_index] != 0 {
                    sample.cell = CellWord::new(-1, 7, 0);
                }
            }
            let cell = TransvoxelTransitionCell::from_full_resolution(full);
            assert_eq!(cell.case_index(), case_index);
            let mesh = extract_transvoxel_transition_cell(TransitionFace::PositiveZ, [0, 0], &cell);
            assert_eq!(mesh.vertices.len(), cell.topology().vertex_count());
            assert_eq!(mesh.indices.len(), cell.topology().triangle_count() * 3);
            assert!(mesh
                .indices
                .iter()
                .all(|index| *index < mesh.vertices.len() as u32));
            for vertex in &mesh.vertices {
                assert!(vertex.vertex.position.into_iter().all(f32::is_finite));
                assert!(vertex.vertex.normal.into_iter().all(f32::is_finite));
                assert!((0.0..=1.0).contains(&vertex.depth_fraction));
            }
        }
    }

    #[test]
    fn fixture_uses_a_65_square_fine_grid_and_matches_neighboring_cells() {
        let fixture = TransvoxelTransitionFaceFixture::new(
            ExtractionFixtureKind::SharpCorner,
            PageKey::new(1, [-1, -1, -1]),
            TransitionFace::PositiveZ,
        )
        .unwrap();
        assert!(fixture.full_resolution_sample([64, 64]).is_some());
        assert!(fixture.full_resolution_sample([65, 0]).is_none());
        assert!(fixture.cell([31, 31]).is_some());
        assert!(fixture.cell([32, 0]).is_none());

        for v in 0..32_u8 {
            for u in 0..31_u8 {
                let left = fixture.cell([u, v]).unwrap();
                let right = fixture.cell([u + 1, v]).unwrap();
                for row in 0..3 {
                    assert_eq!(left.samples()[row * 3 + 2], right.samples()[row * 3]);
                }
            }
        }
        for v in 0..31_u8 {
            for u in 0..32_u8 {
                let bottom = fixture.cell([u, v]).unwrap();
                let top = fixture.cell([u, v + 1]).unwrap();
                for column in 0..3 {
                    assert_eq!(bottom.samples()[6 + column], top.samples()[column]);
                }
            }
        }
    }

    #[test]
    fn neighboring_transition_cells_publish_identical_lateral_seam_vertices() {
        let fixture = TransvoxelTransitionFaceFixture::new(
            ExtractionFixtureKind::Plane,
            PageKey::new(1, [0, -1, 0]),
            TransitionFace::PositiveZ,
        )
        .unwrap();
        let mut nonempty_seams = 0;
        for u in 0..31_u8 {
            let left = extract_transvoxel_transition_cell(
                fixture.face(),
                [u, 31],
                &fixture.cell([u, 31]).unwrap(),
            );
            let right = extract_transvoxel_transition_cell(
                fixture.face(),
                [u + 1, 31],
                &fixture.cell([u + 1, 31]).unwrap(),
            );
            let boundary = f32::from(u + 1);
            let left_keys = seam_vertex_keys(&left, 0, boundary);
            let right_keys = seam_vertex_keys(&right, 0, boundary);
            assert_eq!(left_keys, right_keys, "u seam {boundary}");
            nonempty_seams += usize::from(!left_keys.is_empty());
        }
        assert_eq!(nonempty_seams, 31);
    }

    #[test]
    fn secondary_offset_is_tangent_projected_and_never_moves_the_fine_face() {
        let basis = transition_face_basis(TransitionFace::PositiveZ);
        let tangent = project_onto_tangent([0.0, 0.0, -0.25], [1.0, 0.0, 0.0]);
        assert_eq!(tangent, [0.0, 0.0, -0.25]);
        let normal = project_onto_tangent([0.0, 0.0, -0.25], basis.outward);
        assert_eq!(normal, [0.0; 3]);

        let mut full = [sample(1, basis.u_axis); 9];
        for (index, uv) in TRANSITION_FULL_SAMPLE_UV.into_iter().enumerate() {
            full[index].cell = CellWord::new(if uv[0] == 0 { -1 } else { 1 }, 1, 0);
        }
        let mesh = extract_transvoxel_transition_cell(
            TransitionFace::PositiveZ,
            [0, 0],
            &TransvoxelTransitionCell::from_full_resolution(full),
        );
        let mut saw_half_side = false;
        for vertex in mesh.vertices {
            let primary = basis.map(vertex.face_uv[0], vertex.face_uv[1], 0.0);
            let displacement = sub3(vertex.vertex.position, primary);
            assert!((dot(displacement, vertex.vertex.normal)).abs() <= 1.0e-6);
            if vertex.depth_fraction == 0.0 {
                assert_eq!(displacement, [0.0; 3]);
            } else {
                saw_half_side = true;
                assert!(length(displacement) <= TRANSITION_CELL_WIDTH_COARSE_CELLS + 1.0e-6);
            }
        }
        assert!(saw_half_side);
    }

    #[test]
    fn planar_transition_winding_matches_density_gradient_on_every_face() {
        for face in TransitionFace::ALL {
            let basis = transition_face_basis(face);
            let mut full = [sample(1, basis.u_axis); 9];
            for (index, uv) in TRANSITION_FULL_SAMPLE_UV.into_iter().enumerate() {
                full[index].cell = CellWord::new(if uv[0] == 0 { -1 } else { 1 }, 1, 0);
            }
            let mesh = extract_transvoxel_transition_cell(
                face,
                [7, 11],
                &TransvoxelTransitionCell::from_full_resolution(full),
            );
            assert!(!mesh.indices.is_empty(), "{face:?}");
            for triangle in mesh.indices.chunks_exact(3) {
                let a = mesh.vertices[triangle[0] as usize].vertex.position;
                let b = mesh.vertices[triangle[1] as usize].vertex.position;
                let c = mesh.vertices[triangle[2] as usize].vertex.position;
                let geometric = cross(sub3(b, a), sub3(c, a));
                assert!(dot(geometric, basis.u_axis) > 0.0, "{face:?} {triangle:?}");
            }
        }
    }

    #[test]
    fn full_face_extraction_is_deterministic_and_bounded() {
        for face in TransitionFace::ALL {
            let fixture = TransvoxelTransitionFaceFixture::new(
                ExtractionFixtureKind::Plane,
                PageKey::new(1, [0, -1, 0]),
                face,
            )
            .unwrap();
            let first = fixture.extract();
            let second = fixture.extract();
            assert_eq!(first, second, "{face:?}");
            assert_eq!(first.cell_ranges.len(), 32 * 32);
            assert!(first.vertices.len() <= 32 * 32 * 12);
            assert!(first.indices.len() <= 32 * 32 * 12 * 3);
            assert!(first
                .indices
                .iter()
                .all(|index| *index < first.vertices.len() as u32));
        }
    }

    fn sample(density: i16, gradient: [f32; 3]) -> TransvoxelTransitionSample {
        TransvoxelTransitionSample {
            cell: CellWord::new(density, u8::from(density <= 0), 0),
            gradient,
        }
    }

    fn cross(left: [f32; 3], right: [f32; 3]) -> [f32; 3] {
        [
            left[1] * right[2] - left[2] * right[1],
            left[2] * right[0] - left[0] * right[2],
            left[0] * right[1] - left[1] * right[0],
        ]
    }

    fn sub3(left: [f32; 3], right: [f32; 3]) -> [f32; 3] {
        [left[0] - right[0], left[1] - right[1], left[2] - right[2]]
    }

    fn length(value: [f32; 3]) -> f32 {
        dot(value, value).sqrt()
    }

    fn seam_vertex_keys(
        mesh: &TransvoxelTransitionMesh,
        face_axis: usize,
        coordinate: f32,
    ) -> std::collections::BTreeSet<([i32; 3], [i32; 3], u32, i32)> {
        mesh.vertices
            .iter()
            .filter(|vertex| (vertex.face_uv[face_axis] - coordinate).abs() <= 1.0e-6)
            .map(|vertex| {
                (
                    vertex.vertex.position.map(quantize),
                    vertex.vertex.normal.map(quantize),
                    vertex.vertex.material,
                    quantize(vertex.depth_fraction),
                )
            })
            .collect()
    }

    fn quantize(value: f32) -> i32 {
        (value * 100_000.0).round() as i32
    }
}
