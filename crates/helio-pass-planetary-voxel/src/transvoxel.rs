//! Audited CPU reference for the official Transvoxel lookup tables.
//!
//! The production extractor will consume the same generated data on the GPU.
//! Keeping a small CPU decoder lets tests exhaustively check every table case,
//! winding flag, edge endpoint, and triangle index before shader integration.

mod generated {
    include!("transvoxel_tables.rs");
}

pub use generated::TRANSVOXEL_UPSTREAM_REVISION;

pub const REGULAR_CASE_COUNT: usize = 256;
pub const TRANSITION_CASE_COUNT: usize = 512;
pub const REGULAR_CORNER_COUNT: u8 = 8;
pub const TRANSITION_CORNER_COUNT: u8 = 13;

/// Corner coordinates used by Lengyel's regular-cell tables. The fixture
/// module intentionally uses the conventional cyclic Marching Cubes order, so
/// callers must use [`fixture_case_to_regular_case`] at that boundary.
pub const TRANSVOXEL_REGULAR_CORNERS: [[u8; 3]; REGULAR_CORNER_COUNT as usize] = [
    [0, 0, 0],
    [1, 0, 0],
    [0, 1, 0],
    [1, 1, 0],
    [0, 0, 1],
    [1, 0, 1],
    [0, 1, 1],
    [1, 1, 1],
];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TransvoxelCellKind {
    Regular,
    Transition,
}

impl TransvoxelCellKind {
    const fn name(self) -> &'static str {
        match self {
            Self::Regular => "regular",
            Self::Transition => "transition",
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct TransvoxelVertexCode(u16);

impl TransvoxelVertexCode {
    pub const fn from_raw(raw: u16) -> Self {
        Self(raw)
    }

    pub const fn raw(self) -> u16 {
        self.0
    }

    /// Endpoint indexes encoded in the low byte, in the same order as the
    /// official table notation.
    pub const fn endpoints(self) -> [u8; 2] {
        let edge = self.0 as u8;
        [edge >> 4, edge & 0x0f]
    }

    pub const fn reuse(self) -> u8 {
        (self.0 >> 8) as u8
    }
}

/// A borrowed, allocation-free decode of one official table case.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TransvoxelCaseTopology {
    kind: TransvoxelCellKind,
    case_index: u16,
    class_index: u8,
    reverse_winding: bool,
    vertex_codes: &'static [u16],
    triangle_indices: &'static [u8],
}

impl TransvoxelCaseTopology {
    pub const fn kind(self) -> TransvoxelCellKind {
        self.kind
    }

    pub const fn case_index(self) -> u16 {
        self.case_index
    }

    pub const fn class_index(self) -> u8 {
        self.class_index
    }

    pub const fn reverse_winding(self) -> bool {
        self.reverse_winding
    }

    pub const fn vertex_count(self) -> usize {
        self.vertex_codes.len()
    }

    pub const fn triangle_count(self) -> usize {
        self.triangle_indices.len() / 3
    }

    pub fn vertex(self, index: usize) -> Option<TransvoxelVertexCode> {
        self.vertex_codes
            .get(index)
            .copied()
            .map(TransvoxelVertexCode::from_raw)
    }

    /// Returns a triangle with the inverse-case winding already applied.
    pub fn triangle(self, index: usize) -> Option<[u8; 3]> {
        let offset = index.checked_mul(3)?;
        let triangle = self.triangle_indices.get(offset..offset + 3)?;
        Some(if self.reverse_winding {
            [triangle[0], triangle[2], triangle[1]]
        } else {
            [triangle[0], triangle[1], triangle[2]]
        })
    }

    pub fn fingerprint(self) -> u64 {
        let mut fingerprint = FNV_OFFSET;
        fingerprint = fnv_byte(fingerprint, self.kind as u8);
        for byte in self.case_index.to_le_bytes() {
            fingerprint = fnv_byte(fingerprint, byte);
        }
        fingerprint = fnv_byte(fingerprint, self.class_index);
        fingerprint = fnv_byte(fingerprint, u8::from(self.reverse_winding));
        fingerprint = fnv_byte(fingerprint, self.vertex_count() as u8);
        fingerprint = fnv_byte(fingerprint, self.triangle_count() as u8);
        for code in self.vertex_codes {
            for byte in code.to_le_bytes() {
                fingerprint = fnv_byte(fingerprint, byte);
            }
        }
        for triangle in 0..self.triangle_count() {
            for index in self
                .triangle(triangle)
                .expect("triangle count is derived from the index slice")
            {
                fingerprint = fnv_byte(fingerprint, index);
            }
        }
        fingerprint
    }
}

/// Converts the fixture module's cyclic cube-corner ordering to the bit order
/// required by the official regular-cell tables.
pub const fn fixture_case_to_regular_case(fixture_case: u8) -> u8 {
    let mut regular_case = 0_u8;
    let fixture_bit_for_regular_corner = [0_u8, 1, 3, 2, 4, 5, 7, 6];
    let mut regular_corner = 0_u8;
    while regular_corner < REGULAR_CORNER_COUNT {
        let fixture_corner = fixture_bit_for_regular_corner[regular_corner as usize];
        if fixture_case & (1 << fixture_corner) != 0 {
            regular_case |= 1 << regular_corner;
        }
        regular_corner += 1;
    }
    regular_case
}

pub fn regular_case(case_index: u8) -> TransvoxelCaseTopology {
    let case = usize::from(case_index);
    let class_index = generated::REGULAR_CELL_CLASS[case];
    let counts = generated::REGULAR_CELL_GEOMETRY_COUNTS[usize::from(class_index)];
    let vertex_count = usize::from(counts >> 4);
    let index_count = usize::from(counts & 0x0f) * 3;
    TransvoxelCaseTopology {
        kind: TransvoxelCellKind::Regular,
        case_index: u16::from(case_index),
        class_index,
        reverse_winding: false,
        vertex_codes: &generated::REGULAR_VERTEX_DATA[case][..vertex_count],
        triangle_indices: &generated::REGULAR_CELL_VERTEX_INDEX[usize::from(class_index)]
            [..index_count],
    }
}

pub fn regular_case_from_fixture(fixture_case: u8) -> TransvoxelCaseTopology {
    regular_case(fixture_case_to_regular_case(fixture_case))
}

pub fn transition_case(case_index: u16) -> Option<TransvoxelCaseTopology> {
    let case = usize::from(case_index);
    let class_code = *generated::TRANSITION_CELL_CLASS.get(case)?;
    let class_index = class_code & 0x7f;
    let counts = generated::TRANSITION_CELL_GEOMETRY_COUNTS[usize::from(class_index)];
    let vertex_count = usize::from(counts >> 4);
    let index_count = usize::from(counts & 0x0f) * 3;
    Some(TransvoxelCaseTopology {
        kind: TransvoxelCellKind::Transition,
        case_index,
        class_index,
        reverse_winding: class_code & 0x80 != 0,
        vertex_codes: &generated::TRANSITION_VERTEX_DATA[case][..vertex_count],
        triangle_indices: &generated::TRANSITION_CELL_VERTEX_INDEX[usize::from(class_index)]
            [..index_count],
    })
}

pub const fn transition_corner_reuse(corner: u8) -> Option<u8> {
    if (corner as usize) < generated::TRANSITION_CORNER_DATA.len() {
        Some(generated::TRANSITION_CORNER_DATA[corner as usize])
    } else {
        None
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct TransvoxelTableAudit {
    pub regular_cases: u16,
    pub transition_cases: u16,
    pub regular_vertices: u32,
    pub regular_triangles: u32,
    pub transition_vertices: u32,
    pub transition_triangles: u32,
    pub max_regular_vertices: u8,
    pub max_regular_triangles: u8,
    pub max_transition_vertices: u8,
    pub max_transition_triangles: u8,
    pub fingerprint: u64,
}

pub fn validate_transvoxel_tables() -> Result<TransvoxelTableAudit, TransvoxelTableError> {
    let mut audit = TransvoxelTableAudit {
        regular_cases: REGULAR_CASE_COUNT as u16,
        transition_cases: TRANSITION_CASE_COUNT as u16,
        fingerprint: FNV_OFFSET,
        ..TransvoxelTableAudit::default()
    };

    for case_index in 0..REGULAR_CASE_COUNT {
        let topology = regular_case(case_index as u8);
        validate_topology(topology)?;
        for vertex in topology.vertex_codes {
            let [first, second] = TransvoxelVertexCode::from_raw(*vertex).endpoints();
            let first_inside = case_index & (1 << first) != 0;
            let second_inside = case_index & (1 << second) != 0;
            if first_inside == second_inside {
                return Err(TransvoxelTableError::new(
                    TransvoxelCellKind::Regular,
                    case_index as u16,
                    "vertex edge does not cross the regular-cell surface",
                ));
            }
        }
        audit.regular_vertices += topology.vertex_count() as u32;
        audit.regular_triangles += topology.triangle_count() as u32;
        audit.max_regular_vertices = audit
            .max_regular_vertices
            .max(topology.vertex_count() as u8);
        audit.max_regular_triangles = audit
            .max_regular_triangles
            .max(topology.triangle_count() as u8);
        audit.fingerprint = fnv_u64(audit.fingerprint, topology.fingerprint());
    }

    for case_index in 0..TRANSITION_CASE_COUNT {
        let topology = transition_case(case_index as u16).ok_or_else(|| {
            TransvoxelTableError::new(
                TransvoxelCellKind::Transition,
                case_index as u16,
                "missing transition case",
            )
        })?;
        validate_topology(topology)?;
        audit.transition_vertices += topology.vertex_count() as u32;
        audit.transition_triangles += topology.triangle_count() as u32;
        audit.max_transition_vertices = audit
            .max_transition_vertices
            .max(topology.vertex_count() as u8);
        audit.max_transition_triangles = audit
            .max_transition_triangles
            .max(topology.triangle_count() as u8);
        audit.fingerprint = fnv_u64(audit.fingerprint, topology.fingerprint());
    }

    Ok(audit)
}

fn validate_topology(topology: TransvoxelCaseTopology) -> Result<(), TransvoxelTableError> {
    let (class_count, max_vertices, max_triangles, corner_count) = match topology.kind {
        TransvoxelCellKind::Regular => (16, 12, 5, REGULAR_CORNER_COUNT),
        TransvoxelCellKind::Transition => (56, 12, 12, TRANSITION_CORNER_COUNT),
    };
    if usize::from(topology.class_index) >= class_count {
        return Err(TransvoxelTableError::new(
            topology.kind,
            topology.case_index,
            "class index is out of range",
        ));
    }
    if topology.vertex_count() > max_vertices || topology.triangle_count() > max_triangles {
        return Err(TransvoxelTableError::new(
            topology.kind,
            topology.case_index,
            "geometry counts exceed table capacity",
        ));
    }
    for code in topology.vertex_codes {
        let [first, second] = TransvoxelVertexCode::from_raw(*code).endpoints();
        if first >= corner_count || second >= corner_count || first == second {
            return Err(TransvoxelTableError::new(
                topology.kind,
                topology.case_index,
                "vertex edge endpoints are invalid",
            ));
        }
    }
    for triangle in 0..topology.triangle_count() {
        let indexes = topology
            .triangle(triangle)
            .expect("triangle count is derived from the index slice");
        if indexes
            .into_iter()
            .any(|index| usize::from(index) >= topology.vertex_count())
        {
            return Err(TransvoxelTableError::new(
                topology.kind,
                topology.case_index,
                "triangle references a missing vertex",
            ));
        }
    }
    Ok(())
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
#[error("invalid {kind} Transvoxel case {case_index}: {detail}")]
pub struct TransvoxelTableError {
    kind: &'static str,
    case_index: u16,
    detail: &'static str,
}

impl TransvoxelTableError {
    const fn new(kind: TransvoxelCellKind, case_index: u16, detail: &'static str) -> Self {
        Self {
            kind: kind.name(),
            case_index,
            detail,
        }
    }
}

const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

const fn fnv_byte(fingerprint: u64, byte: u8) -> u64 {
    (fingerprint ^ byte as u64).wrapping_mul(FNV_PRIME)
}

fn fnv_u64(mut fingerprint: u64, value: u64) -> u64 {
    for byte in value.to_le_bytes() {
        fingerprint = fnv_byte(fingerprint, byte);
    }
    fingerprint
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ExtractionFixture, ExtractionFixtureKind};
    use helio_planet_voxel_core::{PageKey, PAGE_EDGE};

    #[test]
    fn every_official_case_is_index_safe_and_deterministic() {
        let first = validate_transvoxel_tables().unwrap();
        let second = validate_transvoxel_tables().unwrap();
        assert_eq!(first, second);
        assert_eq!(first.regular_cases, 256);
        assert_eq!(first.transition_cases, 512);
        assert_eq!(first.regular_vertices, 1_536);
        assert_eq!(first.regular_triangles, 820);
        assert_eq!(first.transition_vertices, 4_096);
        assert_eq!(first.transition_triangles, 2_640);
        assert_eq!(first.max_regular_vertices, 12);
        assert_eq!(first.max_regular_triangles, 5);
        assert_eq!(first.max_transition_vertices, 12);
        assert_eq!(first.max_transition_triangles, 12);
        assert_eq!(first.fingerprint, 0xf7b4_c5a8_1c51_4273);
    }

    #[test]
    fn transition_inverse_flag_reverses_triangle_winding() {
        let topology = (0..TRANSITION_CASE_COUNT as u16)
            .filter_map(transition_case)
            .find(|case| case.reverse_winding() && case.triangle_count() > 0)
            .unwrap();
        let raw: [u8; 3] = topology.triangle_indices[..3].try_into().unwrap();
        assert_eq!(topology.triangle(0), Some([raw[0], raw[2], raw[1]]));
    }

    #[test]
    fn fixture_corner_adapter_preserves_all_corner_states() {
        for fixture_case in 0..=u8::MAX {
            let regular_case = fixture_case_to_regular_case(fixture_case);
            for (regular_corner, fixture_corner) in
                [0_u8, 1, 3, 2, 4, 5, 7, 6].into_iter().enumerate()
            {
                assert_eq!(
                    regular_case & (1 << regular_corner) != 0,
                    fixture_case & (1 << fixture_corner) != 0
                );
            }
        }
    }

    #[test]
    fn canonical_fixtures_decode_all_active_regular_cells() {
        for kind in ExtractionFixtureKind::ALL {
            let page_xyz = match kind {
                ExtractionFixtureKind::Plane
                | ExtractionFixtureKind::ThinSlab
                | ExtractionFixtureKind::MaterialSeam => [0, -1, 0],
                ExtractionFixtureKind::Sphere
                | ExtractionFixtureKind::Cave
                | ExtractionFixtureKind::SharpCorner => [0, 0, 0],
            };
            let fixture = ExtractionFixture::new(kind, PageKey::new(0, page_xyz)).unwrap();
            let mut decoded = 0_u32;
            for z in 0..PAGE_EDGE as u8 {
                for y in 0..PAGE_EDGE as u8 {
                    for x in 0..PAGE_EDGE as u8 {
                        let fixture_case = fixture.cell_case([x, y, z]).unwrap();
                        if !matches!(fixture_case, 0 | u8::MAX) {
                            let topology = regular_case_from_fixture(fixture_case);
                            assert!(topology.vertex_count() >= 3);
                            assert!(topology.triangle_count() >= 1);
                            decoded += 1;
                        }
                    }
                }
            }
            assert_eq!(decoded, fixture.metrics().active_cells, "{}", kind.name());
        }
    }

    #[test]
    fn upstream_revision_is_pinned() {
        assert_eq!(
            TRANSVOXEL_UPSTREAM_REVISION,
            "51a494f03c5b024cd153b596bcc7152eb3cc93a6"
        );
    }
}
