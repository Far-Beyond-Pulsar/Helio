use helio_planet_voxel_core::{AddressError, CellWord, PageKey, PAGE_EDGE};

pub const EXTRACTION_HALO: i32 = 1;
pub const EXTRACTION_SAMPLE_EDGE: usize = PAGE_EDGE + 2 * EXTRACTION_HALO as usize;
pub const EXTRACTION_SAMPLE_COUNT: usize =
    EXTRACTION_SAMPLE_EDGE * EXTRACTION_SAMPLE_EDGE * EXTRACTION_SAMPLE_EDGE;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExtractionFixtureKind {
    Plane,
    Sphere,
    Cave,
    SharpCorner,
    ThinSlab,
    MaterialSeam,
}

impl ExtractionFixtureKind {
    pub const ALL: [Self; 6] = [
        Self::Plane,
        Self::Sphere,
        Self::Cave,
        Self::SharpCorner,
        Self::ThinSlab,
        Self::MaterialSeam,
    ];

    pub const fn name(self) -> &'static str {
        match self {
            Self::Plane => "plane",
            Self::Sphere => "sphere",
            Self::Cave => "cave",
            Self::SharpCorner => "sharp_corner",
            Self::ThinSlab => "thin_slab",
            Self::MaterialSeam => "material_seam",
        }
    }

    pub(crate) fn sample_canonical(self, position: [i64; 3]) -> CellWord {
        let [x, y, z] = position;
        let density = match self {
            Self::Plane | Self::MaterialSeam => y.saturating_add(1),
            Self::Sphere => x
                .saturating_mul(x)
                .saturating_add(y.saturating_mul(y))
                .saturating_add(z.saturating_mul(z))
                .saturating_sub(12 * 12),
            Self::Cave => (12 * 12_i64).saturating_sub(
                x.saturating_mul(x)
                    .saturating_add(y.saturating_mul(y))
                    .saturating_add(z.saturating_mul(z)),
            ),
            Self::SharpCorner => x.max(y).max(z),
            Self::ThinSlab => y.saturating_abs().saturating_sub(1),
        };
        let density = density.clamp(i64::from(i16::MIN), i64::from(i16::MAX)) as i16;
        let material = if density <= 0 {
            if self == Self::MaterialSeam && x >= 0 {
                2
            } else {
                1
            }
        } else {
            0
        };
        CellWord::new(density, material, 0)
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ExtractionFixtureMetrics {
    pub solid_samples: u32,
    pub air_samples: u32,
    pub active_cells: u32,
    pub active_microbrick_mask: u64,
    pub fingerprint: u64,
}

/// A deterministic page plus one sample of halo on every face. The scalar
/// field is evaluated in canonical LOD0 coordinates, so adjacent pages and
/// different LODs see the same source field.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExtractionFixture {
    kind: ExtractionFixtureKind,
    page: PageKey,
    samples: Box<[CellWord]>,
    metrics: ExtractionFixtureMetrics,
}

impl ExtractionFixture {
    pub fn new(kind: ExtractionFixtureKind, page: PageKey) -> Result<Self, FixtureError> {
        page.validate()?;
        let page_min = page.lod0_cell_min()?;
        let scale = 1_i64
            .checked_shl(u32::from(page.lod))
            .ok_or(FixtureError::Address(AddressError::CoordinateOverflow))?;
        let mut samples = Vec::with_capacity(EXTRACTION_SAMPLE_COUNT);
        for z in -EXTRACTION_HALO..PAGE_EDGE as i32 + EXTRACTION_HALO {
            for y in -EXTRACTION_HALO..PAGE_EDGE as i32 + EXTRACTION_HALO {
                for x in -EXTRACTION_HALO..PAGE_EDGE as i32 + EXTRACTION_HALO {
                    let local = [x, y, z];
                    let mut position = [0_i64; 3];
                    for axis in 0..3 {
                        position[axis] = page_min[axis]
                            .checked_add(i64::from(local[axis]).saturating_mul(scale))
                            .ok_or(FixtureError::Address(AddressError::CoordinateOverflow))?;
                    }
                    samples.push(kind.sample_canonical(position));
                }
            }
        }
        let mut fixture = Self {
            kind,
            page,
            samples: samples.into_boxed_slice(),
            metrics: ExtractionFixtureMetrics::default(),
        };
        fixture.metrics = fixture.measure();
        Ok(fixture)
    }

    pub const fn kind(&self) -> ExtractionFixtureKind {
        self.kind
    }

    pub const fn page(&self) -> PageKey {
        self.page
    }

    pub fn samples(&self) -> &[CellWord] {
        &self.samples
    }

    pub const fn metrics(&self) -> ExtractionFixtureMetrics {
        self.metrics
    }

    pub fn sample(&self, local: [i32; 3]) -> Option<CellWord> {
        fixture_index(local).map(|index| self.samples[index])
    }

    pub fn cell_case(&self, cell: [u8; 3]) -> Option<u8> {
        if cell.iter().any(|axis| usize::from(*axis) >= PAGE_EDGE) {
            return None;
        }
        let mut case_index = 0_u8;
        for (corner, offset) in CUBE_CORNERS.into_iter().enumerate() {
            let local = [
                i32::from(cell[0]) + offset[0],
                i32::from(cell[1]) + offset[1],
                i32::from(cell[2]) + offset[2],
            ];
            if self.sample(local)?.is_solid() {
                case_index |= 1 << corner;
            }
        }
        Some(case_index)
    }

    pub fn is_active_cell(&self, cell: [u8; 3]) -> bool {
        self.cell_case(cell)
            .is_some_and(|case_index| !matches!(case_index, 0 | u8::MAX))
    }

    fn measure(&self) -> ExtractionFixtureMetrics {
        let mut metrics = ExtractionFixtureMetrics::default();
        let mut fingerprint = 0xcbf2_9ce4_8422_2325_u64;
        for sample in &self.samples {
            if sample.is_solid() {
                metrics.solid_samples += 1;
            } else {
                metrics.air_samples += 1;
            }
            for byte in sample.0.to_le_bytes() {
                fingerprint ^= u64::from(byte);
                fingerprint = fingerprint.wrapping_mul(0x0000_0100_0000_01b3);
            }
        }
        for z in 0..PAGE_EDGE as u8 {
            for y in 0..PAGE_EDGE as u8 {
                for x in 0..PAGE_EDGE as u8 {
                    if self.is_active_cell([x, y, z]) {
                        metrics.active_cells += 1;
                        let microbrick =
                            usize::from(x / 8) + usize::from(y / 8) * 4 + usize::from(z / 8) * 16;
                        metrics.active_microbrick_mask |= 1_u64 << microbrick;
                    }
                }
            }
        }
        metrics.fingerprint = fingerprint;
        metrics
    }
}

const CUBE_CORNERS: [[i32; 3]; 8] = [
    [0, 0, 0],
    [1, 0, 0],
    [1, 1, 0],
    [0, 1, 0],
    [0, 0, 1],
    [1, 0, 1],
    [1, 1, 1],
    [0, 1, 1],
];

fn fixture_index(local: [i32; 3]) -> Option<usize> {
    let shifted = local.map(|axis| axis.checked_add(EXTRACTION_HALO));
    if shifted.iter().any(
        |axis| !matches!(axis, Some(value) if (0..EXTRACTION_SAMPLE_EDGE as i32).contains(value)),
    ) {
        return None;
    }
    let [x, y, z] = shifted.map(|axis| axis.expect("sample bounds were checked") as usize);
    Some(x + y * EXTRACTION_SAMPLE_EDGE + z * EXTRACTION_SAMPLE_EDGE * EXTRACTION_SAMPLE_EDGE)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
pub enum FixtureError {
    #[error(transparent)]
    Address(#[from] AddressError),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_fixtures_are_deterministic_and_have_surface_cells() {
        for kind in ExtractionFixtureKind::ALL {
            let page_xyz = match kind {
                ExtractionFixtureKind::Plane
                | ExtractionFixtureKind::ThinSlab
                | ExtractionFixtureKind::MaterialSeam => [0, -1, 0],
                ExtractionFixtureKind::Sphere
                | ExtractionFixtureKind::Cave
                | ExtractionFixtureKind::SharpCorner => [0, 0, 0],
            };
            let first = ExtractionFixture::new(kind, PageKey::new(0, page_xyz)).unwrap();
            let second = ExtractionFixture::new(kind, PageKey::new(0, page_xyz)).unwrap();
            assert_eq!(first.metrics(), second.metrics(), "{}", kind.name());
            assert!(first.metrics().active_cells > 0, "{}", kind.name());
            assert_ne!(first.metrics().active_microbrick_mask, 0, "{}", kind.name());
            assert_eq!(
                first.metrics().solid_samples + first.metrics().air_samples,
                EXTRACTION_SAMPLE_COUNT as u32
            );
        }
    }

    #[test]
    fn adjacent_page_halos_sample_identical_canonical_cells() {
        for kind in ExtractionFixtureKind::ALL {
            let left = ExtractionFixture::new(kind, PageKey::new(0, [-1, -1, -1])).unwrap();
            let right = ExtractionFixture::new(kind, PageKey::new(0, [0, -1, -1])).unwrap();
            for z in -1..=32 {
                for y in -1..=32 {
                    assert_eq!(
                        left.sample([32, y, z]),
                        right.sample([0, y, z]),
                        "{} at y={y}, z={z}",
                        kind.name()
                    );
                    assert_eq!(
                        left.sample([31, y, z]),
                        right.sample([-1, y, z]),
                        "{} halo at y={y}, z={z}",
                        kind.name()
                    );
                }
            }
        }
    }

    #[test]
    fn material_seam_changes_material_without_changing_density() {
        let material = ExtractionFixture::new(
            ExtractionFixtureKind::MaterialSeam,
            PageKey::new(0, [-1, -1, -1]),
        )
        .unwrap();
        let plane =
            ExtractionFixture::new(ExtractionFixtureKind::Plane, PageKey::new(0, [-1, -1, -1]))
                .unwrap();
        for z in -1..=32 {
            for y in -1..=32 {
                for x in -1..=32 {
                    assert_eq!(
                        material.sample([x, y, z]).unwrap().density(),
                        plane.sample([x, y, z]).unwrap().density()
                    );
                }
            }
        }
        assert_eq!(material.sample([31, 31, 31]).unwrap().material(), 1);
        assert_eq!(material.sample([32, 31, 31]).unwrap().material(), 2);
    }

    #[test]
    fn classifier_includes_positive_face_cells_using_the_halo() {
        let fixture =
            ExtractionFixture::new(ExtractionFixtureKind::Plane, PageKey::new(0, [0, -1, 0]))
                .unwrap();
        assert!(fixture.is_active_cell([31, 31, 31]));
        assert_eq!(fixture.cell_case([32, 0, 0]), None);
    }
}
