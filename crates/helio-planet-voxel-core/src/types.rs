use bytemuck::{Pod, Zeroable};

pub type MaterialId = u8;

pub const LOD0_CELL_SIZE_METERS: f64 = 0.1;
pub const PAGE_EDGE_CELLS: i64 = 32;
/// Largest camera-local distance at which an `f32` unit in the last place is
/// no larger than one millimeter. Canonical positions never use this bound;
/// it applies only to the bounded render/interaction bubble.
pub const MILLIMETER_INTERACTION_RADIUS_METERS: f64 = 8_192.0;
pub const PAGE_EDGE: usize = PAGE_EDGE_CELLS as usize;
pub const PAGE_CELL_COUNT: usize = PAGE_EDGE * PAGE_EDGE * PAGE_EDGE;
pub const PAGE_CELL_BYTES: usize = PAGE_CELL_COUNT * core::mem::size_of::<CellWord>();
pub const MICROBRICK_EDGE: usize = 8;
pub const MICROBRICKS_PER_AXIS: usize = PAGE_EDGE / MICROBRICK_EDGE;
pub const MICROBRICK_COUNT: usize =
    MICROBRICKS_PER_AXIS * MICROBRICKS_PER_AXIS * MICROBRICKS_PER_AXIS;
pub const MAX_ADDRESSABLE_LOD: u8 = 57;
pub const TRANSITION_FACE_MASK: u8 = 0b00_111111;

/// Stable bit order used by visibility, extraction, and shaders.
#[repr(u8)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum TransitionFace {
    #[default]
    NegativeX = 0,
    PositiveX = 1,
    NegativeY = 2,
    PositiveY = 3,
    NegativeZ = 4,
    PositiveZ = 5,
}

impl TransitionFace {
    pub const ALL: [Self; 6] = [
        Self::NegativeX,
        Self::PositiveX,
        Self::NegativeY,
        Self::PositiveY,
        Self::NegativeZ,
        Self::PositiveZ,
    ];

    pub const fn index(self) -> u8 {
        self as u8
    }

    pub const fn bit(self) -> u8 {
        1 << self.index()
    }

    pub const fn axis(self) -> usize {
        (self.index() / 2) as usize
    }

    pub const fn is_positive(self) -> bool {
        self.index() & 1 != 0
    }

    pub const fn from_index(index: u8) -> Option<Self> {
        Some(match index {
            0 => Self::NegativeX,
            1 => Self::PositiveX,
            2 => Self::NegativeY,
            3 => Self::PositiveY,
            4 => Self::NegativeZ,
            5 => Self::PositiveZ,
            _ => return None,
        })
    }
}

#[repr(transparent)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, PartialOrd, Ord, Pod, Zeroable)]
pub struct PlanetId(pub [u8; 16]);

/// Canonical planet-space position.
///
/// The integer cell address is authoritative. The sub-cell remainder is
/// normalized to `[0, LOD0_CELL_SIZE_METERS)` on every axis, including for
/// negative positions. Absolute world positions therefore never depend on a
/// large `f32` coordinate.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PlanetPosition {
    lod0_cell: [i64; 3],
    subcell_m: [f64; 3],
}

impl PlanetPosition {
    pub fn new(lod0_cell: [i64; 3], mut subcell_m: [f64; 3]) -> Result<Self, PositionError> {
        for value in &mut subcell_m {
            if !value.is_finite() {
                return Err(PositionError::NonFinite);
            }
            if !(0.0..LOD0_CELL_SIZE_METERS).contains(value) {
                return Err(PositionError::SubcellOutOfRange);
            }
            if *value == -0.0 {
                *value = 0.0;
            }
        }
        Ok(Self {
            lod0_cell,
            subcell_m,
        })
    }

    pub const fn from_lod0_cell(lod0_cell: [i64; 3]) -> Self {
        Self {
            lod0_cell,
            subcell_m: [0.0; 3],
        }
    }

    /// Convenience conversion for camera/input values. Persistent terrain
    /// addresses should already arrive as integer cells plus a remainder.
    pub fn from_meters(meters: [f64; 3]) -> Result<Self, PositionError> {
        let mut cells = [0_i64; 3];
        let mut subcell_m = [0.0_f64; 3];
        for axis in 0..3 {
            let value = meters[axis];
            if !value.is_finite() {
                return Err(PositionError::NonFinite);
            }
            let cell = (value / LOD0_CELL_SIZE_METERS).floor();
            if cell < i64::MIN as f64 || cell >= -(i64::MIN as f64) {
                return Err(PositionError::CoordinateOverflow);
            }
            cells[axis] = cell as i64;
            let mut remainder = value - cell * LOD0_CELL_SIZE_METERS;
            // Floating-point division can leave a value one rounding step
            // outside the canonical interval at an exact cell boundary.
            if remainder < 0.0 {
                cells[axis] = cells[axis]
                    .checked_sub(1)
                    .ok_or(PositionError::CoordinateOverflow)?;
                remainder += LOD0_CELL_SIZE_METERS;
            } else if remainder >= LOD0_CELL_SIZE_METERS {
                cells[axis] = cells[axis]
                    .checked_add(1)
                    .ok_or(PositionError::CoordinateOverflow)?;
                remainder -= LOD0_CELL_SIZE_METERS;
            }
            if remainder == -0.0 {
                remainder = 0.0;
            }
            subcell_m[axis] = remainder;
        }
        Self::new(cells, subcell_m)
    }

    pub const fn lod0_cell(self) -> [i64; 3] {
        self.lod0_cell
    }

    pub const fn subcell_m(self) -> [f64; 3] {
        self.subcell_m
    }

    /// Computes `self - origin` without first constructing a large absolute
    /// floating-point coordinate.
    pub fn relative_meters(self, origin: Self) -> Result<[f64; 3], PositionError> {
        let mut relative = [0.0_f64; 3];
        for (axis, output) in relative.iter_mut().enumerate() {
            let cells = self.lod0_cell[axis]
                .checked_sub(origin.lod0_cell[axis])
                .ok_or(PositionError::CoordinateOverflow)?;
            *output = cells as f64 * LOD0_CELL_SIZE_METERS
                + (self.subcell_m[axis] - origin.subcell_m[axis]);
        }
        Ok(relative)
    }

    pub fn relative_to_lod0_cell(
        self,
        origin_lod0_cell: [i64; 3],
    ) -> Result<[f64; 3], PositionError> {
        self.relative_meters(Self::from_lod0_cell(origin_lod0_cell))
    }
}

/// Page-snapped camera frame used to derive all bounded GPU coordinates.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PlanetRenderFrame {
    planet: PlanetId,
    camera: PlanetPosition,
    origin_lod0_cell: [i64; 3],
    frame_index: u64,
}

impl PlanetRenderFrame {
    pub fn new(planet: PlanetId, camera: PlanetPosition, frame_index: u64) -> Self {
        let origin_lod0_cell = camera
            .lod0_cell
            .map(|axis| axis.div_euclid(PAGE_EDGE_CELLS) * PAGE_EDGE_CELLS);
        Self {
            planet,
            camera,
            origin_lod0_cell,
            frame_index,
        }
    }

    pub const fn planet(self) -> PlanetId {
        self.planet
    }

    pub const fn camera(self) -> PlanetPosition {
        self.camera
    }

    pub const fn origin_lod0_cell(self) -> [i64; 3] {
        self.origin_lod0_cell
    }

    pub const fn frame_index(self) -> u64 {
        self.frame_index
    }

    pub fn camera_relative_m(self) -> [f64; 3] {
        self.camera
            .relative_to_lod0_cell(self.origin_lod0_cell)
            .expect("a page-snapped camera origin cannot overflow")
    }

    pub fn camera_local_meters(self, position: PlanetPosition) -> Result<[f64; 3], PositionError> {
        position.relative_meters(self.camera)
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct PageKey {
    pub lod: u8,
    pub page_xyz: [i64; 3],
}

impl PageKey {
    pub const fn new(lod: u8, page_xyz: [i64; 3]) -> Self {
        Self { lod, page_xyz }
    }

    pub fn validate(self) -> Result<(), AddressError> {
        if self.lod > MAX_ADDRESSABLE_LOD {
            return Err(AddressError::UnsupportedLod(self.lod));
        }
        self.lod0_cell_min()?;
        Ok(())
    }

    pub fn parent(self) -> Option<Self> {
        let lod = self.lod.checked_add(1)?;
        (lod <= MAX_ADDRESSABLE_LOD).then(|| Self {
            lod,
            page_xyz: self.page_xyz.map(|axis| axis.div_euclid(2)),
        })
    }

    pub fn lod0_cell_span(self) -> Result<i64, AddressError> {
        if self.lod > MAX_ADDRESSABLE_LOD {
            return Err(AddressError::UnsupportedLod(self.lod));
        }
        PAGE_EDGE_CELLS
            .checked_shl(u32::from(self.lod))
            .ok_or(AddressError::CoordinateOverflow)
    }

    pub fn lod0_cell_min(self) -> Result<[i64; 3], AddressError> {
        let span = self.lod0_cell_span()?;
        Ok([
            self.page_xyz[0]
                .checked_mul(span)
                .ok_or(AddressError::CoordinateOverflow)?,
            self.page_xyz[1]
                .checked_mul(span)
                .ok_or(AddressError::CoordinateOverflow)?,
            self.page_xyz[2]
                .checked_mul(span)
                .ok_or(AddressError::CoordinateOverflow)?,
        ])
    }

    /// Converts an absolute page address into a bounded camera-local GPU
    /// address. The subtraction happens in canonical integer space before the
    /// checked narrowing to `i32`.
    pub fn relative_lod0_cell_min(
        self,
        frame_origin_lod0_cell: [i64; 3],
    ) -> Result<[i32; 3], AddressError> {
        let absolute = self.lod0_cell_min()?;
        let mut relative = [0_i32; 3];
        for axis in 0..3 {
            let delta = absolute[axis]
                .checked_sub(frame_origin_lod0_cell[axis])
                .ok_or(AddressError::CoordinateOverflow)?;
            relative[axis] = i32::try_from(delta).map_err(|_| AddressError::OutsideRenderFrame)?;
        }
        Ok(relative)
    }

    pub fn address_lod0_cell(lod: u8, cell_xyz: [i64; 3]) -> Result<(Self, [u8; 3]), AddressError> {
        let key = Self::new(lod, [0; 3]);
        let scale = key.lod0_cell_span()? / PAGE_EDGE_CELLS;
        let span = PAGE_EDGE_CELLS
            .checked_mul(scale)
            .ok_or(AddressError::CoordinateOverflow)?;
        let page_xyz = cell_xyz.map(|axis| axis.div_euclid(span));
        let local = cell_xyz.map(|axis| (axis.rem_euclid(span) / scale) as u8);
        Ok((Self { lod, page_xyz }, local))
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct PlanetPageKey {
    pub planet: PlanetId,
    pub page: PageKey,
}

impl PlanetPageKey {
    pub const fn new(planet: PlanetId, page: PageKey) -> Self {
        Self { planet, page }
    }

    pub fn validate(self) -> Result<(), AddressError> {
        self.page.validate()
    }
}

#[repr(transparent)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Pod, Zeroable)]
pub struct CellWord(pub u32);

impl CellWord {
    pub const AIR: Self = Self::new(i16::MAX, 0, 0);

    pub const fn new(density: i16, material: MaterialId, flags: u8) -> Self {
        Self((density as u16 as u32) | ((material as u32) << 16) | ((flags as u32) << 24))
    }

    pub const fn density(self) -> i16 {
        self.0 as u16 as i16
    }

    pub const fn material(self) -> MaterialId {
        (self.0 >> 16) as u8
    }

    pub const fn flags(self) -> u8 {
        (self.0 >> 24) as u8
    }

    pub const fn is_solid(self) -> bool {
        self.density() <= 0
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
pub enum AddressError {
    #[error("planetary page LOD {0} exceeds the addressable maximum")]
    UnsupportedLod(u8),
    #[error("planetary page coordinate arithmetic overflowed")]
    CoordinateOverflow,
    #[error("planetary page lies outside the current camera-local render frame")]
    OutsideRenderFrame,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
pub enum PositionError {
    #[error("planet position contains a non-finite value")]
    NonFinite,
    #[error("planet position sub-cell remainder must be in [0, 0.1) meters")]
    SubcellOutOfRange,
    #[error("planet position coordinate arithmetic overflowed")]
    CoordinateOverflow,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transition_face_bits_are_stable_and_exhaust_the_mask() {
        let mut mask = 0_u8;
        for (index, face) in TransitionFace::ALL.into_iter().enumerate() {
            assert_eq!(face.index(), index as u8);
            assert_eq!(TransitionFace::from_index(index as u8), Some(face));
            assert_eq!(face.axis(), index / 2);
            assert_eq!(face.is_positive(), index & 1 != 0);
            mask |= face.bit();
        }
        assert_eq!(mask, TRANSITION_FACE_MASK);
        assert_eq!(TransitionFace::from_index(6), None);
    }

    #[test]
    fn cell_word_matches_the_authoritative_four_byte_layout() {
        assert_eq!(core::mem::size_of::<CellWord>(), 4);
        let word = CellWord::new(-123, 17, 9);
        assert_eq!(word.0, 0x0911_ff85);
        assert_eq!(word.density(), -123);
        assert_eq!(word.material(), 17);
        assert_eq!(word.flags(), 9);
        assert!(word.is_solid());
        assert!(!CellWord::AIR.is_solid());
    }

    #[test]
    fn negative_page_boundaries_use_euclidean_division() {
        for lod in 0..=30 {
            let scale = 1_i64 << lod;
            let span = PAGE_EDGE_CELLS * scale;
            for coordinate in [-span - 1, -span, -span + 1, -1, 0, 1, span - 1, span] {
                let (key, local) = PageKey::address_lod0_cell(lod, [coordinate; 3]).unwrap();
                let minimum = key.lod0_cell_min().unwrap();
                for axis in 0..3 {
                    assert!(usize::from(local[axis]) < PAGE_EDGE);
                    let reconstructed = minimum[axis] + i64::from(local[axis]) * scale;
                    assert!(reconstructed <= coordinate);
                    assert!(coordinate < reconstructed + scale);
                }
            }
        }
    }

    #[test]
    fn camera_local_narrowing_is_checked() {
        let page = PageKey::new(0, [4, -2, 1]);
        assert_eq!(
            page.relative_lod0_cell_min([64, -64, 0]).unwrap(),
            [64, 0, 32]
        );
        assert_eq!(
            PageKey::new(0, [i64::from(i32::MAX), 0, 0]).relative_lod0_cell_min([0; 3]),
            Err(AddressError::OutsideRenderFrame)
        );
    }

    #[test]
    fn canonical_positions_normalize_negative_meter_coordinates() {
        let position = PlanetPosition::from_meters([-0.001, -0.1, -3.201]).unwrap();
        assert_eq!(position.lod0_cell(), [-1, -1, -33]);
        let expected = [0.099, 0.0, 0.099];
        for (actual, expected) in position.subcell_m().into_iter().zip(expected) {
            assert!((actual - expected).abs() < 1.0e-12);
        }
    }

    #[test]
    fn canonical_positions_reject_invalid_remainders_and_remove_negative_zero() {
        assert_eq!(
            PlanetPosition::new([0; 3], [0.1, 0.0, 0.0]),
            Err(PositionError::SubcellOutOfRange)
        );
        assert_eq!(
            PlanetPosition::new([0; 3], [f64::NAN, 0.0, 0.0]),
            Err(PositionError::NonFinite)
        );
        let zero = PlanetPosition::new([0; 3], [-0.0, 0.0, 0.0]).unwrap();
        assert_eq!(zero.subcell_m()[0].to_bits(), 0.0_f64.to_bits());
    }

    #[test]
    fn render_frames_are_page_snapped_on_both_sides_of_zero() {
        let positive = PlanetRenderFrame::new(
            PlanetId::default(),
            PlanetPosition::new([63_710_017, 5, 0], [0.025, 0.0, 0.0]).unwrap(),
            7,
        );
        assert_eq!(positive.origin_lod0_cell(), [63_710_016, 0, 0]);
        assert_eq!(positive.camera_relative_m(), [0.125, 0.5, 0.0]);

        let negative = PlanetRenderFrame::new(
            PlanetId::default(),
            PlanetPosition::new([-63_710_017, -1, 0], [0.025, 0.05, 0.0]).unwrap(),
            8,
        );
        assert_eq!(negative.origin_lod0_cell(), [-63_710_048, -32, 0]);
        assert_eq!(negative.camera_relative_m(), [3.125, 3.15, 0.0]);
    }

    #[test]
    fn earth_radius_and_orbit_offsets_do_not_use_absolute_f32() {
        let ground =
            PlanetPosition::new([63_710_000, -63_710_000, 0], [0.001, 0.099, 0.05]).unwrap();
        let orbit =
            PlanetPosition::new([67_710_000, -63_710_000, 0], [0.001, 0.099, 0.05]).unwrap();
        assert_eq!(
            orbit.relative_meters(ground).unwrap(),
            [400_000.0, 0.0, 0.0]
        );

        let frame = PlanetRenderFrame::new(PlanetId([3; 16]), ground, 42);
        let nearby =
            PlanetPosition::new([63_710_007, -63_710_004, 0], [0.0015, 0.0985, 0.0505]).unwrap();
        let local = frame.camera_local_meters(nearby).unwrap();
        let expected = [0.7005, -0.4005, 0.0005];
        for (actual, expected) in local.into_iter().zip(expected) {
            assert!((actual - expected).abs() < 1.0e-12);
        }
    }
}
