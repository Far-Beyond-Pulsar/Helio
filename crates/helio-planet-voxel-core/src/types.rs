use bytemuck::{Pod, Zeroable};

pub type MaterialId = u8;

pub const LOD0_CELL_SIZE_METERS: f64 = 0.1;
pub const PAGE_EDGE_CELLS: i64 = 32;
pub const PAGE_EDGE: usize = PAGE_EDGE_CELLS as usize;
pub const PAGE_CELL_COUNT: usize = PAGE_EDGE * PAGE_EDGE * PAGE_EDGE;
pub const PAGE_CELL_BYTES: usize = PAGE_CELL_COUNT * core::mem::size_of::<CellWord>();
pub const MICROBRICK_EDGE: usize = 8;
pub const MICROBRICKS_PER_AXIS: usize = PAGE_EDGE / MICROBRICK_EDGE;
pub const MICROBRICK_COUNT: usize =
    MICROBRICKS_PER_AXIS * MICROBRICKS_PER_AXIS * MICROBRICKS_PER_AXIS;
pub const MAX_ADDRESSABLE_LOD: u8 = 57;
pub const TRANSITION_FACE_MASK: u8 = 0b00_111111;

#[repr(transparent)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, PartialOrd, Ord, Pod, Zeroable)]
pub struct PlanetId(pub [u8; 16]);

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

#[cfg(test)]
mod tests {
    use super::*;

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
}
