use crate::{
    CellWord, GpuVoxelMaterial, PlanetId, PlanetPageKey, PAGE_CELL_COUNT, TRANSITION_FACE_MASK,
};
use std::collections::BTreeSet;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PageUpload {
    pub key: PlanetPageKey,
    pub generation: u64,
    pub cells: Box<[CellWord]>,
}

impl PageUpload {
    pub fn new(
        key: PlanetPageKey,
        generation: u64,
        cells: Vec<CellWord>,
    ) -> Result<Self, ContractError> {
        let upload = Self {
            key,
            generation,
            cells: cells.into_boxed_slice(),
        };
        upload.validate()?;
        Ok(upload)
    }

    pub fn validate(&self) -> Result<(), ContractError> {
        self.key.validate()?;
        if self.cells.len() != PAGE_CELL_COUNT {
            return Err(ContractError::CellCount(self.cells.len()));
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PageEvict {
    pub key: PlanetPageKey,
    pub generation: u64,
}

impl PageEvict {
    pub fn validate(self) -> Result<(), ContractError> {
        self.key.validate()?;
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VisiblePage {
    pub key: PlanetPageKey,
    pub generation: u64,
    pub transition_mask: u8,
}

impl VisiblePage {
    pub fn validate(self) -> Result<(), ContractError> {
        self.key.validate()?;
        if self.transition_mask & !TRANSITION_FACE_MASK != 0 {
            return Err(ContractError::TransitionMask(self.transition_mask));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct VisiblePageSet {
    pub frame_index: u64,
    pub pages: Vec<VisiblePage>,
}

impl VisiblePageSet {
    pub fn validate(&self, max_pages: usize) -> Result<(), ContractError> {
        if self.pages.len() > max_pages {
            return Err(ContractError::VisiblePageCount {
                actual: self.pages.len(),
                maximum: max_pages,
            });
        }
        let mut keys = BTreeSet::new();
        for page in &self.pages {
            page.validate()?;
            if !keys.insert(page.key) {
                return Err(ContractError::DuplicateVisiblePage(page.key));
            }
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct MaterialPaletteDelta {
    pub planet: PlanetId,
    pub generation: u64,
    pub first_material: u16,
    pub materials: Vec<GpuVoxelMaterial>,
}

impl MaterialPaletteDelta {
    pub fn validate(&self) -> Result<(), ContractError> {
        let end = usize::from(self.first_material)
            .checked_add(self.materials.len())
            .ok_or(ContractError::PaletteRange)?;
        if end > usize::from(u8::MAX) + 1 {
            return Err(ContractError::PaletteRange);
        }
        if self.materials.iter().any(|material| {
            material
                .base_color_roughness
                .iter()
                .chain(material.emissive_metalness.iter())
                .any(|value| !value.is_finite())
        }) {
            return Err(ContractError::NonFiniteMaterial);
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum ContractError {
    #[error(transparent)]
    Address(#[from] crate::AddressError),
    #[error("planetary page has {0} cells; exactly {PAGE_CELL_COUNT} are required")]
    CellCount(usize),
    #[error("transition mask {0:#010b} uses bits outside the six page faces")]
    TransitionMask(u8),
    #[error("visible page set has {actual} entries; the configured maximum is {maximum}")]
    VisiblePageCount { actual: usize, maximum: usize },
    #[error("visible page set contains duplicate key {0:?}")]
    DuplicateVisiblePage(PlanetPageKey),
    #[error("material palette delta exceeds the 256-entry material address space")]
    PaletteRange,
    #[error("material palette delta contains a non-finite value")]
    NonFiniteMaterial,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{PageKey, PlanetId};

    fn key(index: i64) -> PlanetPageKey {
        PlanetPageKey::new(PlanetId([1; 16]), PageKey::new(0, [index, 0, 0]))
    }

    #[test]
    fn page_upload_requires_one_complete_page() {
        assert_eq!(
            PageUpload::new(key(0), 0, vec![CellWord::AIR; 3]),
            Err(ContractError::CellCount(3))
        );
    }

    #[test]
    fn visible_sets_reject_duplicate_keys_and_invalid_face_bits() {
        let page = VisiblePage {
            key: key(0),
            generation: 1,
            transition_mask: 0,
        };
        assert_eq!(
            VisiblePageSet {
                frame_index: 1,
                pages: vec![page, page],
            }
            .validate(2),
            Err(ContractError::DuplicateVisiblePage(key(0)))
        );
        assert_eq!(
            VisiblePage {
                transition_mask: 0x80,
                ..page
            }
            .validate(),
            Err(ContractError::TransitionMask(0x80))
        );
    }

    #[test]
    fn palette_deltas_are_finite_and_stay_in_the_material_id_range() {
        let material = GpuVoxelMaterial {
            base_color_roughness: [1.0; 4],
            emissive_metalness: [0.0; 4],
        };
        assert!(MaterialPaletteDelta {
            planet: PlanetId::default(),
            generation: 1,
            first_material: 255,
            materials: vec![material],
        }
        .validate()
        .is_ok());
        assert_eq!(
            MaterialPaletteDelta {
                planet: PlanetId::default(),
                generation: 1,
                first_material: 255,
                materials: vec![material, material],
            }
            .validate(),
            Err(ContractError::PaletteRange)
        );
        assert_eq!(
            MaterialPaletteDelta {
                planet: PlanetId::default(),
                generation: 1,
                first_material: 0,
                materials: vec![GpuVoxelMaterial {
                    base_color_roughness: [f32::NAN; 4],
                    emissive_metalness: [0.0; 4],
                }],
            }
            .validate(),
            Err(ContractError::NonFiniteMaterial)
        );
    }
}
