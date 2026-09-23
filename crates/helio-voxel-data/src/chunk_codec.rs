//! Canonical material chunk encoding shared by voxel producers.

use crate::VOXEL_CHUNK_SAMPLES;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum VoxelChunkCodecError {
    InvalidLength(usize),
    PaletteTooLarge(usize),
    MaterialSlotOutOfRange { slot: u8, palette_len: usize },
}

/// One canonical 8³ chunk. `x` is the fastest-moving coordinate.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VoxelMaterialChunk {
    samples: [u8; VOXEL_CHUNK_SAMPLES],
}

impl Default for VoxelMaterialChunk {
    fn default() -> Self {
        Self {
            samples: [0; VOXEL_CHUNK_SAMPLES],
        }
    }
}

impl VoxelMaterialChunk {
    pub fn decode(bytes: &[u8]) -> Result<Self, VoxelChunkCodecError> {
        if bytes.is_empty() || bytes.len() > VOXEL_CHUNK_SAMPLES {
            return Err(VoxelChunkCodecError::InvalidLength(bytes.len()));
        }
        let mut result = Self::default();
        result.samples[..bytes.len()].copy_from_slice(bytes);
        Ok(result)
    }

    pub fn samples(&self) -> &[u8; VOXEL_CHUNK_SAMPLES] {
        &self.samples
    }

    pub(crate) fn samples_mut(&mut self) -> &mut [u8; VOXEL_CHUNK_SAMPLES] {
        &mut self.samples
    }

    pub fn sample(&self, x: usize, y: usize, z: usize) -> u8 {
        self.samples[z * 64 + y * 8 + x]
    }

    pub fn set(&mut self, x: usize, y: usize, z: usize, slot: u8) {
        self.samples[z * 64 + y * 8 + x] = slot;
    }

    pub fn validate_palette(&self, material_ids: &[u32]) -> Result<(), VoxelChunkCodecError> {
        if material_ids.len() > u8::MAX as usize {
            return Err(VoxelChunkCodecError::PaletteTooLarge(material_ids.len()));
        }
        if let Some(&slot) = self
            .samples
            .iter()
            .find(|&&slot| slot != 0 && usize::from(slot) > material_ids.len())
        {
            return Err(VoxelChunkCodecError::MaterialSlotOutOfRange {
                slot,
                palette_len: material_ids.len(),
            });
        }
        Ok(())
    }

    /// Encode a full chunk for publication. A deletion uses a batch Delete op
    /// instead of an implicit empty payload.
    pub fn encode(&self) -> [u8; VOXEL_CHUNK_SAMPLES] {
        self.samples
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn short_payload_zero_extends_and_palette_slots_are_one_based() {
        let chunk = VoxelMaterialChunk::decode(&[1, 2]).unwrap();
        assert_eq!(chunk.sample(0, 0, 0), 1);
        assert_eq!(chunk.sample(1, 0, 0), 2);
        assert_eq!(chunk.sample(2, 0, 0), 0);
        assert!(chunk.validate_palette(&[17, 23]).is_ok());
        assert_eq!(
            chunk.validate_palette(&[17]),
            Err(VoxelChunkCodecError::MaterialSlotOutOfRange {
                slot: 2,
                palette_len: 1,
            })
        );
    }
}
