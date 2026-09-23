//! Canonical material chunk and the existing mesh extractor's 9³ halo layout.

use crate::{VoxelChunkKey, VoxelDomain, VoxelUpdateError, VOXEL_CHUNK_EDGE, VOXEL_CHUNK_SAMPLES};

pub const VOXEL_PADDED_EDGE: usize = VOXEL_CHUNK_EDGE + 1;
pub const VOXEL_PADDED_WORDS: usize =
    (VOXEL_PADDED_EDGE * VOXEL_PADDED_EDGE * VOXEL_PADDED_EDGE).div_ceil(4);

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum VoxelChunkCodecError {
    InvalidLength(usize),
    PaletteTooLarge(usize),
    MaterialSlotOutOfRange { slot: u8, palette_len: usize },
    Domain(VoxelUpdateError),
    AddressOverflow(VoxelChunkKey),
    MissingChunk(VoxelChunkKey),
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

/// Build the packed 9³ input expected by `voxel_surface_extract.wgsl`.
/// Missing in-domain neighbors are unresolved, never silently treated as air.
/// Only samples beyond a bounded domain are known air.
pub fn bake_padded_chunk<'a>(
    center: VoxelChunkKey,
    domain: VoxelDomain,
    mut lookup: impl FnMut(VoxelChunkKey) -> Option<&'a [u8]>,
) -> Result<[u32; VOXEL_PADDED_WORDS], VoxelChunkCodecError> {
    domain
        .validate_key(center)
        .map_err(VoxelChunkCodecError::Domain)?;
    let mut neighbors: [Option<VoxelMaterialChunk>; 8] = std::array::from_fn(|_| None);
    for bits in 0..8usize {
        let next = [
            center.x.checked_add((bits & 1) as i64),
            center.y.checked_add(((bits >> 1) & 1) as i64),
            center.z.checked_add(((bits >> 2) & 1) as i64),
        ];
        let [Some(x), Some(y), Some(z)] = next else {
            if matches!(domain, VoxelDomain::Bounded { .. }) {
                continue;
            }
            return Err(VoxelChunkCodecError::AddressOverflow(center));
        };
        let key = VoxelChunkKey::new(x, y, z, center.lod);
        match domain.validate_key(key) {
            Ok(()) => {
                let bytes = lookup(key).ok_or(VoxelChunkCodecError::MissingChunk(key))?;
                neighbors[bits] = Some(VoxelMaterialChunk::decode(bytes)?);
            }
            Err(VoxelUpdateError::ChunkOutOfDomain(_)) => {}
            Err(error) => return Err(VoxelChunkCodecError::Domain(error)),
        }
    }

    let mut words = [0u32; VOXEL_PADDED_WORDS];
    for z in 0..VOXEL_PADDED_EDGE {
        for y in 0..VOXEL_PADDED_EDGE {
            for x in 0..VOXEL_PADDED_EDGE {
                let neighbor = usize::from(x == VOXEL_CHUNK_EDGE)
                    | (usize::from(y == VOXEL_CHUNK_EDGE) << 1)
                    | (usize::from(z == VOXEL_CHUNK_EDGE) << 2);
                let sample = neighbors[neighbor]
                    .as_ref()
                    .map_or(0, |chunk| chunk.sample(x % 8, y % 8, z % 8));
                let linear = z * 81 + y * 9 + x;
                words[linear / 4] |= u32::from(sample) << ((linear % 4) * 8);
            }
        }
    }
    Ok(words)
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

    #[test]
    fn padded_halo_reads_diagonal_neighbors_and_bounded_outside_is_air() {
        let center = VoxelChunkKey::new(0, 0, 0, 0);
        let all = VoxelDomain::Bounded {
            min: [0; 3],
            max: [1; 3],
            max_lod: 0,
        };
        let words = bake_padded_chunk(center, all, |key| {
            if key == center {
                Some(&[1u8][..])
            } else if key == VoxelChunkKey::new(1, 1, 1, 0) {
                Some(&[7u8][..])
            } else {
                Some(&[0u8][..])
            }
        })
        .unwrap();
        assert_eq!(words[0] & 0xff, 1);
        let corner = 8 * 81 + 8 * 9 + 8;
        assert_eq!((words[corner / 4] >> ((corner % 4) * 8)) & 0xff, 7);
        let boundary = VoxelDomain::Bounded {
            min: [0; 3],
            max: [0; 3],
            max_lod: 0,
        };
        assert!(bake_padded_chunk(center, boundary, |key| {
            (key == center).then_some(&[1u8][..])
        })
        .is_ok());
    }

    #[test]
    fn unbounded_missing_neighbor_is_deferred_not_air() {
        let center = VoxelChunkKey::new(0, 0, 0, 0);
        assert!(matches!(
            bake_padded_chunk(center, VoxelDomain::Unbounded { max_lod: 0 }, |key| {
                (key == center).then_some(&[1u8][..])
            }),
            Err(VoxelChunkCodecError::MissingChunk(_))
        ));
    }
}
