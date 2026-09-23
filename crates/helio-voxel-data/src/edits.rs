//! Revisioned sample edits over the SceneDB-owned material chunk map.

use std::collections::{BTreeMap, HashSet};

use crate::{
    VoxelBatchReceipt, VoxelBatchRevision, VoxelChunkBatch, VoxelChunkCodecError, VoxelChunkKey,
    VoxelChunkOp, VoxelChunkPayload, VoxelChunkUpdate, VoxelDomain, VoxelMaterialChunk,
    VoxelSourceWriter, VoxelUpdateError, MAX_VOXEL_BATCH_UPDATES, VOXEL_CHUNK_ENCODING_RAW,
    VOXEL_CHUNK_SCHEMA_VERSION,
};

/// Coordinates are signed sample-space coordinates, including for negative chunks.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct VoxelSampleEdit {
    pub xyz: [i64; 3],
    pub lod: u8,
    /// Zero removes a sample; positive values are one-based palette slots.
    pub material_slot: u8,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum VoxelEditError {
    TooManyEdits(usize),
    TooManyChunks(usize),
    DuplicateSample([i64; 3], u8),
    MaterialSlotOutOfRange { slot: u8, palette_len: usize },
    InvalidPaletteLength(usize),
    Update(VoxelUpdateError),
    Codec(VoxelChunkCodecError),
}

impl From<VoxelUpdateError> for VoxelEditError {
    fn from(value: VoxelUpdateError) -> Self {
        Self::Update(value)
    }
}

impl From<VoxelChunkCodecError> for VoxelEditError {
    fn from(value: VoxelChunkCodecError) -> Self {
        Self::Codec(value)
    }
}

impl VoxelSourceWriter {
    /// Stage edits to touched chunks only, then publish them as one optimistic
    /// revision. Call from a worker/script thread; a concurrent writer returns
    /// a stale-revision error and the caller can retry from fresh state.
    pub fn publish_sample_edits(
        &self,
        edits: &[VoxelSampleEdit],
        domain: VoxelDomain,
        material_ids: &[u32],
    ) -> Result<VoxelBatchReceipt, VoxelEditError> {
        if material_ids.len() > u8::MAX as usize {
            return Err(VoxelEditError::InvalidPaletteLength(material_ids.len()));
        }
        if edits.len() > MAX_VOXEL_BATCH_UPDATES {
            return Err(VoxelEditError::TooManyEdits(edits.len()));
        }
        let mut seen = HashSet::with_capacity(edits.len());
        let mut keys = BTreeMap::<VoxelChunkKey, Vec<(usize, u8)>>::new();
        for edit in edits {
            if !seen.insert((edit.xyz, edit.lod)) {
                return Err(VoxelEditError::DuplicateSample(edit.xyz, edit.lod));
            }
            if usize::from(edit.material_slot) > material_ids.len() {
                return Err(VoxelEditError::MaterialSlotOutOfRange {
                    slot: edit.material_slot,
                    palette_len: material_ids.len(),
                });
            }
            let [x, y, z] = edit.xyz;
            let key =
                VoxelChunkKey::new(x.div_euclid(8), y.div_euclid(8), z.div_euclid(8), edit.lod);
            domain.validate_key(key)?;
            let offset = z.rem_euclid(8) as usize * 64
                + y.rem_euclid(8) as usize * 8
                + x.rem_euclid(8) as usize;
            keys.entry(key)
                .or_default()
                .push((offset, edit.material_slot));
        }
        if keys.len() > MAX_VOXEL_BATCH_UPDATES {
            return Err(VoxelEditError::TooManyChunks(keys.len()));
        }

        let snapshot = self.snapshot_keys(keys.keys().copied())?;
        let mut encoded = Vec::with_capacity(keys.len());
        for (key, patches) in keys {
            let mut chunk = match snapshot.get(&key) {
                Some(Some(bytes)) => VoxelMaterialChunk::decode(bytes)?,
                _ => VoxelMaterialChunk::default(),
            };
            for (offset, slot) in patches {
                chunk.samples_mut()[offset] = slot;
            }
            chunk.validate_palette(material_ids)?;
            encoded.push((key, chunk.encode()));
        }
        let ops: Vec<_> = encoded
            .iter()
            .map(|(key, bytes)| {
                VoxelChunkOp::Upsert(VoxelChunkUpdate {
                    key: *key,
                    payload: VoxelChunkPayload {
                        encoding: VOXEL_CHUNK_ENCODING_RAW,
                        schema_version: VOXEL_CHUNK_SCHEMA_VERSION,
                        bytes,
                    },
                })
            })
            .collect();
        let expected = snapshot.revision;
        let publish = if ops.is_empty() {
            expected
        } else {
            expected.checked_add(1).ok_or(VoxelEditError::Update(
                VoxelUpdateError::NonSequentialRevision {
                    expected_next: None,
                    publish: expected,
                },
            ))?
        };
        self.publish_batch(&VoxelChunkBatch {
            terrain: self.terrain_id(),
            source: self.source_id(),
            revision: VoxelBatchRevision { expected, publish },
            domain,
            ops: &ops,
        })
        .map_err(Into::into)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{VoxelSourceId, VoxelTerrainId};
    use std::{
        collections::HashMap,
        sync::{Arc, RwLock},
    };

    #[test]
    fn edits_negative_coordinates_and_rejects_duplicate_or_invalid_slots() {
        let writer = VoxelSourceWriter::new(
            VoxelTerrainId(1),
            VoxelSourceId(2),
            Arc::new(RwLock::new((0, HashMap::new()))),
        );
        let domain = VoxelDomain::Unbounded { max_lod: 0 };
        let edit = VoxelSampleEdit {
            xyz: [-1, 0, -8],
            lod: 0,
            material_slot: 1,
        };
        assert_eq!(
            writer
                .publish_sample_edits(&[edit], domain, &[7])
                .unwrap()
                .revision,
            1
        );
        let bytes = writer.snapshot().unwrap();
        let chunk =
            VoxelMaterialChunk::decode(bytes.get(VoxelChunkKey::new(-1, 0, -1, 0)).unwrap())
                .unwrap();
        assert_eq!(chunk.sample(7, 0, 0), 1);
        assert!(matches!(
            writer.publish_sample_edits(&[edit, edit], domain, &[7]),
            Err(VoxelEditError::DuplicateSample(..))
        ));
        assert!(matches!(
            writer.publish_sample_edits(
                &[VoxelSampleEdit {
                    material_slot: 2,
                    ..edit
                }],
                domain,
                &[7]
            ),
            Err(VoxelEditError::MaterialSlotOutOfRange { .. })
        ));
    }
}
