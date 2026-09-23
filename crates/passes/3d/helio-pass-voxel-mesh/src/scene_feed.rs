//! Off-frame preparation of SceneDB voxel rows for transient GPU residency.

use std::{
    collections::{HashMap, HashSet},
    hash::{Hash, Hasher},
    sync::{
        mpsc::{self, Receiver, SyncSender, TryRecvError, TrySendError},
        Arc, Mutex,
    },
    thread,
};

use crate::{
    bake_padded_chunk_with_policy, VoxelChunkKey, VoxelDomain, VoxelEntryId, VoxelMaterialChunk,
    VoxelMissingChunkPolicy, VoxelPayloadStore, VoxelPreparedBrick, VoxelSourceId,
    VoxelSourceWriter, VoxelTerrainId, VOXEL_MESH_MAX_BRICKS, VOXEL_MODE_CUBES, VOXEL_MODE_SURFACE,
};

/// A short-lived description collected from one live SceneDB component row.
/// The Arc points to canonical in-memory bytes; no GPU handle enters the worker.
#[derive(Clone)]
pub struct VoxelSceneEntry {
    pub id: VoxelEntryId,
    pub store: VoxelPayloadStore,
    pub domain: VoxelDomain,
    pub source_revision: u64,
    pub origin: [f64; 3],
    pub voxel_size: f64,
    pub material_ids: Vec<u32>,
    pub smooth_surface: bool,
}

#[derive(Clone, Debug, Default)]
pub struct VoxelSceneFeedStatus {
    pub tracked_entries: usize,
    pub in_flight_entries: usize,
    pub deferred_requests: usize,
    pub failed_entries: usize,
    pub stale_results: u64,
    pub prepared_entries: u64,
    pub selected_bricks: usize,
    pub truncated_entries: usize,
    pub last_error: Option<String>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Tag {
    pub generation: u64,
    pub revision: u64,
    center: [i64; 3],
}

struct Tracked {
    config_hash: u64,
    store_ptr: usize,
    entry: VoxelSceneEntry,
    desired: Tag,
    in_flight: Option<Tag>,
    retry_at: u64,
    last_error: Option<String>,
}

struct PrepRequest {
    entry: VoxelSceneEntry,
    tag: Tag,
}

pub(crate) struct PrepResult {
    pub id: VoxelEntryId,
    pub tag: Tag,
    pub bricks: Result<Vec<VoxelPreparedBrick>, String>,
    pub truncated: bool,
}

pub(crate) struct VoxelSceneFeed {
    request_tx: SyncSender<PrepRequest>,
    result_rx: Mutex<Receiver<PrepResult>>,
    tracked: HashMap<VoxelEntryId, Tracked>,
    tick: u64,
    stale_results: u64,
    prepared_entries: u64,
    selected_bricks: usize,
    truncated_entries: usize,
    blocked_reads: usize,
    last_error: Option<String>,
}

impl VoxelSceneFeed {
    pub fn new() -> Self {
        let (request_tx, request_rx) = mpsc::sync_channel::<PrepRequest>(2);
        let (result_tx, result_rx) = mpsc::sync_channel::<PrepResult>(2);
        // The worker owns only CPU Arc handles. Dropping the pass closes both
        // channels and lets an in-progress bounded preparation finish/exit.
        thread::Builder::new()
            .name("voxel-scene-prep".into())
            .spawn(move || {
                while let Ok(request) = request_rx.recv() {
                    let result = prepare_entry(request);
                    if result_tx.send(result).is_err() {
                        break;
                    }
                }
            })
            .expect("voxel CPU preparation worker must start");
        Self {
            request_tx,
            result_rx: Mutex::new(result_rx),
            tracked: HashMap::new(),
            tick: 0,
            stale_results: 0,
            prepared_entries: 0,
            selected_bricks: 0,
            truncated_entries: 0,
            blocked_reads: 0,
            last_error: None,
        }
    }

    /// Nonblocking reconciliation. `already_queued` includes active and
    /// staging revisions, while `needs_rebuild` identifies evicted cache rows.
    pub fn reconcile(
        &mut self,
        entries: impl IntoIterator<Item = VoxelSceneEntry>,
        camera: [f64; 3],
        mut already_queued: impl FnMut(VoxelEntryId, u64, u64) -> bool,
        mut needs_rebuild: impl FnMut(VoxelEntryId) -> bool,
    ) -> Vec<VoxelEntryId> {
        self.tick = self.tick.saturating_add(1);
        self.blocked_reads = 0;
        let mut seen = HashSet::new();
        for entry in entries {
            let id = entry.id;
            seen.insert(id);
            let revision = match entry.store.try_read() {
                Ok(state) => state.0,
                Err(_) => {
                    self.blocked_reads += 1;
                    continue;
                }
            };
            let center = chunk_center(camera, entry.origin, entry.voxel_size);
            let config_hash = hash_config(&entry);
            let store_ptr = Arc::as_ptr(&entry.store) as usize;
            let tracked = self.tracked.entry(id).or_insert_with(|| Tracked {
                config_hash,
                store_ptr,
                entry: entry.clone(),
                desired: Tag {
                    generation: 1,
                    revision,
                    center,
                },
                in_flight: None,
                retry_at: 0,
                last_error: None,
            });
            if tracked.config_hash != config_hash
                || tracked.store_ptr != store_ptr
                || tracked.desired.center != center
            {
                tracked.desired.generation = tracked.desired.generation.saturating_add(1);
                tracked.config_hash = config_hash;
                tracked.store_ptr = store_ptr;
                tracked.retry_at = 0;
                tracked.last_error = None;
            }
            if tracked.desired.revision != revision {
                tracked.retry_at = 0;
                tracked.last_error = None;
            }
            tracked.entry = entry;
            tracked.desired.revision = revision;
            tracked.desired.center = center;
            if tracked.in_flight.is_none()
                && self.tick >= tracked.retry_at
                && (!already_queued(id, tracked.desired.generation, revision) || needs_rebuild(id))
            {
                let request = PrepRequest {
                    entry: tracked.entry.clone(),
                    tag: tracked.desired,
                };
                match self.request_tx.try_send(request) {
                    Ok(()) => tracked.in_flight = Some(tracked.desired),
                    Err(TrySendError::Full(_)) => {}
                    Err(TrySendError::Disconnected(_)) => {
                        tracked.last_error = Some("voxel preparation worker stopped".into());
                        tracked.retry_at = self.tick.saturating_add(60);
                    }
                }
            }
        }
        let removed: Vec<_> = self
            .tracked
            .keys()
            .filter(|id| !seen.contains(id))
            .copied()
            .collect();
        for id in &removed {
            self.tracked.remove(id);
        }
        removed
    }

    pub fn drain_ready(&mut self) -> Vec<PrepResult> {
        let mut ready = Vec::new();
        let result_rx = self
            .result_rx
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        loop {
            let result = match result_rx.try_recv() {
                Ok(result) => result,
                Err(TryRecvError::Empty | TryRecvError::Disconnected) => break,
            };
            let Some(tracked) = self.tracked.get_mut(&result.id) else {
                self.stale_results += 1;
                continue;
            };
            if tracked.in_flight == Some(result.tag) {
                tracked.in_flight = None;
            }
            if tracked.desired != result.tag {
                self.stale_results += 1;
                continue;
            }
            if let Ok(bricks) = &result.bricks {
                self.prepared_entries += 1;
                self.selected_bricks = bricks.len();
                self.truncated_entries = usize::from(result.truncated);
                tracked.last_error = None;
            } else if let Err(message) = &result.bricks {
                tracked.last_error = Some(message.clone());
                tracked.retry_at = self.tick.saturating_add(60);
                self.last_error = Some(message.clone());
            }
            ready.push(result);
        }
        ready
    }

    pub fn record_residency_error(&mut self, id: VoxelEntryId, message: String) {
        if let Some(tracked) = self.tracked.get_mut(&id) {
            tracked.last_error = Some(message.clone());
            tracked.retry_at = self.tick.saturating_add(60);
        }
        self.last_error = Some(message);
    }

    pub fn status(
        &self,
        mut already_queued: impl FnMut(VoxelEntryId, u64, u64) -> bool,
    ) -> VoxelSceneFeedStatus {
        VoxelSceneFeedStatus {
            tracked_entries: self.tracked.len(),
            in_flight_entries: self
                .tracked
                .values()
                .filter(|entry| entry.in_flight.is_some())
                .count(),
            deferred_requests: self.blocked_reads
                + self
                    .tracked
                    .iter()
                    .filter(|(id, entry)| {
                        entry.in_flight.is_none()
                            && !already_queued(
                                **id,
                                entry.desired.generation,
                                entry.desired.revision,
                            )
                    })
                    .count(),
            failed_entries: self
                .tracked
                .values()
                .filter(|entry| entry.last_error.is_some())
                .count(),
            stale_results: self.stale_results,
            prepared_entries: self.prepared_entries,
            selected_bricks: self.selected_bricks,
            truncated_entries: self.truncated_entries,
            last_error: self.last_error.clone(),
        }
    }
}

fn chunk_center(camera: [f64; 3], origin: [f64; 3], voxel_size: f64) -> [i64; 3] {
    if !voxel_size.is_finite() || voxel_size <= 0.0 {
        return [0; 3];
    }
    std::array::from_fn(|axis| {
        let chunk = ((camera[axis] - origin[axis]) / (voxel_size * 8.0)).floor();
        if !chunk.is_finite() {
            0
        } else {
            chunk.clamp(i64::MIN as f64, i64::MAX as f64) as i64
        }
    })
}

fn hash_config(entry: &VoxelSceneEntry) -> u64 {
    let mut hash = std::collections::hash_map::DefaultHasher::new();
    entry.source_revision.hash(&mut hash);
    entry.voxel_size.to_bits().hash(&mut hash);
    entry.origin.map(f64::to_bits).hash(&mut hash);
    entry.material_ids.hash(&mut hash);
    entry.smooth_surface.hash(&mut hash);
    match entry.domain {
        VoxelDomain::Unbounded { max_lod } => {
            1u8.hash(&mut hash);
            max_lod.hash(&mut hash);
        }
        VoxelDomain::Bounded { min, max, max_lod } => {
            0u8.hash(&mut hash);
            min.hash(&mut hash);
            max.hash(&mut hash);
            max_lod.hash(&mut hash);
        }
    }
    hash.finish()
}

fn smooth_cell_owner_mask(key: VoxelChunkKey, selected: &HashSet<VoxelChunkKey>) -> u8 {
    let mut owned = 0u8;
    for negative in 0..8u8 {
        let Some(base_x) = key.x.checked_sub(i64::from(negative & 1)) else {
            continue;
        };
        let Some(base_y) = key.y.checked_sub(i64::from((negative >> 1) & 1)) else {
            continue;
        };
        let Some(base_z) = key.z.checked_sub(i64::from((negative >> 2) & 1)) else {
            continue;
        };
        let mut winner: Option<VoxelChunkKey> = None;
        for subset in 0..8u8 {
            if subset & !negative != 0 {
                continue;
            }
            let candidate = VoxelChunkKey::new(
                base_x + i64::from(subset & 1),
                base_y + i64::from((subset >> 1) & 1),
                base_z + i64::from((subset >> 2) & 1),
                key.lod,
            );
            if selected.contains(&candidate) && winner.is_none_or(|current| candidate < current) {
                winner = Some(candidate);
            }
        }
        if winner == Some(key) {
            owned |= 1 << negative;
        }
    }
    owned
}

fn prepare_entry(request: PrepRequest) -> PrepResult {
    let id = request.entry.id;
    let tag = request.tag;
    let result = (|| -> Result<(Vec<VoxelPreparedBrick>, bool), String> {
        let entry = request.entry;
        if !entry.voxel_size.is_finite() || entry.voxel_size <= 0.0 {
            return Err("voxel_size must be finite and positive".into());
        }
        if entry.material_ids.len() > 255 {
            return Err("voxel material palette exceeds 255 IDs".into());
        }
        let writer = VoxelSourceWriter::new(
            VoxelTerrainId(u128::from(id.entity_bits)),
            VoxelSourceId(0),
            entry.store,
        );
        let selected = writer
            .select_nearest_with_halo(tag.center, VOXEL_MESH_MAX_BRICKS as usize)
            .map_err(|error| format!("voxel snapshot selection failed: {error:?}"))?;
        if selected.revision != tag.revision {
            return Err("voxel source revision changed during preparation".into());
        }
        let center_set: HashSet<_> = selected.centers.iter().copied().collect();
        let mut bricks = Vec::with_capacity(selected.centers.len());
        for key in selected.centers {
            let center_bytes = selected.chunks.get(&key).expect("selected center exists");
            VoxelMaterialChunk::decode(center_bytes)
                .map_err(|e| format!("voxel chunk {key:?}: {e:?}"))?
                .validate_palette(&entry.material_ids)
                .map_err(|e| format!("voxel chunk {key:?}: {e:?}"))?;
            let words = bake_padded_chunk_with_policy(
                key,
                entry.domain,
                VoxelMissingChunkPolicy::KnownAir,
                |neighbor| selected.chunks.get(&neighbor).map(AsRef::as_ref),
            )
            .map_err(|e| format!("voxel halo {key:?}: {e:?}"))?;
            let scale = entry.voxel_size * 2f64.powi(i32::from(key.lod));
            let origin = [
                entry.origin[0] + key.x as f64 * 8.0 * scale,
                entry.origin[1] + key.y as f64 * 8.0 * scale,
                entry.origin[2] + key.z as f64 * 8.0 * scale,
            ];
            if origin.iter().any(|value| !value.is_finite())
                || !scale.is_finite()
                || scale <= 0.0
                || scale > f32::MAX as f64
            {
                return Err(format!(
                    "voxel chunk {key:?} has an unrepresentable GPU transform"
                ));
            }
            bricks.push(VoxelPreparedBrick {
                key,
                words,
                origin,
                voxel_size: scale as f32,
                mode: if entry.smooth_surface {
                    VOXEL_MODE_SURFACE
                } else {
                    VOXEL_MODE_CUBES
                },
                owner_mask: smooth_cell_owner_mask(key, &center_set),
                material_ids: entry.material_ids.clone(),
            });
        }
        Ok((bricks, selected.truncated))
    })();
    match result {
        Ok((bricks, truncated)) => PrepResult {
            id,
            tag,
            bricks: Ok(bricks),
            truncated,
        },
        Err(error) => PrepResult {
            id,
            tag,
            bricks: Err(error),
            truncated: false,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::RwLock;

    #[test]
    fn sparse_scene_chunk_prepares_with_known_air_halo_and_scene_palette() {
        let mut map = HashMap::new();
        map.insert([0, 0, 0, 0], Arc::<[u8]>::from([1]));
        let entry = VoxelSceneEntry {
            id: VoxelEntryId {
                entity_bits: 7,
                kind: 0,
            },
            store: Arc::new(RwLock::new((1, map))),
            domain: VoxelDomain::Unbounded { max_lod: 0 },
            source_revision: 0,
            origin: [0.0; 3],
            voxel_size: 1.0,
            material_ids: vec![42],
            smooth_surface: false,
        };
        let result = prepare_entry(PrepRequest {
            entry,
            tag: Tag {
                generation: 1,
                revision: 1,
                center: [0; 3],
            },
        });
        let bricks = result.bricks.unwrap();
        assert_eq!(bricks.len(), 1);
        assert_eq!((bricks[0].words[111 / 4] >> ((111 % 4) * 8)) & 0xff, 1);
        assert_eq!(bricks[0].material_ids, [42]);
        assert_eq!(bricks[0].owner_mask, 0xff);
    }

    #[test]
    fn adjacent_selected_chunks_have_one_smooth_boundary_owner() {
        let a = VoxelChunkKey::new(-1, 0, 0, 0);
        let b = VoxelChunkKey::new(0, 0, 0, 0);
        let selected = HashSet::from([a, b]);
        assert_eq!(smooth_cell_owner_mask(b, &selected) & 0b10, 0);
        assert_ne!(smooth_cell_owner_mask(a, &selected) & 0b1, 0);
    }

    #[test]
    fn contended_component_read_remains_deferred_until_reconciled() {
        let entry = VoxelSceneEntry {
            id: VoxelEntryId {
                entity_bits: 11,
                kind: 0,
            },
            store: Arc::new(RwLock::new((0, HashMap::new()))),
            domain: VoxelDomain::Bounded {
                min: [0; 3],
                max: [0; 3],
                max_lod: 0,
            },
            source_revision: 0,
            origin: [0.0; 3],
            voxel_size: 1.0,
            material_ids: vec![0],
            smooth_surface: false,
        };
        let mut feed = VoxelSceneFeed::new();
        let held = entry.store.write().unwrap();
        feed.reconcile([entry.clone()], [0.0; 3], |_, _, _| false, |_| false);
        assert_eq!(feed.status(|_, _, _| false).deferred_requests, 1);
        drop(held);
        feed.reconcile([entry], [0.0; 3], |_, _, _| false, |_| false);
        assert_eq!(feed.status(|_, _, _| false).in_flight_entries, 1);
    }
}
