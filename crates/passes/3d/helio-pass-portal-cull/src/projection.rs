//! Projection bridge from resolved authored portal topology to GPU rows.
//!
//! The resolver owns authored/runtime topology. This module owns the derived
//! frame representation consumed by the existing SceneDB portal buffers. A
//! view row represents one resolved source/target projection, rather than
//! mutating the authored `PortalComponent` row. That distinction matters when
//! one indexed sublevel is instanced more than once: the same authored portal
//! can then legitimately produce several projection rows.

use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fmt;

use glam::Mat4;
use pulsar_scenedb::Entity;

use crate::components::{
    PortalChainComponent, PortalProjectionCountsComponent, PortalViewComponent,
};
use crate::resolver::{
    PortalOccurrence, PortalProjection, ResolutionError, ResolvedPortalChain, SubLevelResolver,
};
use crate::{MAX_CHAIN_DEPTH, MAX_PORTAL_CHAINS};
use helio_pass_gbuffer::SubLevelIndex;

/// The coordinate-space slot reserved for root/world coordinates.
pub const IDENTITY_COORDINATE_SPACE_SLOT: u32 = 0;

/// The G-buffer coordinate-space table has 32 entries, including identity.
pub const MAX_COORDINATE_SPACES: usize = 32;

/// A stable identity for one runtime portal occurrence.
///
/// The authored entity alone is not sufficient because an indexed sublevel
/// can be placed repeatedly, and each actor path creates a distinct runtime
/// occurrence.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct RuntimePortalKey {
    pub entity: Entity,
    pub sublevel_index: SubLevelIndex,
    pub actor_path: Vec<Entity>,
}

impl RuntimePortalKey {
    pub fn from_occurrence(occurrence: &PortalOccurrence) -> Self {
        Self {
            entity: occurrence.entity,
            sublevel_index: occurrence.context.sublevel_index,
            actor_path: occurrence.context.actor_path.clone(),
        }
    }

    fn cmp_path(&self, other: &Self) -> Ordering {
        self.actor_path
            .iter()
            .map(|entity| entity.bits())
            .cmp(other.actor_path.iter().map(|entity| entity.bits()))
    }
}

impl Ord for RuntimePortalKey {
    fn cmp(&self, other: &Self) -> Ordering {
        self.sublevel_index
            .cmp(&other.sublevel_index)
            .then_with(|| self.cmp_path(other))
            .then_with(|| self.entity.bits().cmp(&other.entity.bits()))
    }
}

impl PartialOrd for RuntimePortalKey {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Stable identity for a derived view row: a source occurrence displaying a
/// particular target occurrence.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct PortalProjectionKey {
    pub source: RuntimePortalKey,
    pub target: RuntimePortalKey,
}

impl PortalProjectionKey {
    pub fn from_projection(projection: &PortalProjection) -> Self {
        Self {
            source: RuntimePortalKey::from_occurrence(&projection.source),
            target: RuntimePortalKey::from_occurrence(&projection.target),
        }
    }
}

impl Ord for PortalProjectionKey {
    fn cmp(&self, other: &Self) -> Ordering {
        self.source
            .cmp(&other.source)
            .then_with(|| self.target.cmp(&other.target))
    }
}

impl PartialOrd for PortalProjectionKey {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Errors raised while converting a valid resolver result into the fixed GPU
/// projection ABI.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ProjectionError {
    Resolution(ResolutionError),
    CoordinateSpaceCapacity {
        requested: usize,
        maximum: usize,
    },
    ChainCapacity {
        requested: usize,
        maximum: usize,
    },
    SceneRowCount {
        kind: &'static str,
        expected: usize,
        actual: usize,
    },
    NonDenseSceneRow {
        kind: &'static str,
        dense_index: usize,
        entity_index: u32,
    },
}

impl fmt::Display for ProjectionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Resolution(error) => error.fmt(f),
            Self::CoordinateSpaceCapacity { requested, maximum } => write!(
                f,
                "portal projection needs {requested} coordinate-space slots; maximum is {maximum}"
            ),
            Self::ChainCapacity { requested, maximum } => write!(
                f,
                "portal projection produced {requested} chains; maximum is {maximum}"
            ),
            Self::SceneRowCount {
                kind,
                expected,
                actual,
            } => write!(
                f,
                "{kind} projection row list has {actual} entries; expected {expected}"
            ),
            Self::NonDenseSceneRow {
                kind,
                dense_index,
                entity_index,
            } => write!(
                f,
                "{kind} projection row {dense_index} is backed by SceneDB entity index {entity_index}"
            ),
        }
    }
}

impl std::error::Error for ProjectionError {}

impl From<ResolutionError> for ProjectionError {
    fn from(error: ResolutionError) -> Self {
        Self::Resolution(error)
    }
}

/// Dense, deterministic derived rows for one resolved portal frame.
#[derive(Clone, Debug, PartialEq)]
pub struct PortalProjectionFrame {
    /// Full coordinate-space table. Slot zero is always identity.
    pub coordinate_spaces: Vec<Mat4>,
    /// Dense view rows. Their indices are the IDs stored in chain rows.
    pub portal_views: Vec<PortalViewComponent>,
    /// Dense chain rows. Each `portals[]` entry indexes `portal_views`.
    pub portal_chains: Vec<PortalChainComponent>,
    /// Stable dense IDs for all resolved runtime occurrences.
    pub occurrence_indices: BTreeMap<RuntimePortalKey, u32>,
    /// Stable dense IDs for source/target projection rows.
    pub projection_indices: BTreeMap<PortalProjectionKey, u32>,
    /// All view rows generated by a runtime occurrence, useful when one
    /// source has multiple cross-sublevel target placements.
    pub view_indices_by_occurrence: BTreeMap<RuntimePortalKey, Vec<u32>>,
    /// Explicit active counts for the SceneDB-facing derived count row.
    pub counts: PortalProjectionCountsComponent,
}

impl PortalProjectionFrame {
    /// Coordinate spaces excluding slot zero, suitable for the existing
    /// `Renderer::set_coordinate_spaces` API, which reserves slot zero itself.
    pub fn renderer_coordinate_spaces(&self) -> &[Mat4] {
        &self.coordinate_spaces[1..]
    }

    pub fn portal_view_index(&self, key: &PortalProjectionKey) -> Option<u32> {
        self.projection_indices.get(key).copied()
    }

    /// Publish the derived rows through the existing SceneDB component seam.
    ///
    /// The portal GPU ABI is entity-indexed. Callers therefore reserve a
    /// contiguous projection entity range whose entity indices match the
    /// dense row IDs, and keep those entities separate from authored portal
    /// entities. This method intentionally does not spawn or repurpose those
    /// entities, which leaves lifetime/slot policy with the host application.
    pub fn publish_to_world(
        &self,
        world: &mut pulsar_scenedb::World,
        view_entities: &[Entity],
        chain_entities: &[Entity],
        counts_entity: Entity,
    ) -> Result<(), ProjectionError> {
        if view_entities.len() != self.portal_views.len() {
            return Err(ProjectionError::SceneRowCount {
                kind: "portal view",
                expected: self.portal_views.len(),
                actual: view_entities.len(),
            });
        }
        if chain_entities.len() != self.portal_chains.len() {
            return Err(ProjectionError::SceneRowCount {
                kind: "portal chain",
                expected: self.portal_chains.len(),
                actual: chain_entities.len(),
            });
        }
        for (dense_index, entity) in view_entities.iter().enumerate() {
            if entity.index() as usize != dense_index {
                return Err(ProjectionError::NonDenseSceneRow {
                    kind: "portal view",
                    dense_index,
                    entity_index: entity.index(),
                });
            }
        }
        for (dense_index, entity) in chain_entities.iter().enumerate() {
            if entity.index() as usize != dense_index {
                return Err(ProjectionError::NonDenseSceneRow {
                    kind: "portal chain",
                    dense_index,
                    entity_index: entity.index(),
                });
            }
        }
        for (entity, row) in view_entities.iter().zip(&self.portal_views) {
            world.insert(*entity, *row);
        }
        for (entity, row) in chain_entities.iter().zip(&self.portal_chains) {
            world.insert(*entity, *row);
        }
        world.insert(counts_entity, self.counts);
        Ok(())
    }
}

/// Converts resolver output into the existing SceneDB/GPU projection seam.
#[derive(Clone, Copy, Debug)]
pub struct PortalProjectionBridge {
    max_chain_depth: usize,
}

impl Default for PortalProjectionBridge {
    fn default() -> Self {
        Self {
            max_chain_depth: MAX_CHAIN_DEPTH,
        }
    }
}

impl PortalProjectionBridge {
    pub fn new(max_chain_depth: usize) -> Result<Self, ResolutionError> {
        if max_chain_depth == 0 || max_chain_depth > MAX_CHAIN_DEPTH {
            return Err(ResolutionError::InvalidRecursionDepth {
                requested: max_chain_depth,
                maximum: MAX_CHAIN_DEPTH,
            });
        }
        Ok(Self { max_chain_depth })
    }

    pub fn max_chain_depth(&self) -> usize {
        self.max_chain_depth
    }

    pub fn build(
        &self,
        resolver: &SubLevelResolver,
    ) -> Result<PortalProjectionFrame, ProjectionError> {
        let occurrences = resolver.resolve_portals()?;
        let mut all_occurrences = BTreeMap::<RuntimePortalKey, PortalOccurrence>::new();
        for occurrence in occurrences {
            all_occurrences.insert(RuntimePortalKey::from_occurrence(&occurrence), occurrence);
        }

        let mut projections = resolver.resolve_projections()?;
        let direct_projection_keys: BTreeSet<_> = projections
            .iter()
            .map(PortalProjectionKey::from_projection)
            .collect();
        let mut chains = resolver.resolve_chains(self.max_chain_depth)?;
        for projection in &projections {
            all_occurrences
                .entry(RuntimePortalKey::from_occurrence(&projection.source))
                .or_insert_with(|| projection.source.clone());
            all_occurrences
                .entry(RuntimePortalKey::from_occurrence(&projection.target))
                .or_insert_with(|| projection.target.clone());
        }
        for chain in &chains {
            for projection in &chain.projections {
                all_occurrences
                    .entry(RuntimePortalKey::from_occurrence(&projection.source))
                    .or_insert_with(|| projection.source.clone());
                all_occurrences
                    .entry(RuntimePortalKey::from_occurrence(&projection.target))
                    .or_insert_with(|| projection.target.clone());
            }
        }

        // Chains contain the same first-hop projection data as the direct
        // projection list, so make the row set the union of both sources and
        // sort it by stable runtime keys before assigning any GPU index.
        for chain in &chains {
            projections.extend(chain.projections.iter().cloned());
        }
        projections.sort_by_key(PortalProjectionKey::from_projection);
        projections.dedup_by(|left, right| {
            PortalProjectionKey::from_projection(left)
                == PortalProjectionKey::from_projection(right)
        });

        let mut projection_indices = BTreeMap::new();
        for (index, projection) in projections.iter().enumerate() {
            projection_indices.insert(
                PortalProjectionKey::from_projection(projection),
                index as u32,
            );
        }

        let mut coordinate_spaces = vec![Mat4::IDENTITY];
        let mut coordinate_space_indices = HashMap::<[u32; 16], u32>::new();
        coordinate_space_indices.insert(matrix_key(Mat4::IDENTITY), IDENTITY_COORDINATE_SPACE_SLOT);
        let mut portal_views = Vec::with_capacity(projections.len());
        let mut view_indices_by_occurrence = BTreeMap::<RuntimePortalKey, Vec<u32>>::new();
        for (index, projection) in projections.iter().enumerate() {
            let matrix = projection.target_to_source;
            let key = matrix_key(matrix);
            let coordinate_space = if let Some(&slot) = coordinate_space_indices.get(&key) {
                slot
            } else {
                if coordinate_spaces.len() == MAX_COORDINATE_SPACES {
                    return Err(ProjectionError::CoordinateSpaceCapacity {
                        requested: coordinate_spaces.len() + 1,
                        maximum: MAX_COORDINATE_SPACES,
                    });
                }
                let slot = coordinate_spaces.len() as u32;
                coordinate_spaces.push(matrix);
                coordinate_space_indices.insert(key, slot);
                slot
            };
            let mut view = projection.to_portal_view(coordinate_space);
            if !direct_projection_keys.contains(&PortalProjectionKey::from_projection(projection)) {
                view._pad = PortalViewComponent::FLAG_MASK_HIDDEN;
            }
            portal_views.push(view);
            view_indices_by_occurrence
                .entry(RuntimePortalKey::from_occurrence(&projection.source))
                .or_default()
                .push(index as u32);
        }

        chains.sort_by_key(chain_sort_key);
        chains.dedup_by(|left, right| chain_sort_key(left) == chain_sort_key(right));
        if chains.len() > MAX_PORTAL_CHAINS {
            return Err(ProjectionError::ChainCapacity {
                requested: chains.len(),
                maximum: MAX_PORTAL_CHAINS,
            });
        }
        let mut portal_chains = Vec::with_capacity(chains.len());
        for chain in chains {
            let mut portals = [0; MAX_CHAIN_DEPTH];
            for (index, projection) in chain.projections.iter().enumerate() {
                portals[index] = *projection_indices
                    .get(&PortalProjectionKey::from_projection(projection))
                    .expect("chain projection was included in projection rows");
            }
            portal_chains.push(PortalChainComponent {
                portals,
                depth: chain.projections.len() as u32,
            });
        }

        let occurrence_indices = all_occurrences
            .keys()
            .enumerate()
            .map(|(index, key)| (key.clone(), index as u32))
            .collect();
        let counts = PortalProjectionCountsComponent {
            portal_view_count: portal_views.len() as u32,
            portal_chain_count: portal_chains.len() as u32,
            coordinate_space_count: coordinate_spaces.len() as u32,
            _pad: 0,
        };

        Ok(PortalProjectionFrame {
            coordinate_spaces,
            portal_views,
            portal_chains,
            occurrence_indices,
            projection_indices,
            view_indices_by_occurrence,
            counts,
        })
    }
}

fn matrix_key(matrix: Mat4) -> [u32; 16] {
    matrix.to_cols_array().map(f32::to_bits)
}

fn chain_sort_key(chain: &ResolvedPortalChain) -> Vec<PortalProjectionKey> {
    chain
        .projections
        .iter()
        .map(PortalProjectionKey::from_projection)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::components::PortalComponent;
    use crate::resolver::SubLevelContents;
    use helio_pass_gbuffer::SubLevelActorComponent;

    #[test]
    fn dense_indexing_and_identity_slot_are_deterministic() {
        let mut world = pulsar_scenedb::World::new();
        let a_entity = world.spawn();
        let b_entity = world.spawn();
        let mut resolver = SubLevelResolver::new();
        resolver.insert_sublevel(
            0,
            SubLevelContents::new(
                [],
                [
                    (
                        a_entity,
                        PortalComponent::new(Some(b_entity), Mat4::IDENTITY, [1.0, 1.0]),
                    ),
                    (
                        b_entity,
                        PortalComponent::new(Some(a_entity), Mat4::IDENTITY, [1.0, 1.0]),
                    ),
                ],
            ),
        );
        let frame = PortalProjectionBridge::new(1)
            .unwrap()
            .build(&resolver)
            .unwrap();

        assert_eq!(frame.coordinate_spaces[0], Mat4::IDENTITY);
        // Even identity-authored portal poses get a distinct render-space
        // turn: the view map includes the standard 180-degree portal-plane
        // flip, so target contents land beyond the source opening.
        assert_eq!(frame.coordinate_spaces.len(), 2);
        assert_eq!(frame.portal_views.len(), 2);
        assert_eq!(frame.portal_chains.len(), 2);
        assert_eq!(frame.counts.portal_view_count, 2);
        assert_eq!(frame.counts.portal_chain_count, 2);
        assert!(frame.projection_indices.values().copied().eq(0..2));
    }

    #[test]
    fn same_sublevel_projection_uses_one_dense_pair() {
        let mut world = pulsar_scenedb::World::new();
        let a = world.spawn();
        let b = world.spawn();
        let mut resolver = SubLevelResolver::new();
        resolver.insert_sublevel(
            0,
            SubLevelContents::new(
                [],
                [
                    (
                        a,
                        PortalComponent::new(
                            Some(b),
                            Mat4::from_translation(glam::vec3(2.0, 0.0, 0.0)),
                            [1.0, 1.0],
                        ),
                    ),
                    (
                        b,
                        PortalComponent::new(
                            Some(a),
                            Mat4::from_translation(glam::vec3(-2.0, 0.0, 0.0)),
                            [1.0, 1.0],
                        ),
                    ),
                ],
            ),
        );
        let frame = PortalProjectionBridge::new(1)
            .unwrap()
            .build(&resolver)
            .unwrap();
        assert_eq!(frame.portal_views.len(), 2);
        assert_eq!(frame.coordinate_spaces.len(), 3);
    }

    #[test]
    fn nested_projection_rows_do_not_stamp_virtual_openings() {
        let mut world = pulsar_scenedb::World::new();
        let source = world.spawn();
        let target = world.spawn();
        let mut target_component = PortalComponent::new(
            Some(source),
            Mat4::from_translation(glam::vec3(-2.0, 0.0, 0.0)),
            [1.0, 1.0],
        );
        target_component.flags &= !PortalComponent::FLAG_ENABLED;

        let mut resolver = SubLevelResolver::new();
        resolver.insert_sublevel(
            0,
            SubLevelContents::new(
                [],
                [
                    (
                        source,
                        PortalComponent::new(
                            Some(target),
                            Mat4::from_translation(glam::vec3(2.0, 0.0, 0.0)),
                            [1.0, 1.0],
                        ),
                    ),
                    (target, target_component),
                ],
            ),
        );

        let frame = PortalProjectionBridge::new(3)
            .unwrap()
            .build(&resolver)
            .unwrap();
        assert!(frame
            .portal_views
            .iter()
            .any(|view| view._pad == PortalViewComponent::FLAG_MASK_HIDDEN));
        assert!(frame
            .portal_views
            .iter()
            .any(|view| view._pad == 0));
    }

    #[test]
    fn cross_sublevel_projection_has_a_canonical_target_occurrence() {
        let mut world = pulsar_scenedb::World::new();
        let source = world.spawn();
        let target = world.spawn();
        let mut resolver = SubLevelResolver::new();
        resolver.insert_sublevel(
            0,
            SubLevelContents::new(
                [],
                [(
                    source,
                    PortalComponent::new(Some(target), Mat4::IDENTITY, [1.0, 1.0]),
                )],
            ),
        );
        resolver.insert_sublevel(
            1,
            SubLevelContents::new(
                [],
                [(
                    target,
                    PortalComponent::new(
                        Some(source),
                        Mat4::from_translation(glam::vec3(3.0, 0.0, 0.0)),
                        [1.0, 1.0],
                    ),
                )],
            ),
        );
        let frame = PortalProjectionBridge::new(1)
            .unwrap()
            .build(&resolver)
            .unwrap();
        assert_eq!(frame.occurrence_indices.len(), 2);
        assert_eq!(frame.portal_views.len(), 1);
        assert_eq!(frame.counts.coordinate_space_count, 2);
    }

    #[test]
    fn nested_actor_projection_is_keyed_by_actor_path() {
        let mut world = pulsar_scenedb::World::new();
        let actor_entity = world.spawn();
        let a = world.spawn();
        let b = world.spawn();
        let mut resolver = SubLevelResolver::new();
        resolver.insert_sublevel(
            0,
            SubLevelContents::new(
                [(
                    actor_entity,
                    SubLevelActorComponent::new(
                        1,
                        Mat4::from_translation(glam::vec3(10.0, 0.0, 0.0)),
                    ),
                )],
                [],
            ),
        );
        resolver.insert_sublevel(
            1,
            SubLevelContents::new(
                [],
                [
                    (a, PortalComponent::new(Some(b), Mat4::IDENTITY, [1.0, 1.0])),
                    (
                        b,
                        PortalComponent::new(
                            Some(a),
                            Mat4::from_translation(glam::vec3(0.0, 0.0, 2.0)),
                            [1.0, 1.0],
                        ),
                    ),
                ],
            ),
        );
        let frame = PortalProjectionBridge::new(1)
            .unwrap()
            .build(&resolver)
            .unwrap();
        assert!(frame
            .occurrence_indices
            .keys()
            .any(|key| key.actor_path == vec![actor_entity]));
        assert_eq!(frame.portal_views.len(), 2);
    }

    #[test]
    fn chain_rows_reference_dense_view_rows() {
        let mut world = pulsar_scenedb::World::new();
        let a = world.spawn();
        let b = world.spawn();
        let mut resolver = SubLevelResolver::new();
        resolver.insert_sublevel(
            0,
            SubLevelContents::new(
                [],
                [
                    (a, PortalComponent::new(Some(b), Mat4::IDENTITY, [1.0, 1.0])),
                    (b, PortalComponent::new(Some(a), Mat4::IDENTITY, [1.0, 1.0])),
                ],
            ),
        );
        let frame = PortalProjectionBridge::new(2)
            .unwrap()
            .build(&resolver)
            .unwrap();
        assert_eq!(frame.portal_chains.len(), 4);
        assert!(frame.portal_chains.iter().all(|chain| {
            (chain.depth == 1 || chain.depth == 2)
                && chain
                    .portals
                    .iter()
                    .take(chain.depth as usize)
                    .all(|id| (*id as usize) < frame.portal_views.len())
        }));
    }
}
