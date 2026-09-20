//! CPU-side resolution of indexed sublevels and peer-linked portals.
//!
//! This module is the authored-data boundary for the sublevel/portal system.
//! It consumes `SubLevelActorComponent` and the authored peer-based
//! `PortalComponent`, then produces resolved contexts and portal mappings for
//! the existing coordinate-space, portal-view, and portal-chain projection
//! code. It intentionally does not write SceneDB rows or GPU buffers.

use std::collections::{BTreeMap, HashMap};
use std::fmt;

use glam::Mat4;
use pulsar_scenedb::Entity;

use helio_pass_gbuffer::{SubLevelActorComponent, SubLevelIndex, DEFAULT_SUBLEVEL_INDEX};

use crate::components::{PortalComponent, PortalViewComponent, NO_PORTAL_PEER};
use crate::portal_math::portal_view_map;
use crate::MAX_CHAIN_DEPTH;

/// A component row together with its owning SceneDB entity.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SubLevelActorRecord {
    /// Entity carrying the actor component.
    pub entity: Entity,
    /// Authored actor placement.
    pub component: SubLevelActorComponent,
}

/// A component row together with its owning SceneDB entity.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PortalRecord {
    /// Entity carrying the portal component.
    pub entity: Entity,
    /// Authored portal relationship and local aperture.
    pub component: PortalComponent,
}

/// Authored contents associated with one indexed sublevel.
///
/// The index itself is held by [`SubLevelResolver`]. A sublevel is content,
/// not an ECS component or a placement entity; actor components inside it are
/// what create nested runtime contexts.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct SubLevelContents {
    /// Actors that place other indexed sublevels inside this sublevel.
    pub actors: Vec<SubLevelActorRecord>,
    /// Portals authored in this sublevel.
    pub portals: Vec<PortalRecord>,
}

impl SubLevelContents {
    /// Build contents from the corresponding SceneDB rows.
    pub fn new(
        actors: impl IntoIterator<Item = (Entity, SubLevelActorComponent)>,
        portals: impl IntoIterator<Item = (Entity, PortalComponent)>,
    ) -> Self {
        Self {
            actors: actors
                .into_iter()
                .map(|(entity, component)| SubLevelActorRecord { entity, component })
                .collect(),
            portals: portals
                .into_iter()
                .map(|(entity, component)| PortalRecord { entity, component })
                .collect(),
        }
    }
}

/// A resolved placement of one indexed sublevel.
#[derive(Clone, Debug, PartialEq)]
pub struct SubLevelRuntimeContext {
    /// Indexed content space being placed.
    pub sublevel_index: SubLevelIndex,
    /// Actor entities from the root to this placement, in traversal order.
    pub actor_path: Vec<Entity>,
    /// Transform from this sublevel's authored coordinates into the root
    /// level's coordinates.
    pub transform: Mat4,
}

/// A portal occurrence in one resolved runtime context.
#[derive(Clone, Debug, PartialEq)]
pub struct PortalOccurrence {
    /// Authored portal entity.
    pub entity: Entity,
    /// The resolved sublevel/actor context containing the portal.
    pub context: SubLevelRuntimeContext,
    /// Portal-local to root-level transform for this occurrence.
    pub transform: Mat4,
    /// Portal aperture half extents in local X/Y.
    pub half_extent: [f32; 2],
    /// Authored peer entity.
    pub peer: Entity,
}

/// A resolved mapping that displays the target portal's context through the
/// source portal.
#[derive(Clone, Debug, PartialEq)]
pub struct PortalProjection {
    /// Portal visible to the current view.
    pub source: PortalOccurrence,
    /// Peer portal whose context supplies the displayed contents.
    pub target: PortalOccurrence,
    /// Maps target-context coordinates into the source portal's render space.
    /// This is the standard target-content-to-source-opening view map, which
    /// includes the portal-local 180-degree turn needed to place the target
    /// level beyond the source opening. It is intentionally distinct from
    /// the teleport map used when an actor crosses a portal.
    pub target_to_source: Mat4,
}

impl PortalProjection {
    /// Convert the derived projection into the existing GPU portal-view row.
    ///
    /// The coordinate-space slot is assigned by the later projection bridge;
    /// this method deliberately does not allocate or mutate one.
    pub fn to_portal_view(&self, coordinate_space: u32) -> PortalViewComponent {
        PortalViewComponent {
            transform: self.source.transform.to_cols_array(),
            inverse_transform: self.source.transform.inverse().to_cols_array(),
            half_extent: self.source.half_extent,
            coordinate_space,
            _pad: 0,
        }
    }
}

/// A bounded recursive chain of portal projections.
#[derive(Clone, Debug, PartialEq)]
pub struct ResolvedPortalChain {
    /// Portal entities in outermost-to-deepest order.
    pub portals: Vec<Entity>,
    /// One target-to-source mapping for each portal link.
    pub projections: Vec<PortalProjection>,
    /// True when the authored peer graph continues beyond the requested
    /// depth and was intentionally truncated.
    pub truncated: bool,
}

/// Errors found while resolving authored sublevel and portal relationships.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ResolutionError {
    /// An actor references a sublevel with no registered content.
    MissingSubLevel {
        /// Referenced index.
        index: SubLevelIndex,
        /// Actor that made the reference.
        actor: Entity,
    },
    /// An actor path revisited an index, which would recurse forever.
    ActorCycle {
        /// Repeated index.
        repeated: SubLevelIndex,
        /// Index path at the point of detection.
        path: Vec<SubLevelIndex>,
    },
    /// An enabled portal has no peer handle.
    MissingPortalPeer {
        /// Portal without a peer.
        source: Entity,
    },
    /// A peer handle does not identify an indexed portal.
    UnknownPortalPeer {
        /// Source portal.
        source: Entity,
        /// Missing peer.
        peer: Entity,
    },
    /// Portal recursion must fit the fixed GPU chain ABI.
    InvalidRecursionDepth {
        /// Requested depth.
        requested: usize,
        /// Maximum supported depth.
        maximum: usize,
    },
}

impl fmt::Display for ResolutionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MissingSubLevel { index, actor } => {
                write!(f, "actor {actor:?} references missing sublevel {index}")
            }
            Self::ActorCycle { repeated, path } => {
                write!(f, "sublevel actor cycle at {repeated}: {path:?}")
            }
            Self::MissingPortalPeer { source } => {
                write!(f, "portal {source:?} has no peer")
            }
            Self::UnknownPortalPeer { source, peer } => {
                write!(f, "portal {source:?} references unknown peer {peer:?}")
            }
            Self::InvalidRecursionDepth { requested, maximum } => {
                write!(
                    f,
                    "portal recursion depth {requested} exceeds maximum {maximum}"
                )
            }
        }
    }
}

impl std::error::Error for ResolutionError {}

/// Resolves indexed sublevel content into runtime contexts and peer-linked
/// portal projections.
#[derive(Clone, Debug)]
pub struct SubLevelResolver {
    contents: BTreeMap<SubLevelIndex, SubLevelContents>,
}

impl Default for SubLevelResolver {
    fn default() -> Self {
        Self::new()
    }
}

impl SubLevelResolver {
    /// Create a resolver with the default level (sublevel zero) present as an
    /// identity content space.
    pub fn new() -> Self {
        let mut contents = BTreeMap::new();
        contents.insert(DEFAULT_SUBLEVEL_INDEX, SubLevelContents::default());
        Self { contents }
    }

    /// Replace or insert the authored contents for an indexed sublevel.
    pub fn insert_sublevel(&mut self, index: SubLevelIndex, contents: SubLevelContents) {
        self.contents.insert(index, contents);
    }

    /// Return authored contents for an index.
    pub fn sublevel(&self, index: SubLevelIndex) -> Option<&SubLevelContents> {
        self.contents.get(&index)
    }

    /// Resolve the root and every reachable direct/nested actor placement.
    pub fn resolve_contexts(&self) -> Result<Vec<SubLevelRuntimeContext>, ResolutionError> {
        let mut resolved = Vec::new();
        let root = SubLevelRuntimeContext {
            sublevel_index: DEFAULT_SUBLEVEL_INDEX,
            actor_path: Vec::new(),
            transform: Mat4::IDENTITY,
        };
        resolved.push(root.clone());
        self.walk_contexts(&root, &mut vec![DEFAULT_SUBLEVEL_INDEX], &mut resolved)?;
        Ok(resolved)
    }

    /// Resolve all portal occurrences in directly placed contexts.
    pub fn resolve_portals(&self) -> Result<Vec<PortalOccurrence>, ResolutionError> {
        let contexts = self.resolve_contexts()?;
        let mut occurrences = Vec::new();
        for context in contexts {
            let Some(contents) = self.contents.get(&context.sublevel_index) else {
                return Err(ResolutionError::MissingSubLevel {
                    index: context.sublevel_index,
                    actor: context
                        .actor_path
                        .last()
                        .copied()
                        .unwrap_or(Entity::DANGLING),
                });
            };
            for record in &contents.portals {
                if !record.component.is_enabled() {
                    continue;
                }
                let Some(peer) = record.component.peer_entity() else {
                    return Err(ResolutionError::MissingPortalPeer {
                        source: record.entity,
                    });
                };
                occurrences.push(PortalOccurrence {
                    entity: record.entity,
                    context: context.clone(),
                    transform: context.transform * record.component.transform(),
                    half_extent: record.component.half_extent,
                    peer,
                });
            }
        }
        Ok(occurrences)
    }

    /// Resolve the first-hop source/peer mappings for all directly visible
    /// portal occurrences. A peer in an uninstanced indexed sublevel receives
    /// a canonical identity context so portals do not require a cosmetic
    /// `SubLevelActor` merely to establish a cross-sublevel relationship.
    pub fn resolve_projections(&self) -> Result<Vec<PortalProjection>, ResolutionError> {
        let occurrences = self.resolve_portals()?;
        let index = self.portal_index();
        let mut projections = Vec::new();
        for source in occurrences.iter().cloned() {
            let targets = self.resolve_peer_targets(&source, &occurrences, &index)?;
            projections.extend(targets.into_iter().map(|target| PortalProjection {
                target_to_source: portal_view_map(source.transform, target.transform),
                source: source.clone(),
                target,
            }));
        }
        Ok(projections)
    }

    /// Resolve bounded recursive portal chains. Loops are valid portal
    /// topology; the fixed depth is what makes traversal finite and matches
    /// the existing GPU chain ABI. Every prefix is returned: a recursive
    /// portal image is built from the one-hop, two-hop, ... views, not only
    /// from the deepest walk. This is what makes an ordinary peer graph read
    /// as a recursively repeating space without a demo-authored continuation
    /// object.
    pub fn resolve_chains(
        &self,
        max_depth: usize,
    ) -> Result<Vec<ResolvedPortalChain>, ResolutionError> {
        if max_depth == 0 || max_depth > MAX_CHAIN_DEPTH {
            return Err(ResolutionError::InvalidRecursionDepth {
                requested: max_depth,
                maximum: MAX_CHAIN_DEPTH,
            });
        }

        let occurrences = self.resolve_portals()?;
        let index = self.portal_index();
        let mut chains = Vec::new();
        for source in occurrences.iter().cloned() {
            self.extend_chain(
                source,
                &occurrences,
                &index,
                max_depth,
                Vec::new(),
                Vec::new(),
                &mut chains,
            )?;
        }
        Ok(chains)
    }

    fn walk_contexts(
        &self,
        context: &SubLevelRuntimeContext,
        ancestry: &mut Vec<SubLevelIndex>,
        resolved: &mut Vec<SubLevelRuntimeContext>,
    ) -> Result<(), ResolutionError> {
        let contents = self
            .contents
            .get(&context.sublevel_index)
            .expect("resolved context must have registered contents");
        for actor in &contents.actors {
            if !actor.component.is_enabled() {
                continue;
            }
            let child = actor.component.sublevel_index;
            if ancestry.contains(&child) {
                return Err(ResolutionError::ActorCycle {
                    repeated: child,
                    path: ancestry.clone(),
                });
            }
            if !self.contents.contains_key(&child) {
                return Err(ResolutionError::MissingSubLevel {
                    index: child,
                    actor: actor.entity,
                });
            }
            let mut actor_path = context.actor_path.clone();
            actor_path.push(actor.entity);
            let child_context = SubLevelRuntimeContext {
                sublevel_index: child,
                actor_path,
                transform: context.transform * actor.component.transform(),
            };
            resolved.push(child_context.clone());
            ancestry.push(child);
            self.walk_contexts(&child_context, ancestry, resolved)?;
            ancestry.pop();
        }
        Ok(())
    }

    fn portal_index(&self) -> HashMap<Entity, (SubLevelIndex, PortalRecord)> {
        self.contents
            .iter()
            .flat_map(|(&index, contents)| {
                contents
                    .portals
                    .iter()
                    .copied()
                    .map(move |record| (record.entity, (index, record)))
            })
            .collect()
    }

    fn resolve_peer_targets(
        &self,
        source: &PortalOccurrence,
        occurrences: &[PortalOccurrence],
        portal_index: &HashMap<Entity, (SubLevelIndex, PortalRecord)>,
    ) -> Result<Vec<PortalOccurrence>, ResolutionError> {
        let Some(&(target_index, target_record)) = portal_index.get(&source.peer) else {
            return Err(ResolutionError::UnknownPortalPeer {
                source: source.entity,
                peer: source.peer,
            });
        };

        // A same-sublevel peer belongs to the exact same actor placement as
        // the source. This is what makes portals in one indexed level pair
        // correctly even when that level is nested/instanced.
        if target_index == source.context.sublevel_index {
            return Ok(vec![
                self.occurrence_in_context(target_record, source.context.clone())
            ]);
        }

        // Cross-sublevel peers may have multiple actor placements. Preserve
        // each reachable occurrence; if none is placed, use the target
        // sublevel's canonical local context.
        let placed: Vec<_> = occurrences
            .iter()
            .filter(|occurrence| occurrence.entity == source.peer)
            .cloned()
            .collect();
        if !placed.is_empty() {
            return Ok(placed);
        }

        Ok(vec![self.occurrence_in_context(
            target_record,
            SubLevelRuntimeContext {
                sublevel_index: target_index,
                actor_path: Vec::new(),
                transform: Mat4::IDENTITY,
            },
        )])
    }

    fn occurrence_in_context(
        &self,
        record: PortalRecord,
        context: SubLevelRuntimeContext,
    ) -> PortalOccurrence {
        let peer = record
            .component
            .peer_entity()
            .unwrap_or_else(|| Entity::from_bits(NO_PORTAL_PEER));
        PortalOccurrence {
            entity: record.entity,
            transform: context.transform * record.component.transform(),
            half_extent: record.component.half_extent,
            peer,
            context,
        }
    }

    fn extend_chain(
        &self,
        source: PortalOccurrence,
        occurrences: &[PortalOccurrence],
        portal_index: &HashMap<Entity, (SubLevelIndex, PortalRecord)>,
        max_depth: usize,
        mut portals: Vec<Entity>,
        mut projections: Vec<PortalProjection>,
        output: &mut Vec<ResolvedPortalChain>,
    ) -> Result<(), ResolutionError> {
        portals.push(source.entity);
        let targets = self.resolve_peer_targets(&source, occurrences, portal_index)?;
        for target in targets {
            projections.push(PortalProjection {
                target_to_source: portal_view_map(source.transform, target.transform),
                source: source.clone(),
                target: target.clone(),
            });

            // Publish this prefix before descending. The renderer needs all
            // visible recursion depths so the shallower image can contain the
            // deeper image instead of being replaced by one over-constrained
            // max-depth chain.
            output.push(ResolvedPortalChain {
                portals: portals.clone(),
                projections: projections.clone(),
                truncated: portals.len() == max_depth,
            });

            if portals.len() < max_depth {
                self.extend_chain(
                    target,
                    occurrences,
                    portal_index,
                    max_depth,
                    portals.clone(),
                    projections.clone(),
                    output,
                )?;
            }
            projections.pop();
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn actor(
        world: &mut pulsar_scenedb::World,
        index: SubLevelIndex,
        transform: Mat4,
    ) -> SubLevelActorRecord {
        let entity = world.spawn();
        SubLevelActorRecord {
            entity,
            component: SubLevelActorComponent::new(index, transform),
        }
    }

    #[test]
    fn index_zero_is_identity_context() {
        let resolver = SubLevelResolver::new();
        let contexts = resolver.resolve_contexts().unwrap();
        assert_eq!(
            contexts,
            vec![SubLevelRuntimeContext {
                sublevel_index: DEFAULT_SUBLEVEL_INDEX,
                actor_path: Vec::new(),
                transform: Mat4::IDENTITY,
            }]
        );
    }

    #[test]
    fn nested_actor_transforms_compose_in_path_order() {
        let mut world = pulsar_scenedb::World::new();
        let outer = actor(
            &mut world,
            1,
            Mat4::from_translation(glam::vec3(10.0, 0.0, 0.0)),
        );
        let inner = actor(
            &mut world,
            2,
            Mat4::from_translation(glam::vec3(0.0, 5.0, 0.0)),
        );

        let mut resolver = SubLevelResolver::new();
        resolver.insert_sublevel(
            0,
            SubLevelContents {
                actors: vec![outer],
                portals: Vec::new(),
            },
        );
        resolver.insert_sublevel(
            1,
            SubLevelContents {
                actors: vec![inner],
                portals: Vec::new(),
            },
        );
        resolver.insert_sublevel(2, SubLevelContents::default());

        let contexts = resolver.resolve_contexts().unwrap();
        let nested = contexts
            .iter()
            .find(|context| context.sublevel_index == 2)
            .unwrap();
        assert_eq!(nested.actor_path, vec![outer.entity, inner.entity]);
        assert_eq!(
            nested.transform.transform_point3(glam::Vec3::ZERO),
            glam::vec3(10.0, 5.0, 0.0)
        );
    }

    #[test]
    fn same_sublevel_portals_pair_in_their_runtime_context() {
        let mut world = pulsar_scenedb::World::new();
        let a_entity = world.spawn();
        let b_entity = world.spawn();
        let a = PortalRecord {
            entity: a_entity,
            component: PortalComponent::new(
                Some(b_entity),
                Mat4::from_translation(glam::vec3(0.0, 0.0, 0.0)),
                [1.0, 2.0],
            ),
        };
        let b = PortalRecord {
            entity: b_entity,
            component: PortalComponent::new(
                Some(a_entity),
                Mat4::from_translation(glam::vec3(10.0, 0.0, 0.0)),
                [1.0, 2.0],
            ),
        };
        let mut resolver = SubLevelResolver::new();
        resolver.insert_sublevel(
            0,
            SubLevelContents {
                actors: Vec::new(),
                portals: vec![a, b],
            },
        );

        let projection = resolver
            .resolve_projections()
            .unwrap()
            .into_iter()
            .find(|projection| projection.source.entity == a_entity)
            .unwrap();
        assert_eq!(projection.target.entity, b_entity);
        assert_eq!(projection.target.context.sublevel_index, 0);
        assert_eq!(projection.target.context.actor_path, Vec::<Entity>::new());
        assert_eq!(
            projection
                .target_to_source
                .transform_point3(glam::vec3(10.0, 0.0, 0.0)),
            glam::Vec3::ZERO
        );
    }

    #[test]
    fn cross_sublevel_portal_uses_target_canonical_context() {
        let mut world = pulsar_scenedb::World::new();
        let source_entity = world.spawn();
        let target_entity = world.spawn();
        let source = PortalRecord {
            entity: source_entity,
            component: PortalComponent::new(Some(target_entity), Mat4::IDENTITY, [1.0, 1.0]),
        };
        let target = PortalRecord {
            entity: target_entity,
            component: PortalComponent::new(
                Some(source_entity),
                Mat4::from_translation(glam::vec3(20.0, 0.0, 0.0)),
                [1.0, 1.0],
            ),
        };
        let mut resolver = SubLevelResolver::new();
        resolver.insert_sublevel(
            0,
            SubLevelContents {
                actors: Vec::new(),
                portals: vec![source],
            },
        );
        resolver.insert_sublevel(
            7,
            SubLevelContents {
                actors: Vec::new(),
                portals: vec![target],
            },
        );

        let projection = resolver
            .resolve_projections()
            .unwrap()
            .into_iter()
            .find(|projection| projection.source.entity == source_entity)
            .unwrap();
        assert_eq!(projection.target.context.sublevel_index, 7);
        assert!(projection.target.context.actor_path.is_empty());
        assert_eq!(
            projection
                .target_to_source
                .transform_point3(glam::Vec3::new(20.0, 0.0, 0.0)),
            glam::Vec3::ZERO
        );
    }

    #[test]
    fn actor_cycles_and_missing_peers_are_rejected() {
        let mut world = pulsar_scenedb::World::new();
        let cycle_actor = actor(&mut world, 1, Mat4::IDENTITY);
        let back_actor = actor(&mut world, 0, Mat4::IDENTITY);
        let mut resolver = SubLevelResolver::new();
        resolver.insert_sublevel(
            0,
            SubLevelContents {
                actors: vec![cycle_actor],
                portals: Vec::new(),
            },
        );
        resolver.insert_sublevel(
            1,
            SubLevelContents {
                actors: vec![back_actor],
                portals: Vec::new(),
            },
        );
        assert!(matches!(
            resolver.resolve_contexts(),
            Err(ResolutionError::ActorCycle { .. })
        ));

        let missing_peer = world.spawn();
        let mut peer_resolver = SubLevelResolver::new();
        peer_resolver.insert_sublevel(
            0,
            SubLevelContents {
                actors: Vec::new(),
                portals: vec![PortalRecord {
                    entity: missing_peer,
                    component: PortalComponent::new(None, Mat4::IDENTITY, [1.0, 1.0]),
                }],
            },
        );
        assert!(matches!(
            peer_resolver.resolve_projections(),
            Err(ResolutionError::MissingPortalPeer { source }) if source == missing_peer
        ));
    }

    #[test]
    fn recursion_depth_is_bounded_by_gpu_chain_abi() {
        let resolver = SubLevelResolver::new();
        assert!(matches!(
            resolver.resolve_chains(MAX_CHAIN_DEPTH + 1),
            Err(ResolutionError::InvalidRecursionDepth { .. })
        ));
    }
}
