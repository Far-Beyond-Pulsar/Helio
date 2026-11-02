//! Voxel terrain as a feature: the node that adds a host's chunk pass to
//! preset graphs.
//!
//! The terrain joins at [`Slot::Geometry`] between GI's probe update and its
//! mesh rays: it writes the G-buffer over the spine's and traces the probes'
//! rays. It offers its ground to foliage placement
//! ([`helio_foliage_api::OfferedGrounds`]).

use std::sync::Arc;

use helio_feature_api::{Feature, FeatureRegistrar, NodeDescriptor, ParamDesc, Slot};

use crate::chunks::pass::ChunkPass;

/// Makes a host's chunk pass: its frame source, residency configuration and
/// appearance. Called on every graph build.
pub type ChunkPassMaker = Arc<dyn Fn() -> ChunkPass + Send + Sync>;

/// The voxel terrain a graph draws (a renderer config's `features`): the
/// host's chunk passes. Without it the voxel node builds nothing.
#[derive(Clone, Default)]
pub struct VoxelTerrain {
    pub passes: Vec<ChunkPassMaker>,
}

impl VoxelTerrain {
    /// A terrain drawn by the passes `make` returns.
    pub fn new(make: impl Fn() -> ChunkPass + Send + Sync + 'static) -> Self {
        Self {
            passes: vec![Arc::new(make)],
        }
    }
}

/// Voxel worlds ([`crate`]).
pub struct VoxelFeature;

impl Feature for VoxelFeature {
    fn id(&self) -> &'static str {
        "helio.voxel"
    }

    fn register(&self, registrar: &mut FeatureRegistrar<'_>) {
        registrar.node(
            NodeDescriptor::new(
                "helio.pass.voxel_chunks",
                "Voxel: Chunk Terrain",
                "Passes: Geometry",
                "Voxel worlds from the host's frames: residency, generation, the G-buffer \
                 and the GI probes' rays. Builds nothing without a VoxelTerrain setting.",
                |build| {
                    let makers = build
                        .env
                        .setting::<VoxelTerrain>()
                        .map(|terrain| terrain.passes.clone())
                        .unwrap_or_default();
                    for make in makers {
                        let pass = make();
                        build
                            .env
                            .shared(helio_foliage_api::OfferedGrounds::default)
                            .0
                            .push(pass.foliage_ground());
                        build.graph.add_pass(Box::new(pass));
                    }
                    Ok(())
                },
            )
            .with_param(ParamDesc::boolean("enabled", "Enabled", true))
            .with_generated_pins()
            .in_slot(Slot::Geometry, 10),
        );
    }
}
