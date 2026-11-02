//! Voxel terrain as a feature: the node that adds a host's chunk pass to
//! preset graphs.
//!
//! The terrain is a drawer of the G-buffer phase, after the other drawers:
//! it writes the G-buffer over theirs and traces GI's probe rays (made before
//! the phase). It offers its ground to foliage placement
//! ([`helio_foliage_api::OfferedGrounds`]).

use std::sync::Arc;

use helio_feature_api::{Feature, FeatureRegistrar, NodeDescriptor, ParamDesc};
use helio_frame_api::{FrameNode, Phase};

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
        // With terrain to draw, it traces GI's probes every frame and its sun
        // samples rotate over blocks of pixels for TSR to integrate.
        registrar.prepare(std::sync::Arc::new(|env| {
            let drawn = env
                .setting::<VoxelTerrain>()
                .is_some_and(|terrain| !terrain.passes.is_empty());
            if drawn {
                env.insert_setting(helio_gi_api::TerrainTracer);
                env.insert_setting(helio_pass_tsr::StochasticSamples);
            }
        }));
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
            .drawer(Phase::GBuffer, 40),
        );
    }
}
