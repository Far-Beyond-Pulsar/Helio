//! The features Helio's presets are made of, and the renderer's own nodes.
//!
//! Every pass node is registered by the crate that owns the pass: the spine's
//! by `helio-spine`, each feature's by its crate (`helio_pass_ssr::
//! SsrFeature`, …). What is left here is what belongs to the renderer:
//!
//! - the debug-draw nodes, whose pass and state are the renderer's;
//! - a preparer that turns the renderer's config into the settings the
//!   features read ([`helio_spine::SpineSettings`], the reflections'
//!   switches, the perf overlay's mode),
//!   so no feature reads `RendererConfig`;
//! - [`default_features`], the feature list Helio's presets build from.

use std::sync::{Arc, Mutex};

use helio::{DebugDrawState, RendererConfig};
use helio_feature_api::{pass_node, Feature, FeatureRegistrar, FeatureSet};
use helio_frame_api::{FrameNode, Phase, Slot};

pub use helio_feature_api::{pass_kind as kind, SceneColorFormat};
pub use helio_frame_api::{host_kind, HOST_INPUTS};

/// Helio's features, in registration order: the spine, then every feature
/// that joins the frame's slots and phases, then the renderer's own nodes.
/// An engine adds its features to this set ([`crate::Preset::with_features`]):
/// the features of object kinds Helio does not know (meshes, a voxel world)
/// come from the plugin that owns the kind.
pub fn default_features() -> FeatureSet {
    FeatureSet::new()
        .with(helio_spine::SpineFeature)
        .with(helio_pass_ssr::SsrFeature)
        .with(helio_pass_planar_reflection::PlanarReflectionFeature)
        .with(helio_pass_hlfs::HlfsFeature)
        .with(helio_pass_billboard::BillboardFeature)
        .with(helio_pass_fxaa::FxaaFeature)
        .with(helio_pass_buffer_view::BufferViewFeature)
        .with(helio_pass_perf_overlay::PerfOverlayFeature)
        .with(helio_pass_simple_cube::SimpleCubeFeature)
        .with(RendererNodes)
}

/// The renderer's nodes and the config bridge.
struct RendererNodes;

impl Feature for RendererNodes {
    fn id(&self) -> &'static str {
        "helio.renderer"
    }

    fn register(&self, registrar: &mut FeatureRegistrar<'_>) {
        registrar.prepare(Arc::new(|env| {
            let Some(config) = env.setting::<RendererConfig>().cloned() else {
                return;
            };
            env.insert_setting(helio_spine::SpineSettings {
                background_color: config.background_color,
                debug_mode: config.debug_mode,
                environment_reflections: config.enable_environment_reflections,
            });
            env.insert_setting(helio_pass_ssr::SsrSettings {
                enabled: config.enable_ssr,
            });
            env.insert_setting(helio_pass_planar_reflection::PlanarReflectionSettings {
                enabled: config.enable_planar_reflections,
            });
            env.insert_setting(crate::convert_perf_mode(config.perf_overlay_mode));
        }));

        registrar.node(
            pass_node(
                "debug_draw_scene",
                "Overlay: Grid + Volume Bounds (Editor)",
                "Passes: Overlays",
                "Editor grid and scene volume bounds, depth-tested against the scene.",
                |build| {
                    let state = build.env.require::<Arc<Mutex<DebugDrawState>>>()?.clone();
                    build.graph.add_pass(Box::new(helio::DebugDrawPass::new(
                        build.env.device,
                        build.env.debug_camera_buffer,
                        build.env.scene_format(),
                        state,
                        true,
                        true,
                    )));
                    Ok(())
                },
            )
            .drawer(Phase::Overlay, 40),
        );

        registrar.node(
            pass_node(
                "debug_draw",
                "Output: Debug Lines",
                "Passes: Output",
                "User debug lines and triangles over the final image.",
                |build| {
                    let state = build.env.require::<Arc<Mutex<DebugDrawState>>>()?.clone();
                    build.graph.add_pass(Box::new(helio::DebugDrawPass::new(
                        build.env.device,
                        build.env.debug_camera_buffer,
                        build.env.frame.surface_format,
                        state,
                        false,
                        false,
                    )));
                    Ok(())
                },
            )
            .in_slot(Slot::Display, 30),
        );
    }
}
