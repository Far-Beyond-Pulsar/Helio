//! Helio's default pipeline as render-graph nodes.
//!
//! [`register_nodes`] adds one node per stage of the default deferred graph, so
//! a render-graph asset can reuse, reorder, replace or surround them.
//! [`default_graph_asset`] wires them as `build_default_graph_external_with_context`
//! does, which is the starting point for a new graph.
//!
//! Stages share the performance-overlay state and the HDR lighting
//! configuration through [`StageState`], created by the first stage built.

use std::sync::{Arc, Mutex};

use helio::RendererConfig;
use helio_graph_asset::{
    core_nodes, Connection, GraphAsset, NodeBuild, NodeDescriptor, NodeInstance, NodeRegistry,
    PinDesc, PinRef, PinType,
};
use helio_pass_perf_overlay::PerfOverlayShared;

use crate::{
    add_common_early_passes, add_geometry_passes, add_final_passes, add_late_passes,
    add_scene_linear_chain, declare_common_external_inputs, DecalPass, DeferredLightPass,
    LightCullPass, PerfOverlayAnalyzerPass, PerfOverlayCostAnalyzerPass, PlanarReflectionPass,
    PostProcessPass, SceneAa, SsrPass, TsrPass, FOGGED_HDR_FORMAT,
};

pub const SCENE_INPUTS: &str = "helio.scene_inputs";
pub const EARLY: &str = "helio.stage.early";
pub const GEOMETRY: &str = "helio.stage.geometry";
pub const LIGHTING: &str = "helio.stage.lighting";
pub const ATMOSPHERE_OVERLAYS: &str = "helio.stage.late";
pub const POST_PROCESS: &str = "helio.stage.post_process";
pub const FINAL: &str = "helio.stage.final";

/// Host-supplied resources the default stages read, as the Scene Input node
/// publishes them.
const COMMON_INPUTS: &[&str] =
    &["material_textures", "render_environment", "billboards", "corona_emitters"];

/// What the stages of one build share.
struct StageState {
    perf: Arc<Mutex<PerfOverlayShared>>,
    /// Lighting and everything drawn into it stay scene-linear FP16 until
    /// post-processing tone maps once.
    lighting_config: RendererConfig,
}

fn stage_pins() -> (Vec<PinDesc>, Vec<PinDesc>) {
    (vec![PinDesc::flow("in")], vec![PinDesc::flow("out")])
}

fn stage(
    kind: &str,
    title: &str,
    description: &str,
    build: impl Fn(&mut NodeBuild<'_, '_>) -> Result<(), String> + Send + Sync + 'static,
) -> NodeDescriptor {
    let (inputs, outputs) = stage_pins();
    NodeDescriptor::new(kind, title, "Helio Stages", description, build).with_pins(inputs, outputs)
}

/// The stage state, or an error if the stage that makes it has not run: the
/// later stages need what the early stage builds.
fn state<'a>(build: &'a mut NodeBuild<'_, '_>, stage: &str) -> Result<&'a mut StageState, String> {
    build
        .env
        .shared_existing::<StageState>()
        .ok_or_else(|| format!("{stage} needs the Early stage to come before it"))
}

pub fn register_nodes(registry: &mut NodeRegistry) {
    registry.register(
        NodeDescriptor::new(
            SCENE_INPUTS,
            "Helio Scene Inputs",
            "Helio Stages",
            "The resources Helio's passes expect the renderer to supply every frame.",
            |build| {
                declare_common_external_inputs(build.graph);
                Ok(())
            },
        )
        .with_pins(
            Vec::new(),
            COMMON_INPUTS.iter().map(|name| PinDesc::resource(PinType::SCENE, *name)).collect(),
        ),
    );

    registry.register(stage(
        EARLY,
        "Scene Prep + Shadows",
        "Scene preparation, shadow maps, Hi-Z, occlusion culling and light culling.",
        |build| {
            let (iw, ih) = build.env.internal_size();
            let lighting_config =
                RendererConfig { surface_format: FOGGED_HDR_FORMAT, ..build.env.config };
            let env = &*build.env;
            let perf = add_common_early_passes(
                build.graph,
                env.device,
                env.queue,
                env.camera_buffer,
                &lighting_config,
                env.cull_stats_buffer,
                iw,
                ih,
                env.scene_db.clone(),
            );
            build.graph.add_pass(Box::new(LightCullPass::new(env.device, iw, ih)));
            build.env.shared(|| StageState { perf, lighting_config });
            Ok(())
        },
    ));

    registry.register(stage(
        GEOMETRY,
        "Geometry (GBuffer)",
        "Opaque geometry into the GBuffer, including foliage and virtual geometry.",
        |build| {
            let perf = state(build, "Geometry")?.perf.clone();
            let env = &*build.env;
            add_geometry_passes(
                build.graph,
                env.device,
                env.camera_buffer,
                &env.config,
                &perf,
                env.scene_db.clone(),
            );
            Ok(())
        },
    ));

    registry.register(stage(
        LIGHTING,
        "Deferred Lighting",
        "Decals, reflections (when enabled in settings) and deferred lighting of the GBuffer.",
        |build| {
            let (perf, lighting_config) = {
                let state = state(build, "Lighting")?;
                (state.perf.clone(), state.lighting_config)
            };
            let (iw, ih) = build.env.internal_size();
            let env = &*build.env;
            let (device, queue, camera, config) =
                (env.device, env.queue, env.camera_buffer, env.config);
            let graph = &mut *build.graph;
            graph.add_pass(Box::new(DecalPass::new(device, queue, camera, iw, ih)));
            if config.enable_ssr && helio_core::REFLECTIONS_SUPPORTED {
                graph.add_pass(Box::new(SsrPass::new(device, queue, camera, iw, ih)));
            }
            if config.enable_planar_reflections && helio_core::REFLECTIONS_SUPPORTED {
                graph.add_pass(Box::new(PlanarReflectionPass::new(
                    device,
                    camera,
                    config.surface_format,
                )));
            }
            let mut deferred =
                DeferredLightPass::new(device, queue, camera, lighting_config.surface_format);
            deferred.set_shadow_quality(config.shadow_quality, queue);
            deferred.debug_mode = config.debug_mode;
            deferred.set_env_reflections(config.enable_environment_reflections);
            graph.add_pass(Box::new(deferred));
            graph.add_pass(Box::new(PerfOverlayCostAnalyzerPass::new(perf.clone())));
            graph.add_pass(Box::new(PerfOverlayAnalyzerPass::new(perf)));
            Ok(())
        },
    ));

    registry.register(stage(
        ATMOSPHERE_OVERLAYS,
        "Atmosphere + Overlays",
        "Sky composite, billboards, coronas, water and the editor's debug drawing.",
        |build| {
            let (perf, lighting_config) = {
                let state = state(build, "Atmosphere + Overlays")?;
                (state.perf.clone(), state.lighting_config)
            };
            let (iw, ih) = build.env.internal_size();
            let env = &*build.env;
            add_late_passes(
                build.graph,
                env.device,
                env.queue,
                env.camera_buffer,
                &lighting_config,
                &perf,
                env.debug_state.clone(),
                env.debug_camera_buffer,
                iw,
                ih,
                env.scene_db.clone(),
            );
            Ok(())
        },
    ));

    registry.register(stage(
        POST_PROCESS,
        "Fog, Anti-Aliasing + Post Process",
        "Volumetric fog, transparency, TSR or FXAA, lens, tone mapping and colour grading.",
        |build| {
            let (iw, ih) = build.env.internal_size();
            let env = &*build.env;
            let (device, queue, config) = (env.device, env.queue, env.config);
            let aa = match config.tsr_quality {
                Some(quality) => SceneAa::Tsr(TsrPass::new(
                    device,
                    iw,
                    ih,
                    config.width,
                    config.height,
                    config.surface_format,
                    quality,
                )),
                None => SceneAa::Fxaa,
            };
            let transparent =
                helio_pass_transparent::TransparentPass::new(device, crate::FOGGED_HDR_FORMAT);
            let resolved = add_scene_linear_chain(
                build.graph,
                device,
                FOGGED_HDR_FORMAT,
                Some(transparent),
                aa,
                iw,
                ih,
                None,
            );
            build.graph.add_pass(Box::new(
                PostProcessPass::new_with_user_effects(
                    device,
                    queue,
                    config.width,
                    config.height,
                    config.surface_format,
                    None,
                )
                .with_color_input(resolved),
            ));
            Ok(())
        },
    ));

    registry.register(stage(
        FINAL,
        "Final Overlays",
        "Buffer views, sprites, the performance overlay and debug lines over the final image.",
        |build| {
            let perf = state(build, "Final Overlays")?.perf.clone();
            let env = &*build.env;
            add_final_passes(
                build.graph,
                env.device,
                env.queue,
                &env.config,
                &perf,
                env.debug_state.clone(),
                env.debug_camera_buffer,
                None,
            );
            Ok(())
        },
    ));
}

/// A registry with the core nodes and Helio's stages.
pub fn node_registry() -> NodeRegistry {
    let mut registry = NodeRegistry::with_core();
    register_nodes(&mut registry);
    registry
}

/// The default deferred pipeline as a graph: scene inputs, then each stage in
/// order, then the output.
pub fn default_graph_asset() -> GraphAsset {
    const STAGES: &[&str] = &[EARLY, GEOMETRY, LIGHTING, ATMOSPHERE_OVERLAYS, POST_PROCESS, FINAL];
    let mut asset = GraphAsset { name: "Deferred".into(), ..Default::default() };
    let mut node = |id: &str, kind: &str, x: f32| {
        asset.nodes.push(NodeInstance {
            id: id.into(),
            kind: kind.into(),
            params: Default::default(),
            position: [x, 0.0],
        });
    };
    node("inputs", SCENE_INPUTS, 0.0);
    let ids = ["early", "geometry", "lighting", "late", "post", "final"];
    for (i, (id, kind)) in ids.iter().zip(STAGES).enumerate() {
        node(id, kind, 320.0 * (i as f32 + 1.0));
    }
    node("output", core_nodes::OUTPUT, 320.0 * (STAGES.len() as f32 + 1.0));

    let mut previous = "early";
    for id in &ids[1..] {
        asset.connections.push(Connection {
            from: PinRef::new(previous, "out"),
            to: PinRef::new(*id, "in"),
        });
        previous = id;
    }
    asset.connections.push(Connection {
        from: PinRef::new("final", "out"),
        to: PinRef::new("output", "in"),
    });
    asset
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_graph_asset_validates_and_orders_the_stages() {
        let registry = node_registry();
        let asset = default_graph_asset();
        let validation = helio_graph_asset::validate(&asset, &registry);
        assert!(!validation.has_errors(), "{:?}", validation.diagnostics);
        let plan = validation.plan.expect("valid");
        let kinds: Vec<_> = plan.order.iter().map(|i| asset.nodes[*i].id.as_str()).collect();
        let position = |id: &str| kinds.iter().position(|k| *k == id).unwrap();
        assert!(position("early") < position("geometry"));
        assert!(position("geometry") < position("lighting"));
        assert!(position("post") < position("final"));
        assert_eq!(kinds.last(), Some(&"output"));
    }
}
