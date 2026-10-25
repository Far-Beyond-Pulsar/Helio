//! Compiles a graph asset into a Helio [`RenderGraph`], and wraps it as an
//! [`IntoGraph`] pipeline.

use std::ops::Range;
use std::sync::Arc;

use helio::{GraphRebuilder, IntoGraph, PassBuildContext};
use helio_core::RenderGraph;

use crate::asset::{GraphAsset, NodeInstance};
use crate::registry::{BuildEnv, NodeBuild, NodeRegistry};
use crate::validate::{validate, Diagnostic, Severity};

/// Called after each node is built with the node, the graph so far and the
/// range of pass indices the node added.
pub(crate) type NodeObserver<'a> = &'a mut dyn FnMut(&NodeInstance, &RenderGraph, Range<usize>);

/// Builds `asset`'s graph. Fails with the validation errors, or with the
/// message of the first node that cannot build.
pub fn compile(
    asset: &GraphAsset,
    registry: &NodeRegistry,
    env: &mut BuildEnv<'_>,
) -> Result<RenderGraph, Vec<Diagnostic>> {
    build_graph(asset, registry, env, true, &mut |_, _, _| {})
}

/// The shared build. `lock` finishes the graph (what a renderer needs); a
/// probe leaves it open, since it only wants to read what the passes declare.
pub(crate) fn build_graph(
    asset: &GraphAsset,
    registry: &NodeRegistry,
    env: &mut BuildEnv<'_>,
    lock: bool,
    observe: NodeObserver<'_>,
) -> Result<RenderGraph, Vec<Diagnostic>> {
    let validation = validate(asset, registry);
    for diagnostic in &validation.diagnostics {
        if diagnostic.severity == Severity::Warning {
            log::debug!("render graph `{}`: {diagnostic}", asset.name);
        }
    }
    let Some(plan) = validation.plan else {
        return Err(validation.diagnostics);
    };
    // The plan is about the flattened graph, without its buffer nodes.
    let flat = validation.flat.unwrap_or_else(|| asset.clone());

    let mut graph = if env.owns_device {
        RenderGraph::new(env.device, env.queue)
    } else {
        RenderGraph::new_with_external_device(env.device, env.queue)
    };
    graph.with_xr_mode(env.config.enable_xr);
    if let Some(path) = std::env::var_os("PULSAR_PIPELINE_CACHE") {
        graph.enable_pipeline_cache_persistence(path);
    }

    for &index in &plan.order {
        let node = &flat.nodes[index];
        let descriptor = registry
            .get(&node.kind)
            .expect("validated node kinds are registered");
        let params = descriptor.effective_params(&node.params);
        let first_pass = graph.pass_count();
        {
            let mut build = NodeBuild {
                graph: &mut graph,
                env: &mut *env,
                params: &params,
                node_id: &node.id,
            };
            descriptor.run(&mut build).map_err(|message| {
                vec![Diagnostic {
                    severity: Severity::Error,
                    node: Some(node.id.clone()),
                    message,
                }]
            })?;
            // Performance markers belong to the compiler, not to the graph: a node
            // that ends a phase asks for one, the registry says what it is.
            if descriptor.marker_after {
                if let Some(marker) = registry.marker() {
                    marker(&mut build, &node.kind);
                }
            }
        }
        observe(node, &graph, first_pass..graph.pass_count());
    }

    if lock {
        let (width, height) = env.internal_size();
        graph.lock(width, height);
    }
    Ok(graph)
}

/// Builds the engine's own graph when an asset does not compile.
pub type Fallback = fn(PassBuildContext<'_>) -> RenderGraph;

/// A graph asset as a pipeline for [`helio::RendererBuilder::with_graph`].
///
/// An asset that fails to build is logged and replaced by `fallback`, so a
/// broken graph never leaves the viewport without a renderer; the editor
/// reports the same diagnostics where they can be fixed.
pub struct GraphAssetPipeline {
    asset: Arc<GraphAsset>,
    registry: Arc<NodeRegistry>,
    fallback: Fallback,
}

impl GraphAssetPipeline {
    pub fn new(asset: Arc<GraphAsset>, registry: Arc<NodeRegistry>, fallback: Fallback) -> Self {
        Self {
            asset,
            registry,
            fallback,
        }
    }
}

impl IntoGraph for GraphAssetPipeline {
    fn into_graph(self: Box<Self>, ctx: PassBuildContext<'_>) -> RenderGraph {
        build(&self.asset, &self.registry, self.fallback, ctx)
    }
}

fn build(
    asset: &Arc<GraphAsset>,
    registry: &Arc<NodeRegistry>,
    fallback: Fallback,
    ctx: PassBuildContext<'_>,
) -> RenderGraph {
    let mut env = BuildEnv::new(
        ctx.device,
        ctx.queue,
        ctx.config,
        ctx.debug_state.clone(),
        ctx.camera_buffer,
        ctx.debug_camera_buffer,
        ctx.cull_stats_buffer,
        ctx.owns_device,
        ctx.scene_db.clone(),
    );
    match compile(asset, registry, &mut env) {
        Ok(mut graph) => {
            // Resize and shader reload rebuild from the same asset.
            let asset = Arc::clone(asset);
            let registry = Arc::clone(registry);
            let owns_device = ctx.owns_device;
            let scene_db = ctx.scene_db.clone();
            let rebuilder: GraphRebuilder = Arc::new(
                move |device, queue, config, debug_state, camera, debug_camera, cull_stats| {
                    build(
                        &asset,
                        &registry,
                        fallback,
                        PassBuildContext {
                            device,
                            queue,
                            config,
                            debug_state,
                            camera_buffer: camera,
                            debug_camera_buffer: debug_camera,
                            cull_stats_buffer: cull_stats,
                            owns_device,
                            scene_db: scene_db.clone(),
                        },
                    )
                },
            );
            graph.set_graph_data(rebuilder);
            graph
        }
        Err(diagnostics) => {
            for diagnostic in &diagnostics {
                log::error!(
                    "render graph `{}`: {diagnostic}; using the default pipeline",
                    asset.name
                );
            }
            drop(env);
            fallback(ctx)
        }
    }
}
