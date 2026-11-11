//! Compiles a graph asset into a Helio [`RenderGraph`], and wraps it as an
//! [`IntoGraph`] pipeline.

use std::ops::Range;
use std::sync::Arc;

use helio::{GraphRebuilder, IntoGraph, PassBuildContext, PipelineFailure};
use helio_core::RenderGraph;

use crate::asset::{GraphAsset, NodeInstance};
use crate::validate::{validate, Diagnostic, Severity};
use helio_feature_api::{BuildEnv, NodeBuild, NodeRegistry};

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
    // OpenXR renders one D2 swapchain-layer view per eye, so graph targets stay
    // single-layer: the two-layer multiview allocator would mix D2Array
    // G-buffer attachments with the renderer's single-layer depth and fail wgpu
    // validation in the G-buffer.
    graph.with_xr_mode(false);
    if let Some(path) = std::env::var_os("PULSAR_PIPELINE_CACHE") {
        graph.enable_pipeline_cache_persistence(path);
    }

    for prepare in registry.preparers() {
        prepare(env);
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
    for finish in registry.finishers() {
        finish(env, &mut graph).map_err(|message| {
            vec![Diagnostic {
                severity: Severity::Error,
                node: None,
                message,
            }]
        })?;
    }

    graph.set_scene_derivations(registry.scene_derivations().to_vec());
    graph.set_untraced_geometry(registry.untraced_geometry().to_vec());

    if lock {
        graph.lock_sized(env.frame.render_size());
    }
    Ok(graph)
}

/// Makes the asset a pipeline builds, from the build's environment (its
/// renderer config is the [`helio::RendererConfig`] setting). It may add
/// settings the asset's nodes read.
pub type MakeAsset = Arc<dyn Fn(&NodeRegistry, &mut BuildEnv<'_>) -> GraphAsset + Send + Sync>;

/// Where a pipeline's asset comes from.
#[derive(Clone)]
enum Source {
    /// An authored asset, built as it is.
    Fixed(Arc<GraphAsset>),
    /// An asset made for each build (a shipped preset follows the config).
    Made(MakeAsset),
}

/// A graph asset as a pipeline for [`helio::RendererBuilder::with_graph`].
///
/// An asset that fails to build is the pipeline's failure
/// ([`PipelineFailure`]: the reason, every diagnostic, and a rebuilder so it can
/// be tried again), which the renderer shows on its output and logs. No other
/// graph is built in its place; the editor reports the same diagnostics where
/// they can be fixed.
///
/// `swap_policy` is what shader hot reload may swap on its own (see
/// [`helio_core::SwapPolicy`]); a graph built only of pass nodes whose
/// constructors share builder-created resources the same way carries the
/// shipped graphs' policy. None leaves the graph unswappable (the host
/// rebuilds it whole on a shader reload).
pub struct GraphAssetPipeline {
    source: Source,
    registry: Arc<NodeRegistry>,
    swap_policy: Option<helio_core::SwapPolicy>,
}

impl GraphAssetPipeline {
    pub fn new(asset: Arc<GraphAsset>, registry: Arc<NodeRegistry>) -> Self {
        Self {
            source: Source::Fixed(asset),
            registry,
            swap_policy: None,
        }
    }

    /// A pipeline whose asset `make` makes on every build, resizes and
    /// shader reloads included, so it follows the renderer config.
    pub fn made(make: MakeAsset, registry: Arc<NodeRegistry>) -> Self {
        Self {
            source: Source::Made(make),
            registry,
            swap_policy: None,
        }
    }

    /// Stores the [`helio_core::SwapPolicy`] the built graph reports to shader
    /// hot reload.
    pub fn with_swap_policy(mut self, swap_policy: helio_core::SwapPolicy) -> Self {
        self.swap_policy = Some(swap_policy);
        self
    }
}

impl IntoGraph for GraphAssetPipeline {
    fn into_graph(
        self: Box<Self>,
        ctx: PassBuildContext<'_>,
    ) -> Result<RenderGraph, PipelineFailure> {
        build(Arc::new(*self), ctx)
    }
}

/// The [`BuildEnv`] for a renderer build context: its frame, plus the
/// renderer's config, debug-draw state and scene handle as settings (the
/// default passes' nodes read them; a feature reads its own settings type).
pub fn build_env<'e>(ctx: &PassBuildContext<'e>) -> BuildEnv<'e> {
    let config = &ctx.config;
    BuildEnv::new(
        ctx.device,
        ctx.queue,
        helio_feature_api::FrameDesc {
            width: config.width,
            height: config.height,
            internal_width: config.internal_width(),
            internal_height: config.internal_height(),
            surface_format: config.surface_format,
            xr: config.enable_xr,
        },
        ctx.camera_buffer,
        ctx.debug_camera_buffer,
        ctx.cull_stats_buffer,
        ctx.owns_device,
    )
    .with_feature_settings(&ctx.config.features)
    .with_setting(ctx.config.clone())
    .with_setting(ctx.debug_state.clone())
    .with_setting(ctx.scene_db.clone())
}

/// What a pipeline's failure says about the diagnostics `asset` did not build
/// with: the errors first, then the warnings (routing findings among them).
fn failure(name: &str, diagnostics: &[Diagnostic], rebuilder: GraphRebuilder) -> PipelineFailure {
    let errors = diagnostics
        .iter()
        .filter(|d| d.severity == Severity::Error)
        .count();
    let warnings = diagnostics.len() - errors;
    let ordered = diagnostics
        .iter()
        .filter(|d| d.severity == Severity::Error)
        .chain(
            diagnostics
                .iter()
                .filter(|d| d.severity == Severity::Warning),
        );
    PipelineFailure::new(format!(
        "render graph `{name}` does not build ({errors} error(s), {warnings} warning(s))"
    ))
    .with_details(ordered.map(ToString::to_string))
    .with_rebuilder(rebuilder)
}

fn build(
    pipeline: Arc<GraphAssetPipeline>,
    ctx: PassBuildContext<'_>,
) -> Result<RenderGraph, PipelineFailure> {
    let mut env = build_env(&ctx);
    let asset = match &pipeline.source {
        Source::Fixed(asset) => Arc::clone(asset),
        Source::Made(make) => Arc::new(make(&pipeline.registry, &mut env)),
    };
    // Resize, shader reload and a retry after a failure rebuild the same way.
    let owns_device = ctx.owns_device;
    let scene_db = ctx.scene_db.clone();
    let rebuilder: GraphRebuilder = {
        let pipeline = Arc::clone(&pipeline);
        Arc::new(
            move |device, queue, config, debug_state, camera, debug_camera, cull_stats| {
                build(
                    Arc::clone(&pipeline),
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
        )
    };
    match compile(&asset, &pipeline.registry, &mut env) {
        Ok(mut graph) => {
            // The preset's own passes and what its features declared.
            if let Some(swap_policy) = &pipeline.swap_policy {
                graph.set_swap_policy(swap_policy.clone().merged(pipeline.registry.swap_policy()));
            }
            graph.set_graph_data(rebuilder);
            Ok(graph)
        }
        Err(diagnostics) => Err(failure(&asset.name, &diagnostics, rebuilder)),
    }
}
