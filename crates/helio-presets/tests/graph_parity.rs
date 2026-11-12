//! The shipped pipelines as node graphs: the editor's laid-out template builds
//! the deferred preset's graph, each preset follows the renderer config (the
//! anti-aliasing, reflections), a preset whose lighting model is not registered draws
//! its G-buffer unlit, and a lighting model brings its own pipeline and reads.
//! (An engine's features joining the presets is tested
//! where they live: the voxel terrain's in its plugin's feature crate.)
//!
//! Everything needs a device: the graphs build their passes for real
//! (pipelines, buffers, bind group layouts). Without one the device-gated
//! tests skip; the pure-structure guarantee stays with the registry and
//! validation tests here, and `graphs.rs`'s order tests.
use std::collections::HashSet;
use std::sync::{Arc, Mutex};

use helio::{DebugDrawState, IntoGraph, PassBuildContext, RendererConfig};
use helio_core::{GpuCameraUniforms, PassDeclaration};
use helio_graph_asset::{Connection, GraphAsset, NodeInstance, PinRef};
use helio_presets::Preset;
use helio_presets::{features, graphs};
use pulsar_scenedb::gpu::{EngineGpuContext, GpuMirrorHandle, SceneGpuConfig, SceneGpuStore};

/// One test at a time may hold a wgpu device (Pulsar-Native#1647). Creating and
/// using several devices on one adapter at once crashes some Windows drivers
/// with STATUS_ACCESS_VIOLATION, and libtest runs tests in parallel; this makes
/// `cargo test` safe without `--test-threads=1`. Each test takes it first.
/// (nextest runs one test per process; `.config/nextest.toml` also serializes
/// these suites in a `gpu-device` test group.)
static GPU_DEVICE: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn gpu_serial() -> std::sync::MutexGuard<'static, ()> {
    // A test that panicked while holding the lock does not poison the rest.
    GPU_DEVICE.lock().unwrap_or_else(|e| e.into_inner())
}

fn device() -> Option<(Arc<wgpu::Device>, Arc<wgpu::Queue>)> {
    pollster::block_on(async {
        let instance = wgpu::Instance::new(helio_core::wgpu_instance_descriptor());
        let adapter = instance.request_adapter(&Default::default()).await.ok()?;
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                required_features: helio::required_wgpu_features(adapter.features()),
                required_limits: helio::required_wgpu_limits(adapter.limits()),
                experimental_features: helio::required_experimental_features(adapter.features()),
                ..Default::default()
            })
            .await
            .ok()?;
        Some((Arc::new(device), Arc::new(queue)))
    })
}

fn mirror(
    device: &Arc<wgpu::Device>,
    queue: &Arc<wgpu::Queue>,
) -> pulsar_scenedb::gpu::GpuMirrorHandle {
    let gpu_context = EngineGpuContext::new(Arc::clone(device), Arc::clone(queue));
    let mut gpu_store = SceneGpuStore::new(
        &gpu_context,
        SceneGpuConfig {
            classes: Vec::new(),
            tombstone_headroom: 0,
            max_cells_metadata: 0,
        },
    );
    GpuMirrorHandle::new(Arc::new(gpu_store), Arc::clone(queue))
}

fn uniform_buffer(device: &wgpu::Device, label: &str, size: u64) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::UNIFORM
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

struct Fixtures {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    camera: wgpu::Buffer,
    debug_camera: wgpu::Buffer,
    cull_stats: wgpu::Buffer,
    scene_db: pulsar_scenedb::gpu::GpuMirrorHandle,
    debug_state: Arc<Mutex<DebugDrawState>>,
    config: RendererConfig,
}

fn fixtures() -> Option<Fixtures> {
    let (device, queue) = device()?;
    Some(Fixtures {
        camera: uniform_buffer(
            &device,
            "Scene Camera Buffer",
            (std::mem::size_of::<GpuCameraUniforms>() * 2) as u64,
        ),
        debug_camera: uniform_buffer(&device, "Debug Camera Buffer", 64),
        cull_stats: uniform_buffer(&device, "Cull Stats Buffer", 64),
        scene_db: mirror(&device, &queue),
        debug_state: Arc::new(Mutex::new(DebugDrawState::default())),
        config: RendererConfig::new(320, 180, wgpu::TextureFormat::Rgba8Unorm),
        device,
        queue,
    })
}

impl Fixtures {
    fn context(&self) -> PassBuildContext<'_> {
        PassBuildContext {
            device: &self.device,
            queue: &self.queue,
            config: self.config.clone(),
            debug_state: Arc::clone(&self.debug_state),
            camera_buffer: &self.camera,
            debug_camera_buffer: &self.debug_camera,
            cull_stats_buffer: &self.cull_stats,
            owns_device: false,
            scene_db: self.scene_db.clone(),
        }
    }
}

fn passes(graph: &helio_core::RenderGraph) -> Vec<PassDeclaration> {
    (0..graph.pass_count())
        .filter_map(|index| graph.pass_declaration(index))
        .collect()
}

fn chain(id: &str) -> NodeInstance {
    NodeInstance {
        id: id.into(),
        kind: String::new(),
        params: Default::default(),
        position: [0.0; 2],
    }
}

fn start_and_output(asset: &mut GraphAsset) {
    asset.nodes.push(chain("start"));
    let start = asset.nodes.last_mut().unwrap();
    start.kind = helio_graph_asset::core_nodes::ON_FRAME_START.into();
    asset.nodes.push(chain("out"));
    let out = asset.nodes.last_mut().unwrap();
    out.kind = helio_graph_asset::core_nodes::OUTPUT.into();
}

fn wire(asset: &mut GraphAsset, from: &str, to: &str) {
    asset.connections.push(Connection {
        from: PinRef::new(from, "out"),
        to: PinRef::new(to, "in"),
    });
}

#[test]
fn reflection_nodes_carry_the_platform_capability() {
    let _gpu = gpu_serial();
    let registry = graphs::base_registry();
    for name in ["ssr", "planar_reflection"] {
        let descriptor = registry
            .get(&features::kind(name))
            .expect("reflection nodes are registered");
        assert_eq!(
            descriptor.unsupported.is_some(),
            !helio_core::REFLECTIONS_SUPPORTED,
            "node `{name}` is marked unsupported on this platform"
        );
    }

    let asset = graphs::deferred_graph_asset(&registry);
    let validation = helio_graph_asset::validate(&asset, &registry);
    let capability_warnings: HashSet<&str> = validation
        .diagnostics
        .iter()
        .filter(|d| d.message.contains("builds nothing"))
        .filter_map(|d| d.node.as_deref())
        .collect();
    assert!(capability_warnings.is_empty(), "{capability_warnings:?}");
}

#[test]
fn nodes_of_a_capability_the_platform_lacks_are_reported_at_validation() {
    let _gpu = gpu_serial();
    if helio_core::REFLECTIONS_SUPPORTED {
        eprintln!("REFLECTIONS supported here; the capability path is Apple targets only");
        return;
    }
    let registry = graphs::base_registry();
    let mut asset = GraphAsset::default();
    asset.nodes.push(chain("ssr"));
    let ssr = asset.nodes.last_mut().unwrap();
    ssr.kind = features::kind("ssr");
    start_and_output(&mut asset);
    wire(&mut asset, "start", "ssr");
    wire(&mut asset, "ssr", "out");
    let validation = helio_graph_asset::validate(&asset, &registry);
    assert!(validation
        .diagnostics
        .iter()
        .any(|d| d.node.as_deref() == Some("ssr") && d.message.contains("builds nothing")));
    assert!(!validation.has_errors());
}

/// The preset and the graph asset made from it for `fixtures`' config (what
/// Pulsar's default pipelines plugin writes as its `.rendergraph` file), built
/// as a fixed asset: the same passes with the same declarations in the same
/// order, the same swap policy, size and host inputs.
fn template(fixtures: &Fixtures) -> helio_core::RenderGraph {
    let registry = Arc::new(graphs::base_registry());
    let asset = graphs::deferred_graph_asset(&registry);
    Box::new(helio_graph_asset::GraphAssetPipeline::new(
        Arc::new(asset),
        registry,
    ))
    .into_graph(fixtures.context())
    .expect("the pipeline builds")
}

fn assert_template_matches_preset(fixtures: &Fixtures, what: &str) {
    let preset = Preset::Deferred.build(fixtures.context());
    let template = template(fixtures);

    let policy = |g: &helio_core::RenderGraph| format!("{:?}", g.swap_policy());
    assert!(
        preset.swap_policy().is_some(),
        "the preset declares a policy"
    );
    assert_eq!(
        policy(&preset),
        policy(&template),
        "{what}: swap policies differ"
    );
    assert_eq!(preset.internal_size(), template.internal_size());
    let inputs = |g: &helio_core::RenderGraph| {
        let mut names = g.external_input_names();
        names.sort_unstable();
        names
    };
    assert_eq!(
        inputs(&preset),
        inputs(&template),
        "{what}: host inputs differ"
    );
    assert_eq!(
        passes(&preset),
        passes(&template),
        "{what}: pass list or declarations differ (preset vs template)"
    );
}

#[test]
fn the_template_builds_the_deferred_preset() {
    let _gpu = gpu_serial();
    let Some(mut fixtures) = fixtures() else {
        eprintln!("GPU_VALIDATION_SKIPPED_NO_ADAPTER: deferred template");
        return;
    };
    assert_template_matches_preset(&fixtures, "FXAA");

    // Temporal resolve: the graph has the TSR node at the project's quality, as
    // the preset does, not FXAA.
    let base = fixtures.config.clone();
    for quality in [helio::TsrQuality::Performance, helio::TsrQuality::Quality] {
        fixtures.config = base.clone().with_tsr_quality(quality);
        assert_template_matches_preset(&fixtures, &format!("TSR {quality:?}"));
    }
    let template = template(&fixtures);
    assert!(has::<helio_pass_tsr::TsrPass>(&template));
    assert!(!has::<helio_pass_fxaa::FxaaPass>(&template));

    // The other config switches reach the graph the same way.
    fixtures.config = base;
    fixtures.config.enable_ssr = true;
    assert_template_matches_preset(&fixtures, "SSR");
}

/// The graph carries what the preset used to insert in code (the lighting
/// model's draw phases): with a forward model registered the graph's frame
/// setup node gives the forward phase, so the forward drawer and the lit
/// G-buffer are not both drawn.
#[test]
fn the_graph_sets_the_frame_from_its_frame_setup_node() {
    let _gpu = gpu_serial();
    use helio_frame_api::lighting::{model, LightingModel, Shading};
    let Some(fixtures) = fixtures() else {
        eprintln!("GPU_VALIDATION_SKIPPED_NO_ADAPTER: frame setup");
        return;
    };
    let features = helio_feature_api::FeatureSet::new().with(Model {
        model: LightingModel::new(model::FORWARD, Shading::Forward),
    });
    let registry = Arc::new(graphs::registry_with(&features));
    let asset = graphs::graph_asset(Preset::Forward, &registry);
    let from_graph =
        helio_graph_asset::GraphAssetPipeline::new(Arc::new(asset), Arc::clone(&registry));
    let from_graph = Box::new(from_graph)
        .into_graph(fixtures.context())
        .expect("the pipeline builds");
    let from_preset = Preset::Forward
        .with_registry(registry)
        .build(fixtures.context());
    assert_eq!(passes(&from_graph), passes(&from_preset));
    assert!(!has::<helio_spine::UnlitPass>(&from_graph));
}

fn has<T: helio_core::RenderPass + 'static>(graph: &helio_core::RenderGraph) -> bool {
    graph.find_pass::<T>().is_some()
}

#[test]
fn presets_follow_the_config() {
    let _gpu = gpu_serial();
    let Some(mut fixtures) = fixtures() else {
        eprintln!("GPU_VALIDATION_SKIPPED_NO_ADAPTER: presets follow the config");
        return;
    };
    let base = fixtures.config.clone();

    let graph = Preset::Deferred.build(fixtures.context());
    assert!(has::<helio_pass_fxaa::FxaaPass>(&graph));
    assert!(!has::<helio_pass_tsr::TsrPass>(&graph));
    assert!(!has::<helio_pass_ssr::SsrPass>(&graph));

    fixtures.config = base.clone().with_tsr_quality(helio::TsrQuality::Quality);
    let graph = Preset::Deferred.build(fixtures.context());
    assert!(has::<helio_pass_tsr::TsrPass>(&graph));
    assert!(!has::<helio_pass_fxaa::FxaaPass>(&graph));
    // The FXAA presets never resolve temporally.
    let graph = Preset::DeferredFxaa.build(fixtures.context());
    assert!(has::<helio_pass_fxaa::FxaaPass>(&graph));
    assert!(!has::<helio_pass_tsr::TsrPass>(&graph));

    fixtures.config = base.clone();
    fixtures.config.enable_ssr = true;
    let graph = Preset::Deferred.build(fixtures.context());
    assert_eq!(
        has::<helio_pass_ssr::SsrPass>(&graph),
        helio_core::REFLECTIONS_SUPPORTED
    );
}

/// A plugin's lighting model: the test's stand-in for what the lights
/// register. `id` is the model, `point` where its one pass joins.
struct Model {
    model: helio_frame_api::lighting::LightingModel,
}

impl helio_feature_api::Feature for Model {
    fn id(&self) -> &'static str {
        "test.lighting_model"
    }

    fn register(&self, registrar: &mut helio_feature_api::FeatureRegistrar<'_>) {
        use helio_frame_api::lighting::RegisterLightingModel;
        registrar.lighting_model(self.model.clone());
        if let Some(point) = self.model.point {
            registrar.node(
                helio_feature_api::pass_node("test_model", "Test model", "Test", "", |_| Ok(()))
                    .at(point, 0),
            );
        }
    }
}

/// The forward preset's model is the lights': without it the preset falls
/// back to the unlit stand-in (a G-buffer pipeline), and with it the forward
/// phase runs and nothing resolves a G-buffer.
#[test]
fn the_forward_preset_runs_the_forward_phase_when_its_model_is_registered() {
    let _gpu = gpu_serial();
    use helio_frame_api::lighting::{model, LightingModel, Shading};
    let Some(fixtures) = fixtures() else {
        eprintln!("GPU_VALIDATION_SKIPPED_NO_ADAPTER: forward model");
        return;
    };
    let nodes = graphs::base_registry();
    let phases = Preset::Forward.phases(&nodes);
    assert!(phases.contains(&helio_frame_api::Phase::GBuffer));
    assert!(has::<helio_spine::UnlitPass>(
        &Preset::Forward.build(fixtures.context())
    ));

    let forward = Model {
        model: LightingModel::new(model::FORWARD, Shading::Forward),
    };
    let features = helio_feature_api::FeatureSet::new().with(forward);
    let nodes = graphs::registry_with(&features);
    let phases = Preset::Forward.phases(&nodes);
    assert!(phases.contains(&helio_frame_api::Phase::Forward));
    assert!(!phases.contains(&helio_frame_api::Phase::GBuffer));
    let graph = Preset::Forward
        .with_features(&features)
        .build(fixtures.context());
    assert!(!has::<helio_spine::UnlitPass>(&graph));
}

/// Helio has no lighting model (they are the lights plugin's): every
/// G-buffer preset, and a preset that chose a model nobody registered,
/// resolves its G-buffer with the spine's unlit stand-in, the intended
/// failure, so a host without the lights still shows its surfaces. A feature
/// that registers the chosen model takes its place.
#[test]
fn without_a_lighting_model_the_gbuffer_is_drawn_unlit() {
    let _gpu = gpu_serial();
    use helio_frame_api::lighting::{model, Antialiasing, LightingModel, ModelPipeline, Shading};
    let Some(fixtures) = fixtures() else {
        eprintln!("GPU_VALIDATION_SKIPPED_NO_ADAPTER: unlit stand-in");
        return;
    };
    let absent = Preset::Model(ModelPipeline {
        id: "test.absent",
        label: "Absent model",
        description: "",
        model: "absent",
        aa: Antialiasing::TemporalOrNone,
    });
    // Every built-in preset (the forward one included: without the lights
    // plugin its model is absent too) builds its graph on the stand-in.
    let nodes = graphs::base_registry();
    for preset in Preset::ALL.into_iter().chain([absent]) {
        assert_eq!(preset.model(&nodes).id, model::UNLIT, "{}", preset.name());
        let graph = preset.build(fixtures.context());
        let unlit = passes(&graph)
            .into_iter()
            .find(|pass| pass.type_name == "helio_spine::unlit::UnlitPass")
            .unwrap_or_else(|| panic!("{} draws no unlit stand-in", preset.name()));
        assert!(unlit
            .reads
            .iter()
            .any(|read| read.name == helio_frame_api::gbuffer::GROUP));
        assert!(unlit
            .writes
            .iter()
            .all(|write| write.name != helio_frame_api::gbuffer::GROUP));
    }

    // A registered model at its point replaces the stand-in.
    let deferred = Model {
        model: LightingModel::new(model::DEFERRED, Shading::GBuffer)
            .at(helio_feature_api::ExtensionPoint("test.lighting.model")),
    };
    let with_model = Preset::Deferred
        .with_features(&helio_feature_api::FeatureSet::new().with(deferred))
        .build(fixtures.context());
    assert!(!has::<helio_spine::UnlitPass>(&with_model));
}

/// What a model reads is its registration's: a model that leaves transparency
/// to TSR's reactive mask and writes its own scene format is a preset by
/// registration alone, with no preset of its own in Helio.
#[test]
fn a_model_brings_its_own_pipeline_and_reads() {
    let _gpu = gpu_serial();
    use helio_frame_api::lighting::{
        Antialiasing, InputsOrder, LightingModel, ModelPipeline, Shading, SpecularLayers,
    };
    let Some(mut fixtures) = fixtures() else {
        eprintln!("GPU_VALIDATION_SKIPPED_NO_ADAPTER: model pipeline");
        return;
    };
    let registered = LightingModel::new("test.model", Shading::GBuffer)
        .at(helio_feature_api::ExtensionPoint("test.lighting.model"))
        .with_specular_layers(SpecularLayers::CompositedAfter)
        .with_inputs_order(InputsOrder::BeforeLighting)
        .with_reactive_transparency()
        .with_pipeline(ModelPipeline {
            id: "test.pipeline",
            label: "Test pipeline",
            description: "",
            model: "test.model",
            aa: Antialiasing::TemporalOrNone,
        });
    let pipeline = registered.pipelines[0];
    let features = helio_feature_api::FeatureSet::new().with(Model { model: registered });

    // A host finds its pipeline among the registered models' (the render
    // pipelines plugin makes an asset of each).
    let registry = graphs::registry_with(&features);
    assert!(helio_frame_api::lighting::registered_models(&registry)
        .iter()
        .flat_map(|model| model.pipelines.iter())
        .any(|listed| listed.id == "test.pipeline"));

    // Under TSR it resolves temporally; without, it is not anti-aliased.
    fixtures.config = fixtures
        .config
        .clone()
        .with_tsr_quality(helio::TsrQuality::Quality);
    let graph = Preset::Model(pipeline)
        .with_features(&features)
        .build(fixtures.context());
    assert!(has::<helio_pass_tsr::TsrPass>(&graph));
    assert!(!has::<helio_spine::UnlitPass>(&graph));
    fixtures.config = fixtures.config.clone().without_tsr();
    let graph = Preset::Model(pipeline)
        .with_features(&features)
        .build(fixtures.context());
    assert!(!has::<helio_pass_tsr::TsrPass>(&graph));
    assert!(!has::<helio_pass_fxaa::FxaaPass>(&graph));
}

#[test]
fn the_tsr_node_configures_motion_vectors_and_the_stochastic_mask() {
    let _gpu = gpu_serial();
    let Some((device, queue)) = device() else {
        eprintln!("GPU_VALIDATION_SKIPPED_NO_ADAPTER: TSR node configuration");
        return;
    };
    let scene_db = mirror(&device, &queue);
    let camera = uniform_buffer(&device, "Scene Camera Buffer", 512);

    for stochastic in [false, true] {
        let debug_state = Arc::new(Mutex::new(DebugDrawState::default()));
        let debug_camera = uniform_buffer(&device, "Debug Camera Buffer", 64);
        let cull_stats = uniform_buffer(&device, "Cull Stats Buffer", 64);
        let mut env = helio_graph_asset::build_env(&helio::PassBuildContext {
            device: &device,
            queue: &queue,
            config: RendererConfig::new(320, 180, wgpu::TextureFormat::Rgba8Unorm),
            debug_state: Arc::clone(&debug_state),
            camera_buffer: &camera,
            debug_camera_buffer: &debug_camera,
            cull_stats_buffer: &cull_stats,
            owns_device: false,
            scene_db: scene_db.clone(),
        });

        let registry = graphs::base_registry();
        let mut asset = GraphAsset::default();
        asset.nodes.push(chain("tsr"));
        let tsr = asset.nodes.last_mut().unwrap();
        tsr.kind = features::kind("tsr");
        if stochastic {
            tsr.params
                .insert("stochastic_mask".into(), serde_json::json!(true));
        }
        start_and_output(&mut asset);
        wire(&mut asset, "start", "tsr");
        wire(&mut asset, "tsr", "out");

        let graph =
            helio_graph_asset::compile(&asset, &registry, &mut env).expect("the graph compiles");
        let declarations = passes(&graph);
        let tsr = declarations
            .iter()
            .find(|pass| pass.type_name == "helio_pass_tsr::TsrPass")
            .expect("the tsr node's graph holds a TsrPass");
        let reads: HashSet<&str> = tsr.reads.iter().map(|read| read.name).collect();
        assert!(
            reads.contains(helio_frame_api::gbuffer::VELOCITY),
            "TSR reads {}: {tsr:?}",
            helio_frame_api::gbuffer::VELOCITY
        );
        assert_eq!(
            reads.contains(helio_pass_tsr::STOCHASTIC_MASK),
            stochastic,
            "the stochastic-mask parameter decides {}",
            helio_pass_tsr::STOCHASTIC_MASK
        );
    }
}

#[test]
fn the_debug_overlay_is_a_node_with_no_default_placement() {
    let registry = graphs::base_registry();
    let node = registry
        .get(helio_pass_debug_overlay::DEBUG_OVERLAY_NODE)
        .expect("the debug overlay is a registered node");
    assert_eq!(node.kind, features::kind("debug_overlay"));
    assert!(node.params.iter().any(|p| p.key == "enabled"));
    assert!(node.params.iter().any(|p| p.key == "visible"));
    assert!(
        node.placement.is_none(),
        "no shipped pipeline draws the overlay"
    );
}

#[test]
fn a_graph_with_the_debug_overlay_node_builds_it_over_the_hosts_state() {
    let _gpu = gpu_serial();
    let Some(fixtures) = fixtures() else {
        eprintln!("GPU_VALIDATION_SKIPPED_NO_ADAPTER: debug overlay node");
        return;
    };
    let host = helio_pass_debug_overlay::DebugOverlayState::new();
    let mut config = fixtures.config.clone();
    config = config.with_feature(helio_pass_debug_overlay::DebugOverlayHost(host.clone()));
    let ctx = PassBuildContext {
        config,
        ..fixtures.context()
    };
    let mut env = helio_graph_asset::build_env(&ctx);

    let registry = graphs::base_registry();
    let mut asset = GraphAsset::default();
    asset.nodes.push(chain("overlay"));
    asset.nodes.last_mut().unwrap().kind = helio_pass_debug_overlay::DEBUG_OVERLAY_NODE.into();
    asset
        .nodes
        .last_mut()
        .unwrap()
        .params
        .insert("visible".into(), serde_json::json!(true));
    start_and_output(&mut asset);
    wire(&mut asset, "start", "overlay");
    wire(&mut asset, "overlay", "out");

    let graph =
        helio_graph_asset::compile(&asset, &registry, &mut env).expect("the graph compiles");
    let pass = graph
        .find_pass::<helio_pass_debug_overlay::DebugOverlayPass>()
        .expect("the node builds the overlay pass");
    assert!(
        Arc::ptr_eq(pass.shared(), &host),
        "the pass draws the host's state"
    );
    assert!(
        host.lock().unwrap().enabled,
        "the visible parameter shows it"
    );
}
