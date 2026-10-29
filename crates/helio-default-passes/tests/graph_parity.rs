//! The graph-compiled default deferred pipeline is the legacy builder's twin:
//! one pass declares and sits exactly where the code that used to be this graph
//! put it, the swap policy and lock size agree, and host inputs agree.
//!
//! Known gaps this harness scopes out (Phase 1+): the voxel and GI passes are
//! not nodes yet, so the graph has none and the legacy list is filtered to the
//! passes both declare. Forward, HLFS and simple pipelines are not graphs at
//! all, so there is nothing to compare them with.
//!
//! Everything needs a device: both pipelines build their passes for real
//! (pipelines, buffers, bind group layouts). Without one the device-gated
//! tests skip; the pure-structure guarantee stays with the registry and
//! validation tests here, and `graphs.rs`'s order tests.
use std::collections::HashSet;
use std::sync::{Arc, Mutex};

use helio::{DebugDrawState, PassBuildContext, RenderPipelineRegistry, RendererConfig};
use helio_core::{GpuCameraUniforms, PassDeclaration};
use helio_default_passes::pipelines::{DEFAULT, DEFERRED_GRAPH};
use helio_default_passes::{graph_nodes, graphs};
use helio_graph_asset::{BuildEnv, Connection, GraphAsset, NodeInstance, PinRef};
use pulsar_scenedb::gpu::{EngineGpuContext, GpuMirrorHandle, SceneGpuConfig, SceneGpuStore};

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
    helio_pass_sky::AtmosphereComponent::register_gpu_columns_growable(&mut gpu_store, 4, device);
    helio_pass_gbuffer::MeshComponent::register_gpu_columns_growable(&mut gpu_store, 16, device);
    helio_pass_gbuffer::MaterialComponent::register_gpu_columns_growable(
        &mut gpu_store,
        16,
        device,
    );
    helio_pass_gbuffer::StaticObjectComponent::register_gpu_columns_growable(
        &mut gpu_store,
        16,
        device,
    );
    helio_pass_forward_lit::LightComponent::register_gpu_columns_growable(
        &mut gpu_store,
        16,
        device,
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
    let registry = graphs::base_registry();
    for name in ["ssr", "planar_reflection"] {
        let descriptor = registry
            .get(&graph_nodes::kind(name))
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
    if helio_core::REFLECTIONS_SUPPORTED {
        eprintln!("REFLECTIONS supported here; the capability path is Apple targets only");
        return;
    }
    let registry = graphs::base_registry();
    let mut asset = GraphAsset::default();
    asset.nodes.push(chain("ssr"));
    let ssr = asset.nodes.last_mut().unwrap();
    ssr.kind = graph_nodes::kind("ssr");
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

#[test]
fn the_graph_pipeline_matches_the_legacy_default() {
    let Some(fixtures) = fixtures() else {
        eprintln!("GPU_VALIDATION_SKIPPED_NO_ADAPTER: deferred graph parity");
        return;
    };

    let mut pipelines = RenderPipelineRegistry::new();
    helio_default_passes::pipelines::register_builtin_pipelines(&mut pipelines);
    let legacy = pipelines
        .create(Some(DEFAULT))
        .expect("legacy pipeline registered")
        .into_graph(fixtures.context());
    let graph = pipelines
        .create(Some(DEFERRED_GRAPH))
        .expect("graph pipeline registered")
        .into_graph(fixtures.context());

    let policy = |g: &helio_core::RenderGraph| format!("{:?}", g.swap_policy());
    assert!(legacy.swap_policy().is_some(), "legacy declares a policy");
    assert_eq!(policy(&legacy), policy(&graph), "swap policies differ");
    assert_eq!(
        legacy.internal_size(),
        graph.internal_size(),
        "lock sizes differ"
    );

    let mut legacy_inputs: Vec<_> = legacy.external_input_names();
    let mut graph_inputs: Vec<_> = graph.external_input_names();
    legacy_inputs.sort_unstable();
    graph_inputs.sort_unstable();
    assert_eq!(legacy_inputs, graph_inputs, "host inputs differ");

    // The graph has no voxel or GI nodes yet, so the legacy passes those
    // factories add are scoped out (see this file's doc comment).
    let scoped_out: HashSet<&'static str> = HashSet::from([
        helio_core::graph::type_name_of::<helio_pass_gi::GiPass>(),
        helio_core::graph::type_name_of::<helio_pass_gi::GiMeshTracePass>(),
        helio_core::graph::type_name_of::<helio_pass_gi::GiResolvePass>(),
        helio_core::graph::type_name_of::<helio_pass_sky::CloudTracePass>(),
    ]);
    let legacy_passes: Vec<PassDeclaration> = passes(&legacy)
        .into_iter()
        .filter(|pass| !scoped_out.contains(pass.type_name))
        .collect();
    let graph_passes = passes(&graph);
    assert_eq!(
        legacy_passes, graph_passes,
        "pass list or declarations differ (legacy minus GI/cloud vs graph)"
    );
}

#[test]
fn the_tsr_node_configures_motion_vectors_and_the_stochastic_mask() {
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
        let mut env = BuildEnv::new(
            &device,
            &queue,
            RendererConfig::new(320, 180, wgpu::TextureFormat::Rgba8Unorm),
            Arc::clone(&debug_state),
            &camera,
            &debug_camera,
            &cull_stats,
            false,
            scene_db.clone(),
        );

        let registry = graphs::base_registry();
        let mut asset = GraphAsset::default();
        asset.nodes.push(chain("tsr"));
        let tsr = asset.nodes.last_mut().unwrap();
        tsr.kind = graph_nodes::kind("tsr");
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
            reads.contains(helio_pass_tsr::MOTION_VECTORS),
            "TSR reads {}: {tsr:?}",
            helio_pass_tsr::MOTION_VECTORS
        );
        assert_eq!(
            reads.contains(helio_pass_tsr::STOCHASTIC_MASK),
            stochastic,
            "the stochastic-mask parameter decides {}",
            helio_pass_tsr::STOCHASTIC_MASK
        );
    }
}
