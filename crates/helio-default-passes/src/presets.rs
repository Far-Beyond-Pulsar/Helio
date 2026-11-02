//! Helio's shipped pipelines, as node graphs.
//!
//! A preset is a graph of the pass nodes in [`crate::graph_nodes`], made for
//! the renderer config it is built with: the config picks the anti-aliasing
//! (TSR when `tsr_quality` is set), screen-space and planar reflections,
//! portals and foliage, and the TSR node's settings. The nodes features
//! register for a [`Slot`] go at that slot, so a feature (GI, the clouds, the
//! voxel terrain) joins every preset without a preset naming it. Every build,
//! resizes and shader reloads included, makes the graph again from the
//! renderer's current config.
//!
//! A host builds a renderer with one by function (`presets::deferred`) or by
//! value (`with_graph(Preset::Deferred)`). Project-authored graphs are fixed
//! [`GraphAsset`]s built by `helio_graph_asset::GraphAssetPipeline`.
//!
//! Every preset reads its GPU timings a frame late instead of waiting for
//! them (the renderer's own poll, when it owns the device, or the host's
//! submits deliver them), as the engine's graphs always have.

use std::sync::{Arc, OnceLock};

use helio::{IntoGraph, PassBuildContext, RendererConfig};
use helio_core::RenderGraph;
use helio_graph_asset::{
    core_nodes, Connection, GraphAsset, GraphAssetPipeline, NodeInstance, NodeRegistry, Params,
    PinRef, Slot,
};
use serde_json::Value;

use crate::graph_nodes::{host_kind, kind, SceneColorFormat, HOST_INPUTS};

/// One of Helio's shipped pipelines.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Preset {
    /// Deferred lighting; TSR when the config asks for it, FXAA otherwise.
    Deferred,
    /// Deferred lighting with FXAA, never TSR.
    DeferredFxaa,
    /// Hybrid light-field shading; TSR when the config asks for it, no
    /// anti-aliasing otherwise.
    Hlfs,
    /// Hybrid light-field shading with FXAA, never TSR.
    HlfsFxaa,
    /// One forward pass for opaque geometry, no G-buffer; FXAA.
    Forward,
    /// A test cube straight into the output: checks the device and surface.
    Simple,
}

/// The anti-aliasing a preset ends its scene-linear chain with.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Aa {
    Tsr,
    Fxaa,
    None,
}

impl Preset {
    pub const ALL: [Preset; 6] = [
        Preset::Deferred,
        Preset::DeferredFxaa,
        Preset::Hlfs,
        Preset::HlfsFxaa,
        Preset::Forward,
        Preset::Simple,
    ];

    pub fn name(self) -> &'static str {
        match self {
            Preset::Deferred => "Deferred",
            Preset::DeferredFxaa => "Deferred + FXAA",
            Preset::Hlfs => "HLFS",
            Preset::HlfsFxaa => "HLFS + FXAA",
            Preset::Forward => "Forward",
            Preset::Simple => "Simple",
        }
    }

    fn hlfs(self) -> bool {
        matches!(self, Preset::Hlfs | Preset::HlfsFxaa)
    }

    fn aa(self, config: &RendererConfig) -> Aa {
        let tsr = config.tsr_quality.is_some();
        match self {
            Preset::Deferred if tsr => Aa::Tsr,
            Preset::Hlfs if tsr => Aa::Tsr,
            Preset::Hlfs | Preset::Simple => Aa::None,
            Preset::Deferred | Preset::DeferredFxaa | Preset::HlfsFxaa | Preset::Forward => {
                Aa::Fxaa
            }
        }
    }

    /// The preset's nodes for `config`, in execution order.
    pub fn nodes(self, registry: &NodeRegistry, config: &RendererConfig) -> Vec<NodeInstance> {
        let mut nodes = Nodes {
            registry,
            list: Vec::new(),
        };
        if self == Preset::Simple {
            nodes.pass("simple_cube");
            return nodes.list;
        }
        let portals = config.enable_portals;
        let reflections = helio_core::REFLECTIONS_SUPPORTED;
        let foliage = config
            .features
            .get_or_default::<helio_pass_foliage_place::FoliageSettings>()
            .enabled;
        let aa = self.aa(config);

        // Scene prep: batches, the sky's tables, shadows, culling.
        for name in [
            "object_batch",
            "atmosphere",
            "shadow_matrix",
            "shadow_dirty",
            "shadow",
            "background",
            "indirect_dispatch",
            "hiz",
            "occlusion_cull",
        ] {
            nodes.pass(name);
        }
        if portals {
            nodes.pass("portal_cull");
        }
        if !self.hlfs() {
            nodes.pass("light_cull");
        }

        // Geometry and lighting.
        if self == Preset::Forward {
            nodes.with("forward_lit", [("render_all_opaque", Value::from(true))]);
            nodes.pass("cloud_trace");
        } else {
            // Placement is a compute pass: before the G-buffer, never between
            // it and the foliage draw, whose render passes fuse.
            if foliage {
                nodes.pass("foliage_place");
            }
            nodes.pass("gbuffer");
            if portals {
                nodes.pass("portal_mask");
                nodes.pass("portal_instance");
            }
            if foliage {
                nodes.pass("foliage_gbuffer");
            }
            nodes.pass("virtual_geometry");
            nodes.slot(Slot::Geometry);
            nodes.pass("decal");
            if self.hlfs() {
                nodes.slot(Slot::LightingInputs);
                nodes.pass("hlfs");
                if config.enable_ssr && reflections {
                    nodes.pass("ssr");
                    nodes.pass("ssr_composite");
                }
            } else {
                if config.enable_ssr && reflections {
                    nodes.pass("ssr");
                }
                if config.enable_planar_reflections && reflections {
                    nodes.pass("planar_reflection");
                }
                nodes.slot(Slot::LightingInputs);
                nodes.pass("deferred_light");
            }
            nodes.slot(Slot::Lighting);
        }

        // Over the lit scene: the sky, then what is not seen through the air.
        nodes.pass("atmosphere_composite");
        nodes.pass("corona");
        if portals {
            nodes.pass("portal_editor_overlay");
        }
        nodes.pass("water_sim");
        // The editor's grid and bounds, then the light sprites in front of them
        // (billboards test the scene depth but do not write it).
        nodes.pass("debug_draw_scene");
        nodes.pass("billboard");

        // The scene-linear chain: media, transparency, AA, lens, tone map.
        nodes.pass("post_process_volume_blend");
        nodes.pass("volumetric_fog");
        nodes.pass("fog_composite");
        let reactive = self.hlfs() && aa == Aa::Tsr;
        nodes.with("transparent", [("reactive_mask", Value::from(reactive))]);
        let resolved = match aa {
            Aa::Tsr => {
                let voxel_terrain = config
                    .features
                    .get::<helio_pass_voxel_planet::VoxelTerrain>()
                    .is_some_and(|terrain| !terrain.passes.is_empty());
                nodes.with(
                    "tsr",
                    [
                        ("quality", Value::from(tsr_quality(config))),
                        // The voxel terrain's sun samples rotate over blocks of
                        // pixels; where its cells are a pixel or two wide TSR
                        // integrates them.
                        ("stochastic_mask", Value::from(voxel_terrain)),
                        ("transparency_reactivity", Value::from(self.hlfs())),
                    ],
                );
                "tsr_color"
            }
            Aa::Fxaa => {
                nodes.pass("fxaa");
                "fxaa_color"
            }
            Aa::None => "fogged_hdr",
        };
        nodes.with("lens", [("color_input", Value::from(resolved))]);
        nodes.pass("lens_flare");
        nodes.with("post_process", [("color_input", Value::from("lens_color"))]);

        // Over the final image.
        nodes.pass("buffer_view");
        nodes.pass("sprite_overlay");
        nodes.pass("perf_overlay");
        nodes.pass("debug_draw");
        nodes.slot(Slot::Output);
        nodes.list
    }

    /// The preset for `config` as a graph: On Frame Start, its nodes on one
    /// execution chain, the Output, and the host inputs it reads.
    pub fn asset(self, registry: &NodeRegistry, config: &RendererConfig) -> GraphAsset {
        let mut asset = GraphAsset {
            name: self.name().into(),
            ..Default::default()
        };
        asset.nodes.push(node("start", core_nodes::ON_FRAME_START));
        let chain = self.nodes(registry, config);
        let ids: Vec<String> = chain.iter().map(|node| node.id.clone()).collect();
        asset.nodes.extend(chain);
        asset.nodes.push(node("output", core_nodes::OUTPUT));
        let mut from = "start".to_string();
        for id in ids.into_iter().chain(["output".to_string()]) {
            asset.connections.push(Connection {
                from: PinRef::new(&from, "out"),
                to: PinRef::new(&id, "in"),
            });
            from = id;
        }
        if self != Preset::Simple {
            for name in HOST_INPUTS {
                asset
                    .nodes
                    .push(node(&format!("host_{name}"), &host_kind(name)));
            }
        }
        asset
    }

    /// The preset as a pipeline over `registry`'s nodes.
    pub fn pipeline_with(self, registry: Arc<NodeRegistry>) -> GraphAssetPipeline {
        let pipeline = GraphAssetPipeline::made(
            Arc::new(move |registry, env| {
                let config = env
                    .require::<RendererConfig>()
                    .expect("a renderer build supplies its config")
                    .clone();
                if self.hlfs() {
                    let format = helio_pass_hlfs::HlfsPass::preferred_output_format(env.device);
                    env.insert_setting(SceneColorFormat(format));
                }
                self.asset(registry, &config)
            }),
            registry,
        );
        if self == Preset::Simple {
            pipeline
        } else {
            pipeline.with_swap_policy(crate::default_swap_policy())
        }
    }

    /// The preset as a pipeline over the default features' nodes.
    pub fn pipeline(self) -> GraphAssetPipeline {
        self.pipeline_with(registry())
    }

    /// Builds the preset's graph for `ctx`.
    pub fn build(self, mut ctx: PassBuildContext<'_>) -> RenderGraph {
        ctx.owns_device = false;
        Box::new(self.pipeline()).into_graph(ctx)
    }
}

impl IntoGraph for Preset {
    fn into_graph(self: Box<Self>, ctx: PassBuildContext<'_>) -> RenderGraph {
        self.build(ctx)
    }
}

/// The default features' nodes ([`crate::graphs::base_registry`]), made once.
pub fn registry() -> Arc<NodeRegistry> {
    static REGISTRY: OnceLock<Arc<NodeRegistry>> = OnceLock::new();
    Arc::clone(REGISTRY.get_or_init(|| Arc::new(crate::graphs::base_registry())))
}

pub fn deferred(ctx: PassBuildContext<'_>) -> RenderGraph {
    Preset::Deferred.build(ctx)
}

pub fn deferred_fxaa(ctx: PassBuildContext<'_>) -> RenderGraph {
    Preset::DeferredFxaa.build(ctx)
}

pub fn hlfs(ctx: PassBuildContext<'_>) -> RenderGraph {
    Preset::Hlfs.build(ctx)
}

pub fn hlfs_fxaa(ctx: PassBuildContext<'_>) -> RenderGraph {
    Preset::HlfsFxaa.build(ctx)
}

pub fn forward(ctx: PassBuildContext<'_>) -> RenderGraph {
    Preset::Forward.build(ctx)
}

pub fn simple(ctx: PassBuildContext<'_>) -> RenderGraph {
    Preset::Simple.build(ctx)
}

fn tsr_quality(config: &RendererConfig) -> &'static str {
    use helio::TsrQuality;
    match config.tsr_quality.unwrap_or_default() {
        TsrQuality::Performance => "performance",
        TsrQuality::Balanced => "balanced",
        TsrQuality::Quality => "quality",
        TsrQuality::Native => "native",
    }
}

fn node(id: &str, kind: &str) -> NodeInstance {
    NodeInstance {
        id: id.into(),
        kind: kind.into(),
        params: Params::default(),
        position: [0.0, 0.0],
    }
}

/// A preset's node list as it is made.
struct Nodes<'r> {
    registry: &'r NodeRegistry,
    list: Vec<NodeInstance>,
}

impl Nodes<'_> {
    fn pass(&mut self, name: &str) {
        self.list.push(node(name, &kind(name)));
    }

    fn with<const N: usize>(&mut self, name: &str, params: [(&str, Value); N]) {
        let mut instance = node(name, &kind(name));
        for (key, value) in params {
            instance.params.insert(key.to_string(), value);
        }
        self.list.push(instance);
    }

    /// The nodes registered for `slot`, by order then kind; a node's id is its
    /// kind without the `helio.pass.` prefix.
    fn slot(&mut self, slot: Slot) {
        for descriptor in self.registry.slotted(slot) {
            let id = descriptor
                .kind
                .trim_start_matches("helio.pass.")
                .replace('.', "_");
            self.list.push(node(&id, &descriptor.kind));
        }
    }
}
