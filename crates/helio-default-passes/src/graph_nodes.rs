//! Every pass of Helio's default pipelines as a render-graph node.
//!
//! One node builds one pass (the sprite overlay's culler and batcher are one
//! unit by design and share a node). A node's pins are **not** written here: they
//! are generated from what its pass declares (`helio_graph_asset::probe`), and a
//! node with no probe result has flow pins only, which is all building needs.
//! What is written here is how to *construct* each pass, taken apart from the
//! old `add_*_passes` builders; the graphs that used to be code are now node
//! graphs ([`crate::graphs`]).
//!
//! Handles one pass hands another (the Hi-Z sampler, the portal buffers) are
//! read back from the graph (`find_pass`), or kept in typed shared state, so a
//! node errors by name when the node it needs has not been built.
//!
//! Optional passes (SSR, planar reflections, portals, foliage, water, TSR) are
//! plain nodes: present in a graph means built. Project graphics settings do
//! not switch them; the graph decides.

use std::sync::{Arc, Mutex};

use helio::{DebugDrawState, RendererConfig};
use helio_graph_asset::resources::intern;
use helio_graph_asset::{
    BuildEnv, MarkerFn, NodeBuild, NodeDescriptor, NodeRegistry, ParamDesc, PinType,
};
use helio_pass_perf_overlay::PerfOverlayShared;

use crate::{
    scene_buffer_or_dummy, AtmosphereCompositePass, AtmospherePass, BackgroundPass, BillboardPass,
    CoronaPass, DecalPass, DeferredLightPass, FogCompositePass, FoliageGBufferPass,
    FoliagePlacePass, FoliageQuality, ForwardLitPass, FxaaPass, GBufferPass, HiZBuildPass,
    IndirectDispatchPass, LensFlarePass, LensPass, LightCullPass, ObjectBatchPass,
    OcclusionCullPass, PerfOverlayAnalyzerPass, PerfOverlayCostAnalyzerPass, PerfOverlayPass,
    PlanarReflectionPass, PortalCullPass, PortalEditorOverlayPass, PortalInstancePass,
    PortalMaskPass, PostProcessPass, PostProcessVolumeBlendPass, ShadowDirtyPass, ShadowMatrixPass,
    ShadowPass, SpriteBatchPass, SpriteCullPass, SsrPass, TsrPass, VolumetricFogPass, WaterSimPass,
    FOGGED_HDR, FOGGED_HDR_FORMAT, LENS_COLOR, MAX_VISIBLE_SPRITES, SPOTLIGHT_PNG,
};

/// Resources the renderer supplies every frame. Each has a host node that
/// declares it; the passes that read one get a Scene input for it.
pub const HOST_INPUTS: &[&str] = &[
    "material_textures",
    "render_environment",
    "billboards",
    "corona_emitters",
];

/// The kind of the node for pass `name`.
pub fn kind(name: &str) -> String {
    format!("helio.pass.{name}")
}

/// The kind of the host node for resource `name`.
pub fn host_kind(name: &str) -> String {
    format!("host.{name}")
}

// ── Shared state ─────────────────────────────────────────────────────────────

/// The foliage G-buffer draw the placement node made, for the Foliage GBuffer node.
struct PendingFoliageDraw(Option<FoliageGBufferPass>);

/// The performance overlay's accumulator, made by whichever node needs it first.
struct PerfState(Arc<Mutex<PerfOverlayShared>>);

fn perf(env: &mut BuildEnv<'_>) -> Arc<Mutex<PerfOverlayShared>> {
    let (device, (width, height)) = (env.device, env.internal_size());
    env.shared(|| PerfState(PerfOverlayShared::new(device, width, height)))
        .0
        .clone()
}

/// The shadow-dirty flags the shadow matrix pass fills and the dirty pass reads.
struct ShadowDirtyBuffer(Arc<wgpu::Buffer>);

/// What a node build reads from the environment, as plain references.
struct Ctx<'e> {
    device: &'e Arc<wgpu::Device>,
    queue: &'e Arc<wgpu::Queue>,
    camera: &'e wgpu::Buffer,
    debug_camera: &'e wgpu::Buffer,
    cull_stats: &'e wgpu::Buffer,
    config: RendererConfig,
    iw: u32,
    ih: u32,
    scene_db: helio::SceneDbHandle,
    debug_state: Arc<Mutex<DebugDrawState>>,
}

fn ctx<'e>(env: &BuildEnv<'e>) -> Ctx<'e> {
    let (iw, ih) = env.internal_size();
    Ctx {
        device: env.device,
        queue: env.queue,
        camera: env.camera_buffer,
        debug_camera: env.debug_camera_buffer,
        cull_stats: env.cull_stats_buffer,
        config: env.config,
        iw,
        ih,
        scene_db: env.scene_db.clone(),
        debug_state: env.debug_state.clone(),
    }
}

/// A pass node: one constructor, an `Enabled` toggle, pins generated from the
/// pass.
fn pass_node(
    name: &str,
    title: &str,
    category: &str,
    description: &str,
    build: impl Fn(&mut NodeBuild<'_, '_>) -> Result<(), String> + Send + Sync + 'static,
) -> NodeDescriptor {
    NodeDescriptor::new(&kind(name), title, category, description, build)
        .with_param(ParamDesc::boolean("enabled", "Enabled", true))
        .with_generated_pins()
}

fn missing(node: &str, needs: &str) -> String {
    format!("{node} needs the {needs} node to be built before it")
}

/// Colour resource a post chain node reads, by the name its `color_input`
/// parameter holds.
fn colour_resource(name: &str) -> &'static str {
    match name {
        "tsr_color" => "tsr_color",
        "fogged_hdr" => FOGGED_HDR,
        "lens_color" => LENS_COLOR,
        "pre_aa" => "pre_aa",
        _ => "fxaa_color",
    }
}

fn tsr_quality(name: &str) -> helio::TsrQuality {
    match name {
        "performance" => helio::TsrQuality::Performance,
        "quality" => helio::TsrQuality::Quality,
        "native" => helio::TsrQuality::Native,
        _ => helio::TsrQuality::Balanced,
    }
}

// ── Registration ─────────────────────────────────────────────────────────────

/// Registers every pass node and the host inputs, and the performance marker.
pub fn register_passes(registry: &mut NodeRegistry) {
    register_scene_prep(registry);
    register_geometry(registry);
    register_lighting(registry);
    register_overlays(registry);
    register_post(registry);
    register_output(registry);

    for name in HOST_INPUTS {
        let name: &'static str = name;
        registry.register(
            NodeDescriptor::new(
                &host_kind(name),
                &format!("Host::{name}"),
                "Host",
                "A resource the renderer supplies every frame.",
                move |build| {
                    build.graph.declare_external_input(intern(name));
                    Ok(())
                },
            )
            .with_data_only()
            .with_generated_pins(),
        );
    }

    // After a node that ends a phase, the profiler's marker times it. The graph
    // does not contain them; the compiler adds them.
    let deferred_light = kind("deferred_light");
    let marker: MarkerFn = Arc::new(move |build: &mut NodeBuild<'_, '_>, node_kind: &str| {
        let perf = perf(build.env);
        if node_kind == deferred_light {
            build
                .graph
                .add_pass(Box::new(PerfOverlayCostAnalyzerPass::new(perf.clone())));
        }
        build
            .graph
            .add_pass(Box::new(PerfOverlayAnalyzerPass::new(perf)));
    });
    registry.set_marker(marker);
}

fn register_scene_prep(registry: &mut NodeRegistry) {
    const CATEGORY: &str = "Passes: Scene Prep";

    registry.register(pass_node(
        "object_batch",
        "Scene: Object Batches",
        CATEGORY,
        "Gathers the scene's objects into GPU-driven batches every other pass reads.",
        |build| {
            let c = ctx(build.env);
            build
                .graph
                .add_pass(Box::new(ObjectBatchPass::new(c.device)));
            Ok(())
        },
    ));

    registry.register(pass_node(
        "atmosphere",
        "Sky: Atmosphere Tables",
        CATEGORY,
        "Atmosphere lookup tables and the resolved sky frame. Idle without an atmosphere component.",
        |build| {
            let c = ctx(build.env);
            build.graph.add_pass(Box::new(AtmospherePass::new(c.device)));
            Ok(())
        },
    ));

    registry.register(pass_node(
        "shadow_matrix",
        "Shadows: Light Matrices",
        CATEGORY,
        "Computes the view-projection of every shadow-casting light face.",
        |build| {
            let c = ctx(build.env);
            let lights = scene_buffer_or_dummy(
                &c.scene_db,
                c.device,
                pulsar_scenedb::gpu::BufferKey::of("scene_lights"),
                "SceneDB Lights",
                96,
            );
            let budget = c
                .config
                .shadow_budget
                .validate()
                .map_err(|_| "invalid renderer shadow budget".to_string())?;
            let atlas_size = budget.atlas_size(c.device.limits().max_texture_dimension_2d);
            let matrices = c.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Shadow Matrices"),
                size: helio_pass_shadow_matrix::MAX_SHADOW_FACES as u64
                    * std::mem::size_of::<helio_pass_shadow_matrix::GpuShadowMatrix>() as u64,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });
            let dirty = Arc::new(c.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Shadow Dirty Flags"),
                size: helio_pass_shadow_matrix::MAX_SHADOW_CASTERS as u64 * 4,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
            let hashes = c.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Shadow Hashes"),
                size: helio_pass_shadow_matrix::MAX_SHADOW_CASTERS as u64 * 4,
                usage: wgpu::BufferUsages::STORAGE,
                mapped_at_creation: false,
            });
            build.graph.add_pass(Box::new(
                ShadowMatrixPass::new(
                    c.device,
                    &lights.buffer,
                    &matrices,
                    c.camera,
                    &dirty,
                    &hashes,
                    atlas_size,
                )
                .with_budget(budget, c.ih),
            ));
            build.env.shared(|| ShadowDirtyBuffer(dirty));
            Ok(())
        },
    ));

    registry.register(pass_node(
        "shadow_dirty",
        "Shadows: Find Stale Faces",
        CATEGORY,
        "Marks which shadow faces need re-rendering this frame.",
        |build| {
            let c = ctx(build.env);
            let dirty = build
                .env
                .shared_existing::<ShadowDirtyBuffer>()
                .map(|shared| Arc::clone(&shared.0))
                .ok_or_else(|| missing("Shadow Dirty Flags", "Shadow Matrices"))?;
            build
                .graph
                .add_pass(Box::new(ShadowDirtyPass::new(c.device, dirty)));
            Ok(())
        },
    ));

    registry.register(pass_node(
        "shadow",
        "Shadows: Render Atlas",
        CATEGORY,
        "Renders the dirty shadow faces into the tiled shadow atlas.",
        |build| {
            let c = ctx(build.env);
            let budget = c
                .config
                .shadow_budget
                .validate()
                .map_err(|_| "invalid renderer shadow budget".to_string())?;
            let atlas_size = budget.atlas_size(c.device.limits().max_texture_dimension_2d);
            let (face_dirty, face_geom_count, face_pages) = build
                .graph
                .find_pass::<ShadowDirtyPass>()
                .map(|dirty| {
                    (
                        Arc::clone(&dirty.face_dirty_buf),
                        Arc::clone(&dirty.face_geom_count_buf),
                        Arc::clone(&dirty.face_pages_buf),
                    )
                })
                .ok_or_else(|| missing("Shadow Maps", "Shadow Dirty Flags"))?;
            build.graph.add_pass(Box::new(ShadowPass::new_tiled(
                c.device,
                c.queue,
                face_dirty,
                face_geom_count,
                face_pages,
                atlas_size,
            )));
            Ok(())
        },
    ));

    registry.register(pass_node(
        "background",
        "Scene: Clear Background",
        CATEGORY,
        "Clears the scene colour to the background the sky and lighting draw over.",
        |build| {
            let c = ctx(build.env);
            build.graph.add_pass(Box::new(BackgroundPass::new(
                c.config.surface_format,
                c.config.background_color,
            )));
            Ok(())
        },
    ));

    registry.register(pass_node(
        "indirect_dispatch",
        "Culling: Prepare Dispatch",
        CATEGORY,
        "Prepares the indirect dispatch arguments the culling passes use.",
        |build| {
            let c = ctx(build.env);
            build.graph.add_pass(Box::new(IndirectDispatchPass::new(
                c.device,
                c.cull_stats.clone(),
            )));
            Ok(())
        },
    ));

    registry.register(pass_node(
        "hiz",
        "Culling: Depth Pyramid (Hi-Z)",
        CATEGORY,
        "Builds the hierarchical depth pyramid occlusion culling tests against.",
        |build| {
            let c = ctx(build.env);
            build
                .graph
                .add_pass(Box::new(HiZBuildPass::new(c.device, c.queue, c.iw, c.ih)));
            Ok(())
        },
    ));

    registry.register(pass_node(
        "occlusion_cull",
        "Culling: Frustum + Occlusion",
        CATEGORY,
        "Frustum and Hi-Z occlusion culling of the object batch.",
        |build| {
            let c = ctx(build.env);
            let sampler = build
                .graph
                .find_pass::<HiZBuildPass>()
                .map(|hiz| Arc::clone(&hiz.hiz_sampler))
                .ok_or_else(|| missing("Occlusion Cull", "Hi-Z Build"))?;
            build.graph.add_pass(Box::new(OcclusionCullPass::new(
                c.device,
                sampler,
                c.iw,
                c.ih,
                c.cull_stats.clone(),
            )));
            Ok(())
        },
    ));

    registry.register(pass_node(
        "portal_cull",
        "Portals: Cull",
        CATEGORY,
        "Culls portals and compacts the ones to draw. Portal Instance reads its buffers.",
        |build| {
            let c = ctx(build.env);
            build
                .graph
                .add_pass(Box::new(PortalCullPass::new(c.device)));
            Ok(())
        },
    )
    .with_marker_after());

    registry.register(pass_node(
        "light_cull",
        "Lighting: Tile Light Binning",
        CATEGORY,
        "Bins the scene's lights into screen tiles.",
        |build| {
            let c = ctx(build.env);
            build
                .graph
                .add_pass(Box::new(LightCullPass::new(c.device, c.iw, c.ih)));
            Ok(())
        },
    ));
}

fn register_geometry(registry: &mut NodeRegistry) {
    const CATEGORY: &str = "Passes: Geometry";

    registry.register(pass_node(
        "foliage_place",
        "Foliage: Place Blades",
        CATEGORY,
        "Places foliage blades on the GPU. Must come before the GBuffer so subpass fusion holds.",
        |build| {
            let c = ctx(build.env);
            // The draw shares placement's buffers and its boxes run before the
            // G-buffer: made here, the draw waits for the Foliage GBuffer node.
            let (place, boxes, draw) = crate::foliage_passes(c.device, &c.config);
            build.graph.add_pass(Box::new(place));
            build.graph.add_pass(Box::new(boxes));
            build.env.shared(|| PendingFoliageDraw(None)).0 = Some(draw);
            Ok(())
        },
    ));

    registry.register(pass_node(
        "gbuffer",
        "Geometry: GBuffer",
        CATEGORY,
        "Opaque geometry into the GBuffer and the scene depth.",
        |build| {
            let c = ctx(build.env);
            build.graph.add_pass(Box::new(GBufferPass::new(c.device)));
            Ok(())
        },
    ));

    registry.register(pass_node(
        "portal_mask",
        "Portals: Footprint Mask",
        CATEGORY,
        "Stamps each portal's visible footprint, respecting real occluders.",
        |build| {
            let c = ctx(build.env);
            build
                .graph
                .add_pass(Box::new(PortalMaskPass::new(c.device)));
            Ok(())
        },
    ));

    registry.register(pass_node(
        "portal_instance",
        "Portals: Draw Through",
        CATEGORY,
        "Draws the geometry seen through portals. Needs Portal Cull before it.",
        |build| {
            let c = ctx(build.env);
            let (indirect, compacted, chains, segments) = build
                .graph
                .find_pass::<PortalCullPass>()
                .map(|cull| {
                    (
                        Arc::clone(&cull.portal_indirect_buf),
                        Arc::clone(&cull.portal_compacted_indices_buf),
                        Arc::clone(&cull.portal_compacted_chains_buf),
                        Arc::clone(&cull.segments),
                    )
                })
                .ok_or_else(|| missing("Portal Instances", "Portal Cull"))?;
            build.graph.add_pass(Box::new(PortalInstancePass::new(
                c.device, indirect, compacted, chains, segments,
            )));
            Ok(())
        },
    ));

    registry.register(
        pass_node(
            "foliage_gbuffer",
            "Foliage: Draw to GBuffer",
            CATEGORY,
            "Rasterises placed foliage into the GBuffer. Needs Foliage Placement before it.",
            |build| {
                let c = ctx(build.env);
                let draw = build
                    .env
                    .shared_existing::<PendingFoliageDraw>()
                    .and_then(|pending| pending.0.take())
                    .ok_or_else(|| missing("Foliage GBuffer", "Foliage Placement"))?;
                build.graph.add_pass(Box::new(draw));
                Ok(())
            },
        )
        .with_marker_after(),
    );

    registry.register(
        pass_node(
            "forward_lit",
            "Geometry: Forward Lit",
            CATEGORY,
            "Lit opaque geometry in one forward pass, for pipelines without a GBuffer.",
            |build| {
                let c = ctx(build.env);
                let render_all_opaque = build.boolean("render_all_opaque");
                let mut pass = ForwardLitPass::new(c.device, c.config.surface_format);
                pass.render_all_opaque = render_all_opaque;
                build.graph.add_pass(Box::new(pass));
                Ok(())
            },
        )
        .with_param(ParamDesc::boolean(
            "render_all_opaque",
            "Render all opaque",
            false,
        )),
    );
}

fn register_lighting(registry: &mut NodeRegistry) {
    const CATEGORY: &str = "Passes: Lighting";

    registry.register(pass_node(
        "decal",
        "Geometry: Decals",
        CATEGORY,
        "Projects decals into the GBuffer.",
        |build| {
            let c = ctx(build.env);
            build.graph.add_pass(Box::new(DecalPass::new(
                c.device, c.queue, c.camera, c.iw, c.ih,
            )));
            Ok(())
        },
    ));

    const MISSING_REFLECTIONS: &str = "reflections are not supported on this platform";

    registry.register(
        pass_node(
            "ssr",
            "Reflections: Screen-Space",
            CATEGORY,
            "Traces glossy reflections in screen space. One of the most expensive passes.",
            |build| {
                let c = ctx(build.env);
                if !helio_core::REFLECTIONS_SUPPORTED {
                    log::warn!("Screen-space reflections are not supported on this platform");
                    return Ok(());
                }
                build.graph.add_pass(Box::new(SsrPass::new(
                    c.device, c.queue, c.camera, c.iw, c.ih,
                )));
                Ok(())
            },
        )
        .with_unsupported_when(helio_core::REFLECTIONS_SUPPORTED, MISSING_REFLECTIONS),
    );

    registry.register(
        pass_node(
            "planar_reflection",
            "Reflections: Planar",
            CATEGORY,
            "Reflects the scene across world-space planes.",
            |build| {
                let c = ctx(build.env);
                if !helio_core::REFLECTIONS_SUPPORTED {
                    log::warn!("Planar reflections are not supported on this platform");
                    return Ok(());
                }
                build.graph.add_pass(Box::new(PlanarReflectionPass::new(
                    c.device,
                    c.camera,
                    c.config.surface_format,
                )));
                Ok(())
            },
        )
        .with_unsupported_when(helio_core::REFLECTIONS_SUPPORTED, MISSING_REFLECTIONS),
    );

    registry.register(
        pass_node(
            "deferred_light",
            "Lighting: Deferred",
            CATEGORY,
            "Lights the GBuffer with every light, shadow, reflection and the atmosphere.",
            |build| {
                let c = ctx(build.env);
                let mut pass =
                    DeferredLightPass::new(c.device, c.queue, c.camera, FOGGED_HDR_FORMAT);
                pass.set_shadow_quality(c.config.shadow_quality, c.queue);
                pass.debug_mode = c.config.debug_mode;
                pass.set_env_reflections(c.config.enable_environment_reflections);
                build.graph.add_pass(Box::new(pass));
                Ok(())
            },
        )
        .with_marker_after(),
    );
}

fn register_overlays(registry: &mut NodeRegistry) {
    const CATEGORY: &str = "Passes: Overlays";

    registry.register(pass_node(
        "atmosphere_composite",
        "Sky: Composite",
        CATEGORY,
        "Composites the sky and aerial perspective over the lit scene.",
        |build| {
            let c = ctx(build.env);
            build.graph.add_pass(Box::new(AtmosphereCompositePass::new(
                c.device,
                FOGGED_HDR_FORMAT,
            )));
            Ok(())
        },
    ));

    registry.register(
        pass_node(
            "billboard",
            "Overlay: Billboards",
            CATEGORY,
            "Camera-facing sprites, depth-tested against the scene.",
            |build| {
                let c = ctx(build.env);
                let spotlight = image::load_from_memory(SPOTLIGHT_PNG)
                    .unwrap_or_else(|_| image::DynamicImage::new_rgba8(1, 1))
                    .into_rgba8();
                let (width, height) = spotlight.dimensions();
                let mut pass = BillboardPass::new_with_sprite_rgba(
                    c.device,
                    c.queue,
                    c.camera,
                    FOGGED_HDR_FORMAT,
                    spotlight.as_raw(),
                    width,
                    height,
                );
                pass.set_occluded_by_geometry(true);
                build.graph.add_pass(Box::new(pass));
                Ok(())
            },
        )
        .with_marker_after(),
    );

    registry.register(
        pass_node(
            "corona",
            "Overlay: Light Coronas",
            CATEGORY,
            "Light coronas.",
            |build| {
                let c = ctx(build.env);
                build.graph.add_pass(Box::new(CoronaPass::new(
                    c.device,
                    c.queue,
                    c.camera,
                    FOGGED_HDR_FORMAT,
                )));
                Ok(())
            },
        )
        .with_marker_after(),
    );

    registry.register(
        pass_node(
            "portal_editor_overlay",
            "Overlay: Portal Openings (Editor)",
            CATEGORY,
            "Editor-only checkerboard over each portal's opening.",
            |build| {
                let c = ctx(build.env);
                build.graph.add_pass(Box::new(PortalEditorOverlayPass::new(
                    c.device,
                    FOGGED_HDR_FORMAT,
                )));
                Ok(())
            },
        )
        .with_marker_after(),
    );

    registry.register(
        pass_node(
            "water_sim",
            "Water: Simulate + Draw",
            CATEGORY,
            "Simulates and draws water volumes.",
            |build| {
                let c = ctx(build.env);
                build.graph.add_pass(Box::new(WaterSimPass::new(
                    c.device,
                    c.camera,
                    c.iw,
                    c.ih,
                    FOGGED_HDR_FORMAT,
                )));
                Ok(())
            },
        )
        .with_marker_after(),
    );

    registry.register(pass_node(
        "debug_draw_scene",
        "Overlay: Grid + Volume Bounds (Editor)",
        CATEGORY,
        "Editor grid and scene volume bounds, depth-tested against the scene.",
        |build| {
            let c = ctx(build.env);
            build.graph.add_pass(Box::new(helio::DebugDrawPass::new(
                c.device,
                c.debug_camera,
                FOGGED_HDR_FORMAT,
                c.debug_state.clone(),
                true,
                true,
            )));
            Ok(())
        },
    ));
}

fn register_post(registry: &mut NodeRegistry) {
    const CATEGORY: &str = "Passes: Post";

    registry.register(pass_node(
        "post_process_volume_blend",
        "Post: Blend Volumes",
        CATEGORY,
        "Blends the post-process volumes around the camera.",
        |build| {
            let c = ctx(build.env);
            build
                .graph
                .add_pass(Box::new(PostProcessVolumeBlendPass::new(c.device)));
            Ok(())
        },
    ));

    registry.register(pass_node(
        "volumetric_fog",
        "Fog: Volumetric",
        CATEGORY,
        "Accumulates volumetric fog lit by the shadowed lights.",
        |build| {
            let c = ctx(build.env);
            build
                .graph
                .add_pass(Box::new(VolumetricFogPass::new(c.device)));
            Ok(())
        },
    ));

    registry.register(pass_node(
        "fog_composite",
        "Fog: Composite",
        CATEGORY,
        "Composites the fog over the scene into the fogged HDR colour.",
        |build| {
            let c = ctx(build.env);
            build.graph.add_pass(Box::new(FogCompositePass::with_format(
                c.device,
                FOGGED_HDR_FORMAT,
            )));
            Ok(())
        },
    ));

    registry.register(pass_node(
        "transparent",
        "Transparency: Blend",
        CATEGORY,
        "Alpha-blended geometry over the fogged colour.",
        |build| {
            let c = ctx(build.env);
            build.graph.add_pass(Box::new(
                helio_pass_transparent::TransparentPass::new(c.device, FOGGED_HDR_FORMAT)
                    .with_fogged_target(),
            ));
            Ok(())
        },
    ));

    registry.register(
        pass_node(
            "tsr",
            "Anti-Aliasing: TSR",
            CATEGORY,
            "Temporal anti-aliasing and upscaling. The renderer's internal resolution follows the project's TSR setting.",
            |build| {
                let c = ctx(build.env);
                let quality = tsr_quality(&build.text("quality"));
                let tsr = TsrPass::new(
                    c.device,
                    c.iw,
                    c.ih,
                    c.config.width,
                    c.config.height,
                    c.config.surface_format,
                    quality,
                )
                .with_color_input(FOGGED_HDR)
                .with_intermediate_output()
                .with_motion_vectors();
                let tsr = if build.boolean("stochastic_mask") {
                    tsr.with_stochastic_mask()
                } else {
                    tsr
                };
                build.graph.add_pass(Box::new(tsr));
                Ok(())
            },
        )
        .with_param(ParamDesc::choice(
            "quality",
            "Quality",
            &["performance", "balanced", "quality", "native"],
            "balanced",
        ))
        .with_param(ParamDesc::boolean(
            "stochastic_mask",
            "Stochastic mask",
            false,
        )),
    );

    registry.register(pass_node(
        "fxaa",
        "Anti-Aliasing: FXAA",
        CATEGORY,
        "Fast approximate anti-aliasing of the fogged colour.",
        |build| {
            let c = ctx(build.env);
            build.graph.add_pass(Box::new(
                FxaaPass::new(c.device, FOGGED_HDR_FORMAT)
                    .with_color_input(FOGGED_HDR)
                    .with_intermediate_target(FOGGED_HDR_FORMAT),
            ));
            Ok(())
        },
    ));

    registry.register(
        pass_node(
            "lens",
            "Lens: Elements",
            CATEGORY,
            "Lens elements: water drops, iris shapes and user lenses.",
            |build| {
                let c = ctx(build.env);
                let input = colour_resource(&build.text("color_input"));
                build.graph.add_pass(Box::new(
                    LensPass::new_with_user(c.device, c.iw, c.ih, None)
                        .with_color_input(input)
                        .with_bundled_resources(),
                ));
                Ok(())
            },
        )
        .with_param(ParamDesc::choice(
            "color_input",
            "Colour input",
            &["fxaa_color", "tsr_color", "fogged_hdr"],
            "fxaa_color",
        ))
        .with_alternative_inputs(&["tsr_color", "fxaa_color", FOGGED_HDR], PinType::TEXTURE),
    );

    registry.register(pass_node(
        "lens_flare",
        "Lens: Flares",
        CATEGORY,
        "Lens flares from the scene's bright lights.",
        |build| {
            let c = ctx(build.env);
            build.graph.add_pass(Box::new(
                LensFlarePass::new_hdr(c.device, c.iw, c.ih).with_color_input(LENS_COLOR),
            ));
            Ok(())
        },
    ));

    registry.register(
        pass_node(
            "post_process",
            "Post: Tone Map + Grade",
            CATEGORY,
            "Bloom, exposure, tone mapping and colour grading into the output.",
            |build| {
                let c = ctx(build.env);
                let input = colour_resource(&build.text("color_input"));
                build.graph.add_pass(Box::new(
                    PostProcessPass::new_with_user_effects(
                        c.device,
                        c.queue,
                        c.config.width,
                        c.config.height,
                        c.config.surface_format,
                        None,
                    )
                    .with_color_input(input),
                ));
                Ok(())
            },
        )
        .with_param(ParamDesc::choice(
            "color_input",
            "Colour input",
            &["lens_color", "fxaa_color", "tsr_color", "fogged_hdr"],
            "lens_color",
        ))
        .with_alternative_inputs(
            &[LENS_COLOR, "tsr_color", "fxaa_color", FOGGED_HDR],
            PinType::TEXTURE,
        ),
    );
}

fn register_output(registry: &mut NodeRegistry) {
    const CATEGORY: &str = "Passes: Output";

    registry.register(pass_node(
        "buffer_view",
        "Output: Debug Buffer View",
        CATEGORY,
        "Depth and normal debug views, over the final image. Inert unless one is selected.",
        |build| {
            let c = ctx(build.env);
            build
                .graph
                .add_pass(Box::new(helio_pass_buffer_view::BufferViewPass::new(
                    c.device,
                    c.config.surface_format,
                )));
            Ok(())
        },
    ));

    // The culler and the batcher hand each other buffers and are one unit.
    registry.register(
        pass_node(
            "sprite_overlay",
            "Output: Sprites",
            CATEGORY,
            "Game sprites over the final image: GPU culling and batching.",
            |build| {
                let c = ctx(build.env);
                // The pass draws over the full-resolution image.
                let half = [c.config.width as f32 * 0.5, c.config.height as f32 * 0.5];
                let mut batch = SpriteBatchPass::scene_overlay(
                    c.device,
                    c.queue,
                    c.config.surface_format,
                    helio_mats::MaterialBindingConfig::for_device(c.device),
                );
                batch.set_camera([0.0, 0.0], Some(half));
                let mut cull = SpriteCullPass::new(
                    c.device,
                    c.queue,
                    batch.instances_buffer(),
                    batch.alive_buffer(),
                    1,
                    MAX_VISIBLE_SPRITES,
                )
                .skip_while_empty();
                cull.set_view_rect([0.0, 0.0], half);
                batch.use_gpu_culling(cull.draw_order_buf.clone(), cull.indirect_buf.clone());
                build.graph.add_pass(Box::new(cull));
                build.graph.add_pass(Box::new(batch));
                Ok(())
            },
        )
        .with_marker_after(),
    );

    registry.register(pass_node(
        "perf_overlay",
        "Output: Performance Overlay",
        CATEGORY,
        "Draws the performance overlay over the final image.",
        |build| {
            let c = ctx(build.env);
            let shared = perf(build.env);
            let mut pass = PerfOverlayPass::new(c.device, shared, c.config.surface_format);
            pass.set_mode(crate::convert_perf_mode(c.config.perf_overlay_mode));
            build.graph.add_pass(Box::new(pass));
            Ok(())
        },
    ));

    registry.register(pass_node(
        "debug_draw",
        "Output: Debug Lines",
        CATEGORY,
        "User debug lines and triangles over the final image.",
        |build| {
            let c = ctx(build.env);
            build.graph.add_pass(Box::new(helio::DebugDrawPass::new(
                c.device,
                c.debug_camera,
                c.config.surface_format,
                c.debug_state.clone(),
                false,
                false,
            )));
            Ok(())
        },
    ));
}
