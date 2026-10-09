use std::sync::Arc;

mod background;
pub mod environment_join;
pub mod ray_tracing;
pub mod scene_join;

use helio::DebugDrawState;
use helio::GraphRebuilder;
use helio::PassBuildContext;
use helio::RendererConfig;
use helio_pass_billboard::BillboardPass;
use helio_pass_corona::CoronaPass;
use helio_pass_debug_overlay::{DebugOverlayPass, DebugOverlayState};
use helio_pass_decal::DecalPass;
use helio_pass_deferred_light::DeferredLightPass;
use helio_pass_dof::DofPass;
use helio_pass_flare::LensFlarePass;
use helio_pass_foliage_gbuffer::FoliageGBufferPass;
use helio_pass_foliage_place::FoliagePlacePass;
use helio_pass_foliage_place::FoliageQuality;
use helio_pass_forward_lit::ForwardLitPass;
use helio_pass_fxaa::FxaaPass;
use helio_pass_gbuffer::GBufferPass;
use helio_pass_hiz::HiZBuildPass;
use helio_pass_hlfs::HlfsPass;
use helio_pass_indirect_dispatch::IndirectDispatchPass;
use helio_pass_light_cull::LightCullPass;
use helio_pass_object_batch::ObjectBatchPass;
use helio_pass_occlusion_cull::OcclusionCullPass;
use helio_pass_perf_overlay::{
    PerfOverlayAnalyzerPass, PerfOverlayCostAnalyzerPass, PerfOverlayPass, PerfOverlayShared,
};
use helio_pass_planar_reflection::PlanarReflectionPass;
use helio_pass_portal_cull::PortalCullPass;
use helio_pass_portal_instances::{PortalEditorOverlayPass, PortalInstancePass, PortalMaskPass};
use helio_pass_postprocess::{
    FogCompositePass, PostProcessPass, PostProcessVolumeBlendPass, FOGGED_HDR, FOGGED_HDR_FORMAT,
};
use helio_pass_shadow::ShadowPass;
use helio_pass_shadow_cull::ShadowCullPass;
use helio_pass_shadow_dirty::ShadowDirtyPass;
use helio_pass_shadow_matrix::ShadowMatrixPass;
use helio_pass_simple_cube::SimpleCubePass;
use helio_pass_sky::{AtmosphereCompositePass, AtmospherePass};

use background::BackgroundPass;
use helio_pass_ssr::SsrPass;
use helio_pass_tsr::TsrPass;
use helio_pass_sprite_batch::SpriteBatchPass;
use helio_pass_sprite_cull::SpriteCullPass;
use helio_pass_virtual_geometry::VirtualGeometryPass;
use helio_pass_volumetric_fog::VolumetricFogPass;
use helio_pass_water_sim::WaterSimPass;

use helio_core::RenderGraph;

/// An application-provided pass factory reused on graph resize. The builder
/// chooses its stage; scene data and backend selection remain with the caller.
pub type GraphPassFactory = Arc<
    dyn Fn(&wgpu::Device, &wgpu::Queue, u32, u32) -> Box<dyn helio_core::RenderPass>
        + Send
        + Sync,
>;

/// Factory for format-independent voxel passes in the GBuffer stage.
pub type VoxelPassFactory = GraphPassFactory;

/// Spotlight icon embedded at compile time — used as the editor billboard sprite.
static SPOTLIGHT_PNG: &[u8] = include_bytes!("../../../spotlight.png");

fn scene_buffer_or_dummy(
    scene_db: &helio::SceneDbHandle,
    device: &wgpu::Device,
    key: pulsar_scenedb::gpu::BufferKey,
    label: &str,
    fallback_size: u64,
) -> pulsar_scenedb::gpu::BufferHandle {
    scene_db.store().resolve_buffer_handle(key).unwrap_or_else(|| {
        pulsar_scenedb::gpu::BufferHandle {
            buffer: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: fallback_size.max(4),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            epoch: 0,
            row_bytes: 0,
            content_generation: 0,
        }
    })
}

/// Create a new graph, honouring the caller's device ownership.
///
/// When `config.enable_xr` the graph is put into OpenXR multiview mode:
/// every pool texture is allocated as a 2-layer array and the executor forces
/// `multiview_mask = 0b11` on all render passes. Note that the graph's internal
/// resolution stays `config.internal_width()/internal_height()` — in XR mode
/// the application is expected to size `RendererConfig` to the eye resolution
/// reported by the OpenXR runtime (via `XrSession::width`/`height`), since the
/// graph does not talk to the runtime itself.
fn new_graph(
    device: &Arc<wgpu::Device>,
    queue: &Arc<wgpu::Queue>,
    owns_device: bool,
    config: &RendererConfig,
) -> RenderGraph {
    let mut graph = if owns_device {
        RenderGraph::new(device, queue)
    } else {
        RenderGraph::new_with_external_device(device, queue)
    };
    graph.with_xr_mode(config.enable_xr);
    graph
}

/// Registers the resource names that every full (non-`simple`) default graph
/// relies on the host `Renderer` to supply directly into `ResourceRegistry`
/// every frame, rather than any pass in the graph writing them — see
/// `RenderGraph::declare_external_input` and `docs/helio_3_0_spec.md`
/// §6. `"material_textures"` and `"render_environment"` are read by the
/// geometry/lighting passes that need those backend values;
/// `"vg"` is read only by `VirtualGeometryPass` (`add_geometry_passes`/
/// `add_forward_geometry_passes`); `"billboards"`/`"corona_emitters"` are
/// read only by `BillboardPass`/`CoronaPass` (`add_late_passes`). Every
/// graph builder in this file that calls one of those three helpers needs
/// all four names; `build_simple_graph` uses none of this and is the one
/// entry point that legitimately calls none of these.
fn declare_common_external_inputs(graph: &mut RenderGraph) {
    graph.declare_external_input("material_textures");
    graph.declare_external_input("render_environment");
    graph.declare_external_input("vg");
    graph.declare_external_input("billboards");
    graph.declare_external_input("corona_emitters");
}

/// Which of this crate's passes shader hot reload may swap on their own
/// (see [`helio_core::SwapPolicy`] for what a swap does and why a policy is
/// needed).
///
/// Derived by auditing every constructor call in this file for handles the
/// builder creates and shares between passes. Those passes form a group and
/// are swapped together:
///
/// * shadows: `ShadowMatrixPass` + `ShadowDirtyPass` (dirty-flag buffer),
///   `ShadowDirtyPass` + `ShadowCullPass` + `ShadowPass` (face dirty, geometry
///   count, indirect and count buffers)
/// * `HiZBuildPass` + `OcclusionCullPass` (Hi-Z sampler)
/// * `PortalCullPass` + `PortalInstancePass` (portal output buffers)
/// * `FoliagePlacePass` + `FoliageGBufferPass` (blade arena, tile table,
///   visible blades, indirect buffer)
/// * every perf-overlay pass (one shared `PerfOverlayShared`)
/// * `SpriteCullPass` + `SpriteBatchPass` (draw order and indirect buffers)
///
/// Everything else listed as independent takes only renderer-owned handles
/// (camera, debug camera and cull-stats buffers, debug-draw state, SceneDB,
/// the debug-overlay state) and plain configuration; those are the same objects
/// in the live and the replacement graph. `GBufferPass`, `ForwardLitPass`
/// and `TransparentPass` are left out on purpose: they build part of their
/// WGSL at runtime and keep material/template registrations, so edits to them
/// rebuild the whole graph. So are passes from the application's voxel,
/// lighting and final-pass factories, which this crate cannot vouch for.
///
/// Dropping the replacement's unswapped passes has no side effects beyond the
/// ones a resize already has: none of these constructors spawns a thread,
/// registers into a global or shared registry, or writes a file, and the
/// replacement graph's pipeline cache has no persistence path.
fn default_swap_policy() -> helio_core::SwapPolicy {
    use helio_core::graph::type_name_of as name;
    helio_core::SwapPolicy::new()
        .independent::<ObjectBatchPass>()
        .independent::<IndirectDispatchPass>()
        .independent::<BackgroundPass>()
        .independent::<AtmospherePass>()
        .independent::<AtmosphereCompositePass>()
        .independent::<LightCullPass>()
        .independent::<DecalPass>()
        .independent::<SsrPass>()
        .independent::<helio_pass_ssr::SsrCompositePass>()
        .independent::<PlanarReflectionPass>()
        .independent::<DeferredLightPass>()
        .independent::<HlfsPass>()
        .independent::<VirtualGeometryPass>()
        .independent::<PortalMaskPass>()
        .independent::<PortalEditorOverlayPass>()
        .independent::<BillboardPass>()
        .independent::<CoronaPass>()
        .independent::<WaterSimPass>()
        .independent::<helio::DebugDrawPass>()
        .independent::<DebugOverlayPass>()
        .independent::<PostProcessVolumeBlendPass>()
        .independent::<VolumetricFogPass>()
        .independent::<FogCompositePass>()
        .independent::<TsrPass>()
        .independent::<FxaaPass>()
        .independent::<LensFlarePass>()
        .independent::<PostProcessPass>()
        .independent::<DofPass>()
        .group_types(&[
            name::<ShadowMatrixPass>(),
            name::<ShadowDirtyPass>(),
            name::<ShadowCullPass>(),
            name::<ShadowPass>(),
        ])
        .group_types(&[name::<HiZBuildPass>(), name::<OcclusionCullPass>()])
        .group_types(&[name::<PortalCullPass>(), name::<PortalInstancePass>()])
        .group_types(&[name::<FoliagePlacePass>(), name::<FoliageGBufferPass>()])
        .group_types(&[
            name::<PerfOverlayAnalyzerPass>(),
            name::<PerfOverlayCostAnalyzerPass>(),
            name::<PerfOverlayPass>(),
        ])
        .group_types(&[name::<SpriteCullPass>(), name::<SpriteBatchPass>()])
}

/// Most sprites the 2D overlay draws at once.
const MAX_VISIBLE_SPRITES: u32 = 16_384;

/// The scene's 2D sprites (Pulsar-Native#1060), drawn over the final image
/// after post-processing: culled and sorted by their Z index on the GPU,
/// then drawn by their own orthographic camera (1 unit = 1 output pixel,
/// origin at the centre), so the 3D camera does not move them. Both passes record
/// nothing while the scene holds no sprite.
fn add_sprite_overlay_passes(
    graph: &mut RenderGraph,
    device: &Arc<wgpu::Device>,
    queue: &Arc<wgpu::Queue>,
    config: &RendererConfig,
) {
    // The output's pixels: the graph prepares passes at the internal
    // resolution, but the overlay draws over the full-resolution image.
    let half = [config.width as f32 * 0.5, config.height as f32 * 0.5];
    let mut batch = SpriteBatchPass::scene_overlay(
        device,
        queue,
        config.surface_format,
        helio_mats::MaterialBindingConfig::for_device(device),
    );
    batch.set_camera([0.0, 0.0], Some(half));
    let mut cull = SpriteCullPass::new(
        device,
        queue,
        batch.instances_buffer(),
        batch.alive_buffer(),
        1,
        MAX_VISIBLE_SPRITES,
    )
    .skip_while_empty();
    cull.set_view_rect([0.0, 0.0], half);
    batch.use_gpu_culling(cull.draw_order_buf.clone(), cull.indirect_buf.clone());
    graph.add_pass(Box::new(cull));
    graph.add_pass(Box::new(batch));
}

fn add_common_early_passes(
    graph: &mut RenderGraph,
    device: &Arc<wgpu::Device>,
    queue: &Arc<wgpu::Queue>,
    camera_buf: &wgpu::Buffer,
    config: &RendererConfig,
    cull_stats_buf: &wgpu::Buffer,
    w: u32,
    h: u32,
    scene_db: helio::SceneDbHandle,
) -> Arc<std::sync::Mutex<PerfOverlayShared>> {
    let lights_buf = scene_buffer_or_dummy(
        &scene_db,
        device,
        pulsar_scenedb::gpu::BufferKey::of("scene_lights"),
        "SceneDB Lights",
        96,
    );
    // Owned by the shadow-matrix pass, which computes and publishes it. One
    // matrix per atlas face, rounded up to whole 6-face caster slots. (This was
    // a SceneDB lookup of a key nothing registers: a 64-byte dummy that held
    // one matrix, with nothing publishing it, so no raster shadow rendered.)
    let shadow_face_slots = config.shadow_face_capacity.max(6).div_ceil(6) * 6;
    let shadow_matrices_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Shadow Matrices"),
        size: u64::from(shadow_face_slots)
            * std::mem::size_of::<helio_pass_shadow_matrix::GpuShadowMatrix>() as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    // Must run before every pass below — they all read `object_batch`
    // (instances/draw_calls/indirect/shadow partitions) published by this
    // pass from SceneDB's `StaticObjectComponent` rows. Ordering among
    // `add_pass` calls doesn't matter to the graph's own scheduler (it
    // topologically sorts on declared reads/writes), but this stays first
    // for readability, matching its role as the scene's sole GPU-driven
    // object-batch producer.
    graph.add_pass(Box::new(ObjectBatchPass::new(device)));
    // Atmosphere LUTs and the resolved frame (sun, planet, sky irradiance)
    // that lighting and the composite read; idle without an
    // `AtmosphereComponent` row.
    graph.add_pass(Box::new(AtmospherePass::new(device)));

    let hiz_pass = HiZBuildPass::new(device, queue, w, h);
    let hiz_sampler = Arc::clone(&hiz_pass.hiz_sampler);

    let shadow_dirty_buf = Arc::new(device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Shadow Dirty Flags"),
        size: 42 * 4,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    }));
    let shadow_hashes_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Shadow Hashes"),
        size: 42 * 4,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });

    graph.add_pass(Box::new(ShadowMatrixPass::new(
        device,
        &lights_buf.buffer,
        &shadow_matrices_buf,
        camera_buf,
        &shadow_dirty_buf,
        &shadow_hashes_buf,
        config.shadow_atlas_size,
    )));

    let shadow_dirty_pass = ShadowDirtyPass::new(device, Arc::clone(&shadow_dirty_buf));
    let face_dirty_buf = Arc::clone(&shadow_dirty_pass.face_dirty_buf);
    let face_geom_count_buf = Arc::clone(&shadow_dirty_pass.face_geom_count_buf);
    graph.add_pass(Box::new(shadow_dirty_pass));

    let shadow_cull_pass = ShadowCullPass::new(device, Arc::clone(&face_dirty_buf));
    let face_cull_indirect = Arc::clone(&shadow_cull_pass.face_indirect_buf);
    let face_cull_counts = Arc::clone(&shadow_cull_pass.face_counts_buf);
    graph.add_pass(Box::new(shadow_cull_pass));

    graph.add_pass(Box::new(ShadowPass::new(
        device,
        queue,
        face_dirty_buf,
        face_geom_count_buf,
        face_cull_indirect,
        face_cull_counts,
        config.shadow_atlas_size,
        config.shadow_face_capacity,
    )));

    // The black background the sky (the atmosphere composite) and every
    // lighting pass draw over.
    graph.add_pass(Box::new(BackgroundPass::new(config.surface_format)));

    graph.add_pass(Box::new(IndirectDispatchPass::new(
        device,
        cull_stats_buf.clone(),
    )));
    graph.add_pass(Box::new(hiz_pass));
    let occlusion_cull =
        OcclusionCullPass::new(device, hiz_sampler, w, h, cull_stats_buf.clone());
    // Static occlusion needs no wiring here: after a bake with a PVS,
    // BakeInjectPass publishes `baked_pvs` and the cull uploads it itself.
    graph.add_pass(Box::new(occlusion_cull));

    // Same phase as the frustum/occlusion cull above, not interleaved with
    // GBufferPass/PortalInstancePass later — a compute pass sitting between
    // two fused render passes silently breaks their attachment-based fusion
    // (see `add_geometry_passes`'s own comment on foliage placement for the
    // same reasoning). PortalInstancePass looks this pass back up via
    // `graph.find_pass::<PortalCullPass>()` to get its output buffers.
    if config.enable_portals {
        graph.add_pass(Box::new(PortalCullPass::new(device)));
    }

    let perf_overlay_shared = PerfOverlayShared::new(device, w, h);
    graph.add_pass(Box::new(PerfOverlayAnalyzerPass::new(Arc::clone(
        &perf_overlay_shared,
    ))));

    perf_overlay_shared
}

fn add_geometry_passes(
    graph: &mut RenderGraph,
    device: &Arc<wgpu::Device>,
    camera_buf: &wgpu::Buffer,
    config: &RendererConfig,
    perf: &Arc<std::sync::Mutex<PerfOverlayShared>>,
    _scene_db: helio::SceneDbHandle,
) {

    // Foliage placement is a compute pass and must be added *before* GBufferPass, not
    // between it and FoliageGBufferPass. It is deliberately not `chain_transparent` (it
    // records on the main encoder so it reads this frame's Hi-Z rather than last
    // frame's — plan §6.2), which means the chain scan cannot skip over it: sitting
    // between the two raster passes it would break the very subpass fusion
    // FoliageGBufferPass exists to join. Nothing renders wrong either way, which is
    // exactly why this is worth a comment — the cost is a silent tile store/reload.
    let foliage_buffers = config.enable_foliage.then(|| {
        let place_pass = FoliagePlacePass::new_with_density(
            device,
            FoliageQuality::default(),
            config.foliage_blades_per_m2,
        );
        let handles = (
            Arc::clone(&place_pass.blade_arena),
            Arc::clone(&place_pass.tile_table),
            Arc::clone(&place_pass.visible_blades),
            Arc::clone(&place_pass.foliage_indirect),
            place_pass.blades_per_tile(),
        );
        graph.add_pass(Box::new(place_pass));
        handles
    });

    graph.add_pass(Box::new(GBufferPass::new(device)));

    // Portal-duplicate draws go immediately after GBufferPass — before
    // foliage/VG, same reasoning as those two below: fusion requires an exact
    // attachment-view match, and both foliage and VG must stay last since VG
    // binds only 7 of 8 attachments. Looks the cull pass back up by type
    // rather than threading its buffers through this function's signature.
    //
    // PortalMaskPass runs first: it stamps each portal's true on-screen
    // footprint (respecting real occluders) into `portal_mask` and resets
    // depth to far there, so PortalInstancePass's own screen-space mask
    // check and depth self-occlusion both have something correct to test
    // against. See helio-pass-portal-instances' shaders for why this exists.
    if config.enable_portals {
        if let Some((indirect_buf, compacted_buf, compacted_chains_buf)) =
            graph.find_pass::<PortalCullPass>().map(|p| {
                (
                    Arc::clone(&p.portal_indirect_buf),
                    Arc::clone(&p.portal_compacted_indices_buf),
                    Arc::clone(&p.portal_compacted_chains_buf),
                )
            })
        {
            graph.add_pass(Box::new(PortalMaskPass::new(device)));
            graph.add_pass(Box::new(PortalInstancePass::new(
                device,
                indirect_buf,
                compacted_buf,
                compacted_chains_buf,
            )));
        }
    }

    // Foliage rasterisation goes immediately after GBufferPass/PortalInstance and
    // before VirtualGeometry. After those two because it composites into the same
    // eight targets with LoadOp::Load (chain fusion is transitive across a linear
    // run of exact-attachment-match passes); before VirtualGeometry because VG
    // binds only seven attachments — it omits gbuffer_velocity — so anything
    // downstream of VG can never fuse with the G-buffer.
    if let Some((blade_arena, tile_table, visible_blades, foliage_indirect, blades_per_tile)) =
        foliage_buffers
    {
        graph.add_pass(Box::new(FoliageGBufferPass::new(
            device,
            blade_arena,
            tile_table,
            visible_blades,
            foliage_indirect,
            blades_per_tile,
        )));
    }

    let mut vg_pass = VirtualGeometryPass::new(device, camera_buf);
    vg_pass.debug_mode = config.debug_mode;
    graph.add_pass(Box::new(vg_pass));
    graph.add_pass(Box::new(PerfOverlayAnalyzerPass::new(Arc::clone(perf))));
}

fn add_forward_geometry_passes(
    graph: &mut RenderGraph,
    device: &Arc<wgpu::Device>,
    camera_buf: &wgpu::Buffer,
    config: &RendererConfig,
    perf: &Arc<std::sync::Mutex<PerfOverlayShared>>,
    render_all_opaque: bool,
) {
    let mut fl_pass = ForwardLitPass::new(device, config.surface_format);
    fl_pass.render_all_opaque = render_all_opaque;
    graph.add_pass(Box::new(fl_pass));

    let mut vg_pass = VirtualGeometryPass::new(device, camera_buf);
    vg_pass.debug_mode = config.debug_mode;
    graph.add_pass(Box::new(vg_pass));
    graph.add_pass(Box::new(PerfOverlayAnalyzerPass::new(Arc::clone(perf))));
}

fn add_late_passes(
    graph: &mut RenderGraph,
    device: &Arc<wgpu::Device>,
    queue: &Arc<wgpu::Queue>,
    camera_buf: &wgpu::Buffer,
    config: &RendererConfig,
    perf: &Arc<std::sync::Mutex<PerfOverlayShared>>,
    debug_state: Arc<std::sync::Mutex<DebugDrawState>>,
    debug_camera_buf: &wgpu::Buffer,
    w: u32,
    h: u32,
    _scene_db: helio::SceneDbHandle,
) {
    // The atmosphere over everything lit: the sky where nothing was drawn,
    // aerial perspective over geometry. Before the overlays (billboards,
    // coronas, debug draw), which are not seen through the air.
    graph.add_pass(Box::new(AtmosphereCompositePass::new(device, config.surface_format)));
    let lights_buf = scene_buffer_or_dummy(
        &_scene_db,
        device,
        pulsar_scenedb::gpu::BufferKey::of("scene_lights"),
        "SceneDB Lights",
        96,
    );

    let spotlight = image::load_from_memory(SPOTLIGHT_PNG)
        .unwrap_or_else(|_| image::DynamicImage::new_rgba8(1, 1))
        .into_rgba8();
    let (sw, sh) = spotlight.dimensions();
    let mut billboard_pass = BillboardPass::new_with_sprite_rgba(
        device,
        queue,
        camera_buf,
        config.surface_format,
        spotlight.as_raw(),
        sw,
        sh,
    );
    billboard_pass.set_occluded_by_geometry(true);
    graph.add_pass(Box::new(billboard_pass));
    graph.add_pass(Box::new(PerfOverlayAnalyzerPass::new(Arc::clone(perf))));

    graph.add_pass(Box::new(CoronaPass::new(
        device,
        queue,
        camera_buf,
        config.surface_format,
    )));
    graph.add_pass(Box::new(PerfOverlayAnalyzerPass::new(Arc::clone(perf))));

    // Editor-only checkerboard indicator over each portal's opening —
    // disabled (zero draws) by default; the host application flips it via
    // `renderer.find_pass_mut::<PortalEditorOverlayPass>()` alongside its own
    // editor/game-mode toggle. See that pass's docs for why it isn't wired to
    // `Renderer::is_editor_mode()` automatically.
    if config.enable_portals {
        graph.add_pass(Box::new(PortalEditorOverlayPass::new(
            device,
            config.surface_format,
        )));
    }
    graph.add_pass(Box::new(PerfOverlayAnalyzerPass::new(Arc::clone(perf))));

    graph.add_pass(Box::new(WaterSimPass::new(
        device,
        camera_buf,
        w,
        h,
        config.surface_format,
    )));
    graph.add_pass(Box::new(PerfOverlayAnalyzerPass::new(Arc::clone(perf))));

    // Editor overlay: grid, and the wireframe bounds of every scene volume.
    //
    // Position is load-bearing in both directions. It must come after
    // DeferredLightPass, which writes the same "pre_aa" target and would
    // otherwise paint over it — that is why the grid never appeared while this
    // sat with the early passes. It must also come before FXAA/post-process
    // consume pre_aa, or it would be drawn into an image nothing reads again.
    //
    // Depth-tested against the internal-res scene depth so geometry genuinely
    // in front occludes the bounds, exactly as BillboardPass does.
    graph.add_pass(Box::new(helio::DebugDrawPass::new(
        device,
        debug_camera_buf,
        config.surface_format,
        debug_state,
        true,
        true,
    )));
}

fn convert_perf_mode(mode: helio::PerfOverlayMode) -> helio_pass_perf_overlay::PerfOverlayMode {
    use helio::PerfOverlayMode as H;
    use helio_pass_perf_overlay::PerfOverlayMode as P;
    match mode {
        H::Disabled => P::Disabled,
        H::PassOverdraw => P::PassOverdraw,
        H::ShaderComplexity => P::ShaderComplexity,
        H::TileLightCount => P::TileLightCount,
        H::PassOutput => P::PassOutput,
    }
}

fn add_final_passes(
    graph: &mut RenderGraph,
    device: &Arc<wgpu::Device>,
    queue: &Arc<wgpu::Queue>,
    config: &RendererConfig,
    perf: &Arc<std::sync::Mutex<PerfOverlayShared>>,
    debug_state: Arc<std::sync::Mutex<DebugDrawState>>,
    debug_camera_buf: &wgpu::Buffer,
    debug_overlay: Option<&Arc<std::sync::Mutex<DebugOverlayState>>>,
) {
    // Game content over the final image, under the editor's overlays.
    add_sprite_overlay_passes(graph, device, queue, config);

    graph.add_pass(Box::new(PerfOverlayAnalyzerPass::new(Arc::clone(perf))));

    let mut perf_overlay_pass =
        PerfOverlayPass::new(device, Arc::clone(perf), config.surface_format);
    perf_overlay_pass.set_mode(convert_perf_mode(config.perf_overlay_mode));
    graph.add_pass(Box::new(perf_overlay_pass));

    // User debug lines/tris, drawn at output resolution over the final image.
    // The editor overlay is a separate instance added in add_late_passes,
    // because it needs the internal-res scene depth to occlude correctly.
    graph.add_pass(Box::new(helio::DebugDrawPass::new(
        device,
        debug_camera_buf,
        config.surface_format,
        debug_state,
        false,
        false,
    )));

    if let Some(shared) = debug_overlay {
        graph.add_pass(Box::new(DebugOverlayPass::new(
            device,
            queue,
            Arc::clone(shared),
            config.surface_format,
            config.width,
            config.height,
        )));
    }
}

/// Build the default deferred graph from the shared renderer construction ABI.
///
/// The legacy argument-list entry points below remain for applications that
/// construct graphs directly. New `RendererBuilder` callers should pass this
/// function to `with_pass_build_context` so device, queue, scene, buffers, and
/// configuration are supplied once by the renderer.
pub fn build_default_graph_with_context(ctx: PassBuildContext<'_>) -> RenderGraph {
    build_default_graph_internal(
        ctx.device,
        ctx.queue,
        ctx.camera_buffer,
        ctx.config,
        ctx.debug_state,
        ctx.debug_camera_buffer,
        ctx.cull_stats_buffer,
        ctx.owns_device,
        None,
        None,
        ctx.scene_db.clone(),
        Vec::new(),
        Vec::new(),
        Vec::new(),
    )
}

/// Build the externally-owned default graph from [`PassBuildContext`].
pub fn build_default_graph_external_with_context(ctx: PassBuildContext<'_>) -> RenderGraph {
    let mut ctx = ctx;
    ctx.owns_device = false;
    build_default_graph_with_context(ctx)
}

/// Build the default deferred graph with an application-selected voxel pass.
/// The pass factory is intentionally independent of any voxel format or
/// generation implementation.
pub fn build_default_graph_external_with_voxel_passes(
    ctx: PassBuildContext<'_>,
    voxel_passes: Vec<VoxelPassFactory>,
) -> RenderGraph {
    build_default_graph_external_with_passes(ctx, voxel_passes, Vec::new())
}

/// Add GBuffer voxel passes and final resource consumers before graph locking.
/// Final passes must declare all resource reads/writes. Their factories are
/// retained by graph rebuilds and receive the current internal render size.
pub fn build_default_graph_external_with_passes(
    ctx: PassBuildContext<'_>,
    voxel_passes: Vec<VoxelPassFactory>,
    final_passes: Vec<GraphPassFactory>,
) -> RenderGraph {
    build_default_graph_external_with_lighting_passes(ctx, voxel_passes, Vec::new(), final_passes)
}

/// Backend-owned resolves after opaque lighting and before fog, transparency
/// and antialiasing. Factories survive graph rebuilds at the new internal size.
/// A resolve must declare its resource accesses and preserve unrelated pixels.
pub fn build_default_graph_external_with_lighting_passes(
    mut ctx: PassBuildContext<'_>,
    voxel_passes: Vec<VoxelPassFactory>,
    lighting_passes: Vec<GraphPassFactory>,
    final_passes: Vec<GraphPassFactory>,
) -> RenderGraph {
    ctx.owns_device = false;
    build_default_graph_internal(
        ctx.device,
        ctx.queue,
        ctx.camera_buffer,
        ctx.config,
        ctx.debug_state,
        ctx.debug_camera_buffer,
        ctx.cull_stats_buffer,
        false,
        None,
        None,
        ctx.scene_db,
        voxel_passes,
        final_passes,
        lighting_passes,
    )
}

/// Build the deferred graph with user post-process effects from the shared ABI.
pub fn build_default_graph_with_user_effects_with_context(
    ctx: PassBuildContext<'_>,
    user_effects: &'static str,
) -> RenderGraph {
    build_default_graph_internal(
        ctx.device,
        ctx.queue,
        ctx.camera_buffer,
        ctx.config,
        ctx.debug_state,
        ctx.debug_camera_buffer,
        ctx.cull_stats_buffer,
        ctx.owns_device,
        None,
        Some(user_effects),
        ctx.scene_db.clone(),
        Vec::new(),
        Vec::new(),
        Vec::new(),
    )
}

pub fn build_default_graph(
    device: &Arc<wgpu::Device>,
    queue: &Arc<wgpu::Queue>,
    camera_buf: &wgpu::Buffer,
    config: RendererConfig,
    debug_state: Arc<std::sync::Mutex<DebugDrawState>>,
    debug_camera_buf: &wgpu::Buffer,
    cull_stats_buf: &wgpu::Buffer,
    debug_overlay: Option<&Arc<std::sync::Mutex<DebugOverlayState>>>,
    scene_db: helio::SceneDbHandle,
) -> RenderGraph {
    build_default_graph_internal(
        device,
        queue,
        camera_buf,
        config,
        debug_state,
        debug_camera_buf,
        cull_stats_buf,
        true,
        debug_overlay,
        None,
        scene_db,
        Vec::new(),
        Vec::new(),
        Vec::new(),
    )
}

pub fn build_default_graph_with_user_effects(
    device: &Arc<wgpu::Device>,
    queue: &Arc<wgpu::Queue>,
    camera_buf: &wgpu::Buffer,
    config: RendererConfig,
    debug_state: Arc<std::sync::Mutex<DebugDrawState>>,
    debug_camera_buf: &wgpu::Buffer,
    cull_stats_buf: &wgpu::Buffer,
    debug_overlay: Option<&Arc<std::sync::Mutex<DebugOverlayState>>>,
    user_effects: &'static str,
    scene_db: helio::SceneDbHandle,
) -> RenderGraph {
    build_default_graph_internal(
        device,
        queue,
        camera_buf,
        config,
        debug_state,
        debug_camera_buf,
        cull_stats_buf,
        true,
        debug_overlay,
        Some(user_effects),
        scene_db,
        Vec::new(),
        Vec::new(),
        Vec::new(),
    )
}

pub fn build_default_graph_external(
    device: &Arc<wgpu::Device>,
    queue: &Arc<wgpu::Queue>,
    camera_buf: &wgpu::Buffer,
    config: RendererConfig,
    debug_state: Arc<std::sync::Mutex<DebugDrawState>>,
    debug_camera_buf: &wgpu::Buffer,
    cull_stats_buf: &wgpu::Buffer,
    debug_overlay: Option<&Arc<std::sync::Mutex<DebugOverlayState>>>,
    scene_db: helio::SceneDbHandle,
) -> RenderGraph {
    build_default_graph_internal(
        device,
        queue,
        camera_buf,
        config,
        debug_state,
        debug_camera_buf,
        cull_stats_buf,
        false,
        debug_overlay,
        None,
        scene_db,
        Vec::new(),
        Vec::new(),
        Vec::new(),
    )
}

/// Anti-aliasing stage of [`add_scene_linear_chain`].
enum SceneAa {
    /// Temporal reconstruction; its HDR resolve is published as `tsr_color`.
    Tsr(TsrPass),
    /// FXAA into a linear intermediate of the chain's format (`fxaa_color`).
    Fxaa,
    None,
}

/// Participating media, transparency, anti-aliasing and lens optics, in the
/// one order every graph shares. Returns the key `PostProcessPass` consumes.
///
/// ```text
/// PP settings (camera baseline + volumes) -> medium integration
///   -> medium composite at each opaque surface's depth (fogged_hdr)
///   -> transparency, each fragment fogged at its own depth
///   -> AA -> lens response from the reconstructed image
///   -> PostProcessPass: meter, bloom, lens, exposure, grade, tone map once
/// ```
///
/// Shafts exist only where a medium scatters light, so they reach exposure,
/// bloom and lens extraction as ordinary radiance. The lens reads the AA
/// output so reprojection never sees screen-space optics, and PP applies it
/// after metering so flare cannot feed back into exposure.
fn add_scene_linear_chain(
    graph: &mut RenderGraph,
    device: &Arc<wgpu::Device>,
    hdr_format: wgpu::TextureFormat,
    transparent: Option<helio_pass_transparent::TransparentPass>,
    aa: SceneAa,
    width: u32,
    height: u32,
) -> &'static str {
    graph.add_pass(Box::new(PostProcessVolumeBlendPass::new(device)));
    graph.add_pass(Box::new(VolumetricFogPass::new(device)));
    graph.add_pass(Box::new(FogCompositePass::with_format(device, hdr_format)));
    if let Some(transparent) = transparent {
        graph.add_pass(Box::new(transparent.with_fogged_target()));
    }
    let resolved = match aa {
        SceneAa::Tsr(tsr) => {
            graph.add_pass(Box::new(
                tsr.with_color_input(FOGGED_HDR).with_intermediate_output(),
            ));
            "tsr_color"
        }
        SceneAa::Fxaa => {
            graph.add_pass(Box::new(
                FxaaPass::new(device, hdr_format)
                    .with_color_input(FOGGED_HDR)
                    .with_intermediate_target(hdr_format),
            ));
            "fxaa_color"
        }
        SceneAa::None => FOGGED_HDR,
    };
    graph.add_pass(Box::new(
        LensFlarePass::new_hdr(device, width, height).with_color_input(resolved),
    ));
    resolved
}

fn build_default_graph_internal(
    device: &Arc<wgpu::Device>,
    queue: &Arc<wgpu::Queue>,
    camera_buf: &wgpu::Buffer,
    config: RendererConfig,
    debug_state: Arc<std::sync::Mutex<DebugDrawState>>,
    debug_camera_buf: &wgpu::Buffer,
    cull_stats_buf: &wgpu::Buffer,
    owns_device: bool,
    debug_overlay: Option<&Arc<std::sync::Mutex<DebugOverlayState>>>,
    user_effects: Option<&'static str>,
    scene_db: helio::SceneDbHandle,
    voxel_passes: Vec<VoxelPassFactory>,
    final_passes: Vec<GraphPassFactory>,
    lighting_passes: Vec<GraphPassFactory>,
) -> RenderGraph {
    let iw = config.internal_width();
    let ih = config.internal_height();

    let mut graph = new_graph(device, queue, owns_device, &config);
    declare_common_external_inputs(&mut graph);

    // Lighting and everything drawn into it stay scene-linear FP16 until
    // PostProcessPass tone maps once; a display-format target would clamp
    // emitters at 1.0 so nothing could bloom or flare.
    let lighting_config = RendererConfig { surface_format: FOGGED_HDR_FORMAT, ..config };
    let perf = add_common_early_passes(
        &mut graph,
        device,
        queue,
        camera_buf,
        &lighting_config,
        cull_stats_buf,
        iw,
        ih,
        scene_db.clone(),
    );

    graph.add_pass(Box::new(LightCullPass::new(device, iw, ih)));

    // No RadianceCascadesPass: nothing in this graph publishes `rc_view`,
    // so DeferredLight's `has_rc_gi` (and SSR's RC fallback) never read its
    // output, and it only cost a probe-atlas trace every frame.

    add_geometry_passes(&mut graph, device, camera_buf, &config, &perf, scene_db.clone());

    for factory in &voxel_passes {
        graph.add_pass(factory(device, queue, iw, ih));
    }

    // Decal pass — projects decals into the G-buffer after it's been written.
    // Runs as a compute pass between GBuffer and deferred lighting. Reads
    // SceneDB's `"decals"` buffer directly at execute time; no central
    // buffer to pass in here.
    graph.add_pass(Box::new(DecalPass::new(device, queue, camera_buf, iw, ih)));

    // SSR pass — screen-space reflections for glossy/metallic surfaces.
    // Runs after GBuffer (needs normals + depth + Hi-Z), before deferred lighting.
    //
    // Off by default: this is one of the graph's most expensive passes. When it
    // is absent DeferredLightPass binds its 1×1 black fallback for `ssr_trace`,
    // so the only loss is the reflection contribution. Also compiled out on
    // Apple targets entirely (REFLECTIONS_SUPPORTED), which render it incorrectly.
    if config.enable_ssr && helio_core::REFLECTIONS_SUPPORTED {
        graph.add_pass(Box::new(SsrPass::new(device, queue, camera_buf, iw, ih)));
    }

    // Planar reflection pass — reflects the scene across world-space planes.
    // Runs before deferred lighting so DeferredLightPass can composite its
    // output alongside SSR (planar_reflection texture) in a single draw call.
    //
    // Off by default: cost scales with scene complexity times reflection-plane
    // count. DeferredLightPass falls back to a 1×1 black `planar_reflection`.
    // Also compiled out on Apple targets (REFLECTIONS_SUPPORTED) — see SSR above.
    if config.enable_planar_reflections && helio_core::REFLECTIONS_SUPPORTED {
        graph.add_pass(Box::new(PlanarReflectionPass::new(
            device,
            camera_buf,
            config.surface_format,
        )));
    }

    let mut deferred_light_pass =
        DeferredLightPass::new(device, queue, camera_buf, lighting_config.surface_format);
    deferred_light_pass.set_shadow_quality(config.shadow_quality, queue);
    deferred_light_pass.debug_mode = config.debug_mode;
    deferred_light_pass.set_env_reflections(config.enable_environment_reflections);
    graph.add_pass(Box::new(deferred_light_pass));
    for factory in &lighting_passes {
        graph.add_pass(factory(device, queue, iw, ih));
    }
    graph.add_pass(Box::new(PerfOverlayCostAnalyzerPass::new(perf.clone())));
    graph.add_pass(Box::new(PerfOverlayAnalyzerPass::new(perf.clone())));

    add_late_passes(
        &mut graph,
        device,
        queue,
        camera_buf,
        &lighting_config,
        &perf,
        debug_state.clone(),
        debug_camera_buf,
        iw,
        ih,
        scene_db.clone(),
    );

    // Media, transparency, AA and lens in scene-linear FP16, at internal
    // resolution until TSR upscales. The composite lifts the lighting target
    // into FP16 so nothing downstream clamps radiance.
    //
    // When TSR is active it provides superior temporal anti-aliasing, so FXAA
    // would only add blur on top of an already-sharp image.  Gate FXAA behind
    // the TSR flag so the two don't compete.
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
    // Transparent pass — alpha-blended geometry (simple fixed shader).
    // Its bind group is rebuilt per-frame from `object_batch`/SceneDB
    // resources at execute time, so no buffers are passed in here.
    let transparent = helio_pass_transparent::TransparentPass::new(device, FOGGED_HDR_FORMAT);
    let resolved =
        add_scene_linear_chain(&mut graph, device, FOGGED_HDR_FORMAT, Some(transparent), aa, iw, ih);

    let mut pp = PostProcessPass::new_with_user_effects(
        device,
        queue,
        config.width,
        config.height,
        config.surface_format,
        user_effects,
    )
    .with_color_input(resolved);
    // Enable pre_dof output so the DofPass can read the post-processed image.
    pp.set_output_to_pre_dof(true);
    graph.add_pass(Box::new(pp));

    // Cinematic bokeh DOF — runs after the main uber-shader, reads "pre_dof"
    // (written by PostProcessPass) and writes the final output to ctx.target.
    graph.add_pass(Box::new(DofPass::new(
        device,
        queue,
        config.width,
        config.height,
        config.surface_format,
    )));

    add_final_passes(
        &mut graph,
        device,
        queue,
        &config,
        &perf,
        debug_state,
        debug_camera_buf,
        debug_overlay,
    );

    for factory in &final_passes {
        graph.add_pass(factory(device, queue, iw, ih));
    }
    graph.lock(iw, ih);

    let overlay_owned = debug_overlay.map(Arc::clone);
    let effect_snippet = user_effects;
    let rebuilder: GraphRebuilder = Arc::new(
        move |device, queue, config, debug_state, camera_buf, debug_camera_buf, cull_stats_buf| {
            build_default_graph_internal(
                device,
                queue,
                camera_buf,
                config,
                debug_state,
                debug_camera_buf,
                cull_stats_buf,
                owns_device,
                overlay_owned.as_ref(),
                effect_snippet,
                scene_db.clone(),
                voxel_passes.clone(),
                final_passes.clone(),
                lighting_passes.clone(),
            )
        },
    );
    graph.set_swap_policy(default_swap_policy());
    graph.set_graph_data(rebuilder);

    graph
}

pub fn build_fxaa_graph(
    device: &Arc<wgpu::Device>,
    queue: &Arc<wgpu::Queue>,
    camera_buf: &wgpu::Buffer,
    config: RendererConfig,
    debug_state: Arc<std::sync::Mutex<DebugDrawState>>,
    debug_camera_buf: &wgpu::Buffer,
    cull_stats_buf: &wgpu::Buffer,
    debug_overlay: Option<&Arc<std::sync::Mutex<DebugOverlayState>>>,
    scene_db: helio::SceneDbHandle,
) -> RenderGraph {
    build_fxaa_graph_internal(
        device,
        queue,
        camera_buf,
        config,
        debug_state,
        debug_camera_buf,
        cull_stats_buf,
        true,
        debug_overlay,
        scene_db,
    )
}

/// Build the FXAA graph from the shared renderer construction ABI.
pub fn build_fxaa_graph_with_context(ctx: PassBuildContext<'_>) -> RenderGraph {
    build_fxaa_graph_internal(
        ctx.device,
        ctx.queue,
        ctx.camera_buffer,
        ctx.config,
        ctx.debug_state,
        ctx.camera_buffer,
        ctx.cull_stats_buffer,
        ctx.owns_device,
        None,
        ctx.scene_db.clone(),
    )
}

pub fn build_fxaa_graph_external(
    device: &Arc<wgpu::Device>,
    queue: &Arc<wgpu::Queue>,
    camera_buf: &wgpu::Buffer,
    config: RendererConfig,
    debug_state: Arc<std::sync::Mutex<DebugDrawState>>,
    debug_camera_buf: &wgpu::Buffer,
    cull_stats_buf: &wgpu::Buffer,
    debug_overlay: Option<&Arc<std::sync::Mutex<DebugOverlayState>>>,
    scene_db: helio::SceneDbHandle,
) -> RenderGraph {
    build_fxaa_graph_internal(
        device,
        queue,
        camera_buf,
        config,
        debug_state,
        debug_camera_buf,
        cull_stats_buf,
        false,
        debug_overlay,
        scene_db,
    )
}

fn build_fxaa_graph_internal(
    device: &Arc<wgpu::Device>,
    queue: &Arc<wgpu::Queue>,
    camera_buf: &wgpu::Buffer,
    config: RendererConfig,
    debug_state: Arc<std::sync::Mutex<DebugDrawState>>,
    debug_camera_buf: &wgpu::Buffer,
    cull_stats_buf: &wgpu::Buffer,
    owns_device: bool,
    debug_overlay: Option<&Arc<std::sync::Mutex<DebugOverlayState>>>,
    scene_db: helio::SceneDbHandle,
) -> RenderGraph {
    let iw = config.internal_width();
    let ih = config.internal_height();

    let mut graph = new_graph(device, queue, owns_device, &config);
    declare_common_external_inputs(&mut graph);

    // Lighting and everything drawn into it stay scene-linear FP16 until
    // PostProcessPass tone maps once; a display-format target would clamp
    // emitters at 1.0 so nothing could bloom or flare.
    let lighting_config = RendererConfig { surface_format: FOGGED_HDR_FORMAT, ..config };
    let perf = add_common_early_passes(
        &mut graph,
        device,
        queue,
        camera_buf,
        &lighting_config,
        cull_stats_buf,
        iw,
        ih,
        scene_db.clone(),
    );

    graph.add_pass(Box::new(LightCullPass::new(device, iw, ih)));

    // No RadianceCascadesPass: nothing in this graph publishes `rc_view`,
    // so DeferredLight's `has_rc_gi` (and SSR's RC fallback) never read its
    // output, and it only cost a probe-atlas trace every frame.

    add_geometry_passes(&mut graph, device, camera_buf, &config, &perf, scene_db.clone());

    // Decal pass — reads SceneDB's `"decals"` buffer directly at execute time.
    graph.add_pass(Box::new(DecalPass::new(device, queue, camera_buf, iw, ih)));

    // Both off by default; see the notes in the primary graph builder above.
    // DeferredLightPass binds 1×1 black fallbacks when either pass is absent.
    // Also compiled out on Apple targets (REFLECTIONS_SUPPORTED).
    if config.enable_ssr && helio_core::REFLECTIONS_SUPPORTED {
        graph.add_pass(Box::new(SsrPass::new(device, queue, camera_buf, iw, ih)));
    }

    if config.enable_planar_reflections && helio_core::REFLECTIONS_SUPPORTED {
        graph.add_pass(Box::new(PlanarReflectionPass::new(
            device,
            camera_buf,
            config.surface_format,
        )));
    }

    let mut deferred_light_pass =
        DeferredLightPass::new(device, queue, camera_buf, lighting_config.surface_format);
    deferred_light_pass.set_shadow_quality(config.shadow_quality, queue);
    deferred_light_pass.debug_mode = config.debug_mode;
    deferred_light_pass.set_env_reflections(config.enable_environment_reflections);
    graph.add_pass(Box::new(deferred_light_pass));
    graph.add_pass(Box::new(PerfOverlayCostAnalyzerPass::new(Arc::clone(
        &perf,
    ))));
    graph.add_pass(Box::new(PerfOverlayAnalyzerPass::new(Arc::clone(&perf))));

    add_late_passes(
        &mut graph,
        device,
        queue,
        camera_buf,
        &lighting_config,
        &perf,
        debug_state.clone(),
        debug_camera_buf,
        iw,
        ih,
        scene_db.clone(),
    );

    // TSR provides temporal super-resolution upscaling with its own temporal AA.
    // When TSR is not configured, skip temporal accumulation (render at native res).
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
        None => SceneAa::None,
    };
    let resolved = add_scene_linear_chain(&mut graph, device, FOGGED_HDR_FORMAT, None, aa, iw, ih);

    graph.add_pass(Box::new(
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

    add_final_passes(
        &mut graph,
        device,
        queue,
        &config,
        &perf,
        debug_state,
        debug_camera_buf,
        debug_overlay,
    );

    graph.lock(iw, ih);

    let overlay_owned = debug_overlay.map(Arc::clone);
    let rebuilder: GraphRebuilder = Arc::new(
        move |device, queue, config, debug_state, camera_buf, debug_camera_buf, cull_stats_buf| {
            build_fxaa_graph_internal(
                device,
                queue,
                camera_buf,
                config,
                debug_state,
                debug_camera_buf,
                cull_stats_buf,
                owns_device,
                overlay_owned.as_ref(),
                scene_db.clone(),
            )
        },
    );
    graph.set_swap_policy(default_swap_policy());
    graph.set_graph_data(rebuilder);

    graph
}

fn build_hlfs_graph_internal(
    device: &Arc<wgpu::Device>,
    queue: &Arc<wgpu::Queue>,
    camera_buf: &wgpu::Buffer,
    config: RendererConfig,
    debug_state: Arc<std::sync::Mutex<DebugDrawState>>,
    debug_camera_buf: &wgpu::Buffer,
    cull_stats_buf: &wgpu::Buffer,
    owns_device: bool,
    debug_overlay: Option<&Arc<std::sync::Mutex<DebugOverlayState>>>,
    scene_db: helio::SceneDbHandle,
) -> RenderGraph {
    let iw = config.internal_width();
    let ih = config.internal_height();

    let mut graph = new_graph(device, queue, owns_device, &config);
    declare_common_external_inputs(&mut graph);

    let perf = add_common_early_passes(
        &mut graph,
        device,
        queue,
        camera_buf,
        &config,
        cull_stats_buf,
        iw,
        ih,
        scene_db.clone(),
    );

    add_geometry_passes(&mut graph, device, camera_buf, &config, &perf, scene_db.clone());

    // Decal pass — reads SceneDB's `"decals"` buffer directly at execute time.
    graph.add_pass(Box::new(DecalPass::new(device, queue, camera_buf, iw, ih)));

    // Lighting stays in linear HDR until the post-process pass tonemaps it.
    let lighting_format = HlfsPass::preferred_output_format(device);
    let mut hlfs_pass = HlfsPass::new(device, queue, iw, ih, lighting_format);
    hlfs_pass.set_shadow_quality(config.shadow_quality, queue);
    graph.add_pass(Box::new(hlfs_pass));
    if config.enable_ssr && helio_core::REFLECTIONS_SUPPORTED {
        graph.add_pass(Box::new(SsrPass::new(device, queue, camera_buf, iw, ih)));
        graph.add_pass(Box::new(helio_pass_ssr::SsrCompositePass::new(device, camera_buf, lighting_format)));
    }

    let lighting_config = RendererConfig {
        surface_format: lighting_format,
        ..config
    };

    add_late_passes(
        &mut graph,
        device,
        queue,
        camera_buf,
        &lighting_config,
        &perf,
        debug_state.clone(),
        debug_camera_buf,
        iw,
        ih,
        scene_db.clone(),
    );

    // Blend transparent surfaces in the same linear HDR format as HLFS, over
    // the fogged opaque image.
    let mut transparent = helio_pass_transparent::TransparentPass::new(device, lighting_format);
    if config.tsr_quality.is_some() { transparent = transparent.with_reactive_mask(); }

    // TSR provides temporal super-resolution upscaling with its own temporal AA.
    // When TSR is not configured, skip temporal accumulation (render at native res).
    let aa = match config.tsr_quality {
        Some(quality) => SceneAa::Tsr(TsrPass::new(
            device,
            iw,
            ih,
            config.width,
            config.height,
            config.surface_format,
            quality,
        ).with_transparency_reactivity()),
        None => SceneAa::None,
    };
    let resolved =
        add_scene_linear_chain(&mut graph, device, lighting_format, Some(transparent), aa, iw, ih);

    graph.add_pass(Box::new(
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

    add_final_passes(
        &mut graph,
        device,
        queue,
        &config,
        &perf,
        debug_state,
        debug_camera_buf,
        debug_overlay,
    );

    graph.lock(iw, ih);

    let overlay_owned = debug_overlay.map(Arc::clone);
    let rebuilder: GraphRebuilder = Arc::new(
        move |device, queue, config, debug_state, camera_buf, debug_camera_buf, cull_stats_buf| {
            build_hlfs_graph_internal(
                device,
                queue,
                camera_buf,
                config,
                debug_state,
                debug_camera_buf,
                cull_stats_buf,
                owns_device,
                overlay_owned.as_ref(),
                scene_db.clone(),
            )
        },
    );
    graph.set_swap_policy(default_swap_policy());
    graph.set_graph_data(rebuilder);

    graph
}

/// Build the shared HLFS graph with default `HlfsMode::ScreenSpace` visibility.
pub fn build_hlfs_graph(
    device: &Arc<wgpu::Device>,
    queue: &Arc<wgpu::Queue>,
    camera_buf: &wgpu::Buffer,
    config: RendererConfig,
    debug_state: Arc<std::sync::Mutex<DebugDrawState>>,
    debug_camera_buf: &wgpu::Buffer,
    cull_stats_buf: &wgpu::Buffer,
    debug_overlay: Option<&Arc<std::sync::Mutex<DebugOverlayState>>>,
    scene_db: helio::SceneDbHandle,
) -> RenderGraph {
    build_hlfs_graph_internal(
        device,
        queue,
        camera_buf,
        config,
        debug_state,
        debug_camera_buf,
        cull_stats_buf,
        true,
        debug_overlay,
        scene_db,
    )
}

/// Build the HLFS graph from the shared renderer construction ABI.
pub fn build_hlfs_graph_with_context(ctx: PassBuildContext<'_>) -> RenderGraph {
    build_hlfs_graph_internal(
        ctx.device,
        ctx.queue,
        ctx.camera_buffer,
        ctx.config,
        ctx.debug_state,
        ctx.camera_buffer,
        ctx.cull_stats_buffer,
        ctx.owns_device,
        None,
        ctx.scene_db.clone(),
    )
}

/// Build the FXAA graph with default `HlfsMode::ScreenSpace` visibility.
pub fn build_fxaa_hlfs_graph(
    device: &Arc<wgpu::Device>,
    queue: &Arc<wgpu::Queue>,
    camera_buf: &wgpu::Buffer,
    config: RendererConfig,
    debug_state: Arc<std::sync::Mutex<DebugDrawState>>,
    debug_camera_buf: &wgpu::Buffer,
    cull_stats_buf: &wgpu::Buffer,
    debug_overlay: Option<&Arc<std::sync::Mutex<DebugOverlayState>>>,
    scene_db: helio::SceneDbHandle,
) -> RenderGraph {
    build_fxaa_hlfs_graph_internal(
        device,
        queue,
        camera_buf,
        config,
        debug_state,
        debug_camera_buf,
        cull_stats_buf,
        true,
        debug_overlay,
        scene_db,
    )
}

/// Build the FXAA+HLFS graph from the shared renderer construction ABI.
pub fn build_fxaa_hlfs_graph_with_context(ctx: PassBuildContext<'_>) -> RenderGraph {
    build_fxaa_hlfs_graph_internal(
        ctx.device,
        ctx.queue,
        ctx.camera_buffer,
        ctx.config,
        ctx.debug_state,
        ctx.camera_buffer,
        ctx.cull_stats_buffer,
        ctx.owns_device,
        None,
        ctx.scene_db.clone(),
    )
}

pub fn build_fxaa_hlfs_graph_external(
    device: &Arc<wgpu::Device>,
    queue: &Arc<wgpu::Queue>,
    camera_buf: &wgpu::Buffer,
    config: RendererConfig,
    debug_state: Arc<std::sync::Mutex<DebugDrawState>>,
    debug_camera_buf: &wgpu::Buffer,
    cull_stats_buf: &wgpu::Buffer,
    debug_overlay: Option<&Arc<std::sync::Mutex<DebugOverlayState>>>,
    scene_db: helio::SceneDbHandle,
) -> RenderGraph {
    build_fxaa_hlfs_graph_internal(
        device,
        queue,
        camera_buf,
        config,
        debug_state,
        debug_camera_buf,
        cull_stats_buf,
        false,
        debug_overlay,
        scene_db,
    )
}

fn build_fxaa_hlfs_graph_internal(
    device: &Arc<wgpu::Device>,
    queue: &Arc<wgpu::Queue>,
    camera_buf: &wgpu::Buffer,
    config: RendererConfig,
    debug_state: Arc<std::sync::Mutex<DebugDrawState>>,
    debug_camera_buf: &wgpu::Buffer,
    cull_stats_buf: &wgpu::Buffer,
    owns_device: bool,
    debug_overlay: Option<&Arc<std::sync::Mutex<DebugOverlayState>>>,
    scene_db: helio::SceneDbHandle,
) -> RenderGraph {
    let w = config.internal_width();
    let h = config.internal_height();

    let mut graph = new_graph(device, queue, owns_device, &config);
    declare_common_external_inputs(&mut graph);

    let perf = add_common_early_passes(
        &mut graph,
        device,
        queue,
        camera_buf,
        &config,
        cull_stats_buf,
        w,
        h,
        scene_db.clone(),
    );

    add_geometry_passes(&mut graph, device, camera_buf, &config, &perf, scene_db.clone());

    // Decal pass — reads SceneDB's `"decals"` buffer directly at execute time.
    graph.add_pass(Box::new(DecalPass::new(device, queue, camera_buf, w, h)));

    // Lighting stays in linear HDR until the post-process pass tonemaps it.
    let lighting_format = HlfsPass::preferred_output_format(device);
    let mut hlfs_pass = HlfsPass::new(device, queue, w, h, lighting_format);
    hlfs_pass.set_shadow_quality(config.shadow_quality, queue);
    graph.add_pass(Box::new(hlfs_pass));
    if config.enable_ssr && helio_core::REFLECTIONS_SUPPORTED {
        graph.add_pass(Box::new(SsrPass::new(device, queue, camera_buf, w, h)));
        graph.add_pass(Box::new(helio_pass_ssr::SsrCompositePass::new(device, camera_buf, lighting_format)));
    }

    let lighting_config = RendererConfig {
        surface_format: lighting_format,
        ..config
    };

    add_late_passes(
        &mut graph,
        device,
        queue,
        camera_buf,
        &lighting_config,
        &perf,
        debug_state.clone(),
        debug_camera_buf,
        w,
        h,
        scene_db.clone(),
    );

    // Match the native HLFS graph: transparent glass belongs in linear HDR
    // before anti-aliasing and tonemapping, alongside the opaque lighting.
    let transparent = helio_pass_transparent::TransparentPass::new(device, lighting_format);
    let resolved = add_scene_linear_chain(
        &mut graph, device, lighting_format, Some(transparent), SceneAa::Fxaa, w, h,
    );

    graph.add_pass(Box::new(PostProcessPass::new_with_user_effects(
        device,
        queue,
        config.width,
        config.height,
        config.surface_format,
        None,
    ).with_color_input(resolved)));

    add_final_passes(
        &mut graph,
        device,
        queue,
        &config,
        &perf,
        debug_state,
        debug_camera_buf,
        debug_overlay,
    );

    graph.lock(w, h);

    let overlay_owned = debug_overlay.map(Arc::clone);
    let rebuilder: GraphRebuilder = Arc::new(
        move |device, queue, config, debug_state, camera_buf, debug_camera_buf, cull_stats_buf| {
            build_fxaa_hlfs_graph_internal(
                device,
                queue,
                camera_buf,
                config,
                debug_state,
                debug_camera_buf,
                cull_stats_buf,
                owns_device,
                overlay_owned.as_ref(),
                scene_db.clone(),
            )
        },
    );
    graph.set_swap_policy(default_swap_policy());
    graph.set_graph_data(rebuilder);

    graph
}

pub fn build_simple_graph(
    device: &Arc<wgpu::Device>,
    queue: &Arc<wgpu::Queue>,
    surface_format: wgpu::TextureFormat,
) -> RenderGraph {
    let mut graph = RenderGraph::new(device, queue);
    graph.add_pass(Box::new(SimpleCubePass::new(device, surface_format)));

    let rebuilder: GraphRebuilder = Arc::new(
        move |device, _queue, _scene, _config, _debug_state, _debug_camera_buf, _cull_stats_buf| {
            let mut g = RenderGraph::new(device, _queue);
            g.add_pass(Box::new(SimpleCubePass::new(device, surface_format)));
            g
        },
    );
    graph.set_graph_data(rebuilder);

    graph
}

/// Build the simple graph from the shared renderer construction ABI.
pub fn build_simple_graph_with_context(ctx: PassBuildContext<'_>) -> RenderGraph {
    let mut graph = new_graph(ctx.device, ctx.queue, ctx.owns_device, &ctx.config);
    graph.add_pass(Box::new(SimpleCubePass::new(
        ctx.device,
        ctx.config.surface_format,
    )));
    graph
}

// ── Forward-mode graph builders ─────────────────────────────────────────────

pub fn build_forward_opaque_graph(
    device: &Arc<wgpu::Device>,
    queue: &Arc<wgpu::Queue>,
    camera_buf: &wgpu::Buffer,
    config: RendererConfig,
    debug_state: Arc<std::sync::Mutex<DebugDrawState>>,
    debug_camera_buf: &wgpu::Buffer,
    cull_stats_buf: &wgpu::Buffer,
    debug_overlay: Option<&Arc<std::sync::Mutex<DebugOverlayState>>>,
    scene_db: helio::SceneDbHandle,
) -> RenderGraph {
    build_forward_graph_internal(
        device,
        queue,
        camera_buf,
        config,
        debug_state,
        debug_camera_buf,
        cull_stats_buf,
        true,
        debug_overlay,
        scene_db,
    )
}

/// Build the forward-opaque graph from the shared renderer construction ABI.
pub fn build_forward_opaque_graph_with_context(ctx: PassBuildContext<'_>) -> RenderGraph {
    build_forward_graph_internal(
        ctx.device,
        ctx.queue,
        ctx.camera_buffer,
        ctx.config,
        ctx.debug_state,
        ctx.camera_buffer,
        ctx.cull_stats_buffer,
        ctx.owns_device,
        None,
        ctx.scene_db.clone(),
    )
}

pub fn build_forward_opaque_graph_external(
    device: &Arc<wgpu::Device>,
    queue: &Arc<wgpu::Queue>,
    camera_buf: &wgpu::Buffer,
    config: RendererConfig,
    debug_state: Arc<std::sync::Mutex<DebugDrawState>>,
    debug_camera_buf: &wgpu::Buffer,
    cull_stats_buf: &wgpu::Buffer,
    debug_overlay: Option<&Arc<std::sync::Mutex<DebugOverlayState>>>,
    scene_db: helio::SceneDbHandle,
) -> RenderGraph {
    build_forward_graph_internal(
        device,
        queue,
        camera_buf,
        config,
        debug_state,
        debug_camera_buf,
        cull_stats_buf,
        false,
        debug_overlay,
        scene_db,
    )
}

pub fn build_forward_only_graph(
    device: &Arc<wgpu::Device>,
    queue: &Arc<wgpu::Queue>,
    camera_buf: &wgpu::Buffer,
    config: RendererConfig,
    debug_state: Arc<std::sync::Mutex<DebugDrawState>>,
    debug_camera_buf: &wgpu::Buffer,
    cull_stats_buf: &wgpu::Buffer,
    debug_overlay: Option<&Arc<std::sync::Mutex<DebugOverlayState>>>,
    scene_db: helio::SceneDbHandle,
) -> RenderGraph {
    build_forward_graph_internal(
        device,
        queue,
        camera_buf,
        config,
        debug_state,
        debug_camera_buf,
        cull_stats_buf,
        true,
        debug_overlay,
        scene_db,
    )
}

/// Build the forward-only graph from the shared renderer construction ABI.
pub fn build_forward_only_graph_with_context(ctx: PassBuildContext<'_>) -> RenderGraph {
    build_forward_graph_internal(
        ctx.device,
        ctx.queue,
        ctx.camera_buffer,
        ctx.config,
        ctx.debug_state,
        ctx.camera_buffer,
        ctx.cull_stats_buffer,
        ctx.owns_device,
        None,
        ctx.scene_db.clone(),
    )
}

pub fn build_forward_only_graph_external(
    device: &Arc<wgpu::Device>,
    queue: &Arc<wgpu::Queue>,
    camera_buf: &wgpu::Buffer,
    config: RendererConfig,
    debug_state: Arc<std::sync::Mutex<DebugDrawState>>,
    debug_camera_buf: &wgpu::Buffer,
    cull_stats_buf: &wgpu::Buffer,
    debug_overlay: Option<&Arc<std::sync::Mutex<DebugOverlayState>>>,
    scene_db: helio::SceneDbHandle,
) -> RenderGraph {
    build_forward_graph_internal(
        device,
        queue,
        camera_buf,
        config,
        debug_state,
        debug_camera_buf,
        cull_stats_buf,
        false,
        debug_overlay,
        scene_db,
    )
}

fn build_forward_graph_internal(
    device: &Arc<wgpu::Device>,
    queue: &Arc<wgpu::Queue>,
    camera_buf: &wgpu::Buffer,
    config: RendererConfig,
    debug_state: Arc<std::sync::Mutex<DebugDrawState>>,
    debug_camera_buf: &wgpu::Buffer,
    cull_stats_buf: &wgpu::Buffer,
    owns_device: bool,
    debug_overlay: Option<&Arc<std::sync::Mutex<DebugOverlayState>>>,
    scene_db: helio::SceneDbHandle,
) -> RenderGraph {
    let iw = config.internal_width();
    let ih = config.internal_height();

    let mut graph = new_graph(device, queue, owns_device, &config);
    declare_common_external_inputs(&mut graph);

    // Lighting and everything drawn into it stay scene-linear FP16 until
    // PostProcessPass tone maps once; a display-format target would clamp
    // emitters at 1.0 so nothing could bloom or flare.
    let lighting_config = RendererConfig { surface_format: FOGGED_HDR_FORMAT, ..config };
    let perf = add_common_early_passes(
        &mut graph,
        device,
        queue,
        camera_buf,
        &lighting_config,
        cull_stats_buf,
        iw,
        ih,
        scene_db.clone(),
    );

    graph.add_pass(Box::new(LightCullPass::new(device, iw, ih)));

    // No RadianceCascadesPass: nothing in this graph publishes `rc_view`,
    // so DeferredLight's `has_rc_gi` (and SSR's RC fallback) never read its
    // output, and it only cost a probe-atlas trace every frame.

    // Forward geometry pass replaces G-buffer + decal + deferred light + SSR + planar reflections
    add_forward_geometry_passes(&mut graph, device, camera_buf, &lighting_config, &perf, true);

    add_late_passes(
        &mut graph,
        device,
        queue,
        camera_buf,
        &lighting_config,
        &perf,
        debug_state.clone(),
        debug_camera_buf,
        iw,
        ih,
        scene_db.clone(),
    );

    // Transparent pass — alpha-blended geometry (simple fixed shader).
    // Its bind group is rebuilt per-frame from `object_batch`/SceneDB
    // resources at execute time, so no buffers are passed in here.
    let transparent = helio_pass_transparent::TransparentPass::new(device, FOGGED_HDR_FORMAT);
    let resolved = add_scene_linear_chain(
        &mut graph, device, FOGGED_HDR_FORMAT, Some(transparent), SceneAa::Fxaa, iw, ih,
    );

    graph.add_pass(Box::new(PostProcessPass::new_with_user_effects(
        device,
        queue,
        config.width,
        config.height,
        config.surface_format,
        None,
    ).with_color_input(resolved)));

    add_final_passes(
        &mut graph,
        device,
        queue,
        &config,
        &perf,
        debug_state,
        debug_camera_buf,
        debug_overlay,
    );

    graph.lock(iw, ih);

    let overlay_owned = debug_overlay.map(Arc::clone);
    let rebuilder: GraphRebuilder = Arc::new(
        move |device, queue, config, debug_state, camera_buf, debug_camera_buf, cull_stats_buf| {
            build_forward_graph_internal(
                device,
                queue,
                camera_buf,
                config,
                debug_state,
                debug_camera_buf,
                cull_stats_buf,
                owns_device,
                overlay_owned.as_ref(),
                scene_db.clone(),
            )
        },
    );
    graph.set_swap_policy(default_swap_policy());
    graph.set_graph_data(rebuilder);

    graph
}
