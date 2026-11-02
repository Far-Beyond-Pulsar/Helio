//! Helio's spine as a feature: the core pipeline every preset is built around
//! and its extension points' neighbours.
//!
//! [`SpineFeature`] registers one node per spine pass: scene prep (object
//! batches, shadows, the background, culling, light binning), the G-buffer and
//! forward geometry, deferred lighting, transparency, TSR and the tone map,
//! plus the host-input nodes that declare what the renderer supplies each
//! frame. Features add the rest at the spine's slots.
//!
//! The spine knows no renderer: its nodes read the build environment and
//! [`SpineSettings`], which a graph's builder sets (Helio's presets map their
//! renderer config into it).

use std::sync::Arc;

use helio_feature_api::{
    missing, pass_node, BuildEnv, Feature, FeatureRegistrar, NodeDescriptor, ParamDesc, PinType,
};
use helio_pass_deferred_light::DeferredLightPass;
use helio_pass_forward_lit::ForwardLitPass;
use helio_pass_gbuffer::GBufferPass;
use helio_pass_hiz::HiZBuildPass;
use helio_pass_indirect_dispatch::IndirectDispatchPass;
use helio_pass_light_cull::LightCullPass;
use helio_pass_object_batch::ObjectBatchPass;
use helio_pass_occlusion_cull::OcclusionCullPass;
use helio_pass_postprocess::{
    FogCompositePass, PostProcessPass, PostProcessVolumeBlendPass, FOGGED_HDR,
};
use helio_pass_shadow::ShadowPass;
use helio_pass_shadow_dirty::ShadowDirtyPass;
use helio_pass_shadow_matrix::{ShadowBudget, ShadowMatrixPass, ShadowQuality};
use helio_pass_tsr::{TsrPass, TsrQuality};

mod background;
pub use background::BackgroundPass;

/// The colour the lens writes; the tone map reads it by default. A resource
/// name, so the spine does not depend on the lens feature.
const LENS_COLOR: &str = "lens_color";

/// What the spine's nodes read that is not the frame: a graph's builder sets
/// it (Helio's presets from their renderer config).
#[derive(Clone, Debug)]
pub struct SpineSettings {
    pub shadow_budget: ShadowBudget,
    pub shadow_quality: ShadowQuality,
    /// The colour `pre_aa` is cleared to before anything draws.
    pub background_color: [f32; 4],
    /// The deferred lighting's debug view (0: none).
    pub debug_mode: u32,
    /// Environment-cubemap indirect specular.
    pub environment_reflections: bool,
}

impl Default for SpineSettings {
    fn default() -> Self {
        Self {
            shadow_budget: ShadowBudget::default(),
            shadow_quality: ShadowQuality::Medium,
            background_color: [0.0, 0.0, 0.0, 1.0],
            debug_mode: 0,
            environment_reflections: true,
        }
    }
}

fn settings(env: &BuildEnv<'_>) -> SpineSettings {
    env.setting::<SpineSettings>().cloned().unwrap_or_default()
}

/// The scene handle the spine reads SceneDB's buffers through.
fn scene_db(env: &BuildEnv<'_>) -> Result<pulsar_scenedb::gpu::GpuMirrorHandle, String> {
    env.require::<pulsar_scenedb::gpu::GpuMirrorHandle>()
        .cloned()
}

/// SceneDB's buffer under `key`, or an empty stand-in of `fallback_size`
/// bytes while the scene has none.
pub fn scene_buffer_or_dummy(
    scene_db: &pulsar_scenedb::gpu::GpuMirrorHandle,
    device: &wgpu::Device,
    key: pulsar_scenedb::gpu::BufferKey,
    label: &str,
    fallback_size: u64,
) -> pulsar_scenedb::gpu::BufferHandle {
    scene_db
        .store()
        .resolve_buffer_handle(key)
        .unwrap_or_else(|| pulsar_scenedb::gpu::BufferHandle {
            buffer: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: fallback_size.max(4),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            epoch: 0,
            row_bytes: 0,
            content_generation: 0,
        })
}

/// Resources the renderer supplies every frame. Each has a host node that
/// declares it; the passes that read one get a Scene input for it.
pub const HOST_INPUTS: &[&str] = &[
    "material_textures",
    "render_environment",
    "billboards",
    "corona_emitters",
];

/// The kind of the host node for resource `name`.
pub fn host_kind(name: &str) -> String {
    format!("host.{name}")
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

fn tsr_quality(name: &str) -> TsrQuality {
    match name {
        "performance" => TsrQuality::Performance,
        "quality" => TsrQuality::Quality,
        "native" => TsrQuality::Native,
        _ => TsrQuality::Balanced,
    }
}

/// The shadow-dirty flags the shadow matrix pass fills and the dirty pass reads.
struct ShadowDirtyBuffer(Arc<wgpu::Buffer>);

/// The spine's nodes and host inputs.
pub struct SpineFeature;

impl Feature for SpineFeature {
    fn id(&self) -> &'static str {
        "helio.spine"
    }

    fn register(&self, registrar: &mut FeatureRegistrar<'_>) {
        register_host_inputs(registrar);
        register_scene_prep(registrar);
        register_geometry(registrar);
        register_lighting(registrar);
        register_post(registrar);
    }
}

fn register_host_inputs(registry: &mut FeatureRegistrar<'_>) {
    for name in HOST_INPUTS {
        let name: &'static str = name;
        registry.node(
            NodeDescriptor::new(
                &host_kind(name),
                &format!("Host::{name}"),
                "Host",
                "A resource the renderer supplies every frame.",
                move |build| {
                    build.graph.declare_external_input(name);
                    Ok(())
                },
            )
            .with_data_only()
            .with_generated_pins(),
        );
    }
}

fn shadow_atlas(env: &BuildEnv<'_>) -> Result<(ShadowBudget, u32), String> {
    let budget = settings(env)
        .shadow_budget
        .validate()
        .map_err(|_| "invalid renderer shadow budget".to_string())?;
    let size = budget.atlas_size(env.device.limits().max_texture_dimension_2d);
    Ok((budget, size))
}

fn register_scene_prep(registry: &mut FeatureRegistrar<'_>) {
    const CATEGORY: &str = "Passes: Scene Prep";

    registry.node(pass_node(
        "object_batch",
        "Scene: Object Batches",
        CATEGORY,
        "Gathers the scene's objects into GPU-driven batches every other pass reads.",
        |build| {
            let pass = ObjectBatchPass::new(build.env.device);
            build.graph.add_pass(Box::new(pass));
            Ok(())
        },
    ));

    registry.node(pass_node(
        "shadow_matrix",
        "Shadows: Light Matrices",
        CATEGORY,
        "Computes the view-projection of every shadow-casting light face.",
        |build| {
            let device = build.env.device;
            let (budget, atlas_size) = shadow_atlas(build.env)?;
            let lights = scene_buffer_or_dummy(
                &scene_db(build.env)?,
                device,
                pulsar_scenedb::gpu::BufferKey::of("scene_lights"),
                "SceneDB Lights",
                96,
            );
            let matrices = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Shadow Matrices"),
                size: helio_pass_shadow_matrix::MAX_SHADOW_FACES as u64
                    * std::mem::size_of::<helio_pass_shadow_matrix::GpuShadowMatrix>() as u64,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });
            let dirty = Arc::new(device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Shadow Dirty Flags"),
                size: helio_pass_shadow_matrix::MAX_SHADOW_CASTERS as u64 * 4,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
            let hashes = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Shadow Hashes"),
                size: helio_pass_shadow_matrix::MAX_SHADOW_CASTERS as u64 * 4,
                usage: wgpu::BufferUsages::STORAGE,
                mapped_at_creation: false,
            });
            let (_, ih) = build.env.internal_size();
            build.graph.add_pass(Box::new(
                ShadowMatrixPass::new(
                    device,
                    &lights.buffer,
                    &matrices,
                    build.env.camera_buffer,
                    &dirty,
                    &hashes,
                    atlas_size,
                )
                .with_budget(budget, ih),
            ));
            build.env.shared(|| ShadowDirtyBuffer(dirty));
            Ok(())
        },
    ));

    registry.node(pass_node(
        "shadow_dirty",
        "Shadows: Find Stale Faces",
        CATEGORY,
        "Marks which shadow faces need re-rendering this frame.",
        |build| {
            let dirty = build
                .env
                .shared_existing::<ShadowDirtyBuffer>()
                .map(|shared| Arc::clone(&shared.0))
                .ok_or_else(|| missing("Shadow Dirty Flags", "Shadow Matrices"))?;
            let pass = ShadowDirtyPass::new(build.env.device, dirty);
            build.graph.add_pass(Box::new(pass));
            Ok(())
        },
    ));

    registry.node(pass_node(
        "shadow",
        "Shadows: Render Atlas",
        CATEGORY,
        "Renders the dirty shadow faces into the tiled shadow atlas.",
        |build| {
            let (_, atlas_size) = shadow_atlas(build.env)?;
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
                build.env.device,
                build.env.queue,
                face_dirty,
                face_geom_count,
                face_pages,
                atlas_size,
            )));
            Ok(())
        },
    ));

    registry.node(pass_node(
        "background",
        "Scene: Clear Background",
        CATEGORY,
        "Clears the scene colour to the background the sky and lighting draw over.",
        |build| {
            let pass = BackgroundPass::new(
                build.env.scene_format(),
                settings(build.env).background_color,
            );
            build.graph.add_pass(Box::new(pass));
            Ok(())
        },
    ));

    registry.node(pass_node(
        "indirect_dispatch",
        "Culling: Prepare Dispatch",
        CATEGORY,
        "Prepares the indirect dispatch arguments the culling passes use.",
        |build| {
            let pass =
                IndirectDispatchPass::new(build.env.device, build.env.cull_stats_buffer.clone());
            build.graph.add_pass(Box::new(pass));
            Ok(())
        },
    ));

    registry.node(pass_node(
        "hiz",
        "Culling: Depth Pyramid (Hi-Z)",
        CATEGORY,
        "Builds the hierarchical depth pyramid occlusion culling tests against.",
        |build| {
            let (iw, ih) = build.env.internal_size();
            let pass = HiZBuildPass::new(build.env.device, build.env.queue, iw, ih);
            build.graph.add_pass(Box::new(pass));
            Ok(())
        },
    ));

    registry.node(pass_node(
        "occlusion_cull",
        "Culling: Frustum + Occlusion",
        CATEGORY,
        "Frustum and Hi-Z occlusion culling of the object batch.",
        |build| {
            let (iw, ih) = build.env.internal_size();
            let sampler = build
                .graph
                .find_pass::<HiZBuildPass>()
                .map(|hiz| Arc::clone(&hiz.hiz_sampler))
                .ok_or_else(|| missing("Occlusion Cull", "Hi-Z Build"))?;
            let pass = OcclusionCullPass::new(
                build.env.device,
                sampler,
                iw,
                ih,
                build.env.cull_stats_buffer.clone(),
            );
            build.graph.add_pass(Box::new(pass));
            Ok(())
        },
    ));

    registry.node(pass_node(
        "light_cull",
        "Lighting: Tile Light Binning",
        CATEGORY,
        "Bins the scene's lights into screen tiles.",
        |build| {
            let (iw, ih) = build.env.internal_size();
            let pass = LightCullPass::new(build.env.device, iw, ih);
            build.graph.add_pass(Box::new(pass));
            Ok(())
        },
    ));
}

fn register_geometry(registry: &mut FeatureRegistrar<'_>) {
    const CATEGORY: &str = "Passes: Geometry";

    registry.node(pass_node(
        "gbuffer",
        "Geometry: GBuffer",
        CATEGORY,
        "Opaque geometry into the GBuffer and the scene depth.",
        |build| {
            let pass = GBufferPass::new(build.env.device);
            build.graph.add_pass(Box::new(pass));
            Ok(())
        },
    ));

    registry.node(
        pass_node(
            "forward_lit",
            "Geometry: Forward Lit",
            CATEGORY,
            "Lit opaque geometry in one forward pass, for pipelines without a GBuffer.",
            |build| {
                let render_all_opaque = build.boolean("render_all_opaque");
                let mut pass = ForwardLitPass::new(build.env.device, build.env.scene_format());
                pass.render_all_opaque = render_all_opaque;
                build.graph.add_pass(Box::new(pass));
                Ok(())
            },
        )
        .with_param(ParamDesc::boolean(
            "render_all_opaque",
            "Render all opaque",
            false,
        ))
        .with_marker_after(),
    );
}

fn register_lighting(registry: &mut FeatureRegistrar<'_>) {
    const CATEGORY: &str = "Passes: Lighting";

    registry.node(
        pass_node(
            "deferred_light",
            "Lighting: Deferred",
            CATEGORY,
            "Lights the GBuffer with every light, shadow, reflection and the atmosphere.",
            |build| {
                let spine = settings(build.env);
                let queue = build.env.queue;
                let mut pass = DeferredLightPass::new(
                    build.env.device,
                    queue,
                    build.env.camera_buffer,
                    build.env.scene_format(),
                );
                pass.set_shadow_quality(spine.shadow_quality, queue);
                pass.debug_mode = spine.debug_mode;
                pass.set_env_reflections(spine.environment_reflections);
                build.graph.add_pass(Box::new(pass));
                Ok(())
            },
        )
        .with_marker_after(),
    );
}

fn register_post(registry: &mut FeatureRegistrar<'_>) {
    const CATEGORY: &str = "Passes: Post";

    registry.node(pass_node(
        "post_process_volume_blend",
        "Post: Blend Volumes",
        CATEGORY,
        "Blends the post-process volumes around the camera.",
        |build| {
            let pass = PostProcessVolumeBlendPass::new(build.env.device);
            build.graph.add_pass(Box::new(pass));
            Ok(())
        },
    ));

    registry.node(pass_node(
        "fog_composite",
        "Fog: Composite",
        CATEGORY,
        "Composites the fog over the scene into the fogged HDR colour.",
        |build| {
            let pass = FogCompositePass::with_format(build.env.device, build.env.scene_format());
            build.graph.add_pass(Box::new(pass));
            Ok(())
        },
    ));

    registry.node(
        pass_node(
            "transparent",
            "Transparency: Blend",
            CATEGORY,
            "Alpha-blended geometry over the fogged colour.",
            |build| {
                let pass = helio_pass_transparent::TransparentPass::new(
                    build.env.device,
                    build.env.scene_format(),
                );
                let pass = if build.boolean("reactive_mask") {
                    pass.with_reactive_mask()
                } else {
                    pass
                };
                build.graph.add_pass(Box::new(pass.with_fogged_target()));
                Ok(())
            },
        )
        .with_param(ParamDesc::boolean(
            "reactive_mask",
            "Reactive mask (for TSR)",
            false,
        )),
    );

    registry.node(
        pass_node(
            "tsr",
            "Anti-Aliasing: TSR",
            CATEGORY,
            "Temporal anti-aliasing and upscaling. The renderer's internal resolution follows the project's TSR setting.",
            |build| {
                let (iw, ih) = build.env.internal_size();
                let frame = build.env.frame;
                let quality = tsr_quality(&build.text("quality"));
                let tsr = TsrPass::new(
                    build.env.device,
                    iw,
                    ih,
                    frame.width,
                    frame.height,
                    frame.surface_format,
                    quality,
                )
                .with_color_input(FOGGED_HDR)
                .with_intermediate_output()
                .with_motion_vectors();
                let stochastic = build.boolean("stochastic_mask")
                    || build
                        .env
                        .setting::<helio_pass_tsr::StochasticSamples>()
                        .is_some();
                let tsr = if stochastic {
                    tsr.with_stochastic_mask()
                } else {
                    tsr
                };
                let tsr = if build.boolean("transparency_reactivity") {
                    tsr.with_transparency_reactivity()
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
        ))
        .with_param(ParamDesc::boolean(
            "transparency_reactivity",
            "Transparency reactivity",
            false,
        )),
    );

    registry.node(
        pass_node(
            "post_process",
            "Post: Tone Map + Grade",
            CATEGORY,
            "Bloom, exposure, tone mapping and colour grading into the output.",
            |build| {
                let input = colour_resource(&build.text("color_input"));
                let frame = build.env.frame;
                let effects = build
                    .env
                    .setting::<helio_pass_postprocess::UserEffects>()
                    .map(|effects| effects.0);
                build.graph.add_pass(Box::new(
                    PostProcessPass::new_with_user_effects(
                        build.env.device,
                        build.env.queue,
                        frame.width,
                        frame.height,
                        frame.surface_format,
                        effects,
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

#[cfg(test)]
mod tests {
    #[test]
    fn the_default_scene_format_is_the_fogged_colour_format() {
        assert_eq!(
            helio_feature_api::DEFAULT_SCENE_FORMAT,
            helio_pass_postprocess::FOGGED_HDR_FORMAT
        );
    }
}
