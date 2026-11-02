//! GI as a feature: its settings and the nodes it adds to preset graphs.
//!
//! The probes' update and the meshes' tracer join at
//! [`Slot::Geometry`] (the update first, every other tracer between them);
//! the resolve joins at [`Slot::LightingInputs`], before deferred lighting
//! reads `gi_irradiance`.

use helio_feature_api::{Feature, FeatureRegistrar, NodeBuild, NodeDescriptor, ParamDesc, Slot};

use crate::{mesh_rays_supported, GiConfig, GiMeshTracePass, GiPass, GiResolvePass};

/// GI's settings (a renderer config's `features`).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct GiSettings {
    /// The finest probes' spacing (metres), or `None` for the default:
    /// open ground's ([`crate::DEFAULT_FINEST_SPACING`]) with a terrain
    /// tracer, half a metre (rooms, pillars, furniture) without one.
    pub probe_spacing: Option<f32>,
}

impl GiSettings {
    /// The probes' configuration these settings ask for, with a terrain
    /// tracing them every frame or not ([`helio_gi_api::TerrainTracer`]).
    pub fn config(&self, terrain_traced: bool) -> GiConfig {
        let default = if terrain_traced {
            crate::DEFAULT_FINEST_SPACING
        } else {
            0.5
        };
        GiConfig {
            finest_spacing: self.probe_spacing.map_or(default, f64::from),
            needs_tlas: !terrain_traced,
        }
    }

    /// Whether there is GI on `device`: a terrain traces it, or the meshes
    /// can (hardware ray queries).
    pub fn active(&self, device: &wgpu::Device, terrain_traced: bool) -> bool {
        terrain_traced || mesh_rays_supported(device)
    }
}

/// World-space diffuse GI ([`crate`]).
pub struct GiFeature;

const CATEGORY: &str = "Passes: Lighting";

fn settings(build: &NodeBuild<'_, '_>) -> GiSettings {
    build
        .env
        .setting::<GiSettings>()
        .copied()
        .unwrap_or_default()
}

/// Whether a feature traces the probes against its terrain this graph.
fn terrain_traced(build: &NodeBuild<'_, '_>) -> bool {
    build.env.setting::<helio_gi_api::TerrainTracer>().is_some()
}

fn node(
    name: &str,
    title: &str,
    description: &str,
    build: impl Fn(&mut NodeBuild<'_, '_>) -> Result<(), String> + Send + Sync + 'static,
) -> NodeDescriptor {
    NodeDescriptor::new(
        &format!("helio.pass.{name}"),
        title,
        CATEGORY,
        description,
        build,
    )
    .with_param(ParamDesc::boolean("enabled", "Enabled", true))
    .with_generated_pins()
}

impl Feature for GiFeature {
    fn id(&self) -> &'static str {
        "helio.gi"
    }

    fn register(&self, registrar: &mut FeatureRegistrar<'_>) {
        registrar.node(
            node(
                "gi",
                "GI: Probe Update",
                "Integrates last frame's hits into the probe clipmap and writes this frame's rays. \
                 Builds nothing where nothing can trace them.",
                |build| {
                    let settings = settings(build);
                    let traced = terrain_traced(build);
                    if settings.active(build.env.device, traced) {
                        let pass = GiPass::new(build.env.device, settings.config(traced));
                        build.graph.add_pass(Box::new(pass));
                    }
                    Ok(())
                },
            )
            .in_slot(Slot::Geometry, 0),
        );
        registrar.node(
            node(
                "gi_mesh_trace",
                "GI: Mesh Rays",
                "Traces the probes' rays against the scene's TLAS (hardware ray queries).",
                |build| {
                    if let Some(pass) = GiMeshTracePass::new(build.env.device) {
                        build.graph.add_pass(Box::new(pass));
                    }
                    Ok(())
                },
            )
            .in_slot(Slot::Geometry, 20),
        );
        registrar.node(
            node(
                "gi_resolve",
                "GI: Resolve",
                "The probes' irradiance at every pixel (`gi_irradiance`) for deferred lighting.",
                |build| {
                    if settings(build).active(build.env.device, terrain_traced(build)) {
                        build
                            .graph
                            .add_pass(Box::new(GiResolvePass::new(build.env.device)));
                    }
                    Ok(())
                },
            )
            .in_slot(Slot::LightingInputs, 0),
        );
    }
}
