//! Physical media authoring. The authored values reach the volumetric fog
//! pass through their derived rows ([`super::environment_rows`]) and the
//! graph's environment join; nothing here writes pass rows.
use engine_class_derive::{engine_class, register_world_component};
use pulsar_reflection::{ComponentRuntimeBehavior, ComponentRuntimeContext, RuntimeComponentOwner};
use helio_pass_volumetric_fog::GlobalFogComponent as GlobalMedium;
use super::FogMode;

#[engine_class(no_register, clone, debug, serialize, deserialize)]
#[category("Medium", category_color = "#D18F6F")]
#[serde(default)]
pub struct MediumProps {
    #[property(category = "Medium")]
    pub mode: FogMode,
    /// Absorption plus scattering, in inverse metres. One world unit is one metre.
    #[property(min = 0.0, max = 100.0, step = 0.001, category = "Medium")]
    pub extinction: f32,
    /// Fraction of extinction that scatters, per color channel.
    #[property(category = "Medium")]
    pub albedo: [f32; 3],
    /// Scene-linear radiance emitted per metre; independent of extinction.
    #[property(category = "Medium")]
    pub emission: [f32; 3],
    #[property(min = -0.999, max = 0.999, step = 0.001, category = "Medium")]
    pub anisotropy: f32,
    #[property(min = 0.0, max = 100.0, step = 0.001, category = "Medium")]
    pub height_falloff: f32,
    /// World-space reference height in metres.
    #[property(category = "Medium")]
    pub height: f32,
}

impl Default for MediumProps {
    fn default() -> Self {
        let value = GlobalMedium::default();
        Self { mode: FogMode::Uniform, extinction: value.extinction, albedo: value.albedo,
            emission: value.emission, anisotropy: value.anisotropy,
            height_falloff: value.height_falloff, height: value.height }
    }
}

impl MediumProps {
    pub fn to_medium(&self) -> GlobalMedium {
        GlobalMedium {
            enabled: 1,
            mode: match self.mode { FogMode::Uniform => 0, FogMode::HeightBased => 1, FogMode::Smoke => 2 },
            extinction: self.extinction, albedo: self.albedo, emission: self.emission,
            anisotropy: self.anisotropy, height_falloff: self.height_falloff, height: self.height,
            ..Default::default()
        }
    }
}

#[engine_class(category = "Rendering", clone, debug, serialize, deserialize)]
#[category("Medium", category_color = "#D18F6F")]
#[category("Volume", category_color = "#8F8F8F")]
#[serde(default)]
pub struct GlobalFogComponent {
    #[property(category = "Medium")]
    pub enabled: bool,
    #[sub_props]
    #[serde(flatten)]
    pub medium: MediumProps,
}

impl Default for GlobalFogComponent {
    fn default() -> Self { Self { enabled: true, medium: MediumProps::default() } }
}

#[engine_class(category = "Rendering", clone, debug, serialize, deserialize)]
#[category("Medium", category_color = "#D18F6F")]
#[category("Volume", category_color = "#8F8F8F")]
#[serde(default)]
pub struct LocalFogVolumeComponent {
    #[property(category = "Medium")]
    pub enabled: bool,
    /// Full local extent in metres. The owner's scale and rotation produce
    /// the world AABB (the environment join, on the GPU).
    #[property(category = "Volume")]
    pub size: [f32; 3],
    /// Inward fade from the world AABB boundary, in metres.
    #[property(min = 0.0, max = 1000.0, step = 0.1, category = "Volume")]
    pub edge_fade: f32,
    #[sub_props]
    #[serde(flatten)]
    pub medium: MediumProps,
}

impl Default for LocalFogVolumeComponent {
    fn default() -> Self { Self { enabled: true, size: [10.0; 3], edge_fade: 0.0, medium: MediumProps::default() } }
}

#[register_world_component]
impl ComponentRuntimeBehavior for GlobalFogComponent {
    const CLASS_NAME: &'static str = "GlobalFogComponent";
    fn sync_component(_owner: &RuntimeComponentOwner, _index: usize, _component: &Self, _context: &mut dyn ComponentRuntimeContext) {}
}

#[register_world_component]
impl ComponentRuntimeBehavior for LocalFogVolumeComponent {
    const CLASS_NAME: &'static str = "LocalFogVolumeComponent";
    fn sync_component(_owner: &RuntimeComponentOwner, _index: usize, _component: &Self, _context: &mut dyn ComponentRuntimeContext) {}
}
