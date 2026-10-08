//! A planet's air: the sky, the sun's colour through it, the sky's light on
//! surfaces and aerial perspective. The pass-owned SceneDB row
//! (`helio_pass_sky::AtmosphereComponent`) is the runtime authority; any
//! scene may have one: a planet (centred on the owner) or flat ground at the
//! world's origin. The sun is the scene's first directional light.
use engine_class_derive::{engine_class, register_runtime_behavior, register_world_component};
use helio_pass_sky::atmosphere::placement;
use helio_pass_sky::AtmosphereComponent as AtmosphereRow;
use pulsar_reflection::{get_subsystem, ComponentRuntimeBehavior, ComponentRuntimeContext, RuntimeComponentOwner, Reflectable};
use serde::{Deserialize, Serialize};
use crate::subsystems::PendingWorldWrites;

/// Where the atmosphere's planet is.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, Reflectable)]
pub enum AtmospherePlacement {
    /// Flat ground at the world's origin: the planet's centre is straight
    /// below it, `planet_radius_km` down.
    GroundAtOrigin,
    /// A planet centred on the owner's position.
    PlanetAtOwner,
}

impl Default for AtmospherePlacement {
    fn default() -> Self {
        Self::GroundAtOrigin
    }
}

/// Defaults are Earth's air (Hillaire 2020). Coefficients are per kilometre,
/// heights and radii in kilometres.
#[engine_class(category = "Rendering", clone, debug, serialize, deserialize)]
#[category("Atmosphere", category_color = "#6F9FD1")]
#[category("Planet", category_color = "#8F8F8F")]
#[category("Scattering", category_color = "#9FB7D1")]
#[serde(default)]
pub struct AtmosphereComponent {
    #[property(category = "Atmosphere")]
    pub enabled: bool,
    #[property(category = "Planet")]
    pub placement: AtmospherePlacement,
    /// Radius of the planet's surface.
    #[property(min = 0.001, max = 1000000.0, step = 1.0, category = "Planet")]
    pub planet_radius_km: f32,
    /// Height of the top of the air above the surface.
    #[property(min = 0.001, max = 10000.0, step = 1.0, category = "Planet")]
    pub thickness_km: f32,
    /// Light the ground reflects back into the air.
    #[property(category = "Planet")]
    pub ground_albedo: [f32; 3],
    /// Half the sun's apparent diameter.
    #[property(min = 0.01, max = 20.0, step = 0.01, category = "Atmosphere")]
    pub sun_angular_radius_deg: f32,
    /// Molecular (Rayleigh) scattering at the surface: what makes the sky blue.
    #[property(category = "Scattering")]
    pub rayleigh_scattering: [f32; 3],
    #[property(min = 0.01, max = 1000.0, step = 0.1, category = "Scattering")]
    pub rayleigh_scale_height_km: f32,
    /// Aerosol (Mie) scattering at the surface: haze and the glow around the sun.
    #[property(category = "Scattering")]
    pub mie_scattering: [f32; 3],
    #[property(category = "Scattering")]
    pub mie_absorption: [f32; 3],
    #[property(min = 0.01, max = 1000.0, step = 0.1, category = "Scattering")]
    pub mie_scale_height_km: f32,
    /// Forward scattering of aerosols (0 isotropic, towards 1 forward).
    #[property(min = -0.999, max = 0.999, step = 0.001, category = "Scattering")]
    pub mie_anisotropy: f32,
    /// Ozone absorption at the layer's peak: the zenith's blue at dusk.
    #[property(category = "Scattering")]
    pub ozone_absorption: [f32; 3],
    #[property(min = 0.0, max = 1000.0, step = 0.1, category = "Scattering")]
    pub ozone_center_km: f32,
    #[property(min = 0.0, max = 1000.0, step = 0.1, category = "Scattering")]
    pub ozone_width_km: f32,
}

impl Default for AtmosphereComponent {
    fn default() -> Self {
        Self::from_row(&AtmosphereRow::earth())
    }
}

impl AtmosphereComponent {
    fn from_row(row: &AtmosphereRow) -> Self {
        Self {
            enabled: true,
            placement: AtmospherePlacement::GroundAtOrigin,
            planet_radius_km: row.bottom_radius,
            thickness_km: row.top_radius - row.bottom_radius,
            ground_albedo: row.ground_albedo,
            sun_angular_radius_deg: row.sun_angular_radius.to_degrees(),
            rayleigh_scattering: row.rayleigh_scattering,
            rayleigh_scale_height_km: row.rayleigh_scale_height,
            mie_scattering: row.mie_scattering,
            mie_absorption: row.mie_absorption,
            mie_scale_height_km: row.mie_scale_height,
            mie_anisotropy: row.mie_g,
            ozone_absorption: row.ozone_absorption,
            ozone_center_km: row.ozone_center,
            ozone_width_km: row.ozone_width,
        }
    }

    pub fn to_row(&self, owner: &RuntimeComponentOwner) -> AtmosphereRow {
        let bottom = self.planet_radius_km.max(0.001);
        let (placement, center) = match self.placement {
            AtmospherePlacement::GroundAtOrigin => (placement::GROUND_AT_ORIGIN, [0.0; 3]),
            AtmospherePlacement::PlanetAtOwner => (placement::CENTER, owner.position),
        };
        AtmosphereRow {
            center,
            placement,
            rayleigh_scattering: self.rayleigh_scattering,
            rayleigh_scale_height: self.rayleigh_scale_height_km.max(0.01),
            mie_scattering: self.mie_scattering,
            mie_scale_height: self.mie_scale_height_km.max(0.01),
            mie_absorption: self.mie_absorption,
            mie_g: self.mie_anisotropy.clamp(-0.999, 0.999),
            ozone_absorption: self.ozone_absorption,
            ozone_center: self.ozone_center_km,
            ground_albedo: self.ground_albedo,
            ozone_width: self.ozone_width_km.max(0.0),
            bottom_radius: bottom,
            top_radius: bottom + self.thickness_km.max(0.001),
            sun_angular_radius: self.sun_angular_radius_deg.to_radians(),
            enabled: 1,
        }
    }
}

fn remove_atmosphere(world: &mut pulsar_scenedb::World, entity: pulsar_scenedb::Entity) {
    world.remove::<AtmosphereComponent>(entity);
    world.remove::<AtmosphereRow>(entity);
}

#[register_world_component(remove = remove_atmosphere)]
#[register_runtime_behavior]
impl ComponentRuntimeBehavior for AtmosphereComponent {
    const CLASS_NAME: &'static str = "AtmosphereComponent";
    fn sync_component(owner: &RuntimeComponentOwner, _index: usize, component: &Self, context: &mut dyn ComponentRuntimeContext) {
        let Some(entity) = context.subsystems_mut().get_mut::<pulsar_scenedb::Entity>().copied() else { return; };
        let writes = get_subsystem!(context, PendingWorldWrites);
        let row = component.enabled.then(|| component.to_row(owner));
        writes.push(move |world| {
            if let Some(row) = row { world.insert(entity, row); }
            else { world.remove::<AtmosphereRow>(entity); }
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn with_owner<R>(position: [f32; 3], f: impl FnOnce(&RuntimeComponentOwner<'_>) -> R) -> R {
        let props = std::collections::HashMap::new();
        f(&RuntimeComponentOwner { scene_object_id: "air", position, rotation: [0.0; 3], scale: [1.0; 3], props: &props })
    }

    #[test]
    fn defaults_are_earths_air_on_flat_ground() {
        let row = with_owner([5.0, 6.0, 7.0], |owner| AtmosphereComponent::default().to_row(owner));
        let earth = AtmosphereRow::earth();
        assert_eq!(row.placement, placement::GROUND_AT_ORIGIN);
        assert_eq!(row.center, [0.0; 3]);
        assert!((row.bottom_radius - earth.bottom_radius).abs() < 1e-3);
        assert!((row.top_radius - earth.top_radius).abs() < 1e-3);
        assert!((row.sun_angular_radius - earth.sun_angular_radius).abs() < 1e-6);
        assert_eq!(row.rayleigh_scattering, earth.rayleigh_scattering);
        assert_eq!(row.enabled, 1);
    }

    #[test]
    fn a_planet_is_centred_on_its_owner() {
        let component = AtmosphereComponent {
            placement: AtmospherePlacement::PlanetAtOwner,
            planet_radius_km: 1737.0,
            thickness_km: 40.0,
            ..Default::default()
        };
        let row = with_owner([1.0, -2.0, 3.0], |owner| component.to_row(owner));
        assert_eq!(row.placement, placement::CENTER);
        assert_eq!(row.center, [1.0, -2.0, 3.0]);
        assert_eq!((row.bottom_radius, row.top_radius), (1737.0, 1777.0));
    }
}
