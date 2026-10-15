//! A planet's air: the sky, the sun's colour through it, the sky's light on
//! surfaces and aerial perspective. Any scene may have one: a planet
//! (centred on the owner) or flat ground at the world's origin. The sun is
//! the scene's first directional light.
//!
//! SceneDB derives an `AtmosphereSourceRow` (`environment_rows`) from the
//! authored value; the environment join places it as the sky pass's
//! `helio_pass_sky::AtmosphereComponent` row in `"atmospheres"`.
use engine_class_derive::{engine_class, register_world_component};
use helio_pass_sky::AtmosphereComponent as AtmosphereRow;
use helio_pass_sky::atmosphere::placement;
use pulsar_reflection::Reflectable;
use serde::{Deserialize, Serialize};

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

    /// The pass row in the owner's space: a planet placed at its owner has
    /// a zero centre, which the environment join moves to the owner's world
    /// position (`AtmosphereSourceRow`).
    pub fn to_row(&self) -> AtmosphereRow {
        let bottom = self.planet_radius_km.max(0.001);
        let placement = match self.placement {
            AtmospherePlacement::GroundAtOrigin => placement::GROUND_AT_ORIGIN,
            AtmospherePlacement::PlanetAtOwner => placement::CENTER,
        };
        AtmosphereRow {
            center: [0.0; 3],
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
            enabled: u32::from(self.enabled),
        }
    }
}

impl AtmosphereComponent {
    /// The air a voxel world's `air` settings describe, on this sky's
    /// planet (placement, radius, ground and sun kept): `None` for `Custom` (the
    /// authored sky stays); `VoxelAir::None` is a disabled sky. `density`
    /// scales every coefficient, `tint` the molecular scattering per
    /// channel (the sky's colour), `haze` the aerosols.
    pub fn for_air(&self, air: crate::VoxelAir, density: f32, tint: [f32; 3], haze: f32) -> Option<Self> {
        use crate::VoxelAir;
        let preset = match air {
            VoxelAir::Custom => return None,
            VoxelAir::None => return Some(Self { enabled: false, ..self.clone() }),
            VoxelAir::Earthlike => Self::from_row(&AtmosphereRow::earth()),
            VoxelAir::Thin => Self::from_row(&AtmosphereRow::mars()),
            VoxelAir::Dense => {
                // A deep haze over a tall atmosphere (Titan-like).
                let earth = Self::from_row(&AtmosphereRow::earth());
                Self {
                    thickness_km: 200.0,
                    rayleigh_scattering: earth.rayleigh_scattering.map(|v| v * 1.5),
                    rayleigh_scale_height_km: 20.0,
                    mie_scattering: earth.mie_scattering.map(|v| v * 12.0),
                    mie_absorption: [0.012, 0.018, 0.035],
                    mie_scale_height_km: 6.0,
                    mie_anisotropy: 0.6,
                    ozone_absorption: [0.0; 3],
                    ..earth
                }
            }
        };
        let density = density.max(0.0);
        let haze = haze.max(0.0);
        let times = |v: [f32; 3], k: [f32; 3]| [v[0] * k[0], v[1] * k[1], v[2] * k[2]];
        Some(Self {
            enabled: density > 0.0,
            placement: self.placement,
            planet_radius_km: self.planet_radius_km,
            sun_angular_radius_deg: self.sun_angular_radius_deg,
            ground_albedo: self.ground_albedo,
            rayleigh_scattering: times(preset.rayleigh_scattering, tint.map(|t| t.max(0.0) * density)),
            mie_scattering: preset.mie_scattering.map(|v| v * density * haze),
            mie_absorption: preset.mie_absorption.map(|v| v * density * haze),
            ozone_absorption: preset.ozone_absorption.map(|v| v * density),
            ..preset
        })
    }
}

#[register_world_component]
impl AtmosphereComponent {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_are_earths_air_on_flat_ground() {
        let row = AtmosphereComponent::default().to_row();
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
    fn a_worlds_air_derives_its_sky() {
        use crate::VoxelAir;
        let sky = AtmosphereComponent { placement: AtmospherePlacement::PlanetAtOwner, planet_radius_km: 1737.0, ..Default::default() };
        assert!(sky.for_air(VoxelAir::Custom, 1.0, [1.0; 3], 1.0).is_none(), "custom keeps the authored sky");
        assert!(!sky.for_air(VoxelAir::None, 1.0, [1.0; 3], 1.0).unwrap().enabled, "an airless world has no sky");
        let earth = sky.for_air(VoxelAir::Earthlike, 1.0, [1.0; 3], 1.0).unwrap();
        assert!(earth.enabled);
        assert_eq!((earth.placement, earth.planet_radius_km), (AtmospherePlacement::PlanetAtOwner, 1737.0), "the planet stays");
        assert_eq!(earth.rayleigh_scattering, AtmosphereRow::earth().rayleigh_scattering);
        let thin = sky.for_air(VoxelAir::Earthlike, 0.5, [1.0, 0.5, 1.0], 2.0).unwrap();
        assert!((thin.rayleigh_scattering[1] - earth.rayleigh_scattering[1] * 0.25).abs() < 1e-9);
        assert!((thin.mie_scattering[0] - earth.mie_scattering[0]).abs() < 1e-9, "half the air, twice the haze");
        assert!(!sky.for_air(VoxelAir::Earthlike, 0.0, [1.0; 3], 1.0).unwrap().enabled, "no air at all");
    }

    #[test]
    fn a_planet_is_placed_at_its_owner_by_the_join() {
        let component = AtmosphereComponent {
            placement: AtmospherePlacement::PlanetAtOwner,
            planet_radius_km: 1737.0,
            thickness_km: 40.0,
            ..Default::default()
        };
        let row = component.to_row();
        assert_eq!(row.placement, placement::CENTER);
        assert_eq!(
            row.center, [0.0; 3],
            "local: the join adds the owner's position"
        );
        assert_eq!((row.bottom_radius, row.top_radius), (1737.0, 1777.0));
        let disabled = AtmosphereComponent {
            enabled: false,
            ..component
        };
        assert_eq!(disabled.to_row().enabled, 0);
    }
}
