//! Light conversion from SolidRS to Helio

use helio_render_v2::scene::SceneLight;
use solid_rs::scene::{Light, DirectionalLight, PointLight, SpotLight, AreaLight};

/// Convert a SolidRS light to Helio's SceneLight
pub fn convert_light(light: &Light) -> Option<SceneLight> {
    match light {
        Light::Directional(dir_light) => Some(convert_directional(dir_light)),
        Light::Point(point_light) => Some(convert_point(point_light)),
        Light::Spot(spot_light) => Some(convert_spot(spot_light)),
        Light::Area(area_light) => {
            log::warn!(
                "Area light '{}' not supported in Helio - converting to point light",
                area_light.base.name
            );
            Some(convert_area_as_point(area_light))
        }
    }
}

fn convert_directional(light: &DirectionalLight) -> SceneLight {
    // DirectionalLight has a base (name, color, intensity) but no direction field
    // In SolidRS, directional lights point in the -Z direction of their node transform
    // For now, we'll use a default downward direction
    SceneLight::directional(
        [0.0, -1.0, 0.0], // Default: pointing down
        [light.base.color.x, light.base.color.y, light.base.color.z],
        light.base.intensity,
    )
}

fn convert_point(light: &PointLight) -> SceneLight {
    // PointLight has position implicitly from the node transform
    // For now, we'll use origin - the actual position will be set from the node
    let range = light.range.unwrap_or(10.0); // Default range if not specified

    SceneLight::point(
        [0.0, 0.0, 0.0], // Position will be set from node transform
        [light.base.color.x, light.base.color.y, light.base.color.z],
        light.base.intensity,
        range,
    )
}

fn convert_spot(light: &SpotLight) -> SceneLight {
    let range = light.range.unwrap_or(10.0);

    // Convert angles to radians (SolidRS stores in radians already)
    // inner_cone_angle and outer_cone_angle are full cone angles, we need half-angles
    let inner_angle = light.inner_cone_angle / 2.0;
    let outer_angle = light.outer_cone_angle / 2.0;

    SceneLight::spot(
        [0.0, 0.0, 0.0], // Position from node transform
        [0.0, 0.0, -1.0], // Direction from node transform (-Z)
        [light.base.color.x, light.base.color.y, light.base.color.z],
        light.base.intensity,
        range,
        inner_angle,
        outer_angle,
    )
}

fn convert_area_as_point(light: &AreaLight) -> SceneLight {
    // Convert area light to point light as a fallback
    // Use the area size to estimate an appropriate range
    let range = (light.width.max(light.height) * 5.0).max(10.0);

    SceneLight::point(
        [0.0, 0.0, 0.0],
        [light.base.color.x, light.base.color.y, light.base.color.z],
        light.base.intensity,
        range,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::Vec3;
    use solid_rs::scene::LightBase;

    #[test]
    fn test_convert_directional_light() {
        let light = DirectionalLight {
            base: LightBase {
                name: "Sun".to_string(),
                color: Vec3::new(1.0, 0.9, 0.8),
                intensity: 5.0,
            },
        };

        let scene_light = convert_directional(&light);
        assert_eq!(scene_light.color, [1.0, 0.9, 0.8]);
        assert_eq!(scene_light.intensity, 5.0);
    }

    #[test]
    fn test_convert_point_light() {
        let light = PointLight {
            base: LightBase {
                name: "Bulb".to_string(),
                color: Vec3::ONE,
                intensity: 100.0,
            },
            range: Some(15.0),
        };

        let scene_light = convert_point(&light);
        assert_eq!(scene_light.range, 15.0);
        assert_eq!(scene_light.intensity, 100.0);
    }
}
