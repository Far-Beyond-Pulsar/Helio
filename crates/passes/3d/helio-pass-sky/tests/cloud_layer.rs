//! The atmosphere row's cloud fields.
use helio_pass_sky::{AtmosphereComponent, CloudLayer};

#[test]
fn skies_have_no_clouds_until_given_a_layer() {
    assert_eq!(AtmosphereComponent::earth().clouds(), None);
    assert_eq!(AtmosphereComponent::mars().clouds(), None);
    let layer = CloudLayer { coverage: 0.3, wind: [4.0, 0.0, -1.0], ..CloudLayer::cumulus() };
    let row = AtmosphereComponent::earth().around_planet([1.0, 2.0, 3.0], 16_000.0).with_clouds(layer);
    assert_eq!(row.clouds(), Some(layer));
    assert_eq!(row.enabled, 1, "the air stays enabled");
    assert_eq!(row.bottom_radius, 16.0, "the planet stays");
}

#[test]
fn a_layer_is_clamped_to_what_the_shader_can_draw() {
    let row = AtmosphereComponent::earth().with_clouds(CloudLayer {
        bottom_km: 2.0,
        top_km: 1.0,
        coverage: 3.0,
        kind: -1.0,
        extinction: -5.0,
        weather_km: 0.0,
        albedo: [1.5, 0.5, -1.0],
        ..CloudLayer::overcast()
    });
    let clouds = row.clouds().unwrap();
    assert!(clouds.top_km > clouds.bottom_km);
    assert_eq!((clouds.coverage, clouds.kind, clouds.extinction), (1.0, 0.0, 0.0));
    assert!(clouds.weather_km > 0.0);
    assert_eq!(clouds.albedo, [1.0, 0.5, 0.0]);
}
