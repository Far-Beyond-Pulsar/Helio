//! Components this engine keeps but does not consume say so
//! (Pulsar-Native#1035, Phase 4): each declares a reason, which the
//! properties card shows and the first attach logs.

// Links helio_component's registrations into this test binary.
use helio_component::components::LODComponent as _;

#[test]
fn unsupported_components_declare_why() {
    for class in [
        "LODComponent",
        "ReflectionCaptureComponent",
        "PortalComponent",
        "VoxelComponent",
    ] {
        let reason = pulsar_world_registry::unsupported_reason(class);
        assert!(
            reason.is_some_and(|reason| !reason.is_empty()),
            "{class} declares no reason"
        );
    }
    for class in [
        "StaticMeshComponent",
        "LightComponent",
        "GlobalFogComponent",
        "LocalFogVolumeComponent",
        "PostProcessVolumeComponent",
        "CameraPostProcessComponent",
        "WaterVolumeComponent",
        "FoliageComponent",
        "VoxelTerrainComponent",
    ] {
        assert_eq!(
            pulsar_world_registry::unsupported_reason(class),
            None,
            "{class}"
        );
    }
}
