//! Components this engine keeps but does not consume yet say so
//! (Pulsar-Native#1035, Phase 4): each declares a reason and its tracking
//! issue, which the properties card shows and the first attach logs.

// Links helio_component's registrations into this test binary.
use helio_component::components::LODComponent as _;

#[test]
fn unfinished_components_declare_why_and_their_issue() {
    for (class, issue) in [
        ("LODComponent", 1053),
        ("ReflectionCaptureComponent", 1054),
        ("PortalComponent", 1055),
    ] {
        let unfinished = pulsar_world_registry::unfinished_component(class)
            .unwrap_or_else(|| panic!("{class} is not declared unfinished"));
        assert!(!unfinished.reason.is_empty(), "{class} declares no reason");
        assert_eq!(
            unfinished.issue,
            format!("https://github.com/Far-Beyond-Pulsar/Pulsar-Native/issues/{issue}"),
            "{class}"
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
        // Drawn as a mesh since #1056.
        "VoxelComponent",
    ] {
        assert!(
            pulsar_world_registry::unfinished_component(class).is_none(),
            "{class}"
        );
    }
}
