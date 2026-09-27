//! Local fixture timing and continuous motion, using the ordinary full graph.
//! This does not qualify planetary streaming or a populated editor workload.
use super::*;

pub(super) fn run(flight: &mut Flight, output: &Path, selected: &str) {
    let (name, world, eye, forward) = if selected == "ground-close" {
        (
            "ground-close",
            (*flight.world).clone(),
            flight.world.ground_spawn(0.0, 0.0, 3.0),
            Vec3::new(0.0, -0.8, -1.0),
        )
    } else {
        surface_reference::cases((*flight.world).clone())
            .into_iter()
            .find(|(name, _, _, _)| *name == selected)
            .expect("unknown cache benchmark fixture")
    };
    let fixed_jitter = std::env::var_os("HELIO_VOXEL_CACHE_FIXED_JITTER").is_some();
    if fixed_jitter { flight.renderer.set_jitter_enabled(false); flight.renderer.set_camera_jitter_override(Some([0.0, 0.0])); }
    let record = std::env::var_os("HELIO_VOXEL_FLIGHT_RECORD").is_some();
    flight.world = Arc::new(world);
    flight.raytraced_sun = true;
    flight.renderer.set_frame_delta_override(Some(1.0 / 60.0));
    let forward = forward.normalize();
    let right = forward.cross(Vec3::Y).normalize();
    flight.settle("ground_load", eye, forward);
    #[cfg(feature = "voxel-surface-cache")]
    eprintln!(
        "VOXEL_CACHE_BENCH_PATCH {:?}",
        flight
            .renderer
            .find_pass::<LazyEngineVoxelPass>()
            .unwrap()
            .surface_patch_stats()
            .unwrap()
    );
    fs::write(
        output.join("benchmark.json"),
        serde_json::to_vec_pretty(&serde_json::json!({
            "fixture": name, "size": flight.size,
            "voxel_size_m": flight.world.voxel_size(),
            "recording": record, "raytraced_sun": true,
            "warm_frames": 32, "frames_per_stage": 180,
            "motion": "one cycle: 0.5 m sideways, 0.1 m vertically, 0.04 rad yaw",
            "fixed_jitter": fixed_jitter,
            "mesh_enabled": std::env::var_os("HELIO_VOXEL_SURFACE_MESH").is_some(),
            "cache_feature": cfg!(feature="voxel-surface-cache"),
            "cache_off": std::env::var_os("HELIO_VOXEL_SURFACE_CACHE_OFF").is_some(),
            "skip_off": std::env::var_os("HELIO_VOXEL_SURFACE_CACHE_SKIP_OFF").is_some(),
            "scope": "local settled fixture, synchronized headless graph, excludes presentation"
        }))
        .unwrap(),
    )
    .unwrap();
    for stage in ["cache_stationary", "cache_motion"] {
        for _ in 0..32 {
            flight.draw("cache_warm", eye, forward);
        }
        for frame in 0..180 {
            let phase = if stage == "cache_motion" {
                frame as f64 / 179.0 * std::f64::consts::TAU
            } else {
                0.0
            };
            let offset =
                right.as_dvec3() * (phase.sin() * 0.5) + DVec3::Y * ((phase * 2.0).sin() * 0.1);
            let direction = (forward + right * (phase.sin() * 0.04) as f32).normalize();
            flight.draw(stage, eye + offset, direction);
            if record && stage == "cache_motion" {
                flight.capture_options(&output.join(format!("motion-{frame:03}.png")), false);
            }
        }
        // Let asynchronous timestamps for every measured frame arrive. These
        // extra frames have their own label and are excluded from statistics.
        for _ in 0..4 {
            flight.draw("cache_drain", eye, forward);
        }
        flight.capture_options(&output.join(format!("{stage}.png")), false);
    }
    eprintln!("VOXEL_CACHE_BENCH_COMPLETE fixture={name} recording={record}");
}
