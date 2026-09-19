//! Standalone target for exercising `scenedb-inspector-agent` end to end,
//! independent of any Helio example's own threading model. Spawns a small,
//! constantly-churning `World` (two archetypes: Position-only, and
//! Position+Health) and publishes it via the agent whenever
//! `SCENEDB_INSPECTOR_SHM` is set -- i.e. whenever this is launched by the
//! `scenedb_inspector` client rather than run directly.
//!
//! Run directly (agent stays off, just logs to confirm the sim runs):
//!   cargo run -p scenedb-inspector-agent --example demo_target
//!
//! Run under the inspector: see that client's own docs.

use std::sync::{Arc, Mutex};
use std::time::Duration;

use pulsar_reflection::Reflectable;
use pulsar_scenedb::World;

// `Reflectable` is what lets the inspector decode these into structured,
// nested JSON (`ArchetypeColumnSnapshot::rows_reflected`) instead of just
// raw hex -- see world_telemetry.rs's doc comment in the SceneDB patch.
// `Vec3` here is deliberately a nested struct (not a bare [f32; 3]) to
// exercise that nesting, not just top-level primitive fields.

#[repr(C)]
#[derive(Clone, Copy, Reflectable)]
struct Vec3 {
    x: f32,
    y: f32,
    z: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Reflectable)]
struct Position {
    pos: Vec3,
}

#[repr(C)]
#[derive(Clone, Copy, Reflectable)]
struct Health {
    value: f32,
}

fn main() {
    env_logger::try_init().ok();

    let world = Arc::new(Mutex::new(World::new()));

    // Position-only archetype.
    let mut movers = Vec::new();
    for i in 0..8u32 {
        let mut w = world.lock().unwrap();
        let e = w.spawn();
        w.insert(
            e,
            Position {
                pos: Vec3 {
                    x: i as f32,
                    y: 0.0,
                    z: 0.0,
                },
            },
        );
        movers.push(e);
    }

    // Position+Health archetype.
    let mut actors = Vec::new();
    for i in 0..5u32 {
        let mut w = world.lock().unwrap();
        let e = w.spawn();
        w.insert(
            e,
            Position {
                pos: Vec3 {
                    x: 0.0,
                    y: i as f32,
                    z: 0.0,
                },
            },
        );
        w.insert(e, Health { value: 100.0 });
        actors.push(e);
    }

    let agent_world = world.clone();
    let agent = scenedb_inspector_agent::maybe_start(Duration::from_millis(100), move || {
        agent_world.lock().unwrap().telemetry_snapshot()
    });
    match &agent {
        Some(_) => log::info!("demo_target: inspector agent started (SCENEDB_INSPECTOR_SHM set)"),
        None => log::info!(
            "demo_target: inspector agent NOT started (SCENEDB_INSPECTOR_SHM unset) -- \
             running the sim standalone"
        ),
    }

    let mut frame = 0u64;
    loop {
        {
            let mut w = world.lock().unwrap();
            for (i, &e) in movers.iter().enumerate() {
                if let Some(mut pos) = w.get_mut::<Position>(e) {
                    pos.pos.x = i as f32 + (frame as f32 * 0.05).sin();
                }
            }
            for (i, &e) in actors.iter().enumerate() {
                if let Some(mut hp) = w.get_mut::<Health>(e) {
                    hp.value = 100.0 - ((frame as f32 * 0.5 + i as f32 * 10.0) % 100.0);
                }
            }
        }

        if frame % 50 == 0 {
            log::info!("demo_target: frame {frame}");
        }
        frame += 1;
        std::thread::sleep(Duration::from_millis(16));

        if agent.is_none() && frame >= 300 {
            log::info!("demo_target: standalone run complete, exiting");
            break;
        }
    }
}
