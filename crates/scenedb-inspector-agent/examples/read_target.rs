//! Minimal manual reader: attaches to a shared-memory segment a running
//! `demo_target` (or any `maybe_start`-instrumented process) is publishing
//! to, and prints the decoded snapshot's shape once a second. Mainly a
//! smoke test for `ring.rs`'s reader side and a worked reference for the
//! standalone `scenedb_inspector` client's own (hand-copied) reader.
//!
//! Usage:
//!   SCENEDB_INSPECTOR_SHM=demo cargo run -p scenedb-inspector-agent --example demo_target &
//!   cargo run -p scenedb-inspector-agent --example read_target -- demo

use std::time::Duration;

use scenedb_inspector_agent::ring::RingView;
use shared_memory::ShmemConf;

fn main() {
    env_logger::try_init().ok();

    let name = std::env::args().nth(1).unwrap_or_else(|| {
        eprintln!("usage: read_target <shm-name>");
        std::process::exit(1);
    });

    // The agent creates the segment only after the target process starts
    // and calls maybe_start/start, so poll for it rather than failing on
    // the first attempt.
    let shmem = loop {
        match ShmemConf::new().os_id(&name).open() {
            Ok(s) => break s,
            Err(e) => {
                log::info!("waiting for shared memory '{name}' ({e}) ...");
                std::thread::sleep(Duration::from_millis(250));
            }
        }
    };

    // SAFETY: we just successfully opened a segment a `scenedb-inspector-agent`
    // writer has (or is) initializing; `attach` itself validates magic/version
    // before trusting anything else about the layout.
    let view = unsafe { RingView::attach(shmem.as_ptr()) }.expect("attach failed");
    log::info!(
        "attached: {} slots x {} bytes",
        view.slot_count(),
        view.slot_capacity()
    );

    loop {
        match view.read_latest(8) {
            Some(bytes) => match serde_json::from_slice::<serde_json::Value>(&bytes) {
                Ok(v) => {
                    let entity_count = v.get("entity_count").and_then(|x| x.as_u64());
                    let archetype_count = v
                        .get("archetypes")
                        .and_then(|a| a.as_array())
                        .map(|a| a.len());
                    log::info!(
                        "snapshot: {} bytes, entity_count={:?}, archetypes={:?}",
                        bytes.len(),
                        entity_count,
                        archetype_count
                    );
                    if let Some(archetypes) = v.get("archetypes").and_then(|a| a.as_array()) {
                        for arch in archetypes {
                            if let Some(cols) = arch.get("columns").and_then(|c| c.as_array()) {
                                for col in cols {
                                    let reflected = col.get("rows_reflected");
                                    log::info!(
                                        "  column {:?}: rows_reflected = {}",
                                        col.get("type_name"),
                                        serde_json::to_string(reflected.unwrap_or(&serde_json::Value::Null)).unwrap_or_default()
                                    );
                                }
                            }
                        }
                    }
                }
                Err(e) => log::warn!("decode failed: {e}"),
            },
            None => log::info!("no snapshot published yet"),
        }
        std::thread::sleep(Duration::from_secs(1));
    }
}
