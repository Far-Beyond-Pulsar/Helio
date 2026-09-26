//! A bounded, coalescing save worker. Ordinary destruction appends edit records;
//! replacement/undo writes a new atomic checkpoint. Shutdown drains pending work.
use crate::World;
use std::{
    io::Write,
    path::PathBuf,
    sync::{Arc, Condvar, Mutex},
    thread::JoinHandle,
};
struct State {
    pending: Option<Arc<World>>,
    stop: bool,
    error: Option<String>,
}
pub struct JournalWriter {
    state: Arc<(Mutex<State>, Condvar)>,
    worker: Option<JoinHandle<()>>,
}
impl JournalWriter {
    pub fn new(path: PathBuf) -> Self {
        let state = Arc::new((
            Mutex::new(State {
                pending: None,
                stop: false,
                error: None,
            }),
            Condvar::new(),
        ));
        let shared = state.clone();
        let worker = std::thread::Builder::new()
            .name("voxel-save".into())
            .spawn(move || {
                let mut saved = if path.exists() {
                    World::load(&path).ok().map(|w| w.edits)
                } else {
                    None
                };
                loop {
                    let (lock, wake) = &*shared;
                    let mut state = lock.lock().unwrap();
                    while state.pending.is_none() && !state.stop {
                        state = wake.wait(state).unwrap();
                    }
                    let Some(world) = state.pending.take() else {
                        break;
                    };
                    drop(state);
                    let result = (|| -> Result<(), String> {
                        if let Some(previous) = &saved {
                            if previous.len() <= world.edits.len()
                                && previous.iter().eq(world.edits[..previous.len()].iter())
                            {
                                if previous.len() == world.edits.len() {
                                    return Ok(());
                                }
                                let mut file = std::fs::OpenOptions::new()
                                    .append(true)
                                    .open(&path)
                                    .map_err(|e| e.to_string())?;
                                let mut bytes = Vec::new();
                                for edit in &world.edits[previous.len()..] {
                                    bytes.push(b'\n');
                                    serde_json::to_writer(&mut bytes, edit)
                                        .map_err(|e| e.to_string())?;
                                }
                                file.write_all(&bytes)
                                    .and_then(|_| file.sync_data())
                                    .map_err(|e| e.to_string())?;
                                return Ok(());
                            }
                        }
                        world.save(&path)
                    })();
                    match result {
                        Ok(()) => saved = Some(world.edits.clone()),
                        Err(error) => {
                            saved = None;
                            eprintln!("VOXEL_SAVE {}: {error}", path.display());
                            shared.0.lock().unwrap().error = Some(error);
                        }
                    }
                }
            })
            .expect("voxel save worker");
        Self {
            state,
            worker: Some(worker),
        }
    }
    pub fn save(&self, world: Arc<World>) {
        self.state.0.lock().unwrap().pending = Some(world);
        self.state.1.notify_one();
    }
    pub fn take_error(&self) -> Option<String> {
        self.state.0.lock().unwrap().error.take()
    }
}
impl Drop for JournalWriter {
    fn drop(&mut self) {
        self.state.0.lock().unwrap().stop = true;
        self.state.1.notify_one();
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::world::Edit;

    #[test]
    fn shutdown_drains_append_and_replacement_journals() {
        let path = std::env::temp_dir().join(format!(
            "helio-journal-{}-{}.json",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let mut world = World::default();
        let save = |world: &World| {
            let writer = JournalWriter::new(path.clone());
            writer.save(Arc::new(world.clone()));
            drop(writer);
        };
        save(&world);
        let checkpoint = std::fs::read(&path).unwrap();
        world
            .apply_edit(Edit {
                cell: [1, 63_710_000, 1],
                radius: 0.5,
                material: 0,
            })
            .unwrap();
        save(&world);
        let appended = std::fs::read(&path).unwrap();
        assert!(
            appended.starts_with(&checkpoint),
            "ordinary edit must append"
        );
        assert!(World::load(&path).unwrap().edits == world.edits);
        world.edits.clear();
        world.rebuild_edits();
        save(&world);
        assert!(
            World::load(&path).unwrap().edits.is_empty(),
            "undo must replace the checkpoint"
        );
        // A damaged tail must be repaired by a complete checkpoint, not by
        // appending records after corrupt data.
        std::fs::OpenOptions::new()
            .append(true)
            .open(&path)
            .unwrap()
            .write_all(b"\n{broken")
            .unwrap();
        world
            .apply_edit(Edit {
                cell: [1, 63_710_000, 1],
                radius: 0.5,
                material: 3,
            })
            .unwrap();
        save(&world);
        assert!(World::load(&path).unwrap().edits == world.edits);
        std::fs::remove_file(path).unwrap();
    }
}
