//! Bounded experimental cache accelerator for canonical reference rendering.
//! Missing pages always fall back to the same published canonical source.
use crate::{
    surface_cache::{Brick, Cache, Key},
    Params, World,
};
use std::sync::{
    atomic::{AtomicU64, Ordering},
    mpsc, Arc, Condvar, Mutex,
};
use std::time::Instant;
use wgpu::util::DeviceExt;

const SIDE: i32 = 8;
const TILES: usize = SIDE as usize * SIDE as usize * SIDE as usize;
// Worst possible dictionary payload: header + descriptors + 512 mixed blocks.
const WORDS: usize = 1 + 512 + 512 * 4;
const UPLOADS: usize = 8;

#[cfg(test)]
mod tests;

#[derive(Clone, Copy, Default, Debug, serde::Serialize)]
pub struct Stats {
    pub enabled: bool,
    pub revision: u64,
    pub low: [i32; 3],
    pub ready: usize,
    pub requested: usize,
    pub uploaded_bytes: u64,
    pub discarded_results: u64,
    pub construction_ms: f64,
}
struct Request {
    world: Arc<World>,
    low: Key,
    revision: u64,
}
struct Pending {
    request: Option<Request>,
    stop: bool,
}
struct Completed {
    key: Key,
    revision: u64,
    brick: Arc<Brick>,
    milliseconds: f64,
}

pub(super) struct Patch {
    primary: wgpu::ComputePipeline,
    pub settings: wgpu::Buffer,
    pub directory: wgpu::Buffer,
    pub words: wgpu::Buffer,
    pending: Arc<(Mutex<Pending>, Condvar)>,
    serial: Arc<AtomicU64>,
    receiver: Option<Mutex<mpsc::Receiver<Completed>>>,
    worker: Option<std::thread::JoinHandle<()>>,
    world: Option<Arc<World>>,
    low: Key,
    stats: Stats,
}

fn shader_source() -> String {
    format!(
        "{}\n{}\n{}\n{}\n{}",
        crate::SHADER,
        crate::surface_cache::GPU_SHADER,
        include_str!("surface_patch.wgsl"),
        include_str!("surface_patch_predicates.wgsl"),
        include_str!("surface_patch_primary.wgsl")
    )
}

impl Patch {
    pub fn new(device: &wgpu::Device) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("bounded exact surface patch primary"),
            source: wgpu::ShaderSource::Wgsl(shader_source().into()),
        });
        let primary = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("surface_patch_primary"),
            layout: None,
            module: &shader,
            entry_point: Some("surface_patch_primary"),
            compilation_options: Default::default(),
            cache: None,
        });
        let pending = Arc::new((
            Mutex::new(Pending {
                request: None,
                stop: false,
            }),
            Condvar::new(),
        ));
        let serial = Arc::new(AtomicU64::new(0));
        let (sender, receiver) = mpsc::sync_channel(UPLOADS);
        let state = pending.clone();
        let requested = serial.clone();
        let worker = std::thread::Builder::new()
            .name("voxel-surface-patch".into())
            .spawn(move || {
                let mut cache = Cache::new(8 * 1024 * 1024, 1024);
                loop {
                    let request = {
                        let (lock, signal) = &*state;
                        let mut state = lock.lock().unwrap();
                        while state.request.is_none() && !state.stop {
                            state = signal.wait(state).unwrap();
                        }
                        if state.stop {
                            return;
                        }
                        state.request.take().unwrap()
                    };
                    cache.set_world(request.world);
                    let mut keys: Vec<_> = (0..TILES)
                        .map(|i| {
                            Key([
                                request.low.0[0] + (i % 8) as i32,
                                request.low.0[1] + (i / 8 % 8) as i32,
                                request.low.0[2] + (i / 64) as i32,
                            ])
                        })
                        .collect();
                    keys.sort_by_key(|key| {
                        (0..3)
                            .map(|a| (key.0[a] - request.low.0[a] - SIDE / 2).pow(2))
                            .sum::<i32>()
                    });
                    for key in keys {
                        if requested.load(Ordering::Acquire) != request.revision {
                            break;
                        }
                        let start = Instant::now();
                        let brick = cache
                            .get(key)
                            .expect("one brick fits the declared cache budget");
                        let milliseconds = start.elapsed().as_secs_f64() * 1000.0;
                        if requested.load(Ordering::Acquire) != request.revision {
                            break;
                        }
                        // Backpressure bounds completed CPU payloads. A superseded
                        // result may remain queued; the render thread rejects it.
                        if sender
                            .send(Completed {
                                key,
                                revision: request.revision,
                                brick,
                                milliseconds,
                            })
                            .is_err()
                        {
                            return;
                        }
                    }
                }
            })
            .unwrap();
        Self {
            primary,
            settings: device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("surface patch domain"),
                contents: bytemuck::cast_slice(&[0u32; 8]),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            }),
            directory: device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("surface patch directory"),
                contents: bytemuck::cast_slice(&[u32::MAX; TILES]),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            }),
            words: super::terrain::buffer(
                device,
                "surface patch exact materials",
                (TILES * WORDS * 4) as u64,
            ),
            pending,
            serial,
            receiver: Some(Mutex::new(receiver)),
            worker: Some(worker),
            world: None,
            low: Key([0; 3]),
            stats: Stats {
                enabled: std::env::var_os("HELIO_VOXEL_SURFACE_CACHE_OFF").is_none(),
                ..Default::default()
            },
        }
    }

    pub fn stats(&self) -> Stats {
        self.stats
    }

    pub fn encode_primary(
        &self,
        device: &wgpu::Device,
        params: &wgpu::Buffer,
        hits: &wgpu::Buffer,
        encoder: &mut wgpu::CommandEncoder,
        size: [u32; 2],
    ) {
        if !self.stats.enabled {
            return;
        }
        let buffers = [
            (0, params),
            (9, hits),
            (31, &self.settings),
            (32, &self.directory),
            (33, &self.words),
        ];
        let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("surface patch primary inputs"),
            layout: &self.primary.get_bind_group_layout(0),
            entries: &buffers.map(|(binding, b)| wgpu::BindGroupEntry {
                binding,
                resource: b.as_entire_binding(),
            }),
        });
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&self.primary);
        pass.set_bind_group(0, &group, &[]);
        pass.dispatch_workgroups(size[0].div_ceil(8), size[1].div_ceil(8), 1);
    }

    pub fn update(&mut self, queue: &wgpu::Queue, world: &Arc<World>, params: &Params) {
        if !self.stats.enabled {
            return;
        }
        // A bounded prediction, not geometry authority. Flat/horizon views
        // default to 8 m ahead; the procedural radial estimate is capped at
        // 64 m. Edited floating geometry is still queried canonically.
        let eye = std::array::from_fn::<_, 3, _>(|a| {
            f64::from(params.origin[a]) + f64::from(params.fraction[a])
        });
        let radial = glam::DVec3::from_array(eye).normalize_or_zero();
        let forward = glam::DVec3::new(
            f64::from(params.forward[0]),
            f64::from(params.forward[1]),
            f64::from(params.forward[2]),
        );
        let denominator = -forward.dot(radial);
        let travel = if denominator > 0.1 {
            (-world.density(params.origin[..3].try_into().unwrap()) / denominator).clamp(8.0, 64.0)
        } else {
            8.0
        };
        let center = Key::containing(
            std::array::from_fn(|a| (eye[a] + forward[a] * travel * 10.0).floor() as i32),
            world.voxel_step(),
        );
        let same_world = self
            .world
            .as_ref()
            .is_some_and(|old| Arc::ptr_eq(old, world));
        let inside = (0..3).all(|a| {
            center.0[a] >= self.low.0[a] + SIDE / 4 && center.0[a] < self.low.0[a] + SIDE * 3 / 4
        });
        if !same_world || !inside {
            let revision = self.serial.fetch_add(1, Ordering::AcqRel) + 1;
            self.low = Key(center.0.map(|v| v - SIDE / 2));
            self.world = Some(world.clone());
            self.stats = Stats {
                enabled: true,
                revision,
                low: self.low.0,
                requested: TILES,
                ..Default::default()
            };
            // Invalidate the GPU directory before admitting the new source.
            // Payload storage may retain bytes, but no stale address is live.
            queue.write_buffer(&self.directory, 0, bytemuck::cast_slice(&[u32::MAX; TILES]));
            let settings = [
                self.low.0[0] as u32,
                self.low.0[1] as u32,
                self.low.0[2] as u32,
                SIDE as u32,
                world.voxel_step(),
                0,
                0,
                0,
            ];
            queue.write_buffer(&self.settings, 0, bytemuck::cast_slice(&settings));
            let (lock, signal) = &*self.pending;
            lock.lock().unwrap().request = Some(Request {
                world: world.clone(),
                low: self.low,
                revision,
            });
            signal.notify_one();
        }
        for _ in 0..UPLOADS {
            let Ok(result) = self
                .receiver
                .as_mut()
                .unwrap()
                .get_mut()
                .unwrap()
                .try_recv()
            else {
                break;
            };
            if result.revision != self.stats.revision {
                self.stats.discarded_results += 1;
                continue;
            }
            let q = std::array::from_fn::<_, 3, _>(|a| result.key.0[a] - self.low.0[a]);
            assert!(q.iter().all(|v| (0..SIDE).contains(v)));
            let slot = (q[0] + q[1] * SIDE + q[2] * SIDE * SIDE) as usize;
            let base = (slot * WORDS) as u32;
            let words = result.brick.words();
            assert!(words.len() <= WORDS);
            queue.write_buffer(
                &self.words,
                u64::from(base) * 4,
                bytemuck::cast_slice(words),
            );
            queue.write_buffer(&self.directory, slot as u64 * 4, bytemuck::bytes_of(&base));
            self.stats.ready += 1;
            self.stats.uploaded_bytes += (words.len() * 4 + 4) as u64;
            self.stats.construction_ms += result.milliseconds;
        }
    }
}

impl Drop for Patch {
    fn drop(&mut self) {
        self.serial.fetch_add(1, Ordering::AcqRel);
        let (lock, signal) = &*self.pending;
        lock.lock().unwrap().stop = true;
        signal.notify_one();
        // Release a worker blocked by bounded result backpressure before join.
        self.receiver.take();
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}
