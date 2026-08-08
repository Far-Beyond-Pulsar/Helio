//! SceneDB world-mirror frame-boundary integration (Helio #210).
//!
//! This module owns the [`SceneGpuMirrorSubsystem`] — a
//! [`pulsar_scenedb::Subsystem`] that flushes the world-mirror store at the
//! retire/compact boundary each frame — and the [`Scene`]'s
//! [`flush_scenedb_boundary`] method that drives the full phase machine
//! ([`pulsar_scenedb::gpu::FrameDriver`] → simulate → harvest → boundary
//! → compact → sync).
//!
//! ## Two-store design
//!
//! The subsystem's store (`Arc<SceneGpuStore>`) is the *world-mirror* store
//! shared with the [`GpuMirrorHandle`] attached to [`Scene::world`]. Flushing
//! it is a `&self` operation. The *phase* store (`SceneGpuStore`, owned by
//! value) drives the genuine `&mut` witness chain with empty cells, producing
//! the required [`RetiredPhase`] for the subsystem hook. See the foundation
//! brief for the full constraint analysis.

use std::sync::Arc;

use pulsar_scenedb::gpu::{RetiredPhase, SceneGpuStore};
use pulsar_scenedb::Subsystem;

use crate::scene::Scene;

/// Flushes the world-mirror GPU store at the retire/compact boundary.
///
/// Runs inside [`pulsar_scenedb::SubsystemRegistry::boundary`], which is
/// called with the genuine [`RetiredPhase`] produced by driving the
/// frame-boundary chain against the [`Scene`]'s by-value [`SceneGpuStore`].
pub(in crate::scene) struct SceneGpuMirrorSubsystem {
    store: Arc<SceneGpuStore>,
    queue: Arc<wgpu::Queue>,
}

impl SceneGpuMirrorSubsystem {
    pub(in crate::scene) fn new(store: Arc<SceneGpuStore>, queue: Arc<wgpu::Queue>) -> Self {
        Self { store, queue }
    }
}

impl Subsystem for SceneGpuMirrorSubsystem {
    fn name(&self) -> &'static str {
        "scene_gpu_mirror"
    }

    fn boundary(&mut self, _phase: &RetiredPhase) {
        let _stats = self.store.flush_gpu_mirror(&self.queue);
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}

impl Scene {
    /// Drive the SceneDB frame-boundary phase machine for this frame.
    ///
    /// Called from [`Scene::flush`] before the GPU scene submit. Runs the
    /// full phase chain against the by-value [`phase_store`] (which produces
    /// the genuine [`RetiredPhase`] for the subsystem hook), dispatches
    /// simulate/harvest/boundary to every registered subsystem, then
    /// completes the compact→sync bookkeeping.
    pub(in crate::scene) fn flush_scenedb_boundary(&mut self) {
        let sim_a = self.scenedb_driver.begin();
        self.scenedb_subsystems.simulate_a(&mut self.world, &sim_a);

        let sim_b = sim_a.end();
        self.scenedb_subsystems.simulate_b(&mut self.world, &sim_b);

        let harvest = sim_b.end();
        self.scenedb_subsystems.harvest(&self.mirror_store, &harvest);

        let boundary = harvest.end();
        let (retired, _drained) = boundary.retire(&mut self.phase_store, &mut []);
        self.scenedb_subsystems.boundary(&retired);

        let _sync = retired.compact(&mut self.phase_store, &mut []).sync(&mut self.phase_store, &mut []);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pulsar_scenedb::gpu::{
        EngineGpuContext, GpuMirrorHandle, RegionClassConfig, SceneGpuConfig, SimulateA,
    };
    use pulsar_scenedb::{Entity, GpuColumnSet, SubsystemRegistry, World};
    use pulsar_scenedb_derive::SceneStore;

    /// Smoke component for testing the mirror round-trip through the frame chain.
    #[derive(SceneStore, Clone, Copy)]
    struct FrameSmokeComponent {
        #[gpu]
        value: u32,
    }

    /// Insert a known value into a pre-spawned entity during simulate_a.
    struct Writer {
        entity: Entity,
        value: u32,
    }

    impl Subsystem for Writer {
        fn name(&self) -> &'static str {
            "writer"
        }

        fn simulate_a(&mut self, world: &mut World, _witness: &SimulateA) {
            world.insert(self.entity, FrameSmokeComponent { value: self.value });
        }

        fn as_any(&self) -> &dyn std::any::Any {
            self
        }

        fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
            self
        }
    }

    fn test_context(label: &str) -> EngineGpuContext {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
            apply_limit_buckets: false,
        }))
        .expect("no adapter — GPU tests need a local GPU");
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some(label),
            ..Default::default()
        }))
        .expect("device");
        EngineGpuContext::new(Arc::new(device), Arc::new(queue))
    }

    fn readback_u32(ctx: &EngineGpuContext, buf: &wgpu::Buffer, row: u64) -> u32 {
        let staging = ctx.device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback"),
            size: 4,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut enc = ctx.device().create_command_encoder(&Default::default());
        enc.copy_buffer_to_buffer(buf, row * 4, &staging, 0, 4);
        ctx.queue().submit([enc.finish()]);
        let slice = staging.slice(..);
        slice.map_async(wgpu::MapMode::Read, |r| r.expect("map"));
        ctx.device()
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("poll");
        let data = slice.get_mapped_range().expect("mapped range").to_vec();
        staging.unmap();
        u32::from_ne_bytes(data.try_into().unwrap())
    }

    #[test]
    fn world_mirror_frame_chain_round_trip() {
        let ctx = test_context("scenedb-foundation-chain-test");

        let mut store = SceneGpuStore::new(
            &ctx,
            SceneGpuConfig {
                classes: vec![RegionClassConfig {
                    capacity: 64,
                    max_resident_cells: 1,
                }],
                tombstone_headroom: SceneGpuConfig::default_headroom(),
                max_cells_metadata: 16,
            },
        );
        FrameSmokeComponent::register_gpu_columns_growable(&mut store, 8, ctx.device());
        let store = Arc::new(store);

        let mut world = World::new();
        world.attach_gpu_mirror(GpuMirrorHandle::new(Arc::clone(&store), Arc::clone(ctx.queue())));

        let entity = world.spawn();
        let row = entity.index() as u64;

        let value_field_id = FrameSmokeComponent::gpu_columns()
            .iter()
            .find(|c| c.buffer_name == "value")
            .unwrap()
            .field_token
            .id();

        let mut registry = SubsystemRegistry::new();
        registry.register(Writer { entity, value: 1234 });
        registry.register(SceneGpuMirrorSubsystem::new(Arc::clone(&store), Arc::clone(ctx.queue())));

        let mut driver = pulsar_scenedb::gpu::FrameDriver::new();
        let mut phase_store = SceneGpuStore::new(
            &ctx,
            SceneGpuConfig {
                classes: vec![],
                tombstone_headroom: 0,
                max_cells_metadata: 0,
            },
        );

        let sim_a = driver.begin();
        registry.simulate_a(&mut world, &sim_a);
        let sim_b = sim_a.end();
        registry.simulate_b(&mut world, &sim_b);
        let harvest = sim_b.end();
        registry.harvest(&store, &harvest);
        let boundary = harvest.end();
        let (retired, _drained) = boundary.retire(&mut phase_store, &mut []);
        registry.boundary(&retired);
        let _stats = retired.compact(&mut phase_store, &mut []).sync(&mut phase_store, &mut []);

        let mut got = 0u32;
        store.with_dirty_tracked_buffer_for_id(value_field_id, &mut |buf| {
            got = readback_u32(&ctx, buf, row);
        });
        assert_eq!(
            got, 1234,
            "value written at simulate must be flushed to the GPU mirror by the boundary"
        );
    }
}
