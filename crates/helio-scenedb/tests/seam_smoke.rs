//! The seam proof (M3-a T9, design Rev 2 S5): build a real `SceneGpuStore` +
//! one cell through SceneDB's public write/boundary API (mirrors
//! `pulsar_scenedb/tests/gpu_store.rs`'s `test_context`/boundary-driving
//! pattern), construct a `SceneDbBinding` over it, and run a small
//! Helio-owned compute shader -- built from `SCENE_BINDINGS_WGSL` itself,
//! the ACTUAL seam WGSL, not a stand-in -- that copies `instances[0]` and
//! `instance_info[0]` into Helio-owned output buffers. Readback must equal
//! the values written on the CPU side, byte-exact.
//!
//! ## This test IS the wgpu-major version lock
//!
//! `helio-scenedb`'s `Cargo.toml` declares `wgpu = { workspace = true }`
//! (Helio's crates.io wgpu 30) alongside `pulsar_scenedb`'s own
//! `gpu`-feature `wgpu = "30"` dep -- two independent `Cargo.lock` entries
//! that happen to share a major version. Nothing at dependency-resolution
//! time enforces that they stay in lockstep. What enforces it is this file:
//! [`SceneDbBinding::new`] takes a `&pulsar_scenedb::gpu::SceneGpuStore`
//! whose buffer accessors (`transform_buffer()`, etc.) return
//! `&wgpu::Buffer` values built from `pulsar_scenedb`'s wgpu-30 `Device`,
//! and this test feeds those straight into Helio-namespaced
//! `wgpu::BindGroupLayout`/`wgpu::BindGroup`/`wgpu::ComputePipeline`
//! construction. If the two `wgpu` deps ever diverged to different majors,
//! `wgpu::Buffer`/`wgpu::Device`/`wgpu::Queue`/etc. would become DIFFERENT,
//! non-unifiable Rust types across the crate boundary, and this file would
//! fail to COMPILE (type mismatch at every `store.*_buffer()` call site and
//! every `as_entire_binding()`/bind-group construction below) -- long before
//! any runtime assertion could catch it. There is no separate runtime
//! version-lock check anywhere in this crate; this smoke test compiling and
//! passing IS the lock.

use helio_scenedb::{wgsl::SCENE_BINDINGS_WGSL, SceneDbBinding};
use pulsar_scenedb::gpu::{
    CellSlot, ClusterBuffer, EngineGpuContext, FrameDriver, MeshRegistry, MeshletBuffer,
    RegionClassConfig, SceneGpuConfig, SceneGpuStore,
};
use pulsar_scenedb::{CellStorage, CellType, InstanceInfo, TypeToken};
use std::sync::Arc;

/// Mirrors `pulsar_scenedb/tests/gpu_store.rs::test_context` (same
/// upstream-wgpu-30 API forms: `InstanceDescriptor::new_without_display_
/// handle()`, `apply_limit_buckets`, `PollType::wait_indefinitely()`,
/// `get_mapped_range()` returning a `Result`) -- this crate cannot import
/// that test helper across a crate boundary, so it is reproduced here
/// rather than approximated.
///
/// One deliberate difference from the `pulsar_scenedb`-side helper: this
/// test requests `adapter.limits()` instead of `wgpu::Limits::default()`.
/// `SceneDbBinding`'s 8-entry bind group is visible to every shader stage
/// (`ShaderStages::all()`, `src/lib.rs`) and this test adds 2 more compute
/// storage buffers of its own (`out_transform`/`out_info`) — 10 storage
/// buffers bound into the COMPUTE stage in total. `Limits::default()` is
/// the conservative WebGPU-portable baseline
/// (`max_storage_buffers_per_shader_stage` = 8) meant for browser/downlevel
/// targets, not a real constraint of the native hardware this GPU suite
/// runs on; requesting the adapter's actual limits is the standard native-
/// testing idiom (mirrors what a real renderer would do) and is well within
/// any GPU this suite targets.
fn test_context() -> EngineGpuContext {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: None,
        force_fallback_adapter: false,
        apply_limit_buckets: false,
    }))
    .expect("no adapter — GPU tests need a local GPU");
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("helio-scenedb-seam-smoke"),
        required_limits: adapter.limits(),
        ..Default::default()
    }))
    .expect("device");
    EngineGpuContext::new(Arc::new(device), Arc::new(queue))
}

fn readback(ctx: &EngineGpuContext, buf: &wgpu::Buffer, bytes: u64) -> Vec<u8> {
    let staging = ctx.device().create_buffer(&wgpu::BufferDescriptor {
        label: Some("readback"),
        size: bytes,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut enc = ctx.device().create_command_encoder(&Default::default());
    enc.copy_buffer_to_buffer(buf, 0, &staging, 0, bytes);
    ctx.queue().submit([enc.finish()]);
    let slice = staging.slice(..);
    slice.map_async(wgpu::MapMode::Read, |r| r.expect("map"));
    ctx.device().poll(wgpu::PollType::wait_indefinitely()).expect("poll");
    let data = slice.get_mapped_range().expect("mapped range").to_vec();
    staging.unmap();
    data
}

fn mat(seed: f32) -> [f32; 16] {
    core::array::from_fn(|i| seed + i as f32)
}

/// A `CellType`-based cell carrying both the `[f32; 16]` transform column
/// and the `InstanceInfo` column, mirroring `gpu_store.rs::transform_info_
/// cell`.
fn transform_info_cell(capacity: u32) -> CellStorage {
    let ct = CellType::new("helio-seam-smoke")
        .with(TypeToken::of::<[f32; 16]>())
        .with(TypeToken::of::<InstanceInfo>())
        .build()
        .unwrap();
    CellStorage::from_cell_type(&ct, capacity).unwrap()
}

fn scene_cfg() -> SceneGpuConfig {
    SceneGpuConfig {
        classes: vec![RegionClassConfig { capacity: 64, max_resident_cells: 4 }],
        tombstone_headroom: 8,
        max_materials: 16,
        max_cells_metadata: 16,
    }
}

#[test]
fn seam_smoke_shader_reads_scenedb_buffers_byte_exact() {
    let ctx = test_context();
    let device: &wgpu::Device = ctx.device().as_ref();

    // --- Build a real SceneGpuStore + one cell, through SceneDB's public
    // write/boundary API (mirrors tests/gpu_store.rs's pattern exactly:
    // register_cell -> alloc -> write_transform/write_instance_info ->
    // drive the compile-time phase machine's Simulate->Harvest->Boundary
    // chain by value). ---
    let mut store = SceneGpuStore::new(&ctx, scene_cfg());
    let mut cell = transform_info_cell(64);
    let id = store.register_cell(&cell, 0).unwrap();
    let mut frames = FrameDriver::new();
    let sim = frames.begin();
    let h = cell.alloc().unwrap();
    let transform = mat(9.0);
    let info = InstanceInfo { mesh_index: 7, flags: 1 };
    assert!(store.write_transform(id, &mut cell, h, &transform, &sim));
    assert!(store.write_instance_info(id, &mut cell, h, info, &sim));
    {
        let mut slots = [CellSlot { id, cell: &mut cell }];
        // retire -> compact -> sync: the only boundary path available
        // outside pulsar_scenedb (retire_all/compact_all/sync_all are
        // pub(crate)).
        sim.end().end().end().run(&mut store, &mut slots);
    }
    let row = cell.row_of(h).unwrap() as usize;
    let base = store.row_region_base(id) as usize;
    assert_eq!(base + row, 0, "single cell, single alloc, class-0 region base 0 — row 0 for a clean index below");

    // --- The asset-side stores SceneDbBinding also binds. Empty registries
    // are fine here — the smoke test proves the wiring (byte-exact transfer
    // through the bind group), not asset content; those get their own byte-
    // exact coverage in pulsar_scenedb's own gpu_assets suite. ---
    let meshes = MeshRegistry::new(&ctx, 1);
    let clusters = ClusterBuffer::new(&ctx, 1);
    let meshlets = MeshletBuffer::new(&ctx, 1);

    // --- The seam itself. ---
    let binding = SceneDbBinding::new(device, &store, &meshes, &clusters, &meshlets);

    // --- A Helio-owned compute shader, built from the ACTUAL seam WGSL
    // (SCENE_BINDINGS_WGSL) plus a 4-line entry point copying instances[0]
    // and instance_info[0] into Helio-owned output storage buffers (group
    // 1, kept separate from SceneDB's read-only group 0). ---
    let shader_src = format!(
        "{SCENE_BINDINGS_WGSL}\n{}",
        r#"
@group(1) @binding(0) var<storage, read_write> out_transform: array<mat4x4<f32>>;
@group(1) @binding(1) var<storage, read_write> out_info: array<InstanceInfo>;

@compute @workgroup_size(1)
fn main() {
    out_transform[0] = instances[0].transform;
    out_info[0] = instance_info[0];
}
"#
    );
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("seam-smoke-shader"),
        source: wgpu::ShaderSource::Wgsl(shader_src.into()),
    });

    let out_transform_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("out-transform"),
        size: 64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let out_info_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("out-info"),
        size: 8,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let out_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("out-layout"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    });
    let out_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("out-bind-group"),
        layout: &out_layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: out_transform_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: out_info_buf.as_entire_binding() },
        ],
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("seam-smoke-pipeline-layout"),
        bind_group_layouts: &[Some(&binding.layout), Some(&out_layout)],
        immediate_size: 0,
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("seam-smoke-pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });

    let mut encoder = device.create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &binding.bind_group, &[]);
        pass.set_bind_group(1, &out_bind_group, &[]);
        pass.dispatch_workgroups(1, 1, 1);
    }
    ctx.queue().submit([encoder.finish()]);

    // --- Readback: Helio-owned staging buffers, mirroring gpu_store.rs's
    // `readback` helper. ---
    let transform_bytes = readback(&ctx, &out_transform_buf, 64);
    let info_bytes = readback(&ctx, &out_info_buf, 8);

    let got_transform: Vec<f32> = transform_bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    assert_eq!(
        got_transform,
        transform.to_vec(),
        "shader-read instances[0].transform == the CPU transform written via write_transform"
    );

    let got_mesh_index = u32::from_le_bytes(info_bytes[0..4].try_into().unwrap());
    let got_flags = u32::from_le_bytes(info_bytes[4..8].try_into().unwrap());
    assert_eq!(
        (got_mesh_index, got_flags),
        (info.mesh_index, info.flags),
        "shader-read instance_info[0] == the CPU InstanceInfo written via write_instance_info"
    );
}
