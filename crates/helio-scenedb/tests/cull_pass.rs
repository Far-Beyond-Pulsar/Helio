//! `CullPass` end-to-end proof (M3-b T5, design S4): build a real scene
//! through SceneDB's public write/harvest API (mirrors `seam_smoke.rs`'s
//! `test_context`/boundary-driving pattern and `gpu_harvest.rs`'s harvest
//! pattern), upload the harvested tokens via `ViewTokenBuffers` (M3-b T1),
//! dispatch `CullPass`, and read back the command buffer + visible ids +
//! telemetry counters. Every expected value below is HAND-COMPUTED from the
//! fixture's own geometry, not re-derived by re-running the shader's logic
//! in Rust (that CPU-reference equality check is M3-b T6's job).
//!
//! ## The fixture (house law: self-verifying guards -- every category
//! non-empty before trusting the aggregate assertion)
//!
//! Camera at the world origin, looking down -Z, with an identity view
//! matrix (world space IS view space here) and a symmetric fovy=90/aspect=1
//! perspective projection (near=1, far=100 -- `fixture::view_proj_90deg`).
//! Three instances, one cell, fresh allocations (slot == row == index,
//! generation 1 for all three -- no stale/oob category exercised, asserted
//! zero below per the plan's own instruction: "stale/oob are Task 6's job").
//!
//! | row | translation      | mesh (local extents)     | category      |
//! |-----|-------------------|---------------------------|---------------|
//! | 0   | (0, 0, -10)       | mesh0, (0.5,0.5,0.5)      | VISIBLE       |
//! | 1   | (100, 0, -10)     | mesh0, (0.5,0.5,0.5)      | FRUSTUM-CULLED|
//! | 2   | (0, 0, -0.5)      | mesh1, (0.5,0.5,2.0)      | NEAR-CLIP     |
//!
//! Row 0's box (world z in [-10.5,-9.5], all W = -z_view in [9.5,10.5] > 0,
//! well inside the fovy=90 side planes at that depth) is unambiguously
//! visible with no near-clip. Row 1 is the identical box translated far off
//! to the side (x=100 at z=-10; the fovy=90 side plane only admits |x| <=
//! 10 at that depth) -- entirely outside the right frustum plane, cannot
//! trigger near-clip (same z range as row 0). Row 2's box is deliberately
//! LARGE in z (extents.z=2.0) and centered just in front of the camera
//! (translation z=-0.5), so its far corners reach world z=+1.5 -- BEHIND
//! the camera eye (z_view >= 0), giving W = -z_view <= 0 on those corners,
//! which per spec S12 bypasses culling entirely (marked visible
//! unconditionally, near-clip flag set) rather than being frustum-tested.

use helio_scenedb::cull::{CullOutputBuffers, CullPass, CullUniforms, NEAR_CLIP_FLAG};
use helio_scenedb::SceneDbBinding;
use pulsar_scenedb::gpu::{
    CellSlot, ClusterBuffer, EngineGpuContext, FrameDriver, HarvestPipeline, HarvestStaging,
    MaterialRegistry, MeshClass, MeshMetadata, MeshRegistry, MeshletBuffer, RegionClassConfig,
    SceneGpuConfig, SceneGpuStore, View, ViewTokenBuffers,
};
use pulsar_scenedb::{Aabb, InstanceInfo, Scratchpad, SpatialCell};
use std::collections::HashMap;
use std::sync::Arc;

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
        label: Some("helio-scenedb-cull-pass-test"),
        // M3-b T4's whole point: this pass fits under the WebGPU-portable
        // default limits (group(0) 5 + group(2) 3 == 8, wgsl.rs's CULL_WGSL
        // doc has the arithmetic) — no `adapter.limits()` workaround needed.
        required_limits: wgpu::Limits::default(),
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

fn scene_cfg() -> SceneGpuConfig {
    SceneGpuConfig {
        classes: vec![RegionClassConfig { capacity: 64, max_resident_cells: 4 }],
        tombstone_headroom: 8,
        max_cells_metadata: 16,
    }
}

/// Column-major-flat `mat4x4<f32>` for a pure translation (no rotation, no
/// scale) — translation-only matrices are transpose-invariant in this flat
/// 16-float layout (the same 12/13/14 positions hold tx/ty/tz whether the
/// data is read row-major or column-major), which is WHY this fixture
/// deliberately avoids rotation entirely: it sidesteps the row/col-major
/// authoring-convention question raised by `page.rs`'s "row-major mat4"
/// comment (never resolved by any prior task — no shader before this one
/// ever multiplied a vector through this buffer) without depending on an
/// answer this task does not need to give. `CULL_WGSL`'s `abs_mat3`/S11
/// world-AABB math is still exercised generally (it operates on the full
/// upper-left 3x3, not specialized to this fixture's diagonal-identity
/// case) — only the FIXTURE's own hand-computed expectations rely on this
/// translation-only simplification.
fn translation(t: [f32; 3]) -> [f32; 16] {
    #[rustfmt::skip]
    let m = [
        1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        t[0], t[1], t[2], 1.0,
    ];
    m
}

/// Symmetric perspective projection, fovy=90deg (tan(fovy/2)==1), aspect=1,
/// so fovx==90deg too (tan(fovx/2) == aspect*tan(fovy/2) == 1) — the
/// side/top/bottom frustum planes below are this exact projection's planes,
/// independently re-derived (not read back off this matrix), matching
/// `plane_test`'s `n.p+d>=0` convention. `w = -z_view` for ANY such
/// right-handed perspective matrix with the standard `row3 = [0,0,-1,0]`
/// (verified algebraically in the M3-b T5 report) — the near/far values
/// only affect `z_clip`, never `w`, so S12's W<=0 test is insensitive to
/// the exact near/far choice here.
fn view_proj_90deg() -> [f32; 16] {
    let near = 1.0_f32;
    let far = 100.0_f32;
    let a = far / (near - far); // -100/99
    let b = (near * far) / (near - far); // -100/99 (near*far == far here since near==1)
    #[rustfmt::skip]
    let m = [
        1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, a, -1.0,
        0.0, 0.0, b, 0.0,
    ];
    m
}

/// The 6 inward-normal (`n.p+d>=0` == inside) world-space planes of
/// `view_proj_90deg`'s frustum, independently derived from the fovy=90/
/// aspect=1/near=1/far=100 geometry (NOT extracted from the matrix) —
/// `x<=-z` / `x>=z` / `y<=-z` / `y>=z` (both at 45 degree half-angle, since
/// tan==1) plus `z<=-near` / `z>=-far`.
fn frustum_planes_90deg() -> [[f32; 4]; 6] {
    [
        [1.0, 0.0, -1.0, 0.0],  // left:   x - z >= 0
        [-1.0, 0.0, -1.0, 0.0], // right: -x - z >= 0
        [0.0, 1.0, -1.0, 0.0],  // bottom: y - z >= 0
        [0.0, -1.0, -1.0, 0.0], // top:   -y - z >= 0
        [0.0, 0.0, -1.0, -1.0], // near:  -z - 1 >= 0  (z <= -1)
        [0.0, 0.0, 1.0, 100.0], // far:    z + 100 >= 0 (z >= -100)
    ]
}

fn mesh(center: [f32; 3], extents: [f32; 3], index_count: u32, index_offset: u32, base_vertex: i32) -> MeshMetadata {
    MeshMetadata {
        vertex_offset: 0,
        index_offset,
        index_count,
        base_vertex,
        material_index: 0,
        lod_count: 1, // C5 XOR rule: traditional mesh, no cluster table
        lod_distances: [0.0; 4],
        local_aabb_center: center,
        cluster_table_offset: 0,
        local_aabb_extents: extents,
        meshlet_count: 0,
    }
}

#[test]
fn cull_pass_produces_exact_hand_computed_visible_set() {
    let ctx = test_context();
    let device: &wgpu::Device = ctx.device().as_ref();

    // --- Build the scene: one cell, 3 fresh instances (rows 0/1/2, slot ==
    // row, generation 1 for all — no stale/oob category in this fixture,
    // Task 6's job). ---
    let mut store = SceneGpuStore::new(&ctx, scene_cfg());
    let mut cell = SpatialCell::with_transform(64).unwrap();
    // Harvest (broad-phase) AABBs: generous unit boxes around each
    // translation so ONE big harvest query AABB catches all three — the
    // cull SHADER computes its own precise world AABB from mesh_meta +
    // transform independently (S11), this is only the CPU-side coarse
    // query bound.
    let h0 = cell.alloc(Aabb { min: [-1.0, -1.0, -11.0], max: [1.0, 1.0, -9.0] }).unwrap();
    let h1 = cell.alloc(Aabb { min: [99.0, -1.0, -11.0], max: [101.0, 1.0, -9.0] }).unwrap();
    let h2 = cell.alloc(Aabb { min: [-1.0, -1.0, -1.5], max: [1.0, 1.0, 0.5] }).unwrap();

    let id = store.register_cell(cell.storage(), 0).unwrap();
    let base = store.row_region_base(id);
    assert_eq!(base, 0, "single cell, class-0 region base 0 — rows below are global rows verbatim");

    let mut frames = FrameDriver::new();
    let sim = frames.begin();
    assert!(store.write_transform(id, cell.storage_mut(), h0, &translation([0.0, 0.0, -10.0]), &sim));
    assert!(store.write_transform(id, cell.storage_mut(), h1, &translation([100.0, 0.0, -10.0]), &sim));
    assert!(store.write_transform(id, cell.storage_mut(), h2, &translation([0.0, 0.0, -0.5]), &sim));
    assert!(store.write_instance_info(id, cell.storage_mut(), h0, InstanceInfo { mesh_index: 0, flags: 0 }, &sim));
    assert!(store.write_instance_info(id, cell.storage_mut(), h1, InstanceInfo { mesh_index: 0, flags: 0 }, &sim));
    assert!(store.write_instance_info(id, cell.storage_mut(), h2, InstanceInfo { mesh_index: 1, flags: 0 }, &sim));

    // --- Harvest: one big AABB view over the whole cell (S3.1's expected-
    // generation column comes along for free — HarvestPipeline::harvest_cell
    // always emits it positionally aligned with the token). ---
    let harvest_phase = sim.end().end();
    let pipeline = HarvestPipeline::new();
    let mut pad = Scratchpad::new();
    let mut staging = HarvestStaging::new();
    let view = View::Aabb(Aabb { min: [-200.0, -200.0, -200.0], max: [200.0, 200.0, 200.0] });
    let n = pipeline.harvest_cell(&cell, base, MeshClass::Traditional, &view, &mut pad, &mut staging, &harvest_phase);
    assert_eq!(n, 3, "sanity: the broad harvest view catches all three rows");
    assert_eq!(staging.traditional.len(), 3);

    // --- Boundary: retire/compact/sync pushes the transform/instance_info
    // writes above onto the GPU buffers the cull shader will read. ---
    let boundary = harvest_phase.end();
    {
        let mut slots = [CellSlot { id, cell: cell.storage_mut() }];
        boundary.run(&mut store, &mut slots);
    }

    // --- Per-view token upload (M3-b T1). ---
    let mut view_tokens = ViewTokenBuffers::new(&ctx, "cull-pass-test-view", 0);
    view_tokens.upload(&ctx, &staging, MeshClass::Traditional);
    assert_eq!(view_tokens.count(), 3);

    // --- Asset-side stores SceneDbBinding also binds (mesh registry carries
    // the two real meshes this fixture's hand-computed expectations use;
    // cluster/meshlet/material registries are unused by cull, empty is
    // fine — mirrors seam_smoke.rs). ---
    let mut meshes = MeshRegistry::new(&ctx, 4);
    let mesh0 = meshes.register(ctx.queue(), mesh([0.0, 0.0, 0.0], [0.5, 0.5, 0.5], 6, 0, 0)).unwrap();
    let mesh1 = meshes.register(ctx.queue(), mesh([0.0, 0.0, 0.0], [0.5, 0.5, 2.0], 12, 6, 4)).unwrap();
    assert_eq!((mesh0, mesh1), (0, 1), "sanity: mesh_index values the fixture's InstanceInfo rows used above");
    let clusters = ClusterBuffer::new(&ctx, 1);
    let meshlets = MeshletBuffer::new(&ctx, 1);
    let materials = MaterialRegistry::new(&ctx, 1);

    let scene_binding = SceneDbBinding::new(device, &store, &meshes, &clusters, &meshlets, &materials);

    // --- The cull pass itself. ---
    let cull_pass = CullPass::new(device, &scene_binding.cull_layout);
    let output = CullOutputBuffers::new(device, "cull-pass-test-output", 8);
    output.clear_counters(ctx.queue());
    let output_bind_group = cull_pass.build_output_bind_group(device, &view_tokens, &output);

    let uniforms = CullUniforms {
        view_proj: view_proj_90deg(),
        planes: frustum_planes_90deg(),
        count: view_tokens.count(),
        mesh_count: meshes.len(),
        capacity: output.capacity(),
        reserved: 0,
    };
    cull_pass.write_uniforms(ctx.queue(), &uniforms);

    let mut encoder = device.create_command_encoder(&Default::default());
    cull_pass.record(&mut encoder, &scene_binding.cull_bind_group, &output_bind_group, view_tokens.count());
    ctx.queue().submit([encoder.finish()]);

    // --- Readback: 16-byte counters header + up to `capacity` 32-byte
    // records. ---
    let out_bytes = readback(&ctx, output.buffer(), output.byte_size());
    let u32_at = |off: usize| u32::from_le_bytes(out_bytes[off..off + 4].try_into().unwrap());
    let i32_at = |off: usize| i32::from_le_bytes(out_bytes[off..off + 4].try_into().unwrap());

    let visible_count = u32_at(0);
    let stale_drops = u32_at(4);
    let oob_drops = u32_at(8);
    let frustum_drops = u32_at(12);

    // --- Self-verifying fixture guards (house law): every category the
    // fixture is DESIGNED to exercise must be non-empty before the main
    // assertion is trusted, and every category it deliberately does NOT
    // exercise (stale/oob — Task 6's job) must read exactly zero. ---
    assert_eq!(stale_drops, 0, "fixture has no stale-generation row (Task 6's job)");
    assert_eq!(oob_drops, 0, "fixture has no out-of-range mesh_index row (Task 6's job)");
    assert_eq!(frustum_drops, 1, "guard: exactly one frustum-culled row (row 1) — must be non-zero");
    assert_eq!(visible_count, 2, "guard: exactly two visible rows (row 0 + row 2, the near-clip bypass) — must be non-zero");

    // --- Parse the `visible_count` (== 2, within `capacity` == 8, so both
    // slots were written) records, keyed by `row` — command-slot ORDER
    // between concurrently-executing GPU threads is not guaranteed, so this
    // assertion is written to be robust to either row landing in slot 0 or
    // slot 1, while still being fully hand-computed per row. ---
    let mut by_row: HashMap<u32, (u32, u32, u32, i32, u32, u32)> = HashMap::new(); // row -> (index_count, instance_count, first_index, base_vertex, first_instance, flags)
    for slot in 0..visible_count.min(output.capacity()) {
        let base_off = CullOutputBuffers::HEADER_BYTES as usize + slot as usize * CullOutputBuffers::RECORD_BYTES as usize;
        let index_count = u32_at(base_off);
        let instance_count = u32_at(base_off + 4);
        let first_index = u32_at(base_off + 8);
        let base_vertex = i32_at(base_off + 12);
        let first_instance = u32_at(base_off + 16);
        let row = u32_at(base_off + 20);
        let flags = u32_at(base_off + 24);
        assert_eq!(first_instance, slot, "S14.1: first_instance == command slot (C5)");
        assert_eq!(instance_count, 1, "S14.1: instance_count always 1 — no instance merging");
        by_row.insert(row, (index_count, instance_count, first_index, base_vertex, first_instance, flags));
    }
    assert_eq!(by_row.len(), 2, "two distinct rows, no duplicate/lost slot");

    // Row 0 (VISIBLE, mesh0, no near-clip): hand-computed from mesh0's
    // registered fields (index_count=6, index_offset=0, base_vertex=0).
    let row0 = by_row.get(&0).expect("row 0 (VISIBLE) must have a command");
    assert_eq!(row0.0, 6, "row 0 index_count == mesh0.index_count");
    assert_eq!(row0.2, 0, "row 0 first_index == mesh0.index_offset");
    assert_eq!(row0.3, 0, "row 0 base_vertex == mesh0.base_vertex");
    assert_eq!(row0.5 & NEAR_CLIP_FLAG, 0, "row 0 is NOT near-clip — flag bit 0 clear");

    // Row 2 (NEAR-CLIP bypass, mesh1): hand-computed from mesh1's registered
    // fields (index_count=12, index_offset=6, base_vertex=4).
    let row2 = by_row.get(&2).expect("row 2 (NEAR-CLIP) must have a command");
    assert_eq!(row2.0, 12, "row 2 index_count == mesh1.index_count");
    assert_eq!(row2.2, 6, "row 2 first_index == mesh1.index_offset");
    assert_eq!(row2.3, 4, "row 2 base_vertex == mesh1.base_vertex");
    assert_eq!(row2.5 & NEAR_CLIP_FLAG, NEAR_CLIP_FLAG, "row 2 near-clip flag bit 0 MUST be set — its far corners are behind the camera (W<=0)");

    // Row 1 (FRUSTUM-CULLED) must have no command at all.
    assert!(!by_row.contains_key(&1), "row 1 was frustum-culled — no command, no visible id");
}
