//! M3-b T6: Test 2 (stale-token drop, design §3.1/§20.2/C6) + the
//! mesh_index out-of-range twin (closing the coverage gap the M3-b T5
//! review found — T5's own suite never disables the bounds check and
//! notice, since it never injects an out-of-range `mesh_index`).
//!
//! Both tests build a real scene through SceneDB's public write/harvest
//! API (same headless pattern as `cull_pass.rs`/`gpu_harvest.rs`), upload
//! via `ViewTokenBuffers`, dispatch the real `CullPass`, and read back the
//! telemetry counters + command records — never re-deriving expectations
//! by re-running the shader's own logic (that is `cull_equality.rs`'s job).

#[path = "support/mod.rs"]
mod support;

use helio_scenedb::cull::{CullOutputBuffers, CullPass, CullUniforms};
use helio_scenedb::SceneDbBinding;
use pulsar_scenedb::gpu::{
    CellSlot, ClusterBuffer, FrameDriver, HarvestPipeline, HarvestStaging, MaterialRegistry,
    MeshClass, MeshRegistry, MeshletBuffer, SceneGpuStore, View, ViewTokenBuffers,
};
use pulsar_scenedb::{Aabb, Handle, InstanceInfo, Scratchpad, SpatialCell};
use std::collections::{HashMap, HashSet};

use support::{frustum_planes_90deg, mesh, readback, scene_cfg, test_context, translation, view_proj_90deg};

fn box_at(x: f32) -> Aabb {
    Aabb { min: [x - 0.5, -0.5, -10.5], max: [x + 0.5, 0.5, -9.5] }
}

/// Reads back `CullOutputBuffers`' 16-byte atomics header + up to
/// `capacity` 32-byte records, returning `(visible_count, stale_drops,
/// oob_drops, frustum_drops, by_row)` — `by_row` maps `row ->
/// (index_count, first_index, base_vertex, first_instance)`, the same
/// shape `cull_pass.rs` parses, factored out here since both tests below
/// need it.
fn read_cull_output(
    ctx: &pulsar_scenedb::gpu::EngineGpuContext,
    output: &CullOutputBuffers,
) -> (u32, u32, u32, u32, HashMap<u32, (u32, u32, i32, u32)>) {
    let out_bytes = readback(ctx, output.buffer(), output.byte_size());
    let u32_at = |off: usize| u32::from_le_bytes(out_bytes[off..off + 4].try_into().unwrap());
    let i32_at = |off: usize| i32::from_le_bytes(out_bytes[off..off + 4].try_into().unwrap());

    let visible_count = u32_at(0);
    let stale_drops = u32_at(4);
    let oob_drops = u32_at(8);
    let frustum_drops = u32_at(12);

    let mut by_row = HashMap::new();
    for slot in 0..visible_count.min(output.capacity()) {
        let base_off = CullOutputBuffers::HEADER_BYTES as usize + slot as usize * CullOutputBuffers::RECORD_BYTES as usize;
        let index_count = u32_at(base_off);
        let first_index = u32_at(base_off + 8);
        let base_vertex = i32_at(base_off + 12);
        let first_instance = u32_at(base_off + 16);
        let row = u32_at(base_off + 20);
        assert_eq!(first_instance, slot, "S14.1: first_instance == command slot (C5)");
        by_row.insert(row, (index_count, first_index, base_vertex, first_instance));
    }
    (visible_count, stale_drops, oob_drops, frustum_drops, by_row)
}

/// Test 2 (design §3.1/§20.2/C6): a harvested run's `expected_gens`
/// snapshot goes stale the moment its row's occupant is freed and a
/// boundary bumps that slot's live generation — WITHOUT re-harvesting, the
/// cull shader must drop exactly the stale rows, increment `stale_drops`
/// by exactly that count, and leave every OTHER row's command untouched.
///
/// ## Fixture shape (M3-α T8 house law: gen-diverse KEPT set + a
/// self-verifying non-uniformity guard, PLUS two dedicated victim rows)
///
/// Six live rows, one cell: a KEPT set of four (`h0`, `h2`, `h3`, `h1b`)
/// built via the T8 free+compact+realloc churn recipe (frees `h1`,
/// compacts — the then-last live row swaps into `h1`'s slot, permanently
/// breaking row/slot identity — then reallocs `h1b` at the new tail,
/// recycling the freed slot at generation 2) so the kept set genuinely
/// carries non-uniform generations (guarded below, non-vacuously); and two
/// VICTIM rows (`hv1`, `hv2`) allocated LAST, after the churn, so freeing
/// them later requires no compaction swap into the kept set's rows — a
/// self-verifying guard re-checks every kept handle's row is UNCHANGED
/// across the second boundary, so this test does not silently mis-assert
/// if that assumption ever stops holding.
#[test]
fn test2_stale_token_drop_leaves_only_the_injected_rows_dropped() {
    let ctx = test_context();
    let device: &wgpu::Device = ctx.device().as_ref();

    let mut store = SceneGpuStore::new(&ctx, scene_cfg());
    let mut cell = SpatialCell::with_transform(64).unwrap();

    // -- Kept-set churn (T8 recipe): h0, h1(freed), h2, h3 allocated, h1
    // freed + compacted, h1b reallocated at the new tail. --
    let h0 = cell.alloc(box_at(0.0)).unwrap();
    let h1 = cell.alloc(box_at(1.5)).unwrap();
    let h2 = cell.alloc(box_at(3.0)).unwrap();
    let h3 = cell.alloc(box_at(4.5)).unwrap();
    cell.free(h1);
    cell.compact();
    let h1b = cell.alloc(box_at(6.0)).unwrap();
    let kept: Vec<(Handle, f32)> = vec![(h0, 0.0), (h2, 3.0), (h3, 4.5), (h1b, 6.0)];

    // -- Victim rows, allocated LAST (tail — no swap needed when freed). --
    let hv1 = cell.alloc(box_at(7.5)).unwrap();
    let hv2 = cell.alloc(box_at(9.0)).unwrap();
    let victims = [hv1, hv2];
    let victims_with_x: Vec<(Handle, f32)> = vec![(hv1, 7.5), (hv2, 9.0)];

    // Self-verifying guard 1 (house law): the kept set must carry genuinely
    // non-uniform generations, or the stale-check below is not actually
    // exercising generation diversity.
    let distinct_kept_gens: HashSet<u32> = kept.iter().map(|(h, _)| h.generation()).collect();
    assert!(
        distinct_kept_gens.len() > 1,
        "guard: the kept set must carry genuinely different generations (T8 churn) — got {distinct_kept_gens:?}"
    );

    let id = store.register_cell(cell.storage(), 0).unwrap();
    let base = store.row_region_base(id);
    assert_eq!(base, 0, "single cell, class-0 region base 0");

    let mut frames = FrameDriver::new();
    let sim = frames.begin();
    for (h, x) in kept.iter().copied().chain(victims_with_x.iter().copied()) {
        assert!(store.write_transform(id, cell.storage_mut(), h, &translation([x, 0.0, -10.0]), &sim));
        assert!(store.write_instance_info(id, cell.storage_mut(), h, InstanceInfo { mesh_index: 0, flags: 0 }, &sim));
    }

    // -- Harvest all six live rows in one broad query. --
    let harvest_phase = sim.end().end();
    let pipeline = HarvestPipeline::new();
    let mut pad = Scratchpad::new();
    let mut staging = HarvestStaging::new();
    let view = View::Aabb(Aabb { min: [-1.0, -1.0, -11.0], max: [10.0, 1.0, -9.0] });
    let n = pipeline.harvest_cell(&cell, base, MeshClass::Traditional, &view, &mut pad, &mut staging, &harvest_phase);
    assert_eq!(n, 6, "sanity: all six live rows (4 kept + 2 victims) harvested");

    // Capture every kept handle's row BEFORE the second boundary, so the
    // guard after it can prove no collateral row-shuffle happened.
    let kept_rows_pre: HashMap<Handle, u32> = kept.iter().map(|&(h, _)| (h, cell.row_of(h).unwrap())).collect();

    // -- Boundary #1: push the initial (all-fresh) state to GPU — the
    // uploaded tokens/gens below snapshot exactly this state. --
    let boundary = harvest_phase.end();
    {
        let mut slots = [CellSlot { id, cell: cell.storage_mut() }];
        boundary.run(&mut store, &mut slots);
    }

    let mut view_tokens = ViewTokenBuffers::new(&ctx, "test2-stale-view", 0);
    view_tokens.upload(&ctx, &staging, MeshClass::Traditional);
    assert_eq!(view_tokens.count(), 6);

    let mut meshes = MeshRegistry::new(&ctx, 2);
    let mesh0 = meshes.register(ctx.queue(), mesh([0.0, 0.0, 0.0], [0.5, 0.5, 0.5], 6, 0, 0)).unwrap();
    assert_eq!(mesh0, 0);
    let clusters = ClusterBuffer::new(&ctx, 1);
    let meshlets = MeshletBuffer::new(&ctx, 1);
    let materials = MaterialRegistry::new(&ctx, 1);
    let scene_binding = SceneDbBinding::new(device, &store, &meshes, &clusters, &meshlets, &materials);
    let cull_pass = CullPass::new(device, &scene_binding.cull_layout);
    let output = CullOutputBuffers::new(device, "test2-stale-output", 8);

    let uniforms = CullUniforms {
        view_proj: view_proj_90deg(),
        planes: frustum_planes_90deg(),
        count: view_tokens.count(),
        mesh_count: meshes.len(),
        capacity: output.capacity(),
        reserved: 0,
    };
    cull_pass.write_uniforms(ctx.queue(), &uniforms);
    let output_bind_group = cull_pass.build_output_bind_group(device, &view_tokens, &output);

    // -- Dispatch #1 (pre-free baseline): every row is fresh, nothing
    // stale yet — proves the fixture is genuinely all-visible BEFORE
    // staleness is injected (a non-vacuity guard on the fixture itself). --
    output.clear_counters(ctx.queue());
    {
        let mut encoder = device.create_command_encoder(&Default::default());
        cull_pass.record(&mut encoder, &scene_binding.cull_bind_group, &output_bind_group, view_tokens.count());
        ctx.queue().submit([encoder.finish()]);
    }
    let (visible0, stale0, oob0, frustum0, by_row0) = read_cull_output(&ctx, &output);
    assert_eq!(stale0, 0, "baseline: nothing stale yet");
    assert_eq!(oob0, 0, "baseline: no OOB row in this fixture");
    assert_eq!(frustum0, 0, "baseline: every row is comfortably inside the frustum");
    assert_eq!(visible0, 6, "baseline: all six rows visible before staleness is injected");
    assert_eq!(by_row0.len(), 6);

    // -- Make hv1 + hv2 stale: free_deferred -> force_complete -> a SECOND
    // boundary (no re-harvest, no re-upload — the SAME view_tokens buffer
    // from before is reused for dispatch #2 below). --
    let sim2 = frames.begin();
    let serial1 = store.tracker().next_serial();
    assert!(store.free_deferred(id, cell.storage_mut(), hv1, serial1, &sim2));
    let serial2 = store.tracker().next_serial();
    assert!(store.free_deferred(id, cell.storage_mut(), hv2, serial2, &sim2));
    store.tracker().force_complete(serial1);
    store.tracker().force_complete(serial2);
    let harvest2 = sim2.end().end();
    let boundary2 = harvest2.end();
    {
        let mut slots = [CellSlot { id, cell: cell.storage_mut() }];
        boundary2.run(&mut store, &mut slots);
    }

    // Self-verifying guard 2 (house law): the kept set's rows must be
    // UNCHANGED by the second boundary (victims were tail rows, so no
    // compaction swap should have touched anything else) — if this ever
    // stops holding, the "every OTHER row still emits normally" assertion
    // below would be checking the wrong physical rows, so fail loudly here
    // instead.
    for (h, x) in &kept {
        let post_row = cell.row_of(*h).expect("kept handle must still be live");
        assert_eq!(
            kept_rows_pre[h], post_row,
            "guard: kept handle at x={x} must keep its pre-free row (victims were tail rows — no swap expected)"
        );
    }
    let expected_visible_rows: HashSet<u32> = kept.iter().map(|(h, _)| cell.row_of(*h).unwrap()).collect();
    assert_eq!(expected_visible_rows.len(), 4);

    // -- Dispatch #2: SAME view_tokens (uploaded before the free) against
    // the NOW-BUMPED live generations for hv1/hv2's slots. --
    output.clear_counters(ctx.queue());
    {
        let mut encoder = device.create_command_encoder(&Default::default());
        cull_pass.record(&mut encoder, &scene_binding.cull_bind_group, &output_bind_group, view_tokens.count());
        ctx.queue().submit([encoder.finish()]);
    }
    let (visible1, stale1, oob1, frustum1, by_row1) = read_cull_output(&ctx, &output);

    // THE assertions (design §3.1/Test 2): exactly the injected count
    // dropped as stale, nothing else touched.
    assert_eq!(stale1, 2, "stale_drops must equal exactly the 2 injected stale rows");
    assert_eq!(oob1, 0, "no OOB row in this fixture");
    assert_eq!(frustum1, 0, "no row moved out of the frustum — only staleness changed");
    // Non-vacuity guard: a shader that dropped EVERYTHING would also
    // "pass" a naive assert_eq!(visible1, 0) style check — assert the
    // KEPT rows are still exactly, individually, present instead.
    assert_eq!(visible1, 4, "exactly the 4 kept (non-stale) rows must still emit a command");
    let by_row1_keys: HashSet<u32> = by_row1.keys().copied().collect();
    assert_eq!(
        by_row1_keys, expected_visible_rows,
        "the visible set after staleness injection must equal EXACTLY the kept (non-stale) rows"
    );
    for (row, (index_count, first_index, base_vertex, _first_instance)) in &by_row1 {
        assert_eq!(*index_count, 6, "row {row}: index_count == mesh0.index_count");
        assert_eq!(*first_index, 0, "row {row}: first_index == mesh0.index_offset");
        assert_eq!(*base_vertex, 0, "row {row}: base_vertex == mesh0.base_vertex");
    }
    // The stale rows' old physical positions must have no command at all.
    let stale_rows_now: HashSet<u32> = victims.iter().filter_map(|h| cell.row_of(*h)).collect();
    // (victims are dead — `row_of` returns None for both; this is asserted
    // for documentation, not as a load-bearing check.)
    assert!(stale_rows_now.is_empty(), "victims are dead — row_of returns None for both");
}

/// The mesh_index out-of-range twin (M3-b T5 review coverage gap): a row
/// whose `InstanceInfo.mesh_index` is written, via the NORMAL
/// `write_instance_info` path, as `>= meshes.len()` (the mesh-table
/// bounds-check ceiling `CullUniforms::mesh_count` carries) must be dropped
/// with `oob_drops` incremented, and must not disturb a sibling VALID row
/// (the non-vacuity guard). Mutation-kill evidence (WGSL bounds check
/// temporarily disabled, this test observed to fail, then reverted) is
/// recorded in the M3-b T6 report, not in this file.
#[test]
fn mesh_index_out_of_range_is_dropped_with_oob_telemetry() {
    let ctx = test_context();
    let device: &wgpu::Device = ctx.device().as_ref();

    let mut store = SceneGpuStore::new(&ctx, scene_cfg());
    let mut cell = SpatialCell::with_transform(64).unwrap();

    let h_ok = cell.alloc(box_at(0.0)).unwrap();
    let h_oob = cell.alloc(box_at(2.0)).unwrap();

    let id = store.register_cell(cell.storage(), 0).unwrap();
    let base = store.row_region_base(id);

    let mut frames = FrameDriver::new();
    let sim = frames.begin();
    assert!(store.write_transform(id, cell.storage_mut(), h_ok, &translation([0.0, 0.0, -10.0]), &sim));
    assert!(store.write_transform(id, cell.storage_mut(), h_oob, &translation([2.0, 0.0, -10.0]), &sim));
    assert!(store.write_instance_info(id, cell.storage_mut(), h_ok, InstanceInfo { mesh_index: 0, flags: 0 }, &sim));
    // Written via the NORMAL write_instance_info path (per plan: "travels
    // the real pipeline") — mesh_index == meshes.len() exactly (the
    // boundary the `>=` bounds check must catch).
    assert!(store.write_instance_info(id, cell.storage_mut(), h_oob, InstanceInfo { mesh_index: 1, flags: 0 }, &sim));

    let harvest_phase = sim.end().end();
    let pipeline = HarvestPipeline::new();
    let mut pad = Scratchpad::new();
    let mut staging = HarvestStaging::new();
    let view = View::Aabb(Aabb { min: [-1.0, -1.0, -11.0], max: [3.0, 1.0, -9.0] });
    let n = pipeline.harvest_cell(&cell, base, MeshClass::Traditional, &view, &mut pad, &mut staging, &harvest_phase);
    assert_eq!(n, 2, "sanity: both rows harvested");

    let boundary = harvest_phase.end();
    {
        let mut slots = [CellSlot { id, cell: cell.storage_mut() }];
        boundary.run(&mut store, &mut slots);
    }

    let mut view_tokens = ViewTokenBuffers::new(&ctx, "oob-twin-view", 0);
    view_tokens.upload(&ctx, &staging, MeshClass::Traditional);
    assert_eq!(view_tokens.count(), 2);

    // Exactly ONE mesh registered (index 0) — h_oob's mesh_index (1) is
    // exactly `meshes.len()`, the `>=` boundary the bounds check must trip.
    let mut meshes = MeshRegistry::new(&ctx, 2);
    let mesh0 = meshes.register(ctx.queue(), mesh([0.0, 0.0, 0.0], [0.5, 0.5, 0.5], 6, 0, 0)).unwrap();
    assert_eq!(mesh0, 0);
    assert_eq!(meshes.len(), 1, "sanity: h_oob's mesh_index (1) is out of range against this table");
    let clusters = ClusterBuffer::new(&ctx, 1);
    let meshlets = MeshletBuffer::new(&ctx, 1);
    let materials = MaterialRegistry::new(&ctx, 1);
    let scene_binding = SceneDbBinding::new(device, &store, &meshes, &clusters, &meshlets, &materials);
    let cull_pass = CullPass::new(device, &scene_binding.cull_layout);
    let output = CullOutputBuffers::new(device, "oob-twin-output", 8);
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

    let (visible, stale, oob, frustum, by_row) = read_cull_output(&ctx, &output);

    assert_eq!(stale, 0, "no stale row in this fixture");
    assert_eq!(frustum, 0, "no frustum-culled row in this fixture");
    // Non-vacuity guard: prove the OOB row's drop is specific to it, not a
    // shader that dropped everything.
    assert_eq!(visible, 1, "exactly h_ok must be visible");
    assert_eq!(oob, 1, "oob_drops must equal exactly the 1 injected out-of-range row");

    let ok_row = cell.row_of(h_ok).unwrap();
    let oob_row = cell.row_of(h_oob).unwrap();
    assert!(by_row.contains_key(&ok_row), "h_ok's row must have a command");
    assert!(!by_row.contains_key(&oob_row), "h_oob's row must have NO command — dropped by the bounds check");
    let (index_count, first_index, base_vertex, _) = by_row[&ok_row];
    assert_eq!(index_count, 6, "h_ok's command uses mesh0.index_count");
    assert_eq!(first_index, 0, "h_ok's command uses mesh0.index_offset");
    assert_eq!(base_vertex, 0, "h_ok's command uses mesh0.base_vertex");
}
