//! `CullPass` end-to-end proof (M3-b T5, design S4): build a real scene
//! through SceneDB's public write/harvest API (mirrors `seam_smoke.rs`'s
//! `test_context`/boundary-driving pattern and `gpu_harvest.rs`'s harvest
//! pattern), upload the harvested tokens via `ViewTokenBuffers` (M3-b T1),
//! dispatch `CullPass`, and read back the command buffer + visible ids +
//! telemetry counters. Every expected value below is HAND-COMPUTED from the
//! fixture's own geometry, not re-derived by re-running the shader's logic
//! in Rust (that CPU-reference equality check is M3-b T6's job,
//! `tests/cull_equality.rs`).
//!
//! ## The fixture (house law: self-verifying guards -- every category
//! non-empty before trusting the aggregate assertion)
//!
//! Camera at the world origin, looking down -Z, with an identity view
//! matrix (world space IS view space here) and a symmetric fovy=90/aspect=1
//! perspective projection (near=1, far=100 -- `support::view_proj_90deg`).
//! Four instances, one cell, fresh allocations (slot == row == index,
//! generation 1 for all -- no stale/oob category exercised here, that is
//! Task 6's dedicated `tests/cull_stale_and_oob.rs`).
//!
//! | row | translation      | mesh (local extents)     | category      |
//! |-----|-------------------|---------------------------|---------------|
//! | 0   | (0, 0, -10)       | mesh0, (0.5,0.5,0.5)      | VISIBLE       |
//! | 1   | (100, 0, -10)     | mesh0, (0.5,0.5,0.5)      | FRUSTUM-CULLED|
//! | 2   | (0, 0, -0.5)      | mesh1, (0.5,0.5,2.0)      | NEAR-CLIP     |
//! | 3   | rotated, see below | mesh2, (0.5,0.5,2.0)     | VISIBLE (rotation regression pin, M3-b T6 fold-in) |
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
//!
//! ## Row 3 -- the rotation regression pin (M3-b T6 fold-in, T5 review)
//!
//! `page.rs`'s `Pod for [f32; 16]` doc comment pins the instance-transform
//! flattening convention as COLUMN-MAJOR (`array[4*col+row] = M[row][col]`,
//! i.e. `to_cols_array()`, no transpose) and records the exact landmine a
//! transposed (naively "row-major") reading produces: `|M_3x3|` computed
//! from `R^T` instead of `R` silently changes the world-space AABB extents
//! for ANY non-symmetric rotation matrix, with NO scale needed to trigger
//! it (element-wise `abs` does not commute with transpose unless the matrix
//! is symmetric). Every fixture row above this one uses either a pure
//! translation (rotation-free, transpose-invariant in this flat layout) or
//! never exercised in a way that discriminates. Row 3 closes that gap: a
//! genuine two-axis rotation `Rz(30 deg) * Rx(40 deg)` (matrix product,
//! applied to a column vector as `Rz * (Rx * v)`), positioned at the exact
//! `x` translation where the CORRECTLY-read world AABB is (barely) inside
//! the right frustum plane and the WRONGLY-transposed-read world AABB is
//! (barely) outside it -- see the self-verifying guard below, which
//! computes both extents independently (via `support::mat3_abs_mul_vec3`
//! on `R` and on `R^T`) and asserts they straddle the frustum's visibility
//! threshold with a real margin BEFORE trusting the main dispatch assertion
//! that row 3 is visible. If `abs_mat3`/the buffer's column-major reading
//! ever regresses to a transposed interpretation, row 3 flips to
//! frustum-culled and `by_row.get(&3)` panics.

#[path = "support/mod.rs"]
mod support;

use helio_scenedb::cull::{CullOutputBuffers, CullPass, CullUniforms, NEAR_CLIP_FLAG};
use helio_scenedb::SceneDbBinding;
use pulsar_scenedb::gpu::{
    CellSlot, ClusterBuffer, FrameDriver, HarvestPipeline, HarvestStaging, MaterialRegistry,
    MeshClass, MeshRegistry, MeshletBuffer, SceneGpuStore, View, ViewTokenBuffers,
};
use pulsar_scenedb::{Aabb, InstanceInfo, Scratchpad, SpatialCell};
use std::collections::HashMap;

use support::{
    frustum_planes_90deg, mat3_abs_mul_vec3, mat3_mul, mat3_rot_x, mat3_rot_z, mat3_transpose,
    mesh, readback, scene_cfg, test_context, translation, view_proj_90deg, flatten_column_major,
};

#[test]
fn cull_pass_produces_exact_hand_computed_visible_set() {
    let ctx = test_context();
    let device: &wgpu::Device = ctx.device().as_ref();

    // --- Build the scene: one cell, 4 fresh instances (rows 0..3, slot ==
    // row, generation 1 for all — no stale/oob category in this fixture,
    // that lives in tests/cull_stale_and_oob.rs). ---
    let mut store = SceneGpuStore::new(&ctx, scene_cfg());
    let mut cell = SpatialCell::with_transform(64).unwrap();
    // Harvest (broad-phase) AABBs: generous unit boxes around each
    // translation so ONE big harvest query AABB catches all four — the
    // cull SHADER computes its own precise world AABB from mesh_meta +
    // transform independently (S11), this is only the CPU-side coarse
    // query bound.
    let h0 = cell.alloc(Aabb { min: [-1.0, -1.0, -11.0], max: [1.0, 1.0, -9.0] }).unwrap();
    let h1 = cell.alloc(Aabb { min: [99.0, -1.0, -11.0], max: [101.0, 1.0, -9.0] }).unwrap();
    let h2 = cell.alloc(Aabb { min: [-1.0, -1.0, -1.5], max: [1.0, 1.0, 0.5] }).unwrap();

    // --- Row 3's rotation regression pin: Rz(30deg) * Rx(40deg), local
    // extents (0.5, 0.5, 2.0) (same shape as mesh1, row 2), local center
    // at the origin (so world_center == the translation, no extra offset
    // math needed). The self-verifying guard below independently computes
    // BOTH the correct (R) and the transposed-bug (R^T) world extents and
    // derives the x-translation that straddles the right frustum plane's
    // visibility threshold for exactly this rotation/extents pair — this
    // is NOT a hand-copied magic number, it is computed fresh at test time
    // from `support`'s matrix helpers, independently of `abs_mat3` in
    // CULL_WGSL. ---
    let rz30 = mat3_rot_z(30.0);
    let rx40 = mat3_rot_x(40.0);
    let r3 = mat3_mul(rz30, rx40); // Rz * Rx, applied to a column vector as Rz*(Rx*v)
    let r3_wrong = mat3_transpose(r3);
    let row3_local_extents = [0.5_f32, 0.5, 2.0];
    let correct_extents = mat3_abs_mul_vec3(r3, row3_local_extents);
    let wrong_extents = mat3_abs_mul_vec3(r3_wrong, row3_local_extents);
    // Right frustum plane: normal (-1, 0, -1), w=0 (`frustum_planes_90deg`
    // index 1) -- r = extents.x + extents.z (|nx|=|nz|=1, |ny|=0, so
    // extents.y never enters this particular plane's test, which is why
    // this specific plane cleanly isolates the x/z components of the
    // transpose bug for this rotation). Visible requires
    // `x_c + z_c <= r` (derived from `d = -x_c - z_c >= -r`).
    let row3_z = -10.0_f32;
    let thresh_correct = (correct_extents[0] + correct_extents[2]) - row3_z; // r_correct + 10
    let thresh_wrong = (wrong_extents[0] + wrong_extents[2]) - row3_z; // r_wrong + 10
    // Self-verifying guard (house law): the two thresholds must be
    // meaningfully separated (not coincidentally equal) BEFORE trusting a
    // midpoint x_c to discriminate between them — a future change to the
    // rotation/extents pair that collapsed this gap would silently turn
    // row 3 into a non-discriminating fixture.
    assert!(
        (thresh_correct - thresh_wrong).abs() > 0.05,
        "guard: the correct-vs-transposed visibility thresholds must differ by a real margin \
         (correct={thresh_correct}, wrong={thresh_wrong}) — otherwise row 3 cannot pin the \
         row/col-major landmine"
    );
    let row3_x = (thresh_correct + thresh_wrong) / 2.0;
    assert!(
        row3_x < thresh_correct && row3_x > thresh_wrong,
        "guard: chosen x_c ({row3_x}) must sit strictly between the wrong threshold \
         ({thresh_wrong}, would already cull) and the correct one ({thresh_correct}, still visible) \
         — i.e. this fixture instance IS visible under the pinned column-major convention and \
         WOULD be frustum-culled under a transposed (row-major) reading"
    );
    let row3_transform = flatten_column_major(r3, [row3_x, 0.0, row3_z]);
    let h3 = cell
        .alloc(Aabb {
            min: [row3_x - 2.0, -2.0, row3_z - 2.0],
            max: [row3_x + 2.0, 2.0, row3_z + 2.0],
        })
        .unwrap();

    let id = store.register_cell(cell.storage(), 0).unwrap();
    let base = store.row_region_base(id);
    assert_eq!(base, 0, "single cell, class-0 region base 0 — rows below are global rows verbatim");

    let mut frames = FrameDriver::new();
    let sim = frames.begin();
    assert!(store.write_transform(id, cell.storage_mut(), h0, &translation([0.0, 0.0, -10.0]), &sim));
    assert!(store.write_transform(id, cell.storage_mut(), h1, &translation([100.0, 0.0, -10.0]), &sim));
    assert!(store.write_transform(id, cell.storage_mut(), h2, &translation([0.0, 0.0, -0.5]), &sim));
    assert!(store.write_transform(id, cell.storage_mut(), h3, &row3_transform, &sim));
    assert!(store.write_instance_info(id, cell.storage_mut(), h0, InstanceInfo { mesh_index: 0, flags: 0 }, &sim));
    assert!(store.write_instance_info(id, cell.storage_mut(), h1, InstanceInfo { mesh_index: 0, flags: 0 }, &sim));
    assert!(store.write_instance_info(id, cell.storage_mut(), h2, InstanceInfo { mesh_index: 1, flags: 0 }, &sim));
    assert!(store.write_instance_info(id, cell.storage_mut(), h3, InstanceInfo { mesh_index: 2, flags: 0 }, &sim));

    // --- Harvest: one big AABB view over the whole cell (S3.1's expected-
    // generation column comes along for free — HarvestPipeline::harvest_cell
    // always emits it positionally aligned with the token). ---
    let harvest_phase = sim.end().end();
    let pipeline = HarvestPipeline::new();
    let mut pad = Scratchpad::new();
    let mut staging = HarvestStaging::new();
    let view = View::Aabb(Aabb { min: [-200.0, -200.0, -200.0], max: [200.0, 200.0, 200.0] });
    let n = pipeline.harvest_cell(&cell, base, MeshClass::Traditional, &view, &mut pad, &mut staging, &harvest_phase);
    assert_eq!(n, 4, "sanity: the broad harvest view catches all four rows");
    assert_eq!(staging.traditional.len(), 4);

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
    assert_eq!(view_tokens.count(), 4);

    // --- Asset-side stores SceneDbBinding also binds (mesh registry carries
    // the three real meshes this fixture's hand-computed expectations use;
    // cluster/meshlet/material registries are unused by cull, empty is
    // fine — mirrors seam_smoke.rs). ---
    let mut meshes = MeshRegistry::new(&ctx, 4);
    let mesh0 = meshes.register(ctx.queue(), mesh([0.0, 0.0, 0.0], [0.5, 0.5, 0.5], 6, 0, 0)).unwrap();
    let mesh1 = meshes.register(ctx.queue(), mesh([0.0, 0.0, 0.0], [0.5, 0.5, 2.0], 12, 6, 4)).unwrap();
    let mesh2 = meshes.register(ctx.queue(), mesh([0.0, 0.0, 0.0], row3_local_extents, 18, 18, 8)).unwrap();
    assert_eq!((mesh0, mesh1, mesh2), (0, 1, 2), "sanity: mesh_index values the fixture's InstanceInfo rows used above");
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
    // exercise (stale/oob — tests/cull_stale_and_oob.rs's job) must read
    // exactly zero. ---
    assert_eq!(stale_drops, 0, "fixture has no stale-generation row (tests/cull_stale_and_oob.rs's job)");
    assert_eq!(oob_drops, 0, "fixture has no out-of-range mesh_index row (tests/cull_stale_and_oob.rs's job)");
    assert_eq!(frustum_drops, 1, "guard: exactly one frustum-culled row (row 1) — must be non-zero");
    assert_eq!(visible_count, 3, "guard: exactly three visible rows (row 0 + row 2 near-clip + row 3 rotation) — must be non-zero");

    // --- Parse the `visible_count` (== 3, within `capacity` == 8, so all
    // slots were written) records, keyed by `row` — command-slot ORDER
    // between concurrently-executing GPU threads is not guaranteed, so this
    // assertion is written to be robust to any row landing in any slot,
    // while still being fully hand-computed per row. ---
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
    assert_eq!(by_row.len(), 3, "three distinct rows, no duplicate/lost slot");

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

    // Row 3 (VISIBLE, mesh2, the rotation regression pin): hand-computed
    // from mesh2's registered fields (index_count=18, index_offset=18,
    // base_vertex=8). THE assertion that matters most here is simply that
    // this entry EXISTS — under a transposed-matrix read, row 3 would be
    // frustum-culled instead (see the guard above), and `by_row.get(&3)`
    // would panic.
    let row3 = by_row.get(&3).expect(
        "row 3 (rotation regression pin) must have a command — if this fails, the buffer's \
         instance-transform column is being read transposed (row-major) instead of the pinned \
         column-major convention (page.rs's Pod for [f32; 16] doc comment)",
    );
    assert_eq!(row3.0, 18, "row 3 index_count == mesh2.index_count");
    assert_eq!(row3.2, 18, "row 3 first_index == mesh2.index_offset");
    assert_eq!(row3.3, 8, "row 3 base_vertex == mesh2.base_vertex");
    assert_eq!(row3.5 & NEAR_CLIP_FLAG, 0, "row 3 is NOT near-clip — flag bit 0 clear (z=-10, well clear of the camera)");

    // Row 1 (FRUSTUM-CULLED) must have no command at all.
    assert!(!by_row.contains_key(&1), "row 1 was frustum-culled — no command, no visible id");
}
