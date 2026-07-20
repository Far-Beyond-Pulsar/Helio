//! M3-b T9-followup, deliverables 2/3: the PERMANENT equivalence pin the T9
//! review's throwaway probe never left behind. T9 proved
//! `DrawExecutor::record_multi_indirect` (strategy (b'): one
//! `multi_draw_indexed_indirect` call over a repacked, tightly-packed
//! 20-byte args buffer) renders byte-identical output to
//! `DrawExecutor::record` (strategy (a): a CPU-side loop of
//! `draw_indexed_indirect` calls reading the cull pass's own 32-byte-
//! strided `CullRecord` buffer directly) -- then deleted the probe. That
//! coverage does not persist across future edits to either draw path OR to
//! the repack pass this file's sibling module (`src/repack.rs`) now ships.
//! This test file is the replacement: it stays in the tree.
//!
//! ## The fixture (house law: self-verifying guards)
//!
//! Reuses the T9 reviewer's fixture SHAPE, chosen specifically to catch a
//! SCRAMBLED slot->row mapping: >= 20 tokens (24 here, one cell), several
//! (5) visible at DISTINCT x-positions -- reusing `draw_transform_sweep.
//! rs`'s own five already-verified sweep positions (x = -8.75/-3.75/1.25/
//! 6.25/8.75 at z=-10, landing in screen columns 0/2/4/6/7 of an 8-wide
//! coarse grid, `support::view_proj_90deg`'s fovy=90 symmetric projection)
//! -- alternating `mesh_index` 0/1/0/1/0 so the row lookup must resolve
//! PER INSTANCE, not just "some row". If a repack (or `record_multi_
//! indirect` itself) ever scrambled which packed record's `first_instance`
//! maps to which cull slot, a visible instance would either vanish, paint
//! the wrong column, or paint the wrong color -- any of which fails this
//! file's per-color pixel-set assertions or the byte-identical target
//! comparison below. The remaining 19 of 24 tokens sit at x=1000 (the same
//! frustum-culled position `benches/pass_timing.rs`'s fixture and `cull_
//! pass.rs`'s row 1 both use) -- a genuine culled MAJORITY, so both the
//! visible and the culled path are exercised, not just the trivial
//! all-visible case.

#[path = "support/mod.rs"]
mod support;

use helio_scenedb::cull::{CullOutputBuffers, CullPass, CullRecord, CullUniforms};
use helio_scenedb::draw::{mesh_color_rgba8, DrawExecutor, DrawUniforms, OffscreenTarget};
use helio_scenedb::repack::{packed_indirect_buffer, RepackPass};
use helio_scenedb::SceneDbBinding;
use pulsar_scenedb::gpu::{
    CellSlot, ClusterBuffer, FrameDriver, GeometryArena, HarvestPipeline, HarvestStaging, MaterialRegistry,
    MeshClass, MeshRegistry, MeshletBuffer, SceneGpuStore, View, ViewTokenBuffers,
};
use pulsar_scenedb::{Aabb, InstanceInfo, Scratchpad, SpatialCell};

use support::{mesh, readback, scene_cfg, test_context_indirect_first_instance, translation, view_proj_90deg};

const QUAD_VERTICES: [[f32; 3]; 4] = [
    [-0.5, -0.5, 0.0],
    [0.5, -0.5, 0.0],
    [0.5, 0.5, 0.0],
    [-0.5, 0.5, 0.0],
];
const QUAD_INDICES: [u32; 6] = [0, 1, 2, 0, 2, 3];

/// Five visible x-positions, DISTINCT enough to land in 5 distinct 8-wide
/// screen columns (verified against `draw_transform_sweep.rs`'s identical
/// values -- see that file's own comment for the u/column arithmetic) --
/// reused verbatim rather than re-derived, since that file already proves
/// these five discriminate. `z = -10.0` for all (same depth as `cull_pass.
/// rs`'s row 0 / `benches/pass_timing.rs`'s visible position).
const VISIBLE_X: [f32; 5] = [-8.75, -3.75, 1.25, 6.25, 8.75];
const Z: f32 = -10.0;
/// The frustum-culled position -- `benches/pass_timing.rs`'s fixture and
/// `cull_pass.rs`'s row 1 both use x=100/x=1000 as "comfortably outside the
/// fovy=90 side planes at z=-10" (|x| <= 10 is the visibility bound there);
/// 1000 keeps a wide margin.
const CULLED_X: f32 = 1000.0;
const N: u32 = 24;
const VISIBLE_COUNT: u32 = VISIBLE_X.len() as u32; // 5

#[test]
fn record_and_record_multi_indirect_render_byte_identical_targets() {
    let ctx = test_context_indirect_first_instance();
    let device: &wgpu::Device = ctx.device().as_ref();
    let queue = ctx.queue();

    // --- Scene: one cell, N=24 rows. Rows 0..VISIBLE_COUNT sit at the
    // five distinct visible x-positions (alternating mesh_index 0/1);
    // rows VISIBLE_COUNT..N sit at the culled position. ---
    let mut store = SceneGpuStore::new(&ctx, scene_cfg());
    let mut cell = SpatialCell::with_transform(32).unwrap();
    let mut handles = Vec::with_capacity(N as usize);
    for row in 0..N {
        let x = if row < VISIBLE_COUNT { VISIBLE_X[row as usize] } else { CULLED_X };
        let h = cell.alloc(Aabb { min: [x - 1.0, -1.0, Z - 1.0], max: [x + 1.0, 1.0, Z + 1.0] }).unwrap();
        handles.push(h);
    }
    let id = store.register_cell(cell.storage(), 0).unwrap();
    let base = store.row_region_base(id);
    assert_eq!(base, 0, "single cell, class-0 region base 0");

    let mut frames = FrameDriver::new();
    let sim = frames.begin();
    for (row, &h) in handles.iter().enumerate() {
        let row = row as u32;
        let x = if row < VISIBLE_COUNT { VISIBLE_X[row as usize] } else { CULLED_X };
        let mesh_index = if row < VISIBLE_COUNT { row % 2 } else { 0 };
        assert!(store.write_transform(id, cell.storage_mut(), h, &translation([x, 0.0, Z]), &sim));
        assert!(store.write_instance_info(id, cell.storage_mut(), h, InstanceInfo { mesh_index, flags: 0 }, &sim));
    }

    let harvest_phase = sim.end().end();
    let pipeline = HarvestPipeline::new();
    let mut pad = Scratchpad::new();
    let mut staging = HarvestStaging::new();
    let view = View::Aabb(Aabb { min: [-2000.0, -2000.0, -2000.0], max: [2000.0, 2000.0, 2000.0] });
    let harvested = pipeline.harvest_cell(&cell, base, MeshClass::Traditional, &view, &mut pad, &mut staging, &harvest_phase);
    assert_eq!(harvested, N, "sanity: harvest must catch every synthetic row");

    let boundary = harvest_phase.end();
    {
        let mut slots = [CellSlot { id, cell: cell.storage_mut() }];
        boundary.run(&mut store, &mut slots);
    }

    let mut view_tokens = ViewTokenBuffers::new(&ctx, "multi-indirect-equiv-view", 0);
    view_tokens.upload(&ctx, &staging, MeshClass::Traditional);
    assert_eq!(view_tokens.count(), N);

    // --- Geometry: one quad, registered twice (mesh_index 0 and 1) over
    // the SAME vertex/index range -- the two instances used above differ
    // only by InstanceInfo.mesh_index (mirrors `draw_transform_sweep.rs`'s
    // fixture idiom, so a per-color pixel-set assertion below actually
    // proves the row->instance_info->color lookup, not just "some pixel is
    // some color"). ---
    let mut arena = GeometryArena::new(&ctx, 4096, 4096);
    let vbytes: &[u8] = bytemuck::cast_slice(&QUAD_VERTICES);
    let ibytes: &[u8] = bytemuck::cast_slice(&QUAD_INDICES);
    let voff = arena.upload_vertices(queue, vbytes).unwrap();
    let ioff = arena.upload_indices(queue, ibytes).unwrap();
    let base_vertex = (voff / 12) as i32;
    let first_index = ioff / 4;

    let mut meshes = MeshRegistry::new(&ctx, 4);
    let mesh0 = meshes
        .register(queue, mesh([0.0, 0.0, 0.0], [0.5, 0.5, 0.5], QUAD_INDICES.len() as u32, first_index, base_vertex))
        .unwrap();
    let mesh1 = meshes
        .register(queue, mesh([0.0, 0.0, 0.0], [0.5, 0.5, 0.5], QUAD_INDICES.len() as u32, first_index, base_vertex))
        .unwrap();
    assert_eq!((mesh0, mesh1), (0, 1), "sanity: mesh_index values the fixture's InstanceInfo rows used above");

    let clusters = ClusterBuffer::new(&ctx, 1);
    let meshlets = MeshletBuffer::new(&ctx, 1);
    let materials = MaterialRegistry::new(&ctx, 1);
    let scene_binding = SceneDbBinding::new(device, &store, &meshes, &clusters, &meshlets, &materials);

    // --- Cull pass: ONE real dispatch, output capacity == N == 24 (so
    // "the last slot at capacity", slot 23, is a real, addressable, never-
    // written-by-cull slot -- deliverable 3's spot-pin target). ---
    let cull_pass = CullPass::new(device, &scene_binding.cull_layout);
    let output = CullOutputBuffers::new(device, "multi-indirect-equiv-cull-output", N);
    output.clear_counters(queue);
    let cull_output_bind_group = cull_pass.build_output_bind_group(device, &view_tokens, &output);
    let uniforms = CullUniforms {
        view_proj: view_proj_90deg(),
        planes: support::frustum_planes_90deg(),
        count: view_tokens.count(),
        mesh_count: meshes.len(),
        capacity: output.capacity(),
        reserved: 0,
    };
    cull_pass.write_uniforms(queue, &uniforms);
    {
        let mut encoder = device.create_command_encoder(&Default::default());
        cull_pass.record(&mut encoder, &scene_binding.cull_bind_group, &cull_output_bind_group, view_tokens.count());
        queue.submit([encoder.finish()]);
    }

    let cull_bytes = readback(&ctx, output.buffer(), output.byte_size());
    let u32_at = |bytes: &[u8], off: usize| u32::from_le_bytes(bytes[off..off + 4].try_into().unwrap());
    let visible_count = u32_at(&cull_bytes, 0);
    let stale_drops = u32_at(&cull_bytes, 4);
    let oob_drops = u32_at(&cull_bytes, 8);
    let frustum_drops = u32_at(&cull_bytes, 12);
    assert_eq!(stale_drops, 0, "fixture has no stale row");
    assert_eq!(oob_drops, 0, "fixture has no OOB mesh_index row");
    assert_eq!(frustum_drops, N - VISIBLE_COUNT, "guard: every non-visible row must be frustum-culled");
    assert_eq!(visible_count, VISIBLE_COUNT, "guard: exactly the 5 fixture rows placed at visible x-positions must be visible");
    // House-law guard (deliverable 2's explicit requirement): visible count
    // must be strictly between 1 and the token count -- proves BOTH the
    // visible and the culled path are genuinely exercised, not a
    // degenerate all-visible or all-culled fixture.
    assert!(visible_count > 1 && visible_count < N, "guard: 1 < visible_count ({visible_count}) < token_count ({N})");

    // ==================================================================
    // Deliverable 3: repack spot-pin -- compare a few repacked
    // `PackedArgs` records against their source `CullRecord`s FIELD FOR
    // FIELD, at three slots chosen to catch a repack that's plausible but
    // shifted: the first visible slot (0), the first zero-instance/culled
    // slot (== visible_count, the first slot the bounded-atomic cull pass
    // never wrote), and the last slot at capacity (N-1 == 23).
    // ==================================================================
    let repack_pass = RepackPass::new(device);
    repack_pass.write_capacity(queue, output.capacity());
    let packed_buf = packed_indirect_buffer(device, output.capacity());
    let repack_bind_group = repack_pass.build_bind_group(device, &output, &packed_buf);
    {
        let mut encoder = device.create_command_encoder(&Default::default());
        repack_pass.record(&mut encoder, &repack_bind_group, output.capacity());
        queue.submit([encoder.finish()]);
    }
    let packed_bytes = readback(&ctx, &packed_buf, output.capacity() as u64 * 20);

    let cull_record_at = |slot: u32| -> CullRecord {
        let off = CullOutputBuffers::HEADER_BYTES as usize + slot as usize * CullOutputBuffers::RECORD_BYTES as usize;
        CullRecord {
            index_count: u32_at(&cull_bytes, off),
            instance_count: u32_at(&cull_bytes, off + 4),
            first_index: u32_at(&cull_bytes, off + 8),
            base_vertex: i32::from_le_bytes(cull_bytes[off + 12..off + 16].try_into().unwrap()),
            first_instance: u32_at(&cull_bytes, off + 16),
            row: u32_at(&cull_bytes, off + 20),
            flags: u32_at(&cull_bytes, off + 24),
            reserved: u32_at(&cull_bytes, off + 28),
        }
    };
    // `(index_count, instance_count, first_index, base_vertex, first_instance)`
    // -- the packed buffer's own 20-byte shape.
    let packed_args_at = |slot: u32| -> (u32, u32, u32, i32, u32) {
        let off = slot as usize * 20;
        (
            u32_at(&packed_bytes, off),
            u32_at(&packed_bytes, off + 4),
            u32_at(&packed_bytes, off + 8),
            i32::from_le_bytes(packed_bytes[off + 12..off + 16].try_into().unwrap()),
            u32_at(&packed_bytes, off + 16),
        )
    };
    let assert_repack_matches = |slot: u32, label: &str| {
        let src = cull_record_at(slot);
        let packed = packed_args_at(slot);
        assert_eq!(
            packed,
            (src.index_count, src.instance_count, src.first_index, src.base_vertex, src.first_instance),
            "repack spot-pin FAILED at slot {slot} ({label}): packed args {packed:?} must equal source \
             CullRecord's first 20 bytes {src:?} field-for-field (index_count, instance_count, \
             first_index, base_vertex, first_instance) -- first_instance especially, the §14.1 \
             command-slot bindless key"
        );
    };
    assert_repack_matches(0, "first visible slot");
    assert_repack_matches(VISIBLE_COUNT, "first zero-instance/culled slot (never written by the bounded-atomic cull pass)");
    assert_repack_matches(N - 1, "last slot at capacity");
    // Self-verifying guard: the "zero-instance" slot must ACTUALLY be
    // zero-instance (i.e. this fixture's capacity genuinely exceeds
    // visible_count, so the spot-pin above exercises a real never-written
    // slot, not an accidental second visible one).
    assert_eq!(
        cull_record_at(VISIBLE_COUNT).instance_count,
        0,
        "guard: slot {VISIBLE_COUNT} must be a genuine zero-instance/never-written slot"
    );

    // ==================================================================
    // Deliverables 1/2: render the SAME scene twice -- strategy (a) via
    // `DrawExecutor::record` (reads `CullRecord`s directly), strategy (b')
    // via `DrawExecutor::record_multi_indirect` + the just-verified
    // repacked buffer -- into two SEPARATE offscreen targets, then compare
    // bytes. Both issue `output.capacity()` (24) draw "slots": the tail
    // (zero-instance) slots are harmless no-op draws on EITHER path
    // (`instance_count == 0`), so this also exercises the repack's
    // handling of the culled majority, not just the 5 visible rows.
    // ==================================================================
    let draw_executor = DrawExecutor::new(device, &scene_binding.cull_layout);
    let draw_output_bind_group = draw_executor.build_output_bind_group(device, &output);
    draw_executor.write_uniforms(queue, &DrawUniforms { view_proj: view_proj_90deg() });

    let target_a = OffscreenTarget::new(device, "multi-indirect-equiv-target-a");
    {
        let mut encoder = device.create_command_encoder(&Default::default());
        draw_executor.record(
            &mut encoder,
            target_a.view(),
            &scene_binding.cull_bind_group,
            &draw_output_bind_group,
            &output,
            arena.vertex_buffer(),
            arena.index_buffer(),
            output.capacity(),
        );
        queue.submit([encoder.finish()]);
    }
    let pixels_a = target_a.read_pixels(device, queue);

    let target_b = OffscreenTarget::new(device, "multi-indirect-equiv-target-b");
    {
        let mut encoder = device.create_command_encoder(&Default::default());
        draw_executor.record_multi_indirect(
            &mut encoder,
            target_b.view(),
            &scene_binding.cull_bind_group,
            &draw_output_bind_group,
            &packed_buf,
            0,
            output.capacity(),
            arena.vertex_buffer(),
            arena.index_buffer(),
        );
        queue.submit([encoder.finish()]);
    }
    let pixels_b = target_b.read_pixels(device, queue);

    // --- Self-verifying guards (house law): non-empty targets -- two
    // blank targets would match vacuously and defeat the whole point of
    // this comparison. "Painted" == any pixel with alpha == 255 (the
    // clear color is transparent black, alpha 0; `mesh_color`'s alpha is
    // always 1.0). ---
    let painted_count = |pixels: &[u8]| pixels.chunks_exact(4).filter(|px| px[3] == 255).count();
    let painted_a = painted_count(&pixels_a);
    let painted_b = painted_count(&pixels_b);
    assert!(painted_a > 0, "guard: strategy (a)'s target must be non-empty (painted {painted_a} pixels)");
    assert!(painted_b > 0, "guard: strategy (b')'s target must be non-empty (painted {painted_b} pixels)");

    // --- Self-verifying guard: at least two distinct mesh colors present
    // -- proves the row lookup resolved PER INSTANCE (mesh_index 0 vs 1),
    // not one instance repeated across every visible slot. ---
    let color0 = mesh_color_rgba8(0);
    let color1 = mesh_color_rgba8(1);
    assert_ne!(color0, color1, "guard: mesh_index 0/1 colors must differ");
    let has_color = |pixels: &[u8], c: [u8; 4]| pixels.chunks_exact(4).any(|px| px == c);
    assert!(has_color(&pixels_a, color0) && has_color(&pixels_a, color1), "guard: strategy (a) must paint BOTH mesh colors");
    assert!(has_color(&pixels_b, color0) && has_color(&pixels_b, color1), "guard: strategy (b') must paint BOTH mesh colors");

    // --- THE deliverable-gate assertion: byte-identical targets. ---
    let diff_count = pixels_a.iter().zip(pixels_b.iter()).filter(|(a, b)| a != b).count();
    assert_eq!(
        diff_count, 0,
        "strategy (a) [record] and strategy (b') [record_multi_indirect + repack] must render \
         byte-identical targets ({diff_count}/{} bytes differ)",
        pixels_a.len()
    );
    assert_eq!(pixels_a, pixels_b, "byte-identical target check (redundant with diff_count above, kept for a precise Vec diff on failure)");
}
