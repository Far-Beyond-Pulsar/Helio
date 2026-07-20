//! Test 4 (design S5/S14, M3-b T7): the transform sweep. Sweeps ONE
//! instance's transform across six frames (>= 5 positions, INCLUDING a
//! rotated pose -- the column-major transform convention is pinned
//! (`page.rs`'s `Pod for [f32; 16]` doc comment) and the T5/T6 review's
//! rotation regression pin lives cull-side (`cull_pass.rs` row 3). SCOPE,
//! corrected per the M3-b T7 review: this test's rotated pose proves a
//! NON-IDENTITY rotation reached the vertex shader's
//! `instances[row].transform` read (asserted on painted-footprint size,
//! after the sweep loop) -- it does NOT catch a transposed-matrix read,
//! because R(45deg) and R(-45deg) map this centered symmetric quad to the
//! identical point set. Transpose detection lives cull-side, where row 3
//! uses a deliberately non-symmetric two-axis rotation. A draw-side
//! transpose pin would need an off-center or asymmetric quad so rotation
//! DIRECTION moves the centroid -- known gap, recorded not overclaimed):
//! each frame
//! `write_transform` -> boundary -> a REAL `CullPass` dispatch -> a REAL
//! `DrawExecutor` indirect draw over that dispatch's own command buffer ->
//! offscreen-target readback. Every frame re-derives its own expected
//! screen column from the SAME symmetric fovy=90/aspect=1 projection
//! `support::view_proj_90deg` already establishes, independently of the
//! shader (the projection math is `x_ndc = x_world / (-z_world)`, derived
//! by hand from that matrix's own coefficients, not by re-running WGSL in
//! Rust).
//!
//! ## The real cull->draw handoff (report honesty note, restated here)
//!
//! Nothing here is shortcut: `CullPass::record` runs a real compute
//! dispatch each frame against the JUST-boundary-synced transform,
//! `CullOutputBuffers` is read back for its own `visible_count` (the S14.2
//! CPU clamp), and `DrawExecutor::record` issues `draw_indexed_indirect`
//! calls that read their `DrawIndexedIndirectArgs` prefix directly out of
//! THAT SAME buffer the cull dispatch just wrote -- the vertex shader's
//! `draw_cull_output.records[iid].row` lookup resolves through the actual
//! per-frame cull output, not a precomputed/faked row index. The ONE
//! CPU-side value the draw call consumes (`command_count`) comes from a
//! real `map_async` readback of the cull pass's own atomic counter, exactly
//! mirroring the S14.2 clamp-after-readback design (T9 is the dedicated
//! task for measuring this readback's latency; this test pays it once per
//! frame without trying to hide or amortize it).
//!
//! ## Two instances per frame -- proving row indirection, not just presence
//!
//! A lone moving instance would let `first_instance == 0` always resolve to
//! row 0 even if the shader ignored `draw_cull_output.records[iid].row`
//! entirely and just used `iid` as the row directly (row == slot == 0 in
//! that degenerate one-instance case). A SECOND, static "decoy" instance
//! (different row, different `mesh_index`, positioned far off in Y so its
//! screen footprint never spatially overlaps the mover's, regardless of the
//! cull pass's nondeterministic slot-allocation order across its two
//! concurrent GPU threads) forces the shader to genuinely resolve
//! `slot -> row` per draw call: whichever of the two rows lands in which
//! slot, both must still end up in their OWN correct screen position with
//! their OWN correct color, or this test's per-color pixel-set assertions
//! fail.

#[path = "support/mod.rs"]
mod support;

use helio_scenedb::cull::{CullOutputBuffers, CullPass, CullUniforms};
use helio_scenedb::draw::{mesh_color_rgba8, DrawExecutor, DrawUniforms, OffscreenTarget};
use helio_scenedb::SceneDbBinding;
use pulsar_scenedb::gpu::{
    CellSlot, ClusterBuffer, FrameDriver, GeometryArena, HarvestPipeline, HarvestStaging, MaterialRegistry,
    MeshClass, MeshRegistry, MeshletBuffer, SceneGpuStore, View, ViewTokenBuffers,
};
use pulsar_scenedb::{Aabb, InstanceInfo, Scratchpad, SpatialCell};

use support::{
    flatten_column_major, mat3_rot_z, mesh, readback, scene_cfg, test_context_indirect_first_instance, translation,
    view_proj_90deg,
};

/// Local-space unit quad (two triangles), `vec3<f32>` positions -- the
/// trivial mesh `DrawExecutor::new`'s vertex layout expects (`array_stride:
/// 12`). Half-extent 0.5, matching this suite's other fixtures' object
/// scale (`cull_pass.rs`'s mesh0).
const QUAD_VERTICES: [[f32; 3]; 4] = [
    [-0.5, -0.5, 0.0],
    [0.5, -0.5, 0.0],
    [0.5, 0.5, 0.0],
    [-0.5, 0.5, 0.0],
];
const QUAD_INDICES: [u32; 6] = [0, 1, 2, 0, 2, 3];

/// Screen-space 8x8 coarse grid (design plan T7: "coarse grid assert...
/// pixel-exact rasterization comparison is explicitly NOT the goal").
const GRID: u32 = 8;

/// `x_ndc = x_world / (-z_world)` -- hand-derived from `view_proj_90deg`'s
/// own coefficients (`support` module doc / `cull_pass.rs`'s own comments):
/// that matrix's clip.x is untouched world x, clip.w is `-z_world`, and
/// fovy=90/aspect=1 gives a slope of exactly 1, so NDC x IS `x/(-z)` with no
/// extra scale factor. Maps to a column index in `0..GRID`.
fn expected_column(x_world: f32, z_world: f32) -> u32 {
    let ndc_x = x_world / -z_world;
    let u = (ndc_x * 0.5 + 0.5) * OffscreenTarget::WIDTH as f32;
    ((u / (OffscreenTarget::WIDTH as f32 / GRID as f32)) as u32).min(GRID - 1)
}

/// Pixel coordinates (row-major, `OffscreenTarget::WIDTH`-wide) whose RGBA8
/// value matches `expected` within `tol` per channel (Rgba8Unorm round-trip
/// tolerance -- the fragment shader computes `f32(byte)/255.0`, the
/// rasterizer quantizes back to u8; this is designed to round-trip exactly
/// for byte-valued inputs, but a tolerance of 1 absorbs any FP rounding
/// without weakening the "which mesh painted this" identification, since
/// `mesh_color_rgba8(0)` and `mesh_color_rgba8(1)` differ by far more than 2
/// in every channel).
fn pixels_matching(rgba: &[u8], expected: [u8; 4], tol: i32) -> Vec<(u32, u32)> {
    let mut hits = Vec::new();
    for y in 0..OffscreenTarget::HEIGHT {
        for x in 0..OffscreenTarget::WIDTH {
            let i = ((y * OffscreenTarget::WIDTH + x) * 4) as usize;
            let px = [rgba[i], rgba[i + 1], rgba[i + 2], rgba[i + 3]];
            let close = px.iter().zip(expected.iter()).all(|(&a, &b)| (a as i32 - b as i32).abs() <= tol);
            if close {
                hits.push((x, y));
            }
        }
    }
    hits
}

fn centroid(pixels: &[(u32, u32)]) -> (f32, f32) {
    let n = pixels.len() as f32;
    let sx: f32 = pixels.iter().map(|&(x, _)| x as f32).sum();
    let sy: f32 = pixels.iter().map(|&(_, y)| y as f32).sum();
    (sx / n, sy / n)
}

fn column_of(px: f32) -> u32 {
    ((px / (OffscreenTarget::WIDTH as f32 / GRID as f32)) as u32).min(GRID - 1)
}

#[test]
fn transform_sweep_tracks_projected_position_across_frames() {
    let ctx = test_context_indirect_first_instance();
    let device: &wgpu::Device = ctx.device().as_ref();
    let queue = ctx.queue();

    // --- Scene: one moving instance (mesh_index 0) + one static decoy
    // (mesh_index 1, parked at y=8 so its screen row never overlaps the
    // mover's y=0 row regardless of the sweep's x). ---
    let mut store = SceneGpuStore::new(&ctx, scene_cfg());
    let mut cell = SpatialCell::with_transform(64).unwrap();
    let h_move = cell.alloc(Aabb { min: [-9.0, -1.0, -11.0], max: [9.0, 1.0, -9.0] }).unwrap();
    let h_decoy = cell.alloc(Aabb { min: [-1.0, 7.0, -11.0], max: [1.0, 9.0, -9.0] }).unwrap();

    let id = store.register_cell(cell.storage(), 0).unwrap();
    let base = store.row_region_base(id);
    assert_eq!(base, 0, "single cell, class-0 region base 0");

    // --- The sweep: six frames, x in [-8, 8] at fixed z=-10, plus a
    // rotated pose (row3-style Rz(45deg)) at a FRESH x not otherwise used
    // (self-verifying guard below proves every column is distinct enough
    // to discriminate). ---
    let z = -10.0_f32;
    struct Step {
        x: f32,
        rot: Option<[[f32; 3]; 3]>,
    }
    // x values chosen to land near the CENTER of their predicted 8-pixel
    // grid cell (u = x*3.2 + 32, cell width 8px -- picking u == 4 mod 8
    // maximizes margin from any cell boundary), not at a boundary: the
    // quad's own screen footprint is a few pixels wide, so a sweep point
    // placed exactly ON a cell boundary (e.g. x=0.0, which projects to
    // u=32.0 exactly, the col3/col4 boundary) makes the observed centroid
    // vs. predicted-column comparison a coin flip on rasterization
    // rounding -- not a real bug, just a bad fixture choice. Every value
    // below sits solidly inside its column instead.
    let steps = [
        Step { x: -8.75, rot: None },  // u=4   -> column 0
        Step { x: -3.75, rot: None },  // u=20  -> column 2
        Step { x: 1.25, rot: None },   // u=36  -> column 4
        Step { x: 6.25, rot: None },   // u=52  -> column 6
        Step { x: 8.75, rot: None },   // u=60  -> column 7
        Step { x: -6.25, rot: Some(mat3_rot_z(45.0)) }, // u=12 -> column 1, rotation regression pin, draw side
    ];

    // Self-verifying guard (house law): the sweep's own predicted columns
    // must be genuinely non-uniform BEFORE trusting "footprint moves"
    // below -- a degenerate sweep that collapsed to one column would let a
    // static-image shader pass by accident.
    let predicted_columns: Vec<u32> = steps.iter().map(|s| expected_column(s.x, z)).collect();
    assert!(
        predicted_columns.iter().collect::<std::collections::HashSet<_>>().len() >= 4,
        "guard: the sweep must predict at least 4 distinct columns, got {predicted_columns:?}"
    );

    let mut frames = FrameDriver::new();
    let sim0 = frames.begin();
    let first = flatten_column_major([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], [steps[0].x, 0.0, z]);
    assert!(store.write_transform(id, cell.storage_mut(), h_move, &first, &sim0));
    assert!(store.write_transform(id, cell.storage_mut(), h_decoy, &translation([0.0, 8.0, z]), &sim0));
    assert!(store.write_instance_info(id, cell.storage_mut(), h_move, InstanceInfo { mesh_index: 0, flags: 0 }, &sim0));
    assert!(store.write_instance_info(id, cell.storage_mut(), h_decoy, InstanceInfo { mesh_index: 1, flags: 0 }, &sim0));

    // --- Harvest once (broad view; no free/realloc happens across the
    // whole sweep, so rows/generations never go stale and the SAME
    // ViewTokenBuffers upload stays valid for every frame -- the T6
    // `sim.end().end()`-without-reharvest pattern below relies on this). ---
    let harvest_phase = sim0.end().end();
    let pipeline = HarvestPipeline::new();
    let mut pad = Scratchpad::new();
    let mut staging = HarvestStaging::new();
    let view = View::Aabb(Aabb { min: [-200.0, -200.0, -200.0], max: [200.0, 200.0, 200.0] });
    let n = pipeline.harvest_cell(&cell, base, MeshClass::Traditional, &view, &mut pad, &mut staging, &harvest_phase);
    assert_eq!(n, 2, "sanity: both rows harvested");

    let boundary0 = harvest_phase.end();
    {
        let mut slots = [CellSlot { id, cell: cell.storage_mut() }];
        boundary0.run(&mut store, &mut slots);
    }

    let mut view_tokens = ViewTokenBuffers::new(&ctx, "draw-sweep-view", 0);
    view_tokens.upload(&ctx, &staging, MeshClass::Traditional);
    assert_eq!(view_tokens.count(), 2);

    // --- Geometry: ONE quad uploaded through SceneDB's real GeometryArena +
    // MeshRegistry public API (design's explicit requirement: the draw
    // genuinely pulls SceneDB-owned vertex/index data, not a test-local
    // buffer) -- registered TWICE (mesh_index 0 and 1) over the SAME
    // vertex/index range, so the two instances differ only by
    // InstanceInfo.mesh_index (and therefore only by fragment color), never
    // by geometry. ---
    let mut arena = GeometryArena::new(&ctx, 4096, 4096);
    let vbytes: &[u8] = bytemuck::cast_slice(&QUAD_VERTICES);
    let ibytes: &[u8] = bytemuck::cast_slice(&QUAD_INDICES);
    let voff = arena.upload_vertices(queue, vbytes).unwrap();
    let ioff = arena.upload_indices(queue, ibytes).unwrap();
    assert_eq!(voff, 0, "sanity: first vertex alloc in an empty arena lands at byte 0");
    assert_eq!(ioff, 0, "sanity: first index alloc in an empty arena lands at byte 0");
    let base_vertex = (voff / 12) as i32; // vec3<f32> stride
    let first_index = ioff / 4; // u32 index stride

    let mut meshes = MeshRegistry::new(&ctx, 4);
    let mesh0 = meshes
        .register(queue, mesh([0.0, 0.0, 0.0], [0.5, 0.5, 0.05], QUAD_INDICES.len() as u32, first_index, base_vertex))
        .unwrap();
    let mesh1 = meshes
        .register(queue, mesh([0.0, 0.0, 0.0], [0.5, 0.5, 0.05], QUAD_INDICES.len() as u32, first_index, base_vertex))
        .unwrap();
    assert_eq!((mesh0, mesh1), (0, 1), "sanity: mesh_index values the fixture's InstanceInfo rows used above");

    let clusters = ClusterBuffer::new(&ctx, 1);
    let meshlets = MeshletBuffer::new(&ctx, 1);
    let materials = MaterialRegistry::new(&ctx, 1);
    let scene_binding = SceneDbBinding::new(device, &store, &meshes, &clusters, &meshlets, &materials);

    // --- Cull + draw passes, built ONCE (mirrors CullPass/DrawExecutor's
    // "rebuilt at renderer construction" idiom -- nothing here is rebuilt
    // per sweep frame, only re-dispatched/re-recorded). ---
    let cull_pass = CullPass::new(device, &scene_binding.cull_layout);
    let output = CullOutputBuffers::new(device, "draw-sweep-cull-output", 8);
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

    let draw_executor = DrawExecutor::new(device, &scene_binding.cull_layout);
    let draw_output_bind_group = draw_executor.build_output_bind_group(device, &output);
    draw_executor.write_uniforms(queue, &DrawUniforms { view_proj: view_proj_90deg() });
    let target = OffscreenTarget::new(device, "draw-sweep-target");

    let move_color = mesh_color_rgba8(0);
    let decoy_color = mesh_color_rgba8(1);
    // Self-verifying guard: the two meshes' derived colors must actually
    // differ, or the pixel-classification-by-color below cannot
    // discriminate the mover from the decoy.
    assert_ne!(move_color, decoy_color, "guard: mesh_index 0/1 colors must differ");

    let mut observed_columns = Vec::new();
    /// `(was_rotated, painted_pixel_count)` per frame — feeds the
    /// rotation-reached-the-shader assertion after the loop.
    let mut mover_footprints: Vec<(bool, usize)> = Vec::new();

    for (step_i, step) in steps.iter().enumerate() {
        // --- Per-frame transform write + boundary (no re-harvest, matching
        // T6's `sim.end().end()` pattern -- rows/gens are stable). ---
        if step_i > 0 {
            let sim = frames.begin();
            let xf = match step.rot {
                None => translation([step.x, 0.0, z]),
                Some(r) => flatten_column_major(r, [step.x, 0.0, z]),
            };
            assert!(store.write_transform(id, cell.storage_mut(), h_move, &xf, &sim));
            let harvest = sim.end().end();
            let boundary = harvest.end();
            let mut slots = [CellSlot { id, cell: cell.storage_mut() }];
            boundary.run(&mut store, &mut slots);
        }

        // --- Real cull dispatch against the just-synced transform. ---
        output.clear_counters(queue);
        {
            let mut encoder = device.create_command_encoder(&Default::default());
            cull_pass.record(&mut encoder, &scene_binding.cull_bind_group, &cull_output_bind_group, view_tokens.count());
            queue.submit([encoder.finish()]);
        }
        let out_bytes = readback(&ctx, output.buffer(), output.byte_size());
        let u32_at = |off: usize| u32::from_le_bytes(out_bytes[off..off + 4].try_into().unwrap());
        let visible_count = u32_at(0);
        let stale_drops = u32_at(4);
        let oob_drops = u32_at(8);
        let frustum_drops = u32_at(12);
        // Self-verifying non-vacuity guard: both instances must be visible,
        // every frame -- a cull regression that dropped one would silently
        // turn this into a one-instance test (defeating the row-indirection
        // proof, see this file's module doc).
        assert_eq!(stale_drops, 0, "step {step_i}: no stale row in this fixture");
        assert_eq!(oob_drops, 0, "step {step_i}: no OOB row in this fixture");
        assert_eq!(frustum_drops, 0, "step {step_i}: both rows must stay inside the frustum across the whole sweep");
        assert_eq!(visible_count, 2, "step {step_i}: both mover and decoy must be visible");
        let command_count = visible_count.min(output.capacity());

        // --- Real indirect draw over the SAME cull output the dispatch
        // above just wrote. ---
        {
            let mut encoder = device.create_command_encoder(&Default::default());
            draw_executor.record(
                &mut encoder,
                target.view(),
                &scene_binding.cull_bind_group,
                &draw_output_bind_group,
                &output,
                arena.vertex_buffer(),
                arena.index_buffer(),
                command_count,
            );
            queue.submit([encoder.finish()]);
        }

        let pixels = target.read_pixels(device, queue);
        let mover_px = pixels_matching(&pixels, move_color, 1);
        let decoy_px = pixels_matching(&pixels, decoy_color, 1);

        // --- Self-verifying guards (house law): non-empty target (a shader
        // that draws nothing must fail this), AND the color-classified
        // pixel sets must be non-empty for BOTH instances (proves the
        // row->instance_info->color lookup for both concurrently-drawn
        // rows, not just "some pixel is some color"). ---
        assert!(!mover_px.is_empty(), "step {step_i}: mover must paint at least one pixel of its mesh_index-0 color");
        assert!(!decoy_px.is_empty(), "step {step_i}: decoy must paint at least one pixel of its mesh_index-1 color");

        let (cx, cy) = centroid(&mover_px);
        let (_dcx, dcy) = centroid(&decoy_px);
        // The decoy's row (y=8 world) and the mover's row (y=0 world) must
        // land in different screen rows regardless of NDC-y sign
        // convention -- proves the two instances are not simply painting
        // the same location (which would defeat the row-indirection proof
        // even if both color checks above happened to pass via overlap).
        assert!(
            (cy - dcy).abs() > (OffscreenTarget::HEIGHT as f32 / GRID as f32),
            "step {step_i}: mover row {cy} and decoy row {dcy} must be clearly separated"
        );

        let observed_col = column_of(cx);
        let expected_col = expected_column(step.x, z);
        assert_eq!(
            observed_col, expected_col,
            "step {step_i} (x={}, rot={:?}): observed column {observed_col} (centroid px={cx}) \
             must match the CPU-projected column {expected_col}",
            step.x,
            step.rot.is_some()
        );
        observed_columns.push(observed_col);
        mover_footprints.push((step.rot.is_some(), mover_px.len()));
    }

    // The rotated pose must actually CHANGE the rendered footprint (M3-β T7
    // review, defect 1). A centered square quad's centroid is invariant
    // under rotation, so the column assertions above cannot see a rotation
    // at all; without this, the rotated frame would be pure decoration.
    // Comparing painted-pixel COUNT against the unrotated frames' does see
    // it: the same quad rotated 45° covers a different pixel set.
    //
    // Scope, stated honestly: this proves a NON-IDENTITY rotation reached
    // the vertex shader's `instances[row]` read — it does NOT discriminate
    // M from Mᵀ, because R(45°) and R(-45°) map a symmetric square to the
    // same point set (reviewer verified numerically). The column-major
    // convention itself is pinned cull-side by `cull_pass.rs` row 3, which
    // uses a deliberately non-symmetric two-axis rotation for exactly that
    // reason. Pinning a *vertex-shader* transpose specifically would need
    // an off-center or asymmetric quad so rotation direction moves the
    // centroid — recorded as a known gap rather than overclaimed here.
    let unrotated: Vec<usize> =
        mover_footprints.iter().filter(|(r, _)| !r).map(|&(_, n)| n).collect();
    let rotated: Vec<usize> =
        mover_footprints.iter().filter(|(r, _)| *r).map(|&(_, n)| n).collect();
    assert!(!rotated.is_empty(), "guard: the sweep must contain a rotated pose");
    let baseline = unrotated[0];
    assert!(
        unrotated.iter().all(|&n| n == baseline),
        "guard: the unrotated poses must all paint the same footprint size (got {unrotated:?}) — \
         otherwise the rotated-vs-unrotated comparison below is meaningless"
    );
    for &n in &rotated {
        assert_ne!(
            n, baseline,
            "the rotated pose painted the same {n}-pixel footprint as the unrotated baseline — \
             the rotation did not reach the vertex shader's transform read"
        );
    }

    // --- The deliverable-gate assertion (house law): the footprint must
    // actually MOVE between at least two sweep positions -- a static image
    // would otherwise have passed every per-frame assertion above by
    // accident (each frame's centroid checked independently). ---
    let distinct: std::collections::HashSet<u32> = observed_columns.iter().copied().collect();
    assert!(
        distinct.len() >= 2,
        "guard: the sweep's observed columns must include at least 2 distinct values, got {observed_columns:?}"
    );
    assert_ne!(
        observed_columns.first(),
        observed_columns.last(),
        "guard: the FIRST and LAST sweep frame must land in different columns"
    );
}
