//! Test 5 (design §14.2, M3-b T8): the overflow clamp. Oversubscribes the
//! cull pass's command buffer -- more visible instances than the shader is
//! TOLD it may write (`CullUniforms.capacity`) -- and proves the whole
//! §14.2 safety story end to end: the atomic keeps counting true demand
//! past the bound, the shader never writes past it, the CPU-side
//! `min(count, capacity)` clamp is what makes the subsequent indirect draw
//! safe, and the silent-drop count is derivable and reportable.
//!
//! ## The logical/physical capacity split (the mechanism this test hinges on)
//!
//! [`CullOutputBuffers::new`] sizes the underlying `wgpu::Buffer` for
//! whatever `capacity` it is given -- call this the buffer's PHYSICAL
//! capacity. Separately, `CullUniforms.capacity` (written independently via
//! `CullPass::write_uniforms`) is the LOGICAL bound the shader actually
//! enforces (`wgsl.rs`'s `cull_main`: `if (out_slot < cull_uniforms.
//! capacity) { cull_output.records[out_slot] = ... }`). Nothing requires
//! these two numbers to match -- `records` is declared `array<CullRecord>`
//! (WGSL runtime-sized, its length implied by whatever buffer region is
//! bound), so the shader's only real bound is the uniform value.
//!
//! This test deliberately allocates a PHYSICAL buffer (`PHYSICAL_CAPACITY`)
//! LARGER than the LOGICAL bound it tells the shader (`LOGICAL_CAPACITY`),
//! specifically so there is real, GPU-addressable, in-bounds space past the
//! logical bound to plant canary bytes into. This is what makes the canary
//! check meaningful: a shader that ignored `cull_uniforms.capacity` and
//! instead wrote up to the buffer's actual physical extent (a real class of
//! bug -- reading the wrong bound, or comparing against `arrayLength`
//! instead of the uniform) would corrupt the canary region without ever
//! triggering a wgpu OOB validation error, because that region genuinely IS
//! inside the bound buffer. A canary planted in memory that was ALSO out of
//! the physical buffer's bounds would prove nothing new (wgpu's own
//! validation already forbids that unconditionally) -- the interesting,
//! test-worthy failure mode is exactly the in-bounds-but-past-the-logical-
//! limit one this split creates room for.
//!
//! `DEMAND` (the fixture's instance count, all mutually visible -- no
//! stale/oob/frustum-cull category exercised here, that is `cull_stale_and_
//! oob.rs`'s and `cull_pass.rs`'s job) sits strictly between `LOGICAL_
//! CAPACITY` (so the buffer really is oversubscribed) and `PHYSICAL_
//! CAPACITY` (so the canary region is never itself a plausible target row
//! -- the fixture simply does not have enough rows to fill it even if the
//! bound were ignored, keeping the corruption check unambiguous either
//! way).

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
use std::collections::HashSet;

use support::{frustum_planes_90deg, mesh, readback, scene_cfg, test_context_indirect_first_instance, translation, view_proj_90deg};

/// Local-space unit quad (two triangles), `vec3<f32>` positions -- identical
/// to `draw_transform_sweep.rs`'s fixture geometry (the trivial mesh
/// `DrawExecutor::new`'s vertex layout expects, `array_stride: 12`).
const QUAD_VERTICES: [[f32; 3]; 4] = [
    [-0.5, -0.5, 0.0],
    [0.5, -0.5, 0.0],
    [0.5, 0.5, 0.0],
    [-0.5, 0.5, 0.0],
];
const QUAD_INDICES: [u32; 6] = [0, 1, 2, 0, 2, 3];

/// The fixture's total mutually-visible instance count -- the "true demand"
/// the shader's atomic must keep counting past the logical bound.
const DEMAND: u32 = 10;
/// The bound the shader is TOLD (`CullUniforms.capacity`) -- deliberately
/// smaller than `DEMAND` (this test's entire point) and smaller than
/// `PHYSICAL_CAPACITY` (see module doc for why).
const LOGICAL_CAPACITY: u32 = 4;
/// The buffer's actual allocated size, in records -- deliberately larger
/// than both `LOGICAL_CAPACITY` (leaves real, addressable canary room) and
/// `DEMAND` (the fixture cannot accidentally fill the canary region even if
/// the logical bound were ignored).
const PHYSICAL_CAPACITY: u32 = 20;
const _SHAPE_GUARD: () = assert!(LOGICAL_CAPACITY < DEMAND && DEMAND < PHYSICAL_CAPACITY);

#[test]
fn overflow_clamp_counts_demand_writes_exactly_capacity_and_leaves_the_tail_untouched() {
    let ctx = test_context_indirect_first_instance();
    let device: &wgpu::Device = ctx.device().as_ref();
    let queue = ctx.queue();

    // --- Scene: DEMAND instances, evenly spaced in x at a fixed z=-10, all
    // comfortably inside the fovy=90/aspect=1 frustum's side planes (which
    // admit |x| far beyond 9 at this depth, per `cull_pass.rs`'s own plane
    // arithmetic) and with no near-clip (z well clear of the camera). One
    // shared mesh (mesh_index 0) for all rows -- this test is about the
    // command-buffer accounting, not per-row visual identity. ---
    let mut store = SceneGpuStore::new(&ctx, scene_cfg());
    let mut cell = SpatialCell::with_transform(64).unwrap();
    let mut handles = Vec::with_capacity(DEMAND as usize);
    for i in 0..DEMAND {
        let x = -9.0 + i as f32 * 2.0;
        let h = cell.alloc(Aabb { min: [x - 1.0, -1.0, -11.0], max: [x + 1.0, 1.0, -9.0] }).unwrap();
        handles.push(h);
    }

    let id = store.register_cell(cell.storage(), 0).unwrap();
    let base = store.row_region_base(id);
    assert_eq!(base, 0, "single cell, class-0 region base 0");

    let mut frames = FrameDriver::new();
    let sim = frames.begin();
    for (i, &h) in handles.iter().enumerate() {
        let x = -9.0 + i as f32 * 2.0;
        assert!(store.write_transform(id, cell.storage_mut(), h, &translation([x, 0.0, -10.0]), &sim));
        assert!(store.write_instance_info(id, cell.storage_mut(), h, InstanceInfo { mesh_index: 0, flags: 0 }, &sim));
    }

    let harvest_phase = sim.end().end();
    let pipeline = HarvestPipeline::new();
    let mut pad = Scratchpad::new();
    let mut staging = HarvestStaging::new();
    let view = View::Aabb(Aabb { min: [-200.0, -200.0, -200.0], max: [200.0, 200.0, 200.0] });
    let n = pipeline.harvest_cell(&cell, base, MeshClass::Traditional, &view, &mut pad, &mut staging, &harvest_phase);
    assert_eq!(n, DEMAND, "sanity: the broad harvest view catches every row");

    let boundary = harvest_phase.end();
    {
        let mut slots = [CellSlot { id, cell: cell.storage_mut() }];
        boundary.run(&mut store, &mut slots);
    }

    let mut view_tokens = ViewTokenBuffers::new(&ctx, "overflow-clamp-view", 0);
    view_tokens.upload(&ctx, &staging, MeshClass::Traditional);
    assert_eq!(view_tokens.count(), DEMAND);

    // --- Geometry + mesh registration (real SceneDB residency, mirroring
    // `draw_transform_sweep.rs`). ---
    let mut arena = GeometryArena::new(&ctx, 4096, 4096);
    let vbytes: &[u8] = bytemuck::cast_slice(&QUAD_VERTICES);
    let ibytes: &[u8] = bytemuck::cast_slice(&QUAD_INDICES);
    let voff = arena.upload_vertices(queue, vbytes).unwrap();
    let ioff = arena.upload_indices(queue, ibytes).unwrap();
    let base_vertex = (voff / 12) as i32;
    let first_index = ioff / 4;

    let mut meshes = MeshRegistry::new(&ctx, 2);
    let mesh0 = meshes
        .register(queue, mesh([0.0, 0.0, 0.0], [0.5, 0.5, 0.5], QUAD_INDICES.len() as u32, first_index, base_vertex))
        .unwrap();
    assert_eq!(mesh0, 0, "sanity: mesh_index 0, matching the fixture's InstanceInfo rows above");

    let clusters = ClusterBuffer::new(&ctx, 1);
    let meshlets = MeshletBuffer::new(&ctx, 1);
    let materials = MaterialRegistry::new(&ctx, 1);
    let scene_binding = SceneDbBinding::new(device, &store, &meshes, &clusters, &meshlets, &materials);

    // --- The cull pass, output allocated at PHYSICAL_CAPACITY (see module
    // doc). ---
    let cull_pass = CullPass::new(device, &scene_binding.cull_layout);
    let output = CullOutputBuffers::new(device, "overflow-clamp-output", PHYSICAL_CAPACITY);
    output.clear_counters(queue);

    // --- Plant the canary: a distinctive, non-zero, non-plausible-draw-
    // command 32-bit pattern (0xCAFEBABE repeated) across the ENTIRE tail
    // region -- every record slot from LOGICAL_CAPACITY..PHYSICAL_CAPACITY
    // -- BEFORE dispatch. `clear_counters` above only zeroes the 16-byte
    // atomics header (offsets 0..16), a disjoint byte range from this
    // write, so ordering between the two calls does not matter. ---
    let tail_offset = CullOutputBuffers::HEADER_BYTES + LOGICAL_CAPACITY as u64 * CullOutputBuffers::RECORD_BYTES;
    let tail_len = (PHYSICAL_CAPACITY - LOGICAL_CAPACITY) as u64 * CullOutputBuffers::RECORD_BYTES;
    const CANARY_PATTERN: u32 = 0xCAFE_BABE;
    let canary_bytes: Vec<u8> =
        std::iter::repeat_n(CANARY_PATTERN.to_le_bytes(), (tail_len / 4) as usize).flatten().collect();
    assert_eq!(canary_bytes.len() as u64, tail_len, "sanity: canary fills the whole tail exactly");
    queue.write_buffer(output.buffer(), tail_offset, &canary_bytes);

    let output_bind_group = cull_pass.build_output_bind_group(device, &view_tokens, &output);
    let uniforms = CullUniforms {
        view_proj: view_proj_90deg(),
        planes: frustum_planes_90deg(),
        count: view_tokens.count(),
        mesh_count: meshes.len(),
        // THE decoupling this test exists to exercise: the shader is told a
        // bound smaller than both the fixture's demand AND the buffer's
        // real physical size.
        capacity: LOGICAL_CAPACITY,
        reserved: 0,
    };
    cull_pass.write_uniforms(queue, &uniforms);

    {
        let mut encoder = device.create_command_encoder(&Default::default());
        cull_pass.record(&mut encoder, &scene_binding.cull_bind_group, &output_bind_group, view_tokens.count());
        queue.submit([encoder.finish()]);
    }

    let out_bytes = readback(&ctx, output.buffer(), output.byte_size());
    let u32_at = |off: usize| u32::from_le_bytes(out_bytes[off..off + 4].try_into().unwrap());
    let i32_at = |off: usize| i32::from_le_bytes(out_bytes[off..off + 4].try_into().unwrap());

    let visible_count = u32_at(0);
    let stale_drops = u32_at(4);
    let oob_drops = u32_at(8);
    let frustum_drops = u32_at(12);

    // --- Self-verifying guards (house law): this fixture is designed to
    // exercise ONLY the overflow category -- everything else must read
    // exactly zero, or the demand/capacity arithmetic below is not testing
    // what it claims to. ---
    assert_eq!(stale_drops, 0, "fixture has no stale-generation row");
    assert_eq!(oob_drops, 0, "fixture has no out-of-range mesh_index row");
    assert_eq!(frustum_drops, 0, "fixture has no frustum-culled row -- every instance sits well inside the frustum");

    // (1) The atomic counter reads back GREATER than capacity: the shader
    // keeps counting true demand even past the bound.
    assert_eq!(visible_count, DEMAND, "the atomic must count the FULL true demand, not just what fit");
    assert!(
        visible_count > LOGICAL_CAPACITY,
        "guard: this fixture must genuinely oversubscribe the command buffer (visible_count={visible_count} \
         must exceed LOGICAL_CAPACITY={LOGICAL_CAPACITY})"
    );

    // (2) Exactly `capacity` command records were written -- collect the
    // LOGICAL_CAPACITY slots and verify each is a real, distinct, valid
    // draw command (S14.1 shape) rather than assuming which of the 10 rows
    // won the atomic race (GPU thread scheduling order is not guaranteed).
    let mut written_rows: HashSet<u32> = HashSet::new();
    for slot in 0..LOGICAL_CAPACITY {
        let off = CullOutputBuffers::HEADER_BYTES as usize + slot as usize * CullOutputBuffers::RECORD_BYTES as usize;
        let index_count = u32_at(off);
        let instance_count = u32_at(off + 4);
        let rec_first_index = u32_at(off + 8);
        let rec_base_vertex = i32_at(off + 12);
        let rec_first_instance = u32_at(off + 16);
        let row = u32_at(off + 20);
        assert_eq!(instance_count, 1, "slot {slot}: S14.1 instance_count always 1");
        assert_eq!(rec_first_instance, slot, "slot {slot}: S14.1 first_instance == command slot (C5)");
        assert_eq!(index_count, QUAD_INDICES.len() as u32, "slot {slot}: valid command, mesh0's index_count");
        assert_eq!(rec_first_index, first_index, "slot {slot}: valid command, mesh0's index_offset");
        assert_eq!(rec_base_vertex, base_vertex, "slot {slot}: valid command, mesh0's base_vertex");
        assert!(row < DEMAND, "slot {slot}: row {row} must be one of the fixture's real rows (0..{DEMAND})");
        assert!(written_rows.insert(row), "slot {slot}: duplicate row {row} across command slots -- lost a slot");
    }
    assert_eq!(
        written_rows.len() as u32,
        LOGICAL_CAPACITY,
        "exactly LOGICAL_CAPACITY distinct rows must have received a command"
    );

    // (3) The region beyond capacity is untouched: the canary planted above
    // must survive byte-for-byte. A shader that wrote past the logical
    // bound (e.g. comparing against the buffer's physical extent instead of
    // `cull_uniforms.capacity`) would corrupt this.
    let tail_after = &out_bytes[tail_offset as usize..(tail_offset + tail_len) as usize];
    assert_eq!(
        tail_after,
        canary_bytes.as_slice(),
        "canary corrupted -- the shader wrote past the LOGICAL capacity bound into buffer space \
         only the PHYSICAL allocation made addressable"
    );

    // (4) The CPU-side clamp path: `min(count, capacity)` against the SAME
    // logical bound the shader enforced (NOT `output.capacity()`, which is
    // the physical size -- a driver that clamped against the physical size
    // here would issue draw calls reading slots the shader never wrote).
    let command_count = visible_count.min(LOGICAL_CAPACITY);
    assert_eq!(command_count, LOGICAL_CAPACITY, "the clamp must yield a usable, in-bounds draw count");

    // (5) Report the silent-drop count.
    let silent_drops = visible_count - command_count;
    assert_eq!(silent_drops, DEMAND - LOGICAL_CAPACITY, "silent_drops == true demand minus what the clamp kept");
    println!(
        "Test 5 overflow clamp: demand={DEMAND} logical_capacity={LOGICAL_CAPACITY} \
         physical_capacity={PHYSICAL_CAPACITY} silent_drops={silent_drops}"
    );

    // --- Then DRAW with the clamped count: this is what actually proves
    // the clamp makes overflow SAFE, not merely counted. Wrap the encode +
    // submit in a validation error scope -- a real wgpu OOB/validation
    // failure would otherwise either panic (uncaptured-error default
    // handler) or, worse, pass silently on a backend that tolerates it; the
    // error scope makes the absence of an error an explicit, checked
    // assertion instead of "the test process didn't crash". ---
    let draw_executor = DrawExecutor::new(device, &scene_binding.cull_layout);
    let draw_output_bind_group = draw_executor.build_output_bind_group(device, &output);
    draw_executor.write_uniforms(queue, &DrawUniforms { view_proj: view_proj_90deg() });
    let target = OffscreenTarget::new(device, "overflow-clamp-target");

    let error_scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
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
    device.poll(wgpu::PollType::wait_indefinitely()).expect("poll");
    let validation_error = pollster::block_on(error_scope.pop());
    assert!(
        validation_error.is_none(),
        "the clamped draw ({command_count} commands, out of a {PHYSICAL_CAPACITY}-record physical buffer) \
         must not raise a validation error -- that is what 'the clamp makes overflow safe' means \
         concretely; got: {validation_error:?}"
    );

    // Self-verifying non-vacuity guard (house law): the clamped draw must
    // actually paint pixels -- an empty/no-op frame would satisfy the
    // no-validation-error assertion above vacuously.
    let pixels = target.read_pixels(device, queue);
    let mesh0_color = mesh_color_rgba8(0);
    let painted = pixels
        .chunks_exact(4)
        .filter(|px| px[0] == mesh0_color[0] && px[1] == mesh0_color[1] && px[2] == mesh0_color[2])
        .count();
    assert!(painted > 0, "guard: the clamped draw must paint at least one pixel of mesh0's color");
}
