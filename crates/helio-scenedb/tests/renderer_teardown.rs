//! Test 13 (design §5's exact assertion set, CONTRACTS.md C0's binding
//! acceptance criterion, M3-b T8 -- THE MILESTONE GATE): stateless renderer
//! teardown. This is the test that turns the Ownership Law from a sentence
//! in CONTRACTS.md into an executable assertion: build a scene through
//! SceneDB, construct renderer-side objects (`SceneDbBinding` + `CullPass` +
//! `DrawExecutor`, plus the Helio-owned derived scratch that goes with them
//! -- `CullOutputBuffers`/`OffscreenTarget`) as instance A, render N frames,
//! **drop A entirely**, construct a fresh instance B against the SAME
//! `SceneGpuStore` and asset registries, render N more frames, and prove:
//! zero scene-data re-upload happened anywhere in that whole window, every
//! scene SSBO and the device survived, and the two renderer instances
//! produce byte-identical output despite being genuinely different GPU
//! objects.
//!
//! ## Assertion-by-assertion mapping to design §5's exact list
//!
//! | # | Design §5 wording | This test's mechanism |
//! |---|---|---|
//! | a | Σ`SyncStats.bytes` == 0 across the window | every frame in the window runs a REAL `BoundaryPhase::run` (retire/compact/sync) with NO writes queued; each call's `SyncStats.bytes` is summed and the total asserted zero at the end -- not "we skipped calling it" |
//! | b | Δ`generation_write_count` == 0 | captured before the window, asserted equal after |
//! | c | ALL six asset-store upload counters unchanged | `MeshRegistry`/`ClusterBuffer`/`MeshletBuffer`/`MaterialRegistry`/`TextureStore`/`GeometryArena`.`upload_count()`, each captured before the window and asserted equal at 3 checkpoints (after A's frames, after A drops + B constructs, after B's frames) |
//! | d | streaming frozen (or counted separately) | `StreamingGrid::write_cell_metadata` is never called anywhere in this test -- frozen, not counted; see the module-level note below |
//! | e | G-buffer hash byte-identical, jitter/TAA caveat stated | A's final frame and B's converged frame are compared BYTE-FOR-BYTE (strictly stronger than a hash: no collision risk) with an FNV-1a hash also computed and printed for the report; this harness has no TAA/jitter/Halton-phase state at all, so the comparison is trivially stable -- stated honestly below, not oversold as jitter-proof |
//! | f | device + every scene SSBO alive, buffer IDs unchanged | `wgpu::Buffer`/`wgpu::Device` implement `PartialEq`/`Clone` in wgpu 30 proxied to the underlying resource id (confirmed: no `.global_id()` in this wgpu version, `cmp::impl_eq_ord_hash_proxy!` on `Buffer`/`Device` is the sanctioned mechanism) -- every SceneDB-owned buffer accessor's handle is cloned before the window and compared equal after |
//!
//! ## The `write_cell_metadata` decision (honesty note, design §5's stated trap)
//!
//! `StreamingGrid::write_cell_metadata` is, by design, an unconditional full
//! rewrite on every call (`grid.rs`'s own doc: "Simple full rewrite of every
//! materialized entry's 8 bytes on every call") with its own upload counter
//! that increments unconditionally, no dirty-gating. This test does not
//! construct a `StreamingGrid` at all and never calls it -- streaming is
//! FROZEN for the window, the simpler of the two options design §5 offers
//! ("freeze streaming for the window, or count it separately"). This is a
//! sound scope choice for M3 (design's own carve-out: "M3 scopes to
//! non-voxel, non-streaming scenes" is the spirit of the gate at this
//! milestone), not a claim that streaming-driven cell-metadata churn is
//! compatible with the zero-reupload story -- that is out of scope here and
//! recorded as such.
//!
//! ## Scope note: which objects are "A"/"B" vs. which persist
//!
//! Per CONTRACTS C0, `SceneGpuStore`, `MeshRegistry`, `ClusterBuffer`,
//! `MeshletBuffer`, `MaterialRegistry`, `TextureStore`, `GeometryArena`, and
//! `ViewTokenBuffers` are all SceneDB-owned (scene object data / its
//! per-view harvest products) -- they are built ONCE, before the window,
//! and never touched again until the test's own final assertions. Only the
//! Helio-owned derived data -- `SceneDbBinding` (the read-only bind groups
//! over those SceneDB buffers), `CullPass`/`DrawExecutor` (pipelines),
//! `CullOutputBuffers` (draw-command scratch), and `OffscreenTarget`
//! (framebuffer) -- is torn down and rebuilt across the A/B boundary. This
//! matches the design's own framing precisely (`lib.rs`'s module doc: "Helio
//! owns no scene state... pipelines, shaders, Hi-Z, framebuffers,
//! draw-command and payload scratch").

#[path = "support/mod.rs"]
mod support;

use helio_scenedb::cull::{CullOutputBuffers, CullPass, CullUniforms};
use helio_scenedb::draw::{mesh_color_rgba8, DrawExecutor, DrawUniforms, OffscreenTarget};
use helio_scenedb::SceneDbBinding;
use pulsar_scenedb::gpu::{
    CellSlot, ClusterBuffer, EngineGpuContext, FrameDriver, GeometryArena, HarvestPipeline, HarvestStaging,
    MaterialRegistry, MaterialRow, MeshClass, MeshRegistry, MeshletBuffer, MeshletEntry, SceneGpuStore, TextureStore,
    View, ViewTokenBuffers,
};
use pulsar_scenedb::{Aabb, InstanceInfo, Scratchpad, SpatialCell};

use support::{frustum_planes_90deg, mesh, readback, scene_cfg, test_context_indirect_first_instance, translation, view_proj_90deg};

/// Local-space unit quad -- identical fixture geometry to `draw_transform_
/// sweep.rs`/`overflow_clamp.rs`.
const QUAD_VERTICES: [[f32; 3]; 4] = [
    [-0.5, -0.5, 0.0],
    [0.5, -0.5, 0.0],
    [0.5, 0.5, 0.0],
    [-0.5, 0.5, 0.0],
];
const QUAD_INDICES: [u32; 6] = [0, 1, 2, 0, 2, 3];

/// Frames rendered by EACH of executor A and executor B (design: "N ≥ 3").
const N_FRAMES: u32 = 4;
/// Cull-output physical capacity -- generous headroom over this fixture's 2
/// instances; overflow is `overflow_clamp.rs`'s job, not this test's.
const CULL_CAPACITY: u32 = 8;

/// Non-cryptographic FNV-1a over the raw pixel bytes -- used ONLY to print a
/// compact, comparable value for the report table. The actual pass/fail
/// assertion below compares the full byte buffers directly (strictly
/// stronger than any hash: zero collision risk), so this hash is reporting
/// sugar, not the proof.
fn fnv1a_hash(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01B3);
    }
    h
}

/// One frame's worth of cull-dispatch + indirect-draw + readback, run
/// against whichever renderer instance (A or B) owns the passed-in
/// pipelines/bind-groups/scratch buffers. Returns the cull telemetry
/// (visible/stale/oob/frustum) and the rendered target's raw RGBA8 bytes.
/// Shared by both the A-loop and the B-loop below -- the SAME function,
/// proving neither loop gets any special-cased behavior.
#[allow(clippy::too_many_arguments)]
fn render_one_frame(
    ctx: &EngineGpuContext,
    cull: &CullPass,
    cull_bind_group: &wgpu::BindGroup,
    cull_output_bg: &wgpu::BindGroup,
    output: &CullOutputBuffers,
    draw: &DrawExecutor,
    draw_output_bg: &wgpu::BindGroup,
    target: &OffscreenTarget,
    vertex_buffer: &wgpu::Buffer,
    index_buffer: &wgpu::Buffer,
    dispatch_count: u32,
) -> (u32, u32, u32, u32, Vec<u8>) {
    let device: &wgpu::Device = ctx.device().as_ref();
    let queue: &wgpu::Queue = ctx.queue().as_ref();

    output.clear_counters(queue);
    {
        let mut encoder = device.create_command_encoder(&Default::default());
        cull.record(&mut encoder, cull_bind_group, cull_output_bg, dispatch_count);
        queue.submit([encoder.finish()]);
    }
    let out_bytes = readback(ctx, output.buffer(), output.byte_size());
    let u32_at = |off: usize| u32::from_le_bytes(out_bytes[off..off + 4].try_into().unwrap());
    let visible = u32_at(0);
    let stale = u32_at(4);
    let oob = u32_at(8);
    let frustum = u32_at(12);
    let command_count = visible.min(output.capacity());

    {
        let mut encoder = device.create_command_encoder(&Default::default());
        draw.record(&mut encoder, target.view(), cull_bind_group, draw_output_bg, output, vertex_buffer, index_buffer, command_count);
        queue.submit([encoder.finish()]);
    }
    let pixels = target.read_pixels(device, queue);
    (visible, stale, oob, frustum, pixels)
}

#[test]
fn stateless_renderer_teardown_zero_reupload_identical_hash_ssbos_alive() {
    let ctx = test_context_indirect_first_instance();
    let device: &wgpu::Device = ctx.device().as_ref();
    let queue: &wgpu::Queue = ctx.queue().as_ref();

    // =====================================================================
    // Scene build (BEFORE the window): two static instances, different
    // meshes/colors so the "did it really render" guard can look for both.
    // Every SceneDB-owned store used below is constructed exactly once and
    // never rebuilt -- this whole block is the "one real upload" the
    // post-window zero-reupload assertions get to be meaningful against.
    // =====================================================================
    let mut store = SceneGpuStore::new(&ctx, scene_cfg());
    let mut cell = SpatialCell::with_transform(64).unwrap();
    let h_a = cell.alloc(Aabb { min: [-1.0, -1.0, -11.0], max: [1.0, 1.0, -9.0] }).unwrap();
    let h_b = cell.alloc(Aabb { min: [2.0, -1.0, -11.0], max: [4.0, 1.0, -9.0] }).unwrap();

    let id = store.register_cell(cell.storage(), 0).unwrap();
    let base = store.row_region_base(id);
    assert_eq!(base, 0, "single cell, class-0 region base 0");

    let mut frames = FrameDriver::new();
    let sim0 = frames.begin();
    assert!(store.write_transform(id, cell.storage_mut(), h_a, &translation([0.0, 0.0, -10.0]), &sim0));
    assert!(store.write_transform(id, cell.storage_mut(), h_b, &translation([3.0, 0.0, -10.0]), &sim0));
    assert!(store.write_instance_info(id, cell.storage_mut(), h_a, InstanceInfo { mesh_index: 0, flags: 0 }, &sim0));
    assert!(store.write_instance_info(id, cell.storage_mut(), h_b, InstanceInfo { mesh_index: 1, flags: 0 }, &sim0));

    let harvest_phase = sim0.end().end();
    let pipeline = HarvestPipeline::new();
    let mut pad = Scratchpad::new();
    let mut staging = HarvestStaging::new();
    let view = View::Aabb(Aabb { min: [-200.0, -200.0, -200.0], max: [200.0, 200.0, 200.0] });
    let n = pipeline.harvest_cell(&cell, base, MeshClass::Traditional, &view, &mut pad, &mut staging, &harvest_phase);
    assert_eq!(n, 2, "sanity: both rows harvested");

    let boundary0 = harvest_phase.end();
    let initial_sync = {
        let mut slots = [CellSlot { id, cell: cell.storage_mut() }];
        boundary0.run(&mut store, &mut slots)
    };
    assert!(initial_sync.bytes > 0, "sanity: the one real scene upload must actually move bytes");

    // --- A THIRD, throwaway instance: allocated as the cell's LAST row,
    // freed via the deferred-retire path, and drained by a second
    // pre-window boundary. This is the ONLY way `generation_write_count`
    // becomes genuinely non-zero: `register_cell`'s bulk generation upload
    // (the write at `scene_store.rs`'s `rebuild_region` call) seeds
    // `gen_shadow` from the freshly-registered rows' OWN generations, so
    // `write_transform`'s shadow-gated stamp is a no-op for h_a/h_b (their
    // very first write already agrees with what registration already
    // shadowed) -- confirmed empirically: without this block, `generation_
    // write_count()` reads 0 even after the real writes above, which would
    // make the "counters must be non-zero before the window" house-law
    // guard below vacuous for this one counter. Retirement is documented as
    // the OTHER trigger (`write_transform`'s own doc: "a generation reaches
    // VRAM on the first write after alloc and on retirement"), so freeing a
    // fresh handle exercises it directly. Allocated as the LAST row and
    // freed before ever being harvested/uploaded -- swap-and-pop
    // compaction of the last row is a no-op, so this cannot renumber h_a's
    // or h_b's rows out from under the harvest/ViewTokenBuffers state
    // captured below. ---
    let sim_throwaway = frames.begin();
    let h_throwaway = cell.alloc(Aabb { min: [50.0, 50.0, 50.0], max: [51.0, 51.0, 51.0] }).unwrap();
    let throwaway_serial = store.tracker().next_serial();
    assert!(store.free_deferred(id, cell.storage_mut(), h_throwaway, throwaway_serial, &sim_throwaway));
    store.tracker().force_complete(throwaway_serial);
    let boundary_throwaway = sim_throwaway.end().end().end();
    {
        let mut slots = [CellSlot { id, cell: cell.storage_mut() }];
        boundary_throwaway.run(&mut store, &mut slots);
    }
    assert!(
        store.generation_write_count() > 0,
        "sanity: retiring the throwaway handle must stamp its bumped generation to VRAM"
    );

    let mut view_tokens = ViewTokenBuffers::new(&ctx, "teardown-view", 0);
    view_tokens.upload(&ctx, &staging, MeshClass::Traditional);
    assert_eq!(view_tokens.count(), 2);

    let mut arena = GeometryArena::new(&ctx, 4096, 4096);
    let vbytes: &[u8] = bytemuck::cast_slice(&QUAD_VERTICES);
    let ibytes: &[u8] = bytemuck::cast_slice(&QUAD_INDICES);
    let voff = arena.upload_vertices(queue, vbytes).unwrap();
    let ioff = arena.upload_indices(queue, ibytes).unwrap();
    let base_vertex = (voff / 12) as i32;
    let first_index = ioff / 4;

    let mut meshes = MeshRegistry::new(&ctx, 4);
    let mesh0 = meshes
        .register(queue, mesh([0.0, 0.0, 0.0], [0.5, 0.5, 0.05], QUAD_INDICES.len() as u32, first_index, base_vertex))
        .unwrap();
    let mesh1 = meshes
        .register(queue, mesh([0.0, 0.0, 0.0], [0.5, 0.5, 0.05], QUAD_INDICES.len() as u32, first_index, base_vertex))
        .unwrap();
    assert_eq!((mesh0, mesh1), (0, 1), "sanity: mesh_index values match the InstanceInfo rows above");

    // ClusterBuffer's constructor already performs one real upload (the
    // reserved sentinel node, `assets.rs`'s own doc) -- no `append` call
    // needed for its baseline to be non-zero.
    let clusters = ClusterBuffer::new(&ctx, 2);

    // MeshletBuffer's constructor does NOT write anything (module doc: entry
    // 0 is an ordinary allocatable record, no sentinel) -- append exactly
    // one valid entry so its baseline is genuinely non-zero, matching the
    // "counters must be non-zero BEFORE the window" house-law guard below.
    let mut meshlets = MeshletBuffer::new(&ctx, 4);
    meshlets
        .append(
            queue,
            &[MeshletEntry {
                sphere_x: 0.0,
                sphere_y: 0.0,
                sphere_z: 0.0,
                sphere_radius: 1.0,
                cone_packed: 0,
                data_offset: 0,
                counts_packed: (1u32 << 8) | 1u32, // vertex_count=1, triangle_count=1
                reserved: 0,
            }],
        )
        .unwrap();

    let mut materials = MaterialRegistry::new(&ctx, 2);
    materials
        .register(
            queue,
            MaterialRow {
                base_color: 0xFFFF_FFFF,
                metallic: 0.5,
                roughness: 0.5,
                normal_scale: 1.0,
                emissive_r: 0.0,
                emissive_g: 0.0,
                emissive_b: 0.0,
                emissive_intensity: 0.0,
                tex_albedo: 0xFFFF_FFFF,
                tex_normal: 0xFFFF_FFFF,
                tex_metallic_roughness: 0xFFFF_FFFF,
                tex_emissive: 0xFFFF_FFFF,
                radiant_graph_index: 0xFFFF_FFFF,
                flags: 0,
                alpha_cutoff: 1.0,
                reserved: 0,
            },
        )
        .unwrap();

    let mut textures = TextureStore::new(4);
    textures
        .register(
            device,
            queue,
            &wgpu::TextureDescriptor {
                label: Some("teardown-texture"),
                size: wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8Unorm,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            },
            &[255u8, 255, 255, 255],
        )
        .unwrap();

    // --- House-law guard: every counter this test claims stays flat across
    // the window must actually be NON-ZERO right now -- otherwise "unchanged"
    // would be a vacuous truth (a counter that starts and stays at 0 proves
    // nothing about re-upload avoidance). ---
    let gen_writes_baseline = store.generation_write_count();
    let mesh_uploads_baseline = meshes.upload_count();
    let cluster_uploads_baseline = clusters.upload_count();
    let meshlet_uploads_baseline = meshlets.upload_count();
    let material_uploads_baseline = materials.upload_count();
    let texture_uploads_baseline = textures.upload_count();
    let geometry_uploads_baseline = arena.upload_count();
    assert!(gen_writes_baseline > 0, "guard: generation_write_count must be non-zero before the window");
    assert!(mesh_uploads_baseline > 0, "guard: MeshRegistry.upload_count must be non-zero before the window");
    assert!(cluster_uploads_baseline > 0, "guard: ClusterBuffer.upload_count must be non-zero before the window");
    assert!(meshlet_uploads_baseline > 0, "guard: MeshletBuffer.upload_count must be non-zero before the window");
    assert!(material_uploads_baseline > 0, "guard: MaterialRegistry.upload_count must be non-zero before the window");
    assert!(texture_uploads_baseline > 0, "guard: TextureStore.upload_count must be non-zero before the window");
    assert!(geometry_uploads_baseline > 0, "guard: GeometryArena.upload_count must be non-zero before the window");

    // Scene SSBO + device identity, captured before the window (assertion f).
    let device_ptr_before = std::sync::Arc::as_ptr(ctx.device());
    let transform_buf_before = store.transform_buffer().clone();
    let instance_info_buf_before = store.instance_info_buffer().clone();
    let slot_mirror_buf_before = store.slot_mirror_buffer().clone();
    let generation_buf_before = store.generation_buffer().clone();
    let cell_meta_buf_before = store.cell_metadata_buffer().clone();
    let mesh_buf_before = meshes.buffer().clone();
    let cluster_buf_before = clusters.buffer().clone();
    let meshlet_buf_before = meshlets.buffer().clone();
    let material_buf_before = materials.buffer().clone();
    let vertex_buf_before = arena.vertex_buffer().clone();
    let index_buf_before = arena.index_buffer().clone();

    let mesh0_color = mesh_color_rgba8(0);
    let mesh1_color = mesh_color_rgba8(1);
    assert_ne!(mesh0_color, mesh1_color, "guard: the two meshes' derived colors must differ");

    // =====================================================================
    // THE WINDOW starts here: everything from this point until B's last
    // frame must move zero scene-data bytes.
    // =====================================================================
    let mut total_sync_bytes: u64 = 0;

    // --- Construct executor A: SceneDbBinding + CullPass + DrawExecutor,
    // plus the Helio-owned derived scratch that goes with a renderer
    // instance (CullOutputBuffers, OffscreenTarget). ---
    let a_binding = SceneDbBinding::new(device, &store, &meshes, &clusters, &meshlets, &materials);
    let a_cull = CullPass::new(device, &a_binding.cull_layout);
    let a_output = CullOutputBuffers::new(device, "teardown-a-output", CULL_CAPACITY);
    let a_cull_bg = a_cull.build_output_bind_group(device, &view_tokens, &a_output);
    a_cull.write_uniforms(
        queue,
        &CullUniforms {
            view_proj: view_proj_90deg(),
            planes: frustum_planes_90deg(),
            count: view_tokens.count(),
            mesh_count: meshes.len(),
            capacity: a_output.capacity(),
            reserved: 0,
        },
    );
    let a_draw = DrawExecutor::new(device, &a_binding.cull_layout);
    let a_draw_bg = a_draw.build_output_bind_group(device, &a_output);
    a_draw.write_uniforms(queue, &DrawUniforms { view_proj: view_proj_90deg() });
    let a_target = OffscreenTarget::new(device, "teardown-a-target");

    // Identity snapshots of A's Helio-owned derived objects -- held ONLY for
    // the "A and B are genuinely distinct" guard below, captured before A is
    // dropped (cloning a wgpu handle bumps its own internal refcount but
    // does not keep the SCENE buffers alive -- those were never part of A).
    let a_cull_bind_group_id = a_binding.cull_bind_group.clone();
    let a_draw_bind_group_id = a_binding.draw_bind_group.clone();
    let a_output_buffer_id = a_output.buffer().clone();
    let a_target_view_id = a_target.view().clone();

    let mut a_final_pixels: Vec<u8> = Vec::new();
    for frame_i in 0..N_FRAMES {
        // Zero-write frame boundary: a REAL BoundaryPhase::run, no
        // write_transform/write_instance_info/free_deferred queued at all
        // this frame -- proves the mechanism itself is a no-op when nothing
        // changed, rather than the test simply never calling it.
        let sim = frames.begin();
        let harvest = sim.end().end();
        let boundary = harvest.end();
        let stats = {
            let mut slots = [CellSlot { id, cell: cell.storage_mut() }];
            boundary.run(&mut store, &mut slots)
        };
        assert_eq!(stats.bytes, 0, "A frame {frame_i}: zero-write boundary must sync zero bytes");
        assert_eq!(stats.ranges, 0, "A frame {frame_i}: zero-write boundary must touch zero ranges");
        total_sync_bytes += stats.bytes;

        let (visible, stale, oob, frustum, pixels) = render_one_frame(
            &ctx,
            &a_cull,
            &a_binding.cull_bind_group,
            &a_cull_bg,
            &a_output,
            &a_draw,
            &a_draw_bg,
            &a_target,
            arena.vertex_buffer(),
            arena.index_buffer(),
            view_tokens.count(),
        );
        assert_eq!(stale, 0, "A frame {frame_i}: no stale row in this fixture");
        assert_eq!(oob, 0, "A frame {frame_i}: no OOB row in this fixture");
        assert_eq!(frustum, 0, "A frame {frame_i}: both rows stay inside the frustum");
        assert_eq!(visible, 2, "A frame {frame_i}: both instances visible every frame");
        a_final_pixels = pixels;
    }

    // --- Checkpoint 1: after A's N frames, before dropping anything. ---
    assert_eq!(store.generation_write_count(), gen_writes_baseline, "after A's frames: generation_write_count moved");
    assert_eq!(meshes.upload_count(), mesh_uploads_baseline, "after A's frames: MeshRegistry re-uploaded");
    assert_eq!(clusters.upload_count(), cluster_uploads_baseline, "after A's frames: ClusterBuffer re-uploaded");
    assert_eq!(meshlets.upload_count(), meshlet_uploads_baseline, "after A's frames: MeshletBuffer re-uploaded");
    assert_eq!(materials.upload_count(), material_uploads_baseline, "after A's frames: MaterialRegistry re-uploaded");
    assert_eq!(textures.upload_count(), texture_uploads_baseline, "after A's frames: TextureStore re-uploaded");
    assert_eq!(arena.upload_count(), geometry_uploads_baseline, "after A's frames: GeometryArena re-uploaded");

    // =====================================================================
    // DROP A ENTIRELY -- explicit, one object at a time, so the compiler
    // (not just a scope-exit convention) is the proof: any later reference
    // to these bindings is a compile error, not just "went out of scope".
    // =====================================================================
    drop(a_cull_bg);
    drop(a_draw_bg);
    drop(a_output);
    drop(a_target);
    drop(a_cull);
    drop(a_draw);
    drop(a_binding);

    // --- Construct executor B: fresh SceneDbBinding + CullPass +
    // DrawExecutor + derived scratch, against the SAME store/meshes/
    // clusters/meshlets/materials/view_tokens/arena -- none of which were
    // touched by the drop above. ---
    let b_binding = SceneDbBinding::new(device, &store, &meshes, &clusters, &meshlets, &materials);
    let b_cull = CullPass::new(device, &b_binding.cull_layout);
    let b_output = CullOutputBuffers::new(device, "teardown-b-output", CULL_CAPACITY);
    let b_cull_bg = b_cull.build_output_bind_group(device, &view_tokens, &b_output);
    b_cull.write_uniforms(
        queue,
        &CullUniforms {
            view_proj: view_proj_90deg(),
            planes: frustum_planes_90deg(),
            count: view_tokens.count(),
            mesh_count: meshes.len(),
            capacity: b_output.capacity(),
            reserved: 0,
        },
    );
    let b_draw = DrawExecutor::new(device, &b_binding.cull_layout);
    let b_draw_bg = b_draw.build_output_bind_group(device, &b_output);
    b_draw.write_uniforms(queue, &DrawUniforms { view_proj: view_proj_90deg() });
    let b_target = OffscreenTarget::new(device, "teardown-b-target");

    // --- Guard (e): A and B must be genuinely distinct objects. Every
    // Helio-owned derived object B just built is a FRESH wgpu allocation
    // (SceneDbBinding::new/CullOutputBuffers::new/OffscreenTarget::new never
    // cache or reuse) -- wgpu::BindGroup/Buffer/TextureView all proxy
    // PartialEq to the underlying resource id (wgpu 30's `cmp::impl_eq_
    // ord_hash_proxy!`), so this is a real identity check, not a
    // by-construction tautology. ---
    assert_ne!(a_cull_bind_group_id, b_binding.cull_bind_group, "guard: B's cull bind group must differ from A's");
    assert_ne!(a_draw_bind_group_id, b_binding.draw_bind_group, "guard: B's draw bind group must differ from A's");
    assert_ne!(&a_output_buffer_id, b_output.buffer(), "guard: B's cull-output buffer must differ from A's");
    assert_ne!(&a_target_view_id, b_target.view(), "guard: B's offscreen target view must differ from A's");

    // --- Checkpoint 2: immediately after drop(A) + construct(B), before B
    // renders a single frame. ---
    assert_eq!(store.generation_write_count(), gen_writes_baseline, "after drop(A)+construct(B): generation_write_count moved");
    assert_eq!(meshes.upload_count(), mesh_uploads_baseline, "after drop(A)+construct(B): MeshRegistry re-uploaded");
    assert_eq!(clusters.upload_count(), cluster_uploads_baseline, "after drop(A)+construct(B): ClusterBuffer re-uploaded");
    assert_eq!(meshlets.upload_count(), meshlet_uploads_baseline, "after drop(A)+construct(B): MeshletBuffer re-uploaded");
    assert_eq!(materials.upload_count(), material_uploads_baseline, "after drop(A)+construct(B): MaterialRegistry re-uploaded");
    assert_eq!(textures.upload_count(), texture_uploads_baseline, "after drop(A)+construct(B): TextureStore re-uploaded");
    assert_eq!(arena.upload_count(), geometry_uploads_baseline, "after drop(A)+construct(B): GeometryArena re-uploaded");

    let mut b_converged_pixels: Vec<u8> = Vec::new();
    for frame_i in 0..N_FRAMES {
        let sim = frames.begin();
        let harvest = sim.end().end();
        let boundary = harvest.end();
        let stats = {
            let mut slots = [CellSlot { id, cell: cell.storage_mut() }];
            boundary.run(&mut store, &mut slots)
        };
        assert_eq!(stats.bytes, 0, "B frame {frame_i}: zero-write boundary must sync zero bytes");
        assert_eq!(stats.ranges, 0, "B frame {frame_i}: zero-write boundary must touch zero ranges");
        total_sync_bytes += stats.bytes;

        let (visible, stale, oob, frustum, pixels) = render_one_frame(
            &ctx,
            &b_cull,
            &b_binding.cull_bind_group,
            &b_cull_bg,
            &b_output,
            &b_draw,
            &b_draw_bg,
            &b_target,
            arena.vertex_buffer(),
            arena.index_buffer(),
            view_tokens.count(),
        );
        assert_eq!(stale, 0, "B frame {frame_i}: no stale row in this fixture");
        assert_eq!(oob, 0, "B frame {frame_i}: no OOB row in this fixture");
        assert_eq!(frustum, 0, "B frame {frame_i}: both rows stay inside the frustum");
        assert_eq!(visible, 2, "B frame {frame_i}: both instances visible every frame");
        b_converged_pixels = pixels;
    }

    // =====================================================================
    // THE WINDOW ends here (after B's last frame). Final assertions:
    // =====================================================================

    // (a) Σ SyncStats.bytes == 0 across the ENTIRE window (A's frames + B's
    // frames combined, one running total, never reset).
    assert_eq!(total_sync_bytes, 0, "assertion (a): total SyncStats.bytes across the whole window must be zero");

    // (b) Δ generation_write_count == 0.
    let gen_writes_after = store.generation_write_count();
    assert_eq!(gen_writes_after, gen_writes_baseline, "assertion (b): generation_write_count moved across the window");

    // (c) ALL SIX asset-store upload counters unchanged.
    assert_eq!(meshes.upload_count(), mesh_uploads_baseline, "assertion (c): MeshRegistry re-uploaded across the window");
    assert_eq!(clusters.upload_count(), cluster_uploads_baseline, "assertion (c): ClusterBuffer re-uploaded across the window");
    assert_eq!(meshlets.upload_count(), meshlet_uploads_baseline, "assertion (c): MeshletBuffer re-uploaded across the window");
    assert_eq!(materials.upload_count(), material_uploads_baseline, "assertion (c): MaterialRegistry re-uploaded across the window");
    assert_eq!(textures.upload_count(), texture_uploads_baseline, "assertion (c): TextureStore re-uploaded across the window");
    assert_eq!(arena.upload_count(), geometry_uploads_baseline, "assertion (c): GeometryArena re-uploaded across the window");

    // (d) streaming: StreamingGrid::write_cell_metadata was never called in
    // this test at all -- frozen for the window, per the module doc's
    // decision. Nothing to assert beyond "it was never invoked", which is
    // true by inspection of this file (no `StreamingGrid` import exists
    // here at all).

    // Self-verifying non-vacuity guard (house law): the window must
    // actually have RENDERED something in both halves -- a pair of blank
    // frames would satisfy a byte-identity comparison vacuously.
    let a_has_mesh0 = a_final_pixels.chunks_exact(4).any(|px| px[0] == mesh0_color[0] && px[1] == mesh0_color[1] && px[2] == mesh0_color[2]);
    let a_has_mesh1 = a_final_pixels.chunks_exact(4).any(|px| px[0] == mesh1_color[0] && px[1] == mesh1_color[1] && px[2] == mesh1_color[2]);
    let b_has_mesh0 = b_converged_pixels.chunks_exact(4).any(|px| px[0] == mesh0_color[0] && px[1] == mesh0_color[1] && px[2] == mesh0_color[2]);
    let b_has_mesh1 = b_converged_pixels.chunks_exact(4).any(|px| px[0] == mesh1_color[0] && px[1] == mesh1_color[1] && px[2] == mesh1_color[2]);
    assert!(a_has_mesh0 && a_has_mesh1, "guard: A's final frame must paint BOTH mesh colors -- a blank frame would pass byte-identity vacuously");
    assert!(b_has_mesh0 && b_has_mesh1, "guard: B's converged frame must paint BOTH mesh colors -- a blank frame would pass byte-identity vacuously");

    // (e) A-final and B-converged target hashes byte-identical. This
    // harness has NO TAA, NO jitter, NO Halton-phase/frame_count-driven
    // state anywhere in the cull/draw path -- so this comparison is
    // trivially stable (the design's own stated caveat: a jitter-driven
    // renderer would need frame_count/jitter-phase reset accounted for,
    // which does not exist here to get wrong). Compared byte-for-byte
    // (strictly stronger than a hash), with the hash also computed and
    // printed for the report.
    let a_hash = fnv1a_hash(&a_final_pixels);
    let b_hash = fnv1a_hash(&b_converged_pixels);
    println!("Test 13: A-final target hash = {a_hash:#018x}, B-converged target hash = {b_hash:#018x} (no TAA/jitter in this harness)");
    assert_eq!(a_final_pixels.len(), b_converged_pixels.len(), "assertion (e): A/B target byte-length mismatch");
    assert_eq!(a_final_pixels, b_converged_pixels, "assertion (e): A-final and B-converged frames must be byte-identical");
    assert_eq!(a_hash, b_hash, "assertion (e): A/B hashes must agree (implied by the byte-identity check above)");

    // (f) Device + every scene SSBO alive throughout, buffer IDs unchanged.
    // `wgpu::Buffer`/`wgpu::Device` have no `.global_id()` in wgpu 30; their
    // `PartialEq`/`Clone` impls (proxied to the underlying resource id) are
    // the sanctioned identity-comparison mechanism instead.
    assert_eq!(std::sync::Arc::as_ptr(ctx.device()), device_ptr_before, "assertion (f): device pointer changed across the window");
    assert_eq!(store.transform_buffer(), &transform_buf_before, "assertion (f): transform buffer identity changed");
    assert_eq!(store.instance_info_buffer(), &instance_info_buf_before, "assertion (f): instance_info buffer identity changed");
    assert_eq!(store.slot_mirror_buffer(), &slot_mirror_buf_before, "assertion (f): slot_mirror buffer identity changed");
    assert_eq!(store.generation_buffer(), &generation_buf_before, "assertion (f): generation buffer identity changed");
    assert_eq!(store.cell_metadata_buffer(), &cell_meta_buf_before, "assertion (f): cell_metadata buffer identity changed");
    assert_eq!(meshes.buffer(), &mesh_buf_before, "assertion (f): MeshRegistry buffer identity changed");
    assert_eq!(clusters.buffer(), &cluster_buf_before, "assertion (f): ClusterBuffer buffer identity changed");
    assert_eq!(meshlets.buffer(), &meshlet_buf_before, "assertion (f): MeshletBuffer buffer identity changed");
    assert_eq!(materials.buffer(), &material_buf_before, "assertion (f): MaterialRegistry buffer identity changed");
    assert_eq!(arena.vertex_buffer(), &vertex_buf_before, "assertion (f): GeometryArena vertex buffer identity changed");
    assert_eq!(arena.index_buffer(), &index_buf_before, "assertion (f): GeometryArena index buffer identity changed");
}
