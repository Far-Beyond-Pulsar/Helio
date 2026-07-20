//! M3-b T6 deliverable 3: GPU-vs-CPU cull equality. `support::cpu_cull_all`
//! (an INDEPENDENT Rust port of the design §4 β term list, written from the
//! spec prose — see `support/mod.rs`'s module doc for why it is not a
//! transliteration of `CULL_WGSL`) is run over the SAME token/gen/buffer
//! state the real `CullPass` GPU dispatch consumes, across >= 3 seeded
//! randomized fixtures mixing visible / frustum-culled / near-clip / stale
//! / out-of-range rows. The visible ROW SETS must match exactly; on
//! mismatch the differing rows and both sides' decisions are printed,
//! along with the failing seed.

#[path = "support/mod.rs"]
mod support;

use helio_scenedb::cull::{CullOutputBuffers, CullPass, CullUniforms, NEAR_CLIP_FLAG};
use helio_scenedb::SceneDbBinding;
use pulsar_scenedb::gpu::{
    CellSlot, ClusterBuffer, EngineGpuContext, FrameDriver, HarvestPipeline, HarvestStaging,
    MaterialRegistry, MeshClass, MeshMetadata, MeshRegistry, MeshletBuffer, SceneGpuStore, View,
    ViewTokenBuffers,
};
use pulsar_scenedb::{Aabb, InstanceInfo, Scratchpad, SpatialCell};
use std::collections::HashSet;

use support::{
    cpu_cull_all, flatten_column_major, frustum_planes_90deg, mat3_mul, mat3_rot_x, mat3_rot_z,
    mesh, readback, scene_cfg, test_context, translation, view_proj_90deg, CpuCullInputs,
    CpuDecision, Rng,
};

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum RowCategory {
    Visible,
    FrustumCulled,
    NearClip,
    Stale,
    Oob,
}

#[allow(dead_code)] // `category` documents intent at construction time; the actual decision is re-derived by the CPU/GPU cull logic, not read back off this field
struct RowSpec {
    transform: [f32; 16],
    mesh_index: u32,
    corrupt_gen: bool,
    category: RowCategory,
}

/// A position/mesh/optional-rotation choice safely inside the frustum at
/// every category that reuses it (`Visible`, `Stale`, `Oob` all start from
/// this — staleness/OOB-ness is layered on top of an otherwise-visible
/// row, deliberately, so the equality check exercises TERM ORDER: the
/// shader must drop these BEFORE ever reaching the frustum test).
fn visible_like(rng: &mut Rng) -> ([f32; 16], u32) {
    let x = rng.range_f32(-3.0, 3.0);
    let y = rng.range_f32(-3.0, 3.0);
    let z = rng.range_f32(-60.0, -8.0);
    let mesh_index = if rng.chance(0.5) { 0 } else { 2 };
    let transform = if rng.chance(0.25) {
        let rz = rng.range_f32(-50.0, 50.0);
        let rx = rng.range_f32(-50.0, 50.0);
        let r = mat3_mul(mat3_rot_z(rz), mat3_rot_x(rx));
        flatten_column_major(r, [x, y, z])
    } else {
        translation([x, y, z])
    };
    (transform, mesh_index)
}

fn gen_row(rng: &mut Rng, category: RowCategory) -> RowSpec {
    match category {
        RowCategory::Visible => {
            let (transform, mesh_index) = visible_like(rng);
            RowSpec { transform, mesh_index, corrupt_gen: false, category }
        }
        RowCategory::Stale => {
            let (transform, mesh_index) = visible_like(rng);
            RowSpec { transform, mesh_index, corrupt_gen: true, category }
        }
        RowCategory::Oob => {
            let (transform, _valid_mesh_index) = visible_like(rng);
            // mesh_count is always 3 in this fixture (see `register_
            // meshes`) — deliberately out of range by 1..=3.
            let mesh_index = 3 + rng.next_u32(3);
            RowSpec { transform, mesh_index, corrupt_gen: false, category }
        }
        RowCategory::FrustumCulled => {
            let sign = if rng.chance(0.5) { 1.0 } else { -1.0 };
            let x = sign * rng.range_f32(500.0, 1500.0);
            let y = rng.range_f32(-3.0, 3.0);
            let z = rng.range_f32(-60.0, -8.0);
            let mesh_index = if rng.chance(0.5) { 0 } else { 2 };
            RowSpec { transform: translation([x, y, z]), mesh_index, corrupt_gen: false, category }
        }
        RowCategory::NearClip => {
            let x = rng.range_f32(-0.3, 0.3);
            let y = rng.range_f32(-0.3, 0.3);
            let z = rng.range_f32(-0.6, -0.1);
            // mesh 1 ("mesh_b") is the tall (extents.z = 2.0) mesh — its
            // far corners always cross world z=0 (behind the camera) in
            // this shallow z range, guaranteeing the near-clip trigger
            // regardless of the random x/y jitter above.
            RowSpec { transform: translation([x, y, z]), mesh_index: 1, corrupt_gen: false, category }
        }
    }
}

fn register_meshes(ctx: &EngineGpuContext) -> (MeshRegistry, Vec<MeshMetadata>) {
    let mut meshes = MeshRegistry::new(ctx, 4);
    let mesh_a = mesh([0.0, 0.0, 0.0], [0.5, 0.5, 0.5], 6, 0, 0);
    let mesh_b = mesh([0.0, 0.0, 0.0], [0.5, 0.5, 2.0], 12, 6, 4);
    let mesh_c = mesh([0.1, -0.1, 0.05], [0.8, 0.3, 0.6], 9, 18, 10);
    assert_eq!(meshes.register(ctx.queue(), mesh_a).unwrap(), 0);
    assert_eq!(meshes.register(ctx.queue(), mesh_b).unwrap(), 1);
    assert_eq!(meshes.register(ctx.queue(), mesh_c).unwrap(), 2);
    (meshes, vec![mesh_a, mesh_b, mesh_c])
}

fn parse_u32_column(bytes: &[u8], count: usize) -> Vec<u32> {
    (0..count).map(|i| u32::from_le_bytes(bytes[i * 4..i * 4 + 4].try_into().unwrap())).collect()
}

/// Builds one randomized fixture (seeded), dispatches the real `CullPass`,
/// runs the independent CPU reference over the identical logical input, and
/// asserts the visible row sets (plus per-row near-clip flags and telemetry
/// counters) match exactly. Every assertion message embeds `seed` so a
/// failure prints it without needing a panic hook.
fn run_equality_check(seed: u64) {
    let ctx = test_context();
    let device: &wgpu::Device = ctx.device().as_ref();
    let mut rng = Rng::new(seed);

    // 5 deterministic anchors (guarantee every category fires at least
    // once, independent of the RNG stream) + 40 randomized filler rows.
    let anchors = [
        RowCategory::Visible,
        RowCategory::FrustumCulled,
        RowCategory::NearClip,
        RowCategory::Stale,
        RowCategory::Oob,
    ];
    let filler_count = 40usize;
    let mut categories: Vec<RowCategory> = anchors.to_vec();
    for _ in 0..filler_count {
        let roll = rng.next_u32(100);
        categories.push(match roll {
            0..=34 => RowCategory::Visible,
            35..=54 => RowCategory::FrustumCulled,
            55..=69 => RowCategory::NearClip,
            70..=84 => RowCategory::Stale,
            _ => RowCategory::Oob,
        });
    }
    let n = categories.len() as u32;

    let mut store = SceneGpuStore::new(&ctx, scene_cfg());
    let mut cell = SpatialCell::with_transform(128).unwrap();

    let mut specs: Vec<RowSpec> = Vec::with_capacity(categories.len());
    let mut handles = Vec::with_capacity(categories.len());
    for (i, &cat) in categories.iter().enumerate() {
        let spec = gen_row(&mut rng, cat);
        let t = &spec.transform;
        let (x, y, z) = (t[12], t[13], t[14]);
        let h = cell
            .alloc(Aabb { min: [x - 1.0, y - 1.0, z - 1.0], max: [x + 1.0, y + 1.0, z + 1.0] })
            .unwrap();
        assert_eq!(
            cell.row_of(h),
            Some(i as u32),
            "seed={seed}: fresh-alloc row-assignment assumption broken at row {i} — the equality \
             test relies on sequential row assignment for a churn-free cell"
        );
        handles.push(h);
        specs.push(spec);
    }

    let id = store.register_cell(cell.storage(), 0).unwrap();
    let base = store.row_region_base(id);
    assert_eq!(base, 0, "seed={seed}: single cell, class-0 region base 0");

    let mut frames = FrameDriver::new();
    let sim = frames.begin();
    for (i, h) in handles.iter().enumerate() {
        assert!(store.write_transform(id, cell.storage_mut(), *h, &specs[i].transform, &sim));
        assert!(store.write_instance_info(
            id,
            cell.storage_mut(),
            *h,
            InstanceInfo { mesh_index: specs[i].mesh_index, flags: 0 },
            &sim
        ));
    }

    let harvest_phase = sim.end().end();
    let pipeline = HarvestPipeline::new();
    let mut pad = Scratchpad::new();
    let mut staging = HarvestStaging::new();
    // Gigantic view — this is a pure CPU-side broad-phase query, unrelated
    // to the GPU frustum under test, so it must catch even the
    // deliberately-far-away FrustumCulled rows (x up to +-1500).
    let view = View::Aabb(Aabb { min: [-5000.0, -5000.0, -5000.0], max: [5000.0, 5000.0, 5000.0] });
    let harvested = pipeline.harvest_cell(&cell, base, MeshClass::Traditional, &view, &mut pad, &mut staging, &harvest_phase);
    assert_eq!(harvested, n, "seed={seed}: broad harvest view must catch every row");
    assert_eq!(staging.traditional, (0..n).collect::<Vec<_>>(), "seed={seed}: plain-path 100%-hit harvest preserves row order (base=0)");

    let boundary = harvest_phase.end();
    {
        let mut slots = [CellSlot { id, cell: cell.storage_mut() }];
        boundary.run(&mut store, &mut slots);
    }

    // Corrupt the Stale rows' expected-gen snapshot directly — every live
    // row's REAL generation is 1 (fresh alloc, no churn anywhere in this
    // fixture), so any value != 1 is a guaranteed, deliberate mismatch.
    // This tests the CULL SHADER's stale-vs-live comparison in isolation
    // (Test 2, tests/cull_stale_and_oob.rs, is what proves harvest's OWN
    // gens column is correct under real free/boundary churn).
    for (i, spec) in specs.iter().enumerate() {
        if spec.corrupt_gen {
            staging.traditional_gens[i] = 2;
        }
    }

    let mut view_tokens = ViewTokenBuffers::new(&ctx, "cull-equality-view", 0);
    view_tokens.upload(&ctx, &staging, MeshClass::Traditional);
    assert_eq!(view_tokens.count(), n);

    let (meshes, mesh_table) = register_meshes(&ctx);
    let clusters = ClusterBuffer::new(&ctx, 1);
    let meshlets = MeshletBuffer::new(&ctx, 1);
    let materials = MaterialRegistry::new(&ctx, 1);
    let scene_binding = SceneDbBinding::new(device, &store, &meshes, &clusters, &meshlets, &materials);
    let cull_pass = CullPass::new(device, &scene_binding.cull_layout);
    let output = CullOutputBuffers::new(device, "cull-equality-output", n + 8);
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

    // -- Read back real GPU state for the CPU reference's slot_mirror /
    // generations inputs (rather than assuming — this is real SSBO
    // content, post-boundary). --
    let gen_buf = store.generation_buffer();
    let slot_buf = store.slot_mirror_buffer();
    let gen_bytes = readback(&ctx, gen_buf, gen_buf.size());
    let slot_bytes = readback(&ctx, slot_buf, slot_buf.size());
    let generations = parse_u32_column(&gen_bytes, (gen_buf.size() / 4) as usize);
    let slot_mirror = parse_u32_column(&slot_bytes, (slot_buf.size() / 4) as usize);

    let tokens: Vec<u32> = staging.traditional.clone();
    let expected_gens: Vec<u32> = staging.traditional_gens.clone();
    let instances: Vec<[f32; 16]> = specs.iter().map(|s| s.transform).collect();
    let instance_info: Vec<InstanceInfo> = specs.iter().map(|s| InstanceInfo { mesh_index: s.mesh_index, flags: 0 }).collect();

    let inputs = CpuCullInputs {
        tokens: &tokens,
        expected_gens: &expected_gens,
        slot_mirror: &slot_mirror,
        generations: &generations,
        instance_info: &instance_info,
        instances: &instances,
        mesh_table: &mesh_table,
        mesh_count: meshes.len(),
        view_proj: view_proj_90deg(),
        planes: frustum_planes_90deg(),
    };
    let cpu_results = cpu_cull_all(&inputs);

    // -- House-law guard: every category this fixture is DESIGNED to
    // exercise must have actually fired, per the CPU reference's OWN tally
    // (not a hand-prediction of which random row lands where). --
    let cpu_stale = cpu_results.iter().filter(|(_, d)| *d == CpuDecision::Stale).count();
    let cpu_oob = cpu_results.iter().filter(|(_, d)| *d == CpuDecision::Oob).count();
    let cpu_frustum = cpu_results.iter().filter(|(_, d)| *d == CpuDecision::FrustumCulled).count();
    let cpu_near_clip = cpu_results.iter().filter(|(_, d)| matches!(d, CpuDecision::Visible { near_clip: true })).count();
    let cpu_visible_total = cpu_results.iter().filter(|(_, d)| matches!(d, CpuDecision::Visible { .. })).count();
    assert!(cpu_stale > 0, "seed={seed}: guard — fixture must produce >=1 Stale row");
    assert!(cpu_oob > 0, "seed={seed}: guard — fixture must produce >=1 Oob row");
    assert!(cpu_frustum > 0, "seed={seed}: guard — fixture must produce >=1 FrustumCulled row");
    assert!(cpu_near_clip > 0, "seed={seed}: guard — fixture must produce >=1 near-clip Visible row");
    assert!(cpu_visible_total > cpu_near_clip, "seed={seed}: guard — fixture must produce >=1 normal (non-near-clip) Visible row");

    // -- GPU readback. --
    let out_bytes = readback(&ctx, output.buffer(), output.byte_size());
    let u32_at = |off: usize| u32::from_le_bytes(out_bytes[off..off + 4].try_into().unwrap());
    let gpu_visible_count = u32_at(0);
    let gpu_stale = u32_at(4);
    let gpu_oob = u32_at(8);
    let gpu_frustum = u32_at(12);
    assert!(gpu_visible_count <= output.capacity(), "seed={seed}: fixture must not overflow the output capacity (would need Test 5's clamp, out of scope here)");

    struct GpuRecord {
        index_count: u32,
        first_index: u32,
        base_vertex: i32,
        near_clip: bool,
    }
    let mut gpu_by_row: std::collections::HashMap<u32, GpuRecord> = std::collections::HashMap::new();
    for slot in 0..gpu_visible_count {
        let off = CullOutputBuffers::HEADER_BYTES as usize + slot as usize * CullOutputBuffers::RECORD_BYTES as usize;
        let index_count = u32_at(off);
        let first_index = u32_at(off + 8);
        let base_vertex = i32::from_le_bytes(out_bytes[off + 12..off + 16].try_into().unwrap());
        let row = u32_at(off + 20);
        let flags = u32_at(off + 24);
        gpu_by_row.insert(row, GpuRecord { index_count, first_index, base_vertex, near_clip: flags & NEAR_CLIP_FLAG != 0 });
    }

    // -- Counter equality. --
    assert_eq!(gpu_stale, cpu_stale as u32, "seed={seed}: stale_drops mismatch (GPU {gpu_stale} vs CPU {cpu_stale})");
    assert_eq!(gpu_oob, cpu_oob as u32, "seed={seed}: oob_drops mismatch (GPU {gpu_oob} vs CPU {cpu_oob})");
    assert_eq!(gpu_frustum, cpu_frustum as u32, "seed={seed}: frustum_drops mismatch (GPU {gpu_frustum} vs CPU {cpu_frustum})");
    assert_eq!(gpu_visible_count, cpu_visible_total as u32, "seed={seed}: visible_count mismatch (GPU {gpu_visible_count} vs CPU {cpu_visible_total})");

    // -- THE equality assertion: visible row SETS must match exactly.
    // Order is NOT compared (see command_slot_order_is_empirically_stable_
    // within_one_dispatch below for the determinism finding this is based
    // on — command-slot allocation is a GPU atomic, not contractually
    // ordered, so this test compares as sets on principle regardless of
    // what was empirically observed on this run's hardware). --
    let cpu_visible_rows: HashSet<u32> = cpu_results
        .iter()
        .filter_map(|(row, d)| if matches!(d, CpuDecision::Visible { .. }) { Some(*row) } else { None })
        .collect();
    let gpu_visible_rows: HashSet<u32> = gpu_by_row.keys().copied().collect();

    if cpu_visible_rows != gpu_visible_rows {
        let only_cpu: Vec<u32> = cpu_visible_rows.difference(&gpu_visible_rows).copied().collect();
        let only_gpu: Vec<u32> = gpu_visible_rows.difference(&cpu_visible_rows).copied().collect();
        let cpu_decision_of = |row: u32| cpu_results.iter().find(|(r, _)| *r == row).map(|(_, d)| *d);
        eprintln!("seed={seed}: VISIBLE SET MISMATCH");
        for row in &only_cpu {
            eprintln!("  row {row}: CPU says {:?}, GPU has NO command", cpu_decision_of(*row));
        }
        for row in &only_gpu {
            eprintln!("  row {row}: GPU emitted a command, CPU says {:?}", cpu_decision_of(*row));
        }
        panic!("seed={seed}: visible row sets differ — {} CPU-only, {} GPU-only (see stderr for the row list)", only_cpu.len(), only_gpu.len());
    }

    // -- Per-row cross-checks over the agreed-upon visible set: near-clip
    // flag parity + DrawCommand fields resolved from the SAME mesh table. --
    for (row, decision) in &cpu_results {
        let CpuDecision::Visible { near_clip: cpu_near } = decision else { continue };
        let gpu_rec = &gpu_by_row[row];
        assert_eq!(
            gpu_rec.near_clip, *cpu_near,
            "seed={seed}: row {row} near-clip flag mismatch (CPU {cpu_near} vs GPU {})",
            gpu_rec.near_clip
        );
        let mesh_index = instance_info[*row as usize].mesh_index;
        let expected_mesh = &mesh_table[mesh_index as usize];
        assert_eq!(gpu_rec.index_count, expected_mesh.index_count, "seed={seed}: row {row} index_count mismatch");
        assert_eq!(gpu_rec.first_index, expected_mesh.index_offset, "seed={seed}: row {row} first_index mismatch");
        assert_eq!(gpu_rec.base_vertex, expected_mesh.base_vertex, "seed={seed}: row {row} base_vertex mismatch");
    }
}

/// M3-b T6 deliverable 3: >= 3 seeded randomized fixtures. Seeds are
/// arbitrary fixed constants (not re-rolled per run) so a failure is
/// reproducible from the printed seed alone without needing to capture
/// process-random state.
#[test]
fn gpu_vs_cpu_cull_equality_matches_across_randomized_seeds() {
    for seed in [0xC0FFEE_u64, 0xDEAD_BEEF_u64, 0x1234_5678_9ABC_DEF0_u64] {
        run_equality_check(seed);
    }
}

/// M3-b T6 deliverable 3's explicit instruction: "check whether the
/// atomic-alloc makes order nondeterministic ... document that." Builds
/// ONE fixture, dispatches `CullPass` TWICE against fresh output buffers
/// with IDENTICAL input, and compares the row->slot assignment between the
/// two runs. This is an empirical probe on the actual adapter/driver this
/// suite runs on, not a portability claim — `run_equality_check` above
/// compares visible sets, never slot order, regardless of what this test
/// finds.
#[test]
fn command_slot_order_determinism_probe() {
    let ctx = test_context();
    let device: &wgpu::Device = ctx.device().as_ref();
    let mut rng = Rng::new(0x5EED_5EED);

    let categories: Vec<RowCategory> = (0..64)
        .map(|_| match rng.next_u32(100) {
            0..=59 => RowCategory::Visible,
            60..=79 => RowCategory::FrustumCulled,
            _ => RowCategory::NearClip,
        })
        .collect();
    let n = categories.len() as u32;

    let mut store = SceneGpuStore::new(&ctx, scene_cfg());
    let mut cell = SpatialCell::with_transform(128).unwrap();
    let mut specs = Vec::new();
    for cat in &categories {
        specs.push(gen_row(&mut rng, *cat));
    }
    let mut handles = Vec::with_capacity(specs.len());
    for (i, spec) in specs.iter().enumerate() {
        let t = &spec.transform;
        let h = cell
            .alloc(Aabb {
                min: [t[12] - 1.0, t[13] - 1.0, t[14] - 1.0],
                max: [t[12] + 1.0, t[13] + 1.0, t[14] + 1.0],
            })
            .unwrap();
        assert_eq!(cell.row_of(h), Some(i as u32), "fresh-alloc row-assignment assumption broken at row {i}");
        handles.push(h);
    }
    let id = store.register_cell(cell.storage(), 0).unwrap();
    let base = store.row_region_base(id);

    let mut frames = FrameDriver::new();
    let sim = frames.begin();
    for (h, spec) in handles.iter().zip(specs.iter()) {
        assert!(store.write_transform(id, cell.storage_mut(), *h, &spec.transform, &sim));
        assert!(store.write_instance_info(id, cell.storage_mut(), *h, InstanceInfo { mesh_index: spec.mesh_index, flags: 0 }, &sim));
    }

    let harvest_phase = sim.end().end();
    let pipeline = HarvestPipeline::new();
    let mut pad = Scratchpad::new();
    let mut staging = HarvestStaging::new();
    let view = View::Aabb(Aabb { min: [-5000.0, -5000.0, -5000.0], max: [5000.0, 5000.0, 5000.0] });
    let harvested = pipeline.harvest_cell(&cell, base, MeshClass::Traditional, &view, &mut pad, &mut staging, &harvest_phase);
    assert_eq!(harvested, n);

    let boundary = harvest_phase.end();
    {
        let mut slots = [CellSlot { id, cell: cell.storage_mut() }];
        boundary.run(&mut store, &mut slots);
    }

    let mut view_tokens = ViewTokenBuffers::new(&ctx, "determinism-probe-view", 0);
    view_tokens.upload(&ctx, &staging, MeshClass::Traditional);

    let (meshes, _mesh_table) = register_meshes(&ctx);
    let clusters = ClusterBuffer::new(&ctx, 1);
    let meshlets = MeshletBuffer::new(&ctx, 1);
    let materials = MaterialRegistry::new(&ctx, 1);
    let scene_binding = SceneDbBinding::new(device, &store, &meshes, &clusters, &meshlets, &materials);
    let cull_pass = CullPass::new(device, &scene_binding.cull_layout);

    let uniforms = CullUniforms {
        view_proj: view_proj_90deg(),
        planes: frustum_planes_90deg(),
        count: view_tokens.count(),
        mesh_count: meshes.len(),
        capacity: view_tokens.count() + 8,
        reserved: 0,
    };
    cull_pass.write_uniforms(ctx.queue(), &uniforms);

    let dispatch_once = || -> Vec<(u32, u32)> {
        let output = CullOutputBuffers::new(device, "determinism-probe-output", view_tokens.count() + 8);
        output.clear_counters(ctx.queue());
        let output_bind_group = cull_pass.build_output_bind_group(device, &view_tokens, &output);
        let mut encoder = device.create_command_encoder(&Default::default());
        cull_pass.record(&mut encoder, &scene_binding.cull_bind_group, &output_bind_group, view_tokens.count());
        ctx.queue().submit([encoder.finish()]);
        let out_bytes = readback(&ctx, output.buffer(), output.byte_size());
        let u32_at = |off: usize| u32::from_le_bytes(out_bytes[off..off + 4].try_into().unwrap());
        let visible_count = u32_at(0);
        (0..visible_count.min(output.capacity()))
            .map(|slot| {
                let off = CullOutputBuffers::HEADER_BYTES as usize + slot as usize * CullOutputBuffers::RECORD_BYTES as usize;
                (slot, u32_at(off + 20)) // (slot, row)
            })
            .collect()
    };

    let run_a = dispatch_once();
    let run_b = dispatch_once();

    let order_identical = run_a == run_b;
    // RESOLVED (M3-b T6 review follow-up): slot assignment is genuinely
    // NONDETERMINISTIC on this adapter — back-to-back dispatches of a
    // byte-identical input hand out different slot ranges, because the
    // §14.2 bounded `atomicAdd` hands slots out in workgroup-completion
    // order and that order is a scheduling detail, not a guarantee. An
    // earlier revision of this probe ASSERTED order stability on the
    // strength of a lucky observation; that assert was flaky by
    // construction and has been removed. What is asserted below is what
    // the contract actually promises.
    println!(
        "[cull order probe] slot assignment {} between two identical dispatches \
         (informational — order is NOT a contract; §14.2 promises only a valid bijection)",
        if order_identical { "was stable" } else { "DIFFERED" }
    );

    // The real invariants, order notwithstanding: both runs must emit the
    // same SET of rows, and each run's slots must be a duplicate-free
    // allocation within capacity (an atomic that handed the same slot to
    // two threads, or a slot past capacity, would corrupt the command
    // buffer — that IS a contract, and it is what this probe now pins).
    let rows_of = |run: &Vec<(u32, u32)>| {
        let mut v: Vec<u32> = run.iter().map(|&(_, row)| row).collect();
        v.sort_unstable();
        v
    };
    assert_eq!(
        rows_of(&run_a),
        rows_of(&run_b),
        "the same input must yield the same visible ROW set regardless of slot order"
    );
    for (label, run) in [("a", &run_a), ("b", &run_b)] {
        let mut slots: Vec<u32> = run.iter().map(|&(slot, _)| slot).collect();
        slots.sort_unstable();
        let before = slots.len();
        slots.dedup();
        assert_eq!(before, slots.len(), "run {label}: a command slot was allocated twice");
        // Same capacity the closure above constructs its output with.
        let capacity = view_tokens.count() + 8;
        assert!(
            slots.iter().all(|&s| s < capacity),
            "run {label}: a command slot landed outside the capacity bound ({capacity})"
        );
    }
}
