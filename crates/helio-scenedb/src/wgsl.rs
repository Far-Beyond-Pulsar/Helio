//! The renderer-side source of truth for the SceneDB<->Helio seam WGSL
//! (M3-a T9, design Rev 2 S5; two-group split M3-b T4, contract #47).
//!
//! Every struct text below is mirrored VERBATIM from `pulsar_scenedb`'s
//! `tests/gpu_layout.rs` (Test 3's host-vs-naga byte-exact layout proof:
//! `M2A_WGSL`'s `Instance`, `M2B_WGSL`'s `MeshMetadata`/`ClusterNode`,
//! `M3A_WGSL`'s `InstanceInfo`, `M3A_MESHLET_WGSL`'s `MeshletEntry`,
//! `M3A_MATERIAL_WGSL`'s `MaterialRow`) -- field names, field order, and
//! scalar-only shape are all load-bearing:
//! naga's `Layouter` computes offsets purely from declaration order and
//! scalar type, so any drift here (a reordered field, a renamed field, a
//! vector type standing in for scalars) breaks the byte-exact guarantee
//! Test 3 proves on the `pulsar_scenedb` side, even though this copy would
//! still parse. Do not hand-edit a field here without updating
//! `tests/gpu_layout.rs` in lockstep (and this crate's own Test 3 harness,
//! `crates/helio-scenedb/tests/binding_layout.rs`, M3-a T10).
//!
//! `CellMeta` is new here (M3-a T9, no `pulsar_scenedb`-side WGSL twin yet):
//! it mirrors `StreamingGrid::write_cell_metadata`'s packing
//! (`pulsar_scenedb::gpu::grid`) byte-for-byte -- 8 bytes total, `f32 alpha`
//! at offset 0, `u32 domain` at offset 4 (`Domain::Outer` = 0,
//! `Domain::Margin` = 1, `Domain::Inner` = 2).
//!
//! ## Bindings — two groups (M3-b T4, closes contract #47)
//!
//! M3-a shipped all 9 buffers in one `@group(0)`, one storage buffer over
//! the WebGPU default per-stage limit
//! (`Limits::default().max_storage_buffers_per_shader_stage == 8`) --
//! flagged as a hard M3-b requirement on [`crate::SceneDbBinding`]'s doc
//! comment since M3-a T9/T11 (perf-validation report contract #47, MISS).
//! M3-b splits the seam along its actual consumer boundary instead of
//! raising device limits:
//!
//! - **`@group(0)` -- the cull-read set.** Everything the β cull compute
//!   pass's term list touches (design §4: transform fetch, `InstanceInfo.
//!   mesh_index` bounds check, `generations[slot_mirror[row]]` validation,
//!   `MeshMetadata` local-AABB lookup). 5 entries.
//! - **`@group(1)` -- the draw/material set.** Everything the draw pass
//!   consumes once M3-β/γ passes exist (cluster DAG + meshlets for VG
//!   traversal, γ scope; materials for shading; cell metadata for
//!   domain/alpha). 4 entries. The cull pass never binds this group.
//! - **`@group(2)` is NOT declared in this constant** -- it is declared by
//!   [`CULL_WGSL`] below (M3-b T5: `cull_tokens`/`cull_expected_gens`/
//!   `cull_output`/`cull_uniforms`, the cull pass's own per-view inputs and
//!   outputs), which `CullPass::new` concatenates after this one at shader-
//!   build time. T4 deliberately left this index untouched in
//!   `SCENE_BINDINGS_WGSL` so T5 could claim it without renumbering
//!   anything here.
//!
//! | group | binding | contents           | element type    |
//! |-------|---------|---------------------|------------------|
//! | 0     | 0       | instance transforms | `Instance`       |
//! | 0     | 1       | instance info        | `InstanceInfo`   |
//! | 0     | 2       | slot mirror          | `u32`            |
//! | 0     | 3       | generations          | `u32`            |
//! | 0     | 4       | mesh configurator    | `MeshMetadata`   |
//! | 1     | 0       | cluster DAG          | `ClusterNode`    |
//! | 1     | 1       | meshlets              | `MeshletEntry`   |
//! | 1     | 2       | cell metadata         | `CellMeta`       |
//! | 1     | 3       | material registry     | `MaterialRow`    |
//!
//! SceneDB owns every write path (`write_transform`/`write_instance_info`/
//! the frame-boundary drivers, `MeshRegistry::register`,
//! `StreamingGrid::write_cell_metadata`, ...); Helio only ever reads through
//! this bind group pair.
//!
//! `MaterialRow` (M3-a T11, Rev 2.4 R8 approved 2026-07-16): 64 bytes, 16
//! scalar fields -- PBR params, four bindless texture slot indices
//! (sentinel `0xFFFF_FFFF` = unbound), a Radiant shader-graph index
//! (sentinel `0xFFFF_FFFF` = default PBR template), feature flags (bits 0-3
//! defined, 4-31 reserved), and an alpha-test cutoff. See
//! `pulsar_scenedb::gpu::MaterialRow`'s doc comment for the full field-by-
//! field rationale; this struct text must stay byte-identical to that
//! type's layout (this crate's own Test 3 harness,
//! `crates/helio-scenedb/tests/binding_layout.rs`, asserts it).
//!
//! Geometry (vertex/index) buffers bind at draw time (they vary per draw
//! call, unlike everything above which is one persistent SSBO per frame);
//! textures bind through Helio's own bindless array. Neither lives in
//! either group.
pub const SCENE_BINDINGS_WGSL: &str = r#"
struct Instance {
    transform: mat4x4<f32>,
}

struct InstanceInfo {
    mesh_index: u32,
    flags: u32,
}

struct MeshMetadata {
    vertex_offset: u32, index_offset: u32, index_count: u32, base_vertex: i32,
    material_index: u32, lod_count: u32,
    lod_d0: f32, lod_d1: f32, lod_d2: f32, lod_d3: f32,
    aabb_cx: f32, aabb_cy: f32, aabb_cz: f32,
    cluster_table_offset: u32,
    aabb_ex: f32, aabb_ey: f32, aabb_ez: f32,
    meshlet_count: u32,
}

struct ClusterNode {
    meshlet_offset: u32, meshlet_count: u32, parent_error: f32, self_error: f32,
    group_id: u32, child_offset: u32, child_count: u32, padding: u32,
    bs_x: f32, bs_y: f32, bs_z: f32, bs_w: f32,
}

struct MeshletEntry {
    sphere_x: f32, sphere_y: f32, sphere_z: f32, sphere_radius: f32,
    cone_packed: u32, data_offset: u32, counts_packed: u32, reserved: u32,
}

struct CellMeta {
    alpha: f32,
    domain: u32,
}

struct MaterialRow {
    base_color: u32,
    metallic: f32, roughness: f32, normal_scale: f32,
    emissive_r: f32, emissive_g: f32, emissive_b: f32, emissive_intensity: f32,
    tex_albedo: u32, tex_normal: u32, tex_metallic_roughness: u32, tex_emissive: u32,
    radiant_graph_index: u32,
    flags: u32,
    alpha_cutoff: f32,
    reserved: u32,
}

// group 0 -- cull-read set (5 entries; see this file's module doc for the
// per-stage arithmetic and crate::SceneDbBinding for the visibility split).
@group(0) @binding(0) var<storage, read> instances: array<Instance>;
@group(0) @binding(1) var<storage, read> instance_info: array<InstanceInfo>;
@group(0) @binding(2) var<storage, read> slot_mirror: array<u32>;
@group(0) @binding(3) var<storage, read> generations: array<u32>;
@group(0) @binding(4) var<storage, read> mesh_meta: array<MeshMetadata>;

// group 1 -- draw/material set (4 entries). Cull never binds this group.
// group 2 is RESERVED for M3-b T5's per-view ViewTokenBuffers -- not
// declared here.
@group(1) @binding(0) var<storage, read> clusters: array<ClusterNode>;
@group(1) @binding(1) var<storage, read> meshlets: array<MeshletEntry>;
@group(1) @binding(2) var<storage, read> cell_meta: array<CellMeta>;
@group(1) @binding(3) var<storage, read> materials: array<MaterialRow>;
"#;

/// The M3-b T5 cull compute pass: `CullPass::new` concatenates this AFTER
/// [`SCENE_BINDINGS_WGSL`] (`format!("{SCENE_BINDINGS_WGSL}\n{CULL_WGSL}")`,
/// the same wholesale-embedding idiom `tests/seam_smoke.rs` established for
/// M3-b T4) to get `cull_main`'s full source: group(0) (`instances`,
/// `instance_info`, `slot_mirror`, `generations`, `mesh_meta`) comes from
/// `SCENE_BINDINGS_WGSL`; this module supplies group(2) — the cull pass's
/// own per-view inputs/outputs (design S4, S14).
///
/// ## group(1) is genuinely never bound (design S4: "cull never binds
/// group 1")
///
/// `SCENE_BINDINGS_WGSL` also declares `@group(1)`'s four draw/material
/// bindings, but `cull_main` below never references any of them (`clusters`/
/// `meshlets`/`cell_meta`/`materials` go untouched). `CullPass::new` passes
/// `bind_group_layouts: &[Some(cull_layout), None, Some(output_layout)]` to
/// `create_pipeline_layout` — wgpu 30's `PipelineLayoutDescriptor::
/// bind_group_layouts` is `&[Option<&BindGroupLayout>]` (sparse; the M3-b T4
/// smoke test bound all three positions with `Some`, but the type has always
/// supported gaps), so group(1) has NO layout at pipeline-layout index 1 at
/// all, and `CullPass::record` never calls `set_bind_group(1, ..)`. This is
/// stronger than "declared but unused, bound anyway for positional validity"
/// (T4's smoke-test pattern): here group(1) is genuinely absent from the
/// pipeline. This works because wgpu/naga's pipeline-layout validation
/// checks bindings *reachable from the entry point* (`cull_main`), not every
/// module-scope declaration in the shader source — `tests/cull_pass.rs`'s
/// passing dispatch is the empirical proof, alongside this doc's reasoning.
///
/// ## group(2) storage-buffer budget arithmetic (the task's explicit ≤3
/// requirement, on top of group(0)'s 5 COMPUTE-visible entries: 5+3=8, the
/// WebGPU default `max_storage_buffers_per_shader_stage` ceiling — #47 must
/// stay closed)
///
/// The design lists FIVE logical group(2) inputs/outputs (tokens,
/// expected_gens, the output command buffer, the visible-id buffer, the
/// atomic counter) plus a uniform block. Five storage buffers would blow
/// the budget (5 + 5 = 10 > 8). This module ships exactly THREE storage
/// bindings by combining the last three into one:
///
/// - binding 0: `cull_tokens: array<u32>` — [`pulsar_scenedb::gpu::
///   ViewTokenBuffers::tokens_buffer`], read-only (producer-owned, cannot be
///   merged with binding 1 — two independently allocated `wgpu::Buffer`s).
/// - binding 1: `cull_expected_gens: array<u32>` —
///   [`pulsar_scenedb::gpu::ViewTokenBuffers::expected_gens_buffer`],
///   read-only (same reason).
/// - binding 2: `cull_output: CullOutput` (read_write) — COMBINES the
///   atomic counters (`visible_count`/`stale_drops`/`oob_drops`/
///   `frustum_drops`, a fixed 16-byte header) AND the per-command-slot
///   output array (`records: array<CullRecord>`) in ONE binding. Each
///   `CullRecord`'s first 20 bytes (`index_count`/`instance_count`/
///   `first_index`/`base_vertex`/`first_instance`) are field-for-field
///   identical to the standalone `DrawCommand` struct below (same order,
///   same offsets) — a future repack into a tightly-packed
///   `array<DrawCommand>` for `multi_draw_indexed_indirect` (M3-b T7's
///   scope; T7 "documents its choice" per the plan, may pick a
///   count-clamped loop instead and avoid repacking entirely) is a
///   fixed-stride copy, not a redesign. `row` stays its OWN field — the
///   design's row-valued `visible_instance_ids[command_slot] = row` (S3,
///   R11) is never folded into `flags` or the command fields, only
///   co-located in the same binding to fit the budget. `flags` bit 0 is the
///   S12 near-clip carry (see [`CULL_WGSL`]'s `NEAR_CLIP_FLAG` doc).
/// - binding 3: `cull_uniforms: CullUniforms` (`var<uniform>`, NOT a
///   storage buffer — uniform buffers have their own separate WebGPU limit,
///   `max_uniform_buffers_per_shader_stage` default 12, untouched by this
///   arithmetic) — the view-proj matrix (S12 near-clip corner projection),
///   6 frustum planes (S4 frustum term), and `count`/`mesh_count`/
///   `capacity` (the `i < count` guard, the mesh-table bounds-check ceiling,
///   and the S14.2 slot-allocation ceiling).
///
/// **Arithmetic:** group(0) COMPUTE-visible entries (5, `SceneDbBinding`'s
/// doc comment) + group(2) storage entries (3, `cull_tokens`/
/// `cull_expected_gens`/`cull_output`; `cull_uniforms` doesn't count) =
/// **8 / 8** — at the ceiling, not over it. No more than 3 was available;
/// this design uses exactly 3.
pub const CULL_WGSL: &str = r#"
// Standalone indirect-draw wire format (spec S14.1) -- 20 bytes, matches
// wgpu's DrawIndexedIndirectArgs field-for-field. Declared here purely so
// Test 3 pins the wire format byte-exact for M3-b T7's future indirect-draw
// executor; the ACTUAL group(2) binding below uses CullRecord, whose first
// 20 bytes are IDENTICAL field-for-field (see this file's module doc).
struct DrawCommand {
    index_count: u32,
    instance_count: u32,
    first_index: u32,
    base_vertex: i32,
    first_instance: u32,
}

// One combined per-command-slot output record (module doc: the group(2)
// storage-budget packing). First 5 fields identical to DrawCommand
// field-for-field (offsets 0/4/8/12/16); `row` is the design's row-valued
// visible_instance_ids[slot] (S3/R11 -- command-slot-keyed, ROW-valued,
// its own field, never folded into `flags` or the command fields); `flags`
// bit 0 (NEAR_CLIP_FLAG) carries the S12 near-clip bypass for this slot.
struct CullRecord {
    index_count: u32,
    instance_count: u32,
    first_index: u32,
    base_vertex: i32,
    first_instance: u32,
    row: u32,
    flags: u32,
    reserved: u32,
}

const NEAR_CLIP_FLAG: u32 = 1u;

// Fixed 16-byte atomic-counters header, followed by the per-slot record
// array (S14.2 bounded-atomic allocation: overflowing threads still
// atomicAdd `visible_count` but do NOT write past `capacity` -- the CPU
// clamps on readback, Test 5 in T8).
struct CullOutput {
    visible_count: atomic<u32>,
    stale_drops: atomic<u32>,
    oob_drops: atomic<u32>,
    frustum_drops: atomic<u32>,
    records: array<CullRecord>,
}

struct CullUniforms {
    view_proj: mat4x4<f32>,
    planes: array<vec4<f32>, 6>,
    count: u32,
    mesh_count: u32,
    capacity: u32,
    reserved: u32,
}

@group(2) @binding(0) var<storage, read> cull_tokens: array<u32>;
@group(2) @binding(1) var<storage, read> cull_expected_gens: array<u32>;
@group(2) @binding(2) var<storage, read_write> cull_output: CullOutput;
@group(2) @binding(3) var<uniform> cull_uniforms: CullUniforms;

// S11: |M3x3| absolute-value world-extent transform.
fn abs_mat3(m: mat4x4<f32>) -> mat3x3<f32> {
    return mat3x3<f32>(abs(m[0].xyz), abs(m[1].xyz), abs(m[2].xyz));
}

// Positive-vertex AABB-vs-plane test (n.p+d>=0 convention, spatial.rs's
// Frustum doc): true if the box is at least partially on the inside side of
// `plane`; false only when the box is ENTIRELY outside it.
fn plane_test(center: vec3<f32>, extents: vec3<f32>, plane: vec4<f32>) -> bool {
    let r = extents.x * abs(plane.x) + extents.y * abs(plane.y) + extents.z * abs(plane.z);
    let d = dot(plane.xyz, center) + plane.w;
    return d >= -r;
}

@compute @workgroup_size(64)
fn cull_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= cull_uniforms.count) {
        return;
    }
    let row = cull_tokens[i];

    // -- S3.1 generation validation: generations[slot_mirror[row]] vs the
    // harvest-time expected_gens[i] snapshot (view_upload.rs's own doc
    // comment states this exact indexing -- verified against
    // scene_store.rs: slot_mirror[global_row] = slot_base + local_slot,
    // and generations is written at that SAME slot_base + local_slot index
    // (write_generation), so slot_mirror's stored value already IS the
    // generations-buffer index; no extra base offset applies here). --
    let slot = slot_mirror[row];
    let live_gen = generations[slot];
    if (live_gen != cull_expected_gens[i]) {
        atomicAdd(&cull_output.stale_drops, 1u);
        return;
    }

    // -- mesh_index bounds check (M3-a T4 recycled-tail defense, REQUIRED:
    // recycled-region tail bytes are contractually undefined until
    // written). --
    let mesh_index = instance_info[row].mesh_index;
    if (mesh_index >= cull_uniforms.mesh_count) {
        atomicAdd(&cull_output.oob_drops, 1u);
        return;
    }

    let mesh = mesh_meta[mesh_index];
    let m = instances[row].transform;

    // -- S11 |M3x3| world AABB. --
    let local_center = vec3<f32>(mesh.aabb_cx, mesh.aabb_cy, mesh.aabb_cz);
    let local_extents = vec3<f32>(mesh.aabb_ex, mesh.aabb_ey, mesh.aabb_ez);
    let world_center = (m * vec4<f32>(local_center, 1.0)).xyz;
    let world_extents = abs_mat3(m) * local_extents;

    // -- S12 near-plane guard: project all 8 world AABB corners through the
    // view-proj matrix; W<=0 on ANY corner bypasses culling entirely and
    // marks the instance visible with the near-clip flag set (S12.1). --
    var near_clip = false;
    for (var c: u32 = 0u; c < 8u; c = c + 1u) {
        let sx = select(-1.0, 1.0, (c & 1u) != 0u);
        let sy = select(-1.0, 1.0, (c & 2u) != 0u);
        let sz = select(-1.0, 1.0, (c & 4u) != 0u);
        let corner = world_center + vec3<f32>(sx, sy, sz) * world_extents;
        let clip = cull_uniforms.view_proj * vec4<f32>(corner, 1.0);
        if (clip.w <= 0.0) {
            near_clip = true;
        }
    }

    var visible = true;
    if (!near_clip) {
        for (var p: u32 = 0u; p < 6u; p = p + 1u) {
            if (!plane_test(world_center, world_extents, cull_uniforms.planes[p])) {
                visible = false;
            }
        }
        if (!visible) {
            atomicAdd(&cull_output.frustum_drops, 1u);
            return;
        }
    }

    // -- S14.2 bounded-atomic slot allocation: overflowing threads still
    // bump visible_count (so the CPU can clamp and know how many were
    // dropped) but never write past capacity. --
    let out_slot = atomicAdd(&cull_output.visible_count, 1u);
    if (out_slot < cull_uniforms.capacity) {
        var flags = 0u;
        if (near_clip) {
            flags = NEAR_CLIP_FLAG;
        }
        cull_output.records[out_slot] = CullRecord(
            mesh.index_count,
            1u,
            mesh.index_offset,
            mesh.base_vertex,
            out_slot,
            row,
            flags,
            0u,
        );
    }
}
"#;
