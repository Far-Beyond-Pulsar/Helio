//! The renderer-side source of truth for the SceneDB<->Helio seam WGSL
//! (M3-a T9, design Rev 2 S5).
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
//! ## Bindings (group 0), all `var<storage, read>`
//!
//! SceneDB owns every write path (`write_transform`/`write_instance_info`/
//! the frame-boundary drivers, `MeshRegistry::register`,
//! `StreamingGrid::write_cell_metadata`, ...); Helio only ever reads through
//! this bind group:
//!
//! | binding | contents           | element type    |
//! |---------|---------------------|------------------|
//! | 0       | instance transforms | `Instance`       |
//! | 1       | instance info       | `InstanceInfo`   |
//! | 2       | slot mirror         | `u32`            |
//! | 3       | generations         | `u32`            |
//! | 4       | mesh configurator   | `MeshMetadata`   |
//! | 5       | cluster DAG         | `ClusterNode`    |
//! | 6       | meshlets            | `MeshletEntry`   |
//! | 7       | cell metadata       | `CellMeta`       |
//! | 8       | material registry   | `MaterialRow`    |
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
//! textures bind through Helio's own bindless array. Neither lives in this
//! bind group.
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

@group(0) @binding(0) var<storage, read> instances: array<Instance>;
@group(0) @binding(1) var<storage, read> instance_info: array<InstanceInfo>;
@group(0) @binding(2) var<storage, read> slot_mirror: array<u32>;
@group(0) @binding(3) var<storage, read> generations: array<u32>;
@group(0) @binding(4) var<storage, read> mesh_meta: array<MeshMetadata>;
@group(0) @binding(5) var<storage, read> clusters: array<ClusterNode>;
@group(0) @binding(6) var<storage, read> meshlets: array<MeshletEntry>;
@group(0) @binding(7) var<storage, read> cell_meta: array<CellMeta>;
@group(0) @binding(8) var<storage, read> materials: array<MaterialRow>;
"#;
