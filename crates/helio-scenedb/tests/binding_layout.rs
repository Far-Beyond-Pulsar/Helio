//! Test 3 (C5): host struct offsets vs naga reflection of the ACTUAL seam
//! WGSL (`helio_scenedb::wgsl::SCENE_BINDINGS_WGSL`) — byte-exact, mirroring
//! `pulsar_scenedb/tests/gpu_layout.rs`'s harness one-for-one (M3-a T10).
//!
//! This is deliberately NOT a copy of the struct text reflected on the
//! SceneDB side: it parses the renderer-side source of truth directly (the
//! `pub const SCENE_BINDINGS_WGSL` Helio's real shaders will `format!` their
//! own bindings around), so any drift between the two WGSL copies — the
//! exact hazard `wgsl.rs`'s doc comment warns about — fails HERE, on the
//! Helio side, not just on the SceneDB side. This harness is what M3-b/-g
//! extend for every new shader struct that joins `SCENE_BINDINGS_WGSL`.
//!
//! Pure naga (WGSL front-end parse + `Layouter`) — no GPU device, no
//! adapter, no `pollster`. Safe to run anywhere `cargo test` runs.

use helio_scenedb::cull::{CullRecord, CullUniforms, DrawCommand};
use helio_scenedb::wgsl::{CULL_WGSL, SCENE_BINDINGS_WGSL};
use pulsar_scenedb::gpu::{ClusterNode, InstanceInfo, MaterialRow, MeshMetadata, MeshletEntry};

/// Reflect (size, [(member_name, offset)]) for a named struct in WGSL source.
/// Ported verbatim from `pulsar_scenedb/tests/gpu_layout.rs::wgsl_struct_
/// layout` (naga 30 API — same crate major as `pulsar_scenedb`'s `gpu`
/// feature dev-dep; T1 already migrated this helper to naga 30, so the API
/// transfers directly with no adaptation needed).
fn wgsl_struct_layout(src: &str, name: &str) -> (u32, Vec<(String, u32)>) {
    let module = naga::front::wgsl::parse_str(src).expect("valid WGSL");
    let mut layouter = naga::proc::Layouter::default();
    layouter.update(module.to_ctx()).expect("layout");
    let (handle, ty) = module
        .types
        .iter()
        .find(|(_, t)| t.name.as_deref() == Some(name))
        .unwrap_or_else(|| panic!("struct {name} not found"));
    let naga::TypeInner::Struct { members, .. } = &ty.inner else {
        panic!("{name} is not a struct");
    };
    let size = layouter[handle].size;
    let offsets = members
        .iter()
        .map(|m| (m.name.clone().unwrap_or_default(), m.offset))
        .collect();
    (size, offsets)
}

#[test]
fn instance_struct_is_byte_exact() {
    let (size, members) = wgsl_struct_layout(SCENE_BINDINGS_WGSL, "Instance");
    // Host element: [f32; 16], 64 bytes, transform at offset 0 (C5).
    assert_eq!(size, 64, "WGSL Instance size == size_of::<[f32; 16]>()");
    assert_eq!(size as usize, std::mem::size_of::<[f32; 16]>());
    assert_eq!(members, vec![("transform".to_string(), 0)]);
}

#[test]
fn instance_info_struct_is_byte_exact() {
    let (size, members) = wgsl_struct_layout(SCENE_BINDINGS_WGSL, "InstanceInfo");
    // Host element: `pulsar_scenedb::gpu::InstanceInfo` (`spatial.rs`, C5
    // amendment — cull's token->mesh link).
    assert_eq!(size, 8, "WGSL InstanceInfo size == size_of::<InstanceInfo>()");
    assert_eq!(size as usize, std::mem::size_of::<InstanceInfo>());
    assert_eq!(
        members,
        vec![("mesh_index".to_string(), 0), ("flags".to_string(), 4)]
    );
}

#[test]
fn mesh_metadata_struct_is_byte_exact() {
    let (size, members) = wgsl_struct_layout(SCENE_BINDINGS_WGSL, "MeshMetadata");
    // Host element: `pulsar_scenedb::gpu::MeshMetadata`, 72 bytes (C5/§6.1).
    assert_eq!(size, 72, "WGSL MeshMetadata size == size_of::<MeshMetadata>()");
    assert_eq!(size as usize, std::mem::size_of::<MeshMetadata>());
    assert_eq!(
        members,
        vec![
            ("vertex_offset".to_string(), 0),
            ("index_offset".to_string(), 4),
            ("index_count".to_string(), 8),
            ("base_vertex".to_string(), 12),
            ("material_index".to_string(), 16),
            ("lod_count".to_string(), 20),
            ("lod_d0".to_string(), 24),
            ("lod_d1".to_string(), 28),
            ("lod_d2".to_string(), 32),
            ("lod_d3".to_string(), 36),
            ("aabb_cx".to_string(), 40),
            ("aabb_cy".to_string(), 44),
            ("aabb_cz".to_string(), 48),
            ("cluster_table_offset".to_string(), 52),
            ("aabb_ex".to_string(), 56),
            ("aabb_ey".to_string(), 60),
            ("aabb_ez".to_string(), 64),
            ("meshlet_count".to_string(), 68),
        ]
    );
}

#[test]
fn cluster_node_struct_is_byte_exact() {
    let (size, members) = wgsl_struct_layout(SCENE_BINDINGS_WGSL, "ClusterNode");
    // Host element: `pulsar_scenedb::gpu::ClusterNode`, 48 bytes (C5).
    assert_eq!(size, 48, "WGSL ClusterNode size == size_of::<ClusterNode>()");
    assert_eq!(size as usize, std::mem::size_of::<ClusterNode>());
    assert_eq!(
        members,
        vec![
            ("meshlet_offset".to_string(), 0),
            ("meshlet_count".to_string(), 4),
            ("parent_error".to_string(), 8),
            ("self_error".to_string(), 12),
            ("group_id".to_string(), 16),
            ("child_offset".to_string(), 20),
            ("child_count".to_string(), 24),
            ("padding".to_string(), 28),
            ("bs_x".to_string(), 32),
            ("bs_y".to_string(), 36),
            ("bs_z".to_string(), 40),
            ("bs_w".to_string(), 44),
        ]
    );
}

#[test]
fn meshlet_entry_struct_is_byte_exact() {
    let (size, members) = wgsl_struct_layout(SCENE_BINDINGS_WGSL, "MeshletEntry");
    // Host element: `pulsar_scenedb::gpu::MeshletEntry`, 32 bytes (C5
    // amendment / punch-list R12).
    assert_eq!(size, 32, "WGSL MeshletEntry size == size_of::<MeshletEntry>()");
    assert_eq!(size as usize, std::mem::size_of::<MeshletEntry>());
    assert_eq!(
        members,
        vec![
            ("sphere_x".to_string(), 0),
            ("sphere_y".to_string(), 4),
            ("sphere_z".to_string(), 8),
            ("sphere_radius".to_string(), 12),
            ("cone_packed".to_string(), 16),
            ("data_offset".to_string(), 20),
            ("counts_packed".to_string(), 24),
            ("reserved".to_string(), 28),
        ]
    );
}

/// M3-a T11 (Rev 2.4 R8, approved 2026-07-16): host
/// `pulsar_scenedb::gpu::MaterialRow` vs naga reflection of the ACTUAL seam
/// WGSL's `MaterialRow` (`@group(1) @binding(3)` post-M3-b-T4-split; was
/// "binding 8" pre-split, in the original single-group layout), byte-exact
/// -- Rust-twin
/// `std::mem::size_of` cross-check mirrors every other struct test in this
/// file. 16 scalar fields, all 16 offsets asserted.
#[test]
fn material_row_struct_is_byte_exact() {
    let (size, members) = wgsl_struct_layout(SCENE_BINDINGS_WGSL, "MaterialRow");
    // Host element: `pulsar_scenedb::gpu::MaterialRow`, 64 bytes (C5/§10.1,
    // Rev 2.4 R8).
    assert_eq!(size, 64, "WGSL MaterialRow size == size_of::<MaterialRow>()");
    assert_eq!(size as usize, std::mem::size_of::<MaterialRow>());
    assert_eq!(
        members,
        vec![
            ("base_color".to_string(), 0),
            ("metallic".to_string(), 4),
            ("roughness".to_string(), 8),
            ("normal_scale".to_string(), 12),
            ("emissive_r".to_string(), 16),
            ("emissive_g".to_string(), 20),
            ("emissive_b".to_string(), 24),
            ("emissive_intensity".to_string(), 28),
            ("tex_albedo".to_string(), 32),
            ("tex_normal".to_string(), 36),
            ("tex_metallic_roughness".to_string(), 40),
            ("tex_emissive".to_string(), 44),
            ("radiant_graph_index".to_string(), 48),
            ("flags".to_string(), 52),
            ("alpha_cutoff".to_string(), 56),
            ("reserved".to_string(), 60),
        ]
    );
}

/// `CellMeta` has no Rust twin type (`wgsl.rs`'s doc comment, M3-a T9): the
/// streaming grid packs `(f32 alpha, u32 domain)` manually into a raw byte
/// buffer at `dense_id * 8` in `StreamingGrid::write_cell_metadata`
/// (`pulsar_scenedb/src/gpu/grid.rs` lines 377-388) rather than through a
/// `#[repr(C)]` struct + typed-slice `write_buffer`. So this test asserts
/// the WGSL side only — size and both field offsets — with this comment as
/// the pointer to the CPU-side packing it must stay byte-compatible with:
/// `alpha.to_le_bytes()` at `offset..offset+4`, `domain_code.to_le_bytes()`
/// at `offset+4..offset+8` (`Domain::Outer` = 0, `Margin` = 1, `Inner` = 2).
#[test]
fn cell_meta_struct_is_byte_exact() {
    let (size, members) = wgsl_struct_layout(SCENE_BINDINGS_WGSL, "CellMeta");
    assert_eq!(size, 8, "WGSL CellMeta size == 8 bytes (grid.rs:377-388 packing)");
    assert_eq!(
        members,
        vec![("alpha".to_string(), 0), ("domain".to_string(), 4)]
    );
}

/// M3-b T5: [`CULL_WGSL`]'s struct declarations reference nothing outside
/// themselves EXCEPT `cull_main`'s body, which reads `SCENE_BINDINGS_WGSL`'s
/// group(0) globals (`instances`/`instance_info`/`slot_mirror`/
/// `generations`/`mesh_meta`) -- naga's WGSL front end validates a module's
/// function bodies too, so parsing `CULL_WGSL` in isolation fails with
/// "undefined identifier". This mirrors `CullPass::new`'s own
/// `format!("{SCENE_BINDINGS_WGSL}\n{CULL_WGSL}")` concatenation exactly
/// (the ACTUAL source the cull pipeline compiles), not an approximation.
fn cull_wgsl_combined() -> String {
    format!("{SCENE_BINDINGS_WGSL}\n{CULL_WGSL}")
}

/// M3-b T5: host [`DrawCommand`] (Helio-owned, no `pulsar_scenedb` twin --
/// C0: SceneDB never touches draw commands) vs naga reflection of
/// `CULL_WGSL`'s standalone `DrawCommand` struct -- the spec S14.1
/// indexed-indirect wire format, byte-exact.
#[test]
fn draw_command_struct_is_byte_exact() {
    let src = cull_wgsl_combined();
    let (size, members) = wgsl_struct_layout(&src, "DrawCommand");
    assert_eq!(size, 20, "WGSL DrawCommand size == size_of::<DrawCommand>() (spec S14.1)");
    assert_eq!(size as usize, std::mem::size_of::<DrawCommand>());
    assert_eq!(
        members,
        vec![
            ("index_count".to_string(), 0),
            ("instance_count".to_string(), 4),
            ("first_index".to_string(), 8),
            ("base_vertex".to_string(), 12),
            ("first_instance".to_string(), 16),
        ]
    );
}

/// M3-b T5: host [`CullRecord`] vs naga reflection of `CULL_WGSL`'s
/// `CullRecord` -- the combined per-command-slot output record
/// (`wgsl::CULL_WGSL`'s module doc has the group(2) storage-budget
/// rationale for why DrawCommand's 5 fields, `row`, and `flags` all live in
/// ONE record type). First 5 offsets identical to `DrawCommand`'s own test
/// above -- asserted again here directly against `CullRecord`'s OWN
/// reflection (not inferred), so a future edit that broke that field-for-
/// field promise would fail HERE, not just look consistent by construction.
#[test]
fn cull_record_struct_is_byte_exact() {
    let src = cull_wgsl_combined();
    let (size, members) = wgsl_struct_layout(&src, "CullRecord");
    assert_eq!(size, 32, "WGSL CullRecord size == size_of::<CullRecord>()");
    assert_eq!(size as usize, std::mem::size_of::<CullRecord>());
    assert_eq!(
        members,
        vec![
            ("index_count".to_string(), 0),
            ("instance_count".to_string(), 4),
            ("first_index".to_string(), 8),
            ("base_vertex".to_string(), 12),
            ("first_instance".to_string(), 16),
            ("row".to_string(), 20),
            ("flags".to_string(), 24),
            ("reserved".to_string(), 28),
        ]
    );
}

/// M3-b T5: host [`CullUniforms`] vs naga reflection of `CULL_WGSL`'s
/// `CullUniforms` -- the cull pass's per-dispatch uniform block (view-proj
/// matrix, 6 frustum planes, and the three scalar guards).
#[test]
fn cull_uniforms_struct_is_byte_exact() {
    let src = cull_wgsl_combined();
    let (size, members) = wgsl_struct_layout(&src, "CullUniforms");
    assert_eq!(size, 176, "WGSL CullUniforms size == size_of::<CullUniforms>()");
    assert_eq!(size as usize, std::mem::size_of::<CullUniforms>());
    assert_eq!(
        members,
        vec![
            ("view_proj".to_string(), 0),
            ("planes".to_string(), 64),
            ("count".to_string(), 160),
            ("mesh_count".to_string(), 164),
            ("capacity".to_string(), 168),
            ("reserved".to_string(), 172),
        ]
    );
}

/// M3-b T5: `CullOutput`'s header offsets (the 16-byte atomics header --
/// `visible_count`/`stale_drops`/`oob_drops`/`frustum_drops` -- followed by
/// `records` at offset 16). naga's `Layouter` reports the STRUCT's overall
/// `size` as 48, not 16, for a trailing runtime array: its "minimum binding
/// size" convention counts the fixed header PLUS exactly one array-element
/// stride (16 + `CullRecord`'s own 32-byte stride,
/// `cull_record_struct_is_byte_exact` above) -- not the true fixed-header
/// size alone. Asserted here as 48 (not 16) so this test documents naga's
/// real behavior rather than the WGSL-source-level intuition; the MEMBER
/// OFFSETS (0/4/8/12/16) are what this test actually pins, and they are
/// unaffected by the trailing-array size quirk.
#[test]
fn cull_output_header_is_byte_exact() {
    let src = cull_wgsl_combined();
    let (size, members) = wgsl_struct_layout(&src, "CullOutput");
    assert_eq!(
        size, 48,
        "naga Layouter size for a struct with a trailing runtime array == fixed header (16) + ONE array-element stride (32, CullRecord) -- not the header alone"
    );
    assert_eq!(
        members,
        vec![
            ("visible_count".to_string(), 0),
            ("stale_drops".to_string(), 4),
            ("oob_drops".to_string(), 8),
            ("frustum_drops".to_string(), 12),
            ("records".to_string(), 16),
        ]
    );
}
