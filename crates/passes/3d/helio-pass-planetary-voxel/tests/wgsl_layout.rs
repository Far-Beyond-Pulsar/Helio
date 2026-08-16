use helio_pass_planetary_voxel::{
    GpuExtractionCounters, GpuExtractionRange, GpuExtractionRequest, GpuLookupQuery,
    GpuLookupResult, GpuPageTableEntry, GpuResidencyCounters, GpuResidencyUniform,
    GpuSurfaceGatherCounters, GpuSurfaceGatherJob, GpuTerrainDraw, GpuTerrainMeshlet,
    GpuTerrainMeshletBounds, GpuTerrainVertex, GpuTransvoxelCell, GpuTransvoxelCellOffset,
    GpuTransvoxelClassifyCounters, GpuTransvoxelDispatch, GpuTransvoxelEmissionCounters,
    GpuTransvoxelScanBlock, GpuTransvoxelTransitionCell, GpuTransvoxelTransitionCounters,
    GpuTransvoxelTransitionDispatch, EXTRACTION_LAYOUT_WGSL, RESIDENCY_WGSL, SURFACE_DRAW_WGSL,
    SURFACE_GATHER_WGSL, SURFACE_PUBLISH_WGSL, TERRAIN_MESHLET_BUILD_WGSL,
    TERRAIN_MESHLET_CULL_WGSL, TRANSVOXEL_CLASSIFY_WGSL, TRANSVOXEL_EMIT_WGSL,
    TRANSVOXEL_TRANSITION_GPU_WGSL,
};
use std::mem::{align_of, offset_of, size_of};
use wgpu::naga::{
    front::wgsl,
    valid::{Capabilities, ValidationFlags, Validator},
    TypeInner,
};

fn wgsl_struct(name: &str) -> (u32, Vec<(String, u32)>) {
    wgsl_struct_in(RESIDENCY_WGSL, name)
}

fn wgsl_struct_in(source: &str, name: &str) -> (u32, Vec<(String, u32)>) {
    let module = wgsl::parse_str(source)
        .unwrap_or_else(|error| panic!("planetary WGSL contract must parse: {error}"));
    Validator::new(ValidationFlags::all(), Capabilities::all())
        .validate(&module)
        .unwrap_or_else(|error| panic!("planetary WGSL contract must validate: {error}"));
    let (_, ty) = module
        .types
        .iter()
        .find(|(_, ty)| ty.name.as_deref() == Some(name))
        .unwrap_or_else(|| panic!("missing WGSL struct {name}"));
    let TypeInner::Struct { members, span } = &ty.inner else {
        panic!("WGSL type {name} is not a struct");
    };
    (
        *span,
        members
            .iter()
            .map(|member| {
                (
                    member.name.clone().expect("contract field must be named"),
                    member.offset,
                )
            })
            .collect(),
    )
}

#[test]
fn extraction_layout_parses_and_validates() {
    let module = wgsl::parse_str(EXTRACTION_LAYOUT_WGSL).expect("extraction WGSL parses");
    Validator::new(ValidationFlags::all(), Capabilities::all())
        .validate(&module)
        .expect("extraction WGSL validates");
}

#[test]
fn planetary_surface_shaders_parse_and_validate() {
    for (name, source) in [
        ("surface publication", SURFACE_PUBLISH_WGSL),
        ("surface draw", SURFACE_DRAW_WGSL),
        ("terrain meshlet build", TERRAIN_MESHLET_BUILD_WGSL),
        ("terrain meshlet cull", TERRAIN_MESHLET_CULL_WGSL),
    ] {
        let module = wgsl::parse_str(source)
            .unwrap_or_else(|error| panic!("planetary {name} WGSL must parse: {error}"));
        Validator::new(ValidationFlags::all(), Capabilities::all())
            .validate(&module)
            .unwrap_or_else(|error| panic!("planetary {name} WGSL must validate: {error}"));
    }

    assert_eq!(
        wgsl_struct_in(SURFACE_PUBLISH_WGSL, "GpuSurfaceFeedback"),
        (
            32,
            vec![
                ("submitted_jobs".into(), 0),
                ("published_jobs".into(), 4),
                ("stale_rejections".into(), 8),
                ("overflow_rejections".into(), 12),
                ("incomplete_rejections".into(), 16),
                ("_pad0".into(), 20),
                ("_pad1".into(), 24),
                ("_pad2".into(), 28),
            ],
        )
    );
    assert_eq!(
        wgsl_struct_in(SURFACE_PUBLISH_WGSL, "GpuSurfaceState"),
        (
            80,
            vec![
                ("generation_low".into(), 0),
                ("generation_high".into(), 4),
                ("valid".into(), 8),
                ("revision".into(), 12),
                ("regular_first_vertex".into(), 16),
                ("regular_vertex_count".into(), 20),
                ("regular_first_index".into(), 24),
                ("regular_index_count".into(), 28),
                ("regular_first_meshlet".into(), 32),
                ("regular_meshlet_count".into(), 36),
                ("transition_first_vertex".into(), 40),
                ("transition_vertex_count".into(), 44),
                ("transition_first_index".into(), 48),
                ("transition_index_count".into(), 52),
                ("transition_first_meshlet".into(), 56),
                ("transition_meshlet_count".into(), 60),
                ("transition_mask".into(), 64),
                ("_pad0".into(), 68),
                ("_pad1".into(), 72),
                ("_pad2".into(), 76),
            ],
        )
    );
    assert_eq!(wgsl_struct_in(SURFACE_PUBLISH_WGSL, "GpuDrawPage").0, 48);
    assert_eq!(
        wgsl_struct_in(SURFACE_PUBLISH_WGSL, "DrawIndexedIndirectArgs").0,
        20
    );
}

#[test]
fn extraction_request_and_range_match_wgsl_exactly() {
    assert_eq!(align_of::<GpuExtractionRequest>(), 16);
    assert_eq!(size_of::<GpuExtractionRequest>(), 32);
    assert_eq!(
        wgsl_struct_in(EXTRACTION_LAYOUT_WGSL, "GpuExtractionRequest"),
        (
            32,
            vec![
                ("page_slot".into(), 0),
                ("generation_low".into(), 4),
                ("generation_high".into(), 8),
                ("transition_mask".into(), 12),
                ("dirty_microbricks_low".into(), 16),
                ("dirty_microbricks_high".into(), 20),
                ("_pad".into(), 24),
            ],
        )
    );

    assert_eq!(align_of::<GpuExtractionRange>(), 16);
    assert_eq!(size_of::<GpuExtractionRange>(), 32);
    assert_eq!(
        wgsl_struct_in(EXTRACTION_LAYOUT_WGSL, "GpuExtractionRange"),
        (
            32,
            vec![
                ("first_vertex".into(), 0),
                ("vertex_count".into(), 4),
                ("first_index".into(), 8),
                ("index_count".into(), 12),
                ("first_meshlet".into(), 16),
                ("meshlet_count".into(), 20),
                ("generation_low".into(), 24),
                ("generation_high".into(), 28),
            ],
        )
    );
}

#[test]
fn extraction_outputs_and_counters_match_wgsl_exactly() {
    assert_eq!(align_of::<GpuTerrainVertex>(), 16);
    assert_eq!(size_of::<GpuTerrainVertex>(), 32);
    assert_eq!(
        wgsl_struct_in(EXTRACTION_LAYOUT_WGSL, "GpuTerrainVertex"),
        (
            32,
            vec![
                ("position".into(), 0),
                ("material".into(), 12),
                ("normal".into(), 16),
                ("flags".into(), 28),
            ],
        )
    );

    assert_eq!(align_of::<GpuTerrainMeshlet>(), 16);
    assert_eq!(size_of::<GpuTerrainMeshlet>(), 32);
    assert_eq!(
        wgsl_struct_in(EXTRACTION_LAYOUT_WGSL, "GpuTerrainMeshlet"),
        (
            32,
            vec![
                ("first_index".into(), 0),
                ("index_count".into(), 4),
                ("first_vertex".into(), 8),
                ("vertex_count".into(), 12),
                ("bounds_offset".into(), 16),
                ("generation_low".into(), 20),
                ("generation_high".into(), 24),
                ("_pad".into(), 28),
            ],
        )
    );

    assert_eq!(align_of::<GpuTerrainMeshletBounds>(), 16);
    assert_eq!(size_of::<GpuTerrainMeshletBounds>(), 48);
    assert_eq!(
        wgsl_struct_in(EXTRACTION_LAYOUT_WGSL, "GpuTerrainMeshletBounds"),
        (
            48,
            vec![
                ("center".into(), 0),
                ("radius".into(), 12),
                ("cone_apex".into(), 16),
                ("cone_cutoff".into(), 28),
                ("cone_axis".into(), 32),
                ("_pad".into(), 44),
            ],
        )
    );

    assert_eq!(align_of::<GpuTerrainDraw>(), 16);
    assert_eq!(size_of::<GpuTerrainDraw>(), 16);
    assert_eq!(
        wgsl_struct_in(EXTRACTION_LAYOUT_WGSL, "GpuTerrainDraw"),
        (
            16,
            vec![
                ("page_slot".into(), 0),
                ("meshlet_index".into(), 4),
                ("surface_kind".into(), 8),
                ("lod".into(), 12),
            ],
        )
    );

    assert_eq!(align_of::<GpuExtractionCounters>(), 16);
    assert_eq!(size_of::<GpuExtractionCounters>(), 48);
    assert_eq!(
        wgsl_struct_in(EXTRACTION_LAYOUT_WGSL, "GpuExtractionCounters"),
        (
            48,
            vec![
                ("requests".into(), 0),
                ("active_cells".into(), 4),
                ("vertices".into(), 8),
                ("indices".into(), 12),
                ("meshlets".into(), 16),
                ("completed".into(), 20),
                ("stale_rejected".into(), 24),
                ("overflowed".into(), 28),
                ("vertex_overflow".into(), 32),
                ("index_overflow".into(), 36),
                ("meshlet_overflow".into(), 40),
                ("_pad".into(), 44),
            ],
        )
    );
}

#[test]
fn residency_shader_parses_and_validates() {
    let module = wgsl::parse_str(RESIDENCY_WGSL).expect("residency WGSL parses");
    Validator::new(ValidationFlags::all(), Capabilities::all())
        .validate(&module)
        .expect("residency WGSL validates");
}

#[test]
fn page_table_entry_matches_wgsl_exactly() {
    assert_eq!(align_of::<GpuPageTableEntry>(), 16);
    assert_eq!(size_of::<GpuPageTableEntry>(), 64);
    assert_eq!(
        wgsl_struct("GpuPageTableEntry"),
        (
            64,
            vec![
                (
                    "planet_id".into(),
                    offset_of!(GpuPageTableEntry, planet_id) as u32
                ),
                (
                    "lod0_cell_min_x".into(),
                    offset_of!(GpuPageTableEntry, lod0_cell_min_x) as u32
                ),
                (
                    "lod0_cell_min_y".into(),
                    offset_of!(GpuPageTableEntry, lod0_cell_min_y) as u32
                ),
                (
                    "lod0_cell_min_z".into(),
                    offset_of!(GpuPageTableEntry, lod0_cell_min_z) as u32
                ),
                ("lod".into(), offset_of!(GpuPageTableEntry, lod) as u32),
                ("slot".into(), offset_of!(GpuPageTableEntry, slot) as u32),
                (
                    "generation_low".into(),
                    offset_of!(GpuPageTableEntry, generation_low) as u32,
                ),
                (
                    "generation_high".into(),
                    offset_of!(GpuPageTableEntry, generation_high) as u32,
                ),
                ("state".into(), offset_of!(GpuPageTableEntry, state) as u32),
                ("_pad".into(), offset_of!(GpuPageTableEntry, _pad) as u32),
            ],
        )
    );
}

#[test]
fn uniform_query_and_result_match_wgsl_exactly() {
    assert_eq!(align_of::<GpuResidencyUniform>(), 16);
    assert_eq!(size_of::<GpuResidencyUniform>(), 32);
    assert_eq!(
        wgsl_struct("GpuResidencyUniform"),
        (
            32,
            vec![
                (
                    "table_mask".into(),
                    offset_of!(GpuResidencyUniform, table_mask) as u32,
                ),
                (
                    "max_probe".into(),
                    offset_of!(GpuResidencyUniform, max_probe) as u32,
                ),
                (
                    "resident_pages".into(),
                    offset_of!(GpuResidencyUniform, resident_pages) as u32,
                ),
                (
                    "atlas_tiles_x".into(),
                    offset_of!(GpuResidencyUniform, atlas_tiles_x) as u32,
                ),
                (
                    "atlas_tiles_y".into(),
                    offset_of!(GpuResidencyUniform, atlas_tiles_y) as u32,
                ),
                (
                    "atlas_tiles_z".into(),
                    offset_of!(GpuResidencyUniform, atlas_tiles_z) as u32,
                ),
                (
                    "publication_epoch_low".into(),
                    offset_of!(GpuResidencyUniform, publication_epoch_low) as u32,
                ),
                (
                    "publication_epoch_high".into(),
                    offset_of!(GpuResidencyUniform, publication_epoch_high) as u32,
                ),
            ],
        )
    );

    assert_eq!(align_of::<GpuLookupQuery>(), 16);
    assert_eq!(size_of::<GpuLookupQuery>(), 48);
    assert_eq!(
        wgsl_struct("GpuLookupQuery"),
        (
            48,
            vec![
                (
                    "planet_id".into(),
                    offset_of!(GpuLookupQuery, planet_id) as u32
                ),
                (
                    "lod0_cell_min_x".into(),
                    offset_of!(GpuLookupQuery, lod0_cell_min_x) as u32
                ),
                (
                    "lod0_cell_min_y".into(),
                    offset_of!(GpuLookupQuery, lod0_cell_min_y) as u32
                ),
                (
                    "lod0_cell_min_z".into(),
                    offset_of!(GpuLookupQuery, lod0_cell_min_z) as u32
                ),
                ("lod".into(), offset_of!(GpuLookupQuery, lod) as u32),
                ("_pad".into(), offset_of!(GpuLookupQuery, _pad) as u32),
            ],
        )
    );

    assert_eq!(align_of::<GpuLookupResult>(), 16);
    assert_eq!(size_of::<GpuLookupResult>(), 16);
    assert_eq!(
        wgsl_struct("GpuLookupResult"),
        (
            16,
            vec![
                ("slot".into(), offset_of!(GpuLookupResult, slot) as u32),
                (
                    "generation_low".into(),
                    offset_of!(GpuLookupResult, generation_low) as u32,
                ),
                (
                    "generation_high".into(),
                    offset_of!(GpuLookupResult, generation_high) as u32,
                ),
                (
                    "probes_and_found".into(),
                    offset_of!(GpuLookupResult, probes_and_found) as u32,
                ),
            ],
        )
    );
}

#[test]
fn counters_are_explicitly_padded_and_pod_sized() {
    assert_eq!(align_of::<GpuResidencyCounters>(), 16);
    assert_eq!(size_of::<GpuResidencyCounters>(), 96);
    assert_eq!(
        wgsl_struct("GpuResidencyCounters"),
        (
            96,
            vec![
                ("resident_pages".into(), 0),
                ("resident_cell_bytes_low".into(), 4),
                ("resident_cell_bytes_high".into(), 8),
                ("table_occupied".into(), 12),
                ("table_tombstones".into(), 16),
                ("uploads_published".into(), 20),
                ("evictions_published".into(), 24),
                ("stale_rejections".into(), 28),
                ("generation_conflicts".into(), 32),
                ("backpressure_events".into(), 36),
                ("table_saturation_events".into(), 40),
                ("batches_submitted".into(), 44),
                ("cell_bytes_uploaded_low".into(), 48),
                ("cell_bytes_uploaded_high".into(), 52),
                ("peak_resident_pages".into(), 56),
                ("peak_resident_cell_bytes_low".into(), 60),
                ("peak_resident_cell_bytes_high".into(), 64),
                ("allocated_gpu_bytes_low".into(), 68),
                ("allocated_gpu_bytes_high".into(), 72),
                ("resource_buffers".into(), 76),
                ("resource_textures".into(), 80),
                ("atlas_capacity_pages".into(), 84),
                ("device_rebuilds".into(), 88),
                ("_pad".into(), 92),
            ],
        )
    );
}

#[test]
fn surface_gather_layouts_match_wgsl_exactly() {
    assert_eq!(align_of::<GpuSurfaceGatherJob>(), 16);
    assert_eq!(size_of::<GpuSurfaceGatherJob>(), 80);
    assert_eq!(
        wgsl_struct_in(SURFACE_GATHER_WGSL, "GpuSurfaceGatherJob"),
        (
            80,
            vec![
                ("planet_id".into(), 0),
                ("lod0_cell_min_x".into(), 16),
                ("lod0_cell_min_y".into(), 24),
                ("lod0_cell_min_z".into(), 32),
                ("lod".into(), 40),
                ("generation_low".into(), 44),
                ("generation_high".into(), 48),
                ("transition_mask".into(), 52),
                ("target_slot".into(), 56),
                ("residency_epoch_low".into(), 60),
                ("residency_epoch_high".into(), 64),
                ("_pad".into(), 68),
            ],
        )
    );

    assert_eq!(align_of::<GpuSurfaceGatherCounters>(), 16);
    assert_eq!(size_of::<GpuSurfaceGatherCounters>(), 32);
    assert_eq!(
        wgsl_struct_in(SURFACE_GATHER_WGSL, "GpuSurfaceGatherCounters"),
        (
            32,
            vec![
                ("regular_samples".into(), 0),
                ("transition_samples".into(), 4),
                ("table_probes".into(), 8),
                ("page_misses".into(), 12),
                ("stale_targets".into(), 16),
                ("completed".into(), 20),
                ("_pad0".into(), 24),
                ("_pad1".into(), 28),
            ],
        )
    );
}

#[test]
fn transvoxel_classifier_layouts_match_wgsl_exactly() {
    let module =
        wgsl::parse_str(TRANSVOXEL_CLASSIFY_WGSL).expect("Transvoxel classify WGSL parses");
    Validator::new(ValidationFlags::all(), Capabilities::all())
        .validate(&module)
        .expect("Transvoxel classify WGSL validates");

    assert_eq!(align_of::<GpuTransvoxelDispatch>(), 16);
    assert_eq!(size_of::<GpuTransvoxelDispatch>(), 48);
    assert_eq!(
        wgsl_struct_in(TRANSVOXEL_CLASSIFY_WGSL, "GpuTransvoxelDispatch"),
        (
            48,
            vec![
                ("dirty_microbricks_low".into(), 0),
                ("dirty_microbricks_high".into(), 4),
                ("generation_low".into(), 8),
                ("generation_high".into(), 12),
                ("cell_count".into(), 16),
                ("max_vertices".into(), 20),
                ("max_indices".into(), 24),
                ("scan_block_count".into(), 28),
                ("transition_mask".into(), 32),
                ("_pad0".into(), 36),
                ("_pad1".into(), 40),
                ("_pad2".into(), 44),
            ],
        )
    );

    assert_eq!(align_of::<GpuTransvoxelCell>(), 16);
    assert_eq!(size_of::<GpuTransvoxelCell>(), 16);
    assert_eq!(
        wgsl_struct_in(TRANSVOXEL_CLASSIFY_WGSL, "GpuTransvoxelCell"),
        (
            16,
            vec![
                ("packed_case_class_counts".into(), 0),
                ("generation_low".into(), 4),
                ("generation_high".into(), 8),
                ("_pad".into(), 12),
            ],
        )
    );

    assert_eq!(align_of::<GpuTransvoxelClassifyCounters>(), 16);
    assert_eq!(size_of::<GpuTransvoxelClassifyCounters>(), 16);
    assert_eq!(
        wgsl_struct_in(TRANSVOXEL_CLASSIFY_WGSL, "GpuTransvoxelClassifyCounters"),
        (
            16,
            vec![
                ("visited_cells".into(), 0),
                ("active_cells".into(), 4),
                ("vertices".into(), 8),
                ("triangles".into(), 12),
            ],
        )
    );
}

#[test]
fn transvoxel_emission_layouts_match_wgsl_exactly() {
    let module = wgsl::parse_str(TRANSVOXEL_EMIT_WGSL).expect("Transvoxel emission WGSL parses");
    Validator::new(ValidationFlags::all(), Capabilities::all())
        .validate(&module)
        .expect("Transvoxel emission WGSL validates");

    assert_eq!(align_of::<GpuTransvoxelCellOffset>(), 16);
    assert_eq!(size_of::<GpuTransvoxelCellOffset>(), 16);
    assert_eq!(
        wgsl_struct_in(TRANSVOXEL_EMIT_WGSL, "GpuTransvoxelCellOffset"),
        (
            16,
            vec![
                ("first_vertex".into(), 0),
                ("first_index".into(), 4),
                ("generation_low".into(), 8),
                ("generation_high".into(), 12),
            ],
        )
    );

    assert_eq!(align_of::<GpuTransvoxelScanBlock>(), 16);
    assert_eq!(size_of::<GpuTransvoxelScanBlock>(), 16);
    assert_eq!(
        wgsl_struct_in(TRANSVOXEL_EMIT_WGSL, "GpuTransvoxelScanBlock"),
        (
            16,
            vec![
                ("vertex_count".into(), 0),
                ("index_count".into(), 4),
                ("first_vertex".into(), 8),
                ("first_index".into(), 12),
            ],
        )
    );

    assert_eq!(align_of::<GpuTransvoxelEmissionCounters>(), 16);
    assert_eq!(size_of::<GpuTransvoxelEmissionCounters>(), 32);
    assert_eq!(
        wgsl_struct_in(TRANSVOXEL_EMIT_WGSL, "GpuTransvoxelEmissionCounters"),
        (
            32,
            vec![
                ("required_vertices".into(), 0),
                ("required_indices".into(), 4),
                ("emitted_vertices".into(), 8),
                ("emitted_indices".into(), 12),
                ("vertex_overflow".into(), 16),
                ("index_overflow".into(), 20),
                ("completed".into(), 24),
                ("_pad".into(), 28),
            ],
        )
    );

    assert_eq!(
        wgsl_struct_in(TRANSVOXEL_EMIT_WGSL, "GpuTerrainVertex"),
        (
            size_of::<GpuTerrainVertex>() as u32,
            vec![
                ("position".into(), 0),
                ("material".into(), 12),
                ("normal".into(), 16),
                ("flags".into(), 28),
            ],
        )
    );
}

#[test]
fn transvoxel_transition_gpu_layouts_match_wgsl_exactly() {
    let module = wgsl::parse_str(TRANSVOXEL_TRANSITION_GPU_WGSL)
        .expect("Transvoxel transition GPU WGSL parses");
    Validator::new(ValidationFlags::all(), Capabilities::all())
        .validate(&module)
        .expect("Transvoxel transition GPU WGSL validates");

    assert_eq!(align_of::<GpuTransvoxelTransitionDispatch>(), 16);
    assert_eq!(size_of::<GpuTransvoxelTransitionDispatch>(), 32);
    assert_eq!(
        wgsl_struct_in(
            TRANSVOXEL_TRANSITION_GPU_WGSL,
            "GpuTransvoxelTransitionDispatch"
        ),
        (
            32,
            vec![
                ("transition_mask".into(), 0),
                ("generation_low".into(), 4),
                ("generation_high".into(), 8),
                ("cell_count".into(), 12),
                ("max_vertices".into(), 16),
                ("max_indices".into(), 20),
                ("scan_block_count".into(), 24),
                ("_pad".into(), 28),
            ],
        )
    );

    assert_eq!(align_of::<GpuTransvoxelTransitionCell>(), 16);
    assert_eq!(size_of::<GpuTransvoxelTransitionCell>(), 16);
    assert_eq!(
        wgsl_struct_in(
            TRANSVOXEL_TRANSITION_GPU_WGSL,
            "GpuTransvoxelTransitionCell"
        ),
        (
            16,
            vec![
                ("packed_case_class_counts".into(), 0),
                ("generation_low".into(), 4),
                ("generation_high".into(), 8),
                ("_pad".into(), 12),
            ],
        )
    );

    assert_eq!(align_of::<GpuTransvoxelTransitionCounters>(), 16);
    assert_eq!(size_of::<GpuTransvoxelTransitionCounters>(), 48);
    assert_eq!(
        wgsl_struct_in(
            TRANSVOXEL_TRANSITION_GPU_WGSL,
            "GpuTransvoxelTransitionCounters"
        ),
        (
            48,
            vec![
                ("active_cells".into(), 0),
                ("active_faces".into(), 4),
                ("required_vertices".into(), 8),
                ("required_indices".into(), 12),
                ("emitted_vertices".into(), 16),
                ("emitted_indices".into(), 20),
                ("vertex_overflow".into(), 24),
                ("index_overflow".into(), 28),
                ("completed".into(), 32),
                ("_pad".into(), 36),
            ],
        )
    );
}
