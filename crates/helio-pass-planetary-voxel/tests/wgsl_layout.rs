use helio_pass_planetary_voxel::{
    GpuLookupQuery, GpuLookupResult, GpuPageTableEntry, GpuResidencyCounters, GpuResidencyUniform,
    RESIDENCY_WGSL,
};
use std::mem::{align_of, offset_of, size_of};
use wgpu::naga::{
    front::wgsl,
    valid::{Capabilities, ValidationFlags, Validator},
    TypeInner,
};

fn wgsl_struct(name: &str) -> (u32, Vec<(String, u32)>) {
    let module = wgsl::parse_str(RESIDENCY_WGSL)
        .unwrap_or_else(|error| panic!("planetary residency WGSL must parse: {error}"));
    Validator::new(ValidationFlags::all(), Capabilities::all())
        .validate(&module)
        .unwrap_or_else(|error| panic!("planetary residency WGSL must validate: {error}"));
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
fn residency_shader_parses_and_validates() {
    let module = wgsl::parse_str(RESIDENCY_WGSL).expect("residency WGSL parses");
    Validator::new(ValidationFlags::all(), Capabilities::all())
        .validate(&module)
        .expect("residency WGSL validates");
}

#[test]
fn page_table_entry_matches_wgsl_exactly() {
    assert_eq!(align_of::<GpuPageTableEntry>(), 16);
    assert_eq!(size_of::<GpuPageTableEntry>(), 48);
    assert_eq!(
        wgsl_struct("GpuPageTableEntry"),
        (
            48,
            vec![
                (
                    "planet_id".into(),
                    offset_of!(GpuPageTableEntry, planet_id) as u32
                ),
                (
                    "relative_lod0_cell_min".into(),
                    offset_of!(GpuPageTableEntry, relative_lod0_cell_min) as u32,
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
            ],
        )
    );
}

#[test]
fn uniform_query_and_result_match_wgsl_exactly() {
    assert_eq!(align_of::<GpuResidencyUniform>(), 16);
    assert_eq!(size_of::<GpuResidencyUniform>(), 16);
    assert_eq!(
        wgsl_struct("GpuResidencyUniform"),
        (
            16,
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
                ("_pad".into(), offset_of!(GpuResidencyUniform, _pad) as u32),
            ],
        )
    );

    assert_eq!(align_of::<GpuLookupQuery>(), 16);
    assert_eq!(size_of::<GpuLookupQuery>(), 32);
    assert_eq!(
        wgsl_struct("GpuLookupQuery"),
        (
            32,
            vec![
                (
                    "planet_id".into(),
                    offset_of!(GpuLookupQuery, planet_id) as u32
                ),
                (
                    "relative_lod0_cell_min".into(),
                    offset_of!(GpuLookupQuery, relative_lod0_cell_min) as u32,
                ),
                ("lod".into(), offset_of!(GpuLookupQuery, lod) as u32),
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
                ("atlas_shards".into(), 80),
                ("device_rebuilds".into(), 84),
                ("_pad".into(), 88),
            ],
        )
    );
}
