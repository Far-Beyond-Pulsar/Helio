use helio_planet_voxel_core::{
    GpuPageMeta, GpuVoxelMaterial, PlanetFrameUniform, PLANET_VOXEL_LAYOUT_WGSL,
};
use std::mem::{align_of, offset_of, size_of};
use wgpu::naga::{
    front::wgsl,
    valid::{Capabilities, ValidationFlags, Validator},
    TypeInner,
};

fn wgsl_struct(name: &str) -> (u32, Vec<(String, u32)>) {
    let module = wgsl::parse_str(PLANET_VOXEL_LAYOUT_WGSL)
        .unwrap_or_else(|error| panic!("planetary layout WGSL must parse: {error}"));
    Validator::new(ValidationFlags::all(), Capabilities::all())
        .validate(&module)
        .unwrap_or_else(|error| panic!("planetary layout WGSL must validate: {error}"));
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
                    member.name.clone().expect("every contract field is named"),
                    member.offset,
                )
            })
            .collect(),
    )
}

#[test]
fn planet_frame_uniform_matches_wgsl_offsets_exactly() {
    assert_eq!(align_of::<PlanetFrameUniform>(), 16);
    assert_eq!(size_of::<PlanetFrameUniform>(), 80);
    assert_eq!(
        wgsl_struct("PlanetFrameUniform"),
        (
            size_of::<PlanetFrameUniform>() as u32,
            vec![
                (
                    "planet_id".into(),
                    offset_of!(PlanetFrameUniform, planet_id) as u32
                ),
                (
                    "origin_x".into(),
                    offset_of!(PlanetFrameUniform, origin_x) as u32
                ),
                (
                    "origin_y".into(),
                    offset_of!(PlanetFrameUniform, origin_y) as u32
                ),
                (
                    "origin_z".into(),
                    offset_of!(PlanetFrameUniform, origin_z) as u32
                ),
                (
                    "frame_index".into(),
                    offset_of!(PlanetFrameUniform, frame_index) as u32
                ),
                (
                    "camera_relative_m".into(),
                    offset_of!(PlanetFrameUniform, camera_relative_m) as u32,
                ),
                (
                    "lod0_cell_size_m".into(),
                    offset_of!(PlanetFrameUniform, lod0_cell_size_m) as u32,
                ),
                (
                    "page_edge_cells".into(),
                    offset_of!(PlanetFrameUniform, page_edge_cells) as u32,
                ),
                ("_pad".into(), offset_of!(PlanetFrameUniform, _pad) as u32),
            ],
        )
    );
}

#[test]
fn page_meta_matches_wgsl_offsets_exactly() {
    assert_eq!(align_of::<GpuPageMeta>(), 16);
    assert_eq!(size_of::<GpuPageMeta>(), 32);
    assert_eq!(
        wgsl_struct("GpuPageMeta"),
        (
            size_of::<GpuPageMeta>() as u32,
            vec![
                (
                    "relative_lod0_cell_min".into(),
                    offset_of!(GpuPageMeta, relative_lod0_cell_min) as u32,
                ),
                ("lod".into(), offset_of!(GpuPageMeta, lod) as u32),
                ("slot".into(), offset_of!(GpuPageMeta, slot) as u32),
                (
                    "generation_low".into(),
                    offset_of!(GpuPageMeta, generation_low) as u32,
                ),
                (
                    "generation_high".into(),
                    offset_of!(GpuPageMeta, generation_high) as u32,
                ),
                (
                    "transition_mask".into(),
                    offset_of!(GpuPageMeta, transition_mask) as u32,
                ),
            ],
        )
    );
}

#[test]
fn voxel_material_matches_wgsl_offsets_exactly() {
    assert_eq!(align_of::<GpuVoxelMaterial>(), 16);
    assert_eq!(size_of::<GpuVoxelMaterial>(), 32);
    assert_eq!(
        wgsl_struct("GpuVoxelMaterial"),
        (
            size_of::<GpuVoxelMaterial>() as u32,
            vec![
                (
                    "base_color_roughness".into(),
                    offset_of!(GpuVoxelMaterial, base_color_roughness) as u32,
                ),
                (
                    "emissive_metalness".into(),
                    offset_of!(GpuVoxelMaterial, emissive_metalness) as u32,
                ),
            ],
        )
    );
}
