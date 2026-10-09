//! The rows fog volumes, post-process volumes, camera post-process
//! settings, water volumes, foliage, atmospheres and decals cast, as the renderer's environment join reads them
//! (Pulsar-Native#1035, Phase 4).
//!
//! Each is a second GPU registration on its authored component: SceneDB
//! derives the row from the authored value on every insert, guarded write,
//! removal, despawn and mirror replay, keyed by the component instance
//! entity. A row is the pass's own row in the component's local space: the
//! graph-owned join (`helio-default-graphs::environment_join`) gates it on
//! the instance's enabled state, its owner's generation and (for volumes)
//! visibility, and places volumes with the owner's transform, on the GPU.
//! A disabled component derives an inert row (its pass skips it).
//!
//! Layouts, in `u32` words (`shaders/environment_join.wgsl` copies the
//! payload after an optional local-size header):
//!
//! | Row | Words | Pass row it becomes |
//! |---|---|---|
//! | [`GlobalFogSourceRow`] | 16: the medium | `helio_pass_volumetric_fog::GlobalFogComponent` (16) |
//! | [`LocalFogSourceRow`] | 4 size + 20: medium, edge fade | `LocalFogVolumeComponent` (8 bounds + 20) |
//! | [`PostProcessVolumeSourceRow`] | 4 size + 156: the row after its bounds | `helio_pass_postprocess::PostProcessVolumeComponent` (8 bounds + 156) |
//! | [`CameraPostProcessSourceRow`] | 152: the row | `helio_pass_postprocess::CameraPostProcessComponent` (152) |
//! | [`WaterVolumeSourceRow`] | 4 size + 56: the row after its bounds | `helio_pass_water_sim::WaterVolumeComponent` (8 bounds + 56), packed into its leading rows |
//! | [`FoliageSourceRow`] | 24 type + 4 layer + 12 wind | `helio_pass_foliage_place`'s `FoliageTypeComponent` (24), `FoliageLayerComponent` (8) and `FoliageWindComponent` (12), each packed |
//! | [`AtmosphereSourceRow`] | 4 centre + 16 media + 4 ground + 4 shell: the row, its centre in the owner's space | `helio_pass_sky::AtmosphereComponent` (28) |
//! | [`DecalSourceRow`] | 4 size + 32: the row, its transform left to the join | `helio_pass_decal::DecalComponent` (32), packed into its leading rows |

use pulsar_scenedb::gpu::GpuMirrorHandle;
use pulsar_scenedb_derive::SceneStore;

use super::{
    AtmosphereComponent, CameraPostProcessComponent, DecalComponent, FoliageComponent, GlobalFogComponent, LocalFogVolumeComponent,
    PostProcessVolumeComponent, WaterVolumeComponent,
};

pub const GLOBAL_FOG_SOURCES_BUFFER: &str = "global_fog_sources";
pub const LOCAL_FOG_SOURCES_BUFFER: &str = "local_fog_sources";
pub const POST_PROCESS_VOLUME_SOURCES_BUFFER: &str = "post_process_volume_sources";
pub const CAMERA_POST_PROCESS_SOURCES_BUFFER: &str = "camera_postprocess_sources";
pub const WATER_VOLUME_SOURCES_BUFFER: &str = "water_volume_sources";
pub const FOLIAGE_SOURCES_BUFFER: &str = "foliage_sources";
pub const ATMOSPHERE_SOURCES_BUFFER: &str = "atmosphere_sources";
pub const DECAL_SOURCES_BUFFER: &str = "decal_sources";

/// A decal: its box's local size (`xyz`, full extent before the owner's
/// scale), then the decal pass row with a zero transform, which the join
/// replaces with the world-to-decal transform of the owner-placed box. A
/// disabled decal has a zero size, which the join skips.
#[derive(SceneStore, bytemuck::Pod, bytemuck::Zeroable, Clone, Copy, Debug, PartialEq)]
#[repr(C)]
#[gpu(layout = packed, buffer = "decal_sources")]
pub struct DecalSourceRow {
    #[gpu]
    pub size: [f32; 4],
    /// The pass row's transform: zero here, the join's to write.
    #[gpu]
    pub transform: [f32; 16],
    /// The rest of the pass row: colour and opacity, texture slots, blend,
    /// type, fade, age and normal adaptation.
    #[gpu]
    pub params: [f32; 16],
}

impl DecalSourceRow {
    /// The row of `decal` with its albedo texture at `albedo_slot`
    /// (`u32::MAX`: none).
    pub fn of(decal: &DecalComponent, albedo_slot: u32) -> Self {
        if !decal.enabled {
            return bytemuck::Zeroable::zeroed();
        }
        let size = [decal.size[0], decal.size[1], decal.size[2], 0.0];
        read(
            &[
                bytemuck::bytes_of(&size),
                bytemuck::bytes_of(&decal.to_row(albedo_slot)),
            ]
            .concat(),
        )
    }
}

/// An atmosphere (`helio_pass_sky::AtmosphereComponent` pass row, bit for
/// bit): its centre and placement (`center`; the centre in the owner's
/// space, zero: the join adds the owner's position to a planet placed at its
/// owner), Rayleigh and Mie scattering, Mie and ozone absorption (`media`),
/// ground albedo and ozone width (`ground`), and planet and atmosphere
/// radii, sun size and enabled flag (`shell`; `enabled` 0 when the authored
/// component is disabled). The placement and enabled words are `u32`s, kept
/// as bits.
#[derive(SceneStore, bytemuck::Pod, bytemuck::Zeroable, Clone, Copy, Debug, PartialEq)]
#[repr(C)]
#[gpu(layout = packed, buffer = "atmosphere_sources")]
pub struct AtmosphereSourceRow {
    #[gpu]
    pub center: [f32; 4],
    #[gpu]
    pub media: [f32; 16],
    #[gpu]
    pub ground: [f32; 4],
    #[gpu]
    pub shell: [f32; 4],
}

impl AtmosphereSourceRow {
    pub fn of(atmosphere: &AtmosphereComponent) -> Self {
        read(bytemuck::bytes_of(&atmosphere.to_row()))
    }
}

/// A global fog medium (`GlobalFogComponent` pass row, bit for bit;
/// `enabled` 0 when the authored component is disabled).
#[derive(SceneStore, bytemuck::Pod, bytemuck::Zeroable, Clone, Copy, Debug, PartialEq)]
#[repr(C)]
#[gpu(layout = packed, buffer = "global_fog_sources")]
pub struct GlobalFogSourceRow {
    #[gpu]
    pub medium: [f32; 16],
}

/// A local fog volume: its local size (`xyz`, full extent before the owner's
/// scale), then the pass row's medium and edge fade (`x`).
#[derive(SceneStore, bytemuck::Pod, bytemuck::Zeroable, Clone, Copy, Debug, PartialEq)]
#[repr(C)]
#[gpu(layout = packed, buffer = "local_fog_sources")]
pub struct LocalFogSourceRow {
    #[gpu]
    pub size: [f32; 4],
    #[gpu]
    pub medium: [f32; 16],
    #[gpu]
    pub edge_fade: [f32; 4],
}

/// A post-process volume: its local size, then the pass row after its
/// bounds: priority, blend radius, blend weight, unbound flag (`params`, as
/// bits), override mask, settings and lens blocks.
#[derive(SceneStore, bytemuck::Pod, bytemuck::Zeroable, Clone, Copy, Debug, PartialEq)]
#[repr(C)]
#[gpu(layout = packed, buffer = "post_process_volume_sources")]
pub struct PostProcessVolumeSourceRow {
    #[gpu]
    pub size: [f32; 4],
    #[gpu]
    pub params: [u32; 4],
    #[gpu]
    pub mask: [u32; 4],
    #[gpu]
    pub settings: [u32; 116],
    #[gpu]
    pub lens: [f32; 16],
    #[gpu]
    pub lens_ext: [f32; 16],
}

/// A camera's post-process baseline (`CameraPostProcessComponent` pass row,
/// bit for bit: view id and enabled flag in `header`; `enabled` 0 when the
/// authored component is disabled).
#[derive(SceneStore, bytemuck::Pod, bytemuck::Zeroable, Clone, Copy, Debug, PartialEq)]
#[repr(C)]
#[gpu(layout = packed, buffer = "camera_postprocess_sources")]
pub struct CameraPostProcessSourceRow {
    #[gpu]
    pub header: [u32; 4],
    #[gpu]
    pub settings: [u32; 116],
    #[gpu]
    pub lens: [f32; 16],
    #[gpu]
    pub lens_ext: [f32; 16],
}

/// A water volume: its local size (`xyz`, full extent before the owner's
/// scale; `w`, the surface height above the owner's position), then the
/// water volume row after its bounds: waves (wave params, direction,
/// colour, extinction), optics (reflection and refraction, caustics, fog,
/// surface params), shading (shadow, sun direction, SSR, simulation
/// dynamics), wind and a reserved vec4. A disabled component has a zero
/// size, which the join skips.
#[derive(SceneStore, bytemuck::Pod, bytemuck::Zeroable, Clone, Copy, Debug, PartialEq)]
#[repr(C)]
#[gpu(layout = packed, buffer = "water_volume_sources")]
pub struct WaterVolumeSourceRow {
    #[gpu]
    pub size: [f32; 4],
    #[gpu]
    pub waves: [f32; 16],
    #[gpu]
    pub optics: [f32; 16],
    #[gpu]
    pub shading: [f32; 16],
    #[gpu]
    pub wind: [f32; 4],
    #[gpu]
    pub reserved: [f32; 4],
}

impl WaterVolumeSourceRow {
    pub fn of(water: &WaterVolumeComponent) -> Self {
        if !water.enabled {
            return bytemuck::Zeroable::zeroed();
        }
        let size = [
            water.size[0],
            water.size[1],
            water.size[2],
            water.surface_height_offset,
        ];
        let gpu = water.local_gpu();
        // The pass row after its two bound vec4s.
        let after_bounds = &bytemuck::bytes_of(&gpu)[32..];
        read(&[bytemuck::bytes_of(&size), after_bounds].concat())
    }
}

/// Foliage: the foliage type row (density, size, slope and altitude ranges,
/// LOD distances and wind response in `type_params`; the interaction
/// stiffness, material, density layer and kind/flags words in `type_ids`;
/// the impostor id and padding in `type_tail`), its layer (half extent,
/// infinite-extent flag, altitude min and max) and its wind row. A disabled
/// component derives a zero row, which the join skips (it places only rows
/// with a density).
///
/// The layer is a world-aligned square centred on the owner. The foliage
/// passes grow every type in every layer and read the first wind row only,
/// so with several foliage components their types share their layers and
/// the first component's wind applies to all of them.
#[derive(SceneStore, bytemuck::Pod, bytemuck::Zeroable, Clone, Copy, Debug, PartialEq)]
#[repr(C)]
#[gpu(layout = packed, buffer = "foliage_sources")]
pub struct FoliageSourceRow {
    #[gpu]
    pub type_params: [f32; 16],
    #[gpu]
    pub type_ids: [u32; 4],
    #[gpu]
    pub type_tail: [u32; 4],
    #[gpu]
    pub layer: [f32; 4],
    #[gpu]
    pub wind: [f32; 12],
}

impl FoliageSourceRow {
    pub fn of(foliage: &FoliageComponent) -> Self {
        if !foliage.general.enabled {
            return bytemuck::Zeroable::zeroed();
        }
        let foliage_type = super::foliage_component::runtime::gpu_type(foliage);
        let layer = [
            foliage.placement.layer_extent,
            if foliage.placement.has_infinite_extent {
                1.0
            } else {
                0.0
            },
            foliage.placement.altitude_min,
            foliage.placement.altitude_max,
        ];
        let wind = super::foliage_component::runtime::wind(foliage);
        read(
            &[
                bytemuck::bytes_of(&foliage_type),
                bytemuck::bytes_of(&layer),
                bytemuck::bytes_of(&wind),
            ]
            .concat(),
        )
    }
}

/// `bytes`, which must be exactly a `T`, as a `T`.
fn read<T: bytemuck::Pod>(bytes: &[u8]) -> T {
    bytemuck::pod_read_unaligned(bytes)
}

impl GlobalFogSourceRow {
    pub fn of(fog: &GlobalFogComponent) -> Self {
        let mut medium = fog.medium.to_medium();
        medium.enabled = u32::from(fog.enabled);
        read(bytemuck::bytes_of(&medium))
    }
}

impl LocalFogSourceRow {
    pub fn of(fog: &LocalFogVolumeComponent) -> Self {
        let mut medium = fog.medium.to_medium();
        medium.enabled = u32::from(fog.enabled);
        Self {
            size: [fog.size[0], fog.size[1], fog.size[2], 0.0],
            medium: read(bytemuck::bytes_of(&medium)),
            edge_fade: [fog.edge_fade.max(0.0), 0.0, 0.0, 0.0],
        }
    }
}

impl PostProcessVolumeSourceRow {
    pub fn of(volume: &PostProcessVolumeComponent) -> Self {
        let size = [volume.size[0], volume.size[1], volume.size[2], 0.0];
        let mut row: Self = bytemuck::Zeroable::zeroed();
        if volume.enabled {
            let pass = helio_pass_postprocess::PostProcessVolumeComponent::from(
                volume.local_descriptor().to_gpu(),
            );
            // The pass row after its two bound vec4s.
            let after_bounds = &bytemuck::bytes_of(&pass)[32..];
            row = read(&[bytemuck::bytes_of(&size), after_bounds].concat());
        }
        row.size = size;
        row
    }
}

impl CameraPostProcessSourceRow {
    pub fn of(camera: &CameraPostProcessComponent) -> Self {
        let mut row = helio_pass_postprocess::CameraPostProcessComponent::new(
            camera.view_id,
            &camera.settings.to_settings(),
        );
        row.enabled = u32::from(camera.enabled);
        read(bytemuck::bytes_of(&row))
    }
}

macro_rules! derived_row {
    ($authored:ty, $row:ty, $dispatch:ident, $clear:ident) => {
        fn $dispatch(mirror: &GpuMirrorHandle, row: u32, data: *const (), is_new_insert: bool) {
            // SAFETY: SceneDB reaches this only through the authored
            // component's own `ComponentId`, with a pointer to a live value.
            let authored = unsafe { &*(data as *const $authored) };
            pulsar_scenedb::gpu::write_derived_row(
                mirror,
                row,
                &<$row>::of(authored),
                is_new_insert,
            );
        }

        fn $clear(mirror: &GpuMirrorHandle, row: u32) {
            pulsar_scenedb::gpu::clear_derived_row::<$row>(mirror, row);
        }

        pulsar_scenedb::pulsar_reflection::inventory::submit! {
            pulsar_scenedb::gpu::GpuMirrorRegistration {
                component_id: pulsar_scenedb::component_id::<$authored>,
                dispatch: $dispatch,
            }
        }

        pulsar_scenedb::pulsar_reflection::inventory::submit! {
            pulsar_scenedb::gpu::GpuClearRegistration {
                component_id: pulsar_scenedb::component_id::<$authored>,
                clear: $clear,
            }
        }
    };
}

derived_row!(
    GlobalFogComponent,
    GlobalFogSourceRow,
    global_fog_dispatch,
    global_fog_clear
);
derived_row!(
    LocalFogVolumeComponent,
    LocalFogSourceRow,
    local_fog_dispatch,
    local_fog_clear
);
derived_row!(
    PostProcessVolumeComponent,
    PostProcessVolumeSourceRow,
    post_process_volume_dispatch,
    post_process_volume_clear
);
derived_row!(
    CameraPostProcessComponent,
    CameraPostProcessSourceRow,
    camera_post_process_dispatch,
    camera_post_process_clear
);
derived_row!(
    FoliageComponent,
    FoliageSourceRow,
    foliage_dispatch,
    foliage_clear
);
derived_row!(
    AtmosphereComponent,
    AtmosphereSourceRow,
    atmosphere_dispatch,
    atmosphere_clear
);
// A decal's image is registered in the scene's texture store as its row is
// derived, so the row carries its bindless slot.
fn decal_dispatch(mirror: &GpuMirrorHandle, row: u32, data: *const (), is_new_insert: bool) {
    // SAFETY: SceneDB reaches this only through `DecalComponent`'s own
    // `ComponentId`, with a pointer to a live value.
    let decal = unsafe { &*(data as *const DecalComponent) };
    let albedo_slot = decal_texture_slot(&decal.albedo_texture, mirror);
    pulsar_scenedb::gpu::write_derived_row(
        mirror,
        row,
        &DecalSourceRow::of(decal, albedo_slot),
        is_new_insert,
    );
}

/// The texture store slot of the image at `path` (project-relative or
/// absolute), or `u32::MAX` (the tint alone) for none or one that does not
/// load.
fn decal_texture_slot(path: &str, mirror: &GpuMirrorHandle) -> u32 {
    if path.is_empty() {
        return u32::MAX;
    }
    let mut resolved = std::path::PathBuf::from(path);
    if resolved.is_relative() {
        if let Some(root) = engine_state::get_project_path() {
            resolved = std::path::Path::new(&root).join(path);
        }
    }
    match crate::material_textures::register_graph_texture(&resolved, mirror) {
        Ok(slot) => slot,
        Err(error) => {
            tracing::warn!("decal texture '{}': {error}", resolved.display());
            u32::MAX
        }
    }
}

fn decal_clear(mirror: &GpuMirrorHandle, row: u32) {
    pulsar_scenedb::gpu::clear_derived_row::<DecalSourceRow>(mirror, row);
}

pulsar_scenedb::pulsar_reflection::inventory::submit! {
    pulsar_scenedb::gpu::GpuMirrorRegistration {
        component_id: pulsar_scenedb::component_id::<DecalComponent>,
        dispatch: decal_dispatch,
    }
}

pulsar_scenedb::pulsar_reflection::inventory::submit! {
    pulsar_scenedb::gpu::GpuClearRegistration {
        component_id: pulsar_scenedb::component_id::<DecalComponent>,
        clear: decal_clear,
    }
}

derived_row!(
    WaterVolumeComponent,
    WaterVolumeSourceRow,
    water_volume_dispatch,
    water_volume_clear
);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rows_have_the_documented_word_counts() {
        assert_eq!(std::mem::size_of::<GlobalFogSourceRow>(), 16 * 4);
        assert_eq!(std::mem::size_of::<LocalFogSourceRow>(), 24 * 4);
        assert_eq!(std::mem::size_of::<PostProcessVolumeSourceRow>(), 160 * 4);
        assert_eq!(std::mem::size_of::<CameraPostProcessSourceRow>(), 152 * 4);
        assert_eq!(std::mem::size_of::<WaterVolumeSourceRow>(), 60 * 4);
        assert_eq!(std::mem::size_of::<FoliageSourceRow>(), 40 * 4);
        assert_eq!(std::mem::size_of::<AtmosphereSourceRow>(), 28 * 4);
        assert_eq!(std::mem::size_of::<DecalSourceRow>(), 36 * 4);
        assert_eq!(
            std::mem::size_of::<helio_pass_decal::DecalComponent>(),
            32 * 4
        );
        assert_eq!(
            std::mem::size_of::<helio_pass_sky::AtmosphereComponent>(),
            28 * 4
        );
        assert_eq!(
            std::mem::size_of::<helio_pass_foliage_place::components::FoliageTypeComponent>(),
            24 * 4
        );
        assert_eq!(
            std::mem::size_of::<helio_pass_foliage_place::components::FoliageWindComponent>(),
            12 * 4
        );
        assert_eq!(
            std::mem::size_of::<helio_pass_water_sim::WaterVolumeComponent>(),
            64 * 4
        );
        assert_eq!(
            std::mem::size_of::<helio_pass_volumetric_fog::GlobalFogComponent>(),
            16 * 4
        );
        assert_eq!(
            std::mem::size_of::<helio_pass_volumetric_fog::LocalFogVolumeComponent>(),
            28 * 4
        );
        assert_eq!(
            std::mem::size_of::<helio_pass_postprocess::PostProcessVolumeComponent>(),
            164 * 4
        );
        assert_eq!(
            std::mem::size_of::<helio_pass_postprocess::CameraPostProcessComponent>(),
            152 * 4
        );
    }

    #[test]
    fn disabled_components_derive_inert_rows() {
        let enabled = |row: &GlobalFogSourceRow| row.medium[0].to_bits();
        let mut fog = GlobalFogComponent::default();
        assert_eq!(enabled(&GlobalFogSourceRow::of(&fog)), 1);
        fog.enabled = false;
        assert_eq!(enabled(&GlobalFogSourceRow::of(&fog)), 0);

        let mut camera = CameraPostProcessComponent::default();
        camera.view_id = 7;
        let row = CameraPostProcessSourceRow::of(&camera);
        assert_eq!((row.header[0], row.header[1]), (7, 1));
        camera.enabled = false;
        assert_eq!(CameraPostProcessSourceRow::of(&camera).header[1], 0);

        let mut volume = PostProcessVolumeComponent::default();
        volume.enabled = false;
        let row = PostProcessVolumeSourceRow::of(&volume);
        assert!(bytemuck::bytes_of(&row)[16..].iter().all(|byte| *byte == 0));

        let mut water = WaterVolumeComponent::default();
        water.enabled = false;
        let row = WaterVolumeSourceRow::of(&water);
        assert!(bytemuck::bytes_of(&row).iter().all(|byte| *byte == 0));

        let mut decal = DecalComponent::default();
        assert_eq!(
            DecalSourceRow::of(&decal, u32::MAX).size,
            [1.0, 1.0, 1.0, 0.0]
        );
        decal.enabled = false;
        let row = DecalSourceRow::of(&decal, u32::MAX);
        assert!(bytemuck::bytes_of(&row).iter().all(|byte| *byte == 0));

        let mut atmosphere = AtmosphereComponent::default();
        assert_eq!(AtmosphereSourceRow::of(&atmosphere).shell[3].to_bits(), 1);
        atmosphere.enabled = false;
        assert_eq!(AtmosphereSourceRow::of(&atmosphere).shell[3].to_bits(), 0);
    }

    #[test]
    fn an_atmosphere_row_is_the_pass_row_in_its_owners_space() {
        let atmosphere = AtmosphereComponent {
            placement: super::super::AtmospherePlacement::PlanetAtOwner,
            planet_radius_km: 6360.0,
            ..Default::default()
        };
        let row = AtmosphereSourceRow::of(&atmosphere);
        assert_eq!(
            bytemuck::bytes_of(&row),
            bytemuck::bytes_of(&atmosphere.to_row())
        );
        assert_eq!(&row.center[..3], &[0.0; 3], "the join adds the owner's position");
        assert_eq!(row.center[3].to_bits(), helio_pass_sky::atmosphere::placement::CENTER);
    }

    #[test]
    fn a_foliage_row_holds_its_type_layer_and_wind() {
        let mut foliage = FoliageComponent::default();
        foliage.general.enabled = true;
        foliage.general.density = 12.0;
        foliage.placement.layer_extent = 25.0;
        let row = FoliageSourceRow::of(&foliage);
        let foliage_type = super::super::foliage_component::runtime::gpu_type(&foliage);
        assert_eq!(
            &bytemuck::bytes_of(&row)[..96],
            bytemuck::bytes_of(&foliage_type)
        );
        assert_eq!(row.type_params[0], 12.0, "density first: the join's gate");
        assert_eq!(row.layer[0], 25.0);
        foliage.general.enabled = false;
        let row = FoliageSourceRow::of(&foliage);
        assert!(bytemuck::bytes_of(&row).iter().all(|byte| *byte == 0));
    }

    #[test]
    fn a_water_row_is_the_pass_row_after_its_bounds() {
        let mut water = WaterVolumeComponent::default();
        water.surface_height_offset = 2.5;
        let row = WaterVolumeSourceRow::of(&water);
        assert_eq!(
            &bytemuck::bytes_of(&row)[16..],
            &bytemuck::bytes_of(&water.local_gpu())[32..]
        );
        assert_eq!(row.size, [200.0, 60.0, 200.0, 2.5]);
    }

    #[test]
    fn a_post_process_volume_row_is_the_pass_row_after_its_bounds() {
        let volume = PostProcessVolumeComponent::default();
        let row = PostProcessVolumeSourceRow::of(&volume);
        let pass = helio_pass_postprocess::PostProcessVolumeComponent::from(
            volume.local_descriptor().to_gpu(),
        );
        assert_eq!(
            &bytemuck::bytes_of(&row)[16..],
            &bytemuck::bytes_of(&pass)[32..]
        );
        assert_eq!(row.size[..3], volume.size);
    }
}
