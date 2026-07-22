use crate::{
    AddressError, PageKey, PlanetId, PlanetPosition, PlanetRenderFrame, LOD0_CELL_SIZE_METERS,
    TRANSITION_FACE_MASK,
};
use bytemuck::{Pod, Zeroable};

/// Frame-stable values consumed only by the planetary pass. Absolute canonical
/// values are split into integer words; existing Helio transforms and camera
/// uniforms remain camera-local `f32` data.
#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
pub struct PlanetFrameUniform {
    pub planet_id: [u32; 4],
    pub origin_x: [u32; 2],
    pub origin_y: [u32; 2],
    pub origin_z: [u32; 2],
    pub frame_index: [u32; 2],
    pub camera_relative_m: [f32; 3],
    pub lod0_cell_size_m: f32,
    pub page_edge_cells: u32,
    pub _pad: [u32; 3],
}

impl PlanetFrameUniform {
    pub fn from_render_frame(frame: PlanetRenderFrame) -> Self {
        let frame_origin_lod0_cell = frame.origin_lod0_cell();
        let camera_relative_m = frame.camera_relative_m().map(|value| value as f32);
        Self {
            planet_id: planet_words(frame.planet()),
            origin_x: split_i64(frame_origin_lod0_cell[0]),
            origin_y: split_i64(frame_origin_lod0_cell[1]),
            origin_z: split_i64(frame_origin_lod0_cell[2]),
            frame_index: split_u64(frame.frame_index()),
            camera_relative_m,
            lod0_cell_size_m: LOD0_CELL_SIZE_METERS as f32,
            page_edge_cells: crate::PAGE_EDGE_CELLS as u32,
            _pad: [0; 3],
        }
    }

    pub fn from_camera(planet: PlanetId, camera: PlanetPosition, frame_index: u64) -> Self {
        Self::from_render_frame(PlanetRenderFrame::new(planet, camera, frame_index))
    }

    pub fn frame_origin_lod0_cell(self) -> [i64; 3] {
        [
            join_i64(self.origin_x),
            join_i64(self.origin_y),
            join_i64(self.origin_z),
        ]
    }

    pub fn planet_id(self) -> PlanetId {
        let mut bytes = [0_u8; 16];
        for (word, output) in self.planet_id.iter().zip(bytes.chunks_exact_mut(4)) {
            output.copy_from_slice(&word.to_le_bytes());
        }
        PlanetId(bytes)
    }

    pub const fn frame_number(self) -> u64 {
        (self.frame_index[0] as u64) | ((self.frame_index[1] as u64) << 32)
    }
}

#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Pod, Zeroable)]
pub struct GpuPageMeta {
    pub relative_lod0_cell_min: [i32; 3],
    pub lod: u32,
    pub slot: u32,
    pub generation_low: u32,
    pub generation_high: u32,
    pub transition_mask: u32,
}

impl GpuPageMeta {
    pub fn new(
        page: PageKey,
        frame_origin_lod0_cell: [i64; 3],
        slot: u32,
        generation: u64,
        transition_mask: u8,
    ) -> Result<Self, GpuPageMetaError> {
        if transition_mask & !TRANSITION_FACE_MASK != 0 {
            return Err(GpuPageMetaError::TransitionMask(transition_mask));
        }
        let generation = split_u64(generation);
        Ok(Self {
            relative_lod0_cell_min: page
                .relative_lod0_cell_min(frame_origin_lod0_cell)
                .map_err(GpuPageMetaError::Address)?,
            lod: u32::from(page.lod),
            slot,
            generation_low: generation[0],
            generation_high: generation[1],
            transition_mask: u32::from(transition_mask),
        })
    }

    pub const fn generation(self) -> u64 {
        (self.generation_low as u64) | ((self.generation_high as u64) << 32)
    }

    /// CPU mirror of `planet_camera_local_position_m` in the shared WGSL.
    pub fn camera_local_position_m(
        self,
        frame: PlanetFrameUniform,
        local_lod0_cell: [f32; 3],
    ) -> [f32; 3] {
        [
            (self.relative_lod0_cell_min[0] as f32 + local_lod0_cell[0]) * frame.lod0_cell_size_m
                - frame.camera_relative_m[0],
            (self.relative_lod0_cell_min[1] as f32 + local_lod0_cell[1]) * frame.lod0_cell_size_m
                - frame.camera_relative_m[1],
            (self.relative_lod0_cell_min[2] as f32 + local_lod0_cell[2]) * frame.lod0_cell_size_m
                - frame.camera_relative_m[2],
        ]
    }
}

#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, Default, PartialEq, Pod, Zeroable)]
pub struct GpuVoxelMaterial {
    pub base_color_roughness: [f32; 4],
    pub emissive_metalness: [f32; 4],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
pub enum GpuPageMetaError {
    #[error(transparent)]
    Address(AddressError),
    #[error("transition mask {0:#010b} uses bits outside the six page faces")]
    TransitionMask(u8),
}

const fn split_u64(value: u64) -> [u32; 2] {
    [value as u32, (value >> 32) as u32]
}

const fn split_i64(value: i64) -> [u32; 2] {
    split_u64(value as u64)
}

const fn join_i64(words: [u32; 2]) -> i64 {
    ((words[0] as u64) | ((words[1] as u64) << 32)) as i64
}

fn planet_words(planet: PlanetId) -> [u32; 4] {
    let mut words = [0_u32; 4];
    for (word, bytes) in words.iter_mut().zip(planet.0.chunks_exact(4)) {
        *word = u32::from_le_bytes(bytes.try_into().expect("chunks_exact yields four bytes"));
    }
    words
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frame_round_trips_signed_canonical_origin() {
        let camera =
            PlanetPosition::new([-63_i64, 34, i64::from(i32::MAX) * 32 + 3], [0.0; 3]).unwrap();
        let uniform = PlanetFrameUniform::from_camera(PlanetId([7; 16]), camera, 9);
        let origin = [-64_i64, 32, i64::from(i32::MAX) * 32];
        assert_eq!(uniform.frame_origin_lod0_cell(), origin);
        assert_eq!(uniform.frame_index, [9, 0]);
        assert_eq!(uniform.page_edge_cells, 32);
        assert_eq!(uniform.camera_relative_m, [0.1, 0.2, 0.3]);
    }

    #[test]
    fn page_boundary_reconstruction_is_identical_from_both_pages() {
        let camera =
            PlanetPosition::new([63_710_017, -63_710_017, 0], [0.025, 0.075, 0.0]).unwrap();
        let frame = PlanetRenderFrame::new(PlanetId([9; 16]), camera, 4);
        let uniform = PlanetFrameUniform::from_render_frame(frame);
        let left = GpuPageMeta::new(
            PageKey::new(0, [1_990_938, -1_990_940, 0]),
            frame.origin_lod0_cell(),
            0,
            1,
            0,
        )
        .unwrap();
        let right = GpuPageMeta::new(
            PageKey::new(0, [1_990_939, -1_990_940, 0]),
            frame.origin_lod0_cell(),
            1,
            1,
            0,
        )
        .unwrap();
        assert_eq!(
            left.camera_local_position_m(uniform, [32.0, 7.5, 3.25]),
            right.camera_local_position_m(uniform, [0.0, 7.5, 3.25])
        );
    }
}
