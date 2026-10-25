//! Voxel materials drawn with the host's scene materials.
//!
//! A host that owns Helio's material table (scene `MaterialComponent` rows
//! and the descriptor-indexed texture table) maps each voxel material id to
//! scene material rows, one for the top, side and bottom faces. The publish
//! stage then textures the terrain with those materials instead of the
//! flat appearance table's colours, so voxel terrain shades like any other
//! Helio surface (and graph-compiled materials flow through the same rows).
//!
//! Only on devices with texture binding arrays: elsewhere the map is
//! ignored and the flat appearance is drawn.

use bytemuck::{Pod, Zeroable};

/// Voxel material slots (`engine::MATERIALS`).
pub const SLOTS: usize = crate::terrain::MATERIALS;

/// No scene material: the flat appearance colour.
pub const NO_MATERIAL: u32 = u32::MAX;

/// Which scene material rows each voxel material shows on its faces, and
/// how large its textures are in the world.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Pod, Zeroable)]
pub struct VoxelMaterialMap {
    /// Per voxel material: scene material rows of its top, side and bottom
    /// faces (`NO_MATERIAL`: flat), and the side of a texture in the world
    /// (metres, f32 bits; 0: one voxel).
    pub faces: [[u32; 4]; SLOTS],
}

impl Default for VoxelMaterialMap {
    fn default() -> Self {
        Self {
            faces: [[NO_MATERIAL; 4]; SLOTS],
        }
    }
}

impl VoxelMaterialMap {
    /// Voxel material `material` shows scene material rows `top`, `side`
    /// and `bottom`, a texture spanning `tile_m` metres (to the nearest
    /// whole number of voxels; 0: one voxel, a block's face). Textures run
    /// on across voxels, so a world of small voxels shows its surfaces at
    /// their size, not a texture a voxel.
    pub fn set(&mut self, material: u32, top: u32, side: u32, bottom: u32, tile_m: f32) {
        self.faces[material as usize % SLOTS] = [top, side, bottom, tile_m.max(0.0).to_bits()];
    }
}

/// A frame's scene material resources, borrowed from the host's context.
pub struct SceneMaterialInputs<'a> {
    /// `GpuMaterial` rows (112 bytes each), indexed by scene material row.
    pub materials: &'a wgpu::Buffer,
    /// The descriptor-indexed texture table, `texture_capacity` long.
    pub texture_views: &'a [&'a wgpu::TextureView],
    pub samplers: &'a [&'a wgpu::Sampler],
    /// Changes whenever the buffer or any table entry does (a bind group is
    /// rebuilt only then).
    pub version: (u64, u64),
}

/// How a host reads this frame's scene materials from its pass context.
pub type SceneMaterialSource = Box<
    dyn for<'b> Fn(&helio_core::PassContext<'b>) -> Option<SceneMaterialInputs<'b>> + Send + Sync,
>;

/// Texture table entries a device's binding arrays hold, or none where it
/// has no descriptor indexing (mirrors Helio's `MaterialBindingConfig`).
pub fn texture_capacity(device: &wgpu::Device) -> Option<u32> {
    #[cfg(target_arch = "wasm32")]
    {
        let _ = device;
        None
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        let needed = wgpu::Features::TEXTURE_BINDING_ARRAY
            | wgpu::Features::SAMPLED_TEXTURE_AND_STORAGE_BUFFER_ARRAY_NON_UNIFORM_INDEXING;
        if !device.features().contains(needed) {
            return None;
        }
        #[cfg(any(target_os = "macos", target_os = "ios", target_os = "android"))]
        const PLATFORM_MAX: u32 = 16;
        #[cfg(not(any(target_os = "macos", target_os = "ios", target_os = "android")))]
        const PLATFORM_MAX: u32 = 256;
        let limits = device.limits();
        Some(
            PLATFORM_MAX
                .min(limits.max_sampled_textures_per_shader_stage)
                .min(limits.max_samplers_per_shader_stage),
        )
    }
}

/// The line of chunk_draw.wgsl the scene materials replace: no scene surface.
pub(crate) const STUB: &str = "fn scene_surface(m: u32, axis: u32, sign: u32, level: u32, cell: vec3<i32>, in_cell: vec3<f32>, pixel_m: f32, slant: f32) -> SceneSurface {\n    return SceneSurface(vec3<f32>(0.0), 0.0, vec3<f32>(0.0), vec3<f32>(0.0, 0.0, 1.0), 1.0, false);\n}";

/// chunk_draw.wgsl with the scene materials spliced in for a texture table
/// of `capacity` entries, and Helio's specular anti-aliasing, which they
/// shade normal maps with.
pub(crate) fn scene_source(draw: &str, capacity: u32) -> String {
    let real = include_str!("../../shaders/scene_materials.wgsl")
        .replace("MATERIAL_TEXTURES", &format!("{capacity}"))
        + "\n"
        + helio_mats::SPECULAR_AA;
    format!(
        "enable wgpu_binding_array;\n{}",
        draw.replacen(STUB, &real, 1)
    )
}

pub(crate) fn layout_entries(capacity: u32) -> Vec<wgpu::BindGroupLayoutEntry> {
    let count = std::num::NonZeroU32::new(capacity);
    vec![
        wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        },
        wgpu::BindGroupLayoutEntry {
            binding: 1,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        },
        wgpu::BindGroupLayoutEntry {
            binding: 2,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count,
        },
        wgpu::BindGroupLayoutEntry {
            binding: 3,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
            count,
        },
    ]
}
