//! Procedural terrain configuration and GPU uniform structs.

/// Terrain generation style.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TerrainStyle {
    /// Gentle rolling hills (2D heightfield FBM).
    Rolling = 0,
}

/// CPU-side terrain configuration. Use `.build_gpu_params()` to create the GPU uniform.
#[derive(Clone, Debug)]
pub struct TerrainConfig {
    pub style: TerrainStyle,
    pub height: f32,
    pub amplitude: f32,
    pub frequency: f32,
    pub octaves: u32,
    pub lacunarity: f32,
    pub persistence: f32,
}

impl TerrainConfig {
    /// Create a default rolling hills terrain.
    pub fn rolling() -> Self {
        Self {
            style: TerrainStyle::Rolling,
            height: -2.0,
            amplitude: 4.0,
            frequency: 0.08,
            octaves: 5,
            lacunarity: 2.0,
            persistence: 0.5,
        }
    }

    /// Build the GPU-side uniform struct.
    pub fn build_gpu_params(&self) -> GpuTerrainParams {
        GpuTerrainParams {
            enabled: 1,
            style: self.style as u32,
            height: self.height,
            amplitude: self.amplitude,
            frequency: self.frequency,
            octaves: self.octaves,
            lacunarity: self.lacunarity,
            persistence: self.persistence,
        }
    }
}

/// GPU-side terrain parameters (32 bytes, 4-byte aligned).
///
/// Layout must match the WGSL `TerrainParams` struct.
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuTerrainParams {
    pub enabled: u32,
    pub style: u32,
    pub height: f32,
    pub amplitude: f32,
    pub frequency: f32,
    pub octaves: u32,
    pub lacunarity: f32,
    pub persistence: f32,
}

impl GpuTerrainParams {
    /// Disabled terrain (returns 1e10, preserving original behavior).
    pub fn disabled() -> Self {
        Self {
            enabled: 0,
            style: 0,
            height: 0.0,
            amplitude: 0.0,
            frequency: 0.0,
            octaves: 0,
            lacunarity: 0.0,
            persistence: 0.0,
        }
    }
}
