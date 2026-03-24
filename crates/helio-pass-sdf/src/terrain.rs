//! Terrain configuration and GPU params for procedural SDF terrain.

use bytemuck::{Pod, Zeroable};

/// Style of procedural terrain.
#[repr(u32)]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum TerrainStyle {
    Rolling = 0,
}

/// CPU-side terrain configuration.
#[derive(Clone, Debug)]
pub struct TerrainConfig {
    pub style: TerrainStyle,
    /// Y-offset of the ground plane (world-space units).
    pub height: f32,
    /// Maximum height variation above/below `height`.
    pub amplitude: f32,
    /// Base noise frequency (cycles per world-unit).
    pub frequency: f32,
    /// Number of FBM octaves.
    pub octaves: u32,
    /// Frequency multiplier per octave.
    pub lacunarity: f32,
    /// Amplitude multiplier per octave.
    pub persistence: f32,
}

impl TerrainConfig {
    pub fn rolling() -> Self {
        Self {
            style: TerrainStyle::Rolling,
            height: 0.0,
            amplitude: 25.0,
            frequency: 0.015,
            octaves: 6,
            lacunarity: 2.0,
            persistence: 0.5,
        }
    }

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

/// GPU-side terrain parameters (32 bytes, bytemuck).
/// Layout must match the WGSL `TerrainParams` struct exactly.
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
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
    pub fn disabled() -> Self {
        Self {
            enabled: 0,
            style: 0,
            height: 0.0,
            amplitude: 0.0,
            frequency: 0.0,
            octaves: 0,
            lacunarity: 1.0,
            persistence: 0.5,
        }
    }
}

