//! Procedural terrain configuration and GPU uniform structs.

// Compile-time assertion: GpuTerrainParams must be exactly 64 bytes for WGSL uniform buffer compatibility
const _: () = assert!(
    std::mem::size_of::<GpuTerrainParams>() == 64,
    "GpuTerrainParams must be 64 bytes"
);

/// Terrain generation style.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TerrainStyle {
    /// Smooth rolling hills with gentle FBM variation.
    Rolling = 0,
    /// Tall, jagged mountains with stronger relief.
    Mountains = 1,
    /// Deep, carved canyon terrain with erosion-like detail.
    Canyons = 2,
    /// Wind-swept dunes with elongated, directional patterns.
    Dunes = 3,
    /// Domain-warped organic terrain (Inigo Quilez two-layer warp).
    Warped = 4,
}

/// CPU-side terrain configuration.
#[derive(Clone, Debug)]
pub struct TerrainConfig {
    pub style: TerrainStyle,
    pub height: f32,
    pub amplitude: f32,
    pub frequency: f32,
    pub octaves: u32,
    pub lacunarity: f32,
    pub persistence: f32,
    /// Warp strength (kept for future style-specific use).
    pub warp_amount: f32,
}

impl TerrainConfig {
    /// Create a default rolling hills terrain (no domain warping).
    pub fn rolling() -> Self {
        Self {
            style: TerrainStyle::Rolling,
            height: -2.0,
            amplitude: 4.0,
            frequency: 0.08,
            octaves: 5,
            lacunarity: 2.0,
            persistence: 0.5,
            warp_amount: 0.0,
        }
    }

    /// Create tall, jagged mountain terrain with stronger relief.
    pub fn mountains() -> Self {
        Self {
            style: TerrainStyle::Mountains,
            height: -5.0,
            amplitude: 25.0,
            frequency: 0.03,
            octaves: 7,
            lacunarity: 2.0,
            persistence: 0.5,
            warp_amount: 2.0,
        }
    }

    /// Create carved canyon terrain with erosion-like detail.
    pub fn canyons() -> Self {
        Self {
            style: TerrainStyle::Canyons,
            height: -2.0,
            amplitude: 15.0,
            frequency: 0.05,
            octaves: 6,
            lacunarity: 2.0,
            persistence: 0.55,
            warp_amount: 3.0,
        }
    }

    /// Create elongated dune terrain with wind-swept patterns.
    pub fn dunes() -> Self {
        Self {
            style: TerrainStyle::Dunes,
            height: -1.0,
            amplitude: 6.0,
            frequency: 0.15,
            octaves: 4,
            lacunarity: 2.0,
            persistence: 0.6,
            warp_amount: 1.0,
        }
    }

    /// Create domain-warped organic terrain (Inigo Quilez two-layer warp).
    ///
    /// `warp_amount` controls the warp intensity; 4.0 is a good default for
    /// visibly organic-looking shapes. Pass 0.0 for no warp.
    pub fn warped(warp_amount: f32) -> Self {
        Self {
            style: TerrainStyle::Warped,
            height: -2.0,
            amplitude: 4.0,
            frequency: 0.08,
            octaves: 5,
            lacunarity: 2.0,
            persistence: 0.5,
            warp_amount,
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
            warp_amount: self.warp_amount,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
            _pad3: 0,
            _pad4: 0,
            _pad5: 0,
            _pad6: 0,
        }
    }
}

/// GPU-side terrain parameters (64 bytes).
///
/// Layout must match the WGSL `TerrainParams` struct.
/// Note: WGSL uniform buffers require minimum 64 bytes, so we pad to exactly 64 bytes.
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuTerrainParams {
    pub enabled: u32,
    pub style: u32,
    pub height: f32,
    pub amplitude: f32,
    // offset 16
    pub frequency: f32,
    pub octaves: u32,
    pub lacunarity: f32,
    pub persistence: f32,
    // offset 32
    pub warp_amount: f32,
    pub _pad0: u32,
    pub _pad1: u32,
    pub _pad2: u32,
    // offset 48
    pub _pad3: u32,
    pub _pad4: u32,
    pub _pad5: u32,
    pub _pad6: u32,
    // offset 64 - WGSL uniform buffer requires minimum 64 bytes
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
            lacunarity: 0.0,
            persistence: 0.0,
            warp_amount: 0.0,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
            _pad3: 0,
            _pad4: 0,
            _pad5: 0,
            _pad6: 0,
        }
    }
}
