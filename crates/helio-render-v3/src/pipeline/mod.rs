use std::collections::HashMap;
use std::sync::Arc;

/// Compiled pipeline cache keyed by `(shader_source_hash, variant_hash, config_hash)`.
/// Avoids re-compiling the same pipeline across frames or feature changes.
pub struct PipelineCache {
    pipelines:  HashMap<PipelineKey, Arc<wgpu::RenderPipeline>>,
    compute:    HashMap<PipelineKey, Arc<wgpu::ComputePipeline>>,
}

#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct PipelineKey {
    pub shader_hash:  u64,   // FNV-1a of full shader WGSL source
    pub variant_hash: u64,   // FNV-1a of PipelineVariant fields
    pub config_hash:  u64,   // FNV-1a of ShaderConfig (override constants)
}

/// Which geometry variant is this pipeline for.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub enum PipelineVariant {
    Opaque,
    Masked,
    Transparent,
    Shadow,
    DepthPrepass,
    Billboard,
    DebugLine,
    Fullscreen,
}

/// Compile-time override constants that get baked into WGSL via the `override` keyword.
/// Changing these invalidates compiled pipelines (they need recompile) but this should
/// only happen on resize or feature toggle — never mid-frame.
#[derive(Clone, PartialEq, Debug)]
pub struct ShaderConfig {
    pub enable_shadows:    bool,
    pub max_shadow_lights: u32,
    pub enable_rc:         bool,
    pub enable_bloom:      bool,
    pub bloom_threshold:   f32,
    pub bloom_intensity:   f32,
}

impl ShaderConfig {
    pub fn config_hash(&self) -> u64 {
        let mut h = fnv_offset();
        h = fnv(h, self.enable_shadows as u64);
        h = fnv(h, self.max_shadow_lights as u64);
        h = fnv(h, self.enable_rc as u64);
        h = fnv(h, self.enable_bloom as u64);
        h = fnv(h, self.bloom_threshold.to_bits() as u64);
        h = fnv(h, self.bloom_intensity.to_bits() as u64);
        h
    }

    /// Build the WGSL override constant block to prepend to each shader.
    pub fn override_block(&self) -> String {
        format!(
            "override ENABLE_SHADOWS: bool = {};\n\
             override MAX_SHADOW_LIGHTS: u32 = {}u;\n\
             override ENABLE_RC: bool = {};\n\
             override ENABLE_BLOOM: bool = {};\n\
             override BLOOM_THRESHOLD: f32 = {};\n\
             override BLOOM_INTENSITY: f32 = {};\n",
            self.enable_shadows, self.max_shadow_lights,
            self.enable_rc, self.enable_bloom,
            self.bloom_threshold, self.bloom_intensity,
        )
    }
}

impl PipelineCache {
    pub fn new() -> Self {
        PipelineCache {
            pipelines: HashMap::new(),
            compute:   HashMap::new(),
        }
    }

    pub fn get_or_create_render(
        &mut self,
        key:        PipelineKey,
        device:     &wgpu::Device,
        factory:    impl FnOnce() -> wgpu::RenderPipeline,
    ) -> Arc<wgpu::RenderPipeline> {
        self.pipelines
            .entry(key)
            .or_insert_with(|| Arc::new(factory()))
            .clone()
    }

    pub fn get_or_create_compute(
        &mut self,
        key:        PipelineKey,
        device:     &wgpu::Device,
        factory:    impl FnOnce() -> wgpu::ComputePipeline,
    ) -> Arc<wgpu::ComputePipeline> {
        self.compute
            .entry(key)
            .or_insert_with(|| Arc::new(factory()))
            .clone()
    }
}

const fn fnv_offset() -> u64 { 0xcbf29ce484222325 }
const fn fnv_prime()  -> u64 { 0x100000001b3 }
fn fnv(h: u64, v: u64) -> u64 { h.wrapping_mul(fnv_prime()) ^ v }

pub fn fnv1a_hash_bytes(bytes: &[u8]) -> u64 {
    let mut h = fnv_offset();
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(fnv_prime());
    }
    h
}

pub fn fnv1a_str(s: &str) -> u64 { fnv1a_hash_bytes(s.as_bytes()) }
