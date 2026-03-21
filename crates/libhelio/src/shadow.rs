//! Shadow configuration types for runtime quality control.
//!
//! Provides GPU-compatible shadow configuration structs and quality presets
//! for controlling CSM splits, bias, filter radius, and PCSS parameters.

use bytemuck::{Pod, Zeroable};

/// Shadow quality presets for runtime configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShadowQuality {
    /// 8-sample PCF, no PCSS (low-end hardware, competitive gaming)
    Low,
    /// 16-sample PCF, no PCSS (consoles, mid-tier PC)
    Medium,
    /// 16-sample PCSS: 8 blocker + 16 filter (high-end PC, recommended)
    High,
    /// 32-sample PCSS: 16 blocker + 32 filter (cinematic, photo mode)
    Ultra,
}

/// Per-cascade configuration (16 bytes, GPU-aligned).
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct CascadeConfig {
    /// Far plane distance for this cascade (meters)
    pub split_distance: f32,
    /// Base depth bias for this cascade
    pub depth_bias: f32,
    /// PCF filter radius (texels in shadow atlas)
    pub filter_radius: f32,
    /// PCSS light size (meters, 0.0 = disable PCSS for this cascade)
    pub pcss_light_size: f32,
}

/// Global shadow configuration (96 bytes, GPU uniform).
///
/// Uploaded to GPU as a uniform buffer for runtime shadow quality control.
/// Changes to this config require a single buffer write (delta upload).
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct ShadowConfig {
    /// Per-cascade settings (4 cascades × 16 bytes = 64 bytes)
    pub cascades: [CascadeConfig; 4],
    /// Global PCSS toggle (0 = PCF only, 1 = PCSS enabled)
    pub enable_pcss: u32,
    /// Blocker search sample count (8-16 recommended)
    pub pcss_blocker_samples: u32,
    /// PCSS filter sample count (16-32 recommended)
    pub pcss_filter_samples: u32,
    /// Padding for 16-byte alignment
    pub _pad: u32,
}

impl Default for ShadowConfig {
    fn default() -> Self {
        Self::from_quality(ShadowQuality::Medium)
    }
}

impl ShadowConfig {
    /// Create shadow configuration from a quality preset.
    pub fn from_quality(quality: ShadowQuality) -> Self {
        match quality {
            ShadowQuality::Low => Self {
                cascades: [
                    CascadeConfig {
                        split_distance: 16.0,
                        depth_bias: 0.00010,
                        filter_radius: 1.0,
                        pcss_light_size: 0.0, // No PCSS
                    },
                    CascadeConfig {
                        split_distance: 80.0,
                        depth_bias: 0.00015,
                        filter_radius: 1.5,
                        pcss_light_size: 0.0,
                    },
                    CascadeConfig {
                        split_distance: 300.0,
                        depth_bias: 0.00020,
                        filter_radius: 2.0,
                        pcss_light_size: 0.0,
                    },
                    CascadeConfig {
                        split_distance: 1400.0,
                        depth_bias: 0.00030,
                        filter_radius: 3.0,
                        pcss_light_size: 0.0,
                    },
                ],
                enable_pcss: 0,
                pcss_blocker_samples: 8,
                pcss_filter_samples: 8,
                _pad: 0,
            },
            ShadowQuality::Medium => Self {
                cascades: [
                    CascadeConfig {
                        split_distance: 16.0,
                        depth_bias: 0.00010,
                        filter_radius: 1.5,
                        pcss_light_size: 0.0, // No PCSS
                    },
                    CascadeConfig {
                        split_distance: 80.0,
                        depth_bias: 0.00015,
                        filter_radius: 2.0,
                        pcss_light_size: 0.0,
                    },
                    CascadeConfig {
                        split_distance: 300.0,
                        depth_bias: 0.00020,
                        filter_radius: 2.5,
                        pcss_light_size: 0.0,
                    },
                    CascadeConfig {
                        split_distance: 1400.0,
                        depth_bias: 0.00030,
                        filter_radius: 3.5,
                        pcss_light_size: 0.0,
                    },
                ],
                enable_pcss: 0,
                pcss_blocker_samples: 16,
                pcss_filter_samples: 16,
                _pad: 0,
            },
            ShadowQuality::High => Self {
                cascades: [
                    CascadeConfig {
                        split_distance: 16.0,
                        depth_bias: 0.00008,
                        filter_radius: 2.0,
                        pcss_light_size: 2.0, // PCSS enabled
                    },
                    CascadeConfig {
                        split_distance: 80.0,
                        depth_bias: 0.00012,
                        filter_radius: 2.5,
                        pcss_light_size: 4.0,
                    },
                    CascadeConfig {
                        split_distance: 300.0,
                        depth_bias: 0.00018,
                        filter_radius: 3.0,
                        pcss_light_size: 8.0,
                    },
                    CascadeConfig {
                        split_distance: 1400.0,
                        depth_bias: 0.00025,
                        filter_radius: 4.0,
                        pcss_light_size: 16.0,
                    },
                ],
                enable_pcss: 1,
                pcss_blocker_samples: 8,
                pcss_filter_samples: 16,
                _pad: 0,
            },
            ShadowQuality::Ultra => Self {
                cascades: [
                    CascadeConfig {
                        split_distance: 16.0,
                        depth_bias: 0.00008,
                        filter_radius: 2.5,
                        pcss_light_size: 2.0, // PCSS enabled
                    },
                    CascadeConfig {
                        split_distance: 80.0,
                        depth_bias: 0.00012,
                        filter_radius: 3.0,
                        pcss_light_size: 4.0,
                    },
                    CascadeConfig {
                        split_distance: 300.0,
                        depth_bias: 0.00018,
                        filter_radius: 3.5,
                        pcss_light_size: 8.0,
                    },
                    CascadeConfig {
                        split_distance: 1400.0,
                        depth_bias: 0.00025,
                        filter_radius: 4.5,
                        pcss_light_size: 16.0,
                    },
                ],
                enable_pcss: 1,
                pcss_blocker_samples: 16,
                pcss_filter_samples: 32,
                _pad: 0,
            },
        }
    }

    /// PSSM (Practical Split Scheme) cascade distribution.
    ///
    /// Computes cascade split distances using a blend of uniform and logarithmic
    /// distribution. `lambda = 0.5` is a good balance for most scenes.
    ///
    /// # Arguments
    /// - `near`: Camera near plane (meters)
    /// - `far`: Camera far plane (meters)
    /// - `lambda`: Blend factor (0.0 = uniform, 1.0 = logarithmic)
    pub fn pssm_splits(near: f32, far: f32, lambda: f32) -> [f32; 4] {
        let mut splits = [0.0; 4];
        for i in 0..4 {
            let ratio = (i as f32 + 1.0) / 4.0;
            let c_uniform = near + (far - near) * ratio;
            let c_log = near * (far / near).powf(ratio);
            splits[i] = lambda * c_log + (1.0 - lambda) * c_uniform;
        }
        splits
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shadow_config_size() {
        assert_eq!(std::mem::size_of::<CascadeConfig>(), 16);
        assert_eq!(std::mem::size_of::<ShadowConfig>(), 96);
    }

    #[test]
    fn test_shadow_config_alignment() {
        assert_eq!(std::mem::align_of::<ShadowConfig>(), 4);
    }

    #[test]
    fn test_pssm_splits() {
        let splits = ShadowConfig::pssm_splits(0.1, 1000.0, 0.5);
        assert!(splits[0] < splits[1]);
        assert!(splits[1] < splits[2]);
        assert!(splits[2] < splits[3]);
        assert!(splits[3] <= 1000.0);
    }
}
