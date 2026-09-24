//! Deterministic CPU chunk generators. A descriptor is serializable by its
//! owner; executable generator implementations stay in this runtime registry.

use std::{collections::HashMap, sync::Arc};

use crate::{VoxelChunkKey, VoxelDomain, VoxelStoredPayload, VOXEL_CHUNK_SAMPLES};

pub const VOXEL_FLAT_GENERATOR: &str = "helio.flat";
pub const VOXEL_PLANET_GENERATOR: &str = "helio.planet";
pub const VOXEL_BUILTIN_GENERATOR_VERSION: u32 = 1;
pub const MAX_VOXEL_GENERATOR_ID_BYTES: usize = 256;
pub const MAX_VOXEL_GENERATOR_PARAMETERS_BYTES: usize = 1024 * 1024;

#[derive(Clone, Debug, PartialEq)]
pub struct VoxelGeneratorDescriptor {
    pub id: String,
    pub version: u32,
    pub seed: u64,
    pub domain: VoxelDomain,
    pub origin: [f64; 3],
    pub voxel_size: f64,
    /// Logical width of one chunk key at LOD zero in base-resolution voxels.
    pub chunk_edge_voxels: u32,
    /// Spatial scale between adjacent LODs; one disables spatial scaling.
    pub lod_scale: u32,
    /// Opaque settings understood by the registered generator. The simple
    /// built-in generators read this as `VoxelBuiltinGeneratorConfig` JSON.
    pub parameters: String,
}

#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct VoxelBuiltinGeneratorConfig {
    pub planet_radius: f64,
    pub base_height: f64,
    pub amplitude: f64,
    pub wavelength: f64,
    /// One-based slot into the owning component's SceneDB material ID palette.
    pub material_slot: u8,
}

impl Default for VoxelBuiltinGeneratorConfig {
    fn default() -> Self {
        Self {
            planet_radius: 8.0,
            base_height: 0.0,
            amplitude: 0.0,
            wavelength: 16.0,
            material_slot: 1,
        }
    }
}

impl VoxelGeneratorDescriptor {
    pub fn validate(&self) -> Result<(), String> {
        if self.id.is_empty() {
            return Err("generator ID is empty; use external chunk batches instead".into());
        }
        if self.id.len() > MAX_VOXEL_GENERATOR_ID_BYTES
            || self.parameters.len() > MAX_VOXEL_GENERATOR_PARAMETERS_BYTES
        {
            return Err("generator ID or parameters exceed the descriptor limit".into());
        }
        if self.version == 0 {
            return Err("generator version must be positive".into());
        }
        if !self.voxel_size.is_finite()
            || self.voxel_size <= 0.0
            || self.chunk_edge_voxels == 0
            || self.lod_scale == 0
            || self.origin.iter().any(|v| !v.is_finite())
        {
            return Err("generator spatial settings must be finite and positive".into());
        }
        Ok(())
    }

    fn builtin_config(&self, shape_mode: u32) -> Result<VoxelBuiltinGeneratorConfig, String> {
        let config = if self.parameters.is_empty() {
            VoxelBuiltinGeneratorConfig::default()
        } else {
            serde_json::from_str(&self.parameters)
                .map_err(|error| format!("invalid built-in generator parameters: {error}"))?
        };
        if self.chunk_edge_voxels != 8
            || !config.base_height.is_finite()
            || !config.amplitude.is_finite()
            || config.amplitude < 0.0
            || !config.wavelength.is_finite()
            || config.wavelength <= 0.0
            || (shape_mode == 1
                && (!config.planet_radius.is_finite() || config.planet_radius <= 0.0))
            || config.material_slot == 0
        {
            return Err(
                "built-in generator requires an 8-voxel chunk and valid shape/material parameters"
                    .into(),
            );
        }
        Ok(config)
    }
}

/// An external implementation is registered at runtime under an ID and
/// version. It receives only CPU values and returns a complete versioned
/// chunk, or None for known empty space. The caller publishes the result in a
/// revisioned batch; generators never touch renderer/GPU objects.
pub trait VoxelChunkGenerator: Send + Sync + 'static {
    fn generate(
        &self,
        descriptor: &VoxelGeneratorDescriptor,
        key: VoxelChunkKey,
    ) -> Result<Option<VoxelStoredPayload>, String>;
}

#[derive(Default, Clone)]
pub struct VoxelGeneratorRegistry {
    generators: HashMap<(String, u32), Arc<dyn VoxelChunkGenerator>>,
}

impl VoxelGeneratorRegistry {
    pub fn register(
        &mut self,
        id: impl Into<String>,
        version: u32,
        generator: Arc<dyn VoxelChunkGenerator>,
    ) -> Result<(), String> {
        let id = id.into();
        if id.is_empty()
            || id.len() > MAX_VOXEL_GENERATOR_ID_BYTES
            || version == 0
            || id == VOXEL_FLAT_GENERATOR
            || id == VOXEL_PLANET_GENERATOR
        {
            return Err("invalid or reserved voxel generator ID/version".into());
        }
        if self.generators.contains_key(&(id.clone(), version)) {
            return Err("voxel generator ID/version was already registered".into());
        }
        self.generators.insert((id, version), generator);
        Ok(())
    }

    pub fn generate(
        &self,
        descriptor: &VoxelGeneratorDescriptor,
        key: VoxelChunkKey,
    ) -> Result<Option<VoxelStoredPayload>, String> {
        descriptor.validate()?;
        descriptor
            .domain
            .validate_key(key)
            .map_err(|e| format!("chunk key outside generator domain: {e:?}"))?;
        match (descriptor.id.as_str(), descriptor.version) {
            (VOXEL_FLAT_GENERATOR, VOXEL_BUILTIN_GENERATOR_VERSION) => {
                generate_builtin(descriptor, key, 0, &descriptor.builtin_config(0)?)
            }
            (VOXEL_PLANET_GENERATOR, VOXEL_BUILTIN_GENERATOR_VERSION) => {
                generate_builtin(descriptor, key, 1, &descriptor.builtin_config(1)?)
            }
            (VOXEL_FLAT_GENERATOR | VOXEL_PLANET_GENERATOR, _) => {
                Err("built-in generator version or shape is unsupported".into())
            }
            _ => self
                .generators
                .get(&(descriptor.id.clone(), descriptor.version))
                .ok_or_else(|| {
                    format!(
                        "unregistered voxel generator {} v{}",
                        descriptor.id, descriptor.version
                    )
                })?
                .generate(descriptor, key),
        }
    }
}

fn generate_builtin(
    descriptor: &VoxelGeneratorDescriptor,
    key: VoxelChunkKey,
    shape_mode: u32,
    config: &VoxelBuiltinGeneratorConfig,
) -> Result<Option<VoxelStoredPayload>, String> {
    let step = descriptor.voxel_size * f64::from(descriptor.lod_scale).powi(i32::from(key.lod));
    if !step.is_finite() || step <= 0.0 {
        return Err("LOD voxel size is not representable".into());
    }
    let base = [key.x, key.y, key.z].map(|v| i128::from(v) * 8);
    let mut samples = [0u8; VOXEL_CHUNK_SAMPLES];
    let mut non_air = false;
    for z in 0..8usize {
        for y in 0..8usize {
            for x in 0..8usize {
                let xyz = [x, y, z];
                let point = std::array::from_fn::<_, 3, _>(|axis| {
                    descriptor.origin[axis] + (base[axis] as f64 + xyz[axis] as f64 + 0.5) * step
                });
                if point.iter().any(|v| !v.is_finite()) {
                    return Err("generated sample position is not representable".into());
                }
                let local =
                    std::array::from_fn::<_, 3, _>(|axis| point[axis] - descriptor.origin[axis]);
                let elevation = config.base_height
                    + config.amplitude
                        * value_noise(
                            descriptor.seed,
                            local[0] / config.wavelength,
                            local[2] / config.wavelength,
                        );
                let solid = if shape_mode == 0 {
                    local[1] <= elevation
                } else {
                    let radial =
                        (local[0] * local[0] + local[1] * local[1] + local[2] * local[2]).sqrt();
                    radial <= config.planet_radius + elevation
                };
                if solid {
                    samples[z * 64 + y * 8 + x] = config.material_slot;
                    non_air = true;
                }
            }
        }
    }
    Ok(non_air.then(|| VoxelStoredPayload::raw_material(samples)))
}

fn value_noise(seed: u64, x: f64, z: f64) -> f64 {
    let x0 = x.floor();
    let z0 = z.floor();
    let sx = (x - x0).clamp(0.0, 1.0);
    let sz = (z - z0).clamp(0.0, 1.0);
    let sx = sx * sx * (3.0 - 2.0 * sx);
    let sz = sz * sz * (3.0 - 2.0 * sz);
    let hash = |dx: u64, dz: u64| {
        let mut v = seed
            ^ (x0 as i64 as u64)
                .wrapping_add(dx)
                .wrapping_mul(0x9e3779b97f4a7c15)
            ^ (z0 as i64 as u64)
                .wrapping_add(dz)
                .wrapping_mul(0xbf58476d1ce4e5b9);
        v ^= v >> 30;
        v = v.wrapping_mul(0xbf58476d1ce4e5b9);
        v ^= v >> 27;
        v = v.wrapping_mul(0x94d049bb133111eb);
        ((v ^ (v >> 31)) >> 11) as f64 / ((1u64 << 53) as f64) * 2.0 - 1.0
    };
    let a = hash(0, 0) * (1.0 - sx) + hash(1, 0) * sx;
    let b = hash(0, 1) * (1.0 - sx) + hash(1, 1) * sx;
    a * (1.0 - sz) + b * sz
}

#[cfg(test)]
mod tests {
    use super::*;

    fn descriptor(id: &str) -> VoxelGeneratorDescriptor {
        VoxelGeneratorDescriptor {
            id: id.into(),
            version: 1,
            seed: 23,
            domain: VoxelDomain::Unbounded { max_lod: 4 },
            origin: [0.0; 3],
            voxel_size: 1.0,
            chunk_edge_voxels: 8,
            lod_scale: 2,
            parameters: String::new(),
        }
    }

    #[test]
    fn flat_and_planet_are_deterministic_across_signed_chunks_and_lod() {
        let registry = VoxelGeneratorRegistry::default();
        let flat = descriptor(VOXEL_FLAT_GENERATOR);
        assert!(registry
            .generate(&flat, VoxelChunkKey::new(-1, -1, 0, 0))
            .unwrap()
            .is_some());
        assert!(registry
            .generate(&flat, VoxelChunkKey::new(-1, 0, 0, 0))
            .unwrap()
            .is_none());
        let planet = descriptor(VOXEL_PLANET_GENERATOR);
        let key = VoxelChunkKey::new(-1, 0, 0, 1);
        assert_eq!(
            registry.generate(&planet, key).unwrap(),
            registry.generate(&planet, key).unwrap()
        );
        assert!(registry
            .generate(&planet, VoxelChunkKey::new(8, 0, 0, 0))
            .unwrap()
            .is_none());
        assert!(registry
            .generate(&planet, VoxelChunkKey::new(0, 0, 0, 0))
            .unwrap()
            .is_some());
        assert!(registry
            .generate(&planet, VoxelChunkKey::new(0, 0, 0, 5))
            .is_err());
    }

    #[test]
    fn unsupported_parameters_and_generators_fail_explicitly() {
        let registry = VoxelGeneratorRegistry::default();
        let mut spec = descriptor(VOXEL_FLAT_GENERATOR);
        spec.parameters = "not json".into();
        assert!(registry
            .generate(&spec, VoxelChunkKey::new(0, 0, 0, 0))
            .is_err());
        spec.id = "external.unknown".into();
        assert!(registry
            .generate(&spec, VoxelChunkKey::new(0, 0, 0, 0))
            .is_err());
        spec.parameters = serde_json::to_string(&VoxelBuiltinGeneratorConfig {
            wavelength: 0.0,
            ..Default::default()
        })
        .unwrap();
        spec.id = VOXEL_FLAT_GENERATOR.into();
        assert!(registry
            .generate(&spec, VoxelChunkKey::new(0, 0, 0, 0))
            .is_err());
    }

    #[test]
    fn seed_and_version_are_part_of_reproducible_generator_identity() {
        let registry = VoxelGeneratorRegistry::default();
        let mut spec = descriptor(VOXEL_FLAT_GENERATOR);
        spec.parameters = serde_json::to_string(&VoxelBuiltinGeneratorConfig {
            amplitude: 4.0,
            wavelength: 3.0,
            ..Default::default()
        })
        .unwrap();
        let key = VoxelChunkKey::new(0, 0, 0, 0);
        let original = registry.generate(&spec, key).unwrap();
        assert_eq!(original, registry.generate(&spec, key).unwrap());
        spec.seed += 1;
        assert_ne!(original, registry.generate(&spec, key).unwrap());
        spec.version += 1;
        assert!(registry.generate(&spec, key).is_err());
    }

    #[test]
    fn external_generator_is_registered_by_id_and_version() {
        struct Solid;
        impl VoxelChunkGenerator for Solid {
            fn generate(
                &self,
                _descriptor: &VoxelGeneratorDescriptor,
                _key: VoxelChunkKey,
            ) -> Result<Option<VoxelStoredPayload>, String> {
                Ok(Some(VoxelStoredPayload::raw_material(
                    [1; VOXEL_CHUNK_SAMPLES],
                )))
            }
        }
        let mut registry = VoxelGeneratorRegistry::default();
        registry
            .register("example.solid", 1, Arc::new(Solid))
            .unwrap();
        assert!(registry
            .register("example.solid", 1, Arc::new(Solid))
            .is_err());
        assert!(registry
            .register(VOXEL_FLAT_GENERATOR, 1, Arc::new(Solid))
            .is_err());
        let spec = descriptor("example.solid");
        assert_eq!(
            registry
                .generate(&spec, VoxelChunkKey::new(0, 0, 0, 0))
                .unwrap()
                .unwrap(),
            VoxelStoredPayload::raw_material([1; VOXEL_CHUNK_SAMPLES])
        );
    }
}
