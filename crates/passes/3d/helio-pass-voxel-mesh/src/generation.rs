//! Deterministic CPU chunk generators. A descriptor is serializable by its
//! owner; executable adapters stay in this runtime registry.

use std::{collections::HashMap, sync::Arc};

use crate::{VoxelChunkKey, VoxelDomain, VOXEL_CHUNK_SAMPLES};

pub const VOXEL_FLAT_GENERATOR: &str = "helio.flat";
pub const VOXEL_PLANET_GENERATOR: &str = "helio.planet";
pub const VOXEL_BUILTIN_GENERATOR_VERSION: u32 = 1;

#[derive(Clone, Debug, PartialEq)]
pub struct VoxelGeneratorDescriptor {
    pub id: String,
    pub version: u32,
    pub seed: u64,
    /// 0 is a plane, 1 is a planet. This must match the registered generator.
    pub shape_mode: u32,
    pub domain: VoxelDomain,
    pub origin: [f64; 3],
    pub voxel_size: f64,
    pub planet_radius: f64,
    pub base_height: f64,
    pub amplitude: f64,
    pub wavelength: f64,
    /// One-based slot into the owning component's SceneDB material ID palette.
    pub material_slot: u8,
}

impl VoxelGeneratorDescriptor {
    pub fn validate(&self) -> Result<(), String> {
        if self.id.is_empty() {
            return Err("generator ID is empty; use external chunk batches instead".into());
        }
        if self.version == 0 {
            return Err("generator version must be positive".into());
        }
        if self.shape_mode > 1 {
            return Err("generator shape must be plane (0) or planet (1)".into());
        }
        if !self.voxel_size.is_finite()
            || self.voxel_size <= 0.0
            || self.origin.iter().any(|v| !v.is_finite())
            || !self.base_height.is_finite()
            || !self.amplitude.is_finite()
            || self.amplitude < 0.0
            || !self.wavelength.is_finite()
            || self.wavelength <= 0.0
            || (self.shape_mode == 1
                && (!self.planet_radius.is_finite() || self.planet_radius <= 0.0))
            || self.material_slot == 0
        {
            return Err("generator parameters must be finite, positive where required, and use a non-air material slot".into());
        }
        Ok(())
    }
}

/// An external implementation is registered at runtime under an ID and
/// version. It receives only CPU values and returns a complete canonical
/// 8³ chunk, or None for known air. The caller publishes the result in a
/// revisioned batch; adapters never touch renderer/GPU objects.
pub trait VoxelChunkGenerator: Send + Sync + 'static {
    fn generate(
        &self,
        descriptor: &VoxelGeneratorDescriptor,
        key: VoxelChunkKey,
    ) -> Result<Option<[u8; VOXEL_CHUNK_SAMPLES]>, String>;
}

#[derive(Default, Clone)]
pub struct VoxelGeneratorRegistry {
    adapters: HashMap<(String, u32), Arc<dyn VoxelChunkGenerator>>,
}

impl VoxelGeneratorRegistry {
    pub fn register(
        &mut self,
        id: impl Into<String>,
        version: u32,
        adapter: Arc<dyn VoxelChunkGenerator>,
    ) -> Result<(), String> {
        let id = id.into();
        if id.is_empty()
            || version == 0
            || id == VOXEL_FLAT_GENERATOR
            || id == VOXEL_PLANET_GENERATOR
        {
            return Err("invalid or reserved voxel generator ID/version".into());
        }
        if self.adapters.contains_key(&(id.clone(), version)) {
            return Err("voxel generator ID/version was already registered".into());
        }
        self.adapters.insert((id, version), adapter);
        Ok(())
    }

    pub fn generate(
        &self,
        descriptor: &VoxelGeneratorDescriptor,
        key: VoxelChunkKey,
    ) -> Result<Option<[u8; VOXEL_CHUNK_SAMPLES]>, String> {
        descriptor.validate()?;
        descriptor
            .domain
            .validate_key(key)
            .map_err(|e| format!("chunk key outside generator domain: {e:?}"))?;
        match (descriptor.id.as_str(), descriptor.version) {
            (VOXEL_FLAT_GENERATOR, VOXEL_BUILTIN_GENERATOR_VERSION)
                if descriptor.shape_mode == 0 =>
            {
                generate_builtin(descriptor, key)
            }
            (VOXEL_PLANET_GENERATOR, VOXEL_BUILTIN_GENERATOR_VERSION)
                if descriptor.shape_mode == 1 =>
            {
                generate_builtin(descriptor, key)
            }
            (VOXEL_FLAT_GENERATOR | VOXEL_PLANET_GENERATOR, _) => {
                Err("built-in generator version or shape is unsupported".into())
            }
            _ => self
                .adapters
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
) -> Result<Option<[u8; VOXEL_CHUNK_SAMPLES]>, String> {
    let step = descriptor.voxel_size * 2f64.powi(i32::from(key.lod));
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
                let elevation = descriptor.base_height
                    + descriptor.amplitude
                        * value_noise(
                            descriptor.seed,
                            local[0] / descriptor.wavelength,
                            local[2] / descriptor.wavelength,
                        );
                let solid = if descriptor.shape_mode == 0 {
                    local[1] <= elevation
                } else {
                    let radial =
                        (local[0] * local[0] + local[1] * local[1] + local[2] * local[2]).sqrt();
                    radial <= descriptor.planet_radius + elevation
                };
                if solid {
                    samples[z * 64 + y * 8 + x] = descriptor.material_slot;
                    non_air = true;
                }
            }
        }
    }
    Ok(non_air.then_some(samples))
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

    fn descriptor(id: &str, shape_mode: u32) -> VoxelGeneratorDescriptor {
        VoxelGeneratorDescriptor {
            id: id.into(),
            version: 1,
            seed: 23,
            shape_mode,
            domain: VoxelDomain::Unbounded { max_lod: 4 },
            origin: [0.0; 3],
            voxel_size: 1.0,
            planet_radius: 8.0,
            base_height: 0.0,
            amplitude: 0.0,
            wavelength: 16.0,
            material_slot: 1,
        }
    }

    #[test]
    fn flat_and_planet_are_deterministic_across_signed_chunks_and_lod() {
        let registry = VoxelGeneratorRegistry::default();
        let flat = descriptor(VOXEL_FLAT_GENERATOR, 0);
        assert!(registry
            .generate(&flat, VoxelChunkKey::new(-1, -1, 0, 0))
            .unwrap()
            .is_some());
        assert!(registry
            .generate(&flat, VoxelChunkKey::new(-1, 0, 0, 0))
            .unwrap()
            .is_none());
        let planet = descriptor(VOXEL_PLANET_GENERATOR, 1);
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
    fn unsupported_modes_fail_explicitly() {
        let registry = VoxelGeneratorRegistry::default();
        let mut spec = descriptor(VOXEL_FLAT_GENERATOR, 1);
        assert!(registry
            .generate(&spec, VoxelChunkKey::new(0, 0, 0, 0))
            .is_err());
        spec.id = "external.unknown".into();
        assert!(registry
            .generate(&spec, VoxelChunkKey::new(0, 0, 0, 0))
            .is_err());
        spec.wavelength = 0.0;
        assert!(registry
            .generate(&spec, VoxelChunkKey::new(0, 0, 0, 0))
            .is_err());
    }
}
