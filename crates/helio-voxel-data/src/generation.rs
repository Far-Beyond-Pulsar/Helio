//! Deterministic CPU chunk generators: explicit per-sample chunk data for
//! a chunk domain (bounded voxel objects, materialized regions). A
//! descriptor is serializable by its owner; executable generator
//! implementations are registered in a runtime registry.
//!
//! Streamed terrain is generated differently: a terrain generator
//! (`helio_pass_voxel_planet::terrain`) is a field evaluated identically on
//! CPU and GPU, so worlds of any size need no stored chunks.

use std::{collections::HashMap, sync::Arc};

use crate::{VoxelChunkKey, VoxelDomain, VoxelStoredPayload};

/// Helio's built-in landform terrain generator (continents, mountains and
/// hills on planets, planes and infinite planes). Its settings component is
/// `VoxelLandformComponent`.
pub const VOXEL_TERRAIN_GENERATOR: &str = "helio.landform";
pub const VOXEL_TERRAIN_GENERATOR_VERSION: u32 = 1;
/// The streamed voxel terrain renderer, which draws every registered
/// terrain generator.
pub const VOXEL_TERRAIN_RENDERER: &str = "helio.voxel-terrain";
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
    /// Opaque settings understood by the registered generator.
    pub parameters: String,
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
        if id.is_empty() || id.len() > MAX_VOXEL_GENERATOR_ID_BYTES || version == 0 {
            return Err("invalid voxel generator ID/version".into());
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
        self.generators
            .get(&(descriptor.id.clone(), descriptor.version))
            .ok_or_else(|| {
                format!(
                    "unregistered voxel generator {} v{}",
                    descriptor.id, descriptor.version
                )
            })?
            .generate(descriptor, key)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::VOXEL_CHUNK_SAMPLES;

    /// Solid below `y = 0` with a seed-dependent material.
    struct Ground;
    impl VoxelChunkGenerator for Ground {
        fn generate(
            &self,
            descriptor: &VoxelGeneratorDescriptor,
            key: VoxelChunkKey,
        ) -> Result<Option<VoxelStoredPayload>, String> {
            if descriptor.parameters == "invalid" {
                return Err("invalid parameters".into());
            }
            let material = 1 + (descriptor.seed % 7) as u8;
            Ok((key.y < 0).then(|| VoxelStoredPayload::raw_material([material; VOXEL_CHUNK_SAMPLES])))
        }
    }

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

    fn registry() -> VoxelGeneratorRegistry {
        let mut registry = VoxelGeneratorRegistry::default();
        registry.register("example.ground", 1, Arc::new(Ground)).unwrap();
        registry
    }

    #[test]
    fn generators_are_registered_by_id_and_version() {
        let mut registry = registry();
        assert!(registry.register("example.ground", 1, Arc::new(Ground)).is_err());
        assert!(registry.register("", 1, Arc::new(Ground)).is_err());
        assert!(registry.register("example.ground", 0, Arc::new(Ground)).is_err());
        let spec = descriptor("example.ground");
        assert!(registry.generate(&spec, VoxelChunkKey::new(-1, -1, 0, 0)).unwrap().is_some());
        assert!(registry.generate(&spec, VoxelChunkKey::new(-1, 0, 0, 0)).unwrap().is_none());
        assert!(registry.generate(&spec, VoxelChunkKey::new(0, 0, 0, 5)).is_err(), "outside the domain");
    }

    #[test]
    fn unsupported_parameters_and_generators_fail_explicitly() {
        let registry = registry();
        let mut spec = descriptor("example.ground");
        spec.parameters = "invalid".into();
        assert!(registry.generate(&spec, VoxelChunkKey::new(0, -1, 0, 0)).is_err());
        spec.id = "external.unknown".into();
        assert!(registry.generate(&spec, VoxelChunkKey::new(0, -1, 0, 0)).is_err());
    }

    #[test]
    fn seed_and_version_are_part_of_reproducible_generator_identity() {
        let registry = registry();
        let mut spec = descriptor("example.ground");
        let key = VoxelChunkKey::new(0, -1, 0, 0);
        let original = registry.generate(&spec, key).unwrap();
        assert_eq!(original, registry.generate(&spec, key).unwrap());
        spec.seed += 1;
        assert_ne!(original, registry.generate(&spec, key).unwrap());
        spec.version += 1;
        assert!(registry.generate(&spec, key).is_err());
    }
}
