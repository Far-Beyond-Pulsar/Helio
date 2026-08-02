use crate::{GpuPageTableEntry, GpuResidencyCounters, GpuResidencyUniform};
use helio_planet_voxel_core::{GpuPageMeta, ResidencyConfig, PAGE_CELL_BYTES};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PlanetaryVoxelGpuConfig {
    pub max_resident_pages: u32,
    pub table_capacity: u32,
    pub max_probe: u32,
    pub max_batch_pages: u32,
    pub max_eviction_watermarks: u32,
}

impl PlanetaryVoxelGpuConfig {
    pub fn new(
        max_resident_pages: u32,
        table_capacity: u32,
        max_probe: u32,
        max_batch_pages: u32,
        max_eviction_watermarks: u32,
    ) -> Result<Self, GpuConfigError> {
        if max_resident_pages == 0 {
            return Err(GpuConfigError::ZeroResidentPages);
        }
        if !table_capacity.is_power_of_two() {
            return Err(GpuConfigError::TableCapacityNotPowerOfTwo(table_capacity));
        }
        let transient_entries = max_resident_pages
            .checked_add(1)
            .ok_or(GpuConfigError::ArithmeticOverflow)?;
        if transient_entries > table_capacity / 2 {
            return Err(GpuConfigError::TableLoadFactor {
                resident_pages: max_resident_pages,
                table_capacity,
            });
        }
        if max_probe == 0 || max_probe > table_capacity {
            return Err(GpuConfigError::InvalidMaxProbe {
                max_probe,
                table_capacity,
            });
        }
        if max_batch_pages == 0 || max_batch_pages > max_resident_pages {
            return Err(GpuConfigError::InvalidBatchPages {
                batch_pages: max_batch_pages,
                resident_pages: max_resident_pages,
            });
        }
        if max_eviction_watermarks == 0 {
            return Err(GpuConfigError::ZeroEvictionWatermarks);
        }
        let config = Self {
            max_resident_pages,
            table_capacity,
            max_probe,
            max_batch_pages,
            max_eviction_watermarks,
        };
        config.logical_gpu_bytes()?;
        for bytes in [
            config.cell_atlas_bytes()?,
            config.metadata_bytes()?,
            config.page_table_bytes()?,
        ] {
            usize::try_from(bytes).map_err(|_| GpuConfigError::ArithmeticOverflow)?;
        }
        Ok(config)
    }

    pub fn residency_config(self) -> Result<ResidencyConfig, GpuConfigError> {
        let max_resident_pages = usize::try_from(self.max_resident_pages)
            .map_err(|_| GpuConfigError::ArithmeticOverflow)?;
        let max_cell_bytes = usize::try_from(self.cell_atlas_bytes()?)
            .map_err(|_| GpuConfigError::ArithmeticOverflow)?;
        let max_eviction_watermarks = usize::try_from(self.max_eviction_watermarks)
            .map_err(|_| GpuConfigError::ArithmeticOverflow)?;
        ResidencyConfig::new(max_resident_pages, max_cell_bytes, max_eviction_watermarks)
            .map_err(|_| GpuConfigError::ArithmeticOverflow)
    }

    pub fn cell_atlas_bytes(self) -> Result<u64, GpuConfigError> {
        u64::from(self.max_resident_pages)
            .checked_mul(PAGE_CELL_BYTES as u64)
            .ok_or(GpuConfigError::ArithmeticOverflow)
    }

    pub fn metadata_bytes(self) -> Result<u64, GpuConfigError> {
        u64::from(self.max_resident_pages)
            .checked_mul(core::mem::size_of::<GpuPageMeta>() as u64)
            .ok_or(GpuConfigError::ArithmeticOverflow)
    }

    pub fn page_table_bytes(self) -> Result<u64, GpuConfigError> {
        u64::from(self.table_capacity)
            .checked_mul(core::mem::size_of::<GpuPageTableEntry>() as u64)
            .ok_or(GpuConfigError::ArithmeticOverflow)
    }

    /// Device-independent lower bound. [`Self::allocation_plan`] reports the
    /// exact allocation after accounting for unused tiles in the 3D atlas.
    pub fn logical_gpu_bytes(self) -> Result<u64, GpuConfigError> {
        [
            self.cell_atlas_bytes()?,
            self.metadata_bytes()?,
            self.page_table_bytes()?,
            core::mem::size_of::<GpuResidencyUniform>() as u64,
            core::mem::size_of::<GpuResidencyCounters>() as u64,
        ]
        .into_iter()
        .try_fold(0_u64, |total, bytes| {
            total
                .checked_add(bytes)
                .ok_or(GpuConfigError::ArithmeticOverflow)
        })
    }

    pub fn allocation_plan(
        self,
        limits: &wgpu::Limits,
    ) -> Result<GpuAllocationPlan, GpuConfigError> {
        let atlas =
            GpuAtlasTexturePlan::new(self.max_resident_pages, limits.max_texture_dimension_3d)?;
        let max_storage_bytes = limits.max_storage_buffer_binding_size;

        for (name, bytes, storage) in [
            ("metadata", self.metadata_bytes()?, true),
            ("page table", self.page_table_bytes()?, true),
            (
                "residency uniform",
                core::mem::size_of::<GpuResidencyUniform>() as u64,
                false,
            ),
            (
                "residency counters",
                core::mem::size_of::<GpuResidencyCounters>() as u64,
                true,
            ),
        ] {
            if bytes > limits.max_buffer_size || (storage && bytes > max_storage_bytes) {
                return Err(GpuConfigError::DeviceBufferLimit {
                    name,
                    requested: bytes,
                    max_buffer_bytes: limits.max_buffer_size,
                    max_storage_bytes,
                });
            }
        }
        let uniform_bytes = core::mem::size_of::<GpuResidencyUniform>() as u64;
        if uniform_bytes > limits.max_uniform_buffer_binding_size {
            return Err(GpuConfigError::UniformBindingLimit {
                requested: uniform_bytes,
                maximum: limits.max_uniform_buffer_binding_size,
            });
        }
        if limits.max_sampled_textures_per_shader_stage == 0 {
            return Err(GpuConfigError::SampledTextureLimit);
        }
        let total_bytes = [
            atlas.size_bytes,
            self.metadata_bytes()?,
            self.page_table_bytes()?,
            uniform_bytes,
            core::mem::size_of::<GpuResidencyCounters>() as u64,
        ]
        .into_iter()
        .try_fold(0_u64, |total, bytes| {
            total
                .checked_add(bytes)
                .ok_or(GpuConfigError::ArithmeticOverflow)
        })?;
        Ok(GpuAllocationPlan {
            atlas,
            metadata_bytes: self.metadata_bytes()?,
            page_table_bytes: self.page_table_bytes()?,
            total_bytes,
        })
    }
}

impl Default for PlanetaryVoxelGpuConfig {
    fn default() -> Self {
        Self::new(256, 1024, 32, 16, 512).expect("default planetary GPU residency budget is valid")
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GpuAllocationPlan {
    pub atlas: GpuAtlasTexturePlan,
    pub metadata_bytes: u64,
    pub page_table_bytes: u64,
    pub total_bytes: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GpuAtlasTexturePlan {
    pub tile_count: [u32; 3],
    pub extent: [u32; 3],
    pub capacity_pages: u32,
    pub size_bytes: u64,
}

impl GpuAtlasTexturePlan {
    fn new(max_resident_pages: u32, max_dimension: u32) -> Result<Self, GpuConfigError> {
        let page_edge = helio_planet_voxel_core::PAGE_EDGE as u32;
        let max_tiles = max_dimension / page_edge;
        if max_tiles == 0 {
            return Err(GpuConfigError::DeviceCannotFitPageTexture {
                page_edge,
                max_dimension,
            });
        }

        let mut best: Option<([u32; 3], u64, u32)> = None;
        for x in 1..=max_tiles {
            for y in x..=max_tiles {
                let xy = u64::from(x) * u64::from(y);
                let required_z = u64::from(max_resident_pages).div_ceil(xy);
                let z = y.max(
                    u32::try_from(required_z).map_err(|_| GpuConfigError::ArithmeticOverflow)?,
                );
                if z > max_tiles {
                    continue;
                }
                let capacity = xy
                    .checked_mul(u64::from(z))
                    .ok_or(GpuConfigError::ArithmeticOverflow)?;
                let spread = z - x;
                let candidate = ([x, y, z], capacity, spread);
                if best.is_none_or(|(_, best_capacity, best_spread)| {
                    (capacity, spread) < (best_capacity, best_spread)
                }) {
                    best = Some(candidate);
                }
            }
        }
        let (tile_count, capacity, _) = best.ok_or(GpuConfigError::AtlasTextureCapacity {
            required_pages: max_resident_pages,
            maximum_pages: max_tiles.saturating_pow(3),
            max_dimension,
        })?;
        let extent = tile_count.map(|tiles| tiles * page_edge);
        let size_bytes = capacity
            .checked_mul(PAGE_CELL_BYTES as u64)
            .ok_or(GpuConfigError::ArithmeticOverflow)?;
        Ok(Self {
            tile_count,
            extent,
            capacity_pages: u32::try_from(capacity)
                .map_err(|_| GpuConfigError::ArithmeticOverflow)?,
            size_bytes,
        })
    }

    pub fn origin_for_slot(self, slot: u32) -> Option<[u32; 3]> {
        if slot >= self.capacity_pages {
            return None;
        }
        let [tiles_x, tiles_y, _] = self.tile_count;
        let tile_x = slot % tiles_x;
        let tile_y = (slot / tiles_x) % tiles_y;
        let tile_z = slot / (tiles_x * tiles_y);
        let edge = helio_planet_voxel_core::PAGE_EDGE as u32;
        Some([tile_x * edge, tile_y * edge, tile_z * edge])
    }
}

#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum GpuConfigError {
    #[error("planetary GPU residency needs at least one page")]
    ZeroResidentPages,
    #[error("page-table capacity {0} must be a non-zero power of two")]
    TableCapacityNotPowerOfTwo(u32),
    #[error(
        "page-table capacity {table_capacity} must keep resident page count {resident_pages} plus one transactional entry at or below 50% load"
    )]
    TableLoadFactor {
        resident_pages: u32,
        table_capacity: u32,
    },
    #[error("maximum probe count {max_probe} must be within 1..={table_capacity}")]
    InvalidMaxProbe { max_probe: u32, table_capacity: u32 },
    #[error("batch page count {batch_pages} must be within 1..={resident_pages}")]
    InvalidBatchPages {
        batch_pages: u32,
        resident_pages: u32,
    },
    #[error("planetary GPU residency needs at least one eviction watermark")]
    ZeroEvictionWatermarks,
    #[error("planetary GPU residency byte arithmetic overflowed")]
    ArithmeticOverflow,
    #[error(
        "device 3D texture dimension {max_dimension} cannot fit one {page_edge}-cell page edge"
    )]
    DeviceCannotFitPageTexture { page_edge: u32, max_dimension: u32 },
    #[error(
        "3D atlas needs {required_pages} page tiles, but a {max_dimension} texture can hold at most {maximum_pages}"
    )]
    AtlasTextureCapacity {
        required_pages: u32,
        maximum_pages: u32,
        max_dimension: u32,
    },
    #[error("planetary GPU sampling needs one sampled texture binding")]
    SampledTextureLimit,
    #[error(
        "{name} buffer requests {requested} bytes (buffer limit {max_buffer_bytes}, storage binding limit {max_storage_bytes})"
    )]
    DeviceBufferLimit {
        name: &'static str,
        requested: u64,
        max_buffer_bytes: u64,
        max_storage_bytes: u64,
    },
    #[error("residency uniform requests {requested} bytes; device binding limit is {maximum}")]
    UniformBindingLimit { requested: u64, maximum: u64 },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn budgets_use_checked_exact_bytes() {
        let config = PlanetaryVoxelGpuConfig::new(4, 16, 8, 2, 8).unwrap();
        assert_eq!(config.cell_atlas_bytes().unwrap(), 4 * 131_072);
        assert_eq!(config.metadata_bytes().unwrap(), 4 * 32);
        assert_eq!(config.page_table_bytes().unwrap(), 16 * 48);
    }

    #[test]
    fn table_load_and_probe_limits_are_explicit() {
        assert!(matches!(
            PlanetaryVoxelGpuConfig::new(8, 16, 8, 1, 1),
            Err(GpuConfigError::TableLoadFactor { .. })
        ));
        assert!(matches!(
            PlanetaryVoxelGpuConfig::new(4, 16, 17, 1, 1),
            Err(GpuConfigError::InvalidMaxProbe { .. })
        ));
    }

    #[test]
    fn allocation_plan_packs_pages_into_balanced_3d_tiles() {
        let config = PlanetaryVoxelGpuConfig::new(384, 1024, 64, 192, 384).unwrap();
        let limits = wgpu::Limits {
            max_texture_dimension_3d: 2_048,
            ..wgpu::Limits::downlevel_defaults()
        };
        let plan = config.allocation_plan(&limits).unwrap();
        assert_eq!(plan.atlas.tile_count, [6, 8, 8]);
        assert_eq!(plan.atlas.capacity_pages, 384);
        assert_eq!(plan.atlas.extent, [192, 256, 256]);
        assert_eq!(plan.atlas.origin_for_slot(0), Some([0, 0, 0]));
        assert_eq!(plan.atlas.origin_for_slot(383), Some([160, 224, 224]));
        assert_eq!(plan.atlas.origin_for_slot(384), None);
    }

    #[test]
    fn allocation_plan_rejects_insufficient_3d_texture_capacity() {
        let config = PlanetaryVoxelGpuConfig::new(9, 32, 8, 2, 8).unwrap();
        let limits = wgpu::Limits {
            max_texture_dimension_3d: 64,
            ..wgpu::Limits::downlevel_defaults()
        };
        assert_eq!(
            config.allocation_plan(&limits),
            Err(GpuConfigError::AtlasTextureCapacity {
                required_pages: 9,
                maximum_pages: 8,
                max_dimension: 64,
            })
        );
    }
}
