use crate::{GpuPageTableEntry, GpuResidencyCounters, GpuResidencyUniform};
use helio_planet_voxel_core::{GpuPageMeta, ResidencyConfig, PAGE_CELL_BYTES};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PlanetaryVoxelGpuConfig {
    pub max_resident_pages: u32,
    pub table_capacity: u32,
    pub max_probe: u32,
    pub max_batch_pages: u32,
    pub max_eviction_watermarks: u32,
    pub max_atlas_shards: u32,
}

impl PlanetaryVoxelGpuConfig {
    pub fn new(
        max_resident_pages: u32,
        table_capacity: u32,
        max_probe: u32,
        max_batch_pages: u32,
        max_eviction_watermarks: u32,
        max_atlas_shards: u32,
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
        if max_atlas_shards == 0 {
            return Err(GpuConfigError::ZeroAtlasShards);
        }
        let config = Self {
            max_resident_pages,
            table_capacity,
            max_probe,
            max_batch_pages,
            max_eviction_watermarks,
            max_atlas_shards,
        };
        config.total_gpu_bytes()?;
        for bytes in [
            config.cell_atlas_bytes()?,
            config.metadata_bytes()?,
            config.page_table_bytes()?,
            config.staging_bytes()?,
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

    pub fn staging_bytes(self) -> Result<u64, GpuConfigError> {
        u64::from(self.max_batch_pages)
            .checked_mul(PAGE_CELL_BYTES as u64)
            .ok_or(GpuConfigError::ArithmeticOverflow)
    }

    pub fn total_gpu_bytes(self) -> Result<u64, GpuConfigError> {
        [
            self.cell_atlas_bytes()?,
            self.metadata_bytes()?,
            self.page_table_bytes()?,
            self.staging_bytes()?,
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
        let page_bytes = PAGE_CELL_BYTES as u64;
        let max_storage_bytes = limits.max_storage_buffer_binding_size;
        let max_shard_bytes = limits.max_buffer_size.min(max_storage_bytes);
        let pages_per_shard = (max_shard_bytes / page_bytes).min(u64::from(u32::MAX));
        if pages_per_shard == 0 {
            return Err(GpuConfigError::DeviceCannotFitPage {
                page_bytes,
                max_buffer_bytes: limits.max_buffer_size,
                max_storage_bytes,
            });
        }
        let shard_count = u64::from(self.max_resident_pages).div_ceil(pages_per_shard);
        if shard_count > u64::from(self.max_atlas_shards) {
            return Err(GpuConfigError::AtlasShardLimit {
                required: shard_count as u32,
                maximum: self.max_atlas_shards,
            });
        }
        let atlas_binding_limit = limits
            .max_storage_buffers_per_shader_stage
            .saturating_sub(2);
        if shard_count > u64::from(atlas_binding_limit) {
            return Err(GpuConfigError::AtlasBindingLimit {
                required: shard_count as u32,
                available: atlas_binding_limit,
            });
        }

        for (name, bytes, storage) in [
            ("metadata", self.metadata_bytes()?, true),
            ("page table", self.page_table_bytes()?, true),
            ("staging", self.staging_bytes()?, false),
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

        let mut shards = Vec::with_capacity(shard_count as usize);
        let mut page_start = 0_u32;
        while page_start < self.max_resident_pages {
            let remaining = self.max_resident_pages - page_start;
            let page_count = remaining.min(pages_per_shard as u32);
            shards.push(GpuAtlasShardPlan {
                page_start,
                page_count,
                size_bytes: u64::from(page_count) * page_bytes,
            });
            page_start += page_count;
        }
        Ok(GpuAllocationPlan {
            shards,
            metadata_bytes: self.metadata_bytes()?,
            page_table_bytes: self.page_table_bytes()?,
            staging_bytes: self.staging_bytes()?,
            total_bytes: self.total_gpu_bytes()?,
        })
    }
}

impl Default for PlanetaryVoxelGpuConfig {
    fn default() -> Self {
        Self::new(256, 1024, 32, 16, 512, 16)
            .expect("default planetary GPU residency budget is valid")
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GpuAllocationPlan {
    pub shards: Vec<GpuAtlasShardPlan>,
    pub metadata_bytes: u64,
    pub page_table_bytes: u64,
    pub staging_bytes: u64,
    pub total_bytes: u64,
}

impl GpuAllocationPlan {
    pub fn shard_for_slot(&self, slot: u32) -> Option<(usize, u64)> {
        self.shards.iter().enumerate().find_map(|(index, shard)| {
            let local = slot.checked_sub(shard.page_start)?;
            (local < shard.page_count).then_some((index, u64::from(local) * PAGE_CELL_BYTES as u64))
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GpuAtlasShardPlan {
    pub page_start: u32,
    pub page_count: u32,
    pub size_bytes: u64,
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
    #[error("planetary GPU residency needs at least one atlas shard")]
    ZeroAtlasShards,
    #[error("planetary GPU residency byte arithmetic overflowed")]
    ArithmeticOverflow,
    #[error(
        "device cannot fit one {page_bytes}-byte page (buffer {max_buffer_bytes}, storage binding {max_storage_bytes})"
    )]
    DeviceCannotFitPage {
        page_bytes: u64,
        max_buffer_bytes: u64,
        max_storage_bytes: u64,
    },
    #[error("cell atlas needs {required} shards, exceeding configured maximum {maximum}")]
    AtlasShardLimit { required: u32, maximum: u32 },
    #[error(
        "cell atlas needs {required} storage bindings, but the device has {available} after reserving metadata and page-table bindings"
    )]
    AtlasBindingLimit { required: u32, available: u32 },
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
        let config = PlanetaryVoxelGpuConfig::new(4, 16, 8, 2, 8, 4).unwrap();
        assert_eq!(config.cell_atlas_bytes().unwrap(), 4 * 131_072);
        assert_eq!(config.metadata_bytes().unwrap(), 4 * 32);
        assert_eq!(config.page_table_bytes().unwrap(), 16 * 48);
        assert_eq!(config.staging_bytes().unwrap(), 2 * 131_072);
    }

    #[test]
    fn table_load_and_probe_limits_are_explicit() {
        assert!(matches!(
            PlanetaryVoxelGpuConfig::new(8, 16, 8, 1, 1, 1),
            Err(GpuConfigError::TableLoadFactor { .. })
        ));
        assert!(matches!(
            PlanetaryVoxelGpuConfig::new(4, 16, 17, 1, 1, 1),
            Err(GpuConfigError::InvalidMaxProbe { .. })
        ));
    }

    #[test]
    fn allocation_plan_shards_at_storage_binding_limit() {
        let config = PlanetaryVoxelGpuConfig::new(4, 16, 8, 2, 8, 4).unwrap();
        let limits = wgpu::Limits {
            max_buffer_size: 2 * PAGE_CELL_BYTES as u64,
            max_storage_buffer_binding_size: (2 * PAGE_CELL_BYTES) as u64,
            ..wgpu::Limits::downlevel_defaults()
        };
        let plan = config.allocation_plan(&limits).unwrap();
        assert_eq!(plan.shards.len(), 2);
        assert_eq!(plan.shard_for_slot(0), Some((0, 0)));
        assert_eq!(plan.shard_for_slot(2), Some((1, 0)));
        assert_eq!(plan.shard_for_slot(4), None);
    }

    #[test]
    fn allocation_plan_rejects_unbindable_atlas_shards() {
        let config = PlanetaryVoxelGpuConfig::new(4, 16, 8, 2, 8, 4).unwrap();
        let limits = wgpu::Limits {
            max_buffer_size: PAGE_CELL_BYTES as u64,
            max_storage_buffer_binding_size: PAGE_CELL_BYTES as u64,
            max_storage_buffers_per_shader_stage: 4,
            ..wgpu::Limits::downlevel_defaults()
        };
        assert_eq!(
            config.allocation_plan(&limits),
            Err(GpuConfigError::AtlasBindingLimit {
                required: 4,
                available: 2,
            })
        );
    }
}
