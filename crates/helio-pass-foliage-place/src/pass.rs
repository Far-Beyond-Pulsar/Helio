//! The `RenderPass` implementation: buffers, pipelines and the four dispatches.

use std::sync::Arc;

use helio_core::{PassContext, PrepareContext, RenderPass, Result as HelioResult};
use helio_foliage_core::{
    FoliageQuality, GpuBladeInstance, GpuFoliageTile, GpuFoliageType, TileState,
    DEFAULT_MAX_TILES_PER_FRAME, DEFAULT_TILE_RING_CAPACITY, FOLIAGE_TILE_SIZE_METERS,
};

use crate::residency::TileRing;
use crate::uniforms::{FoliageCullUniforms, PlaceUniforms};
use crate::{
    foliage_frame_is_present, COUNTER_PLACEMENT_OVERFLOW, COUNTER_PLACED_BLADES,
    COUNTER_VISIBLE_OVERFLOW, DEFAULT_WPO_EXTENT_METERS, FOLIAGE_COUNTER_COUNT,
    FOLIAGE_LOD_VERTEX_COUNTS, FOLIAGE_VISIBLE_PER_LOD_CAPACITY, MAX_CANDIDATES_PER_TILE,
};

const TILE_BYTES: u64 = std::mem::size_of::<GpuFoliageTile>() as u64;
const BLADE_BYTES: u64 = std::mem::size_of::<GpuBladeInstance>() as u64;
const TYPE_BYTES: u64 = std::mem::size_of::<GpuFoliageType>() as u64;

/// Hard ceiling on the foliage type table.
///
/// `GpuBladeInstance` stores the type id in 8 bits, so 256 is not a budget choice — it is
/// the representable maximum. Publishing more types than this cannot work, and clamping
/// with a warning is better than blades silently rendering as type `id % 256`.
const MAX_FOLIAGE_TYPES: u32 = 256;

/// Compute workgroup size shared by `cs_place`, `cs_tile_cull` and `cs_cluster_cull`.
const WORKGROUP_SIZE: u32 = 64;

/// Foliage tile residency, GPU placement, tile/cluster culling and per-LOD compaction.
///
/// See the crate docs for the design and for the interface contract with
/// `FoliageGBufferPass`. The four `pub` buffer fields below are that contract: the graph
/// builder clones the `Arc`s into the consumer's constructor, the way `ShadowDirtyPass`
/// hands `face_dirty_buf` to `ShadowCullPass`.
pub struct FoliagePlacePass {
    // ── Shared with FoliageGBufferPass ──────────────────────────────────────────
    /// `GpuBladeInstance[]`, partitioned into one fixed slab per ring slot.
    ///
    /// Fixed slabs rather than a bump allocator, and that is a correctness choice as much
    /// as a simplicity one: with a bump allocator a tile's `blade_offset` would depend on
    /// the order tiles happened to be placed in, so an evict/re-place cycle would move a
    /// tile's blades in memory and the arena would fragment under ring churn. Equal slabs
    /// also make `blade_index / blades_per_tile` an exact O(1) recovery of the owning
    /// tile, which is how the consumer finds a blade's tile header.
    pub blade_arena: Arc<wgpu::Buffer>,

    /// `GpuFoliageTile[]` — the ring's slot headers. Written by the CPU for coordinates
    /// and residency state, by `cs_place` for `blade_count` and the vertical bounds.
    pub tile_table: Arc<wgpu::Buffer>,

    /// `u32` blade indices in four contiguous regions, one per LOD, addressed with
    /// [`crate::lod_region_offset`]. Indices are *global* arena indices.
    pub visible_blades: Arc<wgpu::Buffer>,

    /// Exactly four `wgpu::util::DrawIndirectArgs` at byte offsets 0/16/32/48, in LOD
    /// order. Vertex counts are [`FOLIAGE_LOD_VERTEX_COUNTS`]; `first_instance` is 0 for
    /// all four.
    pub foliage_indirect: Arc<wgpu::Buffer>,

    /// Per-LOD visible counts plus the overflow telemetry. See the `COUNTER_*` constants.
    counters: Arc<wgpu::Buffer>,

    // ── Internal ────────────────────────────────────────────────────────────────
    type_table: wgpu::Buffer,
    place_queue: wgpu::Buffer,
    tile_visibility: wgpu::Buffer,
    place_uniforms: wgpu::Buffer,
    cull_uniforms: wgpu::Buffer,

    place_pipeline: wgpu::ComputePipeline,
    tile_cull_pipeline: wgpu::ComputePipeline,
    cluster_cull_pipeline: wgpu::ComputePipeline,
    finalize_pipeline: wgpu::ComputePipeline,

    cull_bgl: wgpu::BindGroupLayout,
    place_bind_group: wgpu::BindGroup,
    cull_bind_group: Option<wgpu::BindGroup>,
    cull_bind_group_key: Option<(usize, usize, usize)>,

    placeholder_hiz: wgpu::TextureView,
    placeholder_hiz_sampler: wgpu::Sampler,
    placeholder_terrain: wgpu::TextureView,
    terrain_sampler: wgpu::Sampler,

    ring: TileRing,
    quality: FoliageQuality,
    ring_capacity: u32,
    blades_per_tile: u32,
    cluster_size: u32,
    clusters_per_tile: u32,

    terrain_transform: Option<([f32; 2], f32)>,

    // Per-frame state handed from `prepare` to `execute`.
    active: bool,
    queued_tile_count: u32,
    cluster_dispatch_width: u32,
    cluster_dispatch_height: u32,
    tile_dispatch_groups: u32,

    last_type_generation: Option<u64>,
    density_scale: f32,
    warned_density_clamp: bool,
    warned_type_overflow: bool,
    commands_recorded: u64,

    /// Non-blocking readback of [`Self::counters`], for diagnosing an empty field.
    ///
    /// Worth its ~100 lines: every way this pass can fail quietly — no tiles queued, no
    /// candidates accepted, everything culled, the visible region overflowing — looks
    /// identical from the outside, namely bare ground. These counters distinguish them.
    counters_readback: wgpu::Buffer,
    readback_state: ReadbackState,
    debug_counters: [u32; FOLIAGE_COUNTER_COUNT],

    header_scratch: Vec<GpuFoliageTile>,
    dirty_scratch: Vec<u32>,
}

impl FoliagePlacePass {
    /// Create the pass and allocate every buffer it will ever use.
    ///
    /// Nothing here grows: the ring capacity, arena size and visible-list capacity are all
    /// fixed by `quality`, so there is no reallocation path and therefore no bind-group
    /// versioning for our own buffers. That is deliberate — a growable arena would make
    /// `blade_offset` depend on history, which is exactly what the fixed-slab layout
    /// exists to prevent.
    pub fn new(device: &wgpu::Device, quality: FoliageQuality) -> Self {
        let tiles_across =
            ((2.0 * quality.ring_radius() / FOLIAGE_TILE_SIZE_METERS).ceil() as u32).max(1);
        // The ring is a hard ceiling. If a preset's footprint ever exceeded it the ring
        // would thrash rather than fail — re-placing the same tiles every frame and
        // turning the amortised placement cost into a per-frame one — so clamp loudly
        // rather than allocating an unbounded table.
        let ring_capacity = tiles_across
            .saturating_mul(tiles_across)
            .min(DEFAULT_TILE_RING_CAPACITY)
            .max(1);
        if tiles_across.saturating_mul(tiles_across) > DEFAULT_TILE_RING_CAPACITY {
            log::warn!(
                "foliage quality {quality:?} needs a {tiles_across}x{tiles_across} tile ring \
                 but the table holds {DEFAULT_TILE_RING_CAPACITY}; the ring will thrash"
            );
        }

        let blades_per_tile = (quality.blade_capacity() / ring_capacity as u64).max(1) as u32;
        let arena_blades = ring_capacity as u64 * blades_per_tile as u64;
        let cluster_size = quality.cluster_granularity().max(1);
        let clusters_per_tile = blades_per_tile.div_ceil(cluster_size).max(1);

        let blade_arena = Arc::new(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Foliage Blade Arena"),
            size: arena_blades * BLADE_BYTES,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));
        let tile_table = Arc::new(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Foliage Tile Table"),
            size: ring_capacity as u64 * TILE_BYTES,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));
        let visible_blades = Arc::new(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Foliage Visible Blades"),
            size: FOLIAGE_LOD_VERTEX_COUNTS.len() as u64
                * FOLIAGE_VISIBLE_PER_LOD_CAPACITY as u64
                * std::mem::size_of::<u32>() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));
        let foliage_indirect = Arc::new(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Foliage Indirect Draws"),
            size: FOLIAGE_LOD_VERTEX_COUNTS.len() as u64
                * std::mem::size_of::<wgpu::util::DrawIndirectArgs>() as u64,
            usage: wgpu::BufferUsages::INDIRECT
                | wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));
        let counters = Arc::new(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Foliage Counters"),
            size: FOLIAGE_COUNTER_COUNT as u64 * 4,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        }));

        let type_table = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Foliage Type Table"),
            size: MAX_FOLIAGE_TYPES as u64 * TYPE_BYTES,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let place_queue = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Foliage Place Queue"),
            size: (DEFAULT_MAX_TILES_PER_FRAME.max(1) as u64) * 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let tile_visibility = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Foliage Tile Visibility"),
            size: ring_capacity as u64 * 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let place_uniforms = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Foliage Place Uniforms"),
            size: std::mem::size_of::<PlaceUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let cull_uniforms = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Foliage Cull Uniforms"),
            size: std::mem::size_of::<FoliageCullUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // ── Placeholders ───────────────────────────────────────────────────────
        //
        // Both exist so a missing upstream resource degrades into a well-defined bind
        // rather than an unwrap. The Hi-Z placeholder is only ever reached before the
        // pyramid is routed, and `hiz_valid` is 0 whenever it is bound, so nothing is
        // culled against it.
        let placeholder_hiz = device
            .create_texture(&wgpu::TextureDescriptor {
                label: Some("Foliage Placeholder HiZ"),
                size: wgpu::Extent3d {
                    width: 1,
                    height: 1,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::R32Float,
                usage: wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            })
            .create_view(&wgpu::TextureViewDescriptor::default());
        let placeholder_hiz_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Foliage Placeholder HiZ Sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            ..Default::default()
        });
        let placeholder_terrain = device
            .create_texture(&wgpu::TextureDescriptor {
                label: Some("Foliage Placeholder Terrain"),
                size: wgpu::Extent3d {
                    width: 1,
                    height: 1,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rg16Float,
                usage: wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            })
            .create_view(&wgpu::TextureViewDescriptor::default());
        // Clamp-to-edge, not repeat: the capture is a camera-relative ring, and a
        // repeating address mode would sample terrain from the opposite side of it.
        let terrain_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Foliage Terrain Sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            ..Default::default()
        });

        // ── Pipelines ──────────────────────────────────────────────────────────
        let place_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Foliage Place Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/foliage_place.wgsl").into()),
        });
        let cull_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Foliage Cull Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/foliage_cull.wgsl").into()),
        });

        let place_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Foliage Place BGL"),
            entries: &[
                uniform_entry(0),
                storage_entry(1, true),
                storage_entry(2, false),
                storage_entry(3, false),
                storage_entry(4, true),
                storage_entry(5, false),
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 7,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        let cull_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Foliage Cull BGL"),
            entries: &[
                // Camera is a read-only storage array (mono uses index 0; a future
                // single-pass stereo cull can union `cameras[0]` and `cameras[1]`).
                storage_entry(0, true),
                uniform_entry(1),
                storage_entry(2, true),
                storage_entry(3, true),
                storage_entry(4, true),
                storage_entry(5, false),
                storage_entry(6, false),
                storage_entry(7, false),
                storage_entry(8, false),
                wgpu::BindGroupLayoutEntry {
                    binding: 9,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 10,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
                    count: None,
                },
            ],
        });

        let place_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Foliage Place PL"),
            bind_group_layouts: &[Some(&place_bgl)],
            immediate_size: 0,
        });
        let cull_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Foliage Cull PL"),
            bind_group_layouts: &[Some(&cull_bgl)],
            immediate_size: 0,
        });

        let place_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Foliage Place Pipeline"),
            layout: Some(&place_layout),
            module: &place_shader,
            entry_point: Some("cs_place"),
            compilation_options: Default::default(),
            cache: None,
        });
        let tile_cull_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Foliage Tile Cull Pipeline"),
            layout: Some(&cull_layout),
            module: &cull_shader,
            entry_point: Some("cs_tile_cull"),
            compilation_options: Default::default(),
            cache: None,
        });
        let cluster_cull_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Foliage Cluster Cull Pipeline"),
                layout: Some(&cull_layout),
                module: &cull_shader,
                entry_point: Some("cs_cluster_cull"),
                compilation_options: Default::default(),
                cache: None,
            });
        let finalize_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Foliage Finalize Pipeline"),
            layout: Some(&cull_layout),
            module: &cull_shader,
            entry_point: Some("cs_finalize"),
            compilation_options: Default::default(),
            cache: None,
        });

        // Every binding in the place group is owned by this pass and never reallocated,
        // so this bind group is built once and never rebuilt.
        let place_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Foliage Place BG"),
            layout: &place_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: place_uniforms.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: type_table.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: tile_table.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: blade_arena.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: place_queue.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: counters.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: wgpu::BindingResource::TextureView(&placeholder_terrain),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: wgpu::BindingResource::Sampler(&terrain_sampler),
                },
            ],
        });

        Self {
            blade_arena,
            tile_table,
            visible_blades,
            foliage_indirect,
            counters,
            type_table,
            place_queue,
            tile_visibility,
            place_uniforms,
            cull_uniforms,
            place_pipeline,
            tile_cull_pipeline,
            cluster_cull_pipeline,
            finalize_pipeline,
            cull_bgl,
            place_bind_group,
            cull_bind_group: None,
            cull_bind_group_key: None,
            placeholder_hiz,
            placeholder_hiz_sampler,
            placeholder_terrain,
            terrain_sampler,
            ring: TileRing::new(
                ring_capacity,
                tiles_across,
                FOLIAGE_TILE_SIZE_METERS,
                DEFAULT_MAX_TILES_PER_FRAME,
            ),
            quality,
            ring_capacity,
            blades_per_tile,
            cluster_size,
            clusters_per_tile,
            terrain_transform: None,
            active: false,
            queued_tile_count: 0,
            cluster_dispatch_width: 1,
            cluster_dispatch_height: 1,
            tile_dispatch_groups: ring_capacity.div_ceil(WORKGROUP_SIZE).max(1),
            last_type_generation: None,
            density_scale: 1.0,
            warned_density_clamp: false,
            warned_type_overflow: false,
            commands_recorded: 0,
            counters_readback: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Foliage Counters Readback"),
                size: FOLIAGE_COUNTER_COUNT as u64 * 4,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            readback_state: ReadbackState::Idle,
            debug_counters: [0; FOLIAGE_COUNTER_COUNT],
            header_scratch: Vec::new(),
            dirty_scratch: Vec::new(),
        }
    }

    /// Quality preset this pass was built for. Changing it means rebuilding the pass —
    /// every buffer capacity is derived from it.
    #[inline]
    pub fn quality(&self) -> FoliageQuality {
        self.quality
    }

    /// Number of slots in the tile ring, i.e. entries in [`FoliagePlacePass::tile_table`].
    #[inline]
    pub fn ring_capacity(&self) -> u32 {
        self.ring_capacity
    }

    /// Blades one tile owns in the arena.
    ///
    /// Part of the interface with `FoliageGBufferPass`: slabs are fixed and equal, so a
    /// visible blade index maps back to its tile header with
    /// `tile_slot = blade_index / blades_per_tile`. There is no search and no side table.
    #[inline]
    pub fn blades_per_tile(&self) -> u32 {
        self.blades_per_tile
    }

    /// Blades per cull cluster, from [`FoliageQuality::cluster_granularity`].
    #[inline]
    pub fn cluster_size(&self) -> u32 {
        self.cluster_size
    }

    /// Per-LOD visible counts and overflow telemetry. See the `COUNTER_*` constants.
    #[inline]
    pub fn counters(&self) -> &Arc<wgpu::Buffer> {
        &self.counters
    }

    /// Fraction of the authored density actually placed, in `0.0..=1.0`.
    ///
    /// Below 1.0 means a tile's stratified grid was clamped so it could not generate more
    /// candidates than its arena slab holds. That is a *uniform thinning* of the grass,
    /// which is the right degradation: letting the candidates run and dropping the
    /// overflow would instead delete whichever blades lost the race, and because the
    /// compaction is ordered that is always the same corner of every tile.
    #[inline]
    pub fn density_scale(&self) -> f32 {
        self.density_scale
    }

    /// Frames on which `execute` actually recorded GPU commands.
    ///
    /// The instrument behind the plan's §10 zero-overhead guarantee: with no foliage
    /// registered this must never advance.
    #[inline]
    pub fn commands_recorded(&self) -> u64 {
        self.commands_recorded
    }

    /// Latest non-blocking readback of the counters buffer. May trail by a frame or two.
    ///
    /// Slots 0..4 are the per-LOD visible instance counts, then
    /// [`COUNTER_VISIBLE_OVERFLOW`], [`COUNTER_PLACEMENT_OVERFLOW`] and
    /// [`COUNTER_PLACED_BLADES`].
    pub fn debug_counters(&self) -> [u32; FOLIAGE_COUNTER_COUNT] {
        self.debug_counters
    }

    /// Tiles queued for placement this frame.
    pub fn queued_tile_count(&self) -> u32 {
        self.queued_tile_count
    }

    /// Advance the non-blocking readback. Called once per frame from `prepare`.
    fn pump_readback(&mut self, device: &wgpu::Device) {
        match std::mem::replace(&mut self.readback_state, ReadbackState::Idle) {
            ReadbackState::Idle => {}
            ReadbackState::CopySubmitted => {
                let flag = std::sync::Arc::new(std::sync::Mutex::new(None));
                let sink = std::sync::Arc::clone(&flag);
                self.counters_readback
                    .slice(..)
                    .map_async(wgpu::MapMode::Read, move |result| {
                        *sink.lock().unwrap() = Some(result);
                    });
                self.readback_state = ReadbackState::Mapping(flag);
            }
            ReadbackState::Mapping(flag) => {
                let done = flag.lock().unwrap().take();
                match done {
                    Some(Ok(())) => {
                        if let Ok(view) = self.counters_readback.slice(..).get_mapped_range() {
                            if let Ok(values) = bytemuck::try_cast_slice::<u8, u32>(&view) {
                                if values.len() >= FOLIAGE_COUNTER_COUNT {
                                    self.debug_counters
                                        .copy_from_slice(&values[..FOLIAGE_COUNTER_COUNT]);
                                }
                            }
                        }
                        self.counters_readback.unmap();
                        self.readback_state = ReadbackState::Idle;
                    }
                    Some(Err(_)) => {
                        self.readback_state = ReadbackState::Idle;
                    }
                    None => {
                        // Still in flight; poll again next frame.
                        self.readback_state = ReadbackState::Mapping(flag);
                    }
                }
            }
        }
        let _ = device;
    }

    /// Read-only view of the residency ring, for tests and debug overlays.
    #[inline]
    pub fn ring(&self) -> &TileRing {
        &self.ring
    }

    /// Publish the terrain capture's ring transform: `(min-corner world XZ, side length
    /// in metres)`.
    ///
    /// Until this is called the placement shader runs its temporary flat-plane fallback,
    /// because `FoliageTerrainPass` does not exist yet and a bound texture with an unknown
    /// transform is worse than no texture at all — it would sample terrain from arbitrary
    /// world positions. Pass `None` to go back to the fallback.
    pub fn set_terrain_transform(&mut self, transform: Option<([f32; 2], f32)>) {
        self.terrain_transform = transform.filter(|(origin, extent)| {
            origin[0].is_finite() && origin[1].is_finite() && extent.is_finite() && *extent > 0.0
        });
    }

    /// Tallest foliage and densest type in the published table.
    fn table_stats(&self, types: &[GpuFoliageType]) -> (f32, f32) {
        let mut max_height = 0.0f32;
        let mut max_density = 0.0f32;
        for foliage in types {
            if foliage.height_range[1].is_finite() {
                max_height = max_height.max(foliage.height_range[1]);
            }
            if foliage.density.is_finite() && foliage.density > 0.0 {
                max_density = max_density.max(foliage.density);
            }
        }
        (max_height, max_density)
    }

    /// Resolution of the stratified candidate grid, and the density fraction it achieves.
    fn candidate_grid(&self, max_density: f32) -> (u32, f32) {
        let scaled_density = max_density * self.quality.density_multiplier();
        if !scaled_density.is_finite() || scaled_density <= 0.0 {
            return (1, 1.0);
        }

        let area = FOLIAGE_TILE_SIZE_METERS * FOLIAGE_TILE_SIZE_METERS;
        let desired = (scaled_density * area).ceil();
        if !desired.is_finite() || desired <= 0.0 {
            return (1, 1.0);
        }

        let ideal = (desired.sqrt().ceil() as u32).max(1);
        // Two independent ceilings: the tile's arena slab, and a fixed bound on the
        // shader's inner loop so a mis-authored density cannot turn one workgroup into a
        // multi-millisecond stall.
        let slab_limit = isqrt(self.blades_per_tile).max(1);
        let loop_limit = isqrt(MAX_CANDIDATES_PER_TILE).max(1);
        let grid = ideal.min(slab_limit).min(loop_limit);
        let achieved = ((grid as f32 * grid as f32) / desired).clamp(0.0, 1.0);
        (grid, achieved)
    }

    /// Upload the headers of every slot the ring touched this frame.
    ///
    /// Contiguous slots are coalesced into one write, which matters on the teleport path
    /// where the whole table turns over in a single frame; in steady state the dirty set
    /// is a perimeter strip and this is a handful of small writes.
    fn upload_tile_headers(&mut self, ctx: &PrepareContext, generation: u32) {
        self.dirty_scratch.clear();
        self.dirty_scratch.extend_from_slice(self.ring.dirty_slots());
        if self.dirty_scratch.is_empty() {
            return;
        }
        self.dirty_scratch.sort_unstable();
        self.dirty_scratch.dedup();

        let mut index = 0usize;
        while index < self.dirty_scratch.len() {
            let start = self.dirty_scratch[index];
            let mut end = index;
            while end + 1 < self.dirty_scratch.len()
                && self.dirty_scratch[end + 1] == self.dirty_scratch[end] + 1
            {
                end += 1;
            }

            self.header_scratch.clear();
            for slot in start..=self.dirty_scratch[end] {
                self.header_scratch.push(match self.ring.slot_coord(slot) {
                    Some(coord) => GpuFoliageTile {
                        tile_coord: coord,
                        blade_offset: slot * self.blades_per_tile,
                        blade_count: 0,
                        bounds_center_y: 0.0,
                        bounds_half_y: 0.0,
                        // The GPU flips this to `Resident` at the end of `cs_place`.
                        // Publishing `Resident` from here instead would let the cull
                        // read a slab that placement has not written yet, drawing the
                        // previous tenant's blades for one frame.
                        state: TileState::Placing.as_u32(),
                        generation,
                    },
                    None => GpuFoliageTile::EMPTY,
                });
            }
            ctx.write_buffer(
                &self.tile_table,
                start as u64 * TILE_BYTES,
                bytemuck::cast_slice(&self.header_scratch),
            );
            index = end + 1;
        }
    }
}

impl RenderPass for FoliagePlacePass {
    fn name(&self) -> &'static str {
        "FoliagePlace"
    }

    /// `"foliage_terrain"` is deliberately absent even though the plan's §10 lists it.
    ///
    /// `RenderGraph::validate_dependencies` rejects a read with no prior writer, and
    /// `FoliageTerrainPass` does not exist yet — declaring the read today makes every
    /// graph containing this pass fail to build. Add it in the same change that adds the
    /// producer.
    fn reads(&self) -> &'static [&'static str] {
        &["hiz", "main_scene"]
    }

    fn writes(&self) -> &'static [&'static str] {
        &["foliage_draws"]
    }

    fn render_pass_descriptor<'a>(
        &'a self,
        _target: &'a wgpu::TextureView,
        _depth: &'a wgpu::TextureView,
        _resources: &'a libhelio::FrameResources<'a>,
    ) -> Option<wgpu::RenderPassDescriptor<'a>> {
        None
    }

    fn prepare(&mut self, ctx: &PrepareContext) -> HelioResult<()> {
        // Zero overhead when absent: an unwritten `foliage` slot means no foliage types
        // are registered, and this returns before touching a buffer. `execute` gates on
        // the same flag and records nothing.
        self.active = false;
        if ctx.frame_num < 3 || ctx.frame_num % 120 == 0 {
            log::debug!(
                "[foliage][gate] frame={} slot_written={} type_count={:?}",
                ctx.frame_num,
                ctx.frame_resources.foliage.is_some(),
                ctx.frame_resources.foliage.get().map(|f| f.type_count),
            );
        }
        if !foliage_frame_is_present(ctx.frame_resources) {
            return Ok(());
        }
        let Some(foliage) = ctx.frame_resources.foliage.get() else {
            return Ok(());
        };

        let mut type_count = foliage.type_count;
        if type_count > MAX_FOLIAGE_TYPES {
            if !self.warned_type_overflow {
                log::warn!(
                    "{type_count} foliage types published but GpuBladeInstance stores the \
                     type id in 8 bits; clamping to {MAX_FOLIAGE_TYPES}"
                );
                self.warned_type_overflow = true;
            }
            type_count = MAX_FOLIAGE_TYPES;
        }

        let Ok(all_types) = bytemuck::try_cast_slice::<u8, GpuFoliageType>(foliage.types) else {
            log::warn!(
                "foliage type table is {} bytes, not a whole number of {TYPE_BYTES}-byte \
                 GpuFoliageType records; skipping foliage this frame",
                foliage.types.len()
            );
            return Ok(());
        };
        if (type_count as usize) > all_types.len() {
            log::warn!(
                "foliage published type_count {type_count} against a table of {} entries; \
                 skipping foliage this frame",
                all_types.len()
            );
            return Ok(());
        }
        let types = &all_types[..type_count as usize];

        // Tables change on authoring edits only — wind must not advance the generation,
        // or this re-uploads every frame and the residency cache stops being free.
        if self.last_type_generation != Some(foliage.generation) {
            ctx.write_buffer(&self.type_table, 0, bytemuck::cast_slice(types));
            self.last_type_generation = Some(foliage.generation);
        }

        let (max_foliage_height, max_density) = self.table_stats(types);
        let (candidate_grid, achieved_density) = self.candidate_grid(max_density);
        self.density_scale = achieved_density;
        if achieved_density < 0.999 && !self.warned_density_clamp {
            log::warn!(
                "foliage density clamped to {:.0}% of the authored value: a tile's \
                 {} candidates do not fit its {}-blade arena slab. Grass will be uniformly \
                 thinner. Raise FoliageQuality::blade_arena_bytes or lower the authored \
                 density.",
                achieved_density * 100.0,
                (max_density
                    * self.quality.density_multiplier()
                    * FOLIAGE_TILE_SIZE_METERS
                    * FOLIAGE_TILE_SIZE_METERS)
                    .ceil(),
                self.blades_per_tile,
            );
            self.warned_density_clamp = true;
        }

        // Residency is keyed on (tile_coord, generation); the generation is truncated to
        // 32 bits because that is what `blade_seed` mixes. Wrapping is harmless — it takes
        // 4 billion authoring edits, and a collision only means one tile keeps its blades
        // through an edit it should have re-rolled.
        let generation = foliage.generation as u32;
        let camera = ctx.scene.camera.position();
        let ring_update = self.ring.update([camera[0], camera[2]], generation);
        if ring_update.evicted > 0 {
            log::debug!(
                "foliage ring evicted {} resident tiles to make room; the ring capacity is \
                 smaller than its window and placement is no longer amortised",
                ring_update.evicted
            );
        }

        self.upload_tile_headers(ctx, generation);

        let queue = self.ring.place_queue();
        self.queued_tile_count = queue.len() as u32;
        if !queue.is_empty() {
            ctx.write_buffer(&self.place_queue, 0, bytemuck::cast_slice(queue));
        }

        let (terrain_origin, terrain_extent, terrain_valid) = match self.terrain_transform {
            Some((origin, extent)) if ctx.frame_resources.foliage_terrain.get().is_some() => {
                (origin, extent, 1)
            }
            _ => ([0.0, 0.0], 1.0, 0),
        };

        ctx.write_buffer(
            &self.place_uniforms,
            0,
            bytemuck::bytes_of(&PlaceUniforms {
                tile_size: FOLIAGE_TILE_SIZE_METERS,
                candidate_grid,
                slab_capacity: self.blades_per_tile,
                queued_tile_count: self.queued_tile_count,
                density_multiplier: self.quality.density_multiplier(),
                max_density: max_density * self.quality.density_multiplier(),
                type_count,
                max_foliage_height,
                terrain_valid,
                terrain_origin_x: terrain_origin[0],
                terrain_origin_z: terrain_origin[1],
                terrain_extent,
                _pad: [0; 4],
            }),
        );

        // 2D dispatch grid, the way `vg_cull.wgsl` does it: at the reference tier the
        // cluster lane count is ~98 k, which is under the single-dimension limit, but a
        // large ring on a device with a small limit must still fit.
        let max_dispatch = ctx
            .device
            .limits()
            .max_compute_workgroups_per_dimension
            .max(1);
        let cluster_groups = (self.ring_capacity as u64 * self.clusters_per_tile as u64)
            .div_ceil(WORKGROUP_SIZE as u64)
            .max(1) as u32;
        self.cluster_dispatch_width = cluster_groups.min(max_dispatch).max(1);
        self.cluster_dispatch_height = cluster_groups.div_ceil(self.cluster_dispatch_width);
        self.tile_dispatch_groups = self.ring_capacity.div_ceil(WORKGROUP_SIZE).max(1);

        let max_dim = ctx.width.max(ctx.height).max(1);
        let hiz_mip_count = (u32::BITS - max_dim.leading_zeros()).max(1);
        // Frame 0 has no pyramid, and an untouched depth texture reads back as 0.0, which
        // the near-is-0.0 convention would treat as "everything is occluded". Same guard
        // as `vg_cull.wgsl`, and the same reason.
        let hiz_valid =
            (ctx.frame_num > 0 && ctx.frame_resources.hiz.get().is_some()) as u32;

        ctx.write_buffer(
            &self.cull_uniforms,
            0,
            bytemuck::bytes_of(&FoliageCullUniforms {
                tile_count: self.ring_capacity,
                screen_width: ctx.width,
                screen_height: ctx.height,
                hiz_mip_count,
                hiz_valid,
                cluster_size: self.cluster_size,
                clusters_per_tile: self.clusters_per_tile,
                per_lod_capacity: FOLIAGE_VISIBLE_PER_LOD_CAPACITY,
                tile_size: FOLIAGE_TILE_SIZE_METERS,
                lod_quality_scale: self.quality.lod_distance_scale(),
                type_count,
                cluster_dispatch_width: self.cluster_dispatch_width,
                max_foliage_height,
                wpo_extent: DEFAULT_WPO_EXTENT_METERS,
                _pad: [0; 2],
            }),
        );

        self.pump_readback(ctx.device);
        if ctx.frame_num % 120 == 0 {
            let c = self.debug_counters;
            log::info!(
                "[foliage] queued_tiles={} placed={} visible L0={} L1={} L2={} L3={} \
                 visible_overflow={} placement_overflow={}",
                self.queued_tile_count,
                c[COUNTER_PLACED_BLADES],
                c[0],
                c[1],
                c[2],
                c[3],
                c[COUNTER_VISIBLE_OVERFLOW],
                c[COUNTER_PLACEMENT_OVERFLOW],
            );
        }

        self.active = true;
        Ok(())
    }

    fn execute(&mut self, ctx: &mut PassContext) -> HelioResult<()> {
        if !self.active {
            return Ok(());
        }

        let hiz_view = ctx.resources.hiz.get().unwrap_or(&self.placeholder_hiz);
        let hiz_sampler = ctx
            .resources
            .hiz_sampler
            .get()
            .unwrap_or(&self.placeholder_hiz_sampler);
        let key = (
            ctx.scene.camera as *const _ as usize,
            hiz_view as *const _ as usize,
            hiz_sampler as *const _ as usize,
        );
        if self.cull_bind_group_key != Some(key) {
            self.cull_bind_group = Some(ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Foliage Cull BG"),
                layout: &self.cull_bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: ctx.scene.camera.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: self.cull_uniforms.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: self.tile_table.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: self.blade_arena.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: self.type_table.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: self.visible_blades.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 6,
                        resource: self.counters.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 7,
                        resource: self.foliage_indirect.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 8,
                        resource: self.tile_visibility.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 9,
                        resource: wgpu::BindingResource::TextureView(hiz_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 10,
                        resource: wgpu::BindingResource::Sampler(hiz_sampler),
                    },
                ],
            }));
            self.cull_bind_group_key = Some(key);
        }
        let Some(cull_bg) = self.cull_bind_group.as_ref() else {
            return Ok(());
        };

        // `ctx.encoder_ptr`, not `ctx.compute_encoder_ptr`: the two encoders are submitted
        // as [compute, render], so anything recorded on the compute encoder runs before
        // *all* render-encoder work and would therefore Hi-Z-test against the previous
        // frame's pyramid. This pass does not declare `chain_transparent` for the same
        // reason. See the plan's §6.2 [audit].
        let encoder = unsafe { &mut *ctx.encoder_ptr };

        // Visible counts are per-frame; the overflow counters are too, so a single bad
        // frame does not look like a permanent budget failure.
        encoder.clear_buffer(&self.counters, 0, None);

        if self.queued_tile_count > 0 {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Foliage Place"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.place_pipeline);
            pass.set_bind_group(0, &self.place_bind_group, &[]);
            // One workgroup per queued tile, bounded by `max_tiles_per_frame`. This is
            // what turns a teleport into a few frames of progressive fill-in instead of
            // a hitch.
            pass.dispatch_workgroups(self.queued_tile_count, 1, 1);
        }

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Foliage Tile Cull"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.tile_cull_pipeline);
            pass.set_bind_group(0, cull_bg, &[]);
            pass.dispatch_workgroups(self.tile_dispatch_groups, 1, 1);
        }

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Foliage Cluster Cull"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.cluster_cull_pipeline);
            pass.set_bind_group(0, cull_bg, &[]);
            pass.dispatch_workgroups(self.cluster_dispatch_width, self.cluster_dispatch_height, 1);
        }

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Foliage Finalize"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.finalize_pipeline);
            pass.set_bind_group(0, cull_bg, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }

        if matches!(self.readback_state, ReadbackState::Idle) {
            encoder.copy_buffer_to_buffer(
                &self.counters,
                0,
                &self.counters_readback,
                0,
                FOLIAGE_COUNTER_COUNT as u64 * 4,
            );
            self.readback_state = ReadbackState::CopySubmitted;
        }

        self.commands_recorded += 1;
        Ok(())
    }
}

fn uniform_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn storage_entry(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

/// State of the non-blocking counter readback. Never blocks the GPU; values may trail the
/// rendered frame by a frame or two, which is fine for telemetry.
enum ReadbackState {
    Idle,
    CopySubmitted,
    Mapping(std::sync::Arc<std::sync::Mutex<Option<Result<(), wgpu::BufferAsyncError>>>>),
}

/// Integer square root. `(x as f32).sqrt()` loses precision above 2^24 and would round a
/// grid resolution *up* past its slab, which is the one direction that overflows.
fn isqrt(value: u32) -> u32 {
    if value == 0 {
        return 0;
    }
    // Both refinement loops compare in u64. In u32 the correction step is not merely
    // imprecise, it does not terminate: near the top of the range `guess + 1` squared
    // exceeds u32, `saturating_mul` pins it at u32::MAX, `u32::MAX <= value` is then true
    // for value == u32::MAX, and the loop walks `guess` up until the increment itself
    // overflows.
    let value = u64::from(value);
    let mut guess = u64::from((value as f64).sqrt() as u32);
    while guess > 0 && guess * guess > value {
        guess -= 1;
    }
    while (guess + 1) * (guess + 1) <= value {
        guess += 1;
    }
    guess as u32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn isqrt_is_exact_at_and_around_perfect_squares() {
        assert_eq!(isqrt(0), 0);
        assert_eq!(isqrt(1), 1);
        for root in 1u32..1024 {
            let square = root * root;
            assert_eq!(isqrt(square), root);
            assert_eq!(isqrt(square - 1), root - 1);
            assert_eq!(isqrt(square + 1), root);
        }
        // The range where the f32 path would have drifted.
        assert_eq!(isqrt(u32::MAX), 65535);
        assert_eq!(isqrt(1 << 30), 1 << 15);
    }

    #[test]
    fn every_quality_preset_fits_the_tile_ring_and_leaves_a_usable_slab() {
        for quality in [
            FoliageQuality::Low,
            FoliageQuality::Medium,
            FoliageQuality::High,
            FoliageQuality::Ultra,
        ] {
            let tiles_across =
                ((2.0 * quality.ring_radius() / FOLIAGE_TILE_SIZE_METERS).ceil() as u32).max(1);
            let ring_capacity = tiles_across * tiles_across;
            assert!(
                ring_capacity <= DEFAULT_TILE_RING_CAPACITY,
                "{quality:?} needs {ring_capacity} ring slots, more than the \
                 {DEFAULT_TILE_RING_CAPACITY}-slot table — the ring would thrash"
            );

            let blades_per_tile = (quality.blade_capacity() / ring_capacity as u64) as u32;
            assert!(
                blades_per_tile >= 256,
                "{quality:?} leaves only {blades_per_tile} blades per 8 m tile"
            );
            // The slab must be a whole number of clusters' worth or the last cluster of
            // every tile is partially outside the slab.
            let clusters = blades_per_tile.div_ceil(quality.cluster_granularity());
            assert!(clusters >= 1);
            assert!(clusters * quality.cluster_granularity() >= blades_per_tile);
        }
    }

    #[test]
    fn the_arena_never_exceeds_the_quality_budget() {
        for quality in [
            FoliageQuality::Low,
            FoliageQuality::Medium,
            FoliageQuality::High,
            FoliageQuality::Ultra,
        ] {
            let tiles_across =
                ((2.0 * quality.ring_radius() / FOLIAGE_TILE_SIZE_METERS).ceil() as u32).max(1);
            let ring_capacity = (tiles_across * tiles_across).min(DEFAULT_TILE_RING_CAPACITY);
            let blades_per_tile = quality.blade_capacity() / ring_capacity as u64;
            let arena_bytes = ring_capacity as u64 * blades_per_tile * BLADE_BYTES;
            assert!(
                arena_bytes <= quality.blade_arena_bytes(),
                "{quality:?} arena of {arena_bytes} B exceeds its {} B budget",
                quality.blade_arena_bytes()
            );
        }
    }

    #[test]
    fn gpu_record_sizes_match_the_wgsl_mirrors() {
        assert_eq!(TILE_BYTES, 32);
        assert_eq!(BLADE_BYTES, 16);
        assert_eq!(TYPE_BYTES, 96);
        assert_eq!(MAX_FOLIAGE_TYPES, 256, "the 8-bit type id caps this");
    }
}
